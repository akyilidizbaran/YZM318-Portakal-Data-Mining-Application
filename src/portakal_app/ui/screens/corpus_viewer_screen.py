from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.corpus_screen import (
    CorpusDocument,
    corpus_documents_from_payload,
    count_words,
    summarize_corpus,
)
from portakal_app.ui.screens.create_corpus_screen import preview_text
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
_KNOWN_CATEGORIES = (
    "business",
    "entertainment",
    "politics",
    "sport",
    "sports",
    "tech",
    "technology",
)


@dataclass(frozen=True)
class CorpusViewerFilterResult:
    row_indexes: tuple[int, ...]
    matching_documents: int
    matches: int
    error: str = ""


class FeatureRow(QFrame):
    def __init__(self, name: str, *, type_code: str = "S", checked: bool = True) -> None:
        super().__init__()
        self.setProperty("featureRow", True)
        self._checkbox = QCheckBox(self)
        self._badge = QLabel(type_code, self)
        self._name_label = QLabel(name, self)

        self._badge.setFixedSize(22, 22)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge_color = "#6f5a44" if type_code == "C" else "#f28c28"
        self._badge.setStyleSheet(
            f"border-radius: 4px; background: {badge_color}; color: white; font-weight: 700;"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        layout.addWidget(self._checkbox)
        layout.addWidget(self._badge)
        layout.addWidget(self._name_label, 1)

        self._checkbox.toggled.connect(self._sync_style)
        self.setChecked(checked)

    def isChecked(self) -> bool:
        return self._checkbox.isChecked()

    def setChecked(self, checked: bool) -> None:
        self._checkbox.setChecked(checked)
        self._sync_style()

    def on_toggled(self, callback) -> None:
        self._checkbox.toggled.connect(callback)

    def mousePressEvent(self, event) -> None:
        self.setChecked(not self.isChecked())
        event.accept()

    def _sync_style(self, *_args: object) -> None:
        if self.isChecked():
            self.setStyleSheet(
                "QFrame[featureRow='true'] {"
                "border: 1px solid #f0a433; border-radius: 6px; background: #fff8ee;"
                "}"
            )
        else:
            self.setStyleSheet(
                "QFrame[featureRow='true'] {"
                "border: 1px solid #d7c9b5; border-radius: 6px; background: #ffffff;"
                "}"
            )


def corpus_viewer_unique_word_count(documents: Sequence[CorpusDocument]) -> int:
    words: set[str] = set()
    for document in documents:
        words.update(token.lower() for token in _WORD_PATTERN.findall(document.text))
    return len(words)


def corpus_viewer_words_from_payload(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates: Sequence[object] = (value,)
    elif isinstance(value, Sequence):
        candidates = value
    else:
        return ()

    words: list[str] = []
    for item in candidates:
        if isinstance(item, str):
            words.extend(token.lower() for token in _WORD_PATTERN.findall(item))
        elif isinstance(item, Sequence) and item and isinstance(item[0], str):
            words.extend(token.lower() for token in _WORD_PATTERN.findall(item[0]))
    return tuple(dict.fromkeys(word for word in words if word))


def infer_corpus_document_category(document: CorpusDocument) -> str:
    for attribute in ("category", "label", "class_name", "topic"):
        value = getattr(document, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = getattr(document, "metadata", None) or getattr(document, "meta", None)
    if isinstance(metadata, dict):
        for key in ("category", "label", "class", "topic"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    searchable = " ".join((document.source, document.title, document.text)).lower()
    for category in _KNOWN_CATEGORIES:
        if re.search(rf"\b{re.escape(category)}\b", searchable):
            if category == "sports":
                return "sport"
            if category == "technology":
                return "tech"
            return category
    return "-"


def filter_corpus_viewer_documents(
    documents: Sequence[CorpusDocument],
    pattern: str,
    *,
    search_category: bool = True,
    search_title: bool = True,
    search_source: bool = True,
    search_content: bool = True,
) -> CorpusViewerFilterResult:
    query = pattern.strip()
    if not query:
        return CorpusViewerFilterResult(
            row_indexes=tuple(range(len(documents))),
            matching_documents=len(documents),
            matches=0,
        )

    try:
        regex = re.compile(query, re.IGNORECASE)
    except re.error as error:
        return CorpusViewerFilterResult((), 0, 0, f"Invalid regular expression: {error}")

    row_indexes: list[int] = []
    match_count = 0
    for index, document in enumerate(documents):
        fields: list[str] = []
        if search_category:
            fields.append(infer_corpus_document_category(document))
        if search_title:
            fields.append(document.title)
        if search_source:
            fields.append(document.source)
        if search_content:
            fields.append(document.text)
        haystack = "\n".join(fields)
        matches = list(regex.finditer(haystack))
        if matches:
            row_indexes.append(index)
            match_count += len(matches)

    return CorpusViewerFilterResult(
        row_indexes=tuple(row_indexes),
        matching_documents=len(row_indexes),
        matches=match_count,
    )


class CorpusViewerScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._highlight_words: tuple[str, ...] = ()
        self._filtered_indexes: tuple[int, ...] = tuple(range(len(self._documents)))
        self._filter_result = CorpusViewerFilterResult(self._filtered_indexes, len(self._documents), 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Corpus Viewer",
                "Inspect corpus documents and pass the same corpus to downstream text mining widgets.",
            )
        )
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_filter_panel())
        layout.addWidget(self._build_document_browser(), 1)

        self._render()

    def sizeHint(self) -> QSize:
        return QSize(1100, 720)

    def minimumSizeHint(self) -> QSize:
        return QSize(820, 560)

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._total_word_count_label = self._build_metric_label("Total Words", "0")
        self._unique_word_count_label = self._build_metric_label("Unique Words", "0")
        self._matching_documents_label = self._build_metric_label("Matching Documents", "0 / 0")
        self._matches_label = self._build_metric_label("Matches", "0")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._total_word_count_label, 1)
        layout.addWidget(self._unique_word_count_label, 1)
        layout.addWidget(self._matching_documents_label, 1)
        layout.addWidget(self._matches_label, 1)
        return frame

    def _build_metric_label(self, title: str, value: str) -> QLabel:
        label = QLabel(f"{title}\n{value}", self)
        label.setProperty("infoCard", True)
        label.setStyleSheet("padding: 10px;")
        label.setWordWrap(True)
        return label

    def _build_filter_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(10)
        filter_layout.addWidget(QLabel("Search / RegExp Filter", self))
        self._filter_input = QLineEdit(self)
        self._filter_input.setPlaceholderText("profit|market")
        self._filter_input.textChanged.connect(self._handle_filter_changed)
        filter_layout.addWidget(self._filter_input, 1)
        layout.addLayout(filter_layout)

        features_layout = QHBoxLayout()
        features_layout.setSpacing(12)

        self._search_category_feature = FeatureRow("Category", type_code="C", checked=True)
        self._search_title_feature = FeatureRow("Title / Name", checked=True)
        self._search_source_feature = FeatureRow("Source / Path", checked=True)
        self._search_content_feature = FeatureRow("Content", checked=True)
        for row in self._search_feature_rows():
            row.on_toggled(self._handle_filter_changed)

        self._display_category_feature = FeatureRow("Category", type_code="C", checked=True)
        self._display_title_feature = FeatureRow("Title / Name", checked=True)
        self._display_source_feature = FeatureRow("Source / Path", checked=True)
        self._display_content_feature = FeatureRow("Content", checked=True)
        for row in self._display_feature_rows():
            row.on_toggled(self._update_detail_panel)

        features_layout.addWidget(
            self._build_feature_list_panel("Search features", self._search_feature_rows()),
            1,
        )
        features_layout.addWidget(
            self._build_feature_list_panel("Display features", self._display_feature_rows()),
            1,
        )
        layout.addLayout(features_layout)
        return frame

    def _search_feature_rows(self) -> tuple[FeatureRow, FeatureRow, FeatureRow, FeatureRow]:
        return (
            self._search_category_feature,
            self._search_title_feature,
            self._search_source_feature,
            self._search_content_feature,
        )

    def _display_feature_rows(self) -> tuple[FeatureRow, FeatureRow, FeatureRow, FeatureRow]:
        return (
            self._display_category_feature,
            self._display_title_feature,
            self._display_source_feature,
            self._display_content_feature,
        )

    def _build_feature_list_panel(self, title: str, rows: Sequence[FeatureRow]) -> QFrame:
        frame = QFrame(self)
        frame.setStyleSheet(
            "QFrame { border: 1px solid #d7c9b5; border-radius: 6px; background: #fffaf2; }"
            "QLabel { border: none; background: transparent; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        title_label = QLabel(title, self)
        title_label.setProperty("muted", True)
        layout.addWidget(title_label)
        for row in rows:
            layout.addWidget(row)
        layout.addStretch(1)
        return frame

    def _build_document_browser(self) -> QSplitter:
        splitter = QSplitter(self)
        splitter.addWidget(self._build_table_panel())
        splitter.addWidget(self._build_detail_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return splitter

    def _build_table_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._status_label = QLabel("", self)
        self._status_label.setProperty("muted", True)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(
            ["Title / Name", "Category", "Source / Path", "Text Preview", "Words"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._update_detail_panel)
        layout.addWidget(self._table, 1)
        return frame

    def _build_detail_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Document Details", self))
        self._detail_category_label = self._build_detail_label("Category", "-")
        self._detail_title_label = self._build_detail_label("Title / Name", "-")
        self._detail_source_label = self._build_detail_label("Source / Path", "-")
        self._detail_words_label = self._build_detail_label("Words", "0")
        self._detail_content_label = self._build_detail_label("Content", "")
        layout.addWidget(self._detail_category_label)
        layout.addWidget(self._detail_title_label)
        layout.addWidget(self._detail_source_label)
        layout.addWidget(self._detail_words_label)
        layout.addWidget(self._detail_content_label)

        self._detail_content = QTextEdit(self)
        self._detail_content.setReadOnly(True)
        self._detail_content.setMinimumHeight(240)
        layout.addWidget(self._detail_content, 1)
        return frame

    def _build_detail_label(self, title: str, value: str) -> QLabel:
        label = QLabel(f"{title}: {value}", self)
        label.setWordWrap(True)
        return label

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = ()
            self._using_input_corpus = False
            self._highlight_words = ()
            self._render()
            self._notify_output_changed()
            return

        if payload.port_label in {"Words", "Selected Words"}:
            self._highlight_words = corpus_viewer_words_from_payload(payload.value)
            self._render()
            self._notify_output_changed()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self._render()
        self._notify_output_changed()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload(
            "Corpus",
            tuple(self._documents[index] for index in self._filtered_indexes),
        )

    def _handle_filter_changed(self, *_args: object) -> None:
        self._render()
        self._notify_output_changed()

    def _selected_document_index(self) -> int | None:
        selected = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if not selected:
            return self._filtered_indexes[0] if self._filtered_indexes else None
        row = selected[0].row()
        if 0 <= row < len(self._filtered_indexes):
            return self._filtered_indexes[row]
        return None

    def _render(self) -> None:
        summary = summarize_corpus(self._documents)
        self._filter_result = filter_corpus_viewer_documents(
            self._documents,
            self._filter_input.text() if hasattr(self, "_filter_input") else "",
            search_category=self._search_category_feature.isChecked()
            if hasattr(self, "_search_category_feature")
            else True,
            search_title=self._search_title_feature.isChecked()
            if hasattr(self, "_search_title_feature")
            else True,
            search_source=self._search_source_feature.isChecked()
            if hasattr(self, "_search_source_feature")
            else True,
            search_content=self._search_content_feature.isChecked()
            if hasattr(self, "_search_content_feature")
            else True,
        )
        self._filtered_indexes = self._filter_result.row_indexes

        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._total_word_count_label.setText(f"Total Words\n{summary.total_word_count}")
        self._unique_word_count_label.setText(
            f"Unique Words\n{corpus_viewer_unique_word_count(self._documents)}"
        )
        self._matching_documents_label.setText(
            f"Matching Documents\n{self._filter_result.matching_documents} / {len(self._documents)}"
        )
        matches_value = "n/a" if not self._filter_input.text().strip() else str(self._filter_result.matches)
        self._matches_label.setText(f"Matches\n{matches_value}")

        if self._filter_result.error:
            status = self._filter_result.error
        elif self._documents and self._using_input_corpus:
            status = "Input corpus is connected and displayed."
            if self._filter_input.text().strip():
                status = (
                    f"Filter matched {self._filter_result.matching_documents} "
                    f"of {len(self._documents)} documents."
                )
        elif self._using_input_corpus:
            status = "Input corpus is connected but empty."
        else:
            status = "Connect a Corpus input to inspect documents."
        self._status_label.setText(status)

        previous_index = self._selected_document_index()
        self._table.setRowCount(len(self._filtered_indexes))
        for row, document_index in enumerate(self._filtered_indexes):
            document = self._documents[document_index]
            self._set_item(row, 0, document.title)
            self._set_item(row, 1, infer_corpus_document_category(document))
            self._set_item(row, 2, document.source)
            self._set_item(row, 3, preview_text(document.text))
            self._set_item(row, 4, str(count_words(document.text)))
        self._table.resizeColumnsToContents()
        self._restore_or_select_first(previous_index)
        self._update_detail_panel()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def _restore_or_select_first(self, previous_index: int | None) -> None:
        if not self._filtered_indexes:
            self._table.clearSelection()
            return
        target_row = 0
        if previous_index in self._filtered_indexes:
            target_row = self._filtered_indexes.index(previous_index)
        self._table.selectRow(target_row)

    def _update_detail_panel(self, *_args: object) -> None:
        if not hasattr(self, "_detail_content"):
            return
        document_index = self._selected_document_index()
        if document_index is None:
            self._detail_category_label.setText("Category: -")
            self._detail_title_label.setText("Title / Name: -")
            self._detail_source_label.setText("Source / Path: -")
            self._detail_words_label.setText("Words: 0")
            self._detail_content_label.setVisible(True)
            self._detail_content.setVisible(True)
            self._detail_content.setHtml("<p style='color:#7e715e;'>No document selected.</p>")
            return

        document = self._documents[document_index]
        show_category = self._display_category_feature.isChecked()
        show_title = self._display_title_feature.isChecked()
        show_source = self._display_source_feature.isChecked()
        show_content = self._display_content_feature.isChecked()

        self._detail_category_label.setVisible(show_category)
        self._detail_title_label.setVisible(show_title)
        self._detail_source_label.setVisible(show_source)
        self._detail_category_label.setText(f"Category: {infer_corpus_document_category(document)}")
        self._detail_title_label.setText(f"Title / Name: {document.title}")
        self._detail_source_label.setText(f"Source / Path: {document.source}")
        self._detail_words_label.setText(f"Words: {count_words(document.text)}")
        self._detail_content_label.setVisible(True)
        self._detail_content.setVisible(True)

        if show_content:
            self._detail_content.setHtml(self._document_content_html(document.text))
        else:
            self._detail_content.setHtml("<p style='color:#7e715e;'>Content display disabled.</p>")

    def _document_content_html(self, text: str) -> str:
        escaped = self._highlighted_text_html(text)
        empty_html = '<span style="color:#7e715e;">Empty document.</span>'
        return (
            "<div style='white-space:pre-wrap; font-family:Menlo, Consolas, monospace; "
            "font-size:12px; line-height:1.45;'>"
            f"{escaped or empty_html}"
            "</div>"
        )

    def _highlighted_text_html(self, text: str) -> str:
        spans = self._content_highlight_spans(text)
        if not spans:
            return html.escape(text)

        parts: list[str] = []
        last = 0
        for start, end in spans:
            parts.append(html.escape(text[last:start]))
            parts.append(
                "<span style='background-color:#ffe58a; color:#3b2a10;'>"
                f"{html.escape(text[start:end])}"
                "</span>"
            )
            last = end
        parts.append(html.escape(text[last:]))
        return "".join(parts)

    def _content_highlight_spans(self, text: str) -> tuple[tuple[int, int], ...]:
        spans: list[tuple[int, int]] = []
        query = self._filter_input.text().strip()
        if query and not self._filter_result.error and self._search_content_feature.isChecked():
            try:
                regex = re.compile(query, re.IGNORECASE)
            except re.error:
                regex = None
            if regex is not None:
                for match in regex.finditer(text):
                    start, end = match.span()
                    if start != end:
                        spans.append((start, end))

        if self._highlight_words:
            word_pattern = "|".join(re.escape(word) for word in self._highlight_words)
            regex = re.compile(rf"\b(?:{word_pattern})\b", re.IGNORECASE)
            spans.extend(match.span() for match in regex.finditer(text))

        return self._merge_highlight_spans(spans)

    def _merge_highlight_spans(self, spans: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        normalized = sorted((start, end) for start, end in spans if start < end)
        if not normalized:
            return ()
        merged: list[tuple[int, int]] = [normalized[0]]
        for start, end in normalized[1:]:
            previous_start, previous_end = merged[-1]
            if start <= previous_end:
                merged[-1] = (previous_start, max(previous_end, end))
            else:
                merged.append((start, end))
        return tuple(merged)

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [
                self._documents[index].title,
                infer_corpus_document_category(self._documents[index]),
                self._documents[index].source,
                preview_text(self._documents[index].text),
                str(count_words(self._documents[index].text)),
            ]
            for index in self._filtered_indexes
        ]
        summary = summarize_corpus(self._documents)
        return {
            "summary": (
                f"Corpus Viewer: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.average_words_per_document:.1f} average words/document"
            ),
            "headers": ["Title / Name", "Category", "Source / Path", "Text Preview", "Words"],
            "rows": rows,
        }
