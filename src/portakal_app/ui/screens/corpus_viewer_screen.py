from __future__ import annotations

import html
import re
from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
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


@dataclass(frozen=True)
class CorpusViewerFilterResult:
    row_indexes: tuple[int, ...]
    matching_documents: int
    matches: int
    error: str = ""


def corpus_viewer_unique_word_count(documents: Sequence[CorpusDocument]) -> int:
    words: set[str] = set()
    for document in documents:
        words.update(token.lower() for token in _WORD_PATTERN.findall(document.text))
    return len(words)


def filter_corpus_viewer_documents(
    documents: Sequence[CorpusDocument],
    pattern: str,
    *,
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
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        layout.addWidget(QLabel("Search / RegExp Filter", self), 0, 0)
        self._filter_input = QLineEdit(self)
        self._filter_input.setPlaceholderText("profit|market")
        self._filter_input.textChanged.connect(self._handle_filter_changed)
        layout.addWidget(self._filter_input, 0, 1, 1, 5)

        self._search_title_checkbox = QCheckBox("Search title", self)
        self._search_source_checkbox = QCheckBox("Search source", self)
        self._search_content_checkbox = QCheckBox("Search content", self)
        for checkbox in (
            self._search_title_checkbox,
            self._search_source_checkbox,
            self._search_content_checkbox,
        ):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._handle_filter_changed)
        layout.addWidget(self._search_title_checkbox, 1, 0)
        layout.addWidget(self._search_source_checkbox, 1, 1)
        layout.addWidget(self._search_content_checkbox, 1, 2)

        self._show_title_checkbox = QCheckBox("Show title", self)
        self._show_source_checkbox = QCheckBox("Show source", self)
        self._show_content_checkbox = QCheckBox("Show content", self)
        for checkbox in (
            self._show_title_checkbox,
            self._show_source_checkbox,
            self._show_content_checkbox,
        ):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._update_detail_panel)
        layout.addWidget(self._show_title_checkbox, 2, 0)
        layout.addWidget(self._show_source_checkbox, 2, 1)
        layout.addWidget(self._show_content_checkbox, 2, 2)
        layout.setColumnStretch(5, 1)
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

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Title", "Source", "Text Preview", "Words"])
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
        self._detail_title_label = self._build_detail_label("Title", "-")
        self._detail_source_label = self._build_detail_label("Source", "-")
        self._detail_words_label = self._build_detail_label("Words", "0")
        layout.addWidget(self._detail_title_label)
        layout.addWidget(self._detail_source_label)
        layout.addWidget(self._detail_words_label)

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
            self._render()
            self._notify_output_changed()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self._render()
        self._notify_output_changed()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Corpus", self._documents)

    def _handle_filter_changed(self, *_args: object) -> None:
        self._render()

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
            search_title=self._search_title_checkbox.isChecked()
            if hasattr(self, "_search_title_checkbox")
            else True,
            search_source=self._search_source_checkbox.isChecked()
            if hasattr(self, "_search_source_checkbox")
            else True,
            search_content=self._search_content_checkbox.isChecked()
            if hasattr(self, "_search_content_checkbox")
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
            self._set_item(row, 1, document.source)
            self._set_item(row, 2, preview_text(document.text))
            self._set_item(row, 3, str(count_words(document.text)))
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
            self._detail_title_label.setText("Title: -")
            self._detail_source_label.setText("Source: -")
            self._detail_words_label.setText("Words: 0")
            self._detail_content.setHtml("<p style='color:#7e715e;'>No document selected.</p>")
            return

        document = self._documents[document_index]
        show_title = self._show_title_checkbox.isChecked()
        show_source = self._show_source_checkbox.isChecked()
        show_content = self._show_content_checkbox.isChecked()

        self._detail_title_label.setVisible(show_title)
        self._detail_source_label.setVisible(show_source)
        self._detail_title_label.setText(f"Title: {document.title}")
        self._detail_source_label.setText(f"Source: {document.source}")
        self._detail_words_label.setText(f"Words: {count_words(document.text)}")

        if show_content:
            self._detail_content.setHtml(self._document_content_html(document.text))
        else:
            self._detail_content.setHtml("<p style='color:#7e715e;'>Content hidden.</p>")

    def _document_content_html(self, text: str) -> str:
        escaped = html.escape(text)
        query = self._filter_input.text().strip()
        if query and not self._filter_result.error and self._search_content_checkbox.isChecked():
            try:
                regex = re.compile(query, re.IGNORECASE)
            except re.error:
                regex = None
            if regex is not None:
                parts: list[str] = []
                last = 0
                for match in regex.finditer(text):
                    start, end = match.span()
                    if start == end:
                        continue
                    parts.append(html.escape(text[last:start]))
                    parts.append(
                        "<span style='background-color:#ffe58a; color:#3b2a10;'>"
                        f"{html.escape(text[start:end])}"
                        "</span>"
                    )
                    last = end
                parts.append(html.escape(text[last:]))
                escaped = "".join(parts)
        empty_html = '<span style="color:#7e715e;">Empty document.</span>'
        return (
            "<div style='white-space:pre-wrap; font-family:Menlo, Consolas, monospace; "
            "font-size:12px; line-height:1.45;'>"
            f"{escaped or empty_html}"
            "</div>"
        )

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [
                self._documents[index].title,
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
            "headers": ["Title", "Source", "Text Preview", "Words"],
            "rows": rows,
        }
