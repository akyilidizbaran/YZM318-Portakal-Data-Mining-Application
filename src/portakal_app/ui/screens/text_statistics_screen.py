from __future__ import annotations

import re
import string
from collections import Counter
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
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.bag_of_words_screen import tokenize_for_bow
from portakal_app.ui.screens.corpus_screen import (
    CorpusDocument,
    corpus_documents_from_payload,
    count_words,
    with_corpus_document_attributes,
)
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


@dataclass(frozen=True)
class TextStatisticsSummary:
    document_count: int
    total_word_count: int
    average_words_per_document: float
    unique_word_count: int
    longest_document_title: str
    shortest_document_title: str


@dataclass(frozen=True)
class DocumentStatisticsOptions:
    word_count: bool = True
    character_count: bool = True
    average_word_length: bool = True
    percent_unique_words: bool = True
    punctuation_count: bool = True
    contains: bool = False
    contains_pattern: str = ""


def word_frequencies(documents: Sequence[CorpusDocument]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for document in documents:
        counter.update(tokenize_for_bow(document.text))
    return counter


def statistics_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return normalized or "pattern"


def document_statistics_features(
    document: CorpusDocument,
    options: DocumentStatisticsOptions,
) -> dict[str, object]:
    tokens = tokenize_for_bow(document.text)
    features: dict[str, object] = {}

    if options.word_count:
        features["word_count"] = len(tokens)
    if options.character_count:
        features["character_count"] = len(document.text)
    if options.average_word_length:
        features["average_word_length"] = (
            round(sum(len(token) for token in tokens) / len(tokens), 2) if tokens else 0.0
        )
    if options.percent_unique_words:
        features["percent_unique_words"] = (
            round((len(set(tokens)) / len(tokens)) * 100, 2) if tokens else 0.0
        )
    if options.punctuation_count:
        features["punctuation_count"] = sum(1 for character in document.text if character in string.punctuation)
    pattern = options.contains_pattern.strip().lower()
    if options.contains and pattern:
        column_name = f"contains_{statistics_column_name(pattern)}"
        features[column_name] = sum(1 for token in tokens if pattern in token.lower())

    return features


def enrich_corpus_with_statistics(
    documents: Sequence[CorpusDocument],
    options: DocumentStatisticsOptions,
) -> tuple[CorpusDocument, ...]:
    return tuple(
        with_corpus_document_attributes(document, document_statistics_features(document, options))
        for document in documents
    )


def summarize_text_statistics(documents: Sequence[CorpusDocument]) -> TextStatisticsSummary:
    document_count = len(documents)
    word_counts = [(document.title, count_words(document.text)) for document in documents]
    total_word_count = sum(count for _title, count in word_counts)
    average = total_word_count / document_count if document_count else 0.0
    frequencies = word_frequencies(documents)
    longest = max(word_counts, key=lambda item: item[1], default=("-", 0))[0]
    shortest = min(word_counts, key=lambda item: item[1], default=("-", 0))[0]
    return TextStatisticsSummary(
        document_count=document_count,
        total_word_count=total_word_count,
        average_words_per_document=average,
        unique_word_count=len(frequencies),
        longest_document_title=longest,
        shortest_document_title=shortest,
    )


class TextStatisticsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._output_documents: tuple[CorpusDocument, ...] = self._documents
        self._using_input_corpus = documents is not None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Statistics",
                "Summarize corpus size, document lengths, vocabulary, and frequent words.",
            )
        )
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_feature_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self.apply_statistics()

    def sizeHint(self) -> QSize:
        return QSize(920, 640)

    def minimumSizeHint(self) -> QSize:
        return QSize(700, 500)

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._total_word_count_label = self._build_metric_label("Total Words", "0")
        self._average_word_count_label = self._build_metric_label("Avg Words / Document", "0.0")
        self._unique_word_count_label = self._build_metric_label("Unique Words", "0")
        self._longest_document_label = self._build_metric_label("Longest Document", "-")
        self._shortest_document_label = self._build_metric_label("Shortest Document", "-")

        labels = (
            self._document_count_label,
            self._total_word_count_label,
            self._average_word_count_label,
            self._unique_word_count_label,
            self._longest_document_label,
            self._shortest_document_label,
        )
        for index, label in enumerate(labels):
            layout.addWidget(label, index // 3, index % 3)
        return frame

    def _build_metric_label(self, title: str, value: str) -> QLabel:
        label = QLabel(f"{title}\n{value}", self)
        label.setProperty("infoCard", True)
        label.setStyleSheet("padding: 10px;")
        label.setWordWrap(True)
        return label

    def _build_feature_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        self._word_count_checkbox = QCheckBox("Word count", self)
        self._word_count_checkbox.setChecked(True)
        self._character_count_checkbox = QCheckBox("Character count", self)
        self._character_count_checkbox.setChecked(True)
        self._average_word_length_checkbox = QCheckBox("Average word length", self)
        self._average_word_length_checkbox.setChecked(True)
        self._percent_unique_words_checkbox = QCheckBox("Percent unique words", self)
        self._percent_unique_words_checkbox.setChecked(True)
        self._punctuation_count_checkbox = QCheckBox("Punctuation count", self)
        self._punctuation_count_checkbox.setChecked(True)
        self._contains_checkbox = QCheckBox("Contains", self)

        self._contains_input = QLineEdit(self)
        self._contains_input.setPlaceholderText("profit")

        checkboxes = (
            self._word_count_checkbox,
            self._character_count_checkbox,
            self._average_word_length_checkbox,
            self._percent_unique_words_checkbox,
            self._punctuation_count_checkbox,
        )
        for index, checkbox in enumerate(checkboxes):
            layout.addWidget(checkbox, index // 3, index % 3)

        contains_layout = QHBoxLayout()
        contains_layout.setSpacing(8)
        contains_layout.addWidget(self._contains_checkbox)
        contains_layout.addWidget(self._contains_input, 1)
        layout.addLayout(contains_layout, 2, 0, 1, 2)

        self._apply_button = QPushButton("Apply", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self.apply_statistics)
        layout.addWidget(self._apply_button, 2, 2)
        return frame

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

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Word", "Frequency"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = ()
            self._output_documents = ()
            self._using_input_corpus = False
            self.apply_statistics()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self.apply_statistics()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Corpus", self._output_documents)

    def selected_statistics_options(self) -> DocumentStatisticsOptions:
        return DocumentStatisticsOptions(
            word_count=self._word_count_checkbox.isChecked(),
            character_count=self._character_count_checkbox.isChecked(),
            average_word_length=self._average_word_length_checkbox.isChecked(),
            percent_unique_words=self._percent_unique_words_checkbox.isChecked(),
            punctuation_count=self._punctuation_count_checkbox.isChecked(),
            contains=self._contains_checkbox.isChecked(),
            contains_pattern=self._contains_input.text(),
        )

    def apply_statistics(self) -> tuple[CorpusDocument, ...]:
        if not hasattr(self, "_word_count_checkbox"):
            self._output_documents = self._documents
        else:
            self._output_documents = enrich_corpus_with_statistics(
                self._documents,
                self.selected_statistics_options(),
            )
        self._render()
        self._notify_output_changed()
        return self._output_documents

    def _render(self) -> None:
        summary = summarize_text_statistics(self._documents)
        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._total_word_count_label.setText(f"Total Words\n{summary.total_word_count}")
        self._average_word_count_label.setText(
            f"Avg Words / Document\n{summary.average_words_per_document:.1f}"
        )
        self._unique_word_count_label.setText(f"Unique Words\n{summary.unique_word_count}")
        self._longest_document_label.setText(f"Longest Document\n{summary.longest_document_title}")
        self._shortest_document_label.setText(f"Shortest Document\n{summary.shortest_document_title}")

        if self._documents and self._using_input_corpus:
            status = "Input corpus is connected and enriched statistics corpus is ready."
        elif self._using_input_corpus:
            status = "Input corpus is connected but empty."
        else:
            status = "Connect a Corpus input to calculate statistics."
        self._status_label.setText(status)

        rows = sorted(word_frequencies(self._documents).items(), key=lambda item: (-item[1], item[0]))[:25]
        self._table.setRowCount(len(rows))
        for row, (word, frequency) in enumerate(rows):
            self._set_item(row, 0, word)
            self._set_item(row, 1, str(frequency))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [word, str(frequency)]
            for word, frequency in sorted(word_frequencies(self._documents).items(), key=lambda item: (-item[1], item[0]))[:25]
        ]
        summary = summarize_text_statistics(self._documents)
        return {
            "summary": (
                f"Statistics: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.unique_word_count} unique words"
            ),
            "headers": ["Word", "Frequency"],
            "rows": rows,
        }
