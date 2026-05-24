from __future__ import annotations

import re
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
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.ui.screens.corpus_screen import CorpusDocument
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


SAMPLE_BOW_DOCUMENTS: tuple[CorpusDocument, ...] = (
    CorpusDocument("Document 1", "data mining workflows explore data"),
    CorpusDocument("Document 2", "text mining converts text into features"),
    CorpusDocument("Document 3", "features summarize documents for models"),
    CorpusDocument("Document 4", "data workflows can inspect document features"),
)


@dataclass(frozen=True)
class BagOfWordsSummary:
    document_count: int
    vocabulary_size: int
    total_token_count: int
    most_frequent_term: str


def tokenize_for_bow(text: str) -> tuple[str, ...]:
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    return tuple(normalized.split())


def build_vocabulary(documents: Sequence[CorpusDocument], min_frequency: int = 1) -> tuple[str, ...]:
    threshold = max(1, min_frequency)
    counter: Counter[str] = Counter()
    for document in documents:
        counter.update(tokenize_for_bow(document.text))
    return tuple(sorted(term for term, count in counter.items() if count >= threshold))


def build_document_term_matrix(
    documents: Sequence[CorpusDocument],
    vocabulary: Sequence[str],
    binary: bool = False,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        _document_term_counts(document, vocabulary, binary=binary)
        for document in documents
    )


def _document_term_counts(
    document: CorpusDocument,
    vocabulary: Sequence[str],
    binary: bool = False,
) -> tuple[int, ...]:
    token_counts = Counter(tokenize_for_bow(document.text))
    if binary:
        return tuple(1 if token_counts[term] else 0 for term in vocabulary)
    return tuple(token_counts[term] for term in vocabulary)


def term_frequencies(matrix: Sequence[Sequence[int]], vocabulary: Sequence[str]) -> dict[str, int]:
    return {
        term: sum(row[index] for row in matrix)
        for index, term in enumerate(vocabulary)
    }


def summarize_bow(
    documents: Sequence[CorpusDocument],
    vocabulary: Sequence[str],
    matrix: Sequence[Sequence[int]],
) -> BagOfWordsSummary:
    frequencies = term_frequencies(matrix, vocabulary)
    most_frequent_term = "None"
    most_frequent_count = 0
    for term in vocabulary:
        count = frequencies[term]
        if count > most_frequent_count:
            most_frequent_term = term
            most_frequent_count = count

    return BagOfWordsSummary(
        document_count=len(documents),
        vocabulary_size=len(vocabulary),
        total_token_count=sum(sum(row) for row in matrix),
        most_frequent_term=most_frequent_term,
    )


class BagOfWordsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(SAMPLE_BOW_DOCUMENTS if documents is None else documents)
        self._vocabulary: tuple[str, ...] = ()
        self._matrix: tuple[tuple[int, ...], ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Bag of Words",
                "Convert text documents into a document-term count representation.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self.apply_options()

    def sizeHint(self) -> QSize:
        return QSize(1020, 680)

    def minimumSizeHint(self) -> QSize:
        return QSize(760, 520)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(8)

        self._binary_checkbox = QCheckBox("Binary counts", self)
        layout.addWidget(self._binary_checkbox, 0, 0)

        self._min_frequency_label = QLabel("Minimum term frequency", self)
        layout.addWidget(self._min_frequency_label, 0, 1)

        self._min_frequency_spinbox = QSpinBox(self)
        self._min_frequency_spinbox.setRange(1, 999)
        self._min_frequency_spinbox.setValue(1)
        layout.addWidget(self._min_frequency_spinbox, 0, 2)

        self._apply_button = QPushButton("Apply Bag of Words", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self.apply_options)
        layout.addWidget(self._apply_button, 0, 3)
        return frame

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._vocabulary_size_label = self._build_metric_label("Vocabulary Size", "0")
        self._total_token_count_label = self._build_metric_label("Total Tokens", "0")
        self._most_frequent_term_label = self._build_metric_label("Most Frequent Term", "None")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._vocabulary_size_label, 1)
        layout.addWidget(self._total_token_count_label, 1)
        layout.addWidget(self._most_frequent_term_label, 1)
        return frame

    def _build_metric_label(self, title: str, value: str) -> QLabel:
        label = QLabel(f"{title}\n{value}", self)
        label.setProperty("infoCard", True)
        label.setStyleSheet("padding: 10px;")
        return label

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
        self._table.setHorizontalHeaderLabels(["Document", "Total"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def apply_options(self) -> tuple[tuple[int, ...], ...]:
        self._vocabulary = build_vocabulary(
            self._documents,
            min_frequency=self._min_frequency_spinbox.value(),
        )
        self._matrix = build_document_term_matrix(
            self._documents,
            self._vocabulary,
            binary=self._binary_checkbox.isChecked(),
        )
        self._render()
        self._notify_output_changed()
        return self._matrix

    def set_documents(self, documents: Sequence[CorpusDocument]) -> None:
        self._documents = tuple(documents)
        self.apply_options()

    def _render(self) -> None:
        summary = summarize_bow(self._documents, self._vocabulary, self._matrix)
        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._vocabulary_size_label.setText(f"Vocabulary Size\n{summary.vocabulary_size}")
        self._total_token_count_label.setText(f"Total Tokens\n{summary.total_token_count}")
        self._most_frequent_term_label.setText(f"Most Frequent Term\n{summary.most_frequent_term}")
        self._status_label.setText(
            "Document-term matrix is ready."
            if self._vocabulary
            else "No vocabulary terms available for the current documents and threshold."
        )

        headers = ["Document", *self._vocabulary, "Total"]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(self._documents) + (1 if self._vocabulary else 0))

        for row, (document, counts) in enumerate(zip(self._documents, self._matrix)):
            self._set_item(row, 0, document.title)
            for column, count in enumerate(counts, start=1):
                self._set_item(row, column, str(count))
            self._set_item(row, len(headers) - 1, str(sum(counts)))

        if self._vocabulary:
            totals = term_frequencies(self._matrix, self._vocabulary)
            total_row = len(self._documents)
            self._set_item(total_row, 0, "Total Frequency")
            for column, term in enumerate(self._vocabulary, start=1):
                self._set_item(total_row, column, str(totals[term]))
            self._set_item(total_row, len(headers) - 1, str(sum(totals.values())))

        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        headers = ["Document", *self._vocabulary, "Total"]
        rows: list[list[str]] = []
        for document, counts in zip(self._documents, self._matrix):
            rows.append([document.title, *(str(count) for count in counts), str(sum(counts))])

        if self._vocabulary:
            totals = term_frequencies(self._matrix, self._vocabulary)
            rows.append(
                [
                    "Total Frequency",
                    *(str(totals[term]) for term in self._vocabulary),
                    str(sum(totals.values())),
                ]
            )

        summary = summarize_bow(self._documents, self._vocabulary, self._matrix)
        return {
            "summary": (
                f"Bag of Words: {summary.document_count} documents, "
                f"{summary.vocabulary_size} terms, "
                f"{summary.total_token_count} total tokens"
            ),
            "headers": headers,
            "rows": rows,
        }
