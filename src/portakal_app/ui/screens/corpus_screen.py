from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


@dataclass(frozen=True)
class CorpusDocument:
    title: str
    text: str
    source: str = "Sample"


@dataclass(frozen=True)
class CorpusSummary:
    document_count: int
    total_word_count: int
    average_words_per_document: float


SAMPLE_CORPUS: tuple[CorpusDocument, ...] = (
    CorpusDocument(
        "Document 1",
        "Portakal introduces visual workflows for exploring data mining tasks.",
    ),
    CorpusDocument(
        "Document 2",
        "A corpus is a collection of documents prepared for text analysis.",
    ),
    CorpusDocument(
        "Document 3",
        "Simple text features can help compare documents by their vocabulary.",
    ),
    CorpusDocument(
        "Document 4",
        "Preprocessing usually cleans text before models or visual summaries use it.",
    ),
    CorpusDocument(
        "Document 5",
        "Bag of words converts documents into counts that tabular widgets can inspect.",
    ),
)


def count_words(text: str) -> int:
    return len(text.split())


def summarize_corpus(documents: Iterable[CorpusDocument]) -> CorpusSummary:
    items = tuple(documents)
    document_count = len(items)
    total_word_count = sum(count_words(document.text) for document in items)
    average = total_word_count / document_count if document_count else 0.0
    return CorpusSummary(document_count, total_word_count, average)


def corpus_documents_from_payload(value: object) -> tuple[CorpusDocument, ...] | None:
    if isinstance(value, CorpusDocument):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None

    documents: list[CorpusDocument] = []
    for item in value:
        if not isinstance(item, CorpusDocument):
            return None
        documents.append(item)
    return tuple(documents)


class CorpusScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None, documents: Sequence[CorpusDocument] | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(SAMPLE_CORPUS if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._input_corpus_count = 1 if documents is not None else 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Corpus",
                "A text corpus is a collection of documents prepared for text mining workflows.",
            )
        )
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._render()

    def sizeHint(self) -> QSize:
        return QSize(860, 620)

    def minimumSizeHint(self) -> QSize:
        return QSize(680, 480)

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._total_word_count_label = self._build_metric_label("Total Words", "0")
        self._average_word_count_label = self._build_metric_label("Avg Words / Document", "0.0")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._total_word_count_label, 1)
        layout.addWidget(self._average_word_count_label, 1)
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

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Title", "Source", "Text Preview"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = tuple(SAMPLE_CORPUS)
            self._using_input_corpus = False
            self._input_corpus_count = 0
            self._render()
            self._notify_output_changed()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._input_corpus_count += 1
        if documents is None:
            documents = ()
        if self._using_input_corpus:
            self._documents = (*self._documents, *documents)
        else:
            self._documents = documents
            self._using_input_corpus = True
        self._render()
        self._notify_output_changed()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Corpus", self._documents)

    def _render(self) -> None:
        summary = summarize_corpus(self._documents)
        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._total_word_count_label.setText(f"Total Words\n{summary.total_word_count}")
        self._average_word_count_label.setText(
            f"Avg Words / Document\n{summary.average_words_per_document:.1f}"
        )
        if self._documents and self._using_input_corpus and self._input_corpus_count > 1:
            status = f"Merged {self._input_corpus_count} input corpora and displayed them in connection order."
        elif self._documents and self._using_input_corpus:
            status = "Input corpus is connected and displayed."
        elif self._documents:
            status = "Built-in sample corpus is ready."
        elif self._using_input_corpus:
            status = "Input corpus is connected but empty."
        else:
            status = "Corpus is empty. Add documents in a later import or creation step."
        self._status_label.setText(status)

        self._table.setRowCount(len(self._documents))
        for row, document in enumerate(self._documents):
            self._set_item(row, 0, document.title)
            self._set_item(row, 1, document.source)
            self._set_item(row, 2, self._preview_text(document.text))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def _preview_text(self, text: str, limit: int = 140) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 3]}..."

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [document.title, document.source, self._preview_text(document.text)]
            for document in self._documents
        ]
        summary = summarize_corpus(self._documents)
        return {
            "summary": (
                f"Corpus: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.average_words_per_document:.1f} average words/document"
            ),
            "headers": ["Title", "Source", "Text Preview"],
            "rows": rows,
        }
