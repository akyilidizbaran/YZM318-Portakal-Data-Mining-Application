from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.ui.screens.corpus_screen import CorpusDocument, count_words, summarize_corpus
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


DEFAULT_SOURCE = "Manual"


def create_default_title(index: int) -> str:
    return f"Document {max(1, index)}"


def preview_text(text: str, max_length: int = 140) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3]}..."


def make_document(title: str, text: str, source: str, index: int = 1) -> CorpusDocument:
    clean_title = title.strip() or create_default_title(index)
    clean_source = source.strip() or DEFAULT_SOURCE
    return CorpusDocument(clean_title, text, clean_source)


class CreateCorpusScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None, documents: Sequence[CorpusDocument] | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(documents or ())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Create Corpus",
                "Create a small text corpus manually with document titles, sources, and text content.",
            )
        )
        layout.addWidget(self._build_editor_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._render()

    def sizeHint(self) -> QSize:
        return QSize(920, 700)

    def minimumSizeHint(self) -> QSize:
        return QSize(720, 540)

    def _build_editor_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._title_input = QLineEdit(self)
        self._title_input.setPlaceholderText("Document title")
        layout.addWidget(QLabel("Title", self))
        layout.addWidget(self._title_input)

        self._source_input = QLineEdit(self)
        self._source_input.setPlaceholderText(DEFAULT_SOURCE)
        layout.addWidget(QLabel("Source / Category", self))
        layout.addWidget(self._source_input)

        self._text_input = QPlainTextEdit(self)
        self._text_input.setPlaceholderText("Enter document text...")
        self._text_input.setMinimumHeight(90)
        layout.addWidget(QLabel("Text Content", self))
        layout.addWidget(self._text_input)

        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(8)

        self._add_button = QPushButton("Add Document", self)
        self._add_button.setProperty("primary", True)
        self._add_button.clicked.connect(self._add_from_inputs)
        actions_layout.addWidget(self._add_button)

        self._clear_button = QPushButton("Clear Corpus", self)
        self._clear_button.clicked.connect(self.clear_corpus)
        actions_layout.addWidget(self._clear_button)
        actions_layout.addStretch(1)

        layout.addLayout(actions_layout)
        return frame

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

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Title", "Source", "Text Preview", "Words"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def _add_from_inputs(self) -> None:
        self.add_document(
            self._title_input.text(),
            self._text_input.toPlainText(),
            self._source_input.text(),
        )
        self._title_input.clear()
        self._text_input.clear()
        self._title_input.setFocus()

    def add_document(self, title: str, text: str, source: str = "") -> CorpusDocument:
        document = make_document(title, text, source, index=len(self._documents) + 1)
        self._documents = (*self._documents, document)
        self._render()
        self._notify_output_changed()
        return document

    def clear_corpus(self) -> None:
        self._documents = ()
        self._render()
        self._notify_output_changed()

    def _render(self) -> None:
        summary = summarize_corpus(self._documents)
        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._total_word_count_label.setText(f"Total Words\n{summary.total_word_count}")
        self._average_word_count_label.setText(
            f"Avg Words / Document\n{summary.average_words_per_document:.1f}"
        )
        self._status_label.setText(
            "Manual corpus is ready for later text mining workflow steps."
            if self._documents
            else "No documents created yet. Add a title, optional source, and text content."
        )

        self._table.setRowCount(len(self._documents))
        for row, document in enumerate(self._documents):
            self._set_item(row, 0, document.title)
            self._set_item(row, 1, document.source)
            self._set_item(row, 2, preview_text(document.text))
            self._set_item(row, 3, str(count_words(document.text)))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [document.title, document.source, preview_text(document.text), str(count_words(document.text))]
            for document in self._documents
        ]
        summary = summarize_corpus(self._documents)
        return {
            "summary": (
                f"Manual Corpus: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.average_words_per_document:.1f} average words/document"
            ),
            "headers": ["Title", "Source", "Text Preview", "Words"],
            "rows": rows,
        }
