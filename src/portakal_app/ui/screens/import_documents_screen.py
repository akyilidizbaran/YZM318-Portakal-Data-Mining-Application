from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.ui.screens.corpus_screen import CorpusDocument, count_words
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".txt", ".md", ".csv"})
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1254", "latin-1")


@dataclass(frozen=True)
class DocumentImportResult:
    documents: tuple[CorpusDocument, ...]
    errors: tuple[str, ...] = ()


def is_supported_document_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS


def read_text_file(path: str | Path) -> str:
    file_path = Path(path)
    last_error: UnicodeDecodeError | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    return file_path.read_text(encoding="utf-8")


def document_from_path(path: str | Path) -> tuple[CorpusDocument, ...]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _documents_from_csv(file_path)
    text = read_text_file(file_path)
    return (CorpusDocument(file_path.name, text, str(file_path)),)


def import_documents_from_paths(paths: Sequence[str | Path]) -> DocumentImportResult:
    documents: list[CorpusDocument] = []
    errors: list[str] = []

    for path in paths:
        file_path = Path(path)
        if not is_supported_document_path(file_path):
            errors.append(f"Unsupported file type: {file_path.name}")
            continue
        try:
            documents.extend(document_from_path(file_path))
        except OSError as error:
            errors.append(f"Could not read {file_path.name}: {error}")
        except UnicodeError as error:
            errors.append(f"Could not decode {file_path.name}: {error}")
        except csv.Error as error:
            errors.append(f"Could not parse {file_path.name}: {error}")

    return DocumentImportResult(tuple(documents), tuple(errors))


def _documents_from_csv(path: Path) -> tuple[CorpusDocument, ...]:
    text = read_text_file(path)
    rows: list[CorpusDocument] = []
    reader = csv.reader(text.splitlines())
    for index, row in enumerate(reader, start=1):
        joined = " ".join(cell.strip() for cell in row if cell.strip())
        if not joined:
            continue
        rows.append(CorpusDocument(f"{path.name} row {index}", joined, str(path)))
    return tuple(rows)


class ImportDocumentsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents: tuple[CorpusDocument, ...] = ()
        self._errors: tuple[str, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Import Documents",
                "Import local .txt, .md, or .csv files into a lightweight text corpus.",
            )
        )
        layout.addWidget(self._build_actions_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._render()

    def sizeHint(self) -> QSize:
        return QSize(920, 640)

    def minimumSizeHint(self) -> QSize:
        return QSize(700, 500)

    def _build_actions_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._import_button = QPushButton("Import Documents", self)
        self._import_button.setProperty("primary", True)
        self._import_button.clicked.connect(self._choose_files)
        layout.addWidget(self._import_button)

        self._summary_label = QLabel("", self)
        self._summary_label.setProperty("muted", True)
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label, 1)
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

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Title", "Source", "Text Preview", "Words"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def _choose_files(self) -> None:
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Import Documents",
            "",
            "Text Documents (*.txt *.md *.csv);;All Files (*.*)",
        )
        if not paths:
            self._status_label.setText("No files selected.")
            return
        self.import_paths(paths)

    def import_paths(self, paths: Sequence[str | Path]) -> DocumentImportResult:
        result = import_documents_from_paths(paths)
        self._documents = result.documents
        self._errors = result.errors
        self._render()
        return result

    def _render(self) -> None:
        word_count = sum(count_words(document.text) for document in self._documents)
        self._summary_label.setText(
            f"{len(self._documents)} documents imported, {word_count} total words."
        )
        if self._errors:
            self._status_label.setText("Import completed with warnings: " + " | ".join(self._errors))
        elif self._documents:
            self._status_label.setText("Imported documents are ready for a later corpus workflow step.")
        else:
            self._status_label.setText("No documents imported yet.")

        self._table.setRowCount(len(self._documents))
        for row, document in enumerate(self._documents):
            self._set_item(row, 0, document.title)
            self._set_item(row, 1, document.source)
            self._set_item(row, 2, self._preview_text(document.text))
            self._set_item(row, 3, str(count_words(document.text)))
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
            [document.title, document.source, self._preview_text(document.text), str(count_words(document.text))]
            for document in self._documents
        ]
        word_count = sum(count_words(document.text) for document in self._documents)
        return {
            "summary": f"Imported Documents: {len(self._documents)} documents, {word_count} total words",
            "headers": ["Title", "Source", "Text Preview", "Words"],
            "rows": rows,
        }
