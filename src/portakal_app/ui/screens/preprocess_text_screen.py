from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.ui.screens.corpus_screen import (
    CorpusDocument,
    SAMPLE_CORPUS,
    count_words,
)
from portakal_app.ui.screens.create_corpus_screen import preview_text
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


ENGLISH_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "is",
        "are",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "this",
        "that",
    }
)
TURKISH_STOPWORDS = frozenset(
    {"ve", "veya", "bir", "bu", "şu", "ile", "için", "de", "da", "mi", "mı"}
)
DEFAULT_STOPWORDS = ENGLISH_STOPWORDS | TURKISH_STOPWORDS


@dataclass(frozen=True)
class PreprocessOptions:
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_extra_whitespace: bool = True
    remove_stopwords: bool = False


@dataclass(frozen=True)
class PreprocessingSummary:
    document_count: int
    total_original_words: int
    total_processed_words: int
    removed_word_count: int


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)


def remove_stopwords(text: str, stopwords: Iterable[str] = DEFAULT_STOPWORDS) -> str:
    stopword_set = {word.lower() for word in stopwords}
    return " ".join(word for word in text.split() if word.lower() not in stopword_set)


_normalize_whitespace = normalize_whitespace
_remove_punctuation = remove_punctuation
_remove_numbers = remove_numbers
_remove_stopwords = remove_stopwords


def preprocess_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    normalize_whitespace: bool = True,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
) -> str:
    processed = text
    if lowercase:
        processed = processed.lower()
    if remove_punctuation:
        processed = _remove_punctuation(processed)
    if remove_numbers:
        processed = _remove_numbers(processed)
    if normalize_whitespace:
        processed = _normalize_whitespace(processed)
    if remove_stopwords:
        processed = _remove_stopwords(processed, stopwords)
        if normalize_whitespace:
            processed = _normalize_whitespace(processed)
    return processed


def preprocess_documents(
    documents: Sequence[CorpusDocument],
    options: PreprocessOptions = PreprocessOptions(),
) -> tuple[CorpusDocument, ...]:
    return tuple(
        CorpusDocument(
            document.title,
            preprocess_text(
                document.text,
                lowercase=options.lowercase,
                remove_punctuation=options.remove_punctuation,
                remove_numbers=options.remove_numbers,
                remove_stopwords=options.remove_stopwords,
                normalize_whitespace=options.remove_extra_whitespace,
            ),
            document.source,
        )
        for document in documents
    )


def summarize_preprocessing(
    original_documents: Sequence[CorpusDocument],
    processed_documents: Sequence[CorpusDocument],
) -> PreprocessingSummary:
    document_count = len(original_documents)
    total_original_words = sum(count_words(document.text) for document in original_documents)
    total_processed_words = sum(count_words(document.text) for document in processed_documents)
    removed_word_count = max(0, total_original_words - total_processed_words)
    return PreprocessingSummary(
        document_count,
        total_original_words,
        total_processed_words,
        removed_word_count,
    )


class PreprocessTextScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._original_documents = tuple(SAMPLE_CORPUS if documents is None else documents)
        self._processed_documents: tuple[CorpusDocument, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Preprocess Text",
                "Apply lightweight preprocessing operations to text documents.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self.apply_preprocessing()

    def sizeHint(self) -> QSize:
        return QSize(980, 680)

    def minimumSizeHint(self) -> QSize:
        return QSize(740, 520)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(8)

        self._lowercase_checkbox = QCheckBox("Lowercase", self)
        self._lowercase_checkbox.setChecked(True)
        self._punctuation_checkbox = QCheckBox("Remove punctuation", self)
        self._punctuation_checkbox.setChecked(True)
        self._numbers_checkbox = QCheckBox("Remove numbers", self)
        self._whitespace_checkbox = QCheckBox("Remove extra whitespace", self)
        self._whitespace_checkbox.setChecked(True)
        self._stopwords_checkbox = QCheckBox("Remove simple stopwords", self)

        layout.addWidget(self._lowercase_checkbox, 0, 0)
        layout.addWidget(self._punctuation_checkbox, 0, 1)
        layout.addWidget(self._numbers_checkbox, 0, 2)
        layout.addWidget(self._whitespace_checkbox, 1, 0)
        layout.addWidget(self._stopwords_checkbox, 1, 1)

        self._apply_button = QPushButton("Apply Preprocessing", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self.apply_preprocessing)
        layout.addWidget(self._apply_button, 1, 2)
        return frame

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._original_words_label = self._build_metric_label("Original Words", "0")
        self._processed_words_label = self._build_metric_label("Processed Words", "0")
        self._removed_words_label = self._build_metric_label("Removed Words", "0")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._original_words_label, 1)
        layout.addWidget(self._processed_words_label, 1)
        layout.addWidget(self._removed_words_label, 1)
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

        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(
            [
                "Title",
                "Original Text",
                "Preprocessed Text",
                "Original Words",
                "Processed Words",
            ]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def _current_options(self) -> PreprocessOptions:
        return PreprocessOptions(
            lowercase=self._lowercase_checkbox.isChecked(),
            remove_punctuation=self._punctuation_checkbox.isChecked(),
            remove_numbers=self._numbers_checkbox.isChecked(),
            remove_extra_whitespace=self._whitespace_checkbox.isChecked(),
            remove_stopwords=self._stopwords_checkbox.isChecked(),
        )

    def apply_preprocessing(self) -> tuple[CorpusDocument, ...]:
        self._processed_documents = preprocess_documents(self._original_documents, self._current_options())
        self._render()
        self._notify_output_changed()
        return self._processed_documents

    def set_documents(self, documents: Sequence[CorpusDocument]) -> None:
        self._original_documents = tuple(documents)
        self.apply_preprocessing()

    def _render(self) -> None:
        summary = summarize_preprocessing(self._original_documents, self._processed_documents)
        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._original_words_label.setText(f"Original Words\n{summary.total_original_words}")
        self._processed_words_label.setText(f"Processed Words\n{summary.total_processed_words}")
        self._removed_words_label.setText(f"Removed Words\n{summary.removed_word_count}")

        self._status_label.setText(
            "Preprocessed sample corpus is ready."
            if self._original_documents
            else "No documents available for preprocessing."
        )

        self._table.setRowCount(len(self._original_documents))
        document_pairs = zip(self._original_documents, self._processed_documents)
        for row, (original, processed) in enumerate(document_pairs):
            self._set_item(row, 0, original.title)
            self._set_item(row, 1, preview_text(original.text))
            self._set_item(row, 2, preview_text(processed.text))
            self._set_item(row, 3, str(count_words(original.text)))
            self._set_item(row, 4, str(count_words(processed.text)))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [
                original.title,
                preview_text(original.text),
                preview_text(processed.text),
                str(count_words(original.text)),
                str(count_words(processed.text)),
            ]
            for original, processed in zip(self._original_documents, self._processed_documents)
        ]
        summary = summarize_preprocessing(self._original_documents, self._processed_documents)
        return {
            "summary": (
                f"Preprocess Text: {summary.document_count} documents, "
                f"{summary.total_original_words} original words, "
                f"{summary.total_processed_words} processed words"
            ),
            "headers": [
                "Title",
                "Original Text",
                "Preprocessed Text",
                "Original Words",
                "Processed Words",
            ],
            "rows": rows,
        }
