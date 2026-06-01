from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.bag_of_words_screen import tokenize_for_bow
from portakal_app.ui.screens.corpus_screen import CorpusDocument, corpus_documents_from_payload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader
from portakal_app.ui.shared.readable_inputs import apply_readable_spin_box_style


@dataclass(frozen=True)
class WordListItem:
    word: str
    frequency: int
    document_count: int
    source: str = "Corpus"


STOPWORDS = frozenset({"the", "to", "of", "and", "a", "in"})
UPDATE_MODES = ("Only input", "Union", "Intersection", "Ignore input")
SORT_MODES = ("Sort by Frequency", "Sort by Documents", "Sort Alphabetically")


def normalize_update_mode(update_mode: str) -> str:
    if update_mode == "Only Corpus":
        return "Only input"
    if update_mode == "Only Custom":
        return "Ignore input"
    return update_mode


def build_word_list(
    documents: Sequence[CorpusDocument],
    min_frequency: int = 1,
) -> tuple[WordListItem, ...]:
    threshold = max(1, int(min_frequency))
    frequencies: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()

    for document in documents:
        tokens = tokenize_for_bow(document.text)
        frequencies.update(tokens)
        document_counts.update(set(tokens))

    items = [
        WordListItem(word, frequency, document_counts[word])
        for word, frequency in frequencies.items()
        if frequency >= threshold
    ]
    return tuple(sorted(items, key=lambda item: (-item.frequency, item.word)))


def build_updated_word_list(
    corpus_items: Sequence[WordListItem],
    custom_words: Sequence[str],
    *,
    input_words: Sequence[str] | None = None,
    update_mode: str = "Only input",
    search_filter: str = "",
    sort_mode: str = "Sort by Frequency",
) -> tuple[WordListItem, ...]:
    corpus_by_word = {item.word: item for item in corpus_items}
    input_set = (
        {word.strip().lower() for word in input_words if word.strip()}
        if input_words is not None
        else set(corpus_by_word)
    )
    custom_set = {word.strip().lower() for word in custom_words if word.strip()}
    mode = normalize_update_mode(update_mode)

    if mode == "Intersection":
        words = input_set.intersection(custom_set)
    elif mode == "Union":
        words = input_set.union(custom_set)
    elif mode == "Ignore input":
        words = set(custom_set)
    elif not input_set and custom_set:
        words = set(custom_set)
    else:
        words = set(input_set)

    query = search_filter.strip().lower()
    rows: list[WordListItem] = []
    for word in words:
        if query and query not in word:
            continue
        corpus_item = corpus_by_word.get(word)
        sources: list[str] = []
        if corpus_item is not None:
            sources.append(corpus_item.source)
        if word in input_set and input_words is not None:
            sources.append("Input")
        if word in custom_set:
            sources.append("Custom")
        if corpus_item is None:
            rows.append(WordListItem(word, 0, 0, " + ".join(dict.fromkeys(sources)) or "Custom"))
            continue
        source = " + ".join(dict.fromkeys(sources)) or corpus_item.source
        rows.append(WordListItem(word, corpus_item.frequency, corpus_item.document_count, source))

    if sort_mode == "Sort by Documents":
        return tuple(sorted(rows, key=lambda item: (-item.document_count, -item.frequency, item.word)))
    if sort_mode == "Sort Alphabetically":
        return tuple(sorted(rows, key=lambda item: item.word))
    return tuple(sorted(rows, key=lambda item: (-item.frequency, item.word)))


class WordListScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._input_words: tuple[str, ...] = ()
        self._custom_words: tuple[str, ...] = ()
        self._corpus_items: tuple[WordListItem, ...] = ()
        self._items: tuple[WordListItem, ...] = ()
        self._generated_dataset_service = GeneratedDatasetService()
        self._output_dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Word List",
                "Build a corpus vocabulary table and combine it with a custom word library.",
            )
        )
        layout.addWidget(self._build_custom_word_panel())
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self.apply_filter()

    def sizeHint(self) -> QSize:
        return QSize(860, 640)

    def minimumSizeHint(self) -> QSize:
        return QSize(680, 500)

    def _build_custom_word_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Custom Word List", self))

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self._custom_word_input = QLineEdit(self)
        self._custom_word_input.setPlaceholderText("Add word...")
        self._custom_word_input.returnPressed.connect(self.add_custom_word_from_input)
        controls.addWidget(self._custom_word_input, 1)

        self._add_word_button = QPushButton("Add Word", self)
        self._add_word_button.setProperty("primary", True)
        self._add_word_button.clicked.connect(self.add_custom_word_from_input)
        controls.addWidget(self._add_word_button)

        self._remove_word_button = QPushButton("Remove Selected", self)
        self._remove_word_button.clicked.connect(self.remove_selected_custom_words)
        controls.addWidget(self._remove_word_button)

        self._clear_words_button = QPushButton("Clear List", self)
        self._clear_words_button.clicked.connect(self.clear_custom_words)
        controls.addWidget(self._clear_words_button)
        layout.addLayout(controls)

        self._custom_table = QTableWidget(0, 1, self)
        self._custom_table.setHorizontalHeaderLabels(["Word"])
        self._custom_table.verticalHeader().setVisible(False)
        self._custom_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._custom_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._custom_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._custom_table.horizontalHeader().setStretchLastSection(True)
        self._custom_table.setMaximumHeight(130)
        layout.addWidget(self._custom_table)
        return frame

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Minimum frequency", self))
        self._min_frequency_spinbox = QSpinBox(self)
        self._min_frequency_spinbox.setRange(1, 999)
        self._min_frequency_spinbox.setValue(1)
        apply_readable_spin_box_style(self._min_frequency_spinbox)
        layout.addWidget(self._min_frequency_spinbox)

        layout.addWidget(QLabel("Update mode", self))
        self._update_mode_combo = QComboBox(self)
        self._update_mode_combo.addItems(UPDATE_MODES)
        self._update_mode_combo.setCurrentText("Only input")
        self._update_mode_combo.currentTextChanged.connect(self.apply_filter)
        layout.addWidget(self._update_mode_combo)

        layout.addWidget(QLabel("Search", self))
        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText("Filter words...")
        self._search_input.textChanged.connect(self.apply_filter)
        layout.addWidget(self._search_input, 1)

        layout.addWidget(QLabel("Sort", self))
        self._sort_combo = QComboBox(self)
        self._sort_combo.addItems(SORT_MODES)
        self._sort_combo.currentTextChanged.connect(self.apply_filter)
        layout.addWidget(self._sort_combo)

        self._apply_button = QPushButton("Apply Filter", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self.apply_filter)
        layout.addWidget(self._apply_button)
        layout.addStretch(1)
        return frame

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._word_count_label = self._build_metric_label("Words", "0")
        self._total_frequency_label = self._build_metric_label("Total Frequency", "0")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._word_count_label, 1)
        layout.addWidget(self._total_frequency_label, 1)
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
        self._table.setHorizontalHeaderLabels(["Word", "Frequency", "Documents", "Source"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._notify_output_changed)
        layout.addWidget(self._table, 1)
        return frame

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = ()
            self._using_input_corpus = False
            self._input_words = ()
            self.apply_filter()
            return

        documents = corpus_documents_from_payload(payload.value)
        if documents is None and payload.port_label == "Words":
            self._input_words = tuple(
                dict.fromkeys(token for item in payload.value if isinstance(item, str) for token in tokenize_for_bow(item))
            ) if isinstance(payload.value, Sequence) and not isinstance(payload.value, (str, bytes, bytearray)) else tuple(tokenize_for_bow(str(payload.value)))
            self.apply_filter()
            return

        if payload.port_label == "Corpus" or documents is not None:
            self._documents = () if documents is None else documents
            self._using_input_corpus = documents is not None
            self.apply_filter()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Data", self._output_dataset)

    def current_output_payloads(self) -> dict[str, WorkflowPayload | None]:
        return {
            "Words": WorkflowPayload("Words", self.active_words()),
            "Selected Words": WorkflowPayload("Selected Words", self.selected_words()),
            "Data": WorkflowPayload("Data", self._output_dataset),
        }

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def active_words(self) -> tuple[str, ...]:
        return tuple(item.word for item in self._items)

    def selected_words(self) -> tuple[str, ...]:
        selected_rows = sorted(
            index.row()
            for index in self._table.selectionModel().selectedRows()
            if 0 <= index.row() < len(self._items)
        )
        return tuple(self._items[row].word for row in selected_rows)

    def add_custom_word_from_input(self) -> None:
        self.add_custom_word(self._custom_word_input.text())
        self._custom_word_input.clear()

    def add_custom_word(self, text: str) -> tuple[str, ...]:
        words = tokenize_for_bow(text)
        if not words:
            return self._custom_words
        merged = sorted(set(self._custom_words).union(words))
        self._custom_words = tuple(merged)
        self._render_custom_words()
        self.apply_filter()
        return self._custom_words

    def remove_selected_custom_words(self) -> None:
        selected_rows = {
            index.row()
            for index in self._custom_table.selectionModel().selectedRows()
            if 0 <= index.row() < len(self._custom_words)
        }
        if not selected_rows:
            return
        self._custom_words = tuple(
            word for row, word in enumerate(self._custom_words) if row not in selected_rows
        )
        self._render_custom_words()
        self.apply_filter()

    def clear_custom_words(self) -> None:
        self._custom_words = ()
        self._render_custom_words()
        self.apply_filter()

    def set_update_mode(self, mode: str) -> None:
        normalized_mode = normalize_update_mode(mode)
        if normalized_mode in UPDATE_MODES:
            self._update_mode_combo.setCurrentText(normalized_mode)
        else:
            self.apply_filter()

    def apply_filter(self, *_args: object) -> tuple[WordListItem, ...]:
        self._corpus_items = build_word_list(self._documents, self._min_frequency_spinbox.value())
        self._items = build_updated_word_list(
            self._corpus_items,
            self._custom_words,
            input_words=self._input_words if self._input_words else None,
            update_mode=self._update_mode_combo.currentText(),
            search_filter=self._search_input.text(),
            sort_mode=self._sort_combo.currentText(),
        )
        self._output_dataset = self._build_output_dataset()
        self._render()
        self._notify_output_changed()
        return self._items

    def _build_output_dataset(self) -> DatasetHandle:
        dataframe = pl.DataFrame(
            {
                "Word": [item.word for item in self._items],
                "Frequency": [item.frequency for item in self._items],
                "Documents": [item.document_count for item in self._items],
                "Source": [item.source for item in self._items],
            },
            schema={
                "Word": pl.Utf8,
                "Frequency": pl.Int64,
                "Documents": pl.Int64,
                "Source": pl.Utf8,
            },
        )
        return self._generated_dataset_service.build_dataset(
            dataframe,
            dataset_id="text-word-list-output",
            display_name="Word List",
            file_name="text-word-list-output.csv",
            role_overrides={
                "Word": "meta",
                "Frequency": "feature",
                "Documents": "feature",
                "Source": "meta",
            },
            annotations={
                "generated_by": "word-list",
                "update_mode": self._update_mode_combo.currentText(),
                "custom_words": list(self._custom_words),
            },
        )

    def _render_custom_words(self) -> None:
        self._custom_table.setRowCount(len(self._custom_words))
        for row, word in enumerate(self._custom_words):
            self._custom_table.setItem(row, 0, QTableWidgetItem(word))
        self._custom_table.resizeColumnsToContents()

    def _render(self) -> None:
        total_frequency = sum(item.frequency for item in self._items)
        self._document_count_label.setText(f"Documents\n{len(self._documents)}")
        self._word_count_label.setText(f"Words\n{len(self._items)}")
        self._total_frequency_label.setText(f"Total Frequency\n{total_frequency}")

        stopword_warning = self._stopword_warning()
        if self._items and self._using_input_corpus:
            status = "Input corpus is connected and word list is ready."
        elif self._items:
            status = "Custom word list is ready. Connect a Corpus input to add frequency counts."
        elif self._using_input_corpus:
            status = "Input corpus is connected but no words match the current settings."
        else:
            status = "Connect a Corpus input or add custom words to build a word list."
        if stopword_warning:
            status = f"{status} {stopword_warning}"
        self._status_label.setText(status)

        self._table.setRowCount(len(self._items))
        for row, item in enumerate(self._items):
            self._set_item(row, 0, item.word)
            self._set_item(row, 1, str(item.frequency))
            self._set_item(row, 2, str(item.document_count))
            self._set_item(row, 3, item.source)
        self._table.resizeColumnsToContents()

    def _stopword_warning(self) -> str:
        top_words = [item.word for item in self._corpus_items[:5]]
        if any(word in STOPWORDS for word in top_words):
            return "Common stopwords detected. Consider using Preprocess Text with stopword removal."
        return ""

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [item.word, str(item.frequency), str(item.document_count), item.source]
            for item in self._items
        ]
        return {
            "summary": (
                f"Word List: {len(self._items)} words, "
                f"{sum(item.frequency for item in self._items)} total frequency"
            ),
            "headers": ["Word", "Frequency", "Documents", "Source"],
            "rows": rows,
        }
