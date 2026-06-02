from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
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


WORD_CLOUD_COLORS = ("#6f1d9a", "#e09b00", "#72826f", "#7b3bb2", "#c15f00", "#5e7468")
WORD_CLOUD_MUTED_COLOR = "#6f7569"


@dataclass(frozen=True)
class CloudWord:
    word: str
    frequency: int
    rank: int
    font_size: int
    color: str
    rect: QRectF


def cloud_word_frequencies(documents: Sequence[CorpusDocument]) -> tuple[tuple[str, int], ...]:
    counter: Counter[str] = Counter()
    for document in documents:
        counter.update(tokenize_for_bow(document.text))
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def word_cloud_text_column(dataset: DatasetHandle | None) -> str:
    if dataset is None:
        return ""
    columns = list(dataset.dataframe.columns)
    if not columns:
        return ""
    normalized = {column.lower().replace("_", " ").strip(): column for column in columns}
    for exact_name in ("text", "content", "full content", "text preview", "preview", "body", "article", "message"):
        if exact_name in normalized:
            return normalized[exact_name]
    for column in columns:
        lowered = column.lower().replace("_", " ")
        if any(part in lowered for part in ("text", "content", "preview", "body", "article", "message")):
            return column
    return ""


def word_cloud_documents_from_dataset(dataset: DatasetHandle | None) -> tuple[CorpusDocument, ...]:
    text_column = word_cloud_text_column(dataset)
    if dataset is None or not text_column:
        return ()

    dataframe = dataset.dataframe
    title_column = next(
        (column for column in dataframe.columns if column.lower() in {"title", "name", "document"}),
        "",
    )
    source_column = next(
        (column for column in dataframe.columns if column.lower() in {"source", "path", "file"}),
        "",
    )
    documents: list[CorpusDocument] = []
    for index, row in enumerate(dataframe.iter_rows(named=True), start=1):
        text = "" if row.get(text_column) is None else str(row.get(text_column))
        if not text.strip():
            continue
        title = str(row.get(title_column) or f"Row {index}") if title_column else f"Row {index}"
        source = str(row.get(source_column) or dataset.display_name) if source_column else dataset.display_name
        documents.append(CorpusDocument(title, text, source))
    return tuple(documents)


def word_cloud_frequencies_from_dataset(dataset: DatasetHandle | None) -> tuple[tuple[str, int], ...]:
    if dataset is None:
        return ()

    dataframe = dataset.dataframe
    columns = list(dataframe.columns)
    normalized = {column.lower().replace("_", " ").strip(): column for column in columns}
    word_column = next(
        (
            normalized[name]
            for name in ("word", "term", "token", "keyword")
            if name in normalized
        ),
        "",
    )
    if not word_column:
        return ()

    frequency_column = next(
        (
            normalized[name]
            for name in ("frequency", "count", "weight", "score")
            if name in normalized
        ),
        "",
    )
    counter: Counter[str] = Counter()
    for row in dataframe.iter_rows(named=True):
        raw_word = row.get(word_column)
        if raw_word is None:
            continue
        word = str(raw_word).strip().lower()
        if not word:
            continue
        frequency = 1
        if frequency_column:
            try:
                frequency = max(1, int(float(row.get(frequency_column) or 0)))
            except (TypeError, ValueError):
                frequency = 1
        counter[word] += frequency
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def scaled_word_size(frequency: int, min_frequency: int, max_frequency: int) -> int:
    if max_frequency <= min_frequency:
        return 24
    min_size = 11
    max_size = 58
    ratio = math.sqrt((frequency - min_frequency) / (max_frequency - min_frequency))
    return min_size + int(ratio * (max_size - min_size))


class WordCloudCanvas(QWidget):
    hoveredWordChanged = Signal(str)
    selectedWordChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(420)
        self._frequencies: tuple[tuple[str, int], ...] = ()
        self._words: tuple[CloudWord, ...] = ()
        self._hovered_word = ""
        self._selected_word = ""
        self._color_words = True

    def set_words(self, frequencies: Sequence[tuple[str, int]], *, color_words: bool = True) -> None:
        self._frequencies = tuple(frequencies)
        self._color_words = color_words
        self._layout_words()
        self.update()

    def set_hovered_word(self, word: str) -> None:
        if self._hovered_word == word:
            return
        self._hovered_word = word
        self.update()

    def set_selected_word(self, word: str) -> None:
        if self._selected_word == word:
            return
        self._selected_word = word
        self.update()

    def words(self) -> tuple[CloudWord, ...]:
        return self._words

    def resizeEvent(self, event) -> None:
        self._layout_words()
        super().resizeEvent(event)

    def mouseMoveEvent(self, event) -> None:
        word = self._word_at(event.position())
        if word is None:
            if self._hovered_word:
                self._hovered_word = ""
                self.hoveredWordChanged.emit("")
                self.update()
            QToolTip.hideText()
            return
        if self._hovered_word != word.word:
            self._hovered_word = word.word
            self.hoveredWordChanged.emit(word.word)
            self.update()
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"{word.word}\nFrequency: {word.frequency}\nRank: {word.rank}",
            self,
        )

    def leaveEvent(self, event) -> None:
        if self._hovered_word:
            self._hovered_word = ""
            self.hoveredWordChanged.emit("")
            self.update()
        QToolTip.hideText()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        word = self._word_at(event.position())
        if word is None:
            return super().mousePressEvent(event)
        self._selected_word = word.word
        self.selectedWordChanged.emit(word.word)
        self.update()
        event.accept()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = self.rect().adjusted(10, 10, -10, -10)
        if not self._words:
            painter.setPen(QColor("#7e715e"))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "No words to display.")
            painter.end()
            return

        for word in self._words:
            hovered = word.word == self._hovered_word
            selected = word.word == self._selected_word
            font = QFont(self.font())
            font.setPointSize(word.font_size)
            font.setWeight(QFont.Weight.Bold if selected or hovered else QFont.Weight.DemiBold)
            painter.setFont(font)

            color = QColor(word.color if self._color_words else WORD_CLOUD_MUTED_COLOR)
            if hovered:
                color = color.darker(125)
            if selected:
                padding_rect = word.rect.adjusted(-7, -4, 7, 4)
                painter.setBrush(QColor("#fff1ce"))
                painter.setPen(QPen(QColor("#d88922"), 1.5))
                painter.drawRoundedRect(padding_rect, 7, 7)
                color = QColor("#7a3f12")
            painter.setPen(color)
            painter.drawText(word.rect, Qt.AlignmentFlag.AlignCenter, word.word)
        painter.end()

    def _word_at(self, position) -> CloudWord | None:
        point = position.toPoint() if hasattr(position, "toPoint") else position
        for word in reversed(self._words):
            if word.rect.adjusted(-5, -5, 5, 5).contains(point):
                return word
        return None

    def _layout_words(self) -> None:
        if not self._frequencies:
            self._words = ()
            return

        area = QRectF(self.rect()).adjusted(16, 16, -16, -16)
        if area.width() <= 40 or area.height() <= 40:
            area = QRectF(16, 16, max(float(self.width() - 32), 640.0), max(float(self.height() - 32), 240.0))

        max_frequency = max(frequency for _word, frequency in self._frequencies)
        min_frequency = min(frequency for _word, frequency in self._frequencies)
        occupied: list[QRectF] = []
        placed: list[CloudWord] = []

        for index, (word, frequency) in enumerate(self._frequencies, start=1):
            placement = self._place_word(area, word, frequency, index, min_frequency, max_frequency, occupied)
            if placement is None:
                continue
            rect, font_size = placement
            occupied.append(rect.adjusted(-8, -5, 8, 5))
            placed.append(
                CloudWord(
                    word=word,
                    frequency=frequency,
                    rank=index,
                    font_size=font_size,
                    color=WORD_CLOUD_COLORS[(index - 1) % len(WORD_CLOUD_COLORS)],
                    rect=rect,
                )
            )
        self._words = tuple(placed)

    def _place_word(
        self,
        area: QRectF,
        word: str,
        frequency: int,
        rank: int,
        min_frequency: int,
        max_frequency: int,
        occupied: Sequence[QRectF],
    ) -> tuple[QRectF, int] | None:
        preferred_size = scaled_word_size(frequency, min_frequency, max_frequency)
        word_count = len(self._frequencies)
        density_scale = 0.72 if word_count >= 80 else 0.86 if word_count >= 50 else 1.0
        density_cap = 42 if word_count >= 80 else 50 if word_count >= 50 else 58
        max_size = min(
            max(8, int(preferred_size * density_scale)),
            density_cap,
            max(18, int(area.height() * 0.28)),
            max(18, int(area.width() * 0.2)),
        )
        min_size = 8 if len(self._frequencies) >= 80 else 10
        for font_size in range(max_size, min_size - 1, -2):
            font = QFont(self.font())
            font.setPointSize(font_size)
            font.setWeight(QFont.Weight.DemiBold)
            metrics = QFontMetrics(font)
            width = metrics.horizontalAdvance(word) + 10
            height = metrics.height() + 4
            if width > area.width() or height > area.height():
                continue
            rect = self._find_rect(area, width, height, occupied, rank)
            if rect is not None:
                return rect, font_size
        return None

    def _find_rect(
        self,
        area: QRectF,
        width: int,
        height: int,
        occupied: Sequence[QRectF],
        rank: int,
    ) -> QRectF | None:
        center = area.center()
        center_candidate = QRectF(center.x() - width / 2, center.y() - height / 2, width, height)
        if not self._collides(center_candidate, occupied) and self._contains_rect(area, center_candidate):
            return center_candidate

        radius_x = max(1.0, (area.width() - width) / 2)
        radius_y = max(1.0, (area.height() - height) / 2)
        max_steps = 5200
        golden_angle = 2.399963229728653
        for step in range(1, max_steps + 1):
            fraction = math.sqrt(step / max_steps)
            angle = (step * golden_angle) + (rank * 0.31)
            x = center.x() + math.cos(angle) * radius_x * fraction - width / 2
            y = center.y() + math.sin(angle) * radius_y * fraction - height / 2
            candidate = QRectF(x, y, width, height)
            if self._contains_rect(area, candidate) and not self._collides(candidate, occupied):
                return candidate

        return self._find_grid_rect(area, width, height, occupied, rank)

    def _find_grid_rect(
        self,
        area: QRectF,
        width: int,
        height: int,
        occupied: Sequence[QRectF],
        rank: int,
    ) -> QRectF | None:
        x_step = max(6, int(width * 0.35))
        y_step = max(6, int(height * 0.45))
        x_positions = list(range(int(area.left()), max(int(area.right() - width), int(area.left())) + 1, x_step))
        y_positions = list(range(int(area.top()), max(int(area.bottom() - height), int(area.top())) + 1, y_step))
        if rank % 2:
            x_positions.reverse()
        if rank % 3:
            y_positions.reverse()
        for y in y_positions:
            for x in x_positions:
                candidate = QRectF(float(x), float(y), width, height)
                if self._contains_rect(area, candidate) and not self._collides(candidate, occupied):
                    return candidate
        return None

    def _contains_rect(self, area: QRectF, rect: QRectF) -> bool:
        return (
            rect.left() >= area.left()
            and rect.right() <= area.right()
            and rect.top() >= area.top()
            and rect.bottom() <= area.bottom()
        )

    def _collides(self, rect: QRectF, occupied: Sequence[QRectF]) -> bool:
        return any(rect.intersects(existing) for existing in occupied)


class WordCloudScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._using_input_data = False
        self._data_text_column = ""
        self._data_status_message = ""
        self._data_frequencies: tuple[tuple[str, int], ...] = ()
        self._frequencies: tuple[tuple[str, int], ...] = ()
        self._generated_dataset_service = GeneratedDatasetService()
        self._word_counts_dataset: DatasetHandle | None = None
        self._hovered_word = ""
        self._selected_word = ""
        self._syncing_selection = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Word Cloud",
                "Show frequent corpus words with larger text for higher frequency.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_cloud_panel(), 3)
        layout.addWidget(self._build_table_panel(), 1)

        self.refresh_cloud()

    def sizeHint(self) -> QSize:
        return QSize(920, 680)

    def minimumSizeHint(self) -> QSize:
        return QSize(700, 520)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Top words", self))
        self._top_words_spinbox = QSpinBox(self)
        self._top_words_spinbox.setRange(5, 100)
        self._top_words_spinbox.setValue(40)
        apply_readable_spin_box_style(self._top_words_spinbox)
        layout.addWidget(self._top_words_spinbox)

        layout.addWidget(QLabel("Sort", self))
        self._sort_combo = QComboBox(self)
        self._sort_combo.addItems(("Frequency", "Alphabetical"))
        self._sort_combo.currentTextChanged.connect(self.refresh_cloud)
        layout.addWidget(self._sort_combo)

        self._color_words_checkbox = QCheckBox("Color words", self)
        self._color_words_checkbox.setChecked(True)
        self._color_words_checkbox.toggled.connect(self._render)
        layout.addWidget(self._color_words_checkbox)

        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.setProperty("primary", True)
        self._refresh_button.clicked.connect(self.refresh_cloud)
        layout.addWidget(self._refresh_button)

        self._summary_label = QLabel("", self)
        self._summary_label.setProperty("muted", True)
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label, 1)
        return frame

    def _build_cloud_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        self._cloud_canvas = WordCloudCanvas(self)
        self._cloud_canvas.hoveredWordChanged.connect(self._set_hovered_word)
        self._cloud_canvas.selectedWordChanged.connect(self._select_word)
        layout.addWidget(self._cloud_canvas, 1)
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
        self._table.setMouseTracking(True)
        self._table.cellEntered.connect(self._handle_table_hover)
        self._table.itemSelectionChanged.connect(self._handle_table_selection_changed)
        layout.addWidget(self._table, 1)
        return frame

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = ()
            self._using_input_corpus = False
            self._using_input_data = False
            self._data_text_column = ""
            self._data_status_message = ""
            self._data_frequencies = ()
            self.refresh_cloud()
            return

        documents = corpus_documents_from_payload(payload.value)
        if documents is not None:
            self._documents = documents
            self._using_input_corpus = True
            self._using_input_data = False
            self._data_text_column = ""
            self._data_status_message = ""
            self._data_frequencies = ()
            self.refresh_cloud()
            return

        if payload.dataset is not None:
            dataset = payload.dataset
            self._data_frequencies = word_cloud_frequencies_from_dataset(dataset)
            self._documents = () if self._data_frequencies else word_cloud_documents_from_dataset(dataset)
            self._using_input_corpus = False
            self._using_input_data = True
            self._data_text_column = word_cloud_text_column(dataset)
            if self._data_frequencies:
                self._data_status_message = ""
            elif self._data_text_column:
                self._data_status_message = ""
            else:
                self._data_status_message = "No text column found for Word Cloud."
            self.refresh_cloud()
            return

        self._documents = ()
        self._using_input_corpus = False
        self._using_input_data = False
        self._data_text_column = ""
        self._data_status_message = ""
        self._data_frequencies = ()
        self.refresh_cloud()

    def refresh_cloud(self, *_args: object) -> tuple[tuple[str, int], ...]:
        frequencies = self._data_frequencies if self._using_input_data and self._data_frequencies else cloud_word_frequencies(self._documents)
        if self._sort_combo.currentText() == "Alphabetical":
            frequencies = tuple(sorted(frequencies, key=lambda item: item[0]))
        self._frequencies = frequencies[: self._top_words_spinbox.value()]
        if self._selected_word and self._selected_word not in {word for word, _frequency in self._frequencies}:
            self._selected_word = ""
        self._word_counts_dataset = self._build_word_counts_dataset()
        self._render()
        self._notify_output_changed()
        return self._frequencies

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Corpus", self._selected_word_documents())

    def current_output_payloads(self) -> dict[str, WorkflowPayload | None]:
        selected_word = self.selected_word()
        return {
            "Corpus": WorkflowPayload("Corpus", self._selected_word_documents()),
            "Selected Word": WorkflowPayload("Selected Word", selected_word),
            "Word Counts": WorkflowPayload("Word Counts", self._word_counts_dataset),
        }

    def selected_word(self) -> str:
        return self._selected_word

    def _selected_word_documents(self) -> tuple[CorpusDocument, ...]:
        word = self.selected_word()
        if not word:
            return self._documents
        return tuple(
            document
            for document in self._documents
            if word in tokenize_for_bow(document.text)
        )

    def _build_word_counts_dataset(self) -> DatasetHandle:
        dataframe = pl.DataFrame(
            {
                "Word": [word for word, _frequency in self._frequencies],
                "Frequency": [frequency for _word, frequency in self._frequencies],
            },
            schema={"Word": pl.Utf8, "Frequency": pl.Int64},
        )
        return self._generated_dataset_service.build_dataset(
            dataframe,
            dataset_id="word-cloud-word-counts",
            display_name="Word Cloud Word Counts",
            file_name="word-cloud-word-counts.csv",
            role_overrides={"Word": "meta", "Frequency": "feature"},
            annotations={"generated_by": "word-cloud"},
        )

    def _render(self, *_args: object) -> None:
        total_frequency = sum(frequency for _word, frequency in self._frequencies)
        self._summary_label.setText(
            f"{len(self._documents)} documents, {len(self._frequencies)} displayed words, {total_frequency} total frequency."
        )

        if self._data_status_message:
            status = self._data_status_message
        elif self._frequencies and self._using_input_data and self._data_frequencies:
            status = "Input data is connected. Using Word/Frequency columns."
        elif self._frequencies and self._using_input_data:
            status = f"Input data is connected. Using text column '{self._data_text_column}'."
        elif self._using_input_data:
            status = "Input data is connected but contains no displayable words."
        elif self._frequencies and self._using_input_corpus:
            status = "Input corpus is connected and word cloud is ready."
        elif self._using_input_corpus:
            status = "Input corpus is connected but contains no displayable words."
        else:
            status = "Connect a Corpus or Data input to build a word cloud."
        self._status_label.setText(status)

        self._cloud_canvas.set_words(
            self._frequencies,
            color_words=self._color_words_checkbox.isChecked(),
        )
        self._cloud_canvas.set_hovered_word(self._hovered_word)
        self._cloud_canvas.set_selected_word(self._selected_word)
        self._table.setRowCount(len(self._frequencies))
        for row, (word, frequency) in enumerate(self._frequencies):
            self._set_item(row, 0, word)
            self._set_item(row, 1, str(frequency))
        self._table.resizeColumnsToContents()
        self._sync_table_selection()

    def _handle_table_hover(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._frequencies):
            self._set_hovered_word(self._frequencies[row][0])

    def _handle_table_selection_changed(self) -> None:
        if self._syncing_selection:
            return
        selection_model = self._table.selectionModel()
        selected_rows = selection_model.selectedRows() if selection_model is not None else []
        if not selected_rows:
            self._select_word("")
            return
        row = selected_rows[0].row()
        if 0 <= row < len(self._frequencies):
            self._select_word(self._frequencies[row][0])

    def _set_hovered_word(self, word: str) -> None:
        self._hovered_word = word
        self._cloud_canvas.set_hovered_word(word)
        self._highlight_table_word(word)

    def _select_word(self, word: str) -> None:
        if self._selected_word == word:
            return
        self._selected_word = word
        self._cloud_canvas.set_selected_word(word)
        self._sync_table_selection()
        self._notify_output_changed()

    def _sync_table_selection(self) -> None:
        self._syncing_selection = True
        try:
            if not self._selected_word:
                self._table.clearSelection()
                return
            for row, (word, _frequency) in enumerate(self._frequencies):
                if word == self._selected_word:
                    self._table.selectRow(row)
                    self._table.scrollToItem(self._table.item(row, 0))
                    return
            self._table.clearSelection()
        finally:
            self._syncing_selection = False

    def _highlight_table_word(self, word: str) -> None:
        for row in range(self._table.rowCount()):
            is_match = bool(word) and self._table.item(row, 0).text() == word
            for column in range(self._table.columnCount()):
                item = self._table.item(row, column)
                if item is None:
                    continue
                item.setBackground(QColor("#fff1ce") if is_match else QColor("#ffffff"))

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [[word, str(frequency)] for word, frequency in self._frequencies]
        return {
            "summary": (
                f"Word Cloud: {len(self._frequencies)} displayed words, "
                f"{sum(frequency for _word, frequency in self._frequencies)} total frequency"
            ),
            "headers": ["Word", "Frequency"],
            "rows": rows,
        }
