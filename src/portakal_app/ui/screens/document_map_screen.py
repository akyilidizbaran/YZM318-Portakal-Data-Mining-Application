from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.corpus_screen import CorpusDocument, corpus_documents_from_payload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


@dataclass(frozen=True)
class DocumentCoordinate:
    document: str
    x: float
    y: float


@dataclass(frozen=True)
class DocumentMapResult:
    coordinates: tuple[DocumentCoordinate, ...] = ()
    status: str = ""


def build_document_map(documents: Sequence[CorpusDocument]) -> DocumentMapResult:
    clean_documents = [document for document in documents if document.text.strip()]
    if len(clean_documents) < 2:
        return DocumentMapResult(status="At least 2 non-empty documents are required for a document map.")

    if len(clean_documents) == 2:
        return DocumentMapResult(
            coordinates=(
                DocumentCoordinate(clean_documents[0].title, -1.0, 0.0),
                DocumentCoordinate(clean_documents[1].title, 1.0, 0.0),
            ),
            status="Mapped 2 documents with a two-point fallback layout.",
        )

    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as error:
        return DocumentMapResult(status=f"Document map backend is unavailable: {error}")

    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b")
    try:
        matrix = vectorizer.fit_transform(document.text for document in clean_documents)
    except ValueError as error:
        return DocumentMapResult(status=f"Could not build a document-term matrix: {error}")

    if matrix.shape[1] < 2:
        return DocumentMapResult(status="At least 2 unique terms are required for a document map.")

    svd = TruncatedSVD(n_components=2, random_state=0)
    points = svd.fit_transform(matrix)
    coordinates = tuple(
        DocumentCoordinate(document.title, float(point[0]), float(point[1]))
        for document, point in zip(clean_documents, points)
    )
    return DocumentMapResult(coordinates=coordinates, status=f"Mapped {len(coordinates)} documents into 2D.")


class DocumentMapCanvas(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("panel", True)
        self.setMinimumHeight(260)
        self._coordinates: tuple[DocumentCoordinate, ...] = ()

    def set_coordinates(self, coordinates: Sequence[DocumentCoordinate]) -> None:
        self._coordinates = tuple(coordinates)
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = self.rect().adjusted(28, 24, -28, -24)
        plot_area = area.adjusted(28, 24, -28, -24)
        painter.setPen(QPen(QColor("#d7c9b5"), 1))
        painter.drawRect(area)

        if not self._coordinates:
            painter.setPen(QColor("#7e715e"))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "No document map available.")
            painter.end()
            return

        xs = [point.x for point in self._coordinates]
        ys = [point.y for point in self._coordinates]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        spread_x = max(max_x - min_x, 1e-9)
        spread_y = max(max_y - min_y, 1e-9)

        def map_point(point: DocumentCoordinate) -> QPointF:
            x = plot_area.left() + ((point.x - min_x) / spread_x) * plot_area.width()
            y = plot_area.bottom() - ((point.y - min_y) / spread_y) * plot_area.height()
            return QPointF(x, y)

        def clamp_label(rect: QRectF) -> QRectF:
            x = min(max(rect.x(), area.left() + 6), area.right() - rect.width() - 6)
            y = min(max(rect.y(), area.top() + 6), area.bottom() - rect.height() - 6)
            return QRectF(x, y, rect.width(), rect.height())

        def label_candidates(anchor: QPointF, label_width: int, label_height: int) -> tuple[QRectF, ...]:
            offsets = (
                QPointF(12, -label_height - 8),
                QPointF(12, 10),
                QPointF(-label_width - 12, -label_height - 8),
                QPointF(-label_width - 12, 10),
                QPointF(-label_width / 2, -label_height - 16),
                QPointF(-label_width / 2, 18),
                QPointF(18, -label_height / 2),
                QPointF(-label_width - 18, -label_height / 2),
            )
            return tuple(
                clamp_label(QRectF(anchor.x() + offset.x(), anchor.y() + offset.y(), label_width, label_height))
                for offset in offsets
            )

        painter.setPen(QPen(QColor("#c77a24"), 1.5))
        painter.setBrush(QColor("#f0a433"))
        metrics = QFontMetrics(painter.font())
        occupied_labels: list[QRectF] = []
        for index, point in enumerate(self._coordinates, start=1):
            mapped = map_point(point)
            painter.drawEllipse(mapped, 5, 5)

            label = f"D{index}"
            label_width = max(28, metrics.horizontalAdvance(label) + 12)
            label_height = metrics.height() + 6
            candidates = label_candidates(mapped, label_width, label_height)
            label_rect = candidates[(index - 1) % len(candidates)]
            for candidate in candidates:
                padded = candidate.adjusted(-3, -3, 3, 3)
                if not any(padded.intersects(existing) for existing in occupied_labels):
                    label_rect = candidate
                    break
            occupied_labels.append(label_rect.adjusted(-3, -3, 3, 3))

            painter.setPen(QColor("#3b2a10"))
            painter.setBrush(QColor("#fff8ee"))
            painter.drawRoundedRect(label_rect, 4, 4)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)
            painter.setPen(QPen(QColor("#c77a24"), 1.5))
            painter.setBrush(QColor("#f0a433"))
        painter.end()


class DocumentMapScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._result = DocumentMapResult()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Document Map",
                "Project corpus documents into a simple 2D similarity map.",
            )
        )
        self._status_label = QLabel("", self)
        self._status_label.setProperty("muted", True)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._canvas = DocumentMapCanvas(self)
        layout.addWidget(self._canvas, 2)
        layout.addWidget(self._build_table_panel(), 1)

        self._build_map()

    def sizeHint(self) -> QSize:
        return QSize(920, 680)

    def minimumSizeHint(self) -> QSize:
        return QSize(700, 520)

    def _build_table_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Document", "X", "Y"])
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
            self._using_input_corpus = False
            self._build_map()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self._build_map()

    def _build_map(self) -> DocumentMapResult:
        self._result = build_document_map(self._documents)
        self._render()
        return self._result

    def _render(self) -> None:
        if not self._using_input_corpus:
            self._status_label.setText("Connect a Corpus input to build a document map.")
        else:
            self._status_label.setText(self._result.status or "Input corpus is connected.")

        self._canvas.set_coordinates(self._result.coordinates)
        self._table.setRowCount(len(self._result.coordinates))
        for row, coordinate in enumerate(self._result.coordinates):
            self._table.setItem(row, 0, QTableWidgetItem(coordinate.document))
            self._table.setItem(row, 1, QTableWidgetItem(f"{coordinate.x:.3f}"))
            self._table.setItem(row, 2, QTableWidgetItem(f"{coordinate.y:.3f}"))
        self._table.resizeColumnsToContents()

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [coordinate.document, f"{coordinate.x:.3f}", f"{coordinate.y:.3f}"]
            for coordinate in self._result.coordinates
        ]
        return {
            "summary": f"Document Map: {len(self._result.coordinates)} mapped documents",
            "headers": ["Document", "X", "Y"],
            "rows": rows,
        }
