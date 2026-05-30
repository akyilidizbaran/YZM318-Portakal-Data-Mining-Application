from __future__ import annotations

from collections.abc import Sequence
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
from portakal_app.ui.screens.bag_of_words_screen import tokenize_for_bow
from portakal_app.ui.screens.corpus_screen import CorpusDocument, corpus_documents_from_payload
from portakal_app.ui.screens.create_corpus_screen import preview_text
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


POSITIVE_WORDS = frozenset(
    {
        "accurate",
        "amazing",
        "başarılı",
        "best",
        "better",
        "excellent",
        "faydalı",
        "good",
        "great",
        "happy",
        "improved",
        "iyi",
        "love",
        "olumlu",
        "perfect",
        "positive",
        "strong",
        "useful",
        "yararlı",
    }
)
NEGATIVE_WORDS = frozenset(
    {
        "bad",
        "başarısız",
        "broken",
        "error",
        "failed",
        "hata",
        "kötü",
        "negative",
        "olumsuz",
        "poor",
        "problem",
        "sad",
        "slow",
        "terrible",
        "weak",
        "worse",
        "wrong",
        "zararlı",
    }
)


@dataclass(frozen=True)
class SentimentResult:
    document: str
    positive_score: int
    negative_score: int
    net_score: int
    label: str
    text_preview: str


def analyze_sentiment(documents: Sequence[CorpusDocument]) -> tuple[SentimentResult, ...]:
    rows: list[SentimentResult] = []
    for document in documents:
        tokens = tokenize_for_bow(document.text)
        positive = sum(1 for token in tokens if token in POSITIVE_WORDS)
        negative = sum(1 for token in tokens if token in NEGATIVE_WORDS)
        net = positive - negative
        if net > 0:
            label = "Positive"
        elif net < 0:
            label = "Negative"
        else:
            label = "Neutral"
        rows.append(SentimentResult(document.title, positive, negative, net, label, preview_text(document.text)))
    return tuple(rows)


class SentimentAnalysisScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._results: tuple[SentimentResult, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Sentiment Analysis",
                "Score documents with a small Turkish and English sentiment lexicon.",
            )
        )
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._analyze()

    def sizeHint(self) -> QSize:
        return QSize(920, 640)

    def minimumSizeHint(self) -> QSize:
        return QSize(700, 500)

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._positive_count_label = self._build_metric_label("Positive", "0")
        self._negative_count_label = self._build_metric_label("Negative", "0")
        self._neutral_count_label = self._build_metric_label("Neutral", "0")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._positive_count_label, 1)
        layout.addWidget(self._negative_count_label, 1)
        layout.addWidget(self._neutral_count_label, 1)
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

        self._table = QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(
            ["Document", "Positive", "Negative", "Net", "Label", "Text Preview"]
        )
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
            self._analyze()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self._analyze()

    def _analyze(self) -> tuple[SentimentResult, ...]:
        self._results = analyze_sentiment(self._documents)
        self._render()
        return self._results

    def _render(self) -> None:
        positive = sum(1 for result in self._results if result.label == "Positive")
        negative = sum(1 for result in self._results if result.label == "Negative")
        neutral = sum(1 for result in self._results if result.label == "Neutral")

        self._document_count_label.setText(f"Documents\n{len(self._documents)}")
        self._positive_count_label.setText(f"Positive\n{positive}")
        self._negative_count_label.setText(f"Negative\n{negative}")
        self._neutral_count_label.setText(f"Neutral\n{neutral}")

        if self._results and self._using_input_corpus:
            status = "Input corpus is connected and sentiment labels are ready."
        elif self._using_input_corpus:
            status = "Input corpus is connected but empty."
        else:
            status = "Connect a Corpus input to analyze sentiment."
        self._status_label.setText(status)

        self._table.setRowCount(len(self._results))
        for row, result in enumerate(self._results):
            self._set_item(row, 0, result.document)
            self._set_item(row, 1, str(result.positive_score))
            self._set_item(row, 2, str(result.negative_score))
            self._set_item(row, 3, str(result.net_score))
            self._set_item(row, 4, result.label)
            self._set_item(row, 5, result.text_preview)
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [
                result.document,
                str(result.positive_score),
                str(result.negative_score),
                str(result.net_score),
                result.label,
                result.text_preview,
            ]
            for result in self._results
        ]
        return {
            "summary": f"Sentiment Analysis: {len(self._results)} documents scored",
            "headers": ["Document", "Positive", "Negative", "Net", "Label", "Text Preview"],
            "rows": rows,
        }
