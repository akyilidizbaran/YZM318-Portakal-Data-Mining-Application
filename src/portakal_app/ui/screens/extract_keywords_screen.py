from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.bag_of_words_screen import tokenize_for_bow
from portakal_app.ui.screens.corpus_screen import CorpusDocument, corpus_documents_from_payload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader
from portakal_app.ui.shared.readable_inputs import apply_readable_spin_box_style


@dataclass(frozen=True)
class KeywordItem:
    document: str
    keyword: str
    score: float
    frequency: int


def extract_keywords(
    documents: Sequence[CorpusDocument],
    *,
    top_n: int = 5,
    min_frequency: int = 1,
) -> tuple[KeywordItem, ...]:
    threshold = max(1, int(min_frequency))
    limit = max(1, int(top_n))
    tokenized = [tokenize_for_bow(document.text) for document in documents]
    document_count = len(tokenized)
    if document_count == 0:
        return ()

    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    rows: list[KeywordItem] = []
    for document, tokens in zip(documents, tokenized):
        counts = Counter(tokens)
        total_terms = sum(counts.values())
        if total_terms == 0:
            continue
        scored: list[KeywordItem] = []
        for term, frequency in counts.items():
            if frequency < threshold:
                continue
            tf = frequency / total_terms
            idf = math.log((1 + document_count) / (1 + document_frequency[term])) + 1
            scored.append(KeywordItem(document.title, term, tf * idf, frequency))
        rows.extend(sorted(scored, key=lambda item: (-item.score, item.keyword))[:limit])
    return tuple(rows)


class ExtractKeywordsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._keywords: tuple[KeywordItem, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Extract Keywords",
                "Extract document keywords with a lightweight TF-IDF style score.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self.apply_options()

    def sizeHint(self) -> QSize:
        return QSize(920, 660)

    def minimumSizeHint(self) -> QSize:
        return QSize(700, 500)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Top keywords", self))
        self._top_n_spinbox = QSpinBox(self)
        self._top_n_spinbox.setRange(1, 30)
        self._top_n_spinbox.setValue(5)
        apply_readable_spin_box_style(self._top_n_spinbox)
        layout.addWidget(self._top_n_spinbox)

        layout.addWidget(QLabel("Minimum frequency", self))
        self._min_frequency_spinbox = QSpinBox(self)
        self._min_frequency_spinbox.setRange(1, 999)
        self._min_frequency_spinbox.setValue(1)
        apply_readable_spin_box_style(self._min_frequency_spinbox)
        layout.addWidget(self._min_frequency_spinbox)

        self._apply_button = QPushButton("Extract", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self.apply_options)
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
        self._keyword_count_label = self._build_metric_label("Keywords", "0")
        self._top_keyword_label = self._build_metric_label("Top Keyword", "-")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._keyword_count_label, 1)
        layout.addWidget(self._top_keyword_label, 1)
        return frame

    def _build_metric_label(self, title: str, value: str) -> QLabel:
        label = QLabel(f"{title}\n{value}", self)
        label.setProperty("infoCard", True)
        label.setStyleSheet("padding: 10px;")
        label.setWordWrap(True)
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
        self._table.setHorizontalHeaderLabels(["Document", "Keyword", "Score", "Frequency"])
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
            self.apply_options()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self.apply_options()

    def current_output_payload(self) -> WorkflowPayload:
        words = tuple(dict.fromkeys(item.keyword for item in self._keywords))
        return WorkflowPayload("Words", words)

    def apply_options(self) -> tuple[KeywordItem, ...]:
        self._keywords = extract_keywords(
            self._documents,
            top_n=self._top_n_spinbox.value(),
            min_frequency=self._min_frequency_spinbox.value(),
        )
        self._render()
        self._notify_output_changed()
        return self._keywords

    def _render(self) -> None:
        top_keyword = self._keywords[0].keyword if self._keywords else "-"
        self._document_count_label.setText(f"Documents\n{len(self._documents)}")
        self._keyword_count_label.setText(f"Keywords\n{len(self._keywords)}")
        self._top_keyword_label.setText(f"Top Keyword\n{top_keyword}")

        if self._keywords and self._using_input_corpus:
            status = "Input corpus is connected and keywords are ready."
        elif self._using_input_corpus:
            status = "Input corpus is connected but no keywords match the current settings."
        else:
            status = "Connect a Corpus input to extract keywords."
        self._status_label.setText(status)

        self._table.setRowCount(len(self._keywords))
        for row, item in enumerate(self._keywords):
            self._set_item(row, 0, item.document)
            self._set_item(row, 1, item.keyword)
            self._set_item(row, 2, f"{item.score:.3f}")
            self._set_item(row, 3, str(item.frequency))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [item.document, item.keyword, f"{item.score:.3f}", str(item.frequency)]
            for item in self._keywords
        ]
        return {
            "summary": f"Extract Keywords: {len(self._keywords)} keywords from {len(self._documents)} documents",
            "headers": ["Document", "Keyword", "Score", "Frequency"],
            "rows": rows,
        }
