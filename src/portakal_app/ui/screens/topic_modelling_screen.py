from __future__ import annotations

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
from portakal_app.ui.screens.corpus_screen import CorpusDocument, corpus_documents_from_payload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader
from portakal_app.ui.shared.readable_inputs import apply_readable_spin_box_style


@dataclass(frozen=True)
class TopicSummaryRow:
    topic: str
    top_words: str


@dataclass(frozen=True)
class DocumentTopicRow:
    document: str
    topic: str
    score: float


@dataclass(frozen=True)
class TopicModelResult:
    topics: tuple[TopicSummaryRow, ...] = ()
    document_topics: tuple[DocumentTopicRow, ...] = ()
    status: str = ""


def build_topic_model(
    documents: Sequence[CorpusDocument],
    *,
    topic_count: int = 3,
    top_words: int = 6,
) -> TopicModelResult:
    clean_documents = [document for document in documents if document.text.strip()]
    if len(clean_documents) < 2:
        return TopicModelResult(status="At least 2 non-empty documents are required for topic modelling.")

    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
    except Exception as error:
        return TopicModelResult(status=f"Topic modelling backend is unavailable: {error}")

    vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", min_df=1)
    try:
        matrix = vectorizer.fit_transform(document.text for document in clean_documents)
    except ValueError as error:
        return TopicModelResult(status=f"Could not build a document-term matrix: {error}")

    terms = vectorizer.get_feature_names_out()
    if matrix.shape[1] < 2:
        return TopicModelResult(status="At least 2 unique terms are required for topic modelling.")

    safe_topic_count = max(1, min(int(topic_count), len(clean_documents), matrix.shape[1]))
    model = LatentDirichletAllocation(
        n_components=safe_topic_count,
        random_state=0,
        learning_method="batch",
        max_iter=20,
    )
    document_topic_matrix = model.fit_transform(matrix)

    topic_rows: list[TopicSummaryRow] = []
    safe_top_words = max(1, min(int(top_words), len(terms)))
    for topic_index, weights in enumerate(model.components_, start=1):
        order = weights.argsort()[::-1][:safe_top_words]
        words = ", ".join(str(terms[index]) for index in order)
        topic_rows.append(TopicSummaryRow(f"Topic {topic_index}", words))

    document_rows: list[DocumentTopicRow] = []
    for document, scores in zip(clean_documents, document_topic_matrix):
        topic_index = int(scores.argmax())
        document_rows.append(
            DocumentTopicRow(
                document.title,
                f"Topic {topic_index + 1}",
                float(scores[topic_index]),
            )
        )

    return TopicModelResult(
        topics=tuple(topic_rows),
        document_topics=tuple(document_rows),
        status=f"Built {safe_topic_count} topics from {len(clean_documents)} documents.",
    )


class TopicModellingScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._result = TopicModelResult()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Topic Modelling",
                "Discover lightweight topics with a count matrix and LDA.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_topics_panel(), 1)
        layout.addWidget(self._build_documents_panel(), 1)

        self.apply_options()

    def sizeHint(self) -> QSize:
        return QSize(980, 720)

    def minimumSizeHint(self) -> QSize:
        return QSize(740, 560)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Topics", self))
        self._topic_count_spinbox = QSpinBox(self)
        self._topic_count_spinbox.setRange(1, 10)
        self._topic_count_spinbox.setValue(3)
        apply_readable_spin_box_style(self._topic_count_spinbox)
        layout.addWidget(self._topic_count_spinbox)

        layout.addWidget(QLabel("Top words", self))
        self._top_words_spinbox = QSpinBox(self)
        self._top_words_spinbox.setRange(2, 20)
        self._top_words_spinbox.setValue(6)
        apply_readable_spin_box_style(self._top_words_spinbox)
        layout.addWidget(self._top_words_spinbox)

        self._apply_button = QPushButton("Model Topics", self)
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
        self._topic_count_label = self._build_metric_label("Topics", "0")
        self._assignment_count_label = self._build_metric_label("Assignments", "0")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._topic_count_label, 1)
        layout.addWidget(self._assignment_count_label, 1)
        return frame

    def _build_metric_label(self, title: str, value: str) -> QLabel:
        label = QLabel(f"{title}\n{value}", self)
        label.setProperty("infoCard", True)
        label.setStyleSheet("padding: 10px;")
        return label

    def _build_topics_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._status_label = QLabel("", self)
        self._status_label.setProperty("muted", True)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._topics_table = QTableWidget(0, 2, self)
        self._topics_table.setHorizontalHeaderLabels(["Topic", "Top Words"])
        self._topics_table.verticalHeader().setVisible(False)
        self._topics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._topics_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._topics_table, 1)
        return frame

    def _build_documents_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._documents_table = QTableWidget(0, 3, self)
        self._documents_table.setHorizontalHeaderLabels(["Document", "Dominant Topic", "Topic Score"])
        self._documents_table.verticalHeader().setVisible(False)
        self._documents_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._documents_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._documents_table, 1)
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
        rows = [[row.topic, row.top_words] for row in self._result.topics]
        return WorkflowPayload("Topics", rows)

    def apply_options(self) -> TopicModelResult:
        self._result = build_topic_model(
            self._documents,
            topic_count=self._topic_count_spinbox.value(),
            top_words=self._top_words_spinbox.value(),
        )
        self._render()
        self._notify_output_changed()
        return self._result

    def _render(self) -> None:
        self._document_count_label.setText(f"Documents\n{len(self._documents)}")
        self._topic_count_label.setText(f"Topics\n{len(self._result.topics)}")
        self._assignment_count_label.setText(f"Assignments\n{len(self._result.document_topics)}")
        if not self._using_input_corpus:
            self._status_label.setText("Connect a Corpus input to model topics.")
        else:
            self._status_label.setText(self._result.status or "Input corpus is connected.")

        self._topics_table.setRowCount(len(self._result.topics))
        for row, topic in enumerate(self._result.topics):
            self._topics_table.setItem(row, 0, QTableWidgetItem(topic.topic))
            self._topics_table.setItem(row, 1, QTableWidgetItem(topic.top_words))
        self._topics_table.resizeColumnsToContents()

        self._documents_table.setRowCount(len(self._result.document_topics))
        for row, item in enumerate(self._result.document_topics):
            self._documents_table.setItem(row, 0, QTableWidgetItem(item.document))
            self._documents_table.setItem(row, 1, QTableWidgetItem(item.topic))
            self._documents_table.setItem(row, 2, QTableWidgetItem(f"{item.score:.3f}"))
        self._documents_table.resizeColumnsToContents()

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [[topic.topic, topic.top_words] for topic in self._result.topics]
        return {
            "summary": f"Topic Modelling: {len(self._result.topics)} topics, {len(self._result.document_topics)} document assignments",
            "headers": ["Topic", "Top Words"],
            "rows": rows,
        }
