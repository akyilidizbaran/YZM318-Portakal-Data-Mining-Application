from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
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
from portakal_app.ui.screens.bag_of_words_screen import BagOfWordsCorpus
from portakal_app.ui.screens.corpus_screen import CorpusDocument, corpus_documents_from_payload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader
from portakal_app.ui.shared.readable_inputs import apply_readable_spin_box_style


METHOD_LDA = "lda"
METHOD_LSI = "lsi"
METHOD_NMF = "nmf"
METHOD_HDP = "hdp"

METHOD_LABELS = {
    METHOD_LDA: "Latent Dirichlet Allocation",
    METHOD_LSI: "Latent Semantic Indexing",
    METHOD_NMF: "Non-negative Matrix Factorization",
    METHOD_HDP: "Hierarchical Dirichlet Process",
}

METHOD_NOTES = {
    METHOD_LDA: "LDA models each document as a mixture of topics over count-based word frequencies.",
    METHOD_LSI: "LSI uses SVD on TF-IDF. Scores are latent coordinates and may be negative.",
    METHOD_NMF: "NMF factorizes TF-IDF into non-negative document-topic and topic-word weights.",
    METHOD_HDP: "HDP is not available in this Portakal build. LDA, LSI, and NMF are supported.",
}


@dataclass(frozen=True)
class TopicSummaryRow:
    topic: str
    top_words: str
    weights: str = ""


@dataclass(frozen=True)
class DocumentTopicRow:
    document: str
    topic: str
    score: float
    scores: tuple[float, ...] = ()


@dataclass(frozen=True)
class TopicModelResult:
    topics: tuple[TopicSummaryRow, ...] = ()
    document_topics: tuple[DocumentTopicRow, ...] = ()
    topic_columns: tuple[str, ...] = ()
    status: str = ""
    method: str = METHOD_LDA
    document_count: int = 0
    vocabulary_size: int = 0
    evaluation: str = ""
    topic_scores: tuple[tuple[float, ...], ...] = ()
    matrix_source: str = ""


def topic_documents_from_payload(value: object) -> tuple[CorpusDocument, ...] | None:
    documents = corpus_documents_from_payload(value)
    if documents is not None:
        return documents

    if isinstance(value, Mapping):
        for key in ("documents", "corpus", "items", "rows", "value"):
            if key in value:
                return topic_documents_from_payload(value[key])
        return None

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None

    converted: list[CorpusDocument] = []
    for index, item in enumerate(value, start=1):
        document = _document_from_unknown(item, index)
        if document is None:
            return None
        converted.append(document)
    return tuple(converted)


def topic_bag_of_words_from_payload(value: object) -> BagOfWordsCorpus | None:
    if isinstance(value, BagOfWordsCorpus):
        return value
    if isinstance(value, Mapping):
        for key in ("bag_of_words", "bow", "document_term_matrix", "value"):
            nested = value.get(key)
            if isinstance(nested, BagOfWordsCorpus):
                return nested
    return None


def build_topic_model(
    documents: Sequence[CorpusDocument],
    *,
    method: str = METHOD_LDA,
    topic_count: int = 3,
    top_words: int = 8,
    max_features: int = 1000,
    min_df: int = 1,
    max_df: float = 1.0,
    random_state: int = 42,
    bag_of_words: BagOfWordsCorpus | None = None,
) -> TopicModelResult:
    clean_documents = tuple(document for document in documents if document.text.strip())
    if method == METHOD_HDP:
        return TopicModelResult(
            status=METHOD_NOTES[METHOD_HDP],
            method=METHOD_HDP,
            document_count=len(clean_documents),
        )

    if len(clean_documents) < 2:
        return TopicModelResult(
            status="At least 2 non-empty documents are required for topic modelling.",
            method=method,
            document_count=len(clean_documents),
        )

    method = method if method in (METHOD_LDA, METHOD_LSI, METHOD_NMF) else METHOD_LDA
    reused_matrix = _matrix_from_bag_of_words(bag_of_words, clean_documents, method)
    if reused_matrix is None:
        try:
            matrix, terms = _vectorize_documents(
                clean_documents,
                method=method,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
            )
        except Exception as error:
            return TopicModelResult(
                status=f"Could not build a document-term matrix: {error}",
                method=method,
                document_count=len(clean_documents),
            )
        matrix_source = "Vectorized text internally."
    else:
        matrix, terms = reused_matrix
        matrix_source = "Reused Bag of Words document-term matrix."

    if matrix.shape[1] < 2:
        return TopicModelResult(
            status="At least 2 unique terms are required for topic modelling.",
            method=method,
            document_count=len(clean_documents),
            vocabulary_size=matrix.shape[1],
        )

    safe_topic_count = _safe_topic_count(topic_count, len(clean_documents), matrix.shape[1])
    try:
        if method == METHOD_LDA:
            result = _fit_lda(clean_documents, matrix, terms, safe_topic_count, top_words, random_state)
        elif method == METHOD_LSI:
            result = _fit_lsi(clean_documents, matrix, terms, safe_topic_count, top_words, random_state)
        else:
            result = _fit_nmf(clean_documents, matrix, terms, safe_topic_count, top_words, random_state)
    except Exception as error:
        return TopicModelResult(
            status=f"Topic modelling failed: {error}",
            method=method,
            document_count=len(clean_documents),
            vocabulary_size=matrix.shape[1],
        )

    return TopicModelResult(
        topics=result.topics,
        document_topics=result.document_topics,
        topic_columns=result.topic_columns,
        status=(
            f"Built {len(result.topics)} topics from {len(clean_documents)} documents. "
            "Preprocessing before topic modelling usually improves topic quality."
        ),
        method=method,
        document_count=len(clean_documents),
        vocabulary_size=len(terms),
        evaluation=result.evaluation,
        topic_scores=result.topic_scores,
        matrix_source=matrix_source,
    )


def _fit_lda(
    documents: Sequence[CorpusDocument],
    matrix: Any,
    terms: Sequence[str],
    topic_count: int,
    top_words: int,
    random_state: int,
) -> TopicModelResult:
    from sklearn.decomposition import LatentDirichletAllocation

    model = LatentDirichletAllocation(
        n_components=topic_count,
        random_state=random_state,
        learning_method="batch",
        max_iter=30,
    )
    document_topic_matrix = model.fit_transform(matrix)
    topics = _component_topic_rows(model.components_, terms, top_words)
    document_topics = _document_topic_rows(documents, document_topic_matrix)
    evaluation = _lda_evaluation(model, matrix)
    return _model_result(METHOD_LDA, topics, document_topics, document_topic_matrix, evaluation)


def _fit_lsi(
    documents: Sequence[CorpusDocument],
    matrix: Any,
    terms: Sequence[str],
    topic_count: int,
    top_words: int,
    random_state: int,
) -> TopicModelResult:
    from sklearn.decomposition import TruncatedSVD

    safe_topic_count = min(topic_count, max(1, min(matrix.shape) - 1)) if min(matrix.shape) > 1 else 1
    model = TruncatedSVD(n_components=safe_topic_count, random_state=random_state)
    document_topic_matrix = model.fit_transform(matrix)
    topics = _lsi_topic_rows(model.components_, terms, top_words)
    document_topics = _document_topic_rows(documents, document_topic_matrix)
    explained = float(getattr(model, "explained_variance_ratio_", ()).sum())
    evaluation = f"Explained variance: {explained:.3f}"
    return _model_result(METHOD_LSI, topics, document_topics, document_topic_matrix, evaluation)


def _fit_nmf(
    documents: Sequence[CorpusDocument],
    matrix: Any,
    terms: Sequence[str],
    topic_count: int,
    top_words: int,
    random_state: int,
) -> TopicModelResult:
    from sklearn.decomposition import NMF

    model = NMF(
        n_components=topic_count,
        init="nndsvda",
        random_state=random_state,
        max_iter=400,
    )
    document_topic_matrix = model.fit_transform(matrix)
    topics = _component_topic_rows(model.components_, terms, top_words)
    document_topics = _document_topic_rows(documents, document_topic_matrix)
    evaluation = f"Reconstruction error: {model.reconstruction_err_:.3f}"
    return _model_result(METHOD_NMF, topics, document_topics, document_topic_matrix, evaluation)


def _model_result(
    method: str,
    topics: Sequence[TopicSummaryRow],
    document_topics: Sequence[DocumentTopicRow],
    document_topic_matrix: Any,
    evaluation: str,
) -> TopicModelResult:
    topic_columns = tuple(f"Topic {index}" for index in range(1, len(topics) + 1))
    topic_scores = tuple(tuple(float(value) for value in row) for row in document_topic_matrix.tolist())
    return TopicModelResult(
        topics=tuple(topics),
        document_topics=tuple(document_topics),
        topic_columns=topic_columns,
        method=method,
        evaluation=evaluation,
        topic_scores=topic_scores,
    )


def _vectorize_documents(
    documents: Sequence[CorpusDocument],
    *,
    method: str,
    max_features: int,
    min_df: int,
    max_df: float,
) -> tuple[Any, tuple[str, ...]]:
    if method == METHOD_LDA:
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            stop_words="english",
            max_features=max(10, int(max_features)),
            min_df=max(1, int(min_df)),
            max_df=_safe_max_df(max_df),
        )
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            stop_words="english",
            max_features=max(10, int(max_features)),
            min_df=max(1, int(min_df)),
            max_df=_safe_max_df(max_df),
        )

    matrix = vectorizer.fit_transform(document.text for document in documents)
    return matrix, tuple(str(term) for term in vectorizer.get_feature_names_out())


def _matrix_from_bag_of_words(
    bag_of_words: BagOfWordsCorpus | None,
    documents: Sequence[CorpusDocument],
    method: str,
) -> tuple[Any, tuple[str, ...]] | None:
    if bag_of_words is None:
        return None
    if len(bag_of_words.documents) != len(documents):
        return None
    if not bag_of_words.vocabulary or not bag_of_words.matrix:
        return None
    if method == METHOD_LDA and bag_of_words.matrix_kind != "count":
        return None

    try:
        import numpy as np
    except Exception:
        return None

    matrix = np.asarray(bag_of_words.matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(documents) or matrix.shape[1] != len(bag_of_words.vocabulary):
        return None
    if method == METHOD_NMF and bool((matrix < 0).any()):
        return None
    return matrix, bag_of_words.vocabulary


def _component_topic_rows(components: Any, terms: Sequence[str], top_words: int) -> tuple[TopicSummaryRow, ...]:
    safe_top_words = max(1, min(int(top_words), len(terms)))
    rows: list[TopicSummaryRow] = []
    for topic_index, weights in enumerate(components, start=1):
        order = weights.argsort()[::-1][:safe_top_words]
        words = ", ".join(str(terms[index]) for index in order)
        formatted_weights = ", ".join(f"{terms[index]}={float(weights[index]):.3f}" for index in order)
        rows.append(TopicSummaryRow(f"Topic {topic_index}", words, formatted_weights))
    return tuple(rows)


def _lsi_topic_rows(components: Any, terms: Sequence[str], top_words: int) -> tuple[TopicSummaryRow, ...]:
    safe_top_words = max(1, min(int(top_words), len(terms)))
    rows: list[TopicSummaryRow] = []
    side_count = max(1, safe_top_words // 2)
    for topic_index, weights in enumerate(components, start=1):
        positive_order = [index for index in weights.argsort()[::-1] if weights[index] > 0][:side_count]
        negative_order = [index for index in weights.argsort() if weights[index] < 0][: safe_top_words - len(positive_order)]
        order = positive_order + negative_order
        if not order:
            order = list(weights.argsort()[::-1][:safe_top_words])
        signed_words = ", ".join(_signed_term(terms[index], float(weights[index])) for index in order)
        formatted_weights = ", ".join(f"{terms[index]}={float(weights[index]):+.3f}" for index in order)
        rows.append(TopicSummaryRow(f"Topic {topic_index}", signed_words, formatted_weights))
    return tuple(rows)


def _document_topic_rows(documents: Sequence[CorpusDocument], document_topic_matrix: Any) -> tuple[DocumentTopicRow, ...]:
    rows: list[DocumentTopicRow] = []
    for document, scores in zip(documents, document_topic_matrix):
        topic_index = int(scores.argmax())
        row_scores = tuple(float(value) for value in scores)
        rows.append(
            DocumentTopicRow(
                document.title,
                f"Topic {topic_index + 1}",
                float(scores[topic_index]),
                row_scores,
            )
        )
    return tuple(rows)


def _lda_evaluation(model: Any, matrix: Any) -> str:
    try:
        log_likelihood = float(model.score(matrix))
        perplexity = float(model.perplexity(matrix))
    except Exception:
        return ""
    return f"Log likelihood: {log_likelihood:.3f}; perplexity: {perplexity:.3f}"


def _safe_topic_count(requested: int, document_count: int, vocabulary_size: int) -> int:
    return max(1, min(int(requested), document_count, vocabulary_size))


def _safe_max_df(value: float) -> float:
    return max(0.1, min(float(value), 1.0))


def _signed_term(term: str, weight: float) -> str:
    prefix = "+" if weight >= 0 else "-"
    return f"{prefix}{term}"


def _document_from_unknown(item: object, index: int) -> CorpusDocument | None:
    if isinstance(item, CorpusDocument):
        return item

    if isinstance(item, Mapping):
        text = _mapping_text(item)
        if text is None:
            return None
        title = str(item.get("title") or item.get("name") or item.get("id") or f"Document {index}")
        source = str(item.get("source") or item.get("category") or item.get("path") or "Input")
        return CorpusDocument(title, text, source)

    text = _attribute_text(item)
    if text is None:
        return None
    title = str(getattr(item, "title", "") or getattr(item, "name", "") or f"Document {index}")
    source = str(getattr(item, "source", "") or getattr(item, "category", "") or "Input")
    return CorpusDocument(title, text, source)


def _mapping_text(item: Mapping[object, object]) -> str | None:
    for key in ("text", "content", "preview", "summary"):
        value = item.get(key)
        if value:
            return str(value)
    tokens = item.get("tokens")
    if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes, bytearray)):
        return " ".join(str(token) for token in tokens)
    return None


def _attribute_text(item: object) -> str | None:
    for name in ("text", "content", "preview", "summary"):
        value = getattr(item, name, None)
        if value:
            return str(value)
    tokens = getattr(item, "tokens", None)
    if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes, bytearray)):
        return " ".join(str(token) for token in tokens)
    return None


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
        self._bag_of_words: BagOfWordsCorpus | None = None
        self._result = TopicModelResult()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Topic Modelling",
                "Discover topics with LDA, LSI, or NMF from corpus documents.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_topics_panel(), 1)
        layout.addWidget(self._build_documents_panel(), 1)

        self.apply_options()

    def sizeHint(self) -> QSize:
        return QSize(1040, 760)

    def minimumSizeHint(self) -> QSize:
        return QSize(760, 580)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(8)

        layout.addWidget(QLabel("Method", self), 0, 0)
        self._method_combo = QComboBox(self)
        self._method_combo.addItem(METHOD_LABELS[METHOD_LDA], METHOD_LDA)
        self._method_combo.addItem(METHOD_LABELS[METHOD_LSI], METHOD_LSI)
        self._method_combo.addItem(METHOD_LABELS[METHOD_NMF], METHOD_NMF)
        self._method_combo.currentIndexChanged.connect(self._update_method_note)
        layout.addWidget(self._method_combo, 0, 1, 1, 3)

        layout.addWidget(QLabel("Topics", self), 0, 4)
        self._topic_count_spinbox = QSpinBox(self)
        self._topic_count_spinbox.setRange(1, 20)
        self._topic_count_spinbox.setValue(3)
        apply_readable_spin_box_style(self._topic_count_spinbox)
        layout.addWidget(self._topic_count_spinbox, 0, 5)

        layout.addWidget(QLabel("Top words", self), 0, 6)
        self._top_words_spinbox = QSpinBox(self)
        self._top_words_spinbox.setRange(2, 30)
        self._top_words_spinbox.setValue(8)
        apply_readable_spin_box_style(self._top_words_spinbox)
        layout.addWidget(self._top_words_spinbox, 0, 7)

        layout.addWidget(QLabel("Max features", self), 1, 0)
        self._max_features_spinbox = QSpinBox(self)
        self._max_features_spinbox.setRange(10, 5000)
        self._max_features_spinbox.setSingleStep(100)
        self._max_features_spinbox.setValue(1000)
        apply_readable_spin_box_style(self._max_features_spinbox)
        layout.addWidget(self._max_features_spinbox, 1, 1)

        layout.addWidget(QLabel("Min document frequency", self), 1, 2)
        self._min_df_spinbox = QSpinBox(self)
        self._min_df_spinbox.setRange(1, 20)
        self._min_df_spinbox.setValue(1)
        apply_readable_spin_box_style(self._min_df_spinbox)
        layout.addWidget(self._min_df_spinbox, 1, 3)

        layout.addWidget(QLabel("Max document frequency", self), 1, 4)
        self._max_df_spinbox = QSpinBox(self)
        self._max_df_spinbox.setRange(10, 100)
        self._max_df_spinbox.setSuffix("%")
        self._max_df_spinbox.setValue(100)
        apply_readable_spin_box_style(self._max_df_spinbox)
        layout.addWidget(self._max_df_spinbox, 1, 5)

        self._apply_button = QPushButton("Model Topics", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self.apply_options)
        layout.addWidget(self._apply_button, 1, 6, 1, 2)

        self._method_note_label = QLabel("", self)
        self._method_note_label.setProperty("muted", True)
        self._method_note_label.setWordWrap(True)
        layout.addWidget(self._method_note_label, 2, 0, 1, 8)
        self._update_method_note()
        return frame

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._method_label = self._build_metric_label("Method", "LDA")
        self._topic_count_label = self._build_metric_label("Topics", "0")
        self._vocabulary_size_label = self._build_metric_label("Vocabulary", "0")
        self._assignment_count_label = self._build_metric_label("Assignments", "0")
        self._evaluation_label = self._build_metric_label("Evaluation", "None")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._method_label, 1)
        layout.addWidget(self._topic_count_label, 1)
        layout.addWidget(self._vocabulary_size_label, 1)
        layout.addWidget(self._assignment_count_label, 1)
        layout.addWidget(self._evaluation_label, 1)
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

        self._topics_table = QTableWidget(0, 3, self)
        self._topics_table.setHorizontalHeaderLabels(["Topic", "Top Words", "Weights / Notes"])
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

    def _update_method_note(self) -> None:
        method = self._method_combo.currentData() if hasattr(self, "_method_combo") else METHOD_LDA
        if hasattr(self, "_method_note_label"):
            self._method_note_label.setText(METHOD_NOTES.get(str(method), METHOD_NOTES[METHOD_LDA]))

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = ()
            self._using_input_corpus = False
            self._bag_of_words = None
            self.apply_options()
            return

        self._bag_of_words = topic_bag_of_words_from_payload(payload.value)
        documents = topic_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self.apply_options()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload(
            "Topics",
            {
                "method": METHOD_LABELS.get(self._result.method, self._result.method),
                "topics": [[row.topic, row.top_words, row.weights] for row in self._result.topics],
                "document_topics": [
                    [row.document, row.topic, row.score, *row.scores]
                    for row in self._result.document_topics
                ],
                "topic_columns": list(self._result.topic_columns),
                "status": self._result.status,
                "evaluation": self._result.evaluation,
                "matrix_source": self._result.matrix_source,
            },
        )

    def apply_options(self) -> TopicModelResult:
        method = str(self._method_combo.currentData() or METHOD_LDA)
        self._result = build_topic_model(
            self._documents,
            method=method,
            topic_count=self._topic_count_spinbox.value(),
            top_words=self._top_words_spinbox.value(),
            max_features=self._max_features_spinbox.value(),
            min_df=self._min_df_spinbox.value(),
            max_df=self._max_df_spinbox.value() / 100,
            bag_of_words=self._bag_of_words,
        )
        self._render()
        self._notify_output_changed()
        return self._result

    def _render(self) -> None:
        method_label = METHOD_LABELS.get(self._result.method, METHOD_LABELS[METHOD_LDA])
        self._document_count_label.setText(f"Documents\n{len(self._documents)}")
        self._method_label.setText(f"Method\n{method_label}")
        self._topic_count_label.setText(f"Topics\n{len(self._result.topics)}")
        self._vocabulary_size_label.setText(f"Vocabulary\n{self._result.vocabulary_size}")
        self._assignment_count_label.setText(f"Assignments\n{len(self._result.document_topics)}")
        self._evaluation_label.setText(f"Evaluation\n{self._result.evaluation or 'None'}")

        if not self._using_input_corpus:
            self._status_label.setText("Connect a Corpus input to model topics.")
        elif not self._documents:
            self._status_label.setText("Input corpus is connected but no usable text documents were found.")
        else:
            status = self._result.status or "Input corpus is connected."
            if self._result.matrix_source:
                status = f"{status} {self._result.matrix_source}"
            self._status_label.setText(status)

        self._topics_table.setRowCount(len(self._result.topics))
        for row, topic in enumerate(self._result.topics):
            self._topics_table.setItem(row, 0, QTableWidgetItem(topic.topic))
            self._topics_table.setItem(row, 1, QTableWidgetItem(topic.top_words))
            self._topics_table.setItem(row, 2, QTableWidgetItem(topic.weights))
        self._topics_table.resizeColumnsToContents()

        headers = ["Document", "Dominant Topic", "Topic Score", *self._result.topic_columns]
        self._documents_table.setColumnCount(len(headers))
        self._documents_table.setHorizontalHeaderLabels(headers)
        self._documents_table.setRowCount(len(self._result.document_topics))
        for row, item in enumerate(self._result.document_topics):
            self._documents_table.setItem(row, 0, QTableWidgetItem(item.document))
            self._documents_table.setItem(row, 1, QTableWidgetItem(item.topic))
            self._documents_table.setItem(row, 2, QTableWidgetItem(f"{item.score:.3f}"))
            for column, score in enumerate(item.scores, start=3):
                self._documents_table.setItem(row, column, QTableWidgetItem(f"{score:.3f}"))
        self._documents_table.resizeColumnsToContents()

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [[topic.topic, topic.top_words, topic.weights] for topic in self._result.topics]
        return {
            "summary": (
                f"Topic Modelling: {len(self._result.topics)} topics, "
                f"{len(self._result.document_topics)} document assignments, "
                f"{METHOD_LABELS.get(self._result.method, self._result.method)}"
            ),
            "headers": ["Topic", "Top Words", "Weights / Notes"],
            "rows": rows,
        }
