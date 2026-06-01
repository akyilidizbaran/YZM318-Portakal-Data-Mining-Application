from __future__ import annotations

import html
import re
import string
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
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

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.corpus_screen import (
    CorpusDocument,
    corpus_documents_from_payload,
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

TOKENIZER_WHITESPACE = "whitespace"
TOKENIZER_WORDPUNCT = "wordpunct"
TOKENIZER_REGEX = "regex"
TOKENIZER_TWEET = "tweet"
TOKENIZERS = {
    TOKENIZER_WHITESPACE,
    TOKENIZER_WORDPUNCT,
    TOKENIZER_REGEX,
    TOKENIZER_TWEET,
}


@dataclass(frozen=True)
class PreprocessOptions:
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_extra_whitespace: bool = True
    remove_stopwords: bool = False
    strip_html: bool = False
    remove_urls: bool = False
    remove_accents: bool = False
    tokenizer: str = TOKENIZER_WORDPUNCT
    regex_pattern: str = r"\b\w+\b"
    custom_stopwords: frozenset[str] = field(default_factory=frozenset)
    min_token_length: int = 1
    max_token_length: int = 0
    keep_alpha_only: bool = False
    remove_tokens_with_numbers: bool = False
    ngram_min: int = 1
    ngram_max: int = 1


@dataclass(frozen=True)
class PreprocessingSummary:
    document_count: int
    total_original_words: int
    total_processed_words: int
    removed_word_count: int
    total_original_tokens: int = 0
    total_processed_tokens: int = 0
    removed_token_count: int = 0
    vocabulary_size: int = 0


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def strip_html(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(without_tags)


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)


def remove_stopwords(text: str, stopwords: Iterable[str] = DEFAULT_STOPWORDS) -> str:
    stopword_set = {word.lower() for word in stopwords}
    return " ".join(word for word in text.split() if word.lower() not in stopword_set)


def parse_custom_stopwords(text: str) -> frozenset[str]:
    return frozenset(token.strip().lower() for token in re.split(r"[,\s]+", text) if token.strip())


_normalize_whitespace = normalize_whitespace
_strip_html = strip_html
_remove_urls = remove_urls
_remove_accents = remove_accents
_remove_punctuation = remove_punctuation
_remove_numbers = remove_numbers
_remove_stopwords = remove_stopwords


def transform_text(
    text: str,
    *,
    lowercase: bool = True,
    strip_html_tags: bool = False,
    remove_url_text: bool = False,
    remove_accent_marks: bool = False,
    remove_punctuation_marks: bool = True,
    remove_number_text: bool = False,
    normalize_spaces: bool = True,
) -> str:
    processed = text
    if strip_html_tags:
        processed = _strip_html(processed)
    if remove_url_text:
        processed = _remove_urls(processed)
    if remove_accent_marks:
        processed = _remove_accents(processed)
    if lowercase:
        processed = processed.lower()
    if remove_punctuation_marks:
        processed = _remove_punctuation(processed)
    if remove_number_text:
        processed = _remove_numbers(processed)
    if normalize_spaces:
        processed = _normalize_whitespace(processed)
    return processed


def tokenize_text(
    text: str,
    *,
    tokenizer: str = TOKENIZER_WORDPUNCT,
    regex_pattern: str = r"\b\w+\b",
) -> tuple[str, ...]:
    if tokenizer not in TOKENIZERS:
        tokenizer = TOKENIZER_WORDPUNCT
    if tokenizer == TOKENIZER_WHITESPACE:
        return tuple(token for token in text.split() if token)
    if tokenizer == TOKENIZER_REGEX:
        try:
            return tuple(match.group(0) for match in re.finditer(regex_pattern or r"\b\w+\b", text))
        except re.error:
            return ()
    if tokenizer == TOKENIZER_TWEET:
        return tuple(
            match.group(0)
            for match in re.finditer(r"[@#]?\w+(?:['-]\w+)*|https?://\S+|[^\w\s]", text, flags=re.UNICODE)
            if match.group(0).strip()
        )
    return tuple(
        match.group(0)
        for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        if match.group(0).strip()
    )


def filter_tokens(
    tokens: Sequence[str],
    *,
    remove_stopword_tokens: bool = False,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
    min_token_length: int = 1,
    max_token_length: int = 0,
    keep_alpha_only: bool = False,
    remove_tokens_with_numbers: bool = False,
) -> tuple[str, ...]:
    stopword_set = {word.lower() for word in stopwords}
    filtered: list[str] = []
    for token in tokens:
        normalized = token.strip()
        if not normalized:
            continue
        if remove_stopword_tokens and normalized.lower() in stopword_set:
            continue
        if min_token_length and len(normalized) < min_token_length:
            continue
        if max_token_length and len(normalized) > max_token_length:
            continue
        if keep_alpha_only and not normalized.isalpha():
            continue
        if remove_tokens_with_numbers and any(character.isdigit() for character in normalized):
            continue
        filtered.append(normalized)
    return tuple(filtered)


def build_ngrams(tokens: Sequence[str], ngram_min: int = 1, ngram_max: int = 1) -> tuple[str, ...]:
    if not tokens:
        return ()
    start = max(1, ngram_min)
    end = max(start, ngram_max)
    ngrams: list[str] = []
    for n in range(start, end + 1):
        if n > len(tokens):
            continue
        for index in range(0, len(tokens) - n + 1):
            ngrams.append("_".join(tokens[index : index + n]))
    return tuple(ngrams)


def preprocess_tokens(
    text: str,
    options: PreprocessOptions = PreprocessOptions(),
) -> tuple[str, ...]:
    transformed = transform_text(
        text,
        lowercase=options.lowercase,
        strip_html_tags=options.strip_html,
        remove_url_text=options.remove_urls,
        remove_accent_marks=options.remove_accents,
        remove_punctuation_marks=options.remove_punctuation,
        remove_number_text=options.remove_numbers,
        normalize_spaces=options.remove_extra_whitespace,
    )
    tokens = tokenize_text(
        transformed,
        tokenizer=options.tokenizer,
        regex_pattern=options.regex_pattern,
    )
    all_stopwords = DEFAULT_STOPWORDS | frozenset(word.lower() for word in options.custom_stopwords)
    filtered = filter_tokens(
        tokens,
        remove_stopword_tokens=options.remove_stopwords,
        stopwords=all_stopwords,
        min_token_length=options.min_token_length,
        max_token_length=options.max_token_length,
        keep_alpha_only=options.keep_alpha_only,
        remove_tokens_with_numbers=options.remove_tokens_with_numbers,
    )
    return build_ngrams(filtered, options.ngram_min, options.ngram_max)


def preprocess_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    normalize_whitespace: bool = True,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
    strip_html_tags: bool = False,
    remove_urls_text: bool = False,
    remove_accents_text: bool = False,
    tokenizer: str = TOKENIZER_WORDPUNCT,
    regex_pattern: str = r"\b\w+\b",
    custom_stopwords: Iterable[str] = (),
    min_token_length: int = 1,
    max_token_length: int = 0,
    keep_alpha_only: bool = False,
    remove_tokens_with_numbers: bool = False,
    ngram_min: int = 1,
    ngram_max: int = 1,
) -> str:
    options = PreprocessOptions(
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        remove_numbers=remove_numbers,
        remove_extra_whitespace=normalize_whitespace,
        remove_stopwords=remove_stopwords,
        strip_html=strip_html_tags,
        remove_urls=remove_urls_text,
        remove_accents=remove_accents_text,
        tokenizer=tokenizer,
        regex_pattern=regex_pattern,
        custom_stopwords=frozenset(word.lower() for word in custom_stopwords),
        min_token_length=min_token_length,
        max_token_length=max_token_length,
        keep_alpha_only=keep_alpha_only,
        remove_tokens_with_numbers=remove_tokens_with_numbers,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    if remove_stopwords and not custom_stopwords:
        options = PreprocessOptions(
            **{
                **options.__dict__,
                "custom_stopwords": frozenset(word.lower() for word in set(stopwords) - set(DEFAULT_STOPWORDS)),
            }
        )
    return " ".join(preprocess_tokens(text, options))


def preprocess_documents(
    documents: Sequence[CorpusDocument],
    options: PreprocessOptions = PreprocessOptions(),
) -> tuple[CorpusDocument, ...]:
    return tuple(
        CorpusDocument(
            document.title,
            " ".join(preprocess_tokens(document.text, options)),
            document.source,
        )
        for document in documents
    )


def summarize_preprocessing(
    original_documents: Sequence[CorpusDocument],
    processed_documents: Sequence[CorpusDocument],
    options: PreprocessOptions = PreprocessOptions(),
) -> PreprocessingSummary:
    document_count = len(original_documents)
    total_original_words = sum(count_words(document.text) for document in original_documents)
    total_processed_words = sum(count_words(document.text) for document in processed_documents)
    original_tokens = sum(len(tokenize_text(document.text, tokenizer=TOKENIZER_WHITESPACE)) for document in original_documents)
    processed_token_sets = [document.text.split() for document in processed_documents]
    processed_tokens = sum(len(tokens) for tokens in processed_token_sets)
    vocabulary = {token for tokens in processed_token_sets for token in tokens}
    removed_words = max(0, total_original_words - total_processed_words)
    removed_tokens = max(0, original_tokens - processed_tokens)
    return PreprocessingSummary(
        document_count,
        total_original_words,
        total_processed_words,
        removed_words,
        original_tokens,
        processed_tokens,
        removed_tokens,
        len(vocabulary),
    )


class PreprocessTextScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._original_documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._processed_documents: tuple[CorpusDocument, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Preprocess Text",
                "Build a lightweight Orange-inspired preprocessing pipeline for text documents.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self.apply_preprocessing()

    def sizeHint(self) -> QSize:
        return QSize(1080, 760)

    def minimumSizeHint(self) -> QSize:
        return QSize(820, 560)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(8)

        self._lowercase_checkbox = QCheckBox("Lowercase", self)
        self._lowercase_checkbox.setChecked(True)
        self._strip_html_checkbox = QCheckBox("Strip HTML", self)
        self._urls_checkbox = QCheckBox("Remove URLs", self)
        self._accents_checkbox = QCheckBox("Remove accents", self)
        self._punctuation_checkbox = QCheckBox("Remove punctuation", self)
        self._punctuation_checkbox.setChecked(True)
        self._numbers_checkbox = QCheckBox("Remove numbers", self)
        self._whitespace_checkbox = QCheckBox("Remove extra whitespace", self)
        self._whitespace_checkbox.setChecked(True)

        layout.addWidget(QLabel("Transformation", self), 0, 0)
        layout.addWidget(self._lowercase_checkbox, 0, 1)
        layout.addWidget(self._strip_html_checkbox, 0, 2)
        layout.addWidget(self._urls_checkbox, 0, 3)
        layout.addWidget(self._accents_checkbox, 0, 4)
        layout.addWidget(self._punctuation_checkbox, 1, 1)
        layout.addWidget(self._numbers_checkbox, 1, 2)
        layout.addWidget(self._whitespace_checkbox, 1, 3)

        layout.addWidget(QLabel("Tokenizer", self), 2, 0)
        self._tokenizer_combo = QComboBox(self)
        self._tokenizer_combo.addItem("Word punctuation", TOKENIZER_WORDPUNCT)
        self._tokenizer_combo.addItem("Whitespace", TOKENIZER_WHITESPACE)
        self._tokenizer_combo.addItem("Regex", TOKENIZER_REGEX)
        self._tokenizer_combo.addItem("Tweet-like", TOKENIZER_TWEET)
        layout.addWidget(self._tokenizer_combo, 2, 1)
        self._regex_input = self._build_line_input(r"\b\w+\b")
        layout.addWidget(self._regex_input, 2, 2, 1, 2)

        layout.addWidget(QLabel("Filtering", self), 3, 0)
        self._stopwords_checkbox = QCheckBox("Remove stopwords", self)
        self._alpha_checkbox = QCheckBox("Keep alphabetic only", self)
        self._token_numbers_checkbox = QCheckBox("Remove tokens with numbers", self)
        layout.addWidget(self._stopwords_checkbox, 3, 1)
        layout.addWidget(self._alpha_checkbox, 3, 2)
        layout.addWidget(self._token_numbers_checkbox, 3, 3)

        layout.addWidget(QLabel("Custom stopwords", self), 4, 0)
        self._custom_stopwords_input = self._build_line_input("comma or space separated")
        layout.addWidget(self._custom_stopwords_input, 4, 1, 1, 2)

        layout.addWidget(QLabel("Min length", self), 4, 3)
        self._min_length_spinbox = QSpinBox(self)
        self._min_length_spinbox.setRange(1, 99)
        self._min_length_spinbox.setValue(1)
        layout.addWidget(self._min_length_spinbox, 4, 4)

        layout.addWidget(QLabel("Max length", self), 5, 3)
        self._max_length_spinbox = QSpinBox(self)
        self._max_length_spinbox.setRange(0, 999)
        self._max_length_spinbox.setValue(0)
        layout.addWidget(self._max_length_spinbox, 5, 4)

        layout.addWidget(QLabel("N-grams", self), 5, 0)
        self._ngram_min_spinbox = QSpinBox(self)
        self._ngram_min_spinbox.setRange(1, 5)
        self._ngram_min_spinbox.setValue(1)
        layout.addWidget(self._ngram_min_spinbox, 5, 1)
        self._ngram_max_spinbox = QSpinBox(self)
        self._ngram_max_spinbox.setRange(1, 5)
        self._ngram_max_spinbox.setValue(1)
        layout.addWidget(self._ngram_max_spinbox, 5, 2)

        self._apply_button = QPushButton("Apply Preprocessing", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self.apply_preprocessing)
        layout.addWidget(self._apply_button, 6, 3, 1, 2)
        return frame

    def _build_line_input(self, placeholder: str) -> QLineEdit:
        line_edit = QLineEdit(self)
        line_edit.setPlaceholderText(placeholder)
        line_edit.setStyleSheet(
            "background: #fffdf9; color: #2b2b2b; border: 1px solid #d1cabf; "
            "border-radius: 8px; padding: 6px 10px;"
        )
        return line_edit

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
        self._processed_tokens_label = self._build_metric_label("Processed Tokens", "0")
        self._vocabulary_label = self._build_metric_label("Vocabulary", "0")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._original_words_label, 1)
        layout.addWidget(self._processed_words_label, 1)
        layout.addWidget(self._removed_words_label, 1)
        layout.addWidget(self._processed_tokens_label, 1)
        layout.addWidget(self._vocabulary_label, 1)
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

        self._table = QTableWidget(0, 7, self)
        self._table.setHorizontalHeaderLabels(
            [
                "Title",
                "Original Text",
                "Preprocessed Text",
                "Tokens",
                "Original Words",
                "Processed Tokens",
                "Removed Tokens",
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
        ngram_min = self._ngram_min_spinbox.value()
        ngram_max = max(ngram_min, self._ngram_max_spinbox.value())
        if self._ngram_max_spinbox.value() != ngram_max:
            self._ngram_max_spinbox.setValue(ngram_max)
        return PreprocessOptions(
            lowercase=self._lowercase_checkbox.isChecked(),
            remove_punctuation=self._punctuation_checkbox.isChecked(),
            remove_numbers=self._numbers_checkbox.isChecked(),
            remove_extra_whitespace=self._whitespace_checkbox.isChecked(),
            remove_stopwords=self._stopwords_checkbox.isChecked(),
            strip_html=self._strip_html_checkbox.isChecked(),
            remove_urls=self._urls_checkbox.isChecked(),
            remove_accents=self._accents_checkbox.isChecked(),
            tokenizer=self._tokenizer_combo.currentData() or TOKENIZER_WORDPUNCT,
            regex_pattern=self._regex_input.text() or r"\b\w+\b",
            custom_stopwords=parse_custom_stopwords(self._custom_stopwords_input.text()),
            min_token_length=self._min_length_spinbox.value(),
            max_token_length=self._max_length_spinbox.value(),
            keep_alpha_only=self._alpha_checkbox.isChecked(),
            remove_tokens_with_numbers=self._token_numbers_checkbox.isChecked(),
            ngram_min=ngram_min,
            ngram_max=ngram_max,
        )

    def apply_preprocessing(self) -> tuple[CorpusDocument, ...]:
        self._processed_documents = preprocess_documents(self._original_documents, self._current_options())
        self._render()
        self._notify_output_changed()
        return self._processed_documents

    def set_documents(self, documents: Sequence[CorpusDocument]) -> None:
        self._original_documents = tuple(documents)
        self._using_input_corpus = True
        self.apply_preprocessing()

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._original_documents = ()
            self._using_input_corpus = False
            self.apply_preprocessing()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._original_documents = () if documents is None else documents
        self._using_input_corpus = True
        self.apply_preprocessing()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Corpus", self._processed_documents)

    def _render(self) -> None:
        options = self._current_options()
        summary = summarize_preprocessing(self._original_documents, self._processed_documents, options)
        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._original_words_label.setText(f"Original Words\n{summary.total_original_words}")
        self._processed_words_label.setText(f"Processed Words\n{summary.total_processed_words}")
        self._removed_words_label.setText(f"Removed Words\n{summary.removed_word_count}")
        self._processed_tokens_label.setText(f"Processed Tokens\n{summary.total_processed_tokens}")
        self._vocabulary_label.setText(f"Vocabulary\n{summary.vocabulary_size}")

        if self._original_documents and self._using_input_corpus:
            status = "Input corpus is connected and preprocessed."
        elif self._using_input_corpus:
            status = "Input corpus is connected but empty."
        else:
            status = "Connect a Corpus input to preprocess documents."
        self._status_label.setText(status + " " + self._pipeline_summary(options))

        self._table.setRowCount(len(self._original_documents))
        document_pairs = zip(self._original_documents, self._processed_documents)
        for row, (original, processed) in enumerate(document_pairs):
            original_tokens = tokenize_text(original.text, tokenizer=TOKENIZER_WHITESPACE)
            processed_tokens = processed.text.split()
            self._set_item(row, 0, original.title)
            self._set_item(row, 1, preview_text(original.text))
            self._set_item(row, 2, preview_text(processed.text))
            self._set_item(row, 3, preview_text(", ".join(processed_tokens), max_length=180))
            self._set_item(row, 4, str(count_words(original.text)))
            self._set_item(row, 5, str(len(processed_tokens)))
            self._set_item(row, 6, str(max(0, len(original_tokens) - len(processed_tokens))))
        self._table.resizeColumnsToContents()

    def _pipeline_summary(self, options: PreprocessOptions) -> str:
        steps: list[str] = []
        if options.lowercase:
            steps.append("lowercase")
        if options.strip_html:
            steps.append("strip HTML")
        if options.remove_urls:
            steps.append("remove URLs")
        if options.remove_accents:
            steps.append("remove accents")
        if options.remove_punctuation:
            steps.append("remove punctuation")
        if options.remove_numbers:
            steps.append("remove numbers")
        steps.append(f"tokenizer={options.tokenizer}")
        if options.remove_stopwords:
            steps.append("stopwords")
        if options.ngram_min != 1 or options.ngram_max != 1:
            steps.append(f"n-grams={options.ngram_min}-{options.ngram_max}")
        return "Pipeline: " + ", ".join(steps) + "."

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [
                original.title,
                preview_text(original.text),
                preview_text(processed.text),
                preview_text(", ".join(processed.text.split()), max_length=180),
                str(count_words(original.text)),
                str(len(processed.text.split())),
            ]
            for original, processed in zip(self._original_documents, self._processed_documents)
        ]
        summary = summarize_preprocessing(self._original_documents, self._processed_documents, self._current_options())
        return {
            "summary": (
                f"Preprocess Text: {summary.document_count} documents, "
                f"{summary.total_original_words} original words, "
                f"{summary.total_processed_tokens} processed tokens, "
                f"{summary.vocabulary_size} vocabulary"
            ),
            "headers": [
                "Title",
                "Original Text",
                "Preprocessed Text",
                "Tokens",
                "Original Words",
                "Processed Tokens",
            ],
            "rows": rows,
        }
