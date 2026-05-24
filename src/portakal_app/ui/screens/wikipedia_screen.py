from __future__ import annotations

import html
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.ui.screens.corpus_screen import (
    CorpusDocument,
    CorpusSummary,
    count_words,
    summarize_corpus,
)
from portakal_app.ui.screens.create_corpus_screen import preview_text
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


DEFAULT_WIKIPEDIA_LANGUAGE = "en"
DEFAULT_WIKIPEDIA_LIMIT = 5
WIKIPEDIA_TIMEOUT_SECONDS = 8


@dataclass(frozen=True)
class WikipediaSearchResult:
    title: str
    snippet: str = ""


@dataclass(frozen=True)
class WikipediaFetchResult:
    documents: tuple[CorpusDocument, ...]
    used_fallback: bool = False
    message: str = ""


def normalize_language_code(language: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]", "", language.strip().lower())
    return cleaned or DEFAULT_WIKIPEDIA_LANGUAGE


def build_wikipedia_search_url(
    query: str,
    language: str = DEFAULT_WIKIPEDIA_LANGUAGE,
    limit: int = DEFAULT_WIKIPEDIA_LIMIT,
) -> str:
    lang = normalize_language_code(language)
    params = urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query.strip(),
            "srlimit": max(1, limit),
            "format": "json",
            "utf8": 1,
        }
    )
    return f"https://{lang}.wikipedia.org/w/api.php?{params}"


def build_wikipedia_summary_url(title: str, language: str = DEFAULT_WIKIPEDIA_LANGUAGE) -> str:
    lang = normalize_language_code(language)
    safe_title = quote(title.strip().replace(" ", "_"), safe="")
    return f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{safe_title}"


def parse_wikipedia_search_response(payload: object) -> tuple[WikipediaSearchResult, ...]:
    if not isinstance(payload, dict):
        return ()
    query = payload.get("query")
    if not isinstance(query, dict):
        return ()
    search_items = query.get("search")
    if not isinstance(search_items, list):
        return ()

    results: list[WikipediaSearchResult] = []
    for item in search_items:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        snippet = item.get("snippet")
        results.append(
            WikipediaSearchResult(title.strip(), _clean_wikipedia_text(snippet or ""))
        )
    return tuple(results)


def parse_wikipedia_summary_response(
    payload: object,
    language: str = DEFAULT_WIKIPEDIA_LANGUAGE,
) -> CorpusDocument | None:
    if not isinstance(payload, dict):
        return None
    title = payload.get("title")
    extract = payload.get("extract")
    description = payload.get("description")
    if not isinstance(title, str) or not title.strip():
        return None
    text_parts = [
        value.strip()
        for value in (extract, description)
        if isinstance(value, str) and value.strip()
    ]
    if not text_parts:
        return None
    text = _clean_wikipedia_text(" ".join(text_parts))
    if not text:
        return None
    return CorpusDocument(title.strip(), text, f"Wikipedia ({normalize_language_code(language)})")


def fallback_wikipedia_documents(
    query: str = "",
    language: str = DEFAULT_WIKIPEDIA_LANGUAGE,
) -> tuple[CorpusDocument, ...]:
    lang = normalize_language_code(language)
    query_label = query.strip() or "Text mining"
    return (
        CorpusDocument(
            f"Wikipedia Sample: {query_label}",
            (
                f"{query_label} is represented here as a deterministic encyclopedia-style "
                "article summary for offline corpus testing."
            ),
            f"Wikipedia sample ({lang})",
        ),
        CorpusDocument(
            "Wikipedia Sample: Corpus",
            "A corpus is a collection of documents that can be searched, summarized, and prepared for text analysis.",
            f"Wikipedia sample ({lang})",
        ),
        CorpusDocument(
            "Wikipedia Sample: Open Knowledge",
            "Wikipedia-like sources provide concise article summaries that are useful for exploratory text workflows.",
            f"Wikipedia sample ({lang})",
        ),
    )


def summarize_documents(documents: Sequence[CorpusDocument]) -> CorpusSummary:
    return summarize_corpus(documents)


def fetch_wikipedia_documents(
    query: str,
    language: str = DEFAULT_WIKIPEDIA_LANGUAGE,
    limit: int = DEFAULT_WIKIPEDIA_LIMIT,
    fetch_json: Callable[[str], object] | None = None,
) -> WikipediaFetchResult:
    clean_query = query.strip()
    lang = normalize_language_code(language)
    if not clean_query:
        return WikipediaFetchResult(
            fallback_wikipedia_documents(clean_query, lang),
            True,
            "Enter a query to fetch live Wikipedia summaries. Showing built-in sample documents.",
        )

    loader = fetch_json or _fetch_json_url
    try:
        search_payload = loader(build_wikipedia_search_url(clean_query, lang, limit))
        search_results = parse_wikipedia_search_response(search_payload)
        documents = _documents_from_search_results(search_results, lang, limit, loader)
    except Exception as error:
        return WikipediaFetchResult(
            fallback_wikipedia_documents(clean_query, lang),
            True,
            f"Wikipedia fetch failed: {error}. Showing built-in sample documents.",
        )

    if not documents:
        return WikipediaFetchResult(
            fallback_wikipedia_documents(clean_query, lang),
            True,
            "No Wikipedia results found. Showing built-in sample documents.",
        )
    return WikipediaFetchResult(
        documents,
        False,
        f"Fetched {len(documents)} Wikipedia summaries for '{clean_query}'.",
    )


def _documents_from_search_results(
    search_results: Sequence[WikipediaSearchResult],
    language: str,
    limit: int,
    fetch_json: Callable[[str], object],
) -> tuple[CorpusDocument, ...]:
    documents: list[CorpusDocument] = []
    for result in search_results[: max(1, limit)]:
        try:
            summary = parse_wikipedia_summary_response(
                fetch_json(build_wikipedia_summary_url(result.title, language)),
                language,
            )
        except Exception:
            summary = None
        if summary is not None:
            documents.append(summary)
        elif result.snippet:
            documents.append(
                CorpusDocument(result.title, result.snippet, f"Wikipedia ({language})")
            )
    return tuple(documents)


def _fetch_json_url(url: str) -> object:
    request = Request(
        url,
        headers={"User-Agent": "PortakalTextMiningWidget/0.1 (educational desktop app)"},
    )
    with urlopen(request, timeout=WIKIPEDIA_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _clean_wikipedia_text(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(without_tags).split())


class WikipediaScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(
            fallback_wikipedia_documents() if documents is None else documents
        )
        self._status_message = "Showing built-in Wikipedia sample. Enter a query to fetch live summaries."

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Wikipedia",
                "Retrieve Wikipedia-style article summaries as lightweight corpus documents.",
            )
        )
        layout.addWidget(self._build_query_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._render()

    def sizeHint(self) -> QSize:
        return QSize(960, 660)

    def minimumSizeHint(self) -> QSize:
        return QSize(720, 500)

    def _build_query_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._query_input = QLineEdit(self)
        self._query_input.setPlaceholderText("Search Wikipedia...")
        layout.addWidget(self._query_input, 1)

        self._language_input = QLineEdit(DEFAULT_WIKIPEDIA_LANGUAGE, self)
        self._language_input.setPlaceholderText(DEFAULT_WIKIPEDIA_LANGUAGE)
        self._language_input.setMaximumWidth(70)
        layout.addWidget(QLabel("Language", self))
        layout.addWidget(self._language_input)

        self._fetch_button = QPushButton("Fetch", self)
        self._fetch_button.setProperty("primary", True)
        self._fetch_button.clicked.connect(self.fetch_articles)
        layout.addWidget(self._fetch_button)
        return frame

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._total_word_count_label = self._build_metric_label("Total Words", "0")
        self._average_word_count_label = self._build_metric_label(
            "Avg Words / Document",
            "0.0",
        )

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._total_word_count_label, 1)
        layout.addWidget(self._average_word_count_label, 1)
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
        self._table.setHorizontalHeaderLabels(["Title", "Source", "Text Preview", "Words"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def fetch_articles(self) -> WikipediaFetchResult:
        result = fetch_wikipedia_documents(
            self._query_input.text(),
            self._language_input.text(),
            limit=DEFAULT_WIKIPEDIA_LIMIT,
        )
        self._documents = result.documents
        self._status_message = result.message
        self._render()
        self._notify_output_changed()
        return result

    def _render(self) -> None:
        summary = summarize_documents(self._documents)
        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._total_word_count_label.setText(f"Total Words\n{summary.total_word_count}")
        self._average_word_count_label.setText(
            f"Avg Words / Document\n{summary.average_words_per_document:.1f}"
        )
        self._status_label.setText(self._status_message)

        self._table.setRowCount(len(self._documents))
        for row, document in enumerate(self._documents):
            self._set_item(row, 0, document.title)
            self._set_item(row, 1, document.source)
            self._set_item(row, 2, preview_text(document.text))
            self._set_item(row, 3, str(count_words(document.text)))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [
                document.title,
                document.source,
                preview_text(document.text),
                str(count_words(document.text)),
            ]
            for document in self._documents
        ]
        summary = summarize_documents(self._documents)
        return {
            "summary": (
                f"Wikipedia: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.average_words_per_document:.1f} average words/document"
            ),
            "headers": ["Title", "Source", "Text Preview", "Words"],
            "rows": rows,
        }
