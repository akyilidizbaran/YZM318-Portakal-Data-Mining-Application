from __future__ import annotations

import html
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
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

from portakal_app.ui.screens.corpus_screen import CorpusSummary, count_words
from portakal_app.ui.screens.create_corpus_screen import preview_text
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


DEFAULT_TWITTER_LIMIT = 5
TWITTER_TIMEOUT_SECONDS = 8
TWITTER_SOURCE = "Twitter/X"
TWITTER_BEARER_TOKEN_ENV = "TWITTER_BEARER_TOKEN"


@dataclass(frozen=True)
class TwitterDocument:
    post_id: str
    text: str
    source: str = TWITTER_SOURCE
    author: str = ""
    publication_date: str = ""


@dataclass(frozen=True)
class TwitterFetchResult:
    documents: tuple[TwitterDocument, ...]
    used_fallback: bool = False
    message: str = ""


def normalize_twitter_limit(limit: int) -> int:
    return max(1, min(20, int(limit)))


def normalize_twitter_api_limit(limit: int) -> int:
    return max(10, min(100, int(limit)))


def get_twitter_bearer_token() -> str:
    return os.environ.get(TWITTER_BEARER_TOKEN_ENV, "").strip()


def build_twitter_search_url(query: str, limit: int = DEFAULT_TWITTER_LIMIT) -> str:
    params = {
        "query": query.strip(),
        "max_results": normalize_twitter_api_limit(limit),
        "tweet.fields": "created_at,author_id",
    }
    return f"https://api.twitter.com/2/tweets/search/recent?{urlencode(params)}"


def parse_twitter_response(payload: object) -> tuple[TwitterDocument, ...]:
    if not isinstance(payload, dict):
        return ()
    data = payload.get("data")
    if not isinstance(data, list):
        return ()

    authors = _twitter_author_lookup(payload)
    documents: list[TwitterDocument] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        text = _clean_twitter_text(_safe_string(item.get("text")))
        if not text:
            continue
        post_id = _safe_string(item.get("id")) or f"post-{index}"
        author_id = _safe_string(item.get("author_id"))
        documents.append(
            TwitterDocument(
                post_id=post_id,
                text=text,
                source=TWITTER_SOURCE,
                author=authors.get(author_id, author_id or "Unknown"),
                publication_date=_safe_string(item.get("created_at")),
            )
        )
    return tuple(documents)


def fallback_twitter_documents(query: str = "") -> tuple[TwitterDocument, ...]:
    query_label = query.strip() or "text mining"
    return (
        TwitterDocument(
            "sample-1",
            (
                f"{query_label} is represented here as a deterministic short social "
                "post for offline corpus testing."
            ),
            TWITTER_SOURCE,
            "sample_user",
            "2026-01-01T09:00:00Z",
        ),
        TwitterDocument(
            "sample-2",
            (
                "Short posts can capture reactions, hashtags, and concise public text "
                "for lightweight corpus exploration."
            ),
            TWITTER_SOURCE,
            "data_notes",
            "2026-01-02T10:30:00Z",
        ),
        TwitterDocument(
            "sample-3",
            (
                "A social-feed source should remain useful without tokens, scraping, "
                "or browser automation."
            ),
            TWITTER_SOURCE,
            "portakal_demo",
            "2026-01-03T12:15:00Z",
        ),
    )


def summarize_documents(documents: Sequence[TwitterDocument]) -> CorpusSummary:
    document_count = len(documents)
    total_word_count = sum(count_words(document.text) for document in documents)
    average = total_word_count / document_count if document_count else 0.0
    return CorpusSummary(document_count, total_word_count, average)


def fetch_twitter_documents(
    query: str,
    limit: int = DEFAULT_TWITTER_LIMIT,
    bearer_token: str | None = None,
    fetch_json: Callable[[str, str], object] | None = None,
) -> TwitterFetchResult:
    clean_query = query.strip()
    normalized_limit = normalize_twitter_limit(limit)
    resolved_token = (
        get_twitter_bearer_token() if bearer_token is None else bearer_token.strip()
    )

    if not clean_query:
        return TwitterFetchResult(
            fallback_twitter_documents(clean_query),
            True,
            "Enter a query to fetch social posts. Showing built-in sample documents.",
        )
    if not resolved_token:
        return TwitterFetchResult(
            fallback_twitter_documents(clean_query),
            True,
            "Set TWITTER_BEARER_TOKEN to fetch live Twitter/X results. Showing built-in sample documents.",
        )

    loader = fetch_json or _fetch_json_url
    try:
        payload = loader(build_twitter_search_url(clean_query, normalized_limit), resolved_token)
        documents = parse_twitter_response(payload)
    except Exception as error:
        return TwitterFetchResult(
            fallback_twitter_documents(clean_query),
            True,
            f"Twitter/X fetch failed: {error}. Showing built-in sample documents.",
        )

    if not documents:
        return TwitterFetchResult(
            fallback_twitter_documents(clean_query),
            True,
            "No Twitter/X results found. Showing built-in sample documents.",
        )
    return TwitterFetchResult(
        documents[:normalized_limit],
        False,
        f"Fetched {min(len(documents), normalized_limit)} Twitter/X posts for '{clean_query}'.",
    )


def _fetch_json_url(url: str, bearer_token: str) -> object:
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {bearer_token}",
            "User-Agent": "PortakalTwitterWidget/0.1 (educational desktop app)",
        },
    )
    with urlopen(request, timeout=TWITTER_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _twitter_author_lookup(payload: dict[str, object]) -> dict[str, str]:
    includes = payload.get("includes")
    if not isinstance(includes, dict):
        return {}
    users = includes.get("users")
    if not isinstance(users, list):
        return {}

    authors: dict[str, str] = {}
    for user in users:
        if not isinstance(user, dict):
            continue
        user_id = _safe_string(user.get("id"))
        username = _safe_string(user.get("username")) or _safe_string(user.get("name"))
        if user_id and username:
            authors[user_id] = username
    return authors


def _clean_twitter_text(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(without_tags).split())


def _safe_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


class TwitterScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[TwitterDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(
            fallback_twitter_documents() if documents is None else documents
        )
        self._status_message = (
            "Showing built-in social-post sample. Enter a query and set "
            "TWITTER_BEARER_TOKEN to fetch live posts."
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Twitter",
                "Retrieve or demonstrate short social-media posts as corpus documents.",
            )
        )
        layout.addWidget(self._build_query_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._render()

    def sizeHint(self) -> QSize:
        return QSize(1020, 660)

    def minimumSizeHint(self) -> QSize:
        return QSize(760, 500)

    def _build_query_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._query_input = QLineEdit(self)
        self._query_input.setPlaceholderText("Search social posts...")
        layout.addWidget(self._query_input, 1)

        layout.addWidget(QLabel("Limit", self))
        self._limit_spinbox = QSpinBox(self)
        self._limit_spinbox.setRange(1, 20)
        self._limit_spinbox.setValue(DEFAULT_TWITTER_LIMIT)
        layout.addWidget(self._limit_spinbox)

        self._fetch_button = QPushButton("Fetch", self)
        self._fetch_button.setProperty("primary", True)
        self._fetch_button.clicked.connect(self.fetch_posts)
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

        self._table = QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(
            ["Post ID", "Author", "Source", "Date", "Text Preview", "Words"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def fetch_posts(self) -> TwitterFetchResult:
        result = fetch_twitter_documents(
            self._query_input.text(),
            self._limit_spinbox.value(),
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
            self._set_item(row, 0, document.post_id)
            self._set_item(row, 1, document.author)
            self._set_item(row, 2, document.source)
            self._set_item(row, 3, document.publication_date)
            self._set_item(row, 4, preview_text(document.text))
            self._set_item(row, 5, str(count_words(document.text)))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [
                document.post_id,
                document.author,
                document.source,
                document.publication_date,
                preview_text(document.text),
                str(count_words(document.text)),
            ]
            for document in self._documents
        ]
        summary = summarize_documents(self._documents)
        return {
            "summary": (
                f"Twitter: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.average_words_per_document:.1f} average words/document"
            ),
            "headers": ["Post ID", "Author", "Source", "Date", "Text Preview", "Words"],
            "rows": rows,
        }
