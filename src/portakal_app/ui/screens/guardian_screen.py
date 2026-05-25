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
from portakal_app.ui.shared.readable_inputs import (
    apply_readable_line_edit_style,
    apply_readable_spin_box_style,
)


DEFAULT_GUARDIAN_LIMIT = 5
GUARDIAN_TIMEOUT_SECONDS = 8
GUARDIAN_SOURCE = "The Guardian"
GUARDIAN_API_KEY_ENV = "GUARDIAN_API_KEY"


@dataclass(frozen=True)
class GuardianDocument:
    title: str
    text: str
    source: str = GUARDIAN_SOURCE
    section: str = ""
    publication_date: str = ""


@dataclass(frozen=True)
class GuardianFetchResult:
    documents: tuple[GuardianDocument, ...]
    used_fallback: bool = False
    message: str = ""


def normalize_guardian_limit(limit: int) -> int:
    return max(1, min(20, int(limit)))


def normalize_guardian_section(section: str | None) -> str:
    if section is None:
        return ""
    return re.sub(r"[^a-z0-9-]", "", section.strip().lower())


def get_guardian_api_key() -> str:
    return os.environ.get(GUARDIAN_API_KEY_ENV, "").strip()


def build_guardian_search_url(
    query: str,
    api_key: str,
    limit: int = DEFAULT_GUARDIAN_LIMIT,
    section: str | None = None,
) -> str:
    params = {
        "q": query.strip(),
        "page-size": normalize_guardian_limit(limit),
        "show-fields": "trailText",
        "order-by": "relevance",
        "api-key": api_key.strip(),
    }
    clean_section = normalize_guardian_section(section)
    if clean_section:
        params["section"] = clean_section
    return f"https://content.guardianapis.com/search?{urlencode(params)}"


def parse_guardian_response(payload: object) -> tuple[GuardianDocument, ...]:
    if not isinstance(payload, dict):
        return ()
    response = payload.get("response")
    if not isinstance(response, dict):
        return ()
    results = response.get("results")
    if not isinstance(results, list):
        return ()

    documents: list[GuardianDocument] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = item.get("webTitle")
        if not isinstance(title, str) or not title.strip():
            continue
        fields = item.get("fields")
        trail_text = fields.get("trailText") if isinstance(fields, dict) else ""
        text = _clean_guardian_text(trail_text if isinstance(trail_text, str) else "")
        if not text:
            text = _clean_guardian_text(title)
        documents.append(
            GuardianDocument(
                title=title.strip(),
                text=text,
                source=GUARDIAN_SOURCE,
                section=_safe_string(item.get("sectionName")),
                publication_date=_safe_string(item.get("webPublicationDate")),
            )
        )
    return tuple(documents)


def fallback_guardian_documents(query: str = "") -> tuple[GuardianDocument, ...]:
    query_label = query.strip() or "text mining news"
    return (
        GuardianDocument(
            f"Guardian Sample: {query_label}",
            (
                f"{query_label} is represented here as a deterministic news-style "
                "article summary for offline corpus testing."
            ),
            GUARDIAN_SOURCE,
            "Sample",
            "2026-01-01T00:00:00Z",
        ),
        GuardianDocument(
            "Guardian Sample: Data journalism",
            (
                "News article metadata can provide headlines, sections, dates, and short "
                "trail text for lightweight corpus exploration."
            ),
            GUARDIAN_SOURCE,
            "Technology",
            "2026-01-02T00:00:00Z",
        ),
        GuardianDocument(
            "Guardian Sample: Public interest reporting",
            (
                "A source widget can turn search results into documents without scraping "
                "full article bodies."
            ),
            GUARDIAN_SOURCE,
            "World news",
            "2026-01-03T00:00:00Z",
        ),
    )


def summarize_documents(documents: Sequence[GuardianDocument]) -> CorpusSummary:
    document_count = len(documents)
    total_word_count = sum(count_words(document.text) for document in documents)
    average = total_word_count / document_count if document_count else 0.0
    return CorpusSummary(document_count, total_word_count, average)


def fetch_guardian_documents(
    query: str,
    limit: int = DEFAULT_GUARDIAN_LIMIT,
    section: str | None = None,
    api_key: str | None = None,
    fetch_json: Callable[[str], object] | None = None,
) -> GuardianFetchResult:
    clean_query = query.strip()
    normalized_limit = normalize_guardian_limit(limit)
    clean_section = normalize_guardian_section(section)
    resolved_api_key = get_guardian_api_key() if api_key is None else api_key.strip()

    if not clean_query:
        return GuardianFetchResult(
            fallback_guardian_documents(clean_query),
            True,
            "Enter a query to fetch Guardian articles. Showing built-in sample documents.",
        )
    if not resolved_api_key:
        return GuardianFetchResult(
            fallback_guardian_documents(clean_query),
            True,
            "Set GUARDIAN_API_KEY to fetch live Guardian results. Showing built-in sample documents.",
        )

    loader = fetch_json or _fetch_json_url
    try:
        payload = loader(
            build_guardian_search_url(
                clean_query,
                resolved_api_key,
                normalized_limit,
                clean_section,
            )
        )
        documents = parse_guardian_response(payload)
    except Exception as error:
        return GuardianFetchResult(
            fallback_guardian_documents(clean_query),
            True,
            f"Guardian fetch failed: {error}. Showing built-in sample documents.",
        )

    if not documents:
        return GuardianFetchResult(
            fallback_guardian_documents(clean_query),
            True,
            "No Guardian results found. Showing built-in sample documents.",
        )
    return GuardianFetchResult(
        documents[:normalized_limit],
        False,
        f"Fetched {min(len(documents), normalized_limit)} Guardian articles for '{clean_query}'.",
    )


def _fetch_json_url(url: str) -> object:
    request = Request(
        url,
        headers={"User-Agent": "PortakalGuardianWidget/0.1 (educational desktop app)"},
    )
    with urlopen(request, timeout=GUARDIAN_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _clean_guardian_text(text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(without_tags).split())


def _safe_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


class GuardianScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[GuardianDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(
            fallback_guardian_documents() if documents is None else documents
        )
        self._status_message = (
            "Showing built-in Guardian sample. Enter a query and set GUARDIAN_API_KEY "
            "to fetch live articles."
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "The Guardian",
                "Retrieve news article summaries as lightweight corpus documents.",
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
        self._query_input.setPlaceholderText("Search Guardian articles...")
        apply_readable_line_edit_style(self._query_input)
        layout.addWidget(self._query_input, 1)

        self._section_input = QLineEdit(self)
        self._section_input.setPlaceholderText("Optional section")
        self._section_input.setMaximumWidth(150)
        apply_readable_line_edit_style(self._section_input)
        layout.addWidget(QLabel("Section", self))
        layout.addWidget(self._section_input)

        layout.addWidget(QLabel("Limit", self))
        self._limit_spinbox = QSpinBox(self)
        self._limit_spinbox.setRange(1, 20)
        self._limit_spinbox.setValue(DEFAULT_GUARDIAN_LIMIT)
        apply_readable_spin_box_style(self._limit_spinbox)
        layout.addWidget(self._limit_spinbox)

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

        self._table = QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(
            ["Title", "Source", "Section", "Date", "Text Preview", "Words"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def fetch_articles(self) -> GuardianFetchResult:
        result = fetch_guardian_documents(
            self._query_input.text(),
            self._limit_spinbox.value(),
            self._section_input.text(),
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
            self._set_item(row, 2, document.section)
            self._set_item(row, 3, document.publication_date)
            self._set_item(row, 4, preview_text(document.text))
            self._set_item(row, 5, str(count_words(document.text)))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [
                document.title,
                document.source,
                document.section,
                document.publication_date,
                preview_text(document.text),
                str(count_words(document.text)),
            ]
            for document in self._documents
        ]
        summary = summarize_documents(self._documents)
        return {
            "summary": (
                f"The Guardian: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.average_words_per_document:.1f} average words/document"
            ),
            "headers": ["Title", "Source", "Section", "Date", "Text Preview", "Words"],
            "rows": rows,
        }
