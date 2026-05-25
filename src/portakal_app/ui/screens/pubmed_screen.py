from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
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


DEFAULT_PUBMED_LIMIT = 5
PUBMED_TIMEOUT_SECONDS = 8
PUBMED_SOURCE = "PubMed"


@dataclass(frozen=True)
class PubMedDocument:
    title: str
    text: str
    source: str = PUBMED_SOURCE
    pmid: str = ""


@dataclass(frozen=True)
class PubMedFetchResult:
    documents: tuple[PubMedDocument, ...]
    used_fallback: bool = False
    message: str = ""


def normalize_pubmed_limit(limit: int) -> int:
    return max(1, min(20, int(limit)))


def build_pubmed_search_url(query: str, limit: int = DEFAULT_PUBMED_LIMIT) -> str:
    params = urlencode(
        {
            "db": "pubmed",
            "term": query.strip(),
            "retmode": "json",
            "retmax": normalize_pubmed_limit(limit),
            "tool": "portakal_text_mining_widget",
        }
    )
    return f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"


def build_pubmed_fetch_url(pmids: Sequence[str]) -> str:
    clean_pmids = [pmid.strip() for pmid in pmids if pmid.strip()]
    params = urlencode(
        {
            "db": "pubmed",
            "id": ",".join(clean_pmids),
            "retmode": "xml",
            "tool": "portakal_text_mining_widget",
        }
    )
    return f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"


def parse_pubmed_search_response(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    result = payload.get("esearchresult")
    if not isinstance(result, dict):
        return ()
    id_list = result.get("idlist")
    if not isinstance(id_list, list):
        return ()
    return tuple(str(pmid).strip() for pmid in id_list if str(pmid).strip())


def parse_pubmed_fetch_response(payload: object) -> tuple[PubMedDocument, ...]:
    if isinstance(payload, bytes):
        xml_text = payload.decode("utf-8", errors="replace")
    elif isinstance(payload, str):
        xml_text = payload
    else:
        return ()

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ()

    documents: list[PubMedDocument] = []
    for article in root.findall(".//PubmedArticle"):
        document = _document_from_pubmed_article(article)
        if document is not None:
            documents.append(document)
    return tuple(documents)


def fallback_pubmed_documents(query: str = "") -> tuple[PubMedDocument, ...]:
    query_label = query.strip() or "biomedical text mining"
    return (
        PubMedDocument(
            f"PubMed Sample: {query_label}",
            (
                f"{query_label} is represented here as a deterministic biomedical "
                "abstract for offline corpus testing."
            ),
            PUBMED_SOURCE,
            "SAMPLE-1",
        ),
        PubMedDocument(
            "PubMed Sample: Clinical notes",
            (
                "Biomedical abstracts often describe cohorts, interventions, outcomes, "
                "and conclusions in concise scientific language."
            ),
            PUBMED_SOURCE,
            "SAMPLE-2",
        ),
        PubMedDocument(
            "PubMed Sample: Literature search",
            (
                "PubMed-style records can provide titles and abstracts that are useful "
                "for exploratory text mining workflows."
            ),
            PUBMED_SOURCE,
            "SAMPLE-3",
        ),
    )


def summarize_documents(documents: Sequence[PubMedDocument]) -> CorpusSummary:
    document_count = len(documents)
    total_word_count = sum(count_words(document.text) for document in documents)
    average = total_word_count / document_count if document_count else 0.0
    return CorpusSummary(document_count, total_word_count, average)


def fetch_pubmed_documents(
    query: str,
    limit: int = DEFAULT_PUBMED_LIMIT,
    fetch_json: Callable[[str], object] | None = None,
    fetch_text: Callable[[str], object] | None = None,
) -> PubMedFetchResult:
    clean_query = query.strip()
    normalized_limit = normalize_pubmed_limit(limit)
    if not clean_query:
        return PubMedFetchResult(
            fallback_pubmed_documents(clean_query),
            True,
            "Enter a query to fetch live PubMed abstracts. Showing built-in sample documents.",
        )

    json_loader = fetch_json or _fetch_json_url
    text_loader = fetch_text or _fetch_text_url
    try:
        search_payload = json_loader(build_pubmed_search_url(clean_query, normalized_limit))
        pmids = parse_pubmed_search_response(search_payload)
        if not pmids:
            return PubMedFetchResult(
                fallback_pubmed_documents(clean_query),
                True,
                "No PubMed results found. Showing built-in sample documents.",
            )
        documents = parse_pubmed_fetch_response(
            text_loader(build_pubmed_fetch_url(pmids[:normalized_limit]))
        )
    except Exception as error:
        return PubMedFetchResult(
            fallback_pubmed_documents(clean_query),
            True,
            f"PubMed fetch failed: {error}. Showing built-in sample documents.",
        )

    if not documents:
        return PubMedFetchResult(
            fallback_pubmed_documents(clean_query),
            True,
            "PubMed records had no parseable abstracts. Showing built-in sample documents.",
        )
    return PubMedFetchResult(
        documents,
        False,
        f"Fetched {len(documents)} PubMed records for '{clean_query}'.",
    )


def _document_from_pubmed_article(article: ET.Element) -> PubMedDocument | None:
    pmid = _text_from_first(article, (".//MedlineCitation/PMID", ".//PMID"))
    title = _text_from_first(article, (".//Article/ArticleTitle", ".//ArticleTitle"))
    abstract = _abstract_text(article)
    if not title:
        return None
    text = abstract or "No abstract available for this PubMed record."
    return PubMedDocument(title, text, PUBMED_SOURCE, pmid)


def _text_from_first(element: ET.Element, paths: Sequence[str]) -> str:
    for path in paths:
        match = element.find(path)
        if match is not None:
            text = _element_text(match)
            if text:
                return text
    return ""


def _abstract_text(article: ET.Element) -> str:
    parts: list[str] = []
    for node in article.findall(".//Abstract/AbstractText"):
        label = node.attrib.get("Label", "").strip()
        text = _element_text(node)
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return " ".join(parts)


def _element_text(element: ET.Element) -> str:
    return " ".join(" ".join(element.itertext()).split())


def _fetch_json_url(url: str) -> object:
    return json.loads(_fetch_text_url(url))


def _fetch_text_url(url: str) -> str:
    request = Request(
        url,
        headers={"User-Agent": "PortakalPubMedWidget/0.1 (educational desktop app)"},
    )
    with urlopen(request, timeout=PUBMED_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="replace")


class PubMedScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[PubMedDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(
            fallback_pubmed_documents() if documents is None else documents
        )
        self._status_message = "Showing built-in PubMed sample. Enter a query to fetch live records."

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "PubMed",
                "Retrieve biomedical article abstracts as lightweight corpus documents.",
            )
        )
        layout.addWidget(self._build_query_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._render()

    def sizeHint(self) -> QSize:
        return QSize(980, 660)

    def minimumSizeHint(self) -> QSize:
        return QSize(740, 500)

    def _build_query_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._query_input = QLineEdit(self)
        self._query_input.setPlaceholderText("Search PubMed...")
        apply_readable_line_edit_style(self._query_input)
        layout.addWidget(self._query_input, 1)

        layout.addWidget(QLabel("Limit", self))
        self._limit_spinbox = QSpinBox(self)
        self._limit_spinbox.setRange(1, 20)
        self._limit_spinbox.setValue(DEFAULT_PUBMED_LIMIT)
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

        self._table = QTableWidget(0, 5, self)
        self._table.setHorizontalHeaderLabels(
            ["Title", "Source", "PMID", "Abstract Preview", "Words"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def fetch_articles(self) -> PubMedFetchResult:
        result = fetch_pubmed_documents(self._query_input.text(), self._limit_spinbox.value())
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
            self._set_item(row, 2, document.pmid)
            self._set_item(row, 3, preview_text(document.text))
            self._set_item(row, 4, str(count_words(document.text)))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [
                document.title,
                document.source,
                document.pmid,
                preview_text(document.text),
                str(count_words(document.text)),
            ]
            for document in self._documents
        ]
        summary = summarize_documents(self._documents)
        return {
            "summary": (
                f"PubMed: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.average_words_per_document:.1f} average words/document"
            ),
            "headers": ["Title", "Source", "PMID", "Abstract Preview", "Words"],
            "rows": rows,
        }
