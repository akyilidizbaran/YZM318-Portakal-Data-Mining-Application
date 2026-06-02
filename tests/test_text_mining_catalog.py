from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor

from portakal_app.app import create_application
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.catalog import build_categories, build_widgets
from portakal_app.ui.icons import get_widget_icon
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.shell.widget_catalog import WidgetCatalogButton
from portakal_app.ui.screens.bag_of_words_screen import (
    DF_IDF,
    DF_SMOOTH_IDF,
    NORM_L1,
    NORM_L2,
    TF_SUBLINEAR,
    BagOfWordsScreen,
    apply_global_weighting,
    apply_local_weighting,
    build_document_term_matrix,
    build_weighted_document_term_matrix,
    build_vocabulary,
    document_frequencies,
    format_matrix_value,
    normalize_matrix,
    summarize_bow,
    term_frequencies,
    tokenize_for_bow,
    weighting_summary,
)
from portakal_app.ui.screens.corpus_screen import (
    CorpusDocument,
    CorpusScreen,
    SAMPLE_CORPORA,
    SAMPLE_CORPUS,
    corpus_document_attributes,
    corpus_documents_from_payload,
    count_words,
    sample_corpus_by_id,
    summarize_corpus,
)
from portakal_app.ui.screens.corpus_viewer_screen import (
    CorpusViewerScreen,
    corpus_viewer_unique_word_count,
    corpus_viewer_words_from_payload,
    filter_corpus_viewer_documents,
    infer_corpus_document_category,
)
from portakal_app.ui.screens.create_corpus_screen import (
    BULK_SPLIT_BLANK_LINE,
    BULK_SPLIT_DASHES,
    CreateCorpusScreen,
    make_document,
    preview_text,
    split_bulk_documents,
)
from portakal_app.ui.screens.data_table_screen import DataTableScreen
from portakal_app.ui.screens.document_map_screen import DocumentMapScreen, build_document_map
from portakal_app.ui.screens.extract_keywords_screen import ExtractKeywordsScreen, extract_keywords
from portakal_app.ui.screens.guardian_screen import (
    GuardianDocument,
    GuardianScreen,
    build_guardian_search_url,
    fallback_guardian_documents,
    fetch_guardian_documents,
    get_guardian_api_key,
    parse_guardian_response,
    summarize_documents as summarize_guardian_documents,
)
from portakal_app.ui.screens.import_documents_screen import (
    CSVColumnMapping,
    DocumentImportOptions,
    DUPLICATE_ALLOW,
    DUPLICATE_REPLACE,
    DUPLICATE_SKIP,
    ImportDocumentsScreen,
    import_documents_from_paths,
    is_supported_document_path,
)
from portakal_app.ui.screens.ny_times_screen import (
    NYTimesDocument,
    NYTimesScreen,
    build_ny_times_search_url,
    fallback_ny_times_documents,
    fetch_ny_times_documents,
    get_ny_times_api_key,
    parse_ny_times_response,
    summarize_documents as summarize_ny_times_documents,
)
from portakal_app.ui.screens.placeholder_screen import PlaceholderScreen
from portakal_app.ui.screens.preprocess_text_screen import (
    PreprocessOptions,
    PreprocessTextScreen,
    TOKENIZER_REGEX,
    TOKENIZER_TWEET,
    TOKENIZER_WHITESPACE,
    build_ngrams,
    filter_tokens,
    parse_custom_stopwords,
    preprocess_documents,
    preprocess_text,
    preprocess_tokens,
    summarize_preprocessing,
    tokenize_text,
)
from portakal_app.ui.screens.pubmed_screen import (
    PubMedDocument,
    PubMedScreen,
    build_pubmed_fetch_url,
    build_pubmed_search_url,
    fallback_pubmed_documents,
    fetch_pubmed_documents,
    parse_pubmed_fetch_response,
    parse_pubmed_search_response,
    summarize_documents as summarize_pubmed_documents,
)
from portakal_app.ui.screens.twitter_screen import (
    TwitterDocument,
    TwitterScreen,
    build_twitter_search_url,
    fallback_twitter_documents,
    fetch_twitter_documents,
    get_twitter_bearer_token,
    parse_twitter_response,
    summarize_documents as summarize_twitter_documents,
)
from portakal_app.ui.screens.text_statistics_screen import (
    DocumentStatisticsOptions,
    TextStatisticsScreen,
    document_statistics_features,
    enrich_corpus_with_statistics,
    summarize_text_statistics,
)
from portakal_app.ui.screens.sentiment_analysis_screen import SentimentAnalysisScreen, analyze_sentiment
from portakal_app.ui.screens.topic_modelling_screen import TopicModellingScreen, build_topic_model
from portakal_app.ui.screens.wikipedia_screen import (
    WikipediaScreen,
    build_wikipedia_search_url,
    build_wikipedia_summary_url,
    fallback_wikipedia_documents,
    fetch_wikipedia_documents,
    parse_wikipedia_search_response,
    parse_wikipedia_summary_response,
    summarize_documents,
)
from portakal_app.ui.screens.word_cloud_screen import (
    WordCloudScreen,
    cloud_word_frequencies,
    word_cloud_documents_from_dataset,
    word_cloud_text_column,
)
from portakal_app.ui.screens.word_list_screen import (
    WordListScreen,
    build_updated_word_list,
    build_word_list,
)


PERSON_A_TEXT_MINING_WIDGETS = [
    ("text-corpus", "Corpus", ("Corpus",), ("Corpus",)),
    ("text-corpus-viewer", "Corpus Viewer", ("Corpus",), ("Corpus",)),
    ("text-import-documents", "Import Documents", (), ("Corpus",)),
    ("text-create-corpus", "Create Corpus", (), ("Corpus",)),
    ("text-preprocess", "Preprocess Text", ("Corpus",), ("Corpus",)),
    ("text-bag-of-words", "Bag of Words", ("Corpus",), ("Corpus",)),
    ("text-statistics", "Statistics", ("Corpus",), ("Corpus",)),
    ("text-word-list", "Word List", ("Corpus", "Words"), ("Words", "Selected Words", "Data")),
    ("text-word-cloud", "Word Cloud", ("Corpus", "Data"), ("Corpus", "Selected Word", "Word Counts")),
    ("text-extract-keywords", "Extract Keywords", ("Corpus",), ("Words",)),
    ("text-sentiment-analysis", "Sentiment Analysis", ("Corpus",), ("Corpus",)),
    ("text-topic-modelling", "Topic Modelling", ("Corpus",), ("Corpus", "Topic")),
    ("text-document-map", "Document Map", ("Corpus",), ()),
]

PERSON_A_TEXT_MINING_ICON_NAMES = {
    "text-corpus": "text_corpus",
    "text-corpus-viewer": "text_corpus_viewer",
    "text-import-documents": "text_import_documents",
    "text-create-corpus": "text_create_corpus",
    "text-preprocess": "text_preprocess",
    "text-bag-of-words": "text_bag_of_words",
    "text-statistics": "text_statistics",
    "text-word-list": "text_word_list",
    "text-word-cloud": "text_word_cloud",
    "text-extract-keywords": "text_extract_keywords",
    "text-sentiment-analysis": "text_sentiment_analysis",
    "text-topic-modelling": "text_topic_modelling",
    "text-document-map": "text_document_map",
}

REMOVED_PERSON_A_SOURCE_WIDGET_IDS = {
    "text-the-guardian",
    "text-ny-times",
    "text-pubmed",
    "text-twitter",
    "text-wikipedia",
}

TEXT_MINING_ASSET_DIR = Path(__file__).resolve().parents[1] / "src" / "portakal_app" / "ui" / "assets"


@pytest.fixture(scope="session")
def app():
    return create_application()


@pytest.fixture(autouse=True)
def english_i18n():
    previous = i18n.current_language()
    i18n.set_language("en")
    try:
        yield
    finally:
        i18n.set_language(previous)


def test_text_mining_category_exists():
    categories = {category.id: category for category in build_categories()}

    assert "text-mining" in categories
    assert categories["text-mining"].label == "Text Mining"
    assert categories["text-mining"].enabled is True


def test_text_mining_category_has_no_sidebar_icon():
    categories = {category.id: category for category in build_categories()}

    assert categories["text-mining"].icon_name == ""


def test_text_mining_category_contains_exact_person_a_widgets():
    text_widgets = [widget for widget in build_widgets() if widget.category_id == "text-mining"]

    expected_labels = [label for _id, label, _inputs, _outputs in PERSON_A_TEXT_MINING_WIDGETS]
    expected_ids = [widget_id for widget_id, _label, _inputs, _outputs in PERSON_A_TEXT_MINING_WIDGETS]

    assert [widget.label for widget in text_widgets] == expected_labels
    assert [widget.id for widget in text_widgets] == expected_ids


def test_text_mining_widget_ids_are_unique():
    widgets = build_widgets()
    widget_ids = [widget.id for widget in widgets]

    assert len(widget_ids) == len(set(widget_ids))


def test_text_mining_widgets_have_orange_like_svg_icons(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    for widget_id, expected_icon_name in PERSON_A_TEXT_MINING_ICON_NAMES.items():
        widget = widgets[widget_id]

        assert widget.icon_name == expected_icon_name
        if widget.icon_name.startswith("text_"):
            assert (TEXT_MINING_ASSET_DIR / f"{widget.icon_name}.svg").exists()
        assert not get_widget_icon(widget.icon_name).isNull()


def test_sidebar_hides_text_mining_category_icon_and_catalog_renders_widget_icons(app):
    window = MainWindow()
    window._sidebar.set_current_category("text-mining")
    app.processEvents()

    category_item = window._sidebar._items_by_category["text-mining"]
    cards = [
        card
        for card in window._catalog.findChildren(WidgetCatalogButton)
        if card.widget_id in PERSON_A_TEXT_MINING_ICON_NAMES
    ]

    assert category_item.icon().isNull()
    assert len(cards) == len(PERSON_A_TEXT_MINING_ICON_NAMES)
    assert all(not card.icon().isNull() for card in cards)


def test_text_mining_corpus_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-corpus"].screen_factory()

    assert isinstance(screen, CorpusScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 5
    assert screen._document_count_label.text().endswith("5")


def test_main_window_can_open_text_corpus_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-corpus")
    window._show_widget("text-corpus")

    assert isinstance(window._workspace.current_widget(), CorpusScreen)


def test_text_mining_corpus_viewer_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-corpus-viewer"].screen_factory()

    assert isinstance(screen, CorpusViewerScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 0
    assert "Connect a Corpus input" in screen._status_label.text()


def test_main_window_can_open_corpus_viewer_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-corpus-viewer")
    window._show_widget("text-corpus-viewer")

    assert isinstance(window._workspace.current_widget(), CorpusViewerScreen)


def test_text_mining_import_documents_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-import-documents"].screen_factory()

    assert isinstance(screen, ImportDocumentsScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 0


def test_main_window_can_open_import_documents_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-import-documents")
    window._show_widget("text-import-documents")

    assert isinstance(window._workspace.current_widget(), ImportDocumentsScreen)


def test_text_mining_create_corpus_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-create-corpus"].screen_factory()

    assert isinstance(screen, CreateCorpusScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 0


def test_main_window_can_open_create_corpus_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-create-corpus")
    window._show_widget("text-create-corpus")

    assert isinstance(window._workspace.current_widget(), CreateCorpusScreen)


def test_text_mining_preprocess_text_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-preprocess"].screen_factory()

    assert isinstance(screen, PreprocessTextScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 0
    assert "Connect a Corpus input" in screen._status_label.text()


def test_main_window_can_open_preprocess_text_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-preprocess")
    window._show_widget("text-preprocess")

    assert isinstance(window._workspace.current_widget(), PreprocessTextScreen)


def test_text_mining_bag_of_words_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-bag-of-words"].screen_factory()

    assert isinstance(screen, BagOfWordsScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 0
    assert "Connect a Corpus input" in screen._status_label.text()


def test_main_window_can_open_bag_of_words_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-bag-of-words")
    window._show_widget("text-bag-of-words")

    assert isinstance(window._workspace.current_widget(), BagOfWordsScreen)


def test_text_mining_statistics_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-statistics"].screen_factory()

    assert isinstance(screen, TextStatisticsScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 0
    assert "Connect a Corpus input" in screen._status_label.text()


def test_main_window_can_open_statistics_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-statistics")
    window._show_widget("text-statistics")

    assert isinstance(window._workspace.current_widget(), TextStatisticsScreen)


def test_text_mining_word_list_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-word-list"].screen_factory()

    assert isinstance(screen, WordListScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 0
    assert "Connect a Corpus input" in screen._status_label.text()


def test_main_window_can_open_word_list_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-word-list")
    window._show_widget("text-word-list")

    assert isinstance(window._workspace.current_widget(), WordListScreen)


def test_text_mining_word_cloud_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-word-cloud"].screen_factory()

    assert isinstance(screen, WordCloudScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 0
    assert "Connect a Corpus or Data input" in screen._status_label.text()


def test_main_window_can_open_word_cloud_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-word-cloud")
    window._show_widget("text-word-cloud")

    assert isinstance(window._workspace.current_widget(), WordCloudScreen)


def test_text_mining_extract_keywords_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-extract-keywords"].screen_factory()

    assert isinstance(screen, ExtractKeywordsScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 0
    assert "Connect a Corpus input" in screen._status_label.text()


def test_main_window_can_open_extract_keywords_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-extract-keywords")
    window._show_widget("text-extract-keywords")

    assert isinstance(window._workspace.current_widget(), ExtractKeywordsScreen)


def test_text_mining_sentiment_analysis_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-sentiment-analysis"].screen_factory()

    assert isinstance(screen, SentimentAnalysisScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 0
    assert "Connect a Corpus input" in screen._status_label.text()


def test_main_window_can_open_sentiment_analysis_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-sentiment-analysis")
    window._show_widget("text-sentiment-analysis")

    assert isinstance(window._workspace.current_widget(), SentimentAnalysisScreen)


def test_text_mining_topic_modelling_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-topic-modelling"].screen_factory()

    assert isinstance(screen, TopicModellingScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._topics_table.rowCount() == 0
    assert "Connect a Corpus input" in screen._status_label.text()


def test_main_window_can_open_topic_modelling_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-topic-modelling")
    window._show_widget("text-topic-modelling")

    assert isinstance(window._workspace.current_widget(), TopicModellingScreen)


def test_text_mining_document_map_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-document-map"].screen_factory()

    assert isinstance(screen, DocumentMapScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert screen._table.rowCount() == 0
    assert "Connect a Corpus input" in screen._status_label.text()


def test_main_window_can_open_document_map_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-document-map")
    window._show_widget("text-document-map")

    assert isinstance(window._workspace.current_widget(), DocumentMapScreen)


def test_removed_person_a_source_widgets_are_not_registered():
    widgets = {widget.id: widget for widget in build_widgets()}

    assert REMOVED_PERSON_A_SOURCE_WIDGET_IDS.isdisjoint(widgets)


def test_text_source_widgets_expose_corpus_output_payloads(app):
    screens = (
        GuardianScreen(documents=(GuardianDocument("Guardian A", "one two", "The Guardian", "Tech"),)),
        NYTimesScreen(documents=(NYTimesDocument("NYT A", "three four", "NY Times", "World"),)),
        PubMedScreen(documents=(PubMedDocument("PubMed A", "five six", "PubMed", "123"),)),
        TwitterScreen(documents=(TwitterDocument("42", "seven eight", "Twitter/X", "user"),)),
        WikipediaScreen(documents=(CorpusDocument("Wiki A", "nine ten", "Wikipedia"),)),
    )
    expected_titles = ("Guardian A", "NYT A", "PubMed A", "Post 42", "Wiki A")

    for screen, expected_title in zip(screens, expected_titles):
        payload = screen.current_output_payload()

        assert payload.port_label == "Corpus"
        assert len(payload.value) == 1
        assert isinstance(payload.value[0], CorpusDocument)
        assert payload.value[0].title == expected_title


def test_no_person_a_text_mining_widget_uses_placeholder_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    for widget_id, _label, _inputs, _outputs in PERSON_A_TEXT_MINING_WIDGETS:
        widget = widgets[widget_id]
        screen = widget.screen_factory()

        assert not isinstance(screen, PlaceholderScreen)
        assert widget.enabled is True
        assert widget.description


def test_text_mining_widgets_register_expected_ports():
    widgets = {widget.id: widget for widget in build_widgets()}

    for widget_id, _label, input_labels, output_labels in PERSON_A_TEXT_MINING_WIDGETS:
        widget = widgets[widget_id]

        assert tuple(port.label for port in widget.input_ports) == input_labels
        assert tuple(port.label for port in widget.output_ports) == output_labels


def test_corpus_widget_accepts_multiple_corpus_inputs():
    widget = {widget.id: widget for widget in build_widgets()}["text-corpus"]

    assert widget.input_channels == ("Corpus",)
    assert widget.multi_input_channels == ("Corpus",)


def test_corpus_summary_counts_documents_and_words():
    documents = (
        CorpusDocument("First", "one two three"),
        CorpusDocument("Second", "four five"),
    )

    summary = summarize_corpus(documents)

    assert summary.document_count == 2
    assert summary.total_word_count == 5
    assert summary.average_words_per_document == 2.5


def test_corpus_screen_exposes_multiple_built_in_sample_corpora(app):
    screen = CorpusScreen()
    sample_ids = [sample.id for sample in SAMPLE_CORPORA]

    assert len(SAMPLE_CORPORA) >= 3
    assert len(sample_ids) == len(set(sample_ids))
    assert screen._sample_combo.count() == len(SAMPLE_CORPORA)
    assert screen._sample_combo.currentData() == SAMPLE_CORPORA[0].id
    assert screen.current_output_payload().value == SAMPLE_CORPUS
    assert SAMPLE_CORPORA[0].documents == SAMPLE_CORPUS
    assert "Built-in sample corpus" in screen._status_label.text()


def test_corpus_screen_sample_selection_updates_visible_documents(app):
    screen = CorpusScreen()
    target_sample = sample_corpus_by_id("product-reviews")

    screen._sample_combo.setCurrentIndex(screen._sample_index(target_sample.id))

    assert screen._table.rowCount() == len(target_sample.documents)
    assert screen._table.item(0, 0).text() == target_sample.documents[0].title
    assert screen._table.item(0, 1).text() == target_sample.documents[0].source
    assert screen.current_output_payload().value == target_sample.documents
    assert target_sample.description in screen._sample_description_label.text()


def test_corpus_screen_input_overrides_sample_and_reset_restores_selected_sample(app):
    screen = CorpusScreen()
    target_sample = sample_corpus_by_id("support-tickets")
    input_documents = (CorpusDocument("Incoming", "one two", "Input"),)

    screen._sample_combo.setCurrentIndex(screen._sample_index(target_sample.id))
    screen.set_input_payload(WorkflowPayload("Corpus", input_documents))

    assert screen._sample_combo.isEnabled() is False
    assert screen._table.rowCount() == 1
    assert screen._table.item(0, 0).text() == "Incoming"
    assert screen.current_output_payload().value == input_documents

    screen.set_input_payload(None)

    assert screen._sample_combo.isEnabled() is True
    assert screen._table.rowCount() == len(target_sample.documents)
    assert screen._table.item(0, 0).text() == target_sample.documents[0].title
    assert screen.current_output_payload().value == target_sample.documents
    assert target_sample.name in screen._status_label.text()


def test_empty_corpus_summary_and_screen_are_safe(app):
    summary = summarize_corpus(())
    screen = CorpusScreen(documents=())
    preview = screen.data_preview_snapshot()

    assert summary.document_count == 0
    assert summary.total_word_count == 0
    assert summary.average_words_per_document == 0.0
    assert screen._table.rowCount() == 0
    assert preview["rows"] == []
    assert "0 documents" in preview["summary"]


def test_corpus_screen_accepts_input_payload_and_resets_to_sample(app):
    documents = (
        CorpusDocument("Manual One", "alpha beta", "Manual"),
        CorpusDocument("Manual Two", "gamma delta epsilon", "Manual"),
    )
    screen = CorpusScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))

    assert screen._table.rowCount() == 2
    assert screen._table.item(0, 0).text() == "Manual One"
    assert screen._table.item(0, 1).text() == "Manual"
    assert screen._document_count_label.text().endswith("2")
    assert "Input corpus is connected" in screen._status_label.text()

    screen.set_input_payload(None)

    assert screen._table.rowCount() == len(SAMPLE_CORPUS)
    assert screen._table.item(0, 0).text() == SAMPLE_CORPUS[0].title
    assert "Built-in sample corpus" in screen._status_label.text()


def test_corpus_screen_merges_multiple_input_payloads_in_order(app):
    manual_documents = (
        CorpusDocument("Manual One", "alpha beta", "Create Corpus"),
        CorpusDocument("Manual Two", "gamma delta", "Create Corpus"),
    )
    wikipedia_documents = (
        CorpusDocument("Wikipedia One", "encyclopedia text", "Wikipedia"),
    )
    screen = CorpusScreen()

    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Corpus", manual_documents))
    screen.set_input_payload(WorkflowPayload("Corpus", wikipedia_documents))

    assert screen._table.rowCount() == 3
    assert screen._table.item(0, 0).text() == "Manual One"
    assert screen._table.item(1, 0).text() == "Manual Two"
    assert screen._table.item(2, 0).text() == "Wikipedia One"
    assert screen.current_output_payload().value == (*manual_documents, *wikipedia_documents)
    assert "Merged 2 input corpora" in screen._status_label.text()


def test_corpus_screen_any_connected_input_overrides_sample_even_when_empty(app):
    screen = CorpusScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", ()))

    assert screen._table.rowCount() == 0
    assert screen.current_output_payload().value == ()
    assert "Input corpus is connected but empty" in screen._status_label.text()


def test_corpus_payload_parser_rejects_non_corpus_items():
    documents = (CorpusDocument("First", "one two"),)

    assert corpus_documents_from_payload(documents) == documents
    assert corpus_documents_from_payload("not a corpus") is None
    assert corpus_documents_from_payload((object(),)) is None


def test_corpus_viewer_accepts_input_and_passes_same_corpus(app):
    documents = (
        CorpusDocument("First", "apple banana", "Manual"),
        CorpusDocument("Second", "carrot", "Manual"),
    )
    screen = CorpusViewerScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))

    assert screen._table.rowCount() == 2
    assert screen._table.item(0, 0).text() == "First"
    assert screen._table.item(0, 4).text() == "2"
    assert screen.current_output_payload().port_label == "Corpus"
    assert screen.current_output_payload().value == documents


def test_corpus_viewer_infers_category_from_known_corpus_words():
    assert (
        infer_corpus_document_category(
            CorpusDocument("Quarterly Update", "The market improved.", "business/report.txt")
        )
        == "business"
    )
    assert (
        infer_corpus_document_category(
            CorpusDocument("Match Review", "The sport result changed quickly.", "News")
        )
        == "sport"
    )
    assert infer_corpus_document_category(CorpusDocument("Plain", "neutral words", "Manual")) == "-"


def test_corpus_viewer_filter_finds_matching_documents_and_counts_matches(app):
    documents = (
        CorpusDocument("Market Report", "profit market growth", "News"),
        CorpusDocument("Weather", "rain tomorrow", "Forecast"),
        CorpusDocument("Profit Note", "profit rose again", "Notes"),
    )
    screen = CorpusViewerScreen()

    result = filter_corpus_viewer_documents(documents, "profit|market")
    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen._filter_input.setText("profit|market")
    app.processEvents()

    assert result.row_indexes == (0, 2)
    assert result.matching_documents == 2
    assert result.matches == 5
    assert corpus_viewer_unique_word_count(documents) == 7
    assert screen._table.rowCount() == 2
    assert screen._table.item(0, 0).text() == "Market Report"
    assert screen._matching_documents_label.text().endswith("2 / 3")
    assert screen._matches_label.text().endswith("5")
    assert "ffe58a" in screen._detail_content.toHtml()
    assert screen.current_output_payload().value == (documents[0], documents[2])


def test_corpus_viewer_highlights_words_payload_without_filter(app):
    documents = (
        CorpusDocument("Market Report", "profit dollar market shares", "News"),
        CorpusDocument("Weather", "rain tomorrow", "Forecast"),
    )
    screen = CorpusViewerScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen.set_input_payload(WorkflowPayload("Words", ("profit", "dollar", "market", "shares")))
    app.processEvents()

    assert corpus_viewer_words_from_payload(("profit", ["dollar", 2], "market shares")) == (
        "profit",
        "dollar",
        "market",
        "shares",
    )
    assert screen._table.rowCount() == 2
    assert len(screen._content_highlight_spans(documents[0].text)) == 4
    assert "ffe58a" in screen._detail_content.toHtml()
    assert screen.current_output_payload().value == documents


def test_corpus_viewer_selected_words_payload_limits_highlight(app):
    documents = (CorpusDocument("Market Report", "profit dollar market shares", "News"),)
    screen = CorpusViewerScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen.set_input_payload(WorkflowPayload("Selected Words", ("profit",)))
    app.processEvents()

    assert screen._highlight_words == ("profit",)
    assert screen._content_highlight_spans(documents[0].text) == ((0, 6),)

    screen.set_input_payload(WorkflowPayload("Selected Words", ()))
    app.processEvents()

    assert screen._highlight_words == ()
    assert screen._content_highlight_spans(documents[0].text) == ()


def test_corpus_viewer_filter_can_match_category_feature(app):
    documents = (
        CorpusDocument("Quarterly Update", "The market improved.", "business/report.txt"),
        CorpusDocument("Weather", "rain tomorrow", "Forecast"),
    )
    screen = CorpusViewerScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen._search_title_feature.setChecked(False)
    screen._search_source_feature.setChecked(False)
    screen._search_content_feature.setChecked(False)
    screen._search_category_feature.setChecked(True)
    screen._filter_input.setText("business")
    app.processEvents()

    assert screen._table.rowCount() == 1
    assert screen._table.item(0, 0).text() == "Quarterly Update"
    assert screen._table.item(0, 1).text() == "business"
    assert screen._detail_category_label.text() == "Category: business"


def test_corpus_viewer_filter_respects_search_scope(app):
    documents = (
        CorpusDocument("Profit Title", "plain body", "Manual"),
        CorpusDocument("Body", "profit in content", "Manual"),
    )
    screen = CorpusViewerScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen._search_category_feature.setChecked(False)
    screen._search_title_feature.setChecked(False)
    screen._search_source_feature.setChecked(False)
    screen._search_content_feature.setChecked(True)
    screen._filter_input.setText("profit")
    app.processEvents()

    assert screen._table.rowCount() == 1
    assert screen._table.item(0, 0).text() == "Body"


def test_corpus_viewer_display_features_can_hide_content_without_crashing(app):
    documents = (CorpusDocument("First", "profit market", "Manual"),)
    screen = CorpusViewerScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen._display_content_feature.setChecked(False)
    app.processEvents()

    assert screen._detail_title_label.text() == "Title / Name: First"
    assert screen._detail_source_label.text() == "Source / Path: Manual"
    assert "Content display disabled" in screen._detail_content.toPlainText()


def test_corpus_viewer_display_features_can_hide_category_without_crashing(app):
    documents = (CorpusDocument("Quarterly Update", "The market improved.", "business/report.txt"),)
    screen = CorpusViewerScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen._display_category_feature.setChecked(False)
    app.processEvents()

    assert screen._detail_category_label.isHidden()
    assert screen._detail_title_label.text() == "Title / Name: Quarterly Update"


def test_corpus_viewer_invalid_regex_is_safe(app):
    documents = (CorpusDocument("First", "profit market", "Manual"),)
    screen = CorpusViewerScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen._filter_input.setText("[")
    app.processEvents()

    assert screen._table.rowCount() == 0
    assert "Invalid regular expression" in screen._status_label.text()
    assert screen.current_output_payload().value == ()


def test_text_statistics_summary_and_screen_use_corpus_words(app):
    documents = (
        CorpusDocument("Long", "apple banana apple", "Manual"),
        CorpusDocument("Short", "banana", "Manual"),
    )
    summary = summarize_text_statistics(documents)
    screen = TextStatisticsScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))

    assert summary.document_count == 2
    assert summary.total_word_count == 4
    assert summary.unique_word_count == 2
    assert summary.longest_document_title == "Long"
    assert summary.shortest_document_title == "Short"
    assert screen._table.item(0, 0).text() == "apple"
    assert screen._table.item(0, 1).text() == "2"
    payload = screen.current_output_payload()
    assert payload.port_label == "Corpus"
    assert len(payload.value) == len(documents)
    attributes = corpus_document_attributes(payload.value[0])
    assert attributes["word_count"] == 3
    assert attributes["character_count"] == len("apple banana apple")
    assert attributes["percent_unique_words"] == 66.67


def test_text_statistics_enriches_corpus_with_selected_features(app):
    document = CorpusDocument("Market", "Profit, market profit!", "Manual")
    options = DocumentStatisticsOptions(contains=True, contains_pattern="profit")

    features = document_statistics_features(document, options)
    enriched = enrich_corpus_with_statistics((document,), options)

    assert features["word_count"] == 3
    assert features["character_count"] == len(document.text)
    assert features["average_word_length"] == 6.0
    assert features["percent_unique_words"] == 66.67
    assert features["punctuation_count"] == 2
    assert features["contains_profit"] == 2
    assert enriched[0].title == document.title
    assert enriched[0].text == document.text
    assert corpus_document_attributes(enriched[0])["contains_profit"] == 2


def test_text_statistics_apply_updates_contains_output(app):
    documents = (CorpusDocument("Market", "profit dollar profit", "Manual"),)
    screen = TextStatisticsScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen._contains_checkbox.setChecked(True)
    screen._contains_input.setText("profit")
    screen.apply_statistics()
    app.processEvents()

    output_documents = screen.current_output_payload().value
    assert len(output_documents) == 1
    assert corpus_document_attributes(output_documents[0])["contains_profit"] == 2


def test_word_list_builds_frequency_and_document_counts(app):
    documents = (
        CorpusDocument("First", "apple apple banana"),
        CorpusDocument("Second", "banana carrot apple"),
    )
    items = build_word_list(documents)
    screen = WordListScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))

    assert items[0].word == "apple"
    assert items[0].frequency == 3
    assert items[0].document_count == 2
    assert screen._table.rowCount() == 3
    assert screen._table.item(0, 0).text() == "apple"
    assert screen._table.item(0, 3).text() == "Corpus"
    payload = screen.current_output_payload()
    dataset = payload.dataset
    assert payload.port_label == "Data"
    assert dataset is not None
    assert dataset.dataframe.columns == ["Word", "Frequency", "Documents", "Source"]
    assert dataset.dataframe.get_column("Word").to_list()[0] == "apple"
    assert dataset.dataframe.get_column("Frequency").to_list()[0] == 3
    assert dataset.dataframe.get_column("Documents").to_list()[0] == 2
    payloads = screen.current_output_payloads()
    assert payloads["Words"].value == ("apple", "banana", "carrot")
    assert payloads["Selected Words"].value == ()
    assert payloads["Data"].dataset is dataset


def test_word_list_custom_words_and_update_modes(app):
    documents = (
        CorpusDocument("First", "apple apple banana"),
        CorpusDocument("Second", "banana carrot apple"),
    )
    screen = WordListScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen.add_custom_word("apple dragonfruit")

    screen.set_update_mode("Intersection")
    app.processEvents()
    assert [screen._table.item(row, 0).text() for row in range(screen._table.rowCount())] == ["apple"]
    assert screen._table.item(0, 3).text() == "Corpus + Custom"

    screen.set_update_mode("Union")
    app.processEvents()
    words = {screen._table.item(row, 0).text() for row in range(screen._table.rowCount())}
    assert {"apple", "banana", "carrot", "dragonfruit"}.issubset(words)
    dragonfruit_row = next(
        row for row in range(screen._table.rowCount()) if screen._table.item(row, 0).text() == "dragonfruit"
    )
    assert screen._table.item(dragonfruit_row, 1).text() == "0"
    assert screen._table.item(dragonfruit_row, 3).text() == "Custom"

    screen.set_update_mode("Ignore input")
    app.processEvents()
    assert {screen._table.item(row, 0).text() for row in range(screen._table.rowCount())} == {
        "apple",
        "dragonfruit",
    }
    payload = screen.current_output_payload()
    dataset = payload.dataset
    assert dataset is not None
    assert set(dataset.dataframe.get_column("Word").to_list()) == {"apple", "dragonfruit"}
    dragonfruit = dataset.dataframe.filter(dataset.dataframe.get_column("Word") == "dragonfruit").row(0, named=True)
    assert dragonfruit["Frequency"] == 0
    assert dragonfruit["Documents"] == 0
    assert dragonfruit["Source"] == "Custom"


def test_word_list_only_corpus_default_search_sort_and_stopword_warning(app):
    documents = (
        CorpusDocument("First", "the the the profit apple"),
        CorpusDocument("Second", "profit market"),
    )
    screen = WordListScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))

    assert screen._update_mode_combo.currentText() == "Only input"
    assert screen._table.item(0, 0).text() == "the"
    assert "Common stopwords detected" in screen._status_label.text()

    screen._search_input.setText("profit")
    app.processEvents()
    assert screen._table.rowCount() == 1
    assert screen._table.item(0, 0).text() == "profit"

    screen._search_input.clear()
    screen._sort_combo.setCurrentText("Sort Alphabetically")
    app.processEvents()
    assert screen._table.item(0, 0).text() == "apple"


def test_word_list_update_helper_modes_are_deterministic():
    corpus_items = (
        build_word_list((CorpusDocument("First", "apple apple banana"),))[0],
        build_word_list((CorpusDocument("Second", "banana carrot"),))[0],
    )

    rows = build_updated_word_list(corpus_items, ("apple", "dragonfruit"), update_mode="Union")

    assert any(row.word == "dragonfruit" and row.source == "Custom" for row in rows)
    assert any(row.word == "apple" and row.source == "Corpus + Custom" for row in rows)

    ignored_input = build_updated_word_list(corpus_items, ("apple", "dragonfruit"), update_mode="Ignore input")
    assert {row.word for row in ignored_input} == {"apple", "dragonfruit"}


def test_word_list_words_selected_words_and_data_outputs(app):
    documents = (
        CorpusDocument("First", "profit profit market"),
        CorpusDocument("Second", "dollar shares market"),
    )
    screen = WordListScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen.add_custom_word("profit, dollar, market, shares")
    screen.set_update_mode("Ignore input")
    app.processEvents()

    payloads = screen.current_output_payloads()
    assert set(payloads["Words"].value) == {"profit", "dollar", "market", "shares"}
    assert payloads["Data"].dataset.dataframe.columns == ["Word", "Frequency", "Documents", "Source"]

    profit_row = next(
        row for row in range(screen._table.rowCount()) if screen._table.item(row, 0).text() == "profit"
    )
    screen._table.selectRow(profit_row)
    app.processEvents()

    payloads = screen.current_output_payloads()
    assert payloads["Selected Words"].value == ("profit",)
    profit = payloads["Data"].dataset.dataframe.filter(
        payloads["Data"].dataset.dataframe.get_column("Word") == "profit"
    ).row(0, named=True)
    assert profit["Frequency"] == 2
    assert profit["Documents"] == 1
    assert profit["Source"] == "Corpus + Custom"


def test_word_list_custom_words_get_frequency_counts_from_corpus_input(app):
    documents = (
        CorpusDocument("Market", "profit profits dollar market shares shares", "Manual"),
        CorpusDocument("Finance", "market dollar profit", "Manual"),
    )
    screen = WordListScreen()

    screen.add_custom_word("profit profits dollar market shares")
    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen.set_update_mode("Ignore input")
    app.processEvents()

    rows = {
        screen._table.item(row, 0).text(): (
            int(screen._table.item(row, 1).text()),
            int(screen._table.item(row, 2).text()),
            screen._table.item(row, 3).text(),
        )
        for row in range(screen._table.rowCount())
    }
    assert rows["profit"] == (2, 2, "Corpus + Custom")
    assert rows["shares"] == (2, 1, "Corpus + Custom")
    assert rows["market"] == (2, 2, "Corpus + Custom")


def test_word_list_custom_words_without_corpus_keep_zero_counts(app):
    screen = WordListScreen()

    screen.add_custom_word("profit dollar")
    app.processEvents()

    rows = {
        screen._table.item(row, 0).text(): (
            screen._table.item(row, 1).text(),
            screen._table.item(row, 2).text(),
            screen._table.item(row, 3).text(),
        )
        for row in range(screen._table.rowCount())
    }
    assert rows == {"dollar": ("0", "0", "Custom"), "profit": ("0", "0", "Custom")}
    assert "Connect a Corpus input to add frequency counts" in screen._status_label.text()


def test_word_list_combines_words_input_with_corpus_counts(app):
    documents = (
        CorpusDocument("Market", "profit profit market", "Manual"),
        CorpusDocument("Dollar", "dollar market", "Manual"),
    )
    screen = WordListScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen.set_input_payload(WorkflowPayload("Words", ("profit", "shares")))
    screen.add_custom_word("market shares")
    screen.set_update_mode("Union")
    app.processEvents()

    rows = {
        screen._table.item(row, 0).text(): (
            int(screen._table.item(row, 1).text()),
            int(screen._table.item(row, 2).text()),
            screen._table.item(row, 3).text(),
        )
        for row in range(screen._table.rowCount())
    }
    assert rows["profit"] == (2, 1, "Corpus + Input")
    assert rows["market"] == (2, 2, "Corpus + Custom")
    assert rows["shares"] == (0, 0, "Input + Custom")


def test_word_cloud_builds_frequency_view_from_corpus(app):
    documents = (
        CorpusDocument("First", "apple apple banana"),
        CorpusDocument("Second", "banana carrot apple"),
    )
    screen = WordCloudScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))

    assert cloud_word_frequencies(documents) == (("apple", 3), ("banana", 2), ("carrot", 1))
    assert screen._table.rowCount() == 3
    assert any(word.word == "apple" for word in screen._cloud_canvas.words())
    assert screen.current_output_payloads()["Word Counts"].dataset.dataframe.columns == ["Word", "Frequency"]


def test_word_cloud_builds_frequency_view_from_data_text_column(app):
    dataset = GeneratedDatasetService().build_dataset(
        pl.DataFrame(
            {
                "Title": ["First", "Second"],
                "Content": ["profit profit market", "market shares"],
            }
        ),
        dataset_id="word-cloud-data",
        display_name="Word Cloud Data",
        file_name="word-cloud-data.csv",
        role_overrides={"Title": "meta", "Content": "meta"},
    )
    screen = WordCloudScreen()

    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()

    assert word_cloud_text_column(dataset) == "Content"
    assert [document.text for document in word_cloud_documents_from_dataset(dataset)] == [
        "profit profit market",
        "market shares",
    ]
    assert screen._table.rowCount() == 3
    assert screen._table.item(0, 0).text() in {"market", "profit"}
    assert "Using text column 'Content'" in screen._status_label.text()


def test_word_cloud_handles_data_without_text_column_without_crashing(app):
    dataset = GeneratedDatasetService().build_dataset(
        pl.DataFrame({"Amount": [1, 2], "Score": [3.5, 4.5]}),
        dataset_id="word-cloud-numeric-data",
        display_name="Numeric Data",
        file_name="word-cloud-numeric-data.csv",
    )
    screen = WordCloudScreen()

    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()

    assert screen._table.rowCount() == 0
    assert screen._frequencies == ()
    assert "No text column found for Word Cloud" in screen._status_label.text()


def test_word_cloud_canvas_avoids_overlapping_word_rects(app):
    text = " ".join(
        word
        for index in range(100)
        for word in [f"word{index}"] * (100 - index)
    )
    screen = WordCloudScreen()
    screen._cloud_canvas.resize(1120, 420)
    screen._top_words_spinbox.setValue(100)

    screen.set_input_payload(WorkflowPayload("Corpus", (CorpusDocument("Dense", text, "Manual"),)))
    app.processEvents()

    words = screen._cloud_canvas.words()
    assert len(words) >= 45
    for index, word in enumerate(words):
        for other in words[index + 1:]:
            assert not word.rect.intersects(other.rect), f"{word.word} overlaps {other.word}"


def test_word_cloud_hover_and_selection_sync_table_and_canvas(app):
    documents = (
        CorpusDocument("First", "profit profit market"),
        CorpusDocument("Second", "market shares"),
    )
    screen = WordCloudScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))
    screen._set_hovered_word("profit")
    app.processEvents()

    assert screen._hovered_word == "profit"
    assert screen._cloud_canvas._hovered_word == "profit"
    profit_row = next(
        row for row in range(screen._table.rowCount()) if screen._table.item(row, 0).text() == "profit"
    )
    assert screen._table.item(profit_row, 0).background().color() == QColor("#fff1ce")

    market_row = next(
        row for row in range(screen._table.rowCount()) if screen._table.item(row, 0).text() == "market"
    )
    screen._table.selectRow(market_row)
    app.processEvents()

    assert screen.selected_word() == "market"
    assert screen._cloud_canvas._selected_word == "market"
    assert screen.current_output_payloads()["Selected Word"].value == "market"
    assert {document.title for document in screen.current_output_payload().value} == {"First", "Second"}


def test_extract_keywords_uses_corpus_terms_and_outputs_keywords(app):
    documents = (
        CorpusDocument("First", "apple apple banana"),
        CorpusDocument("Second", "carrot banana carrot"),
    )
    keywords = extract_keywords(documents, top_n=2)
    screen = ExtractKeywordsScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))

    assert any(item.keyword == "apple" for item in keywords)
    assert screen._table.rowCount() == 4
    assert screen.current_output_payload().port_label == "Words"
    assert screen.current_output_payload().value
    assert screen._table.item(0, 1).text() in {"apple", "carrot"}


def test_sentiment_analysis_scores_positive_negative_and_neutral_documents(app):
    documents = (
        CorpusDocument("Positive", "good useful başarılı"),
        CorpusDocument("Negative", "bad error kötü"),
        CorpusDocument("Neutral", "plain corpus text"),
    )
    results = analyze_sentiment(documents)
    screen = SentimentAnalysisScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))

    assert [result.label for result in results] == ["Positive", "Negative", "Neutral"]
    assert screen._table.rowCount() == 3
    assert screen._table.item(0, 4).text() == "Positive"
    assert screen._table.item(1, 4).text() == "Negative"
    assert screen._neutral_count_label.text().endswith("1")


def test_topic_modelling_builds_topics_and_document_assignments(app):
    documents = (
        CorpusDocument("Fruit A", "apple banana fruit apple"),
        CorpusDocument("Fruit B", "banana orange fruit"),
        CorpusDocument("Tech", "model data mining model"),
    )
    result = build_topic_model(documents, topic_count=2, top_words=3)
    screen = TopicModellingScreen()
    screen._topic_count_spinbox.setValue(2)
    screen._top_words_spinbox.setValue(3)

    screen.set_input_payload(WorkflowPayload("Corpus", documents))

    assert len(result.topics) == 2
    assert len(result.document_topics) == 3
    assert screen._topics_table.rowCount() == 2
    assert screen._documents_table.rowCount() == 3
    assert screen.current_output_payload().port_label == "Corpus"
    assert "Topic" in screen.current_output_payloads()


def test_topic_modelling_handles_too_small_corpus_without_crashing(app):
    screen = TopicModellingScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", (CorpusDocument("Only", "one document"),)))

    assert screen._topics_table.rowCount() == 0
    assert "At least 2" in screen._status_label.text()


def test_document_map_builds_two_document_fallback_and_table(app):
    documents = (
        CorpusDocument("First", "apple banana"),
        CorpusDocument("Second", "carrot date"),
    )
    result = build_document_map(documents)
    screen = DocumentMapScreen()

    screen.set_input_payload(WorkflowPayload("Corpus", documents))

    assert len(result.coordinates) == 2
    assert result.coordinates[0].x == -1.0
    assert result.coordinates[1].x == 1.0
    assert screen._table.rowCount() == 2
    assert screen._table.item(0, 0).text() == "First"


def test_create_corpus_document_uses_title_source_preview_and_word_count():
    document = make_document("Manual Note", "alpha beta gamma", "Notes", index=1)

    assert document.title == "Manual Note"
    assert document.source == "Notes"
    assert preview_text(document.text) == "alpha beta gamma"
    assert count_words(document.text) == 3


def test_create_corpus_empty_title_and_text_are_safe():
    document = make_document("  ", "", "", index=2)

    assert document.title == "Document 2"
    assert document.source == "Manual"
    assert count_words(document.text) == 0


def test_create_corpus_summary_counts_manual_documents():
    documents = (
        make_document("First", "one two", "Manual", index=1),
        make_document("Second", "three four five", "Manual", index=2),
    )

    summary = summarize_corpus(documents)

    assert summary.document_count == 2
    assert summary.total_word_count == 5
    assert summary.average_words_per_document == 2.5


def test_create_corpus_bulk_split_modes():
    assert split_bulk_documents("one\n\ntwo", BULK_SPLIT_BLANK_LINE) == ("one", "two")
    assert split_bulk_documents("one\n---\ntwo", BULK_SPLIT_DASHES) == ("one", "two")
    assert split_bulk_documents("   ", BULK_SPLIT_BLANK_LINE) == ()


def test_create_corpus_screen_adds_and_clears_documents(app):
    screen = CreateCorpusScreen()

    document = screen.add_document("", "one two", "")

    assert document.title == "Document 1"
    assert document.source == "Manual"
    assert screen._table.rowCount() == 1
    assert screen._table.item(0, 0).text() == "Document 1"
    assert screen._table.item(0, 1).text() == "Manual"
    assert screen._table.item(0, 2).text() == "one two"
    assert screen._table.item(0, 3).text() == "2"

    screen.clear_corpus()

    assert screen._table.rowCount() == 0
    assert screen.data_preview_snapshot()["rows"] == []


def test_create_corpus_screen_selects_and_updates_existing_document(app):
    screen = CreateCorpusScreen()
    screen.add_document("Original", "one two", "Draft")

    screen._table.selectRow(0)
    app.processEvents()
    updated = screen.update_selected_document("Updated", "three four five", "Final")

    assert updated == CorpusDocument("Updated", "three four five", "Final")
    assert screen._table.rowCount() == 1
    assert screen._table.item(0, 0).text() == "Updated"
    assert screen._table.item(0, 1).text() == "Final"
    assert screen._table.item(0, 3).text() == "3"
    assert screen.current_output_payload().value == (updated,)


def test_create_corpus_screen_removes_selected_document(app):
    screen = CreateCorpusScreen()
    first = screen.add_document("First", "one", "Manual")
    second = screen.add_document("Second", "two", "Manual")

    screen._table.selectRow(0)
    app.processEvents()
    removed = screen.remove_selected_document()

    assert removed == first
    assert screen.current_output_payload().value == (second,)
    assert screen._table.rowCount() == 1
    assert screen._table.item(0, 0).text() == "Second"


def test_create_corpus_screen_moves_documents_up_and_down(app):
    screen = CreateCorpusScreen()
    first = screen.add_document("First", "one", "Manual")
    second = screen.add_document("Second", "two", "Manual")
    third = screen.add_document("Third", "three", "Manual")

    screen._table.selectRow(2)
    app.processEvents()
    assert screen.move_selected_up() is True

    assert screen.current_output_payload().value == (first, third, second)

    screen._table.selectRow(1)
    app.processEvents()
    assert screen.move_selected_down() is True

    assert screen.current_output_payload().value == (first, second, third)


def test_create_corpus_screen_adds_bulk_documents_and_updates_metadata(app):
    screen = CreateCorpusScreen()

    added = screen.add_bulk_documents(
        "First bulk document\n\nSecond bulk document",
        source="Bulk",
        mode=BULK_SPLIT_BLANK_LINE,
    )

    assert [document.title for document in added] == ["Document 1", "Document 2"]
    assert [document.source for document in added] == ["Bulk", "Bulk"]
    assert screen._table.rowCount() == 2
    assert screen._document_count_label.text().endswith("2")
    assert screen._category_count_label.text().endswith("1")
    assert screen._empty_document_count_label.text().endswith("0")
    assert "1 categories" in screen.data_preview_snapshot()["summary"]


def test_create_corpus_screen_adds_bulk_documents_split_by_dashes(app):
    screen = CreateCorpusScreen()

    added = screen.add_bulk_documents(
        "Alpha text\n---\nBeta text",
        source="Dash",
        mode=BULK_SPLIT_DASHES,
    )

    assert [document.text for document in added] == ["Alpha text", "Beta text"]
    assert screen._table.rowCount() == 2


def test_create_corpus_outputs_corpus_payload(app):
    screen = CreateCorpusScreen()
    document = screen.add_document("First", "one two", "Manual")

    payload = screen.current_output_payload()

    assert payload.port_label == "Corpus"
    assert payload.value == (document,)


def test_create_corpus_output_can_update_connected_corpus_widget(app):
    source = CreateCorpusScreen()
    target = CorpusScreen()
    document = source.add_document("Connected", "one two three", "Manual")

    target.set_input_payload(source.current_output_payload())

    assert target._table.rowCount() == 1
    assert target._table.item(0, 0).text() == "Connected"
    assert target._table.item(0, 1).text() == "Manual"
    assert target.current_output_payload().value == (document,)


def test_create_corpus_empty_output_overrides_corpus_sample(app):
    source = CreateCorpusScreen()
    target = CorpusScreen()

    target.set_input_payload(source.current_output_payload())

    assert target._table.rowCount() == 0
    assert target.current_output_payload().value == ()
    assert "Input corpus is connected but empty" in target._status_label.text()


def test_main_window_can_route_create_corpus_output_to_corpus_widget(app):
    window = MainWindow()
    source_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    target_record = window._workspace.canvas.add_workflow_node("text-corpus")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(source_record.node_id, target_record.node_id)

    source_runtime = window._node_runtimes[source_record.node_id]
    target_runtime = window._node_runtimes[target_record.node_id]
    source_runtime.screen.add_document("Workflow Doc", "alpha beta gamma", "Manual")
    app.processEvents()

    assert isinstance(target_runtime.screen, CorpusScreen)
    assert target_runtime.screen._table.rowCount() == 1
    assert target_runtime.screen._table.item(0, 0).text() == "Workflow Doc"
    assert target_runtime.output_payload is not None
    assert target_runtime.output_payload.port_label == "Corpus"


def test_main_window_can_merge_create_corpus_and_import_documents_into_corpus_widget(app, tmp_path):
    import_path = tmp_path / "merged-import.txt"
    import_path.write_text("imported merge text", encoding="utf-8")
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    import_record = window._workspace.canvas.add_workflow_node("text-import-documents")
    target_record = window._workspace.canvas.add_workflow_node("text-corpus")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(create_record.node_id, target_record.node_id)
    assert scene.create_connection(import_record.node_id, target_record.node_id)

    create_runtime = window._node_runtimes[create_record.node_id]
    import_runtime = window._node_runtimes[import_record.node_id]
    target_runtime = window._node_runtimes[target_record.node_id]
    create_document = create_runtime.screen.add_document(
        "Manual Workflow Doc",
        "alpha beta gamma",
        "Create Corpus",
    )
    import_result = import_runtime.screen.import_paths((import_path,))
    app.processEvents()

    expected_documents = (create_document, *import_result.documents)

    assert isinstance(target_runtime.screen, CorpusScreen)
    assert target_runtime.screen.current_output_payload().value == expected_documents
    assert target_runtime.screen._table.rowCount() == len(expected_documents)
    assert target_runtime.screen._table.item(0, 0).text() == "Manual Workflow Doc"
    assert target_runtime.screen._table.item(1, 0).text() == "merged-import.txt"
    assert "Merged 2 input corpora" in target_runtime.screen._status_label.text()


def test_import_documents_output_can_update_connected_corpus_widget(app, tmp_path):
    path = tmp_path / "imported.txt"
    path.write_text("alpha beta gamma", encoding="utf-8")
    source = ImportDocumentsScreen()
    target = CorpusScreen()

    result = source.import_paths((path,))
    target.set_input_payload(source.current_output_payload())

    assert result.errors == ()
    assert source.current_output_payload().port_label == "Corpus"
    assert source.current_output_payload().value == result.documents
    assert target._table.rowCount() == 1
    assert target._table.item(0, 0).text() == "imported.txt"
    assert target._table.item(0, 2).text() == "alpha beta gamma"


def test_import_documents_empty_output_overrides_corpus_sample(app):
    source = ImportDocumentsScreen()
    target = CorpusScreen()

    target.set_input_payload(source.current_output_payload())

    assert target._table.rowCount() == 0
    assert target.current_output_payload().value == ()
    assert "Input corpus is connected but empty" in target._status_label.text()


def test_main_window_can_route_import_documents_output_to_corpus_widget(app, tmp_path):
    path = tmp_path / "workflow-import.txt"
    path.write_text("imported workflow text", encoding="utf-8")
    window = MainWindow()
    source_record = window._workspace.canvas.add_workflow_node("text-import-documents")
    target_record = window._workspace.canvas.add_workflow_node("text-corpus")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(source_record.node_id, target_record.node_id)

    source_runtime = window._node_runtimes[source_record.node_id]
    target_runtime = window._node_runtimes[target_record.node_id]
    source_runtime.screen.import_paths((path,))
    app.processEvents()

    assert isinstance(target_runtime.screen, CorpusScreen)
    assert target_runtime.screen._table.rowCount() == 1
    assert target_runtime.screen._table.item(0, 0).text() == "workflow-import.txt"
    assert target_runtime.output_payload is not None
    assert target_runtime.output_payload.port_label == "Corpus"


def test_pubmed_output_overrides_corpus_sample_and_updates_preprocess_text(app):
    source = PubMedScreen(
        documents=(PubMedDocument("Clinical Trial", "This NEEDS Cleaning.", "PubMed", "123"),)
    )
    corpus = CorpusScreen()
    preprocess = PreprocessTextScreen()

    corpus.set_input_payload(source.current_output_payload())
    preprocess.set_input_payload(source.current_output_payload())

    assert corpus._table.rowCount() == 1
    assert corpus._table.item(0, 0).text() == "Clinical Trial"
    assert corpus._table.item(0, 1).text() == "PubMed PMID 123"
    assert "Input corpus is connected" in corpus._status_label.text()
    assert preprocess._table.rowCount() == 1
    assert preprocess._table.item(0, 0).text() == "Clinical Trial"
    assert preprocess._table.item(0, 2).text() == "this needs cleaning"


def test_all_text_source_outputs_override_corpus_sample(app, tmp_path):
    import_path = tmp_path / "import-source.txt"
    import_path.write_text("imported text", encoding="utf-8")
    source_screens = (
        CreateCorpusScreen(),
        ImportDocumentsScreen(),
        GuardianScreen(documents=(GuardianDocument("Guardian Source", "one two", "The Guardian"),)),
        NYTimesScreen(documents=(NYTimesDocument("NYT Source", "three four", "NY Times"),)),
        PubMedScreen(documents=(PubMedDocument("PubMed Source", "five six", "PubMed", "123"),)),
        TwitterScreen(documents=(TwitterDocument("42", "seven eight", "Twitter/X", "user"),)),
        WikipediaScreen(documents=(CorpusDocument("Wiki Source", "nine ten", "Wikipedia"),)),
    )
    source_screens[0].add_document("Create Source", "created text", "Manual")
    source_screens[1].import_paths((import_path,))
    expected_titles = (
        "Create Source",
        "import-source.txt",
        "Guardian Source",
        "NYT Source",
        "PubMed Source",
        "Post 42",
        "Wiki Source",
    )

    for source, expected_title in zip(source_screens, expected_titles):
        corpus = CorpusScreen()

        corpus.set_input_payload(source.current_output_payload())

        assert corpus._table.rowCount() == 1
        assert corpus._table.item(0, 0).text() == expected_title
        assert corpus._table.item(0, 0).text() != SAMPLE_CORPUS[0].title
        assert "Input corpus is connected" in corpus._status_label.text()


def test_main_window_can_route_import_documents_output_to_corpus_and_preprocess_text(app, tmp_path):
    path = tmp_path / "preprocess-source.txt"
    path.write_text("This NEEDS Cleaning.", encoding="utf-8")
    window = MainWindow()
    import_record = window._workspace.canvas.add_workflow_node("text-import-documents")
    corpus_record = window._workspace.canvas.add_workflow_node("text-corpus")
    preprocess_record = window._workspace.canvas.add_workflow_node("text-preprocess")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(import_record.node_id, corpus_record.node_id)
    assert scene.create_connection(import_record.node_id, preprocess_record.node_id)

    import_runtime = window._node_runtimes[import_record.node_id]
    corpus_runtime = window._node_runtimes[corpus_record.node_id]
    preprocess_runtime = window._node_runtimes[preprocess_record.node_id]
    import_runtime.screen.import_paths((path,))
    app.processEvents()

    assert isinstance(corpus_runtime.screen, CorpusScreen)
    assert isinstance(preprocess_runtime.screen, PreprocessTextScreen)
    assert corpus_runtime.screen._table.rowCount() == 1
    assert corpus_runtime.screen._table.item(0, 0).text() == "preprocess-source.txt"
    assert corpus_runtime.screen._table.item(0, 0).text() != SAMPLE_CORPUS[0].title
    assert preprocess_runtime.screen._table.rowCount() == 1
    assert preprocess_runtime.screen._table.item(0, 0).text() == "preprocess-source.txt"
    assert preprocess_runtime.screen._table.item(0, 2).text() == "this needs cleaning"


def test_corpus_output_can_update_connected_preprocess_text_widget(app):
    source = CorpusScreen()
    target = PreprocessTextScreen()
    documents = (CorpusDocument("Needs Cleaning", "Hello, WORLD!", "Manual"),)

    source.set_input_payload(WorkflowPayload("Corpus", documents))
    target.set_input_payload(source.current_output_payload())

    assert target._table.rowCount() == 1
    assert target._table.item(0, 0).text() == "Needs Cleaning"
    assert target._table.item(0, 1).text() == "Hello, WORLD!"
    assert target._table.item(0, 2).text() == "hello world"
    assert "Input corpus is connected" in target._status_label.text()

    payload = target.current_output_payload()
    assert payload.port_label == "Corpus"
    assert payload.value == (CorpusDocument("Needs Cleaning", "hello world", "Manual"),)

    target.set_input_payload(None)

    assert target._table.rowCount() == 0
    assert target.current_output_payload().value == ()
    assert "Connect a Corpus input" in target._status_label.text()


def test_main_window_can_route_corpus_output_to_preprocess_text_widget(app):
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    corpus_record = window._workspace.canvas.add_workflow_node("text-corpus")
    preprocess_record = window._workspace.canvas.add_workflow_node("text-preprocess")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(create_record.node_id, corpus_record.node_id)
    assert scene.create_connection(corpus_record.node_id, preprocess_record.node_id)

    create_runtime = window._node_runtimes[create_record.node_id]
    corpus_runtime = window._node_runtimes[corpus_record.node_id]
    preprocess_runtime = window._node_runtimes[preprocess_record.node_id]
    create_runtime.screen.add_document("Workflow Text", "This, NEEDS Cleaning!", "Manual")
    app.processEvents()

    assert isinstance(corpus_runtime.screen, CorpusScreen)
    assert isinstance(preprocess_runtime.screen, PreprocessTextScreen)
    assert corpus_runtime.screen._table.rowCount() == 1
    assert preprocess_runtime.screen._table.rowCount() == 1
    assert preprocess_runtime.screen._table.item(0, 0).text() == "Workflow Text"
    assert preprocess_runtime.screen._table.item(0, 2).text() == "this needs cleaning"
    assert preprocess_runtime.output_payload is not None
    assert preprocess_runtime.output_payload.port_label == "Corpus"
    assert preprocess_runtime.output_payload.value == (
        CorpusDocument("Workflow Text", "this needs cleaning", "Manual"),
    )


def test_main_window_can_route_create_corpus_output_directly_to_preprocess_text_widget(app):
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    preprocess_record = window._workspace.canvas.add_workflow_node("text-preprocess")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(create_record.node_id, preprocess_record.node_id)

    create_runtime = window._node_runtimes[create_record.node_id]
    preprocess_runtime = window._node_runtimes[preprocess_record.node_id]
    create_runtime.screen.add_document("Direct Text", "This, NEEDS Cleaning!", "Manual")
    app.processEvents()

    assert isinstance(preprocess_runtime.screen, PreprocessTextScreen)
    assert preprocess_runtime.screen._table.rowCount() == 1
    assert preprocess_runtime.screen._table.item(0, 0).text() == "Direct Text"
    assert preprocess_runtime.screen._table.item(0, 2).text() == "this needs cleaning"


def test_bag_of_words_accepts_input_payload_and_resets_to_empty(app):
    screen = BagOfWordsScreen()
    documents = (CorpusDocument("Clean Text", "apple apple banana", "Manual"),)

    screen.set_input_payload(WorkflowPayload("Corpus", documents))

    assert screen._table.rowCount() == 2
    assert screen._table.item(0, 0).text() == "Clean Text"
    assert screen._table.horizontalHeaderItem(1).text() == "apple"
    assert screen._table.horizontalHeaderItem(2).text() == "banana"
    assert screen._table.item(0, 1).text() == "2"
    assert screen._table.item(0, 2).text() == "1"
    assert "Input corpus is connected" in screen._status_label.text()

    screen.set_input_payload(None)

    assert screen._table.rowCount() == 0
    assert "Connect a Corpus input" in screen._status_label.text()


def test_preprocess_text_output_can_update_connected_bag_of_words_widget(app):
    source = PreprocessTextScreen(
        documents=(CorpusDocument("Needs Cleaning", "Hello, WORLD! Hello.", "Manual"),)
    )
    target = BagOfWordsScreen(documents=())

    target.set_input_payload(source.current_output_payload())

    assert target._table.rowCount() == 2
    assert target._table.item(0, 0).text() == "Needs Cleaning"
    assert target._table.horizontalHeaderItem(1).text() == "hello"
    assert target._table.horizontalHeaderItem(2).text() == "world"
    assert target._table.item(0, 1).text() == "2"
    assert target._table.item(0, 2).text() == "1"
    assert "3 total tokens" in target.data_preview_snapshot()["summary"]


def test_main_window_can_route_create_corpus_output_directly_to_bag_of_words(app):
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    bow_record = window._workspace.canvas.add_workflow_node("text-bag-of-words")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(create_record.node_id, bow_record.node_id)

    create_runtime = window._node_runtimes[create_record.node_id]
    bow_runtime = window._node_runtimes[bow_record.node_id]
    create_runtime.screen.add_document("Direct Bag", "apple apple banana", "Manual")
    app.processEvents()

    assert isinstance(bow_runtime.screen, BagOfWordsScreen)
    assert bow_runtime.screen._table.rowCount() == 2
    assert bow_runtime.screen._table.item(0, 0).text() == "Direct Bag"
    assert bow_runtime.screen._table.horizontalHeaderItem(1).text() == "apple"
    assert bow_runtime.screen._table.horizontalHeaderItem(2).text() == "banana"
    assert bow_runtime.screen._table.item(0, 1).text() == "2"
    assert bow_runtime.screen._table.item(0, 2).text() == "1"


def test_main_window_can_route_preprocess_text_output_to_bag_of_words_widget(app):
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    corpus_record = window._workspace.canvas.add_workflow_node("text-corpus")
    preprocess_record = window._workspace.canvas.add_workflow_node("text-preprocess")
    bow_record = window._workspace.canvas.add_workflow_node("text-bag-of-words")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(create_record.node_id, corpus_record.node_id)
    assert scene.create_connection(corpus_record.node_id, preprocess_record.node_id)
    assert scene.create_connection(preprocess_record.node_id, bow_record.node_id)

    create_runtime = window._node_runtimes[create_record.node_id]
    preprocess_runtime = window._node_runtimes[preprocess_record.node_id]
    bow_runtime = window._node_runtimes[bow_record.node_id]
    create_runtime.screen.add_document("Workflow Text", "This, NEEDS Cleaning! This.", "Manual")
    app.processEvents()

    assert isinstance(preprocess_runtime.screen, PreprocessTextScreen)
    assert isinstance(bow_runtime.screen, BagOfWordsScreen)
    assert preprocess_runtime.screen._table.item(0, 2).text() == "this needs cleaning this"
    assert bow_runtime.screen._table.rowCount() == 2
    assert bow_runtime.screen._table.item(0, 0).text() == "Workflow Text"
    assert bow_runtime.screen._table.horizontalHeaderItem(1).text() == "cleaning"
    assert bow_runtime.screen._table.horizontalHeaderItem(2).text() == "needs"
    assert bow_runtime.screen._table.horizontalHeaderItem(3).text() == "this"
    assert bow_runtime.screen._table.item(0, 1).text() == "1"
    assert bow_runtime.screen._table.item(0, 2).text() == "1"
    assert bow_runtime.screen._table.item(0, 3).text() == "2"


def test_main_window_can_route_preprocess_text_output_to_person_b_widgets(app):
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    preprocess_record = window._workspace.canvas.add_workflow_node("text-preprocess")
    viewer_record = window._workspace.canvas.add_workflow_node("text-corpus-viewer")
    statistics_record = window._workspace.canvas.add_workflow_node("text-statistics")
    word_cloud_record = window._workspace.canvas.add_workflow_node("text-word-cloud")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(create_record.node_id, preprocess_record.node_id)
    assert scene.create_connection(preprocess_record.node_id, viewer_record.node_id)
    assert scene.create_connection(preprocess_record.node_id, statistics_record.node_id)
    assert scene.create_connection(preprocess_record.node_id, word_cloud_record.node_id)

    create_runtime = window._node_runtimes[create_record.node_id]
    viewer_runtime = window._node_runtimes[viewer_record.node_id]
    statistics_runtime = window._node_runtimes[statistics_record.node_id]
    word_cloud_runtime = window._node_runtimes[word_cloud_record.node_id]
    create_runtime.screen.add_document("Workflow Text", "Apple, apple! Banana.", "Manual")
    app.processEvents()

    assert isinstance(viewer_runtime.screen, CorpusViewerScreen)
    assert isinstance(statistics_runtime.screen, TextStatisticsScreen)
    assert isinstance(word_cloud_runtime.screen, WordCloudScreen)
    assert viewer_runtime.screen._table.item(0, 3).text() == "apple apple banana"
    assert statistics_runtime.screen._table.item(0, 0).text() == "apple"
    assert word_cloud_runtime.screen._table.item(0, 0).text() == "apple"


def test_workflow_connections_can_be_deleted_and_reconnected(app):
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    preprocess_record = window._workspace.canvas.add_workflow_node("text-preprocess")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(create_record.node_id, preprocess_record.node_id)
    assert scene.edge_count() == 1

    edge = scene._edges[0]
    scene.clearSelection()
    edge.setSelected(True)
    assert scene.delete_selected_items() is True
    assert scene.edge_count() == 0

    assert scene.create_connection(create_record.node_id, preprocess_record.node_id)
    assert scene.edge_count() == 1
    assert scene.delete_edge(scene._edges[0]) is True
    assert scene.edge_count() == 0
    assert scene.create_connection(create_record.node_id, preprocess_record.node_id)
    assert scene.edge_count() == 1


def test_main_window_can_route_statistics_output_to_corpus_widgets_and_data_table(app):
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    preprocess_record = window._workspace.canvas.add_workflow_node("text-preprocess")
    statistics_record = window._workspace.canvas.add_workflow_node("text-statistics")
    viewer_record = window._workspace.canvas.add_workflow_node("text-corpus-viewer")
    word_cloud_record = window._workspace.canvas.add_workflow_node("text-word-cloud")
    data_table_record = window._workspace.canvas.add_workflow_node("data-table")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(create_record.node_id, preprocess_record.node_id)
    assert scene.create_connection(preprocess_record.node_id, statistics_record.node_id)
    assert scene.create_connection(statistics_record.node_id, viewer_record.node_id)
    assert scene.create_connection(statistics_record.node_id, word_cloud_record.node_id)
    assert scene.create_connection(statistics_record.node_id, data_table_record.node_id)

    create_runtime = window._node_runtimes[create_record.node_id]
    statistics_runtime = window._node_runtimes[statistics_record.node_id]
    viewer_runtime = window._node_runtimes[viewer_record.node_id]
    word_cloud_runtime = window._node_runtimes[word_cloud_record.node_id]
    data_table_runtime = window._node_runtimes[data_table_record.node_id]
    create_runtime.screen.add_document("Market", "Profit, profit market!", "Manual")
    statistics_runtime.screen._contains_checkbox.setChecked(True)
    statistics_runtime.screen._contains_input.setText("profit")
    statistics_runtime.screen.apply_statistics()
    app.processEvents()

    assert isinstance(statistics_runtime.screen, TextStatisticsScreen)
    assert isinstance(viewer_runtime.screen, CorpusViewerScreen)
    assert isinstance(word_cloud_runtime.screen, WordCloudScreen)
    assert isinstance(data_table_runtime.screen, DataTableScreen)
    output_documents = statistics_runtime.screen.current_output_payload().value
    assert len(output_documents) == 1
    assert corpus_document_attributes(output_documents[0])["word_count"] == 3
    assert corpus_document_attributes(output_documents[0])["contains_profit"] == 2
    assert viewer_runtime.screen._table.rowCount() == 1
    assert word_cloud_runtime.screen._table.item(0, 0).text() == "profit"
    assert data_table_runtime.screen._headers[:3] == ["Document", "Source", "Text"]
    assert "word_count" in data_table_runtime.screen._headers
    assert "character_count" in data_table_runtime.screen._headers
    assert "contains_profit" in data_table_runtime.screen._headers
    assert len(data_table_runtime.screen._rows) == 1


def test_main_window_can_route_corpus_viewer_output_to_word_cloud(app):
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    viewer_record = window._workspace.canvas.add_workflow_node("text-corpus-viewer")
    word_cloud_record = window._workspace.canvas.add_workflow_node("text-word-cloud")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(create_record.node_id, viewer_record.node_id)
    assert scene.create_connection(viewer_record.node_id, word_cloud_record.node_id)

    create_runtime = window._node_runtimes[create_record.node_id]
    viewer_runtime = window._node_runtimes[viewer_record.node_id]
    word_cloud_runtime = window._node_runtimes[word_cloud_record.node_id]
    create_runtime.screen.add_document("Market", "profit market", "Manual")
    create_runtime.screen.add_document("Weather", "rain cloud", "Manual")
    viewer_runtime.screen._filter_input.setText("profit")
    app.processEvents()

    assert viewer_runtime.screen.current_output_payload().value[0].title == "Market"
    assert word_cloud_runtime.screen._table.rowCount() == 2
    assert word_cloud_runtime.screen._table.item(0, 0).text() in {"market", "profit"}


def test_main_window_can_route_word_cloud_word_counts_to_data_table(app):
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    word_cloud_record = window._workspace.canvas.add_workflow_node("text-word-cloud")
    data_table_record = window._workspace.canvas.add_workflow_node("data-table")
    scene = window._workspace.canvas.workflow_scene
    word_cloud_definition = window._widget_index["text-word-cloud"]
    data_table_definition = window._widget_index["data-table"]
    word_counts_output_port_id = next(
        port.id for port in word_cloud_definition.output_ports if port.label == "Word Counts"
    )
    data_input_port_id = data_table_definition.input_ports[0].id

    assert scene.create_connection(create_record.node_id, word_cloud_record.node_id)
    assert scene.create_connection(
        word_cloud_record.node_id,
        data_table_record.node_id,
        source_port_id=word_counts_output_port_id,
        target_port_id=data_input_port_id,
        channel="Word Counts",
    )

    create_runtime = window._node_runtimes[create_record.node_id]
    data_table_runtime = window._node_runtimes[data_table_record.node_id]
    create_runtime.screen.add_document("Market", "profit profit market", "Manual")
    app.processEvents()

    assert data_table_runtime.screen._headers == ["Word", "Frequency"]
    assert data_table_runtime.screen._rows[0] == ["profit", "2"]


def test_main_window_can_route_word_list_output_to_data_table(app, tmp_path):
    path = tmp_path / "word-list-source.txt"
    path.write_text("profit profit market", encoding="utf-8")
    window = MainWindow()
    import_record = window._workspace.canvas.add_workflow_node("text-import-documents")
    keywords_record = window._workspace.canvas.add_workflow_node("text-extract-keywords")
    word_list_record = window._workspace.canvas.add_workflow_node("text-word-list")
    data_table_record = window._workspace.canvas.add_workflow_node("data-table")
    scene = window._workspace.canvas.workflow_scene
    keywords_definition = window._widget_index["text-extract-keywords"]
    word_list_definition = window._widget_index["text-word-list"]
    data_table_definition = window._widget_index["data-table"]
    words_output_port_id = next(
        port.id for port in keywords_definition.output_ports if port.label == "Words"
    )
    words_input_port_id = next(
        port.id for port in word_list_definition.input_ports if port.label == "Words"
    )
    data_output_port_id = next(
        port.id for port in word_list_definition.output_ports if port.label == "Data"
    )
    data_input_port_id = data_table_definition.input_ports[0].id

    assert scene.create_connection(import_record.node_id, keywords_record.node_id)
    assert scene.create_connection(
        keywords_record.node_id,
        word_list_record.node_id,
        source_port_id=words_output_port_id,
        target_port_id=words_input_port_id,
    )
    assert scene.create_connection(
        word_list_record.node_id,
        data_table_record.node_id,
        source_port_id=data_output_port_id,
        target_port_id=data_input_port_id,
        channel="Data",
    )

    import_runtime = window._node_runtimes[import_record.node_id]
    keywords_runtime = window._node_runtimes[keywords_record.node_id]
    word_list_runtime = window._node_runtimes[word_list_record.node_id]
    data_table_runtime = window._node_runtimes[data_table_record.node_id]
    import_runtime.screen.import_paths((path,))
    app.processEvents()

    assert isinstance(keywords_runtime.screen, ExtractKeywordsScreen)
    assert isinstance(word_list_runtime.screen, WordListScreen)
    assert isinstance(data_table_runtime.screen, DataTableScreen)
    assert word_list_runtime.screen.current_output_payload().port_label == "Data"
    assert data_table_runtime.screen._headers == ["Word", "Frequency", "Documents", "Source"]
    assert data_table_runtime.screen._rows
    assert data_table_runtime.screen._rows[0][0] in {"profit", "market"}


def test_word_list_data_channel_to_data_table_persists_when_default_output_port_is_used(app):
    window = MainWindow()
    word_list_record = window._workspace.canvas.add_workflow_node("text-word-list")
    data_table_record = window._workspace.canvas.add_workflow_node("data-table")
    scene = window._workspace.canvas.workflow_scene
    word_list_definition = window._widget_index["text-word-list"]
    data_table_definition = window._widget_index["data-table"]
    words_output_port_id = next(
        port.id for port in word_list_definition.output_ports if port.label == "Words"
    )
    data_input_port_id = data_table_definition.input_ports[0].id

    assert tuple(port.label for port in data_table_definition.input_ports) == ("Data",)
    assert scene.create_connection(
        word_list_record.node_id,
        data_table_record.node_id,
        source_port_id=words_output_port_id,
        target_port_id=data_input_port_id,
        channel="Data",
    )

    word_list_runtime = window._node_runtimes[word_list_record.node_id]
    data_table_runtime = window._node_runtimes[data_table_record.node_id]
    word_list_runtime.screen.add_custom_word("profit dollar market shares")
    word_list_runtime.screen.set_update_mode("Ignore input")
    app.processEvents()

    snapshot = scene.snapshot()
    assert scene.edge_count() == 1
    assert snapshot["edges"][0]["channel"] == "Data"
    assert data_table_runtime.screen._headers == ["Word", "Frequency", "Documents", "Source"]
    assert {row[0] for row in data_table_runtime.screen._rows} == {"profit", "dollar", "market", "shares"}


def test_word_list_words_channel_can_feed_data_table_as_single_word_column(app):
    window = MainWindow()
    word_list_record = window._workspace.canvas.add_workflow_node("text-word-list")
    data_table_record = window._workspace.canvas.add_workflow_node("data-table")
    scene = window._workspace.canvas.workflow_scene
    word_list_definition = window._widget_index["text-word-list"]
    data_table_definition = window._widget_index["data-table"]
    words_output_port_id = next(
        port.id for port in word_list_definition.output_ports if port.label == "Words"
    )

    assert scene.create_connection(
        word_list_record.node_id,
        data_table_record.node_id,
        source_port_id=words_output_port_id,
        target_port_id=data_table_definition.input_ports[0].id,
        channel="Words",
    )

    word_list_runtime = window._node_runtimes[word_list_record.node_id]
    data_table_runtime = window._node_runtimes[data_table_record.node_id]
    word_list_runtime.screen.add_custom_word("profit dollar")
    word_list_runtime.screen.set_update_mode("Ignore input")
    app.processEvents()

    assert data_table_runtime.screen._headers == ["Word"]
    assert {row[0] for row in data_table_runtime.screen._rows} == {"profit", "dollar"}


def test_preprocess_text_output_connects_to_word_list_corpus_input(app):
    window = MainWindow()
    preprocess_record = window._workspace.canvas.add_workflow_node("text-preprocess")
    word_list_record = window._workspace.canvas.add_workflow_node("text-word-list")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(preprocess_record.node_id, word_list_record.node_id)
    snapshot = scene.snapshot()
    word_list_definition = window._widget_index["text-word-list"]
    corpus_input_port_id = next(
        port.id for port in word_list_definition.input_ports if port.label == "Corpus"
    )
    assert snapshot["edges"][0]["target_port_id"] == corpus_input_port_id


def test_import_documents_output_connects_to_word_list_corpus_input(app, tmp_path):
    path = tmp_path / "word-list-corpus.txt"
    path.write_text("profit profit market", encoding="utf-8")
    window = MainWindow()
    import_record = window._workspace.canvas.add_workflow_node("text-import-documents")
    word_list_record = window._workspace.canvas.add_workflow_node("text-word-list")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(import_record.node_id, word_list_record.node_id)

    import_runtime = window._node_runtimes[import_record.node_id]
    word_list_runtime = window._node_runtimes[word_list_record.node_id]
    import_runtime.screen.import_paths((path,))
    word_list_runtime.screen.add_custom_word("profit market")
    word_list_runtime.screen.set_update_mode("Ignore input")
    app.processEvents()

    assert word_list_runtime.screen._table.rowCount() == 2
    profit_row = next(
        row for row in range(word_list_runtime.screen._table.rowCount())
        if word_list_runtime.screen._table.item(row, 0).text() == "profit"
    )
    assert word_list_runtime.screen._table.item(profit_row, 1).text() == "2"
    assert word_list_runtime.screen._table.item(profit_row, 2).text() == "1"


def test_extract_keywords_output_connects_to_word_list_words_input_by_default(app):
    window = MainWindow()
    keywords_record = window._workspace.canvas.add_workflow_node("text-extract-keywords")
    word_list_record = window._workspace.canvas.add_workflow_node("text-word-list")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(keywords_record.node_id, word_list_record.node_id)
    snapshot = scene.snapshot()
    word_list_definition = window._widget_index["text-word-list"]
    words_input_port_id = next(
        port.id for port in word_list_definition.input_ports if port.label == "Words"
    )
    assert snapshot["edges"][0]["target_port_id"] == words_input_port_id


def test_word_list_outputs_do_not_connect_to_corpus_inputs(app):
    window = MainWindow()
    word_list_record = window._workspace.canvas.add_workflow_node("text-word-list")
    viewer_record = window._workspace.canvas.add_workflow_node("text-corpus-viewer")
    word_cloud_record = window._workspace.canvas.add_workflow_node("text-word-cloud")
    scene = window._workspace.canvas.workflow_scene
    word_list_definition = window._widget_index["text-word-list"]
    words_output_port_id = next(
        port.id for port in word_list_definition.output_ports if port.label == "Words"
    )
    data_output_port_id = next(
        port.id for port in word_list_definition.output_ports if port.label == "Data"
    )

    assert not scene.create_connection(
        word_list_record.node_id,
        viewer_record.node_id,
        source_port_id=words_output_port_id,
    )
    assert "provides Words" in window.state.status_message
    assert "expects Corpus" in window.state.status_message
    assert scene.create_connection(
        word_list_record.node_id,
        word_cloud_record.node_id,
        source_port_id=data_output_port_id,
        channel="Data",
    )


def test_data_table_output_connects_to_word_cloud_data_input(app):
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    data_table_record = window._workspace.canvas.add_workflow_node("data-table")
    word_cloud_record = window._workspace.canvas.add_workflow_node("text-word-cloud")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(create_record.node_id, data_table_record.node_id)
    assert scene.create_connection(data_table_record.node_id, word_cloud_record.node_id)

    create_runtime = window._node_runtimes[create_record.node_id]
    data_table_runtime = window._node_runtimes[data_table_record.node_id]
    word_cloud_runtime = window._node_runtimes[word_cloud_record.node_id]
    create_runtime.screen.add_document("Market", "profit profit market", "Manual")
    app.processEvents()

    snapshot = scene.snapshot()
    word_cloud_definition = window._widget_index["text-word-cloud"]
    data_input_port_id = next(
        port.id for port in word_cloud_definition.input_ports if port.label == "Data"
    )
    assert snapshot["edges"][1]["target_port_id"] == data_input_port_id
    assert word_cloud_runtime.screen._table.item(0, 0).text() == "profit"


def test_main_window_can_route_preprocess_text_output_to_advanced_text_widgets(app):
    window = MainWindow()
    create_record = window._workspace.canvas.add_workflow_node("text-create-corpus")
    preprocess_record = window._workspace.canvas.add_workflow_node("text-preprocess")
    keywords_record = window._workspace.canvas.add_workflow_node("text-extract-keywords")
    sentiment_record = window._workspace.canvas.add_workflow_node("text-sentiment-analysis")
    topics_record = window._workspace.canvas.add_workflow_node("text-topic-modelling")
    map_record = window._workspace.canvas.add_workflow_node("text-document-map")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(create_record.node_id, preprocess_record.node_id)
    assert scene.create_connection(preprocess_record.node_id, keywords_record.node_id)
    assert scene.create_connection(preprocess_record.node_id, sentiment_record.node_id)
    assert scene.create_connection(preprocess_record.node_id, topics_record.node_id)
    assert scene.create_connection(preprocess_record.node_id, map_record.node_id)

    create_runtime = window._node_runtimes[create_record.node_id]
    keywords_runtime = window._node_runtimes[keywords_record.node_id]
    sentiment_runtime = window._node_runtimes[sentiment_record.node_id]
    topics_runtime = window._node_runtimes[topics_record.node_id]
    map_runtime = window._node_runtimes[map_record.node_id]
    create_runtime.screen.add_document("Positive Fruit", "Good apple apple banana", "Manual")
    create_runtime.screen.add_document("Negative Tech", "Bad error data model", "Manual")
    app.processEvents()

    assert isinstance(keywords_runtime.screen, ExtractKeywordsScreen)
    assert isinstance(sentiment_runtime.screen, SentimentAnalysisScreen)
    assert isinstance(topics_runtime.screen, TopicModellingScreen)
    assert isinstance(map_runtime.screen, DocumentMapScreen)
    assert keywords_runtime.screen._table.rowCount() > 0
    assert sentiment_runtime.screen._table.rowCount() == 2
    assert topics_runtime.screen._documents_table.rowCount() == 2
    assert map_runtime.screen._table.rowCount() == 2


def test_main_window_can_route_import_documents_output_directly_to_bag_of_words(app, tmp_path):
    path = tmp_path / "bag-source.txt"
    path.write_text("apple apple banana", encoding="utf-8")
    window = MainWindow()
    import_record = window._workspace.canvas.add_workflow_node("text-import-documents")
    bow_record = window._workspace.canvas.add_workflow_node("text-bag-of-words")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(import_record.node_id, bow_record.node_id)

    import_runtime = window._node_runtimes[import_record.node_id]
    bow_runtime = window._node_runtimes[bow_record.node_id]
    import_runtime.screen.import_paths((path,))
    app.processEvents()

    assert isinstance(bow_runtime.screen, BagOfWordsScreen)
    assert bow_runtime.screen._table.rowCount() == 2
    assert bow_runtime.screen._table.item(0, 0).text() == "bag-source.txt"
    assert bow_runtime.screen._table.horizontalHeaderItem(1).text() == "apple"
    assert bow_runtime.screen._table.horizontalHeaderItem(2).text() == "banana"
    assert bow_runtime.screen._table.item(0, 1).text() == "2"
    assert bow_runtime.screen._table.item(0, 2).text() == "1"


def test_main_window_can_route_import_documents_output_to_preprocess_text_and_bag_of_words(
    app,
    tmp_path,
):
    path = tmp_path / "workflow-source.txt"
    path.write_text("This, NEEDS Cleaning! This.", encoding="utf-8")
    window = MainWindow()
    import_record = window._workspace.canvas.add_workflow_node("text-import-documents")
    preprocess_record = window._workspace.canvas.add_workflow_node("text-preprocess")
    bow_record = window._workspace.canvas.add_workflow_node("text-bag-of-words")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(import_record.node_id, preprocess_record.node_id)
    assert scene.create_connection(preprocess_record.node_id, bow_record.node_id)

    import_runtime = window._node_runtimes[import_record.node_id]
    preprocess_runtime = window._node_runtimes[preprocess_record.node_id]
    bow_runtime = window._node_runtimes[bow_record.node_id]
    import_runtime.screen.import_paths((path,))
    app.processEvents()

    assert isinstance(preprocess_runtime.screen, PreprocessTextScreen)
    assert isinstance(bow_runtime.screen, BagOfWordsScreen)
    assert preprocess_runtime.screen._table.rowCount() == 1
    assert preprocess_runtime.screen._table.item(0, 0).text() == "workflow-source.txt"
    assert preprocess_runtime.screen._table.item(0, 2).text() == "this needs cleaning this"
    assert bow_runtime.screen._table.rowCount() == 2
    assert bow_runtime.screen._table.item(0, 0).text() == "workflow-source.txt"
    assert bow_runtime.screen._table.horizontalHeaderItem(1).text() == "cleaning"
    assert bow_runtime.screen._table.horizontalHeaderItem(2).text() == "needs"
    assert bow_runtime.screen._table.horizontalHeaderItem(3).text() == "this"
    assert bow_runtime.screen._table.item(0, 1).text() == "1"
    assert bow_runtime.screen._table.item(0, 2).text() == "1"
    assert bow_runtime.screen._table.item(0, 3).text() == "2"


def test_create_corpus_title_and_source_inputs_use_readable_text_color(app):
    screen = CreateCorpusScreen()

    assert "color: #ffffff" in screen._title_input.styleSheet()
    assert "color: #ffffff" in screen._source_input.styleSheet()


def test_text_source_widget_inputs_use_readable_text_color(app):
    screens = (
        GuardianScreen(),
        NYTimesScreen(),
        PubMedScreen(),
        TwitterScreen(),
        WikipediaScreen(),
    )

    for screen in screens:
        assert "color: #ffffff" in screen._query_input.styleSheet()

    assert "color: #ffffff" in screens[0]._section_input.styleSheet()
    assert "color: #ffffff" in screens[0]._limit_spinbox.styleSheet()
    assert "color: #ffffff" in screens[1]._section_input.styleSheet()
    assert "color: #ffffff" in screens[1]._limit_spinbox.styleSheet()
    assert "color: #ffffff" in screens[2]._limit_spinbox.styleSheet()
    assert "color: #ffffff" in screens[3]._limit_spinbox.styleSheet()
    assert "color: #ffffff" in screens[4]._language_input.styleSheet()


def test_preprocess_text_lowercase_works():
    assert preprocess_text(
        "Hello WORLD",
        lowercase=True,
        remove_punctuation=False,
        remove_numbers=False,
        remove_stopwords=False,
        normalize_whitespace=False,
    ) == "hello world"


def test_preprocess_text_remove_punctuation_works():
    assert preprocess_text(
        "Hello, world!",
        lowercase=False,
        remove_punctuation=True,
        remove_numbers=False,
        remove_stopwords=False,
    ) == "Hello world"


def test_preprocess_text_remove_numbers_works():
    assert preprocess_text(
        "Version 2 has 15 tests",
        lowercase=False,
        remove_punctuation=False,
        remove_numbers=True,
        remove_stopwords=False,
    ) == "Version has tests"


def test_preprocess_text_remove_extra_whitespace_works():
    assert preprocess_text(
        "  alpha\n\n beta\tgamma  ",
        lowercase=False,
        remove_punctuation=False,
        remove_numbers=False,
        remove_stopwords=False,
        normalize_whitespace=True,
    ) == "alpha beta gamma"


def test_preprocess_text_stopword_removal_works():
    assert preprocess_text(
        "this is a useful document",
        lowercase=True,
        remove_punctuation=False,
        remove_numbers=False,
        remove_stopwords=True,
    ) == "useful document"


def test_preprocess_text_combined_options_work_together():
    assert preprocess_text(
        "This is 2024, A Test!",
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,
        remove_stopwords=True,
        normalize_whitespace=True,
    ) == "test"


def test_preprocess_text_strips_html_urls_and_accents():
    assert preprocess_text(
        "<p>Café Visit</p> https://example.com",
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=False,
        strip_html_tags=True,
        remove_urls_text=True,
        remove_accents_text=True,
    ) == "cafe visit"


def test_preprocess_tokenizers_whitespace_regex_and_tweet_like_modes():
    assert tokenize_text("hello, world", tokenizer=TOKENIZER_WHITESPACE) == ("hello,", "world")
    assert tokenize_text("alpha 123 beta", tokenizer=TOKENIZER_REGEX, regex_pattern=r"[a-z]+") == (
        "alpha",
        "beta",
    )
    assert tokenize_text("@user loves #Data!", tokenizer=TOKENIZER_TWEET) == (
        "@user",
        "loves",
        "#Data",
        "!",
    )


def test_preprocess_filter_tokens_supports_custom_stopwords_lengths_and_numeric_rules():
    tokens = ("a", "alpha", "beta2", "gamma", "custom")

    filtered = filter_tokens(
        tokens,
        remove_stopword_tokens=True,
        stopwords={"custom"},
        min_token_length=2,
        max_token_length=5,
        keep_alpha_only=True,
        remove_tokens_with_numbers=True,
    )

    assert filtered == ("alpha", "gamma")


def test_preprocess_custom_stopwords_are_parsed_and_applied():
    options = PreprocessOptions(
        remove_stopwords=True,
        custom_stopwords=parse_custom_stopwords("alpha, beta gamma"),
    )

    assert preprocess_tokens("alpha beta gamma delta", options) == ("delta",)


def test_preprocess_ngrams_build_unigrams_and_bigrams():
    assert build_ngrams(("data", "mining", "tool"), 1, 2) == (
        "data",
        "mining",
        "tool",
        "data_mining",
        "mining_tool",
    )
    assert preprocess_text(
        "Data mining tool",
        ngram_min=2,
        ngram_max=2,
    ) == "data_mining mining_tool"


def test_preprocess_text_handles_invalid_regex_safely():
    assert tokenize_text("alpha beta", tokenizer=TOKENIZER_REGEX, regex_pattern="[") == ()


def test_preprocess_text_empty_and_symbol_only_inputs_are_safe():
    assert preprocess_text("") == ""
    assert preprocess_text("!!!") == ""
    assert preprocess_text("123 456", remove_punctuation=False, remove_numbers=True) == ""


def test_preprocess_documents_and_summary_count_removed_words():
    documents = (
        CorpusDocument("First", "This is 2024, A Test!", "Manual"),
        CorpusDocument("Second", "Useful words remain", "Manual"),
    )
    options = PreprocessOptions(remove_numbers=True, remove_stopwords=True)

    processed = preprocess_documents(documents, options)
    summary = summarize_preprocessing(documents, processed)

    assert [document.text for document in processed] == ["test", "useful words remain"]
    assert summary.document_count == 2
    assert summary.total_original_words == 8
    assert summary.total_processed_words == 4
    assert summary.removed_word_count == 4
    assert summary.total_processed_tokens == 4
    assert summary.vocabulary_size == 4


def test_preprocess_summary_uses_consistent_token_counts_without_transforms():
    documents = (
        CorpusDocument("Punctuation", "hello, world! data-mining", "Manual"),
    )
    options = PreprocessOptions(
        lowercase=False,
        remove_punctuation=False,
        remove_numbers=False,
        remove_extra_whitespace=False,
    )

    processed = preprocess_documents(documents, options)
    summary = summarize_preprocessing(documents, processed, options)

    assert processed == (CorpusDocument("Punctuation", "hello world data mining", "Manual"),)
    assert summary.total_original_tokens == 4
    assert summary.total_processed_tokens == 4
    assert summary.removed_token_count == 0


def test_preprocess_screen_uses_lightweight_pipeline_options(app):
    screen = PreprocessTextScreen(
        documents=(CorpusDocument("HTML", "<b>This café</b> links https://example.com", "Manual"),)
    )
    screen._strip_html_checkbox.setChecked(True)
    screen._urls_checkbox.setChecked(True)
    screen._accents_checkbox.setChecked(True)
    screen._stopwords_checkbox.setChecked(True)
    screen._ngram_min_spinbox.setValue(1)
    screen._ngram_max_spinbox.setValue(2)

    processed = screen.apply_preprocessing()

    assert processed == (
        CorpusDocument("HTML", "cafe links cafe_links", "Manual"),
    )
    assert screen._table.item(0, 2).text() == "cafe links cafe_links"
    assert screen._table.item(0, 3).text() == "cafe, links, cafe_links"
    assert screen._processed_tokens_label.text().endswith("3")
    assert screen._vocabulary_label.text().endswith("3")
    assert "n-grams=1-2" in screen._status_label.text()


def test_empty_preprocessing_summary_is_safe():
    summary = summarize_preprocessing((), ())
    screen = PreprocessTextScreen(documents=())

    assert summary.document_count == 0
    assert summary.total_original_words == 0
    assert summary.total_processed_words == 0
    assert summary.removed_word_count == 0
    assert screen._table.rowCount() == 0
    assert screen.data_preview_snapshot()["rows"] == []


def test_bag_of_words_tokenization_lowercases_text():
    assert tokenize_for_bow("Hello WORLD") == ("hello", "world")


def test_bag_of_words_tokenization_removes_punctuation():
    assert tokenize_for_bow("Hello, world!") == ("hello", "world")


def test_bag_of_words_vocabulary_building_works():
    documents = (
        CorpusDocument("First", "apple banana apple"),
        CorpusDocument("Second", "banana carrot"),
    )

    assert build_vocabulary(documents) == ("apple", "banana", "carrot")


def test_bag_of_words_minimum_term_frequency_threshold_works():
    documents = (
        CorpusDocument("First", "apple banana apple"),
        CorpusDocument("Second", "banana carrot"),
    )

    assert build_vocabulary(documents, min_frequency=2) == ("apple", "banana")


def test_bag_of_words_document_term_matrix_counts_terms():
    documents = (
        CorpusDocument("First", "apple banana apple"),
        CorpusDocument("Second", "banana carrot"),
    )
    vocabulary = ("apple", "banana", "carrot")

    matrix = build_document_term_matrix(documents, vocabulary)

    assert matrix == ((2, 1, 0), (0, 1, 1))


def test_bag_of_words_binary_counts_option_works():
    documents = (
        CorpusDocument("First", "apple banana apple"),
        CorpusDocument("Second", "banana carrot"),
    )
    vocabulary = ("apple", "banana", "carrot")

    matrix = build_document_term_matrix(documents, vocabulary, binary=True)

    assert matrix == ((1, 1, 0), (0, 1, 1))


def test_bag_of_words_sublinear_weighting_works():
    weighted = apply_local_weighting((4, 1, 0), TF_SUBLINEAR)

    assert weighted[0] == pytest.approx(1.0 + 1.386294, rel=1e-5)
    assert weighted[1] == 1.0
    assert weighted[2] == 0.0


def test_bag_of_words_idf_and_smooth_idf_weighting_work():
    matrix = ((2.0, 1.0, 0.0), (0.0, 1.0, 1.0))

    idf = apply_global_weighting(matrix, DF_IDF)
    smooth = apply_global_weighting(matrix, DF_SMOOTH_IDF)

    assert idf[0][0] == pytest.approx(2.0 * 0.693147, rel=1e-5)
    assert idf[0][1] == pytest.approx(0.0)
    assert idf[1][2] == pytest.approx(0.693147, rel=1e-5)
    assert smooth[0][1] == pytest.approx(1.0 * 0.693147, rel=1e-5)


def test_bag_of_words_l1_and_l2_normalization_work():
    matrix = ((3.0, 4.0), (0.0, 0.0))

    l1 = normalize_matrix(matrix, NORM_L1)
    l2 = normalize_matrix(matrix, NORM_L2)

    assert l1[0] == pytest.approx((3 / 7, 4 / 7))
    assert l1[1] == (0.0, 0.0)
    assert l2[0] == pytest.approx((0.6, 0.8))
    assert l2[1] == (0.0, 0.0)


def test_bag_of_words_weighted_document_term_matrix_combines_options():
    documents = (
        CorpusDocument("First", "apple apple banana"),
        CorpusDocument("Second", "banana carrot"),
    )
    vocabulary = ("apple", "banana", "carrot")

    matrix = build_weighted_document_term_matrix(
        documents,
        vocabulary,
        local_weighting=TF_SUBLINEAR,
        global_weighting=DF_SMOOTH_IDF,
        normalization=NORM_L1,
    )

    assert len(matrix) == 2
    assert sum(matrix[0]) == pytest.approx(1.0)
    assert sum(matrix[1]) == pytest.approx(1.0)
    assert matrix[0][0] > matrix[0][1]


def test_bag_of_words_document_frequencies_and_formatting_work():
    matrix = ((2.0, 1.0, 0.0), (0.0, 1.0, 1.0))
    vocabulary = ("apple", "banana", "carrot")

    assert document_frequencies(matrix, vocabulary) == {"apple": 1, "banana": 2, "carrot": 1}
    assert format_matrix_value(1.0) == "1"
    assert format_matrix_value(0.333333) == "0.333"
    assert weighting_summary(TF_SUBLINEAR, DF_IDF, NORM_L2) == "TF=Sublinear, DF=IDF, Norm=L2"


def test_bag_of_words_term_frequency_summary_works():
    matrix = ((2, 1, 0), (1, 1, 1))
    vocabulary = ("apple", "banana", "carrot")

    assert term_frequencies(matrix, vocabulary) == {
        "apple": 3,
        "banana": 2,
        "carrot": 1,
    }


def test_bag_of_words_empty_and_symbol_only_documents_are_safe():
    documents = (
        CorpusDocument("Empty", ""),
        CorpusDocument("Symbols", "!!! ..."),
    )
    vocabulary = build_vocabulary(documents)
    matrix = build_document_term_matrix(documents, vocabulary)
    summary = summarize_bow(documents, vocabulary, matrix)

    assert vocabulary == ()
    assert matrix == ((), ())
    assert summary.document_count == 2
    assert summary.vocabulary_size == 0
    assert summary.total_token_count == 0
    assert summary.most_frequent_term == "None"


def test_bag_of_words_number_only_documents_are_safe():
    documents = (CorpusDocument("Numbers", "123 456 123"),)
    vocabulary = build_vocabulary(documents)
    matrix = build_document_term_matrix(documents, vocabulary)

    assert tokenize_for_bow(documents[0].text) == ("123", "456", "123")
    assert vocabulary == ("123", "456")
    assert matrix == ((2, 1),)


def test_bag_of_words_metadata_returns_expected_values():
    documents = (
        CorpusDocument("First", "apple apple banana"),
        CorpusDocument("Second", "banana carrot apple"),
    )
    vocabulary = build_vocabulary(documents)
    matrix = build_document_term_matrix(documents, vocabulary)

    summary = summarize_bow(documents, vocabulary, matrix)

    assert summary.document_count == 2
    assert summary.vocabulary_size == 3
    assert summary.total_token_count == 6
    assert summary.most_frequent_term == "apple"
    assert summary.nonzero_entries == 5
    assert summary.density == pytest.approx(5 / 6)


def test_bag_of_words_screen_renders_matrix_and_totals(app):
    documents = (
        CorpusDocument("First", "apple apple banana"),
        CorpusDocument("Second", "banana carrot apple"),
    )
    screen = BagOfWordsScreen(documents=documents)

    assert screen._table.rowCount() == 3
    assert screen._table.item(0, 0).text() == "First"
    assert screen._table.item(2, 0).text() == "Total Frequency"
    assert "3 terms" in screen.data_preview_snapshot()["summary"]


def test_bag_of_words_screen_uses_orange_style_weighting_options(app):
    documents = (
        CorpusDocument("First", "apple apple banana"),
        CorpusDocument("Second", "banana carrot"),
    )
    screen = BagOfWordsScreen(documents=documents)
    screen._term_frequency_combo.setCurrentIndex(2)
    screen._document_frequency_combo.setCurrentIndex(2)
    screen._normalization_combo.setCurrentIndex(1)

    matrix = screen.apply_options()

    assert len(matrix) == 2
    assert sum(matrix[0]) == pytest.approx(1.0)
    assert "TF=Sublinear, DF=Smooth IDF, Norm=L1" in screen._status_label.text()
    assert screen._density_label.text().endswith("66.7%")
    assert "." in screen._table.item(0, 1).text()


def test_wikipedia_url_builders_use_query_language_and_title():
    search_url = build_wikipedia_search_url("data mining", "tr", limit=3)
    summary_url = build_wikipedia_summary_url("Data mining", "en")

    assert search_url.startswith("https://tr.wikipedia.org/w/api.php?")
    assert "srsearch=data+mining" in search_url
    assert "srlimit=3" in search_url
    assert summary_url == "https://en.wikipedia.org/api/rest_v1/page/summary/Data_mining"


def test_wikipedia_parsers_handle_malformed_or_empty_payloads_safely():
    assert parse_wikipedia_search_response(None) == ()
    assert parse_wikipedia_search_response({"query": {"search": [{"snippet": "missing title"}]}}) == ()
    assert parse_wikipedia_summary_response(None) is None
    assert parse_wikipedia_summary_response({"title": "No extract"}) is None


def test_wikipedia_search_parser_cleans_snippets():
    results = parse_wikipedia_search_response(
        {"query": {"search": [{"title": "Text mining", "snippet": "<span>Text</span> &amp; mining"}]}}
    )

    assert len(results) == 1
    assert results[0].title == "Text mining"
    assert results[0].snippet == "Text & mining"


def test_wikipedia_summary_parser_returns_corpus_document():
    document = parse_wikipedia_summary_response(
        {
            "title": "Text mining",
            "extract": "Text mining extracts patterns from text.",
            "description": "analysis method",
        },
        "en",
    )

    assert document is not None
    assert document.title == "Text mining"
    assert document.source == "Wikipedia (en)"
    assert document.text == "Text mining extracts patterns from text. analysis method"
    assert count_words(document.text) == 8


def test_wikipedia_empty_query_uses_fallback_without_crashing():
    result = fetch_wikipedia_documents("", "en", fetch_json=lambda _url: pytest.fail("network should not be used"))

    assert result.used_fallback is True
    assert result.documents
    assert "sample" in result.documents[0].source.lower()


def test_wikipedia_fallback_documents_are_deterministic_and_non_empty():
    first = fallback_wikipedia_documents("portakal", "en")
    second = fallback_wikipedia_documents("portakal", "en")

    assert first == second
    assert len(first) == 3
    assert all(document.title and document.source and document.text for document in first)
    assert all(count_words(document.text) > 0 for document in first)


def test_wikipedia_fetch_uses_static_payloads_without_live_network():
    def fake_fetch_json(url: str):
        if "/w/api.php?" in url:
            return {"query": {"search": [{"title": "Text mining", "snippet": "fallback snippet"}]}}
        if url.endswith("/Text_mining"):
            return {"title": "Text mining", "extract": "Text mining finds patterns in documents."}
        return {}

    result = fetch_wikipedia_documents("text mining", "en", limit=1, fetch_json=fake_fetch_json)

    assert result.used_fallback is False
    assert len(result.documents) == 1
    assert result.documents[0].title == "Text mining"
    assert result.documents[0].source == "Wikipedia (en)"


def test_wikipedia_fetch_falls_back_on_network_or_empty_results():
    result = fetch_wikipedia_documents("missing", "en", fetch_json=lambda _url: {"query": {"search": []}})

    assert result.used_fallback is True
    assert result.documents


def test_wikipedia_document_summary_counts_words():
    documents = (
        CorpusDocument("First", "one two", "Wikipedia (en)"),
        CorpusDocument("Second", "three four five", "Wikipedia (en)"),
    )

    summary = summarize_documents(documents)

    assert summary.document_count == 2
    assert summary.total_word_count == 5
    assert summary.average_words_per_document == 2.5


def test_wikipedia_screen_empty_documents_are_safe(app):
    screen = WikipediaScreen(documents=())
    preview = screen.data_preview_snapshot()

    assert screen._table.rowCount() == 0
    assert preview["rows"] == []
    assert "0 documents" in preview["summary"]


def test_pubmed_url_builders_use_query_limit_and_pmids():
    search_url = build_pubmed_search_url("cancer therapy", limit=3)
    fetch_url = build_pubmed_fetch_url(("123", "456"))

    assert search_url.startswith("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?")
    assert "db=pubmed" in search_url
    assert "term=cancer+therapy" in search_url
    assert "retmax=3" in search_url
    assert fetch_url.startswith("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?")
    assert "id=123%2C456" in fetch_url


def test_pubmed_parsers_handle_malformed_or_empty_payloads_safely():
    assert parse_pubmed_search_response(None) == ()
    assert parse_pubmed_search_response({"esearchresult": {"idlist": None}}) == ()
    assert parse_pubmed_fetch_response(None) == ()
    assert parse_pubmed_fetch_response("<not xml") == ()


def test_pubmed_search_parser_returns_pmids():
    payload = {"esearchresult": {"idlist": ["123", "456", ""]}}

    assert parse_pubmed_search_response(payload) == ("123", "456")


def test_pubmed_fetch_parser_reads_static_xml_payload():
    xml_payload = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345</PMID>
          <Article>
            <ArticleTitle>Text mining in biomedical literature</ArticleTitle>
            <Abstract>
              <AbstractText>Biomedical text mining extracts signals from abstracts.</AbstractText>
              <AbstractText Label="Conclusion">The method supports literature review.</AbstractText>
            </Abstract>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    documents = parse_pubmed_fetch_response(xml_payload)

    assert len(documents) == 1
    assert documents[0].title == "Text mining in biomedical literature"
    assert documents[0].source == "PubMed"
    assert documents[0].pmid == "12345"
    assert "Biomedical text mining extracts signals from abstracts." in documents[0].text
    assert "Conclusion: The method supports literature review." in documents[0].text


def test_pubmed_fetch_parser_handles_missing_abstract():
    xml_payload = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>98765</PMID>
          <Article>
            <ArticleTitle>Record without abstract</ArticleTitle>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    documents = parse_pubmed_fetch_response(xml_payload)

    assert len(documents) == 1
    assert documents[0].pmid == "98765"
    assert documents[0].text == "No abstract available for this PubMed record."


def test_pubmed_empty_query_uses_fallback_without_crashing():
    result = fetch_pubmed_documents("", fetch_json=lambda _url: pytest.fail("network should not be used"))

    assert result.used_fallback is True
    assert result.documents
    assert result.documents[0].source == "PubMed"


def test_pubmed_fallback_documents_are_deterministic_and_non_empty():
    first = fallback_pubmed_documents("oncology")
    second = fallback_pubmed_documents("oncology")

    assert first == second
    assert len(first) == 3
    assert all(document.title and document.source and document.pmid and document.text for document in first)
    assert all(count_words(document.text) > 0 for document in first)


def test_pubmed_fetch_uses_static_payloads_without_live_network():
    xml_payload = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345</PMID>
          <Article>
            <ArticleTitle>Biomedical text mining</ArticleTitle>
            <Abstract>
              <AbstractText>Text mining supports biomedical literature analysis.</AbstractText>
            </Abstract>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """

    result = fetch_pubmed_documents(
        "text mining",
        limit=1,
        fetch_json=lambda _url: {"esearchresult": {"idlist": ["12345"]}},
        fetch_text=lambda _url: xml_payload,
    )

    assert result.used_fallback is False
    assert len(result.documents) == 1
    assert result.documents[0].title == "Biomedical text mining"
    assert result.documents[0].pmid == "12345"


def test_pubmed_fetch_falls_back_on_empty_results_or_parse_failures():
    result = fetch_pubmed_documents(
        "missing",
        fetch_json=lambda _url: {"esearchresult": {"idlist": []}},
        fetch_text=lambda _url: pytest.fail("fetch should not be called"),
    )

    assert result.used_fallback is True
    assert result.documents


def test_pubmed_document_summary_counts_words():
    documents = (
        PubMedDocument("First", "one two", "PubMed", "1"),
        PubMedDocument("Second", "three four five", "PubMed", "2"),
    )

    summary = summarize_pubmed_documents(documents)

    assert summary.document_count == 2
    assert summary.total_word_count == 5
    assert summary.average_words_per_document == 2.5


def test_pubmed_screen_empty_documents_are_safe(app):
    screen = PubMedScreen(documents=())
    preview = screen.data_preview_snapshot()

    assert screen._table.rowCount() == 0
    assert preview["rows"] == []
    assert "0 documents" in preview["summary"]


def test_guardian_url_builder_uses_query_limit_section_and_api_key():
    search_url = build_guardian_search_url(
        "climate change",
        "fake-key",
        limit=3,
        section="Technology",
    )

    assert search_url.startswith("https://content.guardianapis.com/search?")
    assert "q=climate+change" in search_url
    assert "page-size=3" in search_url
    assert "show-fields=trailText" in search_url
    assert "section=technology" in search_url
    assert "api-key=fake-key" in search_url


def test_guardian_parsers_handle_malformed_or_empty_payloads_safely():
    assert parse_guardian_response(None) == ()
    assert parse_guardian_response({"response": {"results": None}}) == ()
    assert parse_guardian_response({"response": {"results": [{"fields": {"trailText": "missing title"}}]}}) == ()


def test_guardian_parser_reads_static_json_payload():
    payload = {
        "response": {
            "results": [
                {
                    "webTitle": "Text mining helps newsrooms",
                    "sectionName": "Technology",
                    "webPublicationDate": "2026-05-01T12:00:00Z",
                    "fields": {
                        "trailText": "<p>Text mining &amp; reporting can summarize article collections.</p>"
                    },
                }
            ]
        }
    }

    documents = parse_guardian_response(payload)

    assert len(documents) == 1
    assert documents[0].title == "Text mining helps newsrooms"
    assert documents[0].source == "The Guardian"
    assert documents[0].section == "Technology"
    assert documents[0].publication_date == "2026-05-01T12:00:00Z"
    assert documents[0].text == "Text mining & reporting can summarize article collections."
    assert count_words(documents[0].text) == 8


def test_guardian_missing_api_key_uses_fallback_without_crashing(monkeypatch):
    monkeypatch.delenv("GUARDIAN_API_KEY", raising=False)
    result = fetch_guardian_documents(
        "climate",
        api_key="",
        fetch_json=lambda _url: pytest.fail("network should not be used"),
    )

    assert get_guardian_api_key() == ""
    assert result.used_fallback is True
    assert result.documents
    assert result.documents[0].source == "The Guardian"


def test_guardian_empty_query_uses_fallback_without_crashing():
    result = fetch_guardian_documents(
        "",
        api_key="fake-key",
        fetch_json=lambda _url: pytest.fail("network should not be used"),
    )

    assert result.used_fallback is True
    assert result.documents
    assert result.documents[0].source == "The Guardian"


def test_guardian_fallback_documents_are_deterministic_and_non_empty():
    first = fallback_guardian_documents("climate")
    second = fallback_guardian_documents("climate")

    assert first == second
    assert len(first) == 3
    assert all(
        document.title
        and document.source
        and document.section
        and document.publication_date
        and document.text
        for document in first
    )
    assert all(count_words(document.text) > 0 for document in first)


def test_guardian_fetch_uses_static_payload_without_live_network():
    requested_urls: list[str] = []

    def fake_fetch_json(url: str):
        requested_urls.append(url)
        return {
            "response": {
                "results": [
                    {
                        "webTitle": "Climate data in the news",
                        "sectionName": "Environment",
                        "webPublicationDate": "2026-05-02T08:30:00Z",
                        "fields": {"trailText": "Climate reporting uses datasets and text."},
                    }
                ]
            }
        }

    result = fetch_guardian_documents(
        "climate data",
        limit=1,
        section="environment",
        api_key="fake-key",
        fetch_json=fake_fetch_json,
    )

    assert result.used_fallback is False
    assert len(result.documents) == 1
    assert result.documents[0].title == "Climate data in the news"
    assert result.documents[0].section == "Environment"
    assert "section=environment" in requested_urls[0]


def test_guardian_fetch_falls_back_on_empty_results_or_parse_failures():
    result = fetch_guardian_documents(
        "missing",
        api_key="fake-key",
        fetch_json=lambda _url: {"response": {"results": []}},
    )

    assert result.used_fallback is True
    assert result.documents


def test_guardian_document_summary_counts_words():
    documents = (
        GuardianDocument("First", "one two", "The Guardian", "News", "2026-01-01"),
        GuardianDocument("Second", "three four five", "The Guardian", "News", "2026-01-02"),
    )

    summary = summarize_guardian_documents(documents)

    assert summary.document_count == 2
    assert summary.total_word_count == 5
    assert summary.average_words_per_document == 2.5


def test_guardian_screen_empty_documents_are_safe(app):
    screen = GuardianScreen(documents=())
    preview = screen.data_preview_snapshot()

    assert screen._table.rowCount() == 0
    assert preview["rows"] == []
    assert "0 documents" in preview["summary"]


def test_ny_times_url_builder_uses_query_section_and_api_key():
    search_url = build_ny_times_search_url(
        "climate change",
        "fake-key",
        limit=3,
        section="Technology",
    )

    assert search_url.startswith("https://api.nytimes.com/svc/search/v2/articlesearch.json?")
    assert "q=climate+change" in search_url
    assert "api-key=fake-key" in search_url
    assert "sort=newest" in search_url
    assert "page=0" in search_url
    assert "fq=" in search_url
    assert "Technology" in search_url


def test_ny_times_parsers_handle_malformed_or_empty_payloads_safely():
    assert parse_ny_times_response(None) == ()
    assert parse_ny_times_response({"response": {"docs": None}}) == ()
    assert parse_ny_times_response({"response": {"docs": [{"headline": {}}]}}) == ()


def test_ny_times_parser_reads_static_json_payload():
    payload = {
        "response": {
            "docs": [
                {
                    "headline": {"main": "Text mining helps news archives"},
                    "abstract": "Text mining &amp; reporting can summarize article collections.",
                    "snippet": "unused when abstract exists",
                    "lead_paragraph": "also unused",
                    "section_name": "Technology",
                    "news_desk": "Science",
                    "pub_date": "2026-05-03T12:00:00Z",
                    "web_url": "https://example.com/article",
                }
            ]
        }
    }

    documents = parse_ny_times_response(payload)

    assert len(documents) == 1
    assert documents[0].title == "Text mining helps news archives"
    assert documents[0].source == "NY Times"
    assert documents[0].section == "Technology"
    assert documents[0].publication_date == "2026-05-03T12:00:00Z"
    assert documents[0].url == "https://example.com/article"
    assert documents[0].text == "Text mining & reporting can summarize article collections."
    assert count_words(documents[0].text) == 8


def test_ny_times_missing_api_key_uses_fallback_without_crashing(monkeypatch):
    monkeypatch.delenv("NYTIMES_API_KEY", raising=False)
    result = fetch_ny_times_documents(
        "climate",
        api_key="",
        fetch_json=lambda _url: pytest.fail("network should not be used"),
    )

    assert get_ny_times_api_key() == ""
    assert result.used_fallback is True
    assert result.documents
    assert result.documents[0].source == "NY Times"


def test_ny_times_empty_query_uses_fallback_without_crashing():
    result = fetch_ny_times_documents(
        "",
        api_key="fake-key",
        fetch_json=lambda _url: pytest.fail("network should not be used"),
    )

    assert result.used_fallback is True
    assert result.documents
    assert result.documents[0].source == "NY Times"


def test_ny_times_fallback_documents_are_deterministic_and_non_empty():
    first = fallback_ny_times_documents("climate")
    second = fallback_ny_times_documents("climate")

    assert first == second
    assert len(first) == 3
    assert all(
        document.title
        and document.source
        and document.section
        and document.publication_date
        and document.url
        and document.text
        for document in first
    )
    assert all(count_words(document.text) > 0 for document in first)


def test_ny_times_fetch_uses_static_payload_without_live_network():
    requested_urls: list[str] = []

    def fake_fetch_json(url: str):
        requested_urls.append(url)
        return {
            "response": {
                "docs": [
                    {
                        "headline": {"main": "Climate data in the archive"},
                        "snippet": "Climate reporting uses datasets and text.",
                        "section_name": "Environment",
                        "pub_date": "2026-05-04T08:30:00Z",
                        "web_url": "https://example.com/climate",
                    }
                ]
            }
        }

    result = fetch_ny_times_documents(
        "climate data",
        limit=1,
        section="Environment",
        api_key="fake-key",
        fetch_json=fake_fetch_json,
    )

    assert result.used_fallback is False
    assert len(result.documents) == 1
    assert result.documents[0].title == "Climate data in the archive"
    assert result.documents[0].section == "Environment"
    assert "fq=" in requested_urls[0]
    assert "Environment" in requested_urls[0]


def test_ny_times_fetch_falls_back_on_empty_results_or_parse_failures():
    result = fetch_ny_times_documents(
        "missing",
        api_key="fake-key",
        fetch_json=lambda _url: {"response": {"docs": []}},
    )

    assert result.used_fallback is True
    assert result.documents


def test_ny_times_document_summary_counts_words():
    documents = (
        NYTimesDocument("First", "one two", "NY Times", "News", "2026-01-01"),
        NYTimesDocument("Second", "three four five", "NY Times", "News", "2026-01-02"),
    )

    summary = summarize_ny_times_documents(documents)

    assert summary.document_count == 2
    assert summary.total_word_count == 5
    assert summary.average_words_per_document == 2.5


def test_ny_times_screen_empty_documents_are_safe(app):
    screen = NYTimesScreen(documents=())
    preview = screen.data_preview_snapshot()

    assert screen._table.rowCount() == 0
    assert preview["rows"] == []
    assert "0 documents" in preview["summary"]


def test_twitter_url_builder_uses_query_and_limit():
    search_url = build_twitter_search_url("data mining", limit=12)

    assert search_url.startswith("https://api.twitter.com/2/tweets/search/recent?")
    assert "query=data+mining" in search_url
    assert "max_results=12" in search_url
    assert "tweet.fields=created_at%2Cauthor_id" in search_url


def test_twitter_parsers_handle_malformed_or_empty_payloads_safely():
    assert parse_twitter_response(None) == ()
    assert parse_twitter_response({"data": None}) == ()
    assert parse_twitter_response({"data": [{"id": "1"}]}) == ()


def test_twitter_parser_reads_static_json_payload():
    payload = {
        "data": [
            {
                "id": "1",
                "text": "Social text mining &amp; analysis uses short posts.",
                "created_at": "2026-05-24T10:00:00Z",
                "author_id": "user1",
            }
        ],
        "includes": {
            "users": [
                {
                    "id": "user1",
                    "username": "researcher",
                }
            ]
        },
    }

    documents = parse_twitter_response(payload)

    assert len(documents) == 1
    assert documents[0].post_id == "1"
    assert documents[0].author == "researcher"
    assert documents[0].source == "Twitter/X"
    assert documents[0].publication_date == "2026-05-24T10:00:00Z"
    assert documents[0].text == "Social text mining & analysis uses short posts."
    assert count_words(documents[0].text) == 8


def test_twitter_missing_bearer_token_uses_fallback_without_crashing(monkeypatch):
    monkeypatch.delenv("TWITTER_BEARER_TOKEN", raising=False)
    result = fetch_twitter_documents(
        "text mining",
        bearer_token="",
        fetch_json=lambda *_args: pytest.fail("network should not be used"),
    )

    assert get_twitter_bearer_token() == ""
    assert result.used_fallback is True
    assert result.documents
    assert result.documents[0].source == "Twitter/X"


def test_twitter_empty_query_uses_fallback_without_crashing():
    result = fetch_twitter_documents(
        "",
        bearer_token="fake-token",
        fetch_json=lambda *_args: pytest.fail("network should not be used"),
    )

    assert result.used_fallback is True
    assert result.documents
    assert result.documents[0].source == "Twitter/X"


def test_twitter_fallback_documents_are_deterministic_and_non_empty():
    first = fallback_twitter_documents("text mining")
    second = fallback_twitter_documents("text mining")

    assert first == second
    assert len(first) == 3
    assert all(
        document.post_id
        and document.source
        and document.author
        and document.publication_date
        and document.text
        for document in first
    )
    assert all(count_words(document.text) > 0 for document in first)


def test_twitter_fetch_uses_static_payload_without_live_network():
    requested: list[tuple[str, str]] = []

    def fake_fetch_json(url: str, bearer_token: str):
        requested.append((url, bearer_token))
        return {
            "data": [
                {
                    "id": "42",
                    "text": "Text mining can summarize social posts.",
                    "created_at": "2026-05-24T11:00:00Z",
                    "author_id": "user42",
                }
            ]
        }

    result = fetch_twitter_documents(
        "social text",
        limit=1,
        bearer_token="fake-token",
        fetch_json=fake_fetch_json,
    )

    assert result.used_fallback is False
    assert len(result.documents) == 1
    assert result.documents[0].post_id == "42"
    assert result.documents[0].author == "user42"
    assert requested[0][1] == "fake-token"
    assert "query=social+text" in requested[0][0]


def test_twitter_fetch_falls_back_on_empty_results_or_parse_failures():
    result = fetch_twitter_documents(
        "missing",
        bearer_token="fake-token",
        fetch_json=lambda *_args: {"data": []},
    )

    assert result.used_fallback is True
    assert result.documents


def test_twitter_document_summary_counts_words():
    documents = (
        TwitterDocument("1", "one two", "Twitter/X", "first", "2026-01-01"),
        TwitterDocument("2", "three four five", "Twitter/X", "second", "2026-01-02"),
    )

    summary = summarize_twitter_documents(documents)

    assert summary.document_count == 2
    assert summary.total_word_count == 5
    assert summary.average_words_per_document == 2.5


def test_twitter_screen_empty_documents_are_safe(app):
    screen = TwitterScreen(documents=())
    preview = screen.data_preview_snapshot()

    assert screen._table.rowCount() == 0
    assert preview["rows"] == []
    assert "0 documents" in preview["summary"]


def test_import_documents_supported_extensions_are_recognized():
    assert is_supported_document_path("notes.txt") is True
    assert is_supported_document_path("notes.md") is True
    assert is_supported_document_path("notes.csv") is True
    assert is_supported_document_path("notes.pdf") is False


def test_import_documents_rejects_unsupported_extensions_gracefully(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_text("not supported", encoding="utf-8")

    result = import_documents_from_paths((path,))

    assert result.documents == ()
    assert len(result.errors) == 1
    assert "Unsupported file type" in result.errors[0]


def test_import_documents_reports_read_errors_gracefully(tmp_path):
    path = tmp_path / "missing.txt"

    result = import_documents_from_paths((path,))

    assert result.documents == ()
    assert len(result.errors) == 1
    assert "Could not read missing.txt" in result.errors[0]


def test_import_documents_text_file_imports_title_preview_and_word_count(app, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha beta gamma", encoding="utf-8")
    screen = ImportDocumentsScreen()

    result = screen.import_paths((path,))

    assert len(result.documents) == 1
    assert result.documents[0].title == "sample.txt"
    assert result.documents[0].text == "alpha beta gamma"
    assert screen._table.item(0, 0).text() == "sample.txt"
    assert screen._table.item(0, 2).text() == "alpha beta gamma"
    assert screen._table.item(0, 3).text() == "3"


def test_import_documents_empty_path_list_is_safe():
    result = import_documents_from_paths(())

    assert result.documents == ()
    assert result.errors == ()


def test_import_documents_csv_file_creates_documents_from_rows(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("title,body\nFirst,hello world\nSecond,another row\n", encoding="utf-8")

    result = import_documents_from_paths((path,))

    assert result.errors == ()
    assert [document.title for document in result.documents] == [
        "rows.csv row 1",
        "rows.csv row 2",
        "rows.csv row 3",
    ]
    assert result.documents[1].text == "First hello world"


def test_import_documents_folder_imports_recursively_and_uses_subfolder_categories(tmp_path):
    root = tmp_path / "documents"
    sports = root / "sports"
    politics = root / "politics"
    sports.mkdir(parents=True)
    politics.mkdir(parents=True)
    (sports / "match.txt").write_text("team wins final", encoding="utf-8")
    (politics / "vote.md").write_text("election result update", encoding="utf-8")
    (root / "ignore.pdf").write_text("not imported", encoding="utf-8")

    result = import_documents_from_paths((root,))

    assert [document.title for document in result.documents] == ["vote.md", "match.txt"]
    assert [document.source for document in result.documents] == ["politics", "sports"]
    assert len(result.skipped) == 1
    assert result.skipped[0].extension == ".pdf"
    assert "Unsupported file type" in result.errors[0]


def test_import_documents_folder_import_can_disable_recursion(tmp_path):
    root = tmp_path / "documents"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "top.txt").write_text("top document", encoding="utf-8")
    (nested / "deep.txt").write_text("deep document", encoding="utf-8")

    result = import_documents_from_paths(
        (root,),
        options=DocumentImportOptions(recursive=False),
    )

    assert [document.title for document in result.documents] == ["top.txt"]


def test_import_documents_csv_column_mapping_uses_selected_columns(tmp_path):
    path = tmp_path / "mapped.csv"
    path.write_text(
        "title,category,text,date\n"
        "Review One,positive,this app is useful,2026-01-01\n"
        "Review Two,negative,the screen is confusing,2026-01-02\n",
        encoding="utf-8",
    )
    options = DocumentImportOptions(
        csv_mapping=CSVColumnMapping(
            title_column="title",
            text_column="text",
            source_column="category",
        )
    )

    result = import_documents_from_paths((path,), options=options)

    assert result.errors == ()
    assert [document.title for document in result.documents] == ["Review One", "Review Two"]
    assert [document.source for document in result.documents] == ["positive", "negative"]
    assert result.documents[0].text == "this app is useful"


def test_import_documents_csv_column_mapping_accepts_numeric_columns(tmp_path):
    path = tmp_path / "mapped.csv"
    path.write_text("title,body,category\nFirst,hello world,news\n", encoding="utf-8")
    options = DocumentImportOptions(
        csv_mapping=CSVColumnMapping(title_column="1", text_column="2", source_column="3")
    )

    result = import_documents_from_paths((path,), options=options)

    assert result.documents == (CorpusDocument("First", "hello world", "news"),)


def test_import_documents_screen_shows_skipped_documents_table(app, tmp_path):
    path = tmp_path / "report.pdf"
    path.write_text("not supported", encoding="utf-8")
    screen = ImportDocumentsScreen()

    result = screen.import_paths((path,))

    assert result.documents == ()
    assert screen._skipped_table.rowCount() == 1
    assert screen._skipped_table.item(0, 1).text() == ".pdf"
    assert "Unsupported file type" in screen._skipped_table.item(0, 2).text()
    assert "1 skipped" in screen._summary_label.text()


def test_import_documents_screen_preview_remove_clear_and_reload(app, tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first preview text", encoding="utf-8")
    second.write_text("second preview text", encoding="utf-8")
    screen = ImportDocumentsScreen()

    screen.import_paths((first, second))
    screen._table.selectRow(0)
    app.processEvents()

    assert "first preview text" in screen._preview_label.text()
    removed = screen.remove_selected_document()
    assert removed is not None
    assert removed.title == "first.txt"
    assert screen._table.rowCount() == 1

    screen.clear_imported_documents()
    assert screen._table.rowCount() == 0
    assert screen.current_output_payload().value == ()

    reloaded = screen.reload_last_import()
    assert reloaded is not None
    assert screen._table.rowCount() == 2
    assert [document.title for document in screen.current_output_payload().value] == [
        "first.txt",
        "second.txt",
    ]


def test_import_documents_encoding_selector_reads_cp1254_text(tmp_path):
    path = tmp_path / "turkish.txt"
    path.write_bytes("ğüşi".encode("cp1254"))
    options = DocumentImportOptions(encoding="cp1254")

    result = import_documents_from_paths((path,), options=options)

    assert result.errors == ()
    assert result.documents[0].text == "ğüşi"


def test_import_documents_duplicate_policy_skip_replace_and_allow(tmp_path):
    path = tmp_path / "duplicate.txt"
    path.write_text("first version", encoding="utf-8")
    existing = (CorpusDocument("duplicate.txt", "old version", str(path)),)

    skipped = import_documents_from_paths(
        (path,),
        options=DocumentImportOptions(append_to_existing=True, duplicate_policy=DUPLICATE_SKIP),
        existing_documents=existing,
    )
    replaced = import_documents_from_paths(
        (path,),
        options=DocumentImportOptions(append_to_existing=True, duplicate_policy=DUPLICATE_REPLACE),
        existing_documents=existing,
    )
    allowed = import_documents_from_paths(
        (path,),
        options=DocumentImportOptions(append_to_existing=True, duplicate_policy=DUPLICATE_ALLOW),
        existing_documents=existing,
    )

    assert skipped.documents == existing
    assert "Duplicate document" in skipped.errors[0]
    assert replaced.documents == (CorpusDocument("duplicate.txt", "first version", str(path)),)
    assert allowed.documents == (*existing, CorpusDocument("duplicate.txt", "first version", str(path)))
