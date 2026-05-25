from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from portakal_app.app import create_application
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.catalog import build_categories, build_widgets
from portakal_app.ui.icons import get_widget_icon
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.shell.widget_catalog import WidgetCatalogButton
from portakal_app.ui.screens.bag_of_words_screen import (
    BagOfWordsScreen,
    build_document_term_matrix,
    build_vocabulary,
    summarize_bow,
    term_frequencies,
    tokenize_for_bow,
)
from portakal_app.ui.screens.corpus_screen import (
    CorpusDocument,
    CorpusScreen,
    SAMPLE_CORPUS,
    corpus_documents_from_payload,
    count_words,
    summarize_corpus,
)
from portakal_app.ui.screens.create_corpus_screen import (
    CreateCorpusScreen,
    make_document,
    preview_text,
)
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
    preprocess_documents,
    preprocess_text,
    summarize_preprocessing,
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


PERSON_A_TEXT_MINING_WIDGETS = [
    ("text-corpus", "Corpus", ("Corpus",), ("Corpus",)),
    ("text-import-documents", "Import Documents", (), ("Corpus",)),
    ("text-create-corpus", "Create Corpus", (), ("Corpus",)),
    ("text-the-guardian", "The Guardian", (), ("Corpus",)),
    ("text-ny-times", "NY Times", (), ("Corpus",)),
    ("text-pubmed", "PubMed", (), ("Corpus",)),
    ("text-twitter", "Twitter", (), ("Corpus",)),
    ("text-wikipedia", "Wikipedia", (), ("Corpus",)),
    ("text-preprocess", "Preprocess Text", ("Corpus",), ("Corpus",)),
    ("text-bag-of-words", "Bag of Words", ("Corpus",), ("Data",)),
]

PERSON_A_TEXT_MINING_ICON_NAMES = {
    "text-corpus": "text_corpus",
    "text-import-documents": "text_import_documents",
    "text-create-corpus": "text_create_corpus",
    "text-the-guardian": "text_the_guardian",
    "text-ny-times": "text_ny_times",
    "text-pubmed": "text_pubmed",
    "text-twitter": "text_twitter",
    "text-wikipedia": "text_wikipedia",
    "text-preprocess": "text_preprocess",
    "text-bag-of-words": "text_bag_of_words",
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
    assert len(cards) == 10
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


def test_text_mining_wikipedia_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-wikipedia"].screen_factory()

    assert isinstance(screen, WikipediaScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert widgets["text-wikipedia"].icon_name == "text_wikipedia"
    assert screen._table.rowCount() > 0


def test_main_window_can_open_wikipedia_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-wikipedia")
    window._show_widget("text-wikipedia")

    assert isinstance(window._workspace.current_widget(), WikipediaScreen)


def test_text_mining_pubmed_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-pubmed"].screen_factory()

    assert isinstance(screen, PubMedScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert widgets["text-pubmed"].icon_name == "text_pubmed"
    assert screen._table.rowCount() > 0


def test_main_window_can_open_pubmed_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-pubmed")
    window._show_widget("text-pubmed")

    assert isinstance(window._workspace.current_widget(), PubMedScreen)


def test_text_mining_guardian_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-the-guardian"].screen_factory()

    assert isinstance(screen, GuardianScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert widgets["text-the-guardian"].icon_name == "text_the_guardian"
    assert screen._table.rowCount() > 0


def test_main_window_can_open_guardian_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-the-guardian")
    window._show_widget("text-the-guardian")

    assert isinstance(window._workspace.current_widget(), GuardianScreen)


def test_text_mining_ny_times_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-ny-times"].screen_factory()

    assert isinstance(screen, NYTimesScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert widgets["text-ny-times"].icon_name == "text_ny_times"
    assert screen._table.rowCount() > 0


def test_main_window_can_open_ny_times_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-ny-times")
    window._show_widget("text-ny-times")

    assert isinstance(window._workspace.current_widget(), NYTimesScreen)


def test_text_mining_twitter_widget_uses_real_screen(app):
    widgets = {widget.id: widget for widget in build_widgets()}

    screen = widgets["text-twitter"].screen_factory()

    assert isinstance(screen, TwitterScreen)
    assert not isinstance(screen, PlaceholderScreen)
    assert widgets["text-twitter"].icon_name == "text_twitter"
    assert screen._table.rowCount() > 0


def test_main_window_can_open_twitter_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-twitter")
    window._show_widget("text-twitter")

    assert isinstance(window._workspace.current_widget(), TwitterScreen)


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


def test_corpus_summary_counts_documents_and_words():
    documents = (
        CorpusDocument("First", "one two three"),
        CorpusDocument("Second", "four five"),
    )

    summary = summarize_corpus(documents)

    assert summary.document_count == 2
    assert summary.total_word_count == 5
    assert summary.average_words_per_document == 2.5


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


def test_corpus_payload_parser_rejects_non_corpus_items():
    documents = (CorpusDocument("First", "one two"),)

    assert corpus_documents_from_payload(documents) == documents
    assert corpus_documents_from_payload("not a corpus") is None
    assert corpus_documents_from_payload((object(),)) is None


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
