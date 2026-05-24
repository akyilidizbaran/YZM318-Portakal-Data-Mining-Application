from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from portakal_app.app import create_application
from portakal_app.ui import i18n
from portakal_app.ui.catalog import build_categories, build_widgets
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.corpus_screen import (
    CorpusDocument,
    CorpusScreen,
    summarize_corpus,
)
from portakal_app.ui.screens.placeholder_screen import PlaceholderScreen


PERSON_A_TEXT_MINING_WIDGETS = [
    ("text-corpus", "Corpus", (), ("Corpus",)),
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


def test_other_text_mining_widgets_still_use_placeholder_screens(app):
    for widget in build_widgets():
        if widget.category_id != "text-mining" or widget.id == "text-corpus":
            continue

        screen = widget.screen_factory()

        assert isinstance(screen, PlaceholderScreen)
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
