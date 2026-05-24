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
    count_words,
    summarize_corpus,
)
from portakal_app.ui.screens.create_corpus_screen import (
    CreateCorpusScreen,
    make_document,
    preview_text,
)
from portakal_app.ui.screens.import_documents_screen import (
    ImportDocumentsScreen,
    import_documents_from_paths,
    is_supported_document_path,
)
from portakal_app.ui.screens.placeholder_screen import PlaceholderScreen
from portakal_app.ui.screens.preprocess_text_screen import (
    PreprocessOptions,
    PreprocessTextScreen,
    preprocess_documents,
    preprocess_text,
    summarize_preprocessing,
)


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
    assert screen._table.rowCount() == 5


def test_main_window_can_open_preprocess_text_widget(app):
    window = MainWindow()

    window._workspace.canvas.add_workflow_node("text-preprocess")
    window._show_widget("text-preprocess")

    assert isinstance(window._workspace.current_widget(), PreprocessTextScreen)


def test_other_text_mining_widgets_still_use_placeholder_screens(app):
    placeholder_count = 0
    for widget in build_widgets():
        if widget.category_id != "text-mining" or widget.id in {
            "text-corpus",
            "text-import-documents",
            "text-create-corpus",
            "text-preprocess",
        }:
            continue

        screen = widget.screen_factory()

        assert isinstance(screen, PlaceholderScreen)
        assert widget.enabled is True
        assert widget.description
        placeholder_count += 1

    assert placeholder_count == 6


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
