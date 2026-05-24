# Text Mining Person A Final Audit

Date: 2026-05-24
Branch: `feature/text-mining-a-widgets`
Latest implementation commit before this report: `d3a7117 feat: implement basic twitter widget`

## Summary

Person A's official Text Mining widget list is fully represented in the catalog and all 10 widgets now open real screens instead of `PlaceholderScreen`.

Official Person A widget list:

1. Corpus
2. Import Documents
3. Create Corpus
4. The Guardian
5. NY Times
6. PubMed
7. Twitter
8. Wikipedia
9. Preprocess Text
10. Bag of Words

## Files Involved

Catalog, labels, tests, and audit:

- `src/portakal_app/ui/catalog.py`
- `src/portakal_app/ui/i18n.py`
- `tests/test_text_mining_catalog.py`
- `TEXT_MINING_A_FINAL_AUDIT.md`

Implemented Text Mining screens:

- `src/portakal_app/ui/screens/corpus_screen.py`
- `src/portakal_app/ui/screens/import_documents_screen.py`
- `src/portakal_app/ui/screens/create_corpus_screen.py`
- `src/portakal_app/ui/screens/guardian_screen.py`
- `src/portakal_app/ui/screens/ny_times_screen.py`
- `src/portakal_app/ui/screens/pubmed_screen.py`
- `src/portakal_app/ui/screens/twitter_screen.py`
- `src/portakal_app/ui/screens/wikipedia_screen.py`
- `src/portakal_app/ui/screens/preprocess_text_screen.py`
- `src/portakal_app/ui/screens/bag_of_words_screen.py`

Text Mining SVG assets:

- `src/portakal_app/ui/assets/text_mining.svg`
- `src/portakal_app/ui/assets/text_corpus.svg`
- `src/portakal_app/ui/assets/text_import_documents.svg`
- `src/portakal_app/ui/assets/text_create_corpus.svg`
- `src/portakal_app/ui/assets/text_the_guardian.svg`
- `src/portakal_app/ui/assets/text_ny_times.svg`
- `src/portakal_app/ui/assets/text_pubmed.svg`
- `src/portakal_app/ui/assets/text_twitter.svg`
- `src/portakal_app/ui/assets/text_wikipedia.svg`
- `src/portakal_app/ui/assets/text_preprocess.svg`
- `src/portakal_app/ui/assets/text_bag_of_words.svg`

## Catalog And UI Audit

- `Text Mining` category exists with `text_mining` icon metadata.
- All 10 Person A widget IDs are registered in the expected order.
- Widget IDs are unique across the catalog.
- All 10 Person A widgets resolve to real screen classes.
- No Person A Text Mining widget resolves to `PlaceholderScreen`.
- All 10 widget icon names are non-empty and resolve to existing SVG files.
- The source widgets follow the same basic interaction pattern: query controls, optional limit or source-specific field, fetch action, status message, metadata panel, and results table.
- Corpus, Import Documents, Create Corpus, Preprocess Text, and Bag of Words use the same panel/table/metadata style already established for the Text Mining work.
- Empty document states are covered by tests for the screens that can be constructed with empty input.

## Fallback And Credential Audit

- Wikipedia uses live API helpers with deterministic fallback.
- PubMed uses NCBI E-utilities helpers with deterministic fallback.
- The Guardian uses `GUARDIAN_API_KEY` only when available and falls back without it.
- NY Times uses `NYTIMES_API_KEY` only when available and falls back without it.
- Twitter uses `TWITTER_BEARER_TOKEN` only when available and falls back without it.
- Tests do not require live internet or real API credentials.
- No token values are stored.
- No write actions, login automation, browser automation, or social posting behavior were added.
- No full article scraping or fragile HTML scraping was added.

## Asset Audit

- All 11 Text Mining SVG assets exist.
- The audit found local lightweight SVG assets only.
- No official Orange icon files were copied.
- No official Guardian, NY Times, PubMed, Twitter/X, or Wikipedia brand/logo assets were copied.
- Existing Text Mining icon mappings were preserved while each widget was implemented.

## Code Quality Notes

- No major feature changes were made during this audit.
- No unrelated widgets were modified.
- No heavy NLP dependencies were added.
- No hardcoded secrets were found.
- No debug `print`, `breakpoint`, or `pdb` usage was found in the Text Mining screen/test surface.
- The Text Mining test file is large because it covers registration, UI opening, helper parsing, fallback behavior, and metadata calculations in one focused file. It remains deterministic and passing. A future cleanup could split it by widget once the branch is stable.
- There is intentional lightweight duplication across source widgets to avoid broad refactoring before the PR.

## Test Results

Targeted Text Mining tests:

```text
.venv/bin/python -m pytest tests/test_text_mining_catalog.py -q
111 passed in 1.23s
```

App startup check:

```text
QT_QPA_PLATFORM=offscreen .venv/bin/python -m portakal_app
Passed. No traceback observed. The process was terminated after startup.
Qt warnings observed: missing "Sans Serif" alias cost and offscreen propagateSizeHints support.
```

Full test suite:

```text
.venv/bin/python -m pytest -q
464 passed, 5 failed in 13.73s
```

Known unrelated full-suite failures:

- `tests/test_cn2_rule_viewer_screen.py::test_cn2_induction_service_generates_classifier_for_categorical_target`
  - `CN2InductionSettings.__init__()` does not accept `max_rules`.
- `tests/test_pythagorean_forest_screen.py::test_main_window_handles_non_tabular_tree_output_preview`
  - Expected summary text differs from current non-tabular preview text.
- `tests/test_silhouette_plot_screen.py::test_silhouette_plot_dialog_footer_shows_save_button`
  - `WidgetScreenDialog` does not expose `_save_export_button`.
- `tests/test_transform_services.py::TestPythonScriptService::test_wrong_output_type`
  - Expected error text differs from current Python script service error text.
- `tests/test_venn_diagram_screen.py::test_venn_diagram_dialog_footer_shows_save_button`
  - `WidgetScreenDialog` does not expose `_save_export_button`.

These failures are outside the Person A Text Mining implementation path and were not fixed in this audit.

## Limitations

- No workflow data exchange has been implemented yet.
- No advanced NLP, stemming, lemmatization, TF-IDF, embeddings, or model training was added.
- No heavy external text-mining dependencies were added.
- Source widgets use fallback behavior when credentials are absent, network calls fail, responses are empty, or responses are malformed.
- Wikipedia, PubMed, The Guardian, NY Times, and Twitter screens are lightweight source demonstrations, not full production connectors.
- Full article bodies are not scraped or parsed.
- API keys and bearer tokens are read from environment variables only when needed and are not stored.

## PR Readiness

Assessment: Ready for a Person A pull request.

Rationale:

- The requested Person A scope is complete.
- All 10 official widgets are registered and implemented as real screens.
- The Text Mining category and all widget icons resolve correctly.
- Targeted Text Mining tests pass.
- App startup passes offscreen.
- Remaining full-suite failures are known unrelated failures outside Text Mining.
- `PROJECT_MEMORY.md` remains untracked and should not be included in the PR.

## Recommended Next Steps

For Person B:

- Define or implement the shared workflow payload/data contract for Corpus-like outputs.
- Connect source widgets and manual/import corpus widgets to downstream widgets once the workflow contract is agreed.
- Decide whether common document structures should be centralized after integration requirements are clear.

For Person C:

- Review UI polish across Text Mining screens after the data-flow contract exists.
- Add configuration guidance for optional API credentials without storing secrets.
- Consider splitting `tests/test_text_mining_catalog.py` into focused per-widget test files after the PR is merged or stabilized.
- Coordinate with existing unrelated failing tests before requiring full-suite green status for this branch.
