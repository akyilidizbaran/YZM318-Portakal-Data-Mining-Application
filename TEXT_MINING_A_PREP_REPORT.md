# Text Mining Person A Prep Report

## 1) Repository setup status

* Repository: `https://github.com/halilakbas11/YZM318-Portakal-Data-Mining-Application`
* Local path: `/Users/baran/Desktop/Portakal`
* Fork remote: `origin = https://github.com/akyilidizbaran/YZM318-Portakal-Data-Mining-Application.git`
* Original remote: `upstream = https://github.com/halilakbas11/YZM318-Portakal-Data-Mining-Application.git`
* Base branch used: `develop`
* Feature branch created: `feature/text-mining-a-widgets`
* Upstream sync: `git fetch upstream` and `git pull upstream develop` completed; branch was already up to date.
* Virtual environment: created with `python3 -m venv .venv` because `python` is not available on this machine.
* Installation command used: `.venv/bin/python -m pip install -e '.[dev]'`
* Installation result: succeeded.
* App startup check: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m portakal_app`
  * Result: app started and stayed alive for the bounded 5 second startup window.
  * The process was then terminated intentionally to avoid leaving the Qt event loop running.
  * No startup traceback or stderr output was observed.
* Tests: `.venv/bin/python -m pytest --collect-only -q` collected 358 tests successfully.
* Errors/notes:
  * `python --version` fails with `command not found`; use `.venv/bin/python` or `python3`.
  * The documented command did not require `PYTHONPATH=src` after editable install.

## 2) Project structure summary

Important project files and folders:

* `README.md`: development instructions for editable install and `python -m portakal_app`.
* `pyproject.toml`: package metadata, Python `>=3.11`, dependencies, dev extra with `pytest`, and pytest config.
* `requirements.txt`: alternate dependency list; includes the main runtime stack plus `python-slugify` and `tqdm`.
* `src/portakal_app/app.py`: creates the `QApplication`, applies theme, opens `MainWindow`.
* `src/portakal_app/__main__.py`: module entry point for `python -m portakal_app`.
* `src/portakal_app/models.py`: app state, `CategoryDefinition`, `WidgetDefinition`, `PortDefinition`, `WorkflowPayload`, and workflow port compatibility rules.
* `src/portakal_app/ui/catalog.py`: central category and widget registration via `build_categories()` and `build_widgets()`.
* `src/portakal_app/ui/main_window.py`: builds the main shell, indexes widgets by category, creates widget screens, and recomputes workflow runtime payloads.
* `src/portakal_app/ui/shell/sidebar.py`: left category list.
* `src/portakal_app/ui/shell/widget_catalog.py`: widget card/palette panel with search, click, double-click, and drag behavior.
* `src/portakal_app/ui/shell/workflow_canvas.py`: workflow scene, nodes, edges, drag/drop, connection compatibility, snapshot/restore.
* `src/portakal_app/ui/shell/workflow_workspace.py`: canvas host and floating widget dialogs.
* `src/portakal_app/ui/screens/`: most active widget UI screens.
* `src/portakal_app/widgets/unsupervised/`: newer unsupervised widget classes that are also registered through the same catalog.
* `src/portakal_app/data/models.py`: current tabular data model, centered on `DatasetHandle`.
* `src/portakal_app/data/services/`: data import, transform, profiling, model, and utility services.
* `tests/`: pytest suite for services, main window behavior, catalog registration, UI screens, and workflow interactions.

Where widgets/categories are defined:

* Categories are defined in `build_categories()` in `src/portakal_app/ui/catalog.py`.
* Widgets are defined as `WidgetDefinition(...)` entries in `build_widgets()` in `src/portakal_app/ui/catalog.py`.
* `WidgetDefinition` includes id, category id, label, enabled flag, screen factory, description, icon name, input ports, output ports, and optional Orange-style channel labels.

Where UI palette/sidebar logic lives:

* `SidebarCategoryList` in `src/portakal_app/ui/shell/sidebar.py` renders the category list.
* `WidgetCatalogPanel` and `WidgetCatalogButton` in `src/portakal_app/ui/shell/widget_catalog.py` render the widget palette.
* `MainWindow._on_category_selected()` feeds category widgets into the catalog panel.
* `WorkflowCanvas.dropEvent()` accepts `application/x-portakal-widget` drag data and creates workflow nodes.

Existing node/widget/workflow model:

* Static registration uses `CategoryDefinition`, `WidgetDefinition`, and `PortDefinition`.
* Runtime workflow nodes use `WorkflowNodeRecord`, `WorkflowNodeItem`, `WorkflowScene`, and `NodeRuntime`.
* Workflow payloads are passed as `WorkflowPayload(port_label, value)`.
* Current real data payloads are generally `DatasetHandle` instances.
* Port compatibility defaults to matching labels, with extra exceptions in `WORKFLOW_PORT_COMPATIBILITY_OVERRIDES`.

## 3) Person A official widget list

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

## 4) Proposed implementation plan

Add the Text Mining category:

* Add `CategoryDefinition(id="text-mining", label=i18n.t("Text Mining"))` in `build_categories()`.
* Safest placement is at the end of the category list, after `unsupervised`, because existing tests reference current category positions such as Data at index 0 and Transform at index 1.
* Add Turkish translation for `Text Mining` in `src/portakal_app/ui/i18n.py`.

Represent the widgets initially:

* Add ten `WidgetDefinition` entries in `build_widgets()` with category id `text-mining`.
* Use ids:
  * `corpus`
  * `import-documents`
  * `create-corpus`
  * `the-guardian`
  * `ny-times`
  * `pubmed`
  * `twitter`
  * `wikipedia`
  * `preprocess-text`
  * `bag-of-words`
* Reuse the existing `PlaceholderScreen` through `_placeholder_factory(...)` for the first placeholder phase.
* Keep placeholders enabled so users can add them to the workflow and open the popup.
* Do not emit fake output payloads from placeholders.
* Use planned ports, but keep behavior inert:
  * Corpus/source widgets: no input, output `Corpus`.
  * Preprocess Text: input `Corpus`, output `Corpus`.
  * Bag of Words: input `Corpus`, output `Data`.
* Avoid adding a heavy NLP dependency in the placeholder phase.
* Avoid changing `WORKFLOW_PORT_COMPATIBILITY_OVERRIDES` initially; matching `Corpus` labels are enough for corpus-to-corpus links, and Bag of Words can later bridge `Corpus` to existing `Data` consumers.

Suggested implementation order:

1. Category and catalog placeholders for all 10 Person A widgets.
2. Catalog/registration tests for category, exact widget ids, labels, enabled flags, and port labels.
3. Local/offline source widgets first: Corpus, Import Documents, Create Corpus.
4. Text pipeline widgets next: Preprocess Text, then Bag of Words.
5. External source widgets last: Wikipedia, PubMed, The Guardian, NY Times, Twitter.

Minimal placeholder behavior:

* Each widget opens a popup with a title and a short "not implemented yet" message.
* Source widgets show that they will eventually output `Corpus`.
* Preprocess Text shows that it will eventually transform an incoming `Corpus`.
* Bag of Words shows that it will eventually convert a `Corpus` into tabular `Data`.
* No file operations, network requests, credential prompts, NLP processing, or output payload generation yet.

Later real functionality likely needed:

* A corpus data model, likely a new lightweight `CorpusHandle` or similar structure rather than overloading `DatasetHandle`.
* Corpus import service for folders/files, with document text, metadata, labels, language, and source information.
* Create Corpus UI for manual document entry or constructing a corpus from existing tabular/text columns.
* API/source services:
  * The Guardian and NY Times: API keys, pagination, date filters, rate-limit handling.
  * PubMed: query builder, Entrez-style identifiers/metadata, retry behavior.
  * Twitter: authentication model and API access constraints must be clarified before implementation.
  * Wikipedia: article/category/search import; can likely use HTTP endpoints before adding dependencies.
* Preprocess Text service for lowercase, tokenization, stopwords, lemmatization/stemming, n-grams, filtering, and language-aware options.
* Bag of Words service for document-term matrices; existing `scikit-learn` may be enough for an initial `CountVectorizer`/`TfidfVectorizer` implementation.
* Serialization support for source settings and preprocessing options through `serialize_node_state()` / `restore_node_state()`.
* Output conversion from corpus features to existing `DatasetHandle` for downstream Data widgets.

## 5) Risk notes

* The app currently has a strong tabular-data assumption around `DatasetHandle`; Text Mining needs a separate corpus contract before real processing begins.
* Adding the new category anywhere except the end may break tests or UI assumptions that reference existing sidebar positions.
* `PlaceholderScreen` is available and safe, but it has a generic subtitle that says the module is left for another team; a later text-specific placeholder screen may be cleaner.
* There is an unused `_placeholder_factory(...)` helper in `catalog.py`; using it is safer than creating a new abstraction in the first phase.
* Existing screens mostly live in `src/portakal_app/ui/screens/`, while `src/portakal_app/widgets/unsupervised/` also exists. Text Mining should probably start in `ui/screens/` for UI consistency unless the team decides to move add-on widgets under `widgets/text_mining/`.
* `src/portakal_app/ui/screens/screens/` appears to duplicate some screen files. Do not use or reorganize that directory without a separate cleanup decision.
* External-source widgets can introduce API-key, rate-limit, terms-of-service, and network-test instability.
* Heavy NLP packages should be deferred; current dependencies already include `httpx`, `scikit-learn`, `numpy`, `polars`, and `scipy`.
* Icons can reuse existing standard icon names at first. Custom icon work should be a separate small pass.
* Turkish i18n should be updated for new visible labels to stay consistent with the existing localization approach.

## 6) Next-step checklist

Exact files to edit in the next implementation phase:

* `src/portakal_app/ui/catalog.py`
  * Add `text-mining` category.
  * Add the 10 placeholder `WidgetDefinition` entries.
* `src/portakal_app/ui/i18n.py`
  * Add `Text Mining` and the 10 widget label/description translations.
* `tests/test_text_mining_catalog.py`
  * New test file for category registration, exact Person A widget list, placeholder screen opening, and port definitions.
* `PROJECT_MEMORY.md`
  * Update after implementation decisions/milestones, but keep commits scoped as requested.

Implementation order for the next phase:

1. Add category at the end of `build_categories()`.
2. Add all ten placeholder widget definitions in `build_widgets()`.
3. Use existing `_placeholder_factory(...)` and `PlaceholderScreen`.
4. Add tests that preserve this exact official list:
   * Corpus
   * Import Documents
   * Create Corpus
   * The Guardian
   * NY Times
   * PubMed
   * Twitter
   * Wikipedia
   * Preprocess Text
   * Bag of Words
5. Run targeted tests:
   * `.venv/bin/python -m pytest tests/test_text_mining_catalog.py -q`
   * `.venv/bin/python -m pytest tests/test_main_window.py::test_sidebar_width_expands_for_longest_category_label -q`
6. Run broader collection or suite if time allows:
   * `.venv/bin/python -m pytest --collect-only -q`
   * `.venv/bin/python -m pytest -q`

Suggested commit plan:

* Current prep commit: `docs: prepare text mining person A widget plan`
* Next implementation commit: `feat: add text mining placeholder widgets`
* Later real-model commit: `feat: add corpus data model and local text sources`
* Later processing commit: `feat: implement text preprocessing and bag of words`
* Later external-source commit(s): one source family per commit, starting with the lowest-authentication source.
