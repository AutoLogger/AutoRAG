Packaging and distribution
==========================

AutoRAG is distributed from GitHub, **not** PyPI. Consumers pin to a
git tag:

.. code-block:: bash

    pip install "autorag[all] @ git+https://github.com/AutoLogger/AutoRAG@v0.6.0"

Build backend: ``uv_build``. The package layout is the standard ``src/``
shape:

.. code-block:: text

    src/autorag/             # importable package
    src/autorag/static/viz/  # committed React bundle (shipped in the wheel)
    frontend/                # React source — not shipped

Releasing a new version
-----------------------

1. Bump ``__version__`` in ``src/autorag/__init__.py`` and ``version``
   in ``pyproject.toml``.
2. Run ``uv lock`` to refresh the lock file. Commit.
3. ``git tag v0.x.0 && git push --tags``.

Consumers then pin to the tag.

CI
--

``.github/workflows/ci.yml`` runs on every push and PR to ``main``.
Three parallel jobs:

* **Lint & Type Check** — ``ruff check``, ``ruff format --check``,
  ``mypy`` (installs ``--all-extras`` so mypy can see torch /
  chromadb / etc.).
* **Tests (all extras)** — ``pytest -v`` against the full dep stack.
* **SDK base install (no extras)** — ``uv sync --frozen --no-dev``,
  then asserts ``from autorag import AutoRAG`` boots and the SDK
  methods are callable. This is the regression guard for the
  lazy-import contract — if anyone re-introduces a ``chromadb`` /
  ``torch`` / ``whisper`` / ``pyannote`` / ``yt_dlp`` import at module
  top in ``core.py`` / ``embed.py`` / ``__init__.py`` / ``store.py``
  / ``audio_source.py``, this job fails.

The workflow uses ``uv sync --frozen`` (fails if ``uv.lock`` is out of
sync with ``pyproject.toml``). If you change dependencies, run
``uv lock`` locally before pushing.

Docs build
----------

The Sphinx build is local-only:

.. code-block:: bash

    uv sync --group docs
    uv run make -C docs strict        # treats warnings as errors

The ``[docs]`` extra (sphinx, furo, sphinx-autodoc-typehints,
myst-parser) lives in ``[dependency-groups]`` rather than
``[project.optional-dependencies]`` so it doesn't appear in published
wheels — docs aren't a runtime extra.

``docs/conf.py`` mirrors the runtime extras in
:data:`autodoc_mock_imports <docs.conf.autodoc_mock_imports>` so the
strict docs build works from a base install too. When you add a new
extras-gated import, add it to that list as well.

Third-party stubs
-----------------

These packages have no stubs — covered by mypy
``ignore_missing_imports`` overrides in ``pyproject.toml``:

* ``whisperx``, ``faster_whisper``, ``umap``, ``pydantic_sqlite``,
  ``imageio_ffmpeg``, ``chromadb``, ``pyannote``, ``yt_dlp``.

``langchain-ollama`` and ``langchain-core`` ship inline types — no
overrides needed. ``sklearn`` has no stubs and is suppressed with
``# type: ignore[import-untyped]`` at each import site.

Pylance / Pyright
-----------------

``.vscode/settings.json`` enables Pylance with
``typeCheckingMode: "strict"``. Pylance doesn't read mypy overrides,
so ``[tool.pyright]`` in ``pyproject.toml`` mirrors them:
``reportMissingTypeStubs = "none"`` for the unstubbed third-party set,
and ``reportPrivateUsage = "none"`` for the ``pydantic_sqlite._db``
access. If you add a new untyped third-party dep, add it to both the
mypy overrides and the pyright section.
