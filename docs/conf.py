"""Sphinx configuration for AutoRAG documentation.

The autodoc_mock_imports list mirrors the lazy-import contract described
in CLAUDE.md: the base install only carries typer + pydantic + langchain;
everything heavy (torch, whisperx, chromadb, pyannote, etc.) is gated
behind an extra and imported inside method bodies. Without mocking,
``sphinx-build`` from a base install would fail when autodoc tries to
import modules whose top-level dependencies are not present.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

# Make `autorag` importable for autodoc without requiring an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

project = "AutoRAG"
author = "Kalen Cantrell"
copyright = "2026, Kalen Cantrell"

try:
    release = importlib.metadata.version("autorag")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Allow .. include:: ../README.md / ../CHANGELOG.md through myst.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# --- autodoc -------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
}

# Mock heavy / extras-gated deps so autodoc can import every module from a
# base install. Must stay in sync with [project.optional-dependencies] in
# pyproject.toml and the lazy-import rule in CLAUDE.md.
autodoc_mock_imports = [
    "chromadb",
    "torch",
    "torchaudio",
    "whisperx",
    "faster_whisper",
    "pyannote",
    "pyannote.audio",
    "yt_dlp",
    "umap",
    "sklearn",
    "pydantic_sqlite",
    "fastapi",
    "uvicorn",
    "imageio_ffmpeg",
    "huggingface_hub",
    # Pulled transitively by langchain_core (a *base* dep) via a guarded
    # ``from transformers import GPT2TokenizerFast``. In base+docs CI
    # transformers is absent and that import is swallowed; in an all-extras
    # env the real transformers runs ``os.path.join(constants.HF_HOME, ...)``
    # where ``constants`` is the mocked ``huggingface_hub``, which raises.
    # Mocking transformers makes both environments take the same path.
    "transformers",
    "numpy",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# sphinx-autodoc-typehints inspects Pydantic-internal signatures (Field,
# JsonValue) when documenting subclasses of BaseModel; those references
# can't be resolved from our environment. Squelch the noise without
# hiding genuine forward-reference errors in our own code.
always_document_param_types = True
typehints_fully_qualified = False
suppress_warnings = [
    "sphinx_autodoc_typehints.forward_reference",
]

# --- intersphinx ---------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# --- HTML output ---------------------------------------------------------

html_theme = "furo"
html_title = f"AutoRAG {release}"
html_static_path: list[str] = []

# --- myst-parser ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
