Types, schemas, config, formatting
==================================

Dependency-free utility modules safe to import from a base install:

* :mod:`autorag.types` — ``TypedDict`` shapes for transcripts and
  topic trees (``WordSpan``, ``TopicDict``, ``TopicTree``,
  ``TranscriptionResult``).
* :mod:`autorag.schemas` — Pydantic request/response models used by
  the HTTP API.
* :mod:`autorag.config` — Settings via ``pydantic-settings``.
* :mod:`autorag.blocks` — Time-bucketed, speaker-grouped transcript
  formatter (stdlib only).

Types (``autorag.types``)
-------------------------

.. automodule:: autorag.types
   :members:
   :show-inheritance:

Schemas (``autorag.schemas``)
-----------------------------

.. automodule:: autorag.schemas
   :members:
   :show-inheritance:

Config (``autorag.config``)
---------------------------

.. automodule:: autorag.config
   :members:
   :show-inheritance:

Block formatter (``autorag.blocks``)
------------------------------------

.. automodule:: autorag.blocks
   :members:
   :show-inheritance:
   :member-order: bysource
