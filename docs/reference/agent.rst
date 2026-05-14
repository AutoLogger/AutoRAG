Audio→topics agent (``autorag.agent``)
======================================

Multi-pass L0/L1/L2 topic extraction pipeline. Surfaces:

* :func:`~autorag.agent.transcribe_audio` — Whisper + diarization →
  :data:`~autorag.types.WordSpan` list.
* :func:`~autorag.agent.generate_topics` — pure-LLM topic extraction on
  a pre-computed transcript.
* :func:`~autorag.agent.build_topic_runnable` — the LangChain
  ``Runnable[list[WordSpan], TopicTree]`` used by ``generate_topics``.
* :func:`~autorag.agent.build_agent` — the combined Whisper +
  diarization + topics ``Runnable[Path | str, TranscriptionResult]``.

Most callers should go through :class:`~autorag.core.AutoRAG` rather
than importing this module directly. The pipeline design is documented
in :doc:`../internals/audio-pipeline-design`.

.. automodule:: autorag.agent
   :members:
   :show-inheritance:
   :member-order: bysource
   :exclude-members: TopicDict, TopicTree, TranscriptionResult, WordSpan
