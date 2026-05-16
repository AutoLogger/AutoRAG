Transcription and topic extraction
==================================

AutoRAG's audio pipeline turns an audio file (or YouTube URL) into:

1. A list of timestamped :data:`~autorag.types.WordSpan` records via
   whisperX (faster-whisper + wav2vec2 forced alignment).
2. Speaker labels on every word via pyannote (when ``[diarize]`` is
   installed and ``HF_TOKEN`` is set).
3. A 3-level hierarchical :data:`~autorag.types.TopicTree` produced by
   an LLM in five focused passes.

Steps 1 + 2 live behind :meth:`AutoRAG.transcribe
<autorag.core.AutoRAG.transcribe>`. Step 3 is :meth:`AutoRAG.generate_topics
<autorag.core.AutoRAG.generate_topics>`.

Transcribe a local file
-----------------------

.. code-block:: python

    from autorag import AutoRAG

    rag = AutoRAG()
    words = rag.transcribe("meeting.wav", whisper_model="base", language="en")
    print(words[:3])
    # [{'w': ' Hello', 's': 0.0, 'e': 0.4, 'speaker': '0'}, …]

* ``whisper_model`` accepts the standard Whisper sizes (``tiny``,
  ``base``, ``small``, ``medium``, ``large``).
* ``language`` defaults to English (``"en"``); pass ``language=None``
  (SDK) or ``--language ""`` (CLI) to let Whisper auto-detect.
* Each :data:`~autorag.types.WordSpan` carries the word token, its
  start/end seconds, and the diarization-assigned speaker id
  (``"0"``-indexed in first-appearance order; always ``"0"`` when
  diarization is disabled).

The CTranslate2 model is unloaded after each call so the next run
starts from a clean VRAM budget; the wav2vec2 alignment model is
parked on CPU and re-uploaded on the next call.

Extract topics
--------------

.. code-block:: python

    topics = rag.generate_topics(words)
    print(topics["topics"][0]["title"])

Internally the agent issues five distinct LLM call sets — L1
boundaries, "should this L1 subdivide?", L2 boundaries, per-node
summarization, and an L0 aggregate — for roughly
``2 + N1_long + N1_yes + N1 + N2_total`` total calls. See
:doc:`../internals/audio-pipeline-design` for why the boundaries-vs-
summaries split is structured that way.

Persist
-------

The persistence layer requires the ``[rag]`` extra:

.. code-block:: python

    rag.persist_transcription("meeting.wav", words, title="Weekly sync")
    rag.persist_topics("meeting.wav", topics, words=words, title="Weekly sync")

Session ids are stable: a local path maps to the UUID-5 of its
resolved path, and a YouTube URL collapses to a canonical
``https://www.youtube.com/watch?v=<id>`` form. Re-running on the same
input overwrites the existing row instead of duplicating it.

Cached, dependency-free reads
-----------------------------

Once a clip is in SQLite, :meth:`AutoRAG.transcribe_blocks
<autorag.core.AutoRAG.transcribe_blocks>` can read it back without
loading Whisper or pyannote — only the ``[rag]`` extra is required for
the cache hit. ``[audio]``/``[diarize]`` (and ``[youtube]`` for URLs)
are imported lazily only when the cache misses.

.. code-block:: python

    blocks_text = rag.transcribe_blocks("meeting.wav", seconds=10)
    # 00:00-00:08 Speaker 1: Hello, welcome to the standup …

If you already have a :data:`~autorag.types.WordSpan` list in hand,
``autorag.blocks.format_blocks`` does the same formatting with no deps
at all.
