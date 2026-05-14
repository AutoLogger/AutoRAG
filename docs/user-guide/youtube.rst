YouTube URL inputs
==================

Anywhere AutoRAG accepts an audio file path it also accepts a YouTube
URL, provided the ``[youtube]`` extra is installed:

.. code-block:: bash

    pip install "autorag[audio,diarize,youtube] @ git+https://github.com/AutoLogger/AutoRAG@v0.6.0"

Supported hosts (allowlisted in :func:`autorag.audio_source.is_youtube_url`):
``youtube.com``, ``www.youtube.com``, ``m.youtube.com``,
``music.youtube.com``, ``youtu.be``.

How it works
------------

:func:`autorag.audio_source.resolve_audio_input` is the context
manager that handles both local paths and URLs uniformly:

.. code-block:: python

    from autorag.audio_source import resolve_audio_input

    with resolve_audio_input("https://youtu.be/dQw4w9WgXcQ") as src:
        print(src.path, src.title, src.upload_date, src.duration_s)

For URLs, ``yt-dlp`` is invoked lazily and downloads the best audio
stream into a ``tempfile.TemporaryDirectory(prefix="autorag-yt-")``.
The download is cleaned up when the ``with`` block exits.

What gets propagated
--------------------

The CLI and :meth:`AutoRAG.transcribe
<autorag.core.AutoRAG.transcribe>` both wrap their work in
``resolve_audio_input``, and the CLI forwards four optional metadata
fields onto :meth:`persist_transcription
<autorag.core.AutoRAG.persist_transcription>`:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Effect on persistence
   * - ``source_url``
     - Becomes both the row's ``file_path`` and the seed for its
       stable session id. Survives the temp download being cleaned up.
   * - ``upload_date`` (``YYYYMMDD``)
     - Anchors ``created_at`` and the absolute event timestamps to
       midnight UTC of the publish date rather than the temp-file
       mtime.
   * - ``duration_s``
     - Currently informational; no schema column.
   * - ``title``
     - Used as the clip title if neither ``--title`` nor the
       fallback :func:`~autorag.audio_source.default_title_from` is
       used.

The CLI must own the temp lifetime itself because it calls both
``transcribe`` and ``persist_transcription`` on the same path. The
inner wrapper inside ``core.transcribe`` is a no-op pass-through for
an already-local ``Path``, so the double-wrap is safe.

Canonical URL form
------------------

Different YouTube URL shapes collapse to one canonical form for the
purpose of session-id derivation:

.. code-block:: text

    https://youtu.be/dQw4w9WgXcQ
    https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=10s
    https://www.youtube.com/watch?v=dQw4w9WgXcQ
        ↓
    https://www.youtube.com/watch?v=dQw4w9WgXcQ

Re-running ``autorag generate-topics`` on any of those overwrites the
same SQLite row.
