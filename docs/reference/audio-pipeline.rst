Audio pipeline
==============

Three modules sit behind :meth:`AutoRAG.transcribe
<autorag.core.AutoRAG.transcribe>`:

* :mod:`autorag.whisper_runner` — whisperX (faster-whisper +
  wav2vec2 forced-alignment) transcription with frame-accurate word
  timestamps and a CUDA→CPU fallback.
* :mod:`autorag.diarize` — pyannote 3.1 speaker diarization. Adds the
  ``speaker`` field on every :data:`~autorag.types.WordSpan`.
* :mod:`autorag.audio_source` — YouTube URL detection and a context
  manager that downloads remote audio to a temp file while exposing
  yt-dlp metadata.

.. toctree::
   :maxdepth: 1

   audio-pipeline/whisper_runner
   audio-pipeline/diarize
   audio-pipeline/audio_source
