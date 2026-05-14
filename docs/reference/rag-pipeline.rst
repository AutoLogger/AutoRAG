RAG pipeline
============

Document retrieval-augmented generation modules:

* :mod:`autorag.ingest` — document loading and chunking.
* :mod:`autorag.embed` — Ollama embedding wrapper.
* :mod:`autorag.store` — vector store façade.
* :mod:`autorag.retrieve` — similarity search.
* :mod:`autorag.generate` — LLM response generation.

.. toctree::
   :maxdepth: 1

   rag-pipeline/ingest
   rag-pipeline/embed
   rag-pipeline/store
   rag-pipeline/retrieve
   rag-pipeline/generate
