Visualization
=============

The ``/viz`` page is backed by:

* :mod:`autorag.viz` — UMAP projection, ``/viz`` HTML route, and the
  JSON endpoints (``/viz/data``, ``/viz/search``) consumed by the
  React app.
* :mod:`autorag.topic_cluster` — KMeans clustering and similarity-edge
  construction used to colour and connect points in the projection.

The React/r3f frontend that consumes these endpoints is documented in
:doc:`../internals/frontend`.

.. toctree::
   :maxdepth: 1

   viz/viz
   viz/topic_cluster
