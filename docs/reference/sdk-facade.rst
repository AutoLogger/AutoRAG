SDK facade (``autorag.core``)
==============================

:mod:`autorag.core` exposes the single public class
:class:`~autorag.core.AutoRAG`. Every audio or RAG method performs its
heavy imports inside the method body and raises
:exc:`~autorag.errors.MissingExtraError` if the relevant extra is not
installed. See :doc:`../internals/extras-model` for the mapping of
methods to extras.

.. automodule:: autorag.core
   :members:
   :show-inheritance:
   :member-order: bysource

Errors
------

.. automodule:: autorag.errors
   :members:
   :show-inheritance:
