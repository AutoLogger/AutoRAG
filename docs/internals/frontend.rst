Frontend (``/viz``)
===================

``/viz`` is AutoRAG's only browser surface. It's a Vite + React 18 +
TypeScript + ``@react-three/fiber`` app, served as a built static
bundle by FastAPI.

Layout
------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Path
     - What
   * - ``frontend/``
     - Source — TypeScript, **not** shipped to PyPI.
   * - ``frontend/index.html``
     - Vite entry.
   * - ``frontend/vite.config.ts``
     - ``base: '/viz-assets/'`` + ``outDir: ../src/autorag/static/viz``.
   * - ``frontend/src/main.tsx``
     - ``ReactDOM.createRoot``.
   * - ``frontend/src/App.tsx``
     - Root component.
   * - ``frontend/src/api/``
     - Hand-typed mirror of :mod:`autorag.viz` schemas + fetch
       wrappers.
   * - ``frontend/src/state/``
     - Zustand store for cross-component scene state.
   * - ``frontend/src/hooks/``
     - ``useVizData``, ``useDebouncedSearch``.
   * - ``frontend/src/three/``
     - r3f components — ``Scene``, ``PointsLayer``, etc.
   * - ``frontend/src/ui/``
     - DOM components — ``Rail``, ``Legend``, ``SearchBox``,
       ``Tooltip``, etc.
   * - ``src/autorag/static/viz/``
     - **Committed build output.** ``index.html`` + hashed
       ``assets/*``.

The TypeScript source lives outside ``src/autorag/`` so ``uv`` /
``ruff`` / ``mypy`` don't scan it. The build output lives **inside**
the Python package so wheel packaging picks it up via the existing
``static/`` glob — no ``MANIFEST.in`` changes needed.

Build flow
----------

.. code-block:: bash

    cd frontend && npm install && npm run build

``tsc -b && vite build`` runs the TypeScript project-references build
for typecheck, then emits ``index.html`` + hashed
``assets/index-<hash>.{js,css}`` into ``src/autorag/static/viz/``
(Vite's ``emptyOutDir: true`` clears stale hashes). Commit the rebuilt
bundle alongside any ``frontend/src/`` changes in the same commit so
HTML, source, and assets never drift.

For interactive iteration:

.. code-block:: bash

    cd frontend && npm run dev    # Vite on http://localhost:5173

The dev server proxies ``/viz/data`` and ``/viz/search`` to a
separately running ``autorag serve`` on port 8000 (see ``server.proxy``
in ``vite.config.ts``).

FastAPI wiring
--------------

* :mod:`autorag.viz` resolves ``_VIZ_DIR = static/viz/``, serves
  ``_VIZ_DIR / "index.html"`` at ``GET /viz``, and exports
  :data:`~autorag.viz.viz_assets_dir` for the static mount.
* :mod:`autorag.api` mounts the assets dir at ``/viz-assets`` **inside**
  the existing ``[rag]`` ``try/else``, so ``[server]``-only installs
  (without ``[rag]``) silently skip both the viz endpoints and the
  assets mount.
* ``base: '/viz-assets/'`` in ``vite.config.ts`` is load-bearing — it
  makes built asset URLs (``<script
  src="/viz-assets/assets/index-<hash>.js">``) match the mount.

Built bundle is committed
-------------------------

CI does not run a node build. Rationale:

1. Python-only CI keeps passing with zero new infra.
2. PyPI / git-installed wheels need the built assets anyway — they
   ship via the existing ``static/`` glob.
3. The viz changes infrequently relative to the Python backend.

If a CI build is wanted later: add one GitHub Actions job with
``setup-node@v4`` running ``npm ci && npm run build`` in
``frontend/``. Additive.

Three.js and ``@types/three`` are pinned **exactly** (no ``^``) —
``drei`` 9.x must move in lockstep with any Three bump. Current:
``three@0.165.0``.
