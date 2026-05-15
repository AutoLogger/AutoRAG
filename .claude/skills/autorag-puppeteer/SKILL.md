---
name: autorag-puppeteer
description: Drive, debug, and screenshot the AutoRAG FastAPI app via the puppeteer MCP server. Covers server lifecycle (`autorag serve`), the `/viz` WebGL page, the `/health` / `/ingest` / `/query` API, the Swagger `/docs` UI, and patterns for graduating an interactive session into a pytest browser test. Use when the user asks to "screenshot /viz", "drive AutoRAG in a browser", "test the viz page", "debug the FastAPI server interactively", or similar.
---

# AutoRAG Puppeteer Skill

This skill teaches Claude how to use the puppeteer MCP server (already configured in `/workspace/.mcp.json`, GPU-accelerated Chrome with a visible window) against the AutoRAG app. It is structured the same way as `~/.claude/skills/chrome-settings/SKILL.md`: numbered surfaces, concrete recipes, verification commands, then a quick-reference table.

For any Chrome flag / GPU / HDR tweak, **defer to `~/.claude/skills/chrome-settings/SKILL.md`** — that skill owns those surfaces. This one only references them.

## 0. Preamble — things to know before the first tool call

- **Puppeteer tools are deferred.** In this environment the puppeteer tools are not listed in the top-of-prompt schema block. Before the first use in a session, load schemas with:
  ```
  ToolSearch(query="select:puppeteer_navigate,puppeteer_evaluate,puppeteer_screenshot,puppeteer_click,puppeteer_fill,puppeteer_hover,puppeteer_select", max_results=7)
  ```
  After that block returns, all seven tools are callable for the rest of the session.

- **Browser state persists across tool calls** within one MCP server lifetime: URL, cookies, `localStorage`, anything attached to `window`. There is one Chrome process. To get a clean slate, `puppeteer_navigate('about:blank')` — don't assume the page from a previous turn is gone.

- **`headless: false`** — Chrome opens a real window via display forwarding. A human moving the mouse over it during a session will alter raycast/hover state and dirty screenshots. Note this if the user reports flaky tooltips.

- **The persistent user-data dir** is `/home/node/.config/google-chrome/puppeteer`. It's shared across sessions, so a logged-in session or open tab survives MCP restarts.

## 1. Server lifecycle — `autorag serve`

Entry point: `autorag serve [--host 127.0.0.1] [--port 8000] [--reload]` (defined at `src/autorag/cli.py:36-40`, a thin wrapper over `uvicorn.run("autorag.api:app", ...)`).

### Start (background)

```
Bash(command="autorag serve --host 127.0.0.1 --port 8000", run_in_background=true)
```

Capture the returned `bash_id` — you will need it for cleanup.

Add `--reload` only when iterating on Python source. Reload watches the whole `src/` tree and slows the first response.

### Wait for ready

Do NOT call `puppeteer_navigate` until `/health` answers. Cold-start uvicorn can be 1-3 seconds; navigating before the socket is open leaves the MCP server holding a failed page that you then have to re-navigate.

```bash
curl -sS --retry-connrefused --retry 10 --retry-delay 1 http://127.0.0.1:8000/health
```

Expected: `{"status":"ok"}`. If it never returns OK in ~10s, check the background bash's stdout for an import error (very common: a missing extra).

### Find and kill stale instances

A previous session may have left a server running. Check before starting:

```bash
pgrep -af "autorag serve"     # PIDs + cmdlines
lsof -ti :8000                # PIDs holding port 8000
```

Kill: `kill <pid>`. Escalate to `kill -9 <pid>` only if the process ignores SIGTERM for >2s.

### Cleanup contract

At the end of an interactive session, kill the background pid you started. Do not leave zombies — they hold the port and confuse the next session. If the user explicitly says "leave it running", obviously skip this step.

## 2. The `/viz` page

`/viz` is a **Vite + React 18 + `@react-three/fiber`** app (source in `frontend/src/`), built to `src/autorag/static/viz/index.html` + hashed `assets/*` and served by `src/autorag/viz.py` (`_HTML_PATH = static/viz/index.html`). There is **no `viz.html`** — that vanilla file was deleted; do not look for it. The page is a left rail of DOM controls plus a fullscreen r3f `<canvas>`. CSS selectors reach the rail, overlays, and tooltip; anything drawn in the WebGL scene is unreachable from selectors.

### DOM-targetable selectors (verified against `frontend/src/ui/` + `three/Scene.tsx`)

All `id`s below are hand-written in JSX (stable — no CSS-in-JS hashing). The big behavioral change from the old vanilla page: **`#rail` / `#stats` only exist once data has loaded** (the rail is conditionally mounted, see App.tsx `showScene = !!data && !empty && !error`), and **the canvas has no `id`** (r3f creates it).

| Element | Selector | Notes |
|---|---|---|
| Search box | `#search-input` | text input, debounced **~350ms** (`useDebouncedSearch`) |
| Search hit list | `#search-results .search-hit` | button per hit; children `.search-hit-title`, `.search-hit-score` |
| Color-mode button | `#btn-color-mode` | toggles point coloring; gains `.active` in cluster mode |
| Edges button | `#btn-edges` | toggles edge visibility; gains `.active` when edges shown |
| Topic list | `#topic-list` | rows `.topic-row`; hovered row gets `.topic-row.active` (rail↔scene sync) |
| Stats line | `#stats` | **absent until loaded**, then `"<N> topics · <M> clips · <K> clusters"` (never `—`) |
| Legend | `#legend`, `#size-legend` | mounted with the rail |
| Loading overlay | `#loading-overlay` | always in DOM; gains class `visible` while `loading` |
| Empty / error overlays | `#empty-overlay`, `#error-overlay` (`#error-msg`) | each gains `visible` in its state; mutually exclusive |
| Tooltip | `#tooltip` | always in DOM; empty `<div id="tooltip"/>` until a point is hovered, then gains `visible` + inline position. Children are **classes**: `.tt-title`, `.tt-meta`, `.tt-summary` (`.tt-summary` only if the topic has a summary) |
| Rail | `#rail` | the 290px left column; only mounted when the scene shows |
| Canvas | `document.querySelector('canvas')` | **no `id`** — r3f `<Canvas>` (`three/Scene.tsx`), `position: fixed; inset: 0` |

### Anything NOT in that table is in the WebGL scene

Points, edges, camera, raycast/hover state, projected screen coordinates — none are on `window`. Cross-component scene state (color mode, edges toggle, hover/focus index, search results) lives in a **closure-scoped Zustand store** (`frontend/src/state/vizStore.ts`); the camera/points/raycaster are closure-scoped inside r3f controller components. Nothing is exposed for `puppeteer_evaluate`. Verify with `grep -rn 'window\.\|__viz\|globalThis' frontend/src` — it returns nothing (no debug handle exists).

So today: **DOM + screenshots only.** Read state through the rail's DOM mirror (`#stats`, `#topic-list .topic-row.active`, `#tooltip.visible`), not from JS. Don't hunt for a hidden global.

### Navigation gate

`puppeteer_navigate` resolves on `load`, but the React app then fetches `/viz/data` and r3f builds the scene asynchronously. Screenshotting too early returns the loading overlay or a bare canvas. The old `#stats !== '—'` trick is **dead** — `#stats` no longer exists until data loads. Gate on the scene actually being mounted:

```js
// scene ready: data loaded, rail mounted, loading overlay gone, canvas present
(!document.getElementById('loading-overlay').classList.contains('visible')
  && document.getElementById('rail')
  && document.querySelector('canvas'))
// OR a terminal non-scene state you can also screenshot:
|| document.getElementById('empty-overlay').classList.contains('visible')
|| document.getElementById('error-overlay').classList.contains('visible')
```

Poll via `puppeteer_evaluate` until truthy, then screenshot. (If `canvas` never appears but `#rail` does, that's the WebGL-fail SceneBoundary state — see section 4.)

### Recipe — load + screenshot

```
puppeteer_navigate(url="http://127.0.0.1:8000/viz")
# poll until ready (loop up to ~10s):
puppeteer_evaluate(script="
  (!document.getElementById('loading-overlay').classList.contains('visible')
    && document.getElementById('rail') && document.querySelector('canvas'))
  || document.getElementById('empty-overlay').classList.contains('visible')
  || document.getElementById('error-overlay').classList.contains('visible')
")
puppeteer_screenshot(name="viz-loaded")                  # whole page
puppeteer_screenshot(name="viz-rail",   selector="#rail")
puppeteer_screenshot(name="viz-canvas", selector="canvas")  # no #id — element selector
```

### Recipe — search end-to-end

```
puppeteer_fill(selector="#search-input", value="attention")
# wait ~400ms (debounce 350ms + fetch /viz/search + render):
puppeteer_evaluate(script="
  document.querySelectorAll('#search-results .search-hit').length
")
# click the top hit — onClick calls the Zustand setFocusIndex(point_index),
# which the scene's FocusController watches to pan the camera to that point:
puppeteer_click(selector="#search-results .search-hit:first-child")
puppeteer_screenshot(name="viz-focused-attention")
```

### Upgrade path — canvas-state introspection (future, not required)

If you ever need to assert on what's *drawn* (vs. what's in the rail), the minimum change is to expose the Zustand store on `window` behind a debug query param, so production users don't pay a leaked-global cost:

```ts
// frontend/src/main.tsx, after the store import
import { useVizStore } from "./state/vizStore";
if (new URLSearchParams(location.search).get("debug") === "1") {
  (window as unknown as { __viz: typeof useVizStore }).__viz = useVizStore;
}
```

After that ships and **the bundle is rebuilt and committed** (`cd frontend && npm run build` — the served page is the committed `static/viz/` build, not `frontend/src/`), `puppeteer_navigate("http://127.0.0.1:8000/viz?debug=1")` lets you read store state, e.g. `window.__viz.getState().focusIndex`, `window.__viz.getState().searchResults.length`, or drive it with `window.__viz.getState().setFocusIndex(7)`. Note this still doesn't expose the camera/points (closure-scoped in r3f controllers) — for that you'd additionally need a ref bridged out of `three/Scene.tsx`. **Do not ship this as part of using the skill** — propose it as a separate small change when the user actually needs scene introspection.

## 3. API + Swagger

The API is defined in `src/autorag/api.py`:

- `GET /health` — `{"status":"ok"}`
- `POST /ingest` (`IngestRequest` → `IngestResponse`)
- `POST /query` (`QueryRequest` → `QueryResponse`)
- `GET /viz`, `GET /viz/data`, `GET /viz/search` — **conditionally** mounted (see pitfall below)
- `GET /docs` — Swagger UI (FastAPI auto)
- `GET /openapi.json` — spec

### Hit the API from inside the browser (shares origin with /viz)

```
puppeteer_evaluate(script="await (await fetch('/health')).json()")
puppeteer_evaluate(script="
  await (await fetch('/query', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question: 'what is attention?', top_k: 3})
  })).json()
")
```

Use this when you want the request to share cookies / origin with the page you're driving.

### Hit the API from the shell (cleaner for assertions)

```bash
curl -sS http://127.0.0.1:8000/health | jq
curl -sS -X POST http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"what is attention?","top_k":3}' | jq
```

Use this for scripted checks where you don't need browser context.

### Drive Swagger `/docs` interactively

Sometimes the right artifact for a bug report is a screenshot of Swagger executing a payload:

```
puppeteer_navigate(url="http://127.0.0.1:8000/docs")
puppeteer_click(selector='.opblock-summary[data-path="/query"]')
puppeteer_click(selector='.opblock-section-header button.try-out__btn')
puppeteer_fill(selector='.opblock-body textarea',
               value='{"question":"what is attention?","top_k":3}')
puppeteer_click(selector='.opblock-section-header button.execute')
puppeteer_screenshot(name="swagger-query-200")
```

Read the rendered response with:

```
puppeteer_evaluate(script="
  document.querySelector('.responses-wrapper .response .response-col_description pre').textContent
")
```

### Sanity-check route registration

If `/viz` 404s, **don't assume a bug** — see the conditional-mount pitfall below. Confirm routes from the spec first:

```
puppeteer_evaluate(script="
  Object.keys((await (await fetch('/openapi.json')).json()).paths)
")
```

If `/viz` is missing here, it's an extras problem, not a routing bug.

## 4. Common pitfalls (devcontainer + Chrome + WebGL + AutoRAG)

- **Blank `/viz` canvas ≠ "didn't load".** With the GPU flags in `.mcp.json` (`--ignore-gpu-blocklist`, `--use-gl=angle`, `--use-angle=opengl`, `--enable-gpu-rasterization`), a solid-black `<canvas>` screenshot while `#rail` is mounted and `#stats` shows a topic count means **GPU init failed**, not page load (`/viz/data` succeeded — the rail proves it). Diagnose:
  ```
  puppeteer_evaluate(script="!!document.createElement('canvas').getContext('webgl2')")
  puppeteer_navigate(url="chrome://gpu")
  puppeteer_screenshot(name="chrome-gpu")
  ```
  Then route to `~/.claude/skills/chrome-settings/SKILL.md` for flag/driver fixes.

- **`#rail` present but NO `<canvas>` element = WebGL context couldn't be created at all** (distinct from a black canvas). r3f's `<Canvas>` throws synchronously; the `SceneBoundary` error boundary (`frontend/src/ui/SceneBoundary.tsx`) catches it and renders a plain text div — *"3D view unavailable — WebGL context could not be created. The topic list on the left still works."* — with **no id/class**. Detect with `document.querySelector('#rail') && !document.querySelector('canvas')`, or find the div by text. Headless/software-GL Chrome with swiftshader is known to land here in this devcontainer, so this is the expected failure mode when GPU forwarding is down — the rail + search + topic list still screenshot fine and remain the assertable surface.

- **Navigate-then-screenshot race.** `puppeteer_navigate` resolves on `load`, but the viz fetches `/viz/data` and uploads to WebGL asynchronously. Always poll the navigation gate (section 2) before screenshotting.

- **MCP state persists.** One Chrome process per MCP lifetime — cookies, localStorage, current URL, anything on `window` all survive across tool calls. Navigate to `about:blank` for a clean state.

- **`--disable-dev-shm-usage` is load-bearing.** Do NOT remove it from `.mcp.json`. Chrome OOMs on `/tmp` in the container without it. Same for `--no-sandbox --disable-setuid-sandbox`.

- **`/viz` 404 = `[rag]` extra missing**, not a bug. `viz_router` is mounted inside a `try/except ModuleNotFoundError` at `src/autorag/api.py:22-27` — without `chromadb` / `umap` / `sklearn` the router never loads. Fix: `uv sync --extra rag --extra server` (or `pip install 'autorag[rag,server] @ git+...'`).

- **First `/ingest` / `/query` is slow.** `get_rag()` is `lru_cache`d (`api.py:30-32`) and lazily loads heavy deps (chromadb, embeddings) on first call. Warm with a `/health` hit before any screenshot-sensitive timing.

- **`headless: false` means a real interactive window.** If the user moves their mouse over the Chrome window during a screenshot-sensitive flow, raycast hover will fire and `#tooltip` may flicker. Note this if a screenshot looks wrong but the DOM looks right.

- **Display forwarding can silently fail.** If Chrome launches but every screenshot is blank/black even on simple pages (e.g. `about:blank`), the host display isn't reachable. Try `puppeteer_navigate(url="data:text/html,<h1>hello</h1>")` + screenshot as a baseline before blaming the app.

## 5. Graduating an interactive recipe to a pytest test

When an interactive recipe stabilizes and you want to keep it, promote it. Don't try to reuse `mcp-server-puppeteer` from pytest — it's an MCP server, not a library.

### Recommended target: Playwright for Python

`pip install playwright pytest-playwright && playwright install chromium`. Mirrors the puppeteer API closely; transforms are mechanical.

### Folder layout

```
tests/
  browser/
    __init__.py
    conftest.py            # server fixture (see below)
    test_viz_smoke.py      # one test per stable recipe
    __snapshots__/         # screenshots, gitignored by default
```

Keep `tests/browser/` **separate** from `tests/`. Existing `TestClient`-based tests must not grow a Chrome dependency.

### Proposed `pyproject.toml` extra (separate PR, not part of this skill)

```toml
[project.optional-dependencies]
browser = ["playwright>=1.45", "pytest-playwright>=0.5"]
```

Install: `uv sync --extra browser` then `uv run playwright install chromium`. CI: skip `tests/browser/` by default; opt in via `pytest tests/browser -m browser` or a dedicated job.

### Server fixture (session-scoped)

```python
# tests/browser/conftest.py
from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Iterator

import httpx
import pytest


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def autorag_server() -> Iterator[str]:
    port = _free_port()
    proc = subprocess.Popen(
        ["uv", "run", "autorag", "serve", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                break
        except httpx.RequestError:
            time.sleep(0.2)
    else:
        proc.terminate()
        raise RuntimeError("autorag serve did not become ready in 10s")
    try:
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=5)
```

Do **not** swap this for FastAPI's `TestClient` — Playwright needs a real socket.

### One-to-one transform table

| Interactive (this skill) | Playwright (pytest) |
|---|---|
| `puppeteer_navigate(url=URL)` | `page.goto(URL)` |
| `puppeteer_fill(selector=S, value=V)` | `page.fill(S, V)` |
| `puppeteer_click(selector=S)` | `page.click(S)` |
| `puppeteer_evaluate(script=JS)` | `page.evaluate(JS)` |
| `puppeteer_screenshot(name=N)` | `page.screenshot(path=f"tests/browser/__snapshots__/{N}.png")` |
| navigation-gate poll | `page.wait_for_function("...")` |

### Worked example

```python
# tests/browser/test_viz_smoke.py
from playwright.sync_api import Page


def test_viz_loads_and_search_returns_hits(page: Page, autorag_server: str) -> None:
    page.goto(f"{autorag_server}/viz")
    page.wait_for_function(
        "!document.getElementById('loading-overlay').classList.contains('visible')"
        " && document.getElementById('rail') && document.querySelector('canvas')",
        timeout=10_000,
    )
    page.fill("#search-input", "attention")
    page.wait_for_function(
        "document.querySelectorAll('#search-results .search-hit').length > 0",
        timeout=2_000,  # 350ms debounce + /viz/search round-trip
    )
    hits = page.eval_on_selector_all("#search-results .search-hit", "els => els.length")
    assert hits >= 1
```

### Snapshot artifacts

Save screenshots under `tests/browser/__snapshots__/` and add that path to `.gitignore`. Treat snapshots as debug aids you can open after a failure — **not** as assertion fixtures. Canvas pixels are not deterministic across GPU drivers; assert on DOM and route behavior instead.

## 6. Quick reference

| Action | Tool | Wait condition |
|---|---|---|
| Start server | `Bash(run_in_background=true)` + `curl /health` | `{"status":"ok"}` |
| Find stale | `pgrep -af 'autorag serve'`, `lsof -ti :8000` | — |
| Load `/viz` | `puppeteer_navigate` | `#loading-overlay` not `.visible` AND `#rail` + `canvas` present (or `#empty-overlay`/`#error-overlay` `.visible`) |
| Search | `puppeteer_fill('#search-input')` | `#search-results .search-hit` count > 0 (~350ms debounce) |
| Click hit | `puppeteer_click('#search-results .search-hit:first-child')` | After hits visible |
| Call API (browser) | `puppeteer_evaluate("await (await fetch(...)).json()")` | HTTP 200 |
| Call API (shell) | `curl -sS ... \| jq` | HTTP 200 |
| Drive Swagger | `puppeteer_navigate('/docs')` + clicks | `.responses-wrapper pre` populated |
| Screenshot page | `puppeteer_screenshot(name=...)` | After load gated above |
| Screenshot canvas | `puppeteer_screenshot(name=..., selector='canvas')` | Same (no `#id` — element selector) |
| Screenshot rail | `puppeteer_screenshot(name=..., selector='#rail')` | Same |
| Detect WebGL fail | `puppeteer_evaluate("!!document.querySelector('#rail') && !document.querySelector('canvas')")` | `true` = SceneBoundary fallback |
| Inspect store state | `puppeteer_evaluate('window.__viz.getState()...')` | Requires `?debug=1` Zustand-on-window instrumentation in `frontend/src/main.tsx` (future) |
| Cleanup | `kill <bash_id pid>` | — |
