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

`/viz` (mounted at `src/autorag/viz.py`, served via `src/autorag/static/viz.html`) is a 3D WebGL constellation: a left rail of DOM controls and a fullscreen `<canvas>`. CSS selectors only reach the rail and overlays — anything drawn on the canvas is unreachable from selectors.

### DOM-targetable IDs (verified against `viz.html`)

| Element | Selector | Notes |
|---|---|---|
| Search box | `#search-input` | text input, debounced ~300ms |
| Search hit list | `#search-results .search-hit` | rendered after debounce settles |
| Color-mode button | `#btn-color-mode` | toggles point coloring |
| Edges button | `#btn-edges` | toggles edge visibility |
| Topic list | `#topic-list` | populated after data load |
| Stats line | `#stats` | shows `—` until loaded, then counts |
| Legend | `#legend`, `#size-legend` | populated after data load |
| Loading overlay | `#loading-overlay` | has class `visible` while loading |
| Empty / error overlays | `#empty-overlay`, `#error-overlay` (`#error-msg`) | mutually exclusive states |
| Tooltip | `#tooltip` (`#tt-title`, `#tt-meta`, `#tt-summary`) | follows pointer over canvas |
| Rail | `#rail` | the whole 290px left column |
| Canvas | `#canvas` | the WebGL surface |

### Anything NOT in that table is on the canvas

Points, edges, the camera, the hover/raycast state, projected screen coordinates — all are held in **closure-scoped** JS variables inside `viz.html` (`data`, `scene`, `camera`, `renderer`, `focusPoint`, `runSearch`, `pointWorldPositions`). None are on `window`. Verify with `grep -n 'window\.' src/autorag/static/viz.html` — you'll only see `window.innerWidth`, `window.addEventListener`, `window.devicePixelRatio`.

So today: **DOM + screenshots only.** Don't try to read scene state from `puppeteer_evaluate` and don't waste time hunting for a hidden global.

### Navigation gate

`puppeteer_navigate` resolves on `load`, but Three.js scene construction is async (the `/viz/data` fetch + WebGL buffer uploads). Screenshotting too early returns the loading overlay or an unrendered canvas. Always gate on:

```js
!document.getElementById('loading-overlay').classList.contains('visible')
  && document.getElementById('stats').textContent !== '—'
```

Poll via `puppeteer_evaluate` until both are true, then screenshot.

### Recipe — load + screenshot

```
puppeteer_navigate(url="http://127.0.0.1:8000/viz")
# poll until ready (loop up to ~10s):
puppeteer_evaluate(script="
  !document.getElementById('loading-overlay').classList.contains('visible')
  && document.getElementById('stats').textContent !== '—'
")
puppeteer_screenshot(name="viz-loaded")                 # whole page
puppeteer_screenshot(name="viz-rail",   selector="#rail")
puppeteer_screenshot(name="viz-canvas", selector="#canvas")
```

### Recipe — search end-to-end

```
puppeteer_fill(selector="#search-input", value="attention")
# wait ~350ms (debounce ~300ms + render):
puppeteer_evaluate(script="
  document.querySelectorAll('#search-results .search-hit').length
")
# click the top hit to invoke focusPoint(idx) internally:
puppeteer_click(selector="#search-results .search-hit:first-child")
puppeteer_screenshot(name="viz-focused-attention")
```

### Upgrade path — canvas-state introspection (future, not required)

If you ever need to assert on what's *drawn* (vs. what's in the rail), the minimum change is to expose a debug handle inside `viz.html`, behind a query param so production users don't pay a leaked-global cost:

```js
// inside viz.html, near the bottom of the module
if (new URLSearchParams(location.search).get('debug') === '1') {
  window.__viz = { data, scene, camera, renderer, focusPoint, runSearch, pointWorldPositions };
}
```

After that ships, `puppeteer_navigate("http://127.0.0.1:8000/viz?debug=1")` lets you do things like `window.__viz.data.points.length`, `window.__viz.focusPoint(7)`, or project `pointWorldPositions[i]` to screen coordinates for click testing. **Do not ship this as part of using the skill** — propose it as a separate small change when the user actually needs canvas introspection.

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

- **Blank `/viz` canvas ≠ "didn't load".** With the GPU flags in `.mcp.json` (`--ignore-gpu-blocklist`, `--use-gl=angle`, `--use-angle=opengl`, `--enable-gpu-rasterization`), a solid-black canvas screenshot while `#stats` shows a topic count means **GPU init failed**, not page load. Diagnose:
  ```
  puppeteer_evaluate(script="!!document.createElement('canvas').getContext('webgl2')")
  puppeteer_navigate(url="chrome://gpu")
  puppeteer_screenshot(name="chrome-gpu")
  ```
  Then route to `~/.claude/skills/chrome-settings/SKILL.md` for flag/driver fixes.

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
        " && document.getElementById('stats').textContent !== '—'",
        timeout=10_000,
    )
    page.fill("#search-input", "attention")
    page.wait_for_function(
        "document.querySelectorAll('#search-results .search-hit').length > 0",
        timeout=2_000,
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
| Load `/viz` | `puppeteer_navigate` | `#loading-overlay` not `.visible` AND `#stats !== '—'` |
| Search | `puppeteer_fill('#search-input')` | `#search-results .search-hit` count > 0 (~300ms debounce) |
| Click hit | `puppeteer_click('#search-results .search-hit:first-child')` | After hits visible |
| Call API (browser) | `puppeteer_evaluate("await (await fetch(...)).json()")` | HTTP 200 |
| Call API (shell) | `curl -sS ... \| jq` | HTTP 200 |
| Drive Swagger | `puppeteer_navigate('/docs')` + clicks | `.responses-wrapper pre` populated |
| Screenshot page | `puppeteer_screenshot(name=...)` | After load gated above |
| Screenshot canvas | `puppeteer_screenshot(name=..., selector='#canvas')` | Same |
| Screenshot rail | `puppeteer_screenshot(name=..., selector='#rail')` | Same |
| Inspect Three.js state | `puppeteer_evaluate('window.__viz...')` | Requires `?debug=1` viz.html instrumentation (future) |
| Cleanup | `kill <bash_id pid>` | — |
