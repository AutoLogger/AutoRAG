---
name: commit
description: Stage relevant changes, propose a semver version bump (major/minor/patch with reasoning) when the change merits one, update CHANGELOG.md under [Unreleased] per Keep-a-Changelog 1.1.0, sync `pyproject.toml` + `src/autorag/__init__.py` + `uv.lock` on a bump, scan CLAUDE.md and `docs/` for stale claims, draft a commit message in this repo's house style (imperative subject, no conventional-commit prefix, no co-author footer), and show the full plan for user approval before running `git commit`. Stops at the commit — never pushes or tags. Use this skill whenever the user says "commit", "make a commit", "ship this", "wrap this up", "land it", "release", "bump the version", "update the changelog", or otherwise asks to land work in git, even when they don't mention CHANGELOG or version bumps. Also triggers on freeform overrides like "commit just the API file" or "use this message: …".
---

# Commit Skill

Stage → bump (maybe) → update changelog → sync docs → draft message → **show plan** → commit.

The user always sees the full plan before any `git commit` runs. Their freeform input can change what's staged, force or skip a version bump, override the message, or skip changelog/docs updates.

## 0. Preamble — things to know before starting

- **AutoRAG-specific.** Two version locations to keep in lockstep: `pyproject.toml` (`version = "x.y.z"`) and `src/autorag/__init__.py` (`__version__ = "x.y.z"`). A version bump also requires `uv lock` (per CLAUDE.md, which is the source of truth for release flow).
- **Hooks will run.** A PostToolUse hook auto-runs `ruff check --fix` + `ruff format` on every Python `Edit`/`Write`, and `biome check --write` on `.ts`/`.tsx`. Expect formatting noise after editing `__init__.py` — that's normal and the lint failures still surface to you. Don't try to disable hooks.
- **No co-author footer.** `.claude/settings.json` sets `attribution.commit: "false"` and `gitAttribution: false`. The commit body should NOT include `Co-Authored-By: Claude` or `🤖 Generated with…`. Plain message only.
- **Stop at the commit.** Never `git push`, never `git tag`. The user does those explicitly.
- **Never `--no-verify`.** If a pre-commit hook fails, surface the failure and fix the underlying issue. Skipping verification is a non-starter.

## 1. Read the situation

Before doing anything else, gather state. Run these in parallel — they're independent:

```bash
git status --short
git diff --stat                       # unstaged
git diff --staged --stat              # already-staged
git log --oneline -10                 # house style + recent context
```

Then read the **content** of the diff for any non-trivial changes (`git diff <path>` / `git diff --staged <path>`). The classification, version-bump call, and CHANGELOG bucket all depend on what actually changed — not just file names.

If the working tree is clean and the index empty, tell the user there's nothing to commit and stop.

## 2. Interpret user input

The user's prompt may include freeform overrides. Parse them before planning:

| User says…                                                | Effect on plan                                                              |
|-----------------------------------------------------------|-----------------------------------------------------------------------------|
| `"only commit src/autorag/api.py"` / `"just the API file"`| Stage only the named paths; ignore other dirty files                        |
| `"don't stage the test file"` / `"skip <path>"`           | Exclude those paths from staging                                            |
| `"force major"` / `"this is a minor"` / `"patch only"`    | Use the named bump regardless of what the diff suggests                     |
| `"no version bump"` / `"don't bump"` / `"wip"`            | Skip version + lockfile changes; entries still go to `[Unreleased]`         |
| `"skip changelog"` / `"no changelog"`                     | Don't touch CHANGELOG.md                                                    |
| `"use this message: <text>"` / `"call it <text>"`         | Use that subject verbatim; you may still suggest a body                     |
| `"wip"` / `"checkpoint"`                                  | Skip all of: version, changelog, docs scan. Just stage + commit             |

If the input is ambiguous (e.g., "ship the api work"), proceed with normal flow and call out your interpretation in the plan summary so the user can correct it.

## 3. Decide what to stage

Default: stage everything that contributes to the logical change — including untracked files that are clearly part of it.

Skip by default unless the user says otherwise:
- Build artifacts: `__pycache__/`, `*.pyc`, `.venv/`, `node_modules/`, `dist/`, `build/`, `docs/_build/`, `src/autorag/static/viz/` only if it looks like a stale local build (i.e., committed bundle is up to date — verify against `frontend/src/` changes; per CLAUDE.md, the built bundle SHOULD be committed alongside frontend source changes)
- Local-only configs: `.claude/settings.local.json`
- Anything matching `.gitignore` (git already excludes these, but double-check)

Be wary of:
- Files that look like scratch / experiments (e.g., `transcribe_test.py` at the repo root, `*_test.py` outside `tests/`). Surface these in the plan and ask whether to include — don't silently commit them.
- Lockfile changes (`uv.lock`) without a corresponding version or `pyproject.toml` change — usually means a stray `uv sync` ran. Ask before staging.

Use `git add <path>` per file, not `git add -A` / `git add .`, so the user sees exactly what's going in.

## 4. Classify the change → choose a bump (if any)

**Default: no version bump.** Ordinary commits accumulate under `[Unreleased]` and the version stays put. A bump happens only when:

- The user explicitly signals a release: `"release this"`, `"ship 0.7"`, `"bump to 0.7.0"`, `"cut a release"`, `"this is the 0.7 release"`.
- OR the change is so clearly a release in itself (large additive feature, breaking change) that you want to surface the option in the plan — frame it as a question, not a fait accompli: *"This adds a new public method — want me to bump to 0.7.0 and promote, or leave under [Unreleased]?"*

When a bump IS happening, propose the level. Be honest about why you picked it. SemVer rules (https://semver.org/spec/v2.0.0.html):

- **Major (x.0.0)** — backwards-incompatible change to the public API. For AutoRAG that means: removing or renaming anything on `AutoRAG`, changing a method's signature in a non-additive way, removing a CLI command/flag, dropping an extras name, dropping a public type from `autorag.types`. Renames of `WordSpan` / `TopicTree` / `TranscriptionResult` fields. Changing the lazy-import contract that `test-base` enforces.
- **Minor (0.x.0)** — additive: new public method on `AutoRAG`, new CLI command, new public helper in `audio_source` / `blocks` / `persistence`, new extras flag, new schema field that doesn't break existing callers, new module under `autorag.*`.
- **Patch (0.0.x)** — bug fix, performance fix, internal refactor with no surface change, doc-only change to existing API, dependency pin tightening.
- **No bump** — pure dev-tooling changes (CI, hooks, devcontainer, lint config, `.gitignore`), test-only additions, doc edits to `docs/` or `README.md`, scratch files, skill files. Entries still go under `[Unreleased]` so the next real release picks them up; CHANGELOG just doesn't get promoted.

Edge cases:
- Mixed change (new method + bug fix) → take the highest applicable level (minor wins).
- A "fix" that subtly changes behaviour callers depend on → that's a major. Read the diff, don't trust the file name.
- Pre-1.0.0 (`0.x.y`): the project is on `0.6.0`. Per SemVer §4, anything goes in `0.x.y`. In practice this repo treats `0.MINOR.PATCH` like SemVer, so apply the same rules — additive → minor, fix → patch.

If you're proposing a bump, name the next version explicitly. Read current version from `src/autorag/__init__.py` / `pyproject.toml` (they should match — flag if they don't).

## 5. Update CHANGELOG.md

Format reference: https://keepachangelog.com/en/1.1.0/. Section order:
`### Added` → `### Changed` → `### Deprecated` → `### Removed` → `### Fixed` → `### Security`

### 5a. Always: append to `[Unreleased]`

For every commit that has user-visible behaviour, write one or more bullets under `## [Unreleased]` in the appropriate section header. Create the section header if it doesn't already exist under `[Unreleased]`. Bullets:

- **Imperative or noun-phrase**, past tense is fine if it matches existing style — read existing entries in CHANGELOG.md and match.
- Lead with the **public-facing name** (method, CLI command, file path, flag) in backticks so readers can grep for it.
- One sentence per bullet; wrap long lines (~80 chars) — see existing entries for the wrap style.
- Skip CHANGELOG entirely for: dev-tooling changes (CI, hooks, devcontainer, lint), test-only additions that don't change observable behaviour, doc-only edits, skill files. These are invisible to users.

### 5b. On a version bump: promote `[Unreleased]`

When bumping from current version `vA.B.C` to `vX.Y.Z`:

1. Rename the existing `## [Unreleased]` heading to `## [X.Y.Z] - YYYY-MM-DD` (UTC date — get it from `date -u +%F`).
2. Insert a fresh empty `## [Unreleased]` heading above it.
3. Update the compare-link footer at the bottom of the file:
   - Change `[Unreleased]: …/compare/vA.B.C...HEAD` → `[Unreleased]: …/compare/vX.Y.Z...HEAD`
   - Add a new line: `[X.Y.Z]: https://github.com/AutoLogger/AutoRAG/compare/vA.B.C...vX.Y.Z`
   - Insert it directly under the `[Unreleased]` link, before the existing `[A.B.C]` entry, so versions stay in descending order.

### 5c. If `[Unreleased]` is empty when bumping

That means there are unreleased entries to promote, but you also need to make sure this commit's changes get bullets first. Add the bullets, then promote.

## 6. Sync version files (only on a bump)

Atomically — these three changes go in the same commit:

1. `pyproject.toml`: change `version = "A.B.C"` to `version = "X.Y.Z"` (single line).
2. `src/autorag/__init__.py`: change `__version__ = "A.B.C"` to `__version__ = "X.Y.Z"`.
3. Run `uv lock`. This regenerates `uv.lock` with the new version. Stage the resulting `uv.lock` change.

Verify both string edits landed by re-reading the lines you changed (Edit tool errors loudly if it didn't match — don't re-read just to be sure). Verify the two versions match each other.

## 7. Scan CLAUDE.md and `docs/` for stale claims

These docs make specific claims (file paths, method names, version numbers, "currently…", extras tables). When the diff renames or removes something they reference, the docs go stale silently. Scan for:

- **CLAUDE.md** — search the file for any public name that this commit renamed/removed/added. The "Existing Conventions" and "Packaging" tables and the prose under "SDK facade" / "Audio → transcript + topics agent" / "YouTube URL input" are the highest-risk sections. Update only the lines that are now wrong; don't rewrite working sections.
- **`docs/`** — Sphinx `.rst` files. `docs/changelog.rst` likely just `include`s `CHANGELOG.md`, so no edit needed. Check `docs/reference/` (autodoc directives — usually self-updating from docstrings, but module renames will break them) and `docs/user-guide/` + `docs/quickstart.rst` for hand-written prose that names the changed surface.
- **`README.md`** — quick scan for outdated install commands or feature claims if the change touches install / extras / top-level surface.

Don't touch docs that aren't actually wrong. Stale ≠ "could be improved." If you're unsure, leave it and call out in the plan summary that it might be worth a follow-up.

## 8. Draft the commit message

**Subject line** (~50 chars, hard cap ~72):
- Imperative mood, capitalized first letter, **no trailing period**.
- **No conventional-commit prefix** (no `feat:`, `fix:`, `chore:`). Match existing log style: `git log --oneline -10` shows the house style — use it.
- Examples from this repo: `Add Documentation pointer to README`, `Switch transcription backend from openai-whisper to whisperX`, `Free Whisper and pyannote VRAM after inference`.

**Body** (when the change deserves it — most subject-only commits are fine):
- Blank line after subject.
- Wrap at ~72 chars.
- Explain **why** more than what (the diff shows what). Mention motivation, constraint, or the problem this solves. Reference the affected public name(s) so future-you can `git log --grep`.
- For multi-aspect commits, use short paragraphs separated by blank lines, not bullet lists, unless the existing log shows lists. Keep it tight — a paragraph beats a wall of bullets.
- For a release commit (version bump), the body can briefly summarize the new version's themes; the CHANGELOG carries the detail, so don't duplicate.

**Do not include**:
- `Co-Authored-By: Claude <…>` lines
- `🤖 Generated with [Claude Code]…` footer
- `Signed-off-by:` unless the user asks
- Issue/PR references unless the user provided them

## 9. Show the plan, then commit

Before running `git commit`, present a plan in this shape (adapt sections — omit ones that don't apply, e.g., no "Version bump" section for a docs commit):

```
## Plan

**Stage:**
- src/autorag/foo.py
- tests/test_foo.py
- (skipping: transcribe_test.py — looks like scratch; include?)

**Version bump:** 0.6.0 → 0.7.0 (minor)
Reasoning: adds AutoRAG.bar() — net-new public method.
Files: pyproject.toml, src/autorag/__init__.py, uv.lock

**CHANGELOG.md:** promote [Unreleased] → [0.7.0] - 2026-05-14
New entries under Added:
- `AutoRAG.bar(...)`: <one-liner>

**Docs scan:** CLAUDE.md SDK-facade table updated to list `bar`. No docs/ changes.

**Commit message:**
─────
Add AutoRAG.bar for X

<body if any>
─────

Run `git commit`?
```

Wait for the user's reply. They may approve, ask for tweaks, or change scope. Apply changes and re-show the plan if anything material moved.

When the user approves, stage the planned files (`git add <each path>`), then commit with a HEREDOC so multi-line bodies land cleanly:

```bash
git commit -m "$(cat <<'EOF'
<subject>

<body if any>
EOF
)"
```

Then run `git status` and `git log -1 --stat` and show the user. Done — no push, no tag.

## 10. Failure handling

- **Pre-commit hook fails.** The commit didn't happen. Read the failure, fix the underlying issue (lint error, type error, test failure), re-stage the fix, and run a NEW `git commit` (not `--amend`). Never `--no-verify`.
- **`uv lock` fails.** Probably an unrelated dependency issue. Show the user the error, don't paper over it. Without a clean lockfile, don't proceed with the version bump — the CI base-install job will fail.
- **Version mismatch detected** (`__init__.py` says one thing, `pyproject.toml` says another). Stop and tell the user; they'll want to investigate before bumping further.
- **Edit conflicts on CHANGELOG.md** (e.g., the `[Unreleased]` section already had entries from a previous commit). That's expected — append to existing sections, don't replace them. If the existing entries belong to the version you're now promoting, fold them in.

## Quick reference

| Step | Command / file |
|------|----------------|
| Survey | `git status --short`, `git diff[--staged] [--stat]`, `git log --oneline -10` |
| Date for CHANGELOG | `date -u +%F` |
| Version files | `pyproject.toml` (line ~7), `src/autorag/__init__.py` (line ~15) |
| Regenerate lock | `uv lock` |
| Stage | `git add <path>` (per file, never `-A`) |
| Commit | HEREDOC form per §9 |
| Verify | `git log -1 --stat` |
