# Repo Spec Conventions

**Purpose:** the shared format for every `spec_doc_v<N>.md` in this folder, so specs stay readable by humans and agents
and don't grow bloated. Each spec links here instead of repeating these rules.

**Contents:** [1. Item types](#1-item-types) · [2. Status](#2-status) · [3. Document layout](#3-document-layout) ·
[4. Item format](#4-item-format) · [5. Writing rules](#5-writing-rules) · [6. Lifecycle and concurrency](#6-lifecycle-and-concurrency) ·
[7. History](#7-history)

---

## 1. Item types

| Prefix | Type | What it holds | Status values |
|---|---|---|---|
| `I<n>` | Infrastructure | repo layout, tooling, launchers, results utilities, docs | work status (§2) |
| `M<n>` | Modeling | losses, models, configs, training code | work status |
| `E<n>` | Experiment | a question answered with runs, tables, figures and a write-up | work status |
| `C<n>` | Carryover | open work inherited from an earlier spec | work status |
| `C!<n>` | Caveat | a known issue that changes how existing results should be read | `open` · `resolved` |
| `D<n>` | Decision | a choice with a one-line rationale | `decided` · `pending` |

- **IDs are stable:** never renumbered or reused. A dropped item keeps its ID with status `dropped`.
- **Sub-steps** use dots: `E1.2`, `M7b`.
- **Cross-spec references** carry the version: `v1:M5b`, `v2:C!1`. Within a spec, a bare ID refers to that spec.

## 2. Status

Work status is one of:

| Status | Meaning |
|---|---|
| `planned` | designed, not started |
| `in progress` | being worked on (has an owner) |
| `blocked (on X)` | cannot proceed until X; name X by ID where possible |
| `done` | accepted; `Result` points to the commit(s) or write-up |
| `superseded (by X)` | replaced by another item; keep only a one-line pointer |
| `dropped` | abandoned; one line saying why |

`outline` may be used for an `E` item whose design is not yet complete.

## 3. Document layout

Every spec has, in order:
1. **Header:** title, one-line **purpose**, spec **status** (`active` / `closed <date>`), links to predecessor / successor
   and to this file.
2. **Contents:** a linked table of contents.
3. **Status table:** one row per item, `ID · title · status · depends on · owner`. This is the single place to read the
   state of the spec. Keep it current.
4. **Sections:** scope/purpose detail, conventions specific to this spec (only what differs from this file), items
   grouped by type, backlog.
5. **Change log** (append-only), then `Last updated at: <date> EDT`.

## 4. Item format

```
### E<n> — <title>   · status: <status> · owner: <owner>
- Question / Goal:     (E: the question; I/M: the goal)
- Changes:             what will change (files, launchers, configs)
- Accept:              checkable acceptance criteria
- Result:              commit hash(es) or write-up path; one or two lines
- Depends on / Budget: other IDs; GPU-h per stage for E items
```

- **Owner values:** `user`, `agent:infra`, `agent:modeling`, or `—` (unassigned). An agent sets itself as owner before
  starting an item, so parallel agents do not work on the same item.
- **Caveats** (`C!`) list the affected artifacts (experiments, benchmarks, runs) and what would resolve them.
  Experiment write-ups cite the `C!` ID instead of restating the caveat.

## 5. Writing rules

- **Specs hold intent, acceptance and a pointer to the result.** Details live in code, commit messages, experiment
  write-ups (`scripts/experiments/<slug>/<slug>.md`) and `CONTEXT.md`.
- **Closed items shrink** to their outcome and commit hash. Superseded plans are removed, not kept verbatim; git history
  keeps the old text.
- **No duplication across docs:** specs are plans and records; `CONTEXT.md` describes the current repo; `README.md` is the
  user guide. Link instead of copying.
- Plain language, short sentences, tables for anything list-shaped. No emoji status markers; the status word is enough.

## 6. Lifecycle and concurrency

- **One active spec at a time.** A new version starts when the scope shifts; the old one is closed and frozen.
- **Closing a spec:** set its status to `closed <date>`, move open items to the successor as `C` (open work) or `C!`
  (caveats), and leave a pointer. After closure, only typo fixes and a successor pointer may change.
- **Caveats move forward:** an open `C!` always lives in the active spec, so closed specs never need editing.
- **One editor per spec file at a time.** Before editing, re-read the file (another agent may have changed it); keep edits
  small; stage only the files you changed.

## 7. History

- v1 was renamed to these IDs on 2026-09-29: groundwork `G1`–`G7` → `I1`–`I7`, stage-1 steps `T1`–`T8` → `I8`–`I15`.
  Its headings and status table keep `(was Tn)`, because commit messages cite the `T` numbers.
- These conventions apply from v2 onward (adopted 2026-09-29).

Last updated at: 2026-09-29 EDT
