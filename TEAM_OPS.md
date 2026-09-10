# Team Operations

How this team runs. Owned by Shan (team lead).

> This document exists because the failure mode on a small student team isn't people refusing to work — it's work that was never clearly assigned quietly reverting to the lead. Every task below has exactly one name on it.

---

## 1. Ownership lanes

One owner per lane. The owner is accountable for the lane whether or not they do every task in it themselves.

| Lane | Owner | Scope |
|---|---|---|
| **Firmware** | *unassigned* | `compile.ino` — capture loop, WebSocket clients, sensors, I2S, stability |
| **Backend** | *unassigned* | `app_main.py`, Gemini client, transport, deployment |
| **Dashboard** | *unassigned* | `templates/index.html`, `static/main.js` |
| **Research platform** | *unassigned* | Next.js app, Prisma/Neon schema, ingest |
| **Research + faculty-facing** | Shan | Sessions, analysis, CSUN abstract, Prof. Tang comms |
| **Integration + architecture** | Shan | Cross-cutting decisions, final call on architecture |

**Fill these in before assigning anything else.** An unowned lane is a lane that becomes Shan's by default.

**Rule:** if a task doesn't fall in a lane, it gets an explicit owner or it doesn't get worked on. It does not silently become the lead's.

---

## 2. Task assignment format

Every assignment, without exception:

```
TASK:      [one sentence, concrete]
OWNER:     [one name — never two]
DUE:       [specific date, not "next week"]
DONE MEANS: [observable outcome — what I can check]
```

"Done means" is the part people skip and the part that prevents 80% of misunderstandings. "Look into the mirroring issue" is not a task. "Confirm `set_hmirror` is 0 in the flashed build and post a photo of a left-right test, by Friday" is.

---

## 3. Cadence

### Weekly sync — 20 minutes, same slot every week
Three questions per person:
1. What did you finish?
2. What are you blocked on?
3. What's next, by when?

Not a status theater meeting. If someone has nothing, that's information, and it surfaces in week 1 instead of week 6.

### Async
Task board (or `TEAM_OPS.md` §5 below) is the source of truth. Shan checks it — Shan does not chase people individually. If a task is overdue it's visible to everyone.

### Devlog
Every working session → one `DEVLOG.md` entry. Three minutes.

---

## 4. Escalation ladder

**Decided in advance, on purpose, so it's not a judgment call made while annoyed.**

| Trigger | Action |
|---|---|
| Deadline missed, no message | Direct check-in from Shan within 24h. Assume a real reason. |
| Second miss on the same task, no explanation | Reassign the task or cut its scope. Say so plainly. |
| Pattern across multiple tasks | Raise with Prof. Tang, with the record: what was assigned, when, what landed. |
| Someone blocked >2 days | Shan unblocks it. That's the lead's job. |

**Two things this ladder protects against:**
- Shan silently absorbing unfinished work (this is the actual historical failure)
- Escalating to the professor out of frustration rather than out of process

Raising a documented pattern with an advisor is normal project management, not a complaint.

---

## 5. Active tasks

| Task | Owner | Due | Done means | Status |
|---|---|---|---|---|
| Fill in ownership lanes above | Shan | | Every lane has a name | 🔴 |
| README + onboarding docs in repo | Shan | | Merged, new person can set up unaided | 🟡 |
| Architecture diagram (Excalidraw) | Shan | | Diagram exists; Ian + Alicia can be assigned from it | 🔴 |
| Confirm `set_hmirror` = 0 in flashed build | *unassigned* | | Left/right verified physically, result in DEVLOG | 🔴 |
| I2C/SCCB bus-sharing check (read-only) | *unassigned* | | Answer to: does `apply_framesize()` take the shared mutex? | 🔴 |
| XGA decision — fall-through or retire | Shan | | Decision recorded in PROJECT_CONTEXT §3.2 | 🔴 |
| First flash test — Camera Researcher Control | *unassigned* | | Flashed, resolution + Stream FPS change at runtime | 🔴 |
| Phase 1 — Gemini offline localization | *unassigned* | | Results doc from `recordings/` footage | 🔴 |
| Firmware build hash + SNTP | *unassigned* | | Every session's data identifies its build | 🔴 |

🔴 not started · 🟡 in progress · 🟢 done · ⚠️ blocked

> **The architecture diagram is the unlock.** Until it exists, Ian and Alicia can't be given independent lanes, which means everything routes through Shan. It's the highest-leverage item on this board.

---

## 6. Standing notes for the lead

- **Your job is unblocking, not doing.** When someone is stuck, remove the blocker — get them access, answer the question, loop in Tang. Do not take the task back. Taking it back is what got us here.
- **Cap your hours on this.** Decide the number per week in advance. Team leadership without a cap becomes doing everyone's work with extra meetings.
- **Written beats verbal.** Verbal asks get vague timelines. Written asks with dates get answers or visible silence — both are useful.
- **Assume good faith first.** Everyone here has a course load. A missed deadline is usually a scheduling failure, not disrespect. The ladder handles the rest.
