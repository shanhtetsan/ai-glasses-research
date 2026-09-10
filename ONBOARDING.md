# Onboarding

Welcome. This gets you from zero to useful. Budget about two hours for Day 1.

---

## Read in this order

1. **`README.md`** — what the system is and how the pieces connect.
2. **This file** — how to get set up and how we work.
3. **`PROJECT_CONTEXT.md`** — *the important one.* Why the system is shaped this way, what we already tried, what's broken and why. Skipping this means you will re-derive things we learned the hard way, or "fix" something that's intentional.
4. **`TEAM_OPS.md`** — who owns what, and what you're assigned.

---

## What we're actually building, and who it's for

Glasses that talk. A blind or low-vision person wears them, asks a question about what's around them, and gets spoken guidance back.

That last part is worth sitting with. **A real person may walk based on what this system says.** That changes how we treat bugs. A left/right inversion isn't cosmetic — it points someone the wrong way. Guidance about a clear path, when we have no depth sensor, is a claim we can't back.

We are also a **research** project. We run sessions with real participants through VISIONS. That means data integrity matters as much as working code: if a metric is wrong, conclusions built on it are wrong, and we've been burned by exactly that (see `PROJECT_CONTEXT.md` §4.2, §4.5).

---

## Day 1 setup

### 1. Access
- [ ] GitHub repo access
- [ ] Fly.io access (ask Shan)
- [ ] Google Drive / project docs folder
- [ ] Team channel

### 2. Get the backend running locally
```bash
git clone [repo]
cd [repo]
pip install -r requirements-cloud.txt   # NOT requirements.txt — see README
```
There is no `.env.example` — create `.env` yourself (it's gitignored). Minimum: `GEMINI_API_KEY=...`. See `README.md` → "Running locally" for the full optional-env-var list.

```bash
python3 app_main.py
```
Dashboard: `http://localhost:8081/` (port is hardcoded in `app_main.py`'s `__main__` block, matches `fly.toml` and the `Dockerfile`).

**Trap #1:** `requirements.txt` is a local workstation build (torch, ultralytics). `requirements-cloud.txt` is what actually deploys. Installing the wrong one wastes 20 minutes and several GB.

**Trap #2:** the live dashboard files are `templates/index.html` and `static/main.js`. There are orphaned copies at the repo root. Editing those does nothing. People have lost hours here.

**Trap #3:** if you're flashing firmware, WiFi credentials and the backend host are hardcoded plaintext constants at `compile/compile.ino` lines 42-44 (`WIFI_SSID`, `WIFI_PASS`, `SERVER_HOST`). You'll need to change these to your own network locally to flash — **do not commit your own credentials there.** This is committed to source as-is, which is a known issue (see `PROJECT_CONTEXT.md` §4.10); don't make it worse.

### 3. Look at the dashboard
Open the local dashboard, or the deployed one at `ai-glasses-for-research.fly.dev`. Two tabs: **Session** and **Diagnostics**. Click through both before touching code.

### 4. Read one real session
Ask Shan for the P01/P02 findings document. Read the failure cases. This will teach you more about the system in 30 minutes than reading `app_main.py` will in three hours.

---

## The mental model

```
Device captures  →  streams over 2 WebSockets  →  backend forwards to Gemini
     →  Gemini returns speech  →  backend streams it back  →  device plays it
```

Four places things break, and **naming the layer correctly is most of debugging**:

| Layer | Looks like | Fixed in |
|---|---|---|
| **Device/firmware** | Reboots, dropped sockets, heap exhaustion, I2C timeouts, audio glitches | `compile.ino` — *not* Python |
| **Transport** | Latency, dropped frames, reconnects | Backend WebSocket handlers |
| **Model** | Wrong/stale/irrelevant descriptions | Prompting, context management, architecture |
| **Research/data** | Metrics that don't mean what we think | Instrumentation + analysis |

**Ask "which layer?" before you touch anything.** A lot of time has been lost applying backend fixes to firmware problems.

---

## How we work

### Before you build
- **Read the actual code first.** Don't assume behavior from a filename or a doc. Several specs had errors that only surfaced by reading source — a parser that was compiled out, a variable nothing read, the wrong frontend file.
- **Confirm scope before producing anything.** If the task is ambiguous, ask. Don't guess and build.
- **Don't expand scope.** Do the assigned thing. If you spot something adjacent that's broken, write it down and raise it — don't fix it silently in the same change.

### Logs
**Never interpret a log without knowing what the person was doing during that capture window.** We once chased a "deterministic 153-second reconnection cycle" that was just someone manually restarting the service. Ask what was happening, *then* form a hypothesis.

### Committing
- Small, scoped commits with a clear message
- Note the build hash when a change affects a participant session — we've had two sessions we *think* ran the same build, which made a whole comparison uninterpretable
- Don't flash or deploy without checking with Shan

### The devlog
Every working session, add an entry to `DEVLOG.md`. Three minutes. Format's in the file. This exists because context evaporates and we've lost real time re-deriving things.

---

## Your first task

Ask Shan. But a good default first task is:

**Get the backend running locally, connect the dashboard, and write a `DEVLOG.md` entry describing exactly what you had to do — including anything in this doc that was wrong or missing.**

That's genuinely useful: it validates the setup instructions and gives us a fix for the next person.

---

## Glossary

| Term | Means |
|---|---|
| **Stream FPS** | Target *transmission* rate. Not capture FPS, not actual transmitted FPS. Say which one you mean. |
| **Stale grounding** | Model describes an old frame while fresh ones are arriving. Core failure mode. |
| **STABILITY_MODE** | Firmware `#define` that compiles out the `SET:` command parser |
| **P01, P02** | Participant sessions 1 and 2 |
| **BLV** | Blind and low-vision |
| **VISIONS** | Our partner nonprofit — VISIONS/Services for the Blind and Visually Impaired, NYC |
| **Topological (not metric)** | We aim to know *which place you're near*, not exact coordinates. Hardware can't do metric SLAM. |
| **`speech_end_to_first_i2s_ms`** | Our primary latency metric. Device-measured. The backend total overstates perceived wait ~5×. |

---

## Who to ask

| Topic | Person |
|---|---|
| Anything — start here | Shan (team lead) |
| Research direction, publication | Prof. Hao Tang |
| Participant sessions, VISIONS | Shan |

If you're blocked for more than a day, say so. Silent blockage is the single most expensive thing on a small team.
