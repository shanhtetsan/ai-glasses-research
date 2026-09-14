# Development Log

Append-only. Newest at the top. **Three minutes per entry, not thirty.**

## Why this exists
The last log died because it was too heavy to maintain. This one is deliberately small. The point isn't documentation — it's that in six weeks nobody remembers why a setting is what it is, and re-deriving it costs hours.

## Rules
- **One entry per working session.** Not per day, not per commit.
- If you did nothing, write nothing. Gaps are fine and honest.
- Anyone on the team writes entries, not just the lead.
- **Never delete or rewrite past entries.** If something was wrong, add a new entry saying so.
- Bullet fragments beat prose. Nobody is grading this.

## Template — copy this

```markdown
### YYYY-MM-DD — [Name] — [3-6 word title]

**Did:**
-

**Found:**
-

**Next:**
-

**Blocked:** (delete if not)
-
```

Field meanings:
- **Did** — what you actually changed or ran.
- **Found** — surprises, bugs, things that contradicted an assumption. *This is the field that pays for itself later.*
- **Next** — where you'd pick up. Write it for a version of yourself who's forgotten everything.
- **Blocked** — anything you need from another person. If this field is filled, also say it out loud in the channel. The log is not an escalation path.

---

## Entries

### YYYY-MM-DD — [Name] — Example entry, delete me

**Did:**
- Read `apply_framesize()` in `compile.ino` checking for the shared-mutex question
- Ran backend locally against the deployed dashboard

**Found:**
- Root `main.js` is orphaned — spent 40 min editing it before realizing `static/main.js` is live
- `/api/camera` returns 200 but does nothing under `STABILITY_MODE` (parser compiled out)

**Next:**
- Confirm whether SCCB shares the I2C bus with MLX90640
- Then decide XGA fall-through vs. retire

**Blocked:**
- Need Fly.io access to check production logs
