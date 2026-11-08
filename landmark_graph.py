"""Landmark graph: a small map of the fixed things at one test site.

A tester makes it on the /test page in map mode: stand at a spot, face a
landmark (a door, the stairs, an exit), name it, and the page records the
phone's compass bearing. Any place works — a home hallway is a site as much as
a BMCC corridor. One JSON file per site under GRAPH_ROOT (/data/graphs).

When a trial runs with the `landmark_graph` arm set to a site, every question
also sends Gemini that site's map as text (context_text), with the direction
the tester is facing right now. The graph holds fixed things only; the camera
is still the authority on anything that moved, and the text says so.

What it does not do: know where the tester is. The phone gives heading, not
position, so bearings are described from the spot they were measured at.

File format (version 1):
    {"site": "home", "version": 1, "updated_at": "<iso>",
     "landmarks": [{"id": "front-door", "name": "front door", "kind": "door",
                    "from": "start", "bearing_deg": 85.0, "steps": 6, "note": ""}]}
`from` is "start" or the id of an earlier landmark (measured standing there).
`steps` is the tester's own count and is kept for later work; it is never sent
to the model, because the scorer marks any distance in an answer as a bad guess.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

GRAPH_ROOT = os.getenv("LANDMARK_GRAPH_ROOT", "/data/graphs")
VERSION = 1
SITE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
KINDS = ("door", "stairs", "exit", "elevator", "hallway", "room", "turn", "other")
START = "start"
MAX_LANDMARKS = 50
MAX_NAME = 60
MAX_NOTE = 200
MAX_STEPS = 500


class GraphError(ValueError):
    """A graph that cannot be saved or used, with the reason in plain words."""


def site_path(site: str, root: Optional[str] = None) -> str:
    if not isinstance(site, str) or not SITE_PATTERN.match(site):
        raise GraphError("site must be 1-32 of a-z 0-9 '-', starting with a letter or digit")
    return os.path.join(root or GRAPH_ROOT, f"{site}.json")


def validate(site: str, graph: Any) -> Dict[str, Any]:
    """A clean copy of `graph`, or GraphError. Unknown fields are dropped."""
    site_path(site)
    if not isinstance(graph, dict):
        raise GraphError("graph must be a JSON object")
    landmarks = graph.get("landmarks")
    if not isinstance(landmarks, list) or not landmarks:
        raise GraphError("graph needs at least one landmark")
    if len(landmarks) > MAX_LANDMARKS:
        raise GraphError(f"at most {MAX_LANDMARKS} landmarks")
    clean: List[Dict[str, Any]] = []
    seen = {START}
    for index, item in enumerate(landmarks, start=1):
        where = f"landmark {index}"
        if not isinstance(item, dict):
            raise GraphError(f"{where} must be an object")
        landmark_id = item.get("id")
        if not isinstance(landmark_id, str) or not ID_PATTERN.match(landmark_id):
            raise GraphError(f"{where}: id must be 1-40 of a-z 0-9 '-'")
        if landmark_id in seen:
            raise GraphError(f"{where}: id {landmark_id!r} is used twice (or is 'start')")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > MAX_NAME:
            raise GraphError(f"{where}: name must be 1-{MAX_NAME} characters")
        kind = item.get("kind", "other")
        if kind not in KINDS:
            raise GraphError(f"{where}: kind must be one of {', '.join(KINDS)}")
        origin = item.get("from", START)
        if origin not in seen:
            raise GraphError(f"{where}: 'from' must be 'start' or an earlier landmark id")
        bearing = item.get("bearing_deg")
        if isinstance(bearing, bool) or not isinstance(bearing, (int, float)) \
                or not (0.0 <= float(bearing) < 360.0):
            raise GraphError(f"{where}: bearing_deg must be a number in [0, 360)")
        steps = item.get("steps")
        if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int)
                                  or not 0 <= steps <= MAX_STEPS):
            raise GraphError(f"{where}: steps must be a whole number 0-{MAX_STEPS} or empty")
        note = item.get("note") or ""
        if not isinstance(note, str) or len(note) > MAX_NOTE:
            raise GraphError(f"{where}: note must be text, at most {MAX_NOTE} characters")
        clean.append({
            "id": landmark_id, "name": name.strip(), "kind": kind, "from": origin,
            "bearing_deg": round(float(bearing), 1), "steps": steps, "note": note.strip(),
        })
        seen.add(landmark_id)
    return {"site": site, "version": VERSION, "landmarks": clean}


def save(site: str, graph: Any, root: Optional[str] = None) -> Dict[str, Any]:
    """Validate and write atomically. Returns what was stored, with its sha256."""
    clean = validate(site, graph)
    clean["updated_at"] = datetime.now(timezone.utc).isoformat()
    path = site_path(site, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{site}-", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(clean, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return {**clean, "sha256": graph_sha256(clean)}


def load(site: str, root: Optional[str] = None) -> Dict[str, Any]:
    path = site_path(site, root)
    try:
        with open(path, encoding="utf-8") as handle:
            stored = json.load(handle)
    except FileNotFoundError:
        raise GraphError(f"no graph named {site!r}; make one in map mode first") from None
    except (OSError, ValueError) as exc:
        raise GraphError(f"graph {site!r} cannot be read: {exc}") from None
    clean = validate(site, stored)
    clean["updated_at"] = stored.get("updated_at")
    return {**clean, "sha256": graph_sha256(clean)}


def list_sites(root: Optional[str] = None) -> List[Dict[str, Any]]:
    root = root or GRAPH_ROOT
    if not os.path.isdir(root):
        return []
    sites = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json") or name.startswith("."):
            continue
        site = name[:-5]
        try:
            graph = load(site, root)
        except GraphError:
            continue
        sites.append({"site": site, "landmarks": len(graph["landmarks"]),
                      "updated_at": graph.get("updated_at"), "sha256": graph["sha256"]})
    return sites


def graph_sha256(graph: Dict[str, Any]) -> str:
    """Hash of the map content only (not updated_at), so a trial names the exact map."""
    body = {"site": graph["site"], "version": graph["version"], "landmarks": graph["landmarks"]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_site(value: Any) -> Optional[str]:
    """Config parser for the landmark_graph arm: a site that has a saved graph."""
    if value is None or value == "" or value is False:
        return None
    if not isinstance(value, str):
        raise ValueError("landmark_graph must be a site name")
    site = value.strip().lower()
    load(site)          # raises GraphError (a ValueError) if it does not exist
    return site


def relative_direction(bearing_deg: float, heading_deg: float) -> str:
    """Where `bearing_deg` is for someone facing `heading_deg`, as a clock direction."""
    delta = (bearing_deg - heading_deg + 540.0) % 360.0 - 180.0   # (-180, 180]
    hour = round(delta / 30.0) % 12
    clock = 12 if hour == 0 else hour
    if abs(delta) <= 20:
        side = "straight ahead"
    elif abs(delta) >= 160:
        side = "behind"
    elif delta > 0:
        side = "to the right"
    else:
        side = "to the left"
    return f"{side} ({clock} o'clock)"


def context_text(graph: Dict[str, Any], heading_deg: Optional[float]) -> str:
    """The map as Gemini reads it for one question.

    Ordered as the tester recorded them (start first). Step counts are left
    out on purpose, see the module docstring.
    """
    names = {item["id"]: item["name"] for item in graph["landmarks"]}
    lines = [
        "LANDMARK_MAP — a map of fixed things at this site, made earlier by a sighted tester.",
        "Use it to say which way to go. It can be out of date: if the camera shows something "
        "different, trust the camera and say the map may be wrong. It does not show people or "
        "moved objects. It does not know where the user is now.",
        "Never state a distance, a number of steps, or any measurement.",
    ]
    if heading_deg is None:
        lines.append("The user's current facing direction is unknown, so give no left/right "
                     "from the map alone.")
    else:
        lines.append(f"The user is facing compass {round(heading_deg):d}° now.")
    for item in graph["landmarks"]:
        origin = "the start spot" if item["from"] == START else f"the {names[item['from']]}"
        line = (f"- {item['name']} ({item['kind']}): from {origin}, compass "
                f"{round(item['bearing_deg']):d}°")
        if heading_deg is not None:
            line += (f"; if the user were standing at {origin}, that is "
                     f"{relative_direction(item['bearing_deg'], heading_deg)}")
        if item["note"]:
            line += f". Note: {item['note']}"
        lines.append(line + ".")
    return "\n".join(lines)
