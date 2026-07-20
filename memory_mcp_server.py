"""
Nexus Protocol - Pipeline Memory MCP Server
Runs on the Windows PC alongside mcp_server.py. Actioned from either machine.

Stores durable verdicts about what produced good and bad results - the
machine-readable version of the decision ledgers the pipeline skills require.
Agents record a verdict after review and query before configuring a run, so a
rejected approach is not retried and an approved baseline is not regressed.

Memories are one markdown file each under Instructions/Memory/entries/ with an
INDEX.md digest regenerated on every write. They are plain files: reviewable,
greppable, and diffable in git like the rest of Instructions/.

Transport: streamable-HTTP (works across machines; stdio would be same-host only).

Run on Windows:
    python memory_mcp_server.py
    # serves at http://0.0.0.0:8766/mcp

SECURITY: this binds to your LAN. Keep it on a trusted private network. The token
check below is a minimal gate, not real auth - do NOT expose this to the internet
or port-forward it. For remote access use a VPN / Tailscale, not a public port.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# --- Paths (Windows) ----------------------------------------------------
ROOT = Path(__file__).resolve().parent
MEMORY_DIR = ROOT / "Instructions" / "Memory"
ENTRY_DIR = MEMORY_DIR / "entries"
INDEX = MEMORY_DIR / "INDEX.md"

ENTRY_DIR.mkdir(parents=True, exist_ok=True)

STAGES = {"raster", "geometry", "texture", "postprocess", "validation", "pipeline"}
VERDICTS = {"good", "bad"}

# Minimal shared-secret gate. Set the same value as an env var on both machines.
EXPECTED_TOKEN = os.environ.get("NEXUS_MCP_TOKEN", "")

mcp = FastMCP("nexus-pipeline-memory", host="0.0.0.0", port=8766)


def _auth(token: str) -> None:
    if EXPECTED_TOKEN and token != EXPECTED_TOKEN:
        raise PermissionError("Bad or missing NEXUS_MCP_TOKEN.")


def _slug(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "memory"


def _entry_path(memory_id: str) -> Path | None:
    matches = list(ENTRY_DIR.glob(f"{memory_id}_*.md"))
    return matches[0] if matches else None


def _parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    head, _, body = text.partition("\n---\n")
    meta = {}
    for line in head.splitlines():
        if ":" in line and not line.startswith("---"):
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    meta["tags"] = [t.strip() for t in meta.get("tags", "").split(",") if t.strip()]
    meta["body"] = body.strip()
    meta["file"] = path.name
    return meta


def _all_entries() -> list[dict]:
    entries = [_parse(p) for p in sorted(ENTRY_DIR.glob("*.md"))]
    entries.sort(key=lambda e: e.get("created", ""), reverse=True)
    return entries


def _rebuild_index() -> None:
    lines = [
        "# Pipeline Memory Index",
        "",
        "Machine-written by memory_mcp_server.py - do not edit by hand.",
        "One line per memory, newest first. Full entries live in entries/.",
        "",
    ]
    for e in _all_entries():
        flag = "DEPRECATED " if e.get("status") == "deprecated" else ""
        first = e["body"].splitlines()[0] if e["body"] else ""
        lines.append(
            f"- **{e.get('verdict', '?')}/{e.get('stage', '?')}** {flag}{first} "
            f"(`{e.get('id', '?')}`, asset: {e.get('asset') or '-'})"
        )
    INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


@mcp.tool()
def memory_record(verdict: str, statement: str, why: str, stage: str,
                  asset: str = "", tags: str = "", how_to_apply: str = "",
                  source: str = "", token: str = "") -> str:
    """
    Record one durable learning about what worked or failed.

    verdict:      'good' (approach to keep) or 'bad' (approach to avoid).
    statement:    one-sentence fact, concrete and testable. Write it so a future
                  run can act on it without extra context.
    why:          the observed evidence (what was seen/measured, not a guess).
    stage:        raster | geometry | texture | postprocess | validation | pipeline.
    asset:        asset id it was observed on, if any (e.g. ford_torneo_black_v2).
    tags:         comma-separated keywords for retrieval.
    how_to_apply: what a future run should do differently (optional but preferred).
    source:       run/log/report the evidence lives in.
    Returns JSON: {id, file}.
    """
    _auth(token)
    verdict = verdict.lower()
    stage = stage.lower()
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of: {sorted(VERDICTS)}")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of: {sorted(STAGES)}")
    if not statement.strip() or not why.strip():
        raise ValueError("statement and why are both required")

    memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tag_list = ", ".join(t.strip() for t in tags.split(",") if t.strip())
    body = statement.strip()
    body += f"\n\n**Why:** {why.strip()}"
    if how_to_apply.strip():
        body += f"\n\n**How to apply:** {how_to_apply.strip()}"
    path = ENTRY_DIR / f"{memory_id}_{_slug(statement)}.md"
    path.write_text(
        "---\n"
        f"id: {memory_id}\n"
        f"created: {created}\n"
        f"verdict: {verdict}\n"
        f"stage: {stage}\n"
        f"asset: {asset.strip()}\n"
        f"tags: {tag_list}\n"
        f"status: active\n"
        f"source: {source.strip()}\n"
        "---\n"
        f"{body}\n",
        encoding="utf-8",
    )
    _rebuild_index()
    return json.dumps({"id": memory_id, "file": str(path)})


@mcp.tool()
def memory_query(text: str = "", stage: str = "", verdict: str = "",
                 asset: str = "", tag: str = "", include_deprecated: bool = False,
                 limit: int = 10, token: str = "") -> str:
    """
    Search memories before configuring a run. Filters are ANDed; text is a
    keyword match over statement/why/tags/asset, ranked by hits. Empty query
    returns the newest entries.
    Returns JSON: {count, memories: [{id, verdict, stage, asset, tags, body, ...}]}.
    """
    _auth(token)
    words = [w for w in re.split(r"\W+", text.lower()) if w]
    scored = []
    for e in _all_entries():
        if not include_deprecated and e.get("status") == "deprecated":
            continue
        if stage and e.get("stage") != stage.lower():
            continue
        if verdict and e.get("verdict") != verdict.lower():
            continue
        if asset and asset.lower() not in e.get("asset", "").lower():
            continue
        if tag and tag.lower() not in [t.lower() for t in e["tags"]]:
            continue
        haystack = " ".join([e["body"], " ".join(e["tags"]), e.get("asset", "")]).lower()
        score = sum(haystack.count(w) for w in words)
        if words and score == 0:
            continue
        scored.append((score, e))
    scored.sort(key=lambda item: (item[0], item[1].get("created", "")), reverse=True)
    memories = [e for _, e in scored[: max(1, limit)]]
    return json.dumps({"count": len(memories), "memories": memories})


@mcp.tool()
def memory_get(memory_id: str, token: str = "") -> str:
    """Fetch one memory by id, including deprecated ones."""
    _auth(token)
    path = _entry_path(memory_id)
    if path is None:
        return json.dumps({"error": f"not found: {memory_id}"})
    return json.dumps(_parse(path))


@mcp.tool()
def memory_deprecate(memory_id: str, reason: str, token: str = "") -> str:
    """
    Mark a memory as no longer valid (kept on disk - the decision record
    survives, per the rejection-retention rule; queries skip it by default).
    """
    _auth(token)
    if not reason.strip():
        raise ValueError("reason is required")
    path = _entry_path(memory_id)
    if path is None:
        return json.dumps({"error": f"not found: {memory_id}"})
    text = path.read_text(encoding="utf-8")
    if "status: deprecated" in text:
        return json.dumps({"id": memory_id, "status": "already deprecated"})
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    text = text.replace("status: active\n", "status: deprecated\n", 1)
    text += f"\n**Deprecated {stamp}:** {reason.strip()}\n"
    path.write_text(text, encoding="utf-8")
    _rebuild_index()
    return json.dumps({"id": memory_id, "status": "deprecated"})


@mcp.tool()
def memory_stats(token: str = "") -> str:
    """Counts by verdict/stage plus the index path - a quick health check."""
    _auth(token)
    entries = _all_entries()
    by_stage: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    for e in entries:
        if e.get("status") == "deprecated":
            continue
        by_stage[e.get("stage", "?")] = by_stage.get(e.get("stage", "?"), 0) + 1
        by_verdict[e.get("verdict", "?")] = by_verdict.get(e.get("verdict", "?"), 0) + 1
    return json.dumps({
        "total": len(entries),
        "active_by_verdict": by_verdict,
        "active_by_stage": by_stage,
        "index": str(INDEX),
    })


if __name__ == "__main__":
    # Streamable-HTTP so the laptop can reach it over the LAN.
    mcp.run(transport="streamable-http")
