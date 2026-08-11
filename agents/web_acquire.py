"""
Web acquisition: fetch openly-licensed training photos and open-source model
weights from known registries.

Deliberately narrow. Two sources, both of which publish machine-readable licence
metadata:

  - Wikimedia Commons for photographs, reusing the licence filtering and
    attribution recording already in scripts/fetch_training_images.py.
  - The Hugging Face Hub for model weights.

Two rules this module enforces, because "an agent that downloads things from the
internet" is otherwise an excellent way to get a machine compromised:

  1. It fetches data, never code, and never executes anything it fetches. Model
     downloads default to safetensors and config files. PyTorch .bin checkpoints
     are pickles that run arbitrary code at load time, so they are refused unless
     a caller explicitly opts in.
  2. Downloads are registry-scoped. There is no "fetch this URL" entry point, so
     a prompt cannot talk it into reaching an arbitrary host or a link-local
     metadata endpoint.

Anything a model or a web page says inside fetched content is data, not
instruction. Nothing here acts on the contents of what it downloads.
"""

import json
import logging
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

from . import sandbox

logger = logging.getLogger(__name__)

# Extensions that are inert to load: tensors and text. Everything else, notably
# .bin and .pt/.pth pickles and .py, needs an explicit opt-in.
SAFE_MODEL_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.md"]

# Refuse a model download projected to exceed this unless raised by the caller.
DEFAULT_MAX_MODEL_GB = 6.0

# Everything downloaded lands here first. Nothing else in the app reads from it,
# so arriving files cannot be picked up by the engine or the trainer until they
# are explicitly promoted.
QUARANTINE_ROOT = os.path.abspath("downloads/quarantine")

WORKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "sandboxed_fetch.py")


def _hf_transfer_available() -> bool:
    """True when the accelerated Hub downloader is installed."""
    import importlib.util
    return importlib.util.find_spec("hf_transfer") is not None


def _audit_containment(root: str) -> List[str]:
    """
    Paths under root that actually resolve outside it.

    Belt and braces behind the sandbox: a symlink or a ../ in a repo filename
    would show up here even on a platform where confinement is unavailable.
    """
    escaped = []
    for current, dirs, files in os.walk(root):
        for name in dirs + files:
            full = os.path.join(current, name)
            if not sandbox.is_contained(full, root):
                escaped.append(full)
    return escaped


def _run_worker(argv: List[str], write_root: str, timeout: int = 1800) -> Dict[str, Any]:
    """Run the download worker confined to write_root and parse its JSON result."""
    os.makedirs(write_root, exist_ok=True)
    code, out, err, confined = sandbox.run_sandboxed(
        [sys.executable, WORKER] + argv, write_root=write_root, timeout=timeout,
        # Keep the Hub's cache and temp files inside the writable root too,
        # otherwise the sandbox denies them and the download fails.
        env={
            "HF_HUB_CACHE": os.path.join(write_root, ".hf-cache"),
            # hf_transfer downloads each file in parallel chunks and is several
            # times faster on a fast link. Enabled only when installed, because
            # the flag makes huggingface_hub raise if the package is absent.
            **({"HF_HUB_ENABLE_HF_TRANSFER": "1"} if _hf_transfer_available() else {}),
        },
    )

    # The fetcher prints human-readable progress, so the result is the last line
    # of stdout that parses as JSON rather than the whole stream.
    payload = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
            break
        except ValueError:
            continue
    if payload is None:
        payload = {"success": False, "error": (
            f"worker produced no parsable result (exit {code}). "
            f"stdout tail: {out.strip()[-200:] or '(empty)'} stderr tail: {err.strip()[-200:] or '(empty)'}"
        )}

    escaped = _audit_containment(write_root)
    if escaped:
        logger.error("Download wrote outside the quarantine root: %s", escaped)
        payload["success"] = False
        payload["error"] = f"download escaped the quarantine root: {escaped}"

    payload["sandboxed"] = confined
    payload["quarantine_root"] = write_root
    if not confined:
        payload["warning"] = (
            "sandbox-exec unavailable on this platform, so the download ran unconfined; "
            "paths were audited afterwards but not restricted during the write."
        )
    if code != 0 and payload.get("success"):
        payload["success"] = False
        payload["error"] = payload.get("error") or f"worker exited {code}: {err.strip()[:300]}"
    return payload


def _hf_api():
    from huggingface_hub import HfApi
    return HfApi()


def search_images(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search Wikimedia Commons without downloading anything.

    Returns candidates with their licence and author so a caller can see what
    they would be getting, and on what terms, before committing to a download.
    """
    try:
        from scripts.fetch_training_images import _licence_allowed, _plain, search_commons
    except ImportError as e:  # pragma: no cover - path dependent
        return {"success": False, "error": f"image search unavailable: {e}"}

    try:
        pages = search_commons(query, limit)
    except Exception as e:
        return {"success": False, "error": f"Commons search failed: {e}"}

    results = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        licence = _plain(meta.get("LicenseShortName", {}).get("value"))
        results.append({
            "title": page.get("title"),
            "licence": licence,
            "licence_allowed": _licence_allowed(licence),
            "author": _plain(meta.get("Artist", {}).get("value")),
            "width": info.get("width"),
            "height": info.get("height"),
            "url": info.get("url"),
        })

    allowed = [r for r in results if r["licence_allowed"]]
    return {
        "success": True,
        "query": query,
        "found": len(results),
        "usable": len(allowed),
        "results": results,
    }


def download_images(
    subject: str,
    query: str,
    limit: int = 20,
    caption: Optional[str] = None,
    output_root: str = "static/training_data",
) -> Dict[str, Any]:
    """
    Download openly-licensed photos for one subject into the training tree.

    Delegates to the existing fetcher, which keeps only CC0/CC BY/CC BY-SA/public
    domain files above a minimum resolution and writes the author and licence of
    each into ATTRIBUTION.json, as those licences require.
    """
    if not subject or not subject.strip():
        return {"success": False, "error": "subject is required"}
    if not query or not query.strip():
        return {"success": False, "error": "query is required"}
    if os.sep in subject or subject.startswith("."):
        return {"success": False, "error": "subject must be a plain directory name"}

    root = os.path.join(QUARANTINE_ROOT, "images")
    logger.info("Downloading up to %d images for %r (query=%r) into quarantine", limit, subject, query)
    return _run_worker([
        "images",
        "--subject", subject,
        "--query", query,
        "--caption", caption or subject.replace("_", " "),
        "--limit", str(limit),
        "--output-root", root,
    ], write_root=root)


def search_models(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search the Hugging Face Hub. Metadata only; downloads nothing."""
    try:
        api = _hf_api()
        models = list(api.list_models(search=query, limit=limit))
    except Exception as e:
        return {"success": False, "error": f"Hub search failed: {e}"}

    return {
        "success": True,
        "query": query,
        "results": [{
            "id": m.id,
            "downloads": getattr(m, "downloads", None),
            "likes": getattr(m, "likes", None),
            "tags": (getattr(m, "tags", None) or [])[:8],
            # The Hub reports a licence tag only when the author set one. Absent
            # is reported as absent rather than assumed permissive.
            "licence": next((t.split("license:", 1)[1] for t in (getattr(m, "tags", None) or [])
                             if t.startswith("license:")), None),
        } for m in models],
    }


def inspect_model(repo_id: str) -> Dict[str, Any]:
    """
    List a repo's files and flag which are inert to load.

    Lets a caller see the download size and whether the repo ships pickles before
    committing to fetching it.
    """
    try:
        api = _hf_api()
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as e:
        return {"success": False, "repo_id": repo_id, "error": f"Hub lookup failed: {e}"}

    files = []
    for sibling in (info.siblings or []):
        name = sibling.rfilename
        size = getattr(sibling, "size", None)
        files.append({
            "name": name,
            "size_mb": round(size / (1024 * 1024), 1) if size else None,
            "executes_on_load": name.endswith((".bin", ".pt", ".pth", ".ckpt", ".py", ".pkl")),
        })

    total_mb = sum(f["size_mb"] or 0 for f in files)
    return {
        "success": True,
        "repo_id": repo_id,
        "licence": (info.card_data or {}).get("license") if info.card_data else None,
        "total_size_mb": round(total_mb, 1),
        "has_pickled_weights": any(f["executes_on_load"] for f in files),
        "files": files,
    }


def download_model(
    repo_id: str,
    target_dir: str = "checkpoints/downloaded",
    allow_unsafe: bool = False,
    max_gb: float = DEFAULT_MAX_MODEL_GB,
) -> Dict[str, Any]:
    """
    Download model weights from the Hugging Face Hub.

    Args:
        repo_id: Hub repo, e.g. "runwayml/stable-diffusion-v1-5".
        target_dir: Where to place the snapshot.
        allow_unsafe: Also fetch .bin/.pt/.ckpt pickles, which execute arbitrary
            code when loaded. Off by default; safetensors carry no such risk.
        max_gb: Refuse a download whose reported size exceeds this.

    The weights are written to disk and never loaded here, so nothing downloaded
    can run as a side effect of fetching it.
    """
    if not repo_id or "/" not in repo_id:
        return {"success": False, "error": "repo_id must look like 'owner/name'"}

    info = inspect_model(repo_id)
    if not info.get("success"):
        return info

    size_gb = (info["total_size_mb"] or 0) / 1024
    if size_gb > max_gb:
        return {
            "success": False,
            "repo_id": repo_id,
            "error": f"repo is {size_gb:.1f}GB, above the {max_gb}GB limit; raise max_gb to proceed",
        }

    root = os.path.join(QUARANTINE_ROOT, "models")
    argv = ["model", "--repo-id", repo_id, "--output-root", root]
    if allow_unsafe:
        argv.append("--allow-unsafe")

    logger.info("Downloading %s into quarantine (allow_unsafe=%s)", repo_id, allow_unsafe)
    result = _run_worker(argv, write_root=root)
    if not result.get("success"):
        return {"repo_id": repo_id, "licence": info.get("licence"), **result}

    path = result.get("path") or os.path.join(root, repo_id.replace("/", "__"))
    files = []
    for current, _dirs, names in os.walk(path):
        for name in names:
            full = os.path.join(current, name)
            files.append({"name": os.path.relpath(full, path),
                          "size_mb": round(os.path.getsize(full) / (1024 * 1024), 1)})

    return {
        **result,
        "repo_id": repo_id,
        "path": path,
        "licence": info.get("licence"),
        "files": files,
        "downloaded_mb": round(sum(f["size_mb"] for f in files), 1),
        "pickles_skipped": bool(info.get("has_pickled_weights")) and not allow_unsafe,
        "note": ("Written to quarantine and not loaded. Nothing downloaded has been executed. "
                 "Call promote() to move it somewhere the app will use."),
    }


def promote(relative_path: str, destination_root: str = "checkpoints") -> Dict[str, Any]:
    """
    Move a vetted download out of quarantine into a directory the app reads.

    Quarantine is only meaningful if leaving it is a deliberate act, so this is a
    separate call rather than something a download does for itself. Both ends are
    checked for containment, so neither a crafted relative_path nor a symlink can
    place files somewhere else.
    """
    source = os.path.join(QUARANTINE_ROOT, relative_path)
    if not sandbox.is_contained(source, QUARANTINE_ROOT):
        return {"success": False, "error": "path is outside the quarantine root"}
    if not os.path.exists(source):
        return {"success": False, "error": f"nothing at {relative_path} in quarantine"}

    destination_root = os.path.abspath(destination_root)
    destination = os.path.join(destination_root, os.path.basename(source.rstrip(os.sep)))
    if not sandbox.is_contained(destination, destination_root):
        return {"success": False, "error": "destination is outside the destination root"}
    if os.path.exists(destination):
        return {"success": False, "error": f"{destination} already exists; remove it first"}

    os.makedirs(destination_root, exist_ok=True)
    shutil.move(source, destination)
    logger.info("Promoted %s out of quarantine to %s", relative_path, destination)
    return {"success": True, "from": source, "to": destination}


def list_quarantine() -> Dict[str, Any]:
    """What is sitting in quarantine, and how big it is."""
    entries = []
    for kind in ("images", "models"):
        root = os.path.join(QUARANTINE_ROOT, kind)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if name.startswith("."):
                continue
            full = os.path.join(root, name)
            size = sum(os.path.getsize(os.path.join(c, f))
                       for c, _d, fs in os.walk(full) for f in fs) if os.path.isdir(full) else os.path.getsize(full)
            entries.append({
                "relative_path": os.path.join(kind, name),
                "size_mb": round(size / (1024 * 1024), 1),
            })
    return {"success": True, "quarantine_root": QUARANTINE_ROOT, "entries": entries}
