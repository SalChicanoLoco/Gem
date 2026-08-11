#!/usr/bin/env python3
"""
Download worker, intended to be run under agents.sandbox.run_sandboxed.

Kept as a separate process on purpose: the confinement applies to a process, so
the code that touches the network and writes files has to be the thing that is
confined. It prints a single JSON object on stdout and nothing else, so the
caller can parse a result even when a library has written noise to stderr.

It downloads and never imports or loads what it downloaded.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAFE_MODEL_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.md"]


def fetch_images(args) -> dict:
    from scripts.fetch_training_images import fetch_subject
    result = fetch_subject(
        subject=args.subject,
        query=args.query,
        caption=args.caption or args.subject.replace("_", " "),
        limit=args.limit,
        output_root=args.output_root,
    )
    return {"success": True, "kind": "images", "subject": args.subject, **(result or {})}


def fetch_model(args) -> dict:
    from huggingface_hub import snapshot_download
    destination = os.path.join(args.output_root, args.repo_id.replace("/", "__"))
    os.makedirs(destination, exist_ok=True)
    # The Hub client defaults to 8 workers but shards are large and latency-bound,
    # so more parallel connections is the single biggest win on a fast link.
    # hf_transfer (Rust, multi-part per file) is used automatically when present.
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=destination,
        allow_patterns=None if args.allow_unsafe else SAFE_MODEL_PATTERNS,
        max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "16")),
    )
    return {"success": True, "kind": "model", "repo_id": args.repo_id, "path": path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)

    images = sub.add_parser("images")
    images.add_argument("--subject", required=True)
    images.add_argument("--query", required=True)
    images.add_argument("--caption")
    images.add_argument("--limit", type=int, default=20)
    images.add_argument("--output-root", required=True)

    model = sub.add_parser("model")
    model.add_argument("--repo-id", required=True)
    model.add_argument("--output-root", required=True)
    model.add_argument("--allow-unsafe", action="store_true")

    args = parser.parse_args()
    try:
        payload = fetch_images(args) if args.kind == "images" else fetch_model(args)
    except Exception as e:
        payload = {"success": False, "kind": args.kind, "error": f"{type(e).__name__}: {e}"}

    sys.stdout.write(json.dumps(payload))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
