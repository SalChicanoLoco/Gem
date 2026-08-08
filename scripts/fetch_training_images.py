#!/usr/bin/env python3
"""
Training image fetcher (scripts/fetch_training_images.py)

Collects real photographs from Wikimedia Commons for LoRA training, keeping only
openly-licensed files and recording the attribution each licence requires.

Why Commons: every file carries machine-readable licence and author metadata, so
the resulting dataset can state where each image came from. Images are written
alongside a .txt caption sidecar (the format agents/lora_trainer.py expects) and
an ATTRIBUTION.json per subject.

Usage:
    python scripts/fetch_training_images.py --subject roadrunner \\
        --query "Geococcyx californianus" --limit 25

    python scripts/fetch_training_images.py --all
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

USER_AGENT = (
    "SenaAIgent-TrainingFetch/0.1 "
    "(local model training; https://github.com/SalChicanoLoco/Gem)"
)

# Licences that permit reuse and redistribution. Anything else is skipped.
ALLOWED_LICENCE_PATTERNS = (
    re.compile(r"^cc0", re.I),
    re.compile(r"^cc[ -]by([ -]sa)?[ -]?\d", re.I),
    re.compile(r"public domain", re.I),
)

MIN_WIDTH = 640
MIN_HEIGHT = 640
DOWNLOAD_WIDTH = 1024

# Subjects worth training on. Portrait/person categories are deliberately absent;
# see the note in the module README and the project discussion.
DEFAULT_SUBJECTS: Dict[str, Dict[str, str]] = {
    "roadrunner": {
        "query": "Geococcyx californianus",
        "caption": "a greater roadrunner, Geococcyx californianus",
    },
    "roadrunner_desert": {
        "query": "Geococcyx californianus desert",
        "caption": "a greater roadrunner in desert scrub habitat",
    },
    "rainbow_trout": {
        "query": "Oncorhynchus mykiss",
        "caption": "a rainbow trout, Oncorhynchus mykiss",
    },
    "sports_car": {
        "query": "sports car photograph",
        "caption": "a sports car",
    },
    "new_mexico_landscape": {
        "query": "New Mexico desert landscape",
        "caption": "a New Mexico desert landscape",
    },
}


def _licence_allowed(licence: str) -> bool:
    return any(pattern.search(licence or "") for pattern in ALLOWED_LICENCE_PATTERNS)


def _plain(value: Optional[str]) -> str:
    """Strip the HTML Commons embeds in its metadata fields."""
    if not value:
        return ""
    return re.sub(r"<[^>]+>", "", value).strip()


def search_commons(query: str, limit: int) -> List[Dict[str, Any]]:
    """Search Commons for bitmap files matching a query."""
    params = {
        "action": "query",
        "generator": "search",
        # Ask for more than needed; licence and size filters discard some.
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": "6",
        "gsrlimit": str(min(limit * 3, 200)),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": str(DOWNLOAD_WIDTH),
        "format": "json",
    }
    resp = requests.get(COMMONS_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    return list(pages.values())


def fetch_subject(
    subject: str,
    query: str,
    caption: str,
    limit: int,
    output_root: str,
) -> Dict[str, Any]:
    """Download openly-licensed photos for one subject."""
    target_dir = os.path.join(output_root, subject)
    os.makedirs(target_dir, exist_ok=True)

    try:
        candidates = search_commons(query, limit)
    except requests.RequestException as e:
        return {"subject": subject, "downloaded": 0, "error": f"search failed: {e}"}

    attribution: List[Dict[str, str]] = []
    attribution_path = os.path.join(target_dir, "ATTRIBUTION.json")
    if os.path.exists(attribution_path):
        try:
            with open(attribution_path) as f:
                attribution = json.load(f)
        except (OSError, json.JSONDecodeError):
            attribution = []

    already = {entry["source_file"] for entry in attribution}
    downloaded, skipped_licence, skipped_small = 0, 0, 0

    for page in candidates:
        if downloaded >= limit:
            break

        info = page.get("imageinfo", [{}])[0]
        meta = info.get("extmetadata", {})
        title = page.get("title", "")

        if title in already:
            continue
        if not str(info.get("mime", "")).startswith("image/"):
            continue

        licence = _plain(meta.get("LicenseShortName", {}).get("value"))
        if not _licence_allowed(licence):
            skipped_licence += 1
            continue

        if info.get("width", 0) < MIN_WIDTH or info.get("height", 0) < MIN_HEIGHT:
            skipped_small += 1
            continue

        url = info.get("thumburl") or info.get("url")
        if not url:
            continue

        # Truncating long Commons titles makes distinct files collide, so append a
        # digest of the full title to keep names unique.
        stem = re.sub(r"[^A-Za-z0-9]+", "_", title.replace("File:", "")).strip("_")[:60]
        digest = hashlib.md5(title.encode()).hexdigest()[:8]
        safe = f"{stem}_{digest}"
        ext = os.path.splitext(url)[1].lower() or ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"
        image_path = os.path.join(target_dir, f"{safe}{ext}")

        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
            r.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(r.content)
        except requests.RequestException as e:
            print(f"   ! failed {title}: {e}", file=sys.stderr)
            continue

        # Caption sidecar in the format lora_trainer.discover_training_pairs reads.
        with open(os.path.splitext(image_path)[0] + ".txt", "w") as f:
            f.write(caption)

        attribution.append({
            "file": os.path.basename(image_path),
            "source_file": title,
            "source_page": _plain(info.get("descriptionurl")) or info.get("descriptionurl", ""),
            "author": _plain(meta.get("Artist", {}).get("value")) or "unknown",
            "licence": licence,
            "credit": _plain(meta.get("Credit", {}).get("value")),
        })
        downloaded += 1
        print(f"   + {os.path.basename(image_path)}  [{licence}]")

        # Be polite to the API between downloads.
        time.sleep(0.3)

    with open(attribution_path, "w") as f:
        json.dump(attribution, f, indent=2)

    return {
        "subject": subject,
        "downloaded": downloaded,
        "total_in_dir": len(attribution),
        "skipped_licence": skipped_licence,
        "skipped_too_small": skipped_small,
        "dir": target_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", help="Subject directory name, e.g. roadrunner")
    parser.add_argument("--query", help="Commons search query")
    parser.add_argument("--caption", help="Caption written to each .txt sidecar")
    parser.add_argument("--limit", type=int, default=20, help="Images per subject (default 20)")
    parser.add_argument("--all", action="store_true", help="Fetch every default subject")
    parser.add_argument("--output-root", default="static/training_data")
    args = parser.parse_args()

    if args.all:
        targets = DEFAULT_SUBJECTS
    elif args.subject:
        if not args.query:
            parser.error("--query is required with --subject")
        targets = {args.subject: {"query": args.query, "caption": args.caption or args.subject.replace("_", " ")}}
    else:
        parser.error("pass --all or --subject with --query")

    results = []
    for subject, spec in targets.items():
        print(f"\n[{subject}] searching Commons for {spec['query']!r}...")
        result = fetch_subject(
            subject=subject,
            query=spec["query"],
            caption=spec["caption"],
            limit=args.limit,
            output_root=args.output_root,
        )
        results.append(result)
        print(f"   -> {result['downloaded']} new, {result.get('total_in_dir', 0)} total in {result['dir']}")

    print("\n" + "=" * 60)
    total = sum(r["downloaded"] for r in results)
    print(f"Downloaded {total} openly-licensed images across {len(results)} subjects.")
    print("Attribution recorded per subject in ATTRIBUTION.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
