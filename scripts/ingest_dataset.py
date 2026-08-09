#!/usr/bin/env python3
"""
HuggingFace dataset ingestor (scripts/ingest_dataset.py)

Streams an image dataset from the HuggingFace Hub and writes real photographs
plus caption sidecars into static/training_data/<subject>/, in the layout
agents/lora_trainer.py reads.

Schema note: bghira/photo-concept-bucket has no `image` column. Each row carries
a `url` (Pexels) and a `cogvlm_caption`. A previous version of this script called
row["image"], which is always None, so it silently downloaded nothing. This
version fetches the URL and pairs it with the real caption, and it fails loudly
if the columns it needs are absent.

Usage:
    # Everything, unfiltered
    python scripts/ingest_dataset.py --subject photo_concept --limit 200

    # Only rows whose caption/tags mention a keyword
    python scripts/ingest_dataset.py --subject sports_car --match "sports car,supercar" --limit 60
    python scripts/ingest_dataset.py --subject desert --match "desert,arid,dune" --limit 60
"""

import argparse
import io
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

DEFAULT_DATASET = "bghira/photo-concept-bucket"

# Columns this script relies on, checked up front rather than assumed.
URL_COLUMNS = ("url", "image_url")
CAPTION_COLUMNS = ("cogvlm_caption", "caption", "title", "alt")
# Fields searched by --match.
MATCH_COLUMNS = ("cogvlm_caption", "caption", "title", "alt", "tags", "class_label", "slug")

MAX_EDGE = 1024
MIN_EDGE = 512
USER_AGENT = "SenaAIgent-DatasetIngest/0.1 (local model training)"


def _first_present(row: Dict[str, Any], names) -> Optional[str]:
    for name in names:
        value = row.get(name)
        if value:
            return name
    return None


def _row_matches(row: Dict[str, Any], keywords: List[str]) -> bool:
    if not keywords:
        return True
    haystack = " ".join(str(row.get(col, "")) for col in MATCH_COLUMNS).lower()
    return any(kw in haystack for kw in keywords)


def ingest(
    dataset_name: str,
    subject: str,
    limit: int,
    keywords: List[str],
    output_root: str,
    max_scanned: int,
) -> Dict[str, Any]:
    """Stream a dataset and save matching images with caption sidecars."""
    from datasets import load_dataset
    from PIL import Image

    target_dir = os.path.join(output_root, subject)
    os.makedirs(target_dir, exist_ok=True)

    print(f"Streaming {dataset_name!r} -> {target_dir}")
    if keywords:
        print(f"Filtering on: {', '.join(keywords)}")

    stream = load_dataset(dataset_name, split="train", streaming=True)
    iterator = iter(stream)

    try:
        probe = next(iterator)
    except StopIteration:
        return {"success": False, "error": "dataset stream was empty", "saved": 0}

    url_col = _first_present(probe, URL_COLUMNS)
    caption_col = _first_present(probe, CAPTION_COLUMNS)
    if not url_col:
        return {
            "success": False,
            "saved": 0,
            "error": (
                f"{dataset_name} exposes no image URL column "
                f"(looked for {URL_COLUMNS}); columns are {list(probe.keys())}"
            ),
        }
    print(f"Using url column {url_col!r}, caption column {caption_col!r}")

    provenance: List[Dict[str, str]] = []
    provenance_path = os.path.join(target_dir, "PROVENANCE.json")
    if os.path.exists(provenance_path):
        try:
            with open(provenance_path) as f:
                provenance = json.load(f)
        except (OSError, json.JSONDecodeError):
            provenance = []
    seen_urls = {entry["source_url"] for entry in provenance}

    saved = 0
    scanned = 0
    skipped_small = 0
    failed = 0
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    def handle(row: Dict[str, Any]) -> bool:
        """Save one row. Returns True if an image was written."""
        nonlocal saved, skipped_small, failed

        url = row.get(url_col)
        if not url or url in seen_urls:
            return False
        if not _row_matches(row, keywords):
            return False

        caption = str(row.get(caption_col) or subject.replace("_", " ")).strip()
        caption = " ".join(caption.split())[:300]

        try:
            resp = session.get(url, timeout=45)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except (requests.RequestException, OSError) as e:
            failed += 1
            print(f"   ! {str(e)[:70]}", file=sys.stderr)
            return False

        if min(image.size) < MIN_EDGE:
            skipped_small += 1
            return False

        image.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
        stem = f"{subject}_{saved + 1:04d}"
        image_path = os.path.join(target_dir, f"{stem}.jpg")
        image.save(image_path, "JPEG", quality=92)

        with open(os.path.join(target_dir, f"{stem}.txt"), "w") as f:
            f.write(caption)

        seen_urls.add(url)
        provenance.append({
            "file": os.path.basename(image_path),
            "source_url": url,
            "dataset": dataset_name,
            "caption": caption,
        })
        saved += 1
        print(f"   + [{saved}/{limit}] {stem}.jpg  {caption[:60]}")
        return True

    handle(probe)
    scanned += 1

    for row in iterator:
        if saved >= limit or scanned >= max_scanned:
            break
        scanned += 1
        handle(row)
        if scanned % 200 == 0:
            print(f"   ... scanned {scanned}, saved {saved}")
        time.sleep(0.02)

    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)

    return {
        "success": saved > 0,
        "saved": saved,
        "scanned": scanned,
        "skipped_too_small": skipped_small,
        "download_failures": failed,
        "dir": target_dir,
        "total_in_dir": len(provenance),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subject", required=True, help="Subdirectory name under the output root")
    parser.add_argument("--limit", type=int, default=100, help="Images to save (default 100)")
    parser.add_argument("--match", default="", help="Comma-separated keywords to filter rows")
    parser.add_argument("--max-scanned", type=int, default=20000, help="Give up after scanning this many rows")
    parser.add_argument("--output-root", default="static/training_data")
    args = parser.parse_args()

    keywords = [k.strip().lower() for k in args.match.split(",") if k.strip()]

    result = ingest(
        dataset_name=args.dataset,
        subject=args.subject,
        limit=args.limit,
        keywords=keywords,
        output_root=args.output_root,
        max_scanned=args.max_scanned,
    )

    print("=" * 60)
    if not result["success"]:
        print(f"No images saved. {result.get('error', '')}")
        return 1

    print(f"Saved {result['saved']} images to {result['dir']}")
    print(f"Scanned {result['scanned']} rows | too small: {result['skipped_too_small']} | failed: {result['download_failures']}")
    print(f"Directory now holds {result['total_in_dir']} images. Provenance in PROVENANCE.json")
    print("\nTrain with:")
    print(f"  ./venv/bin/python -c \"from agents.lora_trainer import LoRALocalTrainer;"
          f" print(LoRALocalTrainer().train('{result['dir']}', max_steps=1000))\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
