#!/usr/bin/env python3
"""
Video frame extractor for LoRA training (scripts/extract_video_frames.py)

Pulls training-quality stills out of a video. Naively dumping every frame
produces a dataset that is mostly near-duplicates and motion blur, which makes a
LoRA overfit to a handful of shots. This script filters on three axes:

  sharpness  - variance of the Laplacian; drops motion blur and transitions.
  novelty    - perceptual dHash with a Hamming-distance floor; drops frames that
               are near-identical to one already kept, including whole static
               shots held for seconds.
  framing    - trims letterbox bars and generator watermarks, then crops to the
               requested aspect with a configurable vertical bias, so vertical
               footage of people does not lose the faces.

Frames are written with .txt caption sidecars and a PROVENANCE.json recording
the source video and frame index for each still, so synthetic training material
stays traceable.

Usage:
    python scripts/extract_video_frames.py --video clip.mp4 --subject my_style \\
        --caption "in the style of my_style" --limit 60
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def dhash(image: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: compares each pixel to its right-hand neighbour."""
    small = np.asarray(image.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def sharpness(image: Image.Image, size: int = 256) -> float:
    """
    Variance of the Laplacian. Low values mean blur, whether from motion,
    defocus, or a cross-fade between shots.
    """
    gray = np.asarray(image.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float64)
    # 4-neighbour Laplacian via shifts, avoiding a scipy dependency.
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var())


def crop_to_aspect(image: Image.Image, width: int, height: int, top_bias: float = 0.35) -> Image.Image:
    """
    Crop to the target aspect ratio, then resize.

    ``top_bias`` places the crop window vertically: 0.5 is centred, lower values
    favour the top of the frame. Vertical source footage of people puts faces in
    the upper third, and a centred square crop beheads them, so the default sits
    above centre.
    """
    src_w, src_h = image.size
    target_aspect = width / height
    src_aspect = src_w / src_h

    if src_aspect > target_aspect:
        # Source is wider than target: trim the sides, keeping the centre.
        new_w = int(src_h * target_aspect)
        left = (src_w - new_w) // 2
        box = (left, 0, left + new_w, src_h)
    else:
        # Source is taller than target: trim vertically using the bias.
        new_h = int(src_w / target_aspect)
        top = int((src_h - new_h) * top_bias)
        top = max(0, min(top, src_h - new_h))
        box = (0, top, src_w, top + new_h)

    return image.crop(box).resize((width, height), Image.LANCZOS)


def trim_letterbox(image: Image.Image, threshold: int = 18) -> Image.Image:
    """
    Remove near-black letterbox/pillarbox bars.

    Reels splice together clips of differing aspect ratios, leaving black bars on
    some shots. Left in, the adapter learns to draw the bars.
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    rows = gray.max(axis=1) > threshold
    cols = gray.max(axis=0) > threshold
    if not rows.any() or not cols.any():
        return image

    top, bottom = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    left, right = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))

    # Ignore trims that would gut the frame; that means the shot is simply dark.
    if (bottom - top) < gray.shape[0] * 0.4 or (right - left) < gray.shape[1] * 0.4:
        return image
    return image.crop((left, top, right, bottom))


def crop_watermark_strip(image: Image.Image, fraction: float) -> Image.Image:
    """
    Trim a strip off the bottom, where generator watermarks usually sit.

    A watermark left in the training set gets learned as part of the style and
    reappears in generated output.
    """
    if fraction <= 0:
        return image
    w, h = image.size
    return image.crop((0, 0, w, int(h * (1 - fraction))))


def extract(
    video_path: str,
    subject: str,
    caption: str,
    limit: int,
    output_root: str,
    width: int,
    height: int,
    top_bias: float,
    min_sharpness: float,
    min_distance: int,
    watermark_crop: float,
    sample_every: int,
) -> Dict[str, Any]:
    """Extract distinct, sharp frames from a video into a training directory."""
    import av

    if not os.path.exists(video_path):
        return {"success": False, "error": f"No such video: {video_path}", "saved": 0}

    target_dir = os.path.join(output_root, subject)
    os.makedirs(target_dir, exist_ok=True)

    provenance: List[Dict[str, Any]] = []
    provenance_path = os.path.join(target_dir, "PROVENANCE.json")
    if os.path.exists(provenance_path):
        try:
            with open(provenance_path) as f:
                provenance = json.load(f)
        except (OSError, json.JSONDecodeError):
            provenance = []

    kept_hashes: List[int] = []
    saved = 0
    scanned = 0
    rejected_blur = 0
    rejected_dupe = 0

    container = av.open(video_path)
    for index, frame in enumerate(container.decode(video=0)):
        if saved >= limit:
            break
        if index % sample_every:
            continue

        scanned += 1
        image = frame.to_image()
        # Bars first: a watermark crop measured against a letterboxed frame
        # would trim the wrong strip.
        image = trim_letterbox(image)
        image = crop_watermark_strip(image, watermark_crop)

        sharp = sharpness(image)
        if sharp < min_sharpness:
            rejected_blur += 1
            continue

        square = crop_to_aspect(image, width, height, top_bias)
        digest = dhash(square)
        if any(hamming(digest, seen) < min_distance for seen in kept_hashes):
            rejected_dupe += 1
            continue

        kept_hashes.append(digest)
        stem = f"{subject}_{saved + 1:04d}"
        square.save(os.path.join(target_dir, f"{stem}.jpg"), "JPEG", quality=94)
        with open(os.path.join(target_dir, f"{stem}.txt"), "w") as f:
            f.write(caption)

        provenance.append({
            "file": f"{stem}.jpg",
            "source_video": os.path.basename(video_path),
            "frame_index": index,
            "sharpness": round(sharp, 1),
            "synthetic": True,
            "caption": caption,
        })
        saved += 1
        print(f"   + {stem}.jpg  frame {index}  sharpness {sharp:.0f}")

    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)

    return {
        "success": saved > 0,
        "saved": saved,
        "frames_examined": scanned,
        "rejected_blur": rejected_blur,
        "rejected_duplicate": rejected_dupe,
        "dir": target_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--caption", default="", help="Caption written to every sidecar")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768,
                        help="Use 512x768 for vertical source, 512x512 for square")
    parser.add_argument("--top-bias", type=float, default=0.35,
                        help="0.5 centres the crop; lower keeps more of the frame top (faces)")
    parser.add_argument("--min-sharpness", type=float, default=60.0)
    parser.add_argument("--min-distance", type=int, default=12,
                        help="Minimum dHash Hamming distance from every kept frame (0-64)")
    parser.add_argument("--watermark-crop", type=float, default=0.0,
                        help="Fraction of frame height to trim off the bottom, e.g. 0.06")
    parser.add_argument("--sample-every", type=int, default=3, help="Examine every Nth frame")
    parser.add_argument("--output-root", default="static/training_data")
    args = parser.parse_args()

    result = extract(
        video_path=args.video,
        subject=args.subject,
        caption=args.caption or args.subject.replace("_", " "),
        limit=args.limit,
        output_root=args.output_root,
        width=args.width,
        height=args.height,
        top_bias=args.top_bias,
        min_sharpness=args.min_sharpness,
        min_distance=args.min_distance,
        watermark_crop=args.watermark_crop,
        sample_every=args.sample_every,
    )

    print("=" * 60)
    if not result["success"]:
        print(f"No frames saved. {result.get('error', '')}")
        return 1
    print(f"Saved {result['saved']} frames to {result['dir']}")
    print(f"Examined {result['frames_examined']} | blur rejected {result['rejected_blur']}"
          f" | duplicates rejected {result['rejected_duplicate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
