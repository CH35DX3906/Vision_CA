"""Read both protocols from CSV manifests backed by one raw dataset."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath

SPLITS = ("train", "val", "test")
FIELDS = ("image_id", "class_name", "user_id", "source_image", "source_annotation")
DEFAULT_PROTOCOL_ROOT = Path(__file__).resolve().parent / "data" / "evaluation_protocols"
PROTOCOL_DIRECTORIES = {"A": "A_random_image", "B": "B_subject_independent"}


def assignment_digest(partitions):
    # Hash membership independently of CSV row order and local paths.
    pairs = sorted((r["image_id"], split) for split, rows in partitions.items() for r in rows)
    content = "".join(f"{image_id},{split}\n" for image_id, split in pairs)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_manifest(path):
    # Validate the shared schema and require portable relative source paths.
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"Unexpected manifest columns: {path}")
        rows = list(reader)
    seen = set()
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in FIELDS):
            raise ValueError(f"Empty manifest field: {path}")
        if row["image_id"] in seen:
            raise ValueError(f"Duplicate image ID: {row['image_id']}")
        seen.add(row["image_id"])
        for key in ("source_image", "source_annotation"):
            relative = PurePosixPath(row[key])
            if relative.is_absolute() or ".." in relative.parts or "\\" in row[key] or ":" in row[key]:
                raise ValueError(f"Expected a portable relative path: {row[key]}")
    return rows


def load_split(protocol, split, protocol_root=DEFAULT_PROTOCOL_ROOT, source=None):
    """Return CSV fields, image_path, annotation_path, and original annotation.

    Join feature caches by image_id. All hands of an image inherit its split.
    """
    # Allow source relocation without rewriting the committed manifests.
    protocol = {"random_image": "A", "subject_independent": "B"}.get(protocol, protocol)
    protocol = PROTOCOL_DIRECTORIES.get(protocol.upper(), protocol)
    if protocol not in PROTOCOL_DIRECTORIES.values() or split not in SPLITS:
        raise ValueError("Expected A_random_image/B_subject_independent and train/val/test.")
    directory = Path(protocol_root) / protocol
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    source = Path(source).resolve() if source is not None else (directory / summary["source_root"]).resolve()
    rows = read_manifest(directory / f"{split}.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_annotation"]].append(row)
    for relative, items in grouped.items():
        # Load one class JSON at a time and retain only requested annotations.
        annotation_path = (source / relative).resolve()
        if not annotation_path.is_relative_to(source):
            raise ValueError(f"Annotation escapes source: {annotation_path}")
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        for row in items:
            image_path = (source / row["source_image"]).resolve()
            if not image_path.is_relative_to(source) or not image_path.is_file():
                raise ValueError(f"Missing or unsafe image: {image_path}")
            annotation = annotations[row["image_id"]]
            if (image_path.stem != row["image_id"] or annotation.get("user_id") != row["user_id"]
                    or len(annotation.get("labels", [])) != len(annotation.get("bboxes", []))):
                raise ValueError(f"Manifest and source disagree: {row['image_id']}")
            row.update(image_path=image_path, annotation_path=annotation_path, annotation=annotation)
    return rows
