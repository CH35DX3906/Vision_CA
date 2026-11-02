"""Prepare reproducible A/B CSV splits without copying images or annotations."""

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from dataset_protocols import (FIELDS, PROTOCOL_DIRECTORIES, SPLITS,
                               assignment_digest, read_manifest)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def write_json(path, data):
    # Keep deterministic metadata readable in code review.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_counts(total, ratios):
    # Use largest remainders so rounded counts always sum to the total.
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def load_records(source):
    # Match local images, not every entry in the larger annotation inventory.
    image_root = source / "hagrid_30k"
    annotation_root = source / "ann_train_val"
    if not image_root.is_dir() or not annotation_root.is_dir():
        raise ValueError("The source must contain hagrid_30k and ann_train_val.")
    by_class = defaultdict(list)
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            by_class[path.parent.name.removeprefix("train_val_")].append(path)
    records, seen, annotated_total = [], set(), 0
    for class_name, paths in sorted(by_class.items()):
        annotation_path = annotation_root / f"{class_name}.json"
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        annotated_total += len(annotations)
        for path in paths:
            image_id = path.stem
            if image_id in seen:
                raise ValueError(f"Duplicate image ID: {image_id}")
            annotation = annotations.get(image_id)
            if annotation is None:
                raise ValueError(f"Missing annotation: {image_id}")
            user_id = annotation.get("user_id")
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(f"Missing or invalid user_id: {image_id}")
            if len(annotation.get("bboxes", [])) != len(annotation.get("labels", [])):
                raise ValueError(f"Box and label counts differ: {image_id}")
            seen.add(image_id)
            records.append({"image_id": image_id, "class_name": class_name,
                            "user_id": user_id, "source": path, "annotation": annotation,
                            "source_image": path.relative_to(source).as_posix(),
                            "source_annotation": annotation_path.relative_to(source).as_posix()})
    if not records:
        raise ValueError("No local images were found.")
    return records, annotated_total


def load_reference(records, reference, ratios, seed):
    # Preserve all saved A assignments; never fill gaps or silently reshuffle.
    summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    if (summary["seed"] != seed or summary["requested_ratios"] != dict(zip(SPLITS, ratios))
            or summary["mode"] != "random_image"):
        raise ValueError("A reference seed, ratios, and mode must match requested settings.")
    by_id = {record["image_id"]: record for record in records}
    result, seen = {split: [] for split in SPLITS}, set()
    for split in SPLITS:
        manifest = reference / f"{split}.csv"
        if manifest.exists():
            rows = read_manifest(manifest)
        else:
            with (reference / split / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        for row in rows:
            record = by_id.get(row["image_id"])
            if record is None or row["image_id"] in seen:
                raise ValueError(f"Unknown or repeated A image: {row['image_id']}")
            if (row["class_name"] != record["class_name"] or row["user_id"] != record["user_id"]
                    or row["source_image"].replace("\\", "/") != record["source_image"]
                    or ("source_annotation" in row
                        and row["source_annotation"] != record["source_annotation"])):
                raise ValueError(f"A reference metadata differs from source: {row['image_id']}")
            seen.add(row["image_id"])
            result[split].append(record)
    if seen != set(by_id):
        raise ValueError("A reference must cover exactly the current source images.")
    saved_digest = summary.get("assignments_sha256")
    if saved_digest and saved_digest != assignment_digest(result):
        raise ValueError("A assignments differ from their saved checksum.")
    return result


def make_image_split(records, ratios, seed):
    # Only new datasets use the original per-class random image algorithm.
    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        groups[record["class_name"]].append(record)
    result = {split: [] for split in SPLITS}
    for class_name in sorted(groups):
        items = sorted(groups[class_name], key=lambda row: row["image_id"])
        rng.shuffle(items)
        offset = 0
        for split, count in zip(SPLITS, split_counts(len(items), ratios)):
            result[split].extend(items[offset:offset + count])
            offset += count
    return result


def make_subject_split(records, ratios, seed):
    # Assign whole users, allowing image ratios to differ from user ratios.
    users = sorted({record["user_id"] for record in records})
    random.Random(seed).shuffle(users)
    counts = split_counts(len(users), ratios)
    if any(count == 0 for count in counts):
        raise ValueError("There are too few users for three nonempty subsets.")
    user_split, offset = {}, 0
    for split, count in zip(SPLITS, counts):
        user_split.update({user: split for user in users[offset:offset + count]})
        offset += count
    result = {split: [] for split in SPLITS}
    for record in records:
        result[user_split[record["user_id"]]].append(record)
    return result


def validate_partitions(partitions, records, subject_independent):
    # Check full image coverage and user isolation before publishing manifests.
    expected = {record["image_id"] for record in records}
    classes = {record["class_name"] for record in records}
    seen, users = set(), {}
    for split in SPLITS:
        items = partitions[split]
        if not items:
            raise ValueError(f"Empty subset: {split}")
        for record in items:
            image_id = record["image_id"]
            if image_id not in expected or image_id in seen:
                raise ValueError(f"Unknown or repeated image: {image_id}")
            seen.add(image_id)
        users[split] = {record["user_id"] for record in items}
        if {record["class_name"] for record in items} != classes:
            raise ValueError(f"Subset is missing source classes: {split}")
    if seen != expected:
        raise ValueError("Partitions do not cover all source images.")
    overlaps = {f"{left}_{right}": len(users[left] & users[right])
                for index, left in enumerate(SPLITS) for right in SPLITS[index + 1:]}
    if subject_independent and any(overlaps.values()):
        raise ValueError("Users overlap in protocol B.")
    return overlaps


def export_dataset(output, partitions, records, source, ratios, seed, protocol,
                   annotated_total, preserved_digest=None):
    # Output CSV and statistics only; images and labels stay in the raw source.
    overlaps = validate_partitions(partitions, records, protocol == "B")
    expected_files = {f"{split}.csv" for split in SPLITS} | {"summary.json"}
    if output.exists() and any(path.name not in expected_files or not path.is_file()
                               for path in output.iterdir()):
        raise ValueError(f"Output contains unexpected files: {output}")
    digest = assignment_digest(partitions)
    summary = {"schema_version": 1, "protocol": protocol,
               "mode": "random_image" if protocol == "A" else "subject_independent",
               "seed": seed, "requested_ratios": dict(zip(SPLITS, ratios)),
               "ratio_unit": "images_per_class" if protocol == "A" else "users",
               "source_root": Path(os.path.relpath(source, output)).as_posix(),
               "manifest_paths_relative_to": "source_root", "image_storage": "shared raw source",
               "total_images": len(records), "total_users": len({r["user_id"] for r in records}),
               "annotation_entries_in_source_classes": annotated_total,
               "classes": sorted({r["class_name"] for r in records}),
               "assignments_sha256": digest, "user_overlap_counts": overlaps, "splits": {}}
    if preserved_digest is not None:
        if digest != preserved_digest:
            raise ValueError("A membership changed during migration.")
        summary["preserved_reference_assignments_sha256"] = preserved_digest
    summary["folder_label_mismatches"] = [
        {"image_id": r["image_id"], "class_name": r["class_name"], "labels": r["annotation"]["labels"]}
        for r in records if r["class_name"] not in r["annotation"]["labels"]]
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        items = sorted(partitions[split], key=lambda row: (row["class_name"], row["image_id"]))
        with (output / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows({key: record[key] for key in FIELDS} for record in items)
        summary["splits"][split] = {
            "images": len(items), "users": len({r["user_id"] for r in items}),
            "image_fraction": len(items) / len(records),
            "images_per_class": dict(sorted(Counter(r["class_name"] for r in items).items())),
            "annotations_per_label": dict(sorted(Counter(
                label for r in items for label in r["annotation"]["labels"]).items()))}
        print(f"{protocol}/{split}: {len(items)} images, {summary['splits'][split]['users']} users")
    write_json(output / "summary.json", summary)
    return summary


def main():
    # Existing A manifests always take precedence over a new random split.
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=project / "data/raw_train_data/hagrid-sample-30k-384p")
    parser.add_argument("--output-root", type=Path, default=project / "data/evaluation_protocols")
    parser.add_argument("--a-reference", type=Path,
                        help="Existing A directory in legacy or CSV-only format.")
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", choices=("A_random_image", "B_subject_independent",
                                            "random_image", "subject_independent", "A", "B",
                                            "p1", "p2", "both"), default="both")
    args = parser.parse_args()
    if (any(not math.isfinite(value) or value <= 0 for value in args.ratios)
            or abs(sum(args.ratios) - 1) > 1e-9):
        parser.error("Ratios must be finite, positive, and add up to 1.")
    source, output_root = args.source.resolve(), args.output_root.resolve()
    # All generated files must stay within this project workspace.
    if not output_root.is_relative_to(project):
        parser.error("The output root must be inside the project workspace.")
    records, annotated_total = load_records(source)
    protocol = {"A_random_image": "A", "B_subject_independent": "B",
                "random_image": "A", "subject_independent": "B",
                "p1": "A", "p2": "B"}.get(args.protocol, args.protocol)
    pending = []
    if protocol in ("A", "both"):
        reference = args.a_reference
        if reference is None:
            candidates = [output_root / "A_random_image", project / "data/evaluation_protocols/A_random_image",
                          project / "data/P1_random_image"]
            reference = next((p for p in candidates if (p / "summary.json").is_file()), None)
        partitions = (load_reference(records, reference.resolve(), args.ratios, args.seed)
                      if reference is not None else make_image_split(records, args.ratios, args.seed))
        digest = assignment_digest(partitions) if reference is not None else None
        pending.append(("A", partitions, digest))
    if protocol in ("B", "both"):
        pending.append(("B", make_subject_split(records, args.ratios, args.seed), None))
    # Validate both protocols before changing either protocol's manifests.
    for name, partitions, digest in pending:
        validate_partitions(partitions, records, name == "B")
    for name, partitions, digest in pending:
        export_dataset(output_root / PROTOCOL_DIRECTORIES[name], partitions, records, source, args.ratios,
                       args.seed, name, annotated_total, digest)


if __name__ == "__main__":
    main()
