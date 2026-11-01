"""Build image-random and user-grouped HaGRID datasets."""

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def write_json(path, data):
    # Keep all original annotation fields in readable JSON files.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_counts(total, ratios):
    # Use the largest remainders so the counts always add up to the total.
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def load_records(source):
    # Match local images to their class JSON instead of expecting every annotated image.
    image_root = source / "hagrid_30k"
    annotation_root = source / "ann_train_val"
    if not image_root.is_dir() or not annotation_root.is_dir():
        raise ValueError("The source must contain hagrid_30k and ann_train_val.")
    image_paths = sorted(path for path in image_root.rglob("*")
                         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    by_class = defaultdict(list)
    for path in image_paths:
        class_name = path.parent.name.removeprefix("train_val_")
        by_class[class_name].append(path)
    records = []
    seen_ids = set()
    annotated_total = 0
    for class_name, paths in sorted(by_class.items()):
        annotations = json.loads((annotation_root / f"{class_name}.json").read_text(encoding="utf-8"))
        annotated_total += len(annotations)
        for path in paths:
            image_id = path.stem
            if image_id in seen_ids:
                raise ValueError(f"Duplicate image ID: {image_id}")
            if image_id not in annotations:
                raise ValueError(f"Missing annotation for {path}")
            annotation = annotations[image_id]
            user_id = annotation.get("user_id")
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError(f"Missing or invalid user_id: {image_id}")
            if len(annotation.get("bboxes", [])) != len(annotation.get("labels", [])):
                raise ValueError(f"Box and label counts differ: {image_id}")
            seen_ids.add(image_id)
            records.append({"image_id": image_id, "class_name": class_name,
                            "user_id": user_id, "source": path, "annotation": annotation})
    if not records:
        raise ValueError("No local images were found.")
    return records, annotated_total


def make_image_split(records, ratios, seed, existing_output=None):
    # Shuffle each class separately to keep similar class ratios in all subsets.
    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        groups[record["class_name"]].append(record)
    result = {split: [] for split in SPLITS}
    assigned = set()
    if existing_output is not None:
        # Keep existing image assignments while checking their metadata and bytes.
        by_id = {record["image_id"]: record for record in records}
        allowed_files = {existing_output / "summary.json"}
        saved = json.loads((existing_output / "summary.json").read_text(encoding="utf-8"))
        if (saved["mode"] != "random_image" or saved["seed"] != seed
                or saved["requested_ratios"] != dict(zip(SPLITS, ratios))):
            raise ValueError("P1 settings must match the existing seed and ratios.")
        for split in SPLITS:
            split_dir = existing_output / split
            manifest = split_dir / "manifest.csv"
            allowed_files.add(manifest)
            expected_annotations = defaultdict(dict)
            with manifest.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    record = by_id[row["image_id"]]
                    relative = Path("images") / record["class_name"] / record["source"].name
                    if (record["image_id"] in assigned or row["image_path"] != relative.as_posix()
                            or row["class_name"] != record["class_name"]
                            or row["user_id"] != record["user_id"]):
                        raise ValueError(f"Conflicting P1 manifest entry: {row['image_id']}")
                    image = split_dir / relative
                    if image.read_bytes() != record["source"].read_bytes():
                        raise ValueError(f"Image differs from source: {image}")
                    allowed_files.add(image)
                    assigned.add(record["image_id"])
                    result[split].append(record)
                    expected_annotations[record["class_name"]][record["image_id"]] = record["annotation"]
            for path in (split_dir / "annotations").glob("*.json"):
                allowed_files.add(path)
                if json.loads(path.read_text(encoding="utf-8")) != expected_annotations.pop(path.stem, {}):
                    raise ValueError(f"Conflicting P1 annotations: {path}")
            if expected_annotations:
                raise ValueError(f"Missing P1 annotations: {split}")
        if any(path not in allowed_files for path in existing_output.rglob("*") if path.is_file()):
            raise ValueError("P1 contains unexpected files.")
    for class_name in sorted(groups):
        items = sorted((r for r in groups[class_name] if r["image_id"] not in assigned),
                       key=lambda record: record["image_id"])
        rng.shuffle(items)
        offset = 0
        for split, target in zip(SPLITS, split_counts(len(groups[class_name]), ratios)):
            # Fill each class's remaining quota without moving existing images.
            count = target - sum(r["class_name"] == class_name for r in result[split])
            if count < 0:
                raise ValueError(f"Existing P1 split exceeds its target: {split}/{class_name}")
            result[split].extend(items[offset:offset + count])
            offset += count
    return result


def make_subject_split(records, ratios, seed):
    # Split shuffled users, then put every image of each user in the same subset.
    users = sorted({record["user_id"] for record in records})
    random.Random(seed).shuffle(users)
    counts = split_counts(len(users), ratios)
    if any(count == 0 for count in counts):
        raise ValueError("There are too few users for three nonempty subsets.")
    user_split = {}
    offset = 0
    for split, count in zip(SPLITS, counts):
        for user_id in users[offset:offset + count]:
            user_split[user_id] = split
        offset += count
    result = {split: [] for split in SPLITS}
    for record in records:
        result[user_split[record["user_id"]]].append(record)
    return result


def export_user_dataset(output, records, source):
    # Preserve P2's user folders and include every locally available class.
    groups = defaultdict(list)
    expected_images = {}
    for record in records:
        groups[record["user_id"]].append(record)
        relative = (Path(record["user_id"]) / "images" / record["class_name"]
                    / record["source"].name)
        expected_images[relative] = record
    if output.exists():
        # Check existing data before extending it, so unrelated files are protected.
        for path in output.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(output)
            if relative in expected_images:
                continue
            if relative.as_posix() in {"manifest.csv", "summary.json"}:
                continue
            if (len(relative.parts) == 2 and relative.name == "annotations.json"
                    and relative.parts[0] in groups):
                annotations = json.loads(path.read_text(encoding="utf-8"))
                expected = {r["image_id"]: r["annotation"] for r in groups[relative.parts[0]]}
                if all(key in expected and value == expected[key]
                       for key, value in annotations.items()):
                    continue
            raise ValueError(f"Unexpected or conflicting P2 file: {path}")
    output.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "user_id", "class_name", "image_path",
              "annotation_path", "source_image"]
    rows = []
    label_counts = Counter()
    for relative, record in sorted(expected_images.items()):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(record["source"], destination)
        # Verify both existing and newly copied images against the source bytes.
        source_hash = hashlib.sha256(record["source"].read_bytes()).digest()
        if hashlib.sha256(destination.read_bytes()).digest() != source_hash:
            raise ValueError(f"Image differs from source: {destination}")
        label_counts.update(record["annotation"]["labels"])
        rows.append({"image_id": record["image_id"], "user_id": record["user_id"],
                     "class_name": record["class_name"], "image_path": relative.as_posix(),
                     "annotation_path": f'{record["user_id"]}/annotations.json',
                     "source_image": record["source"].relative_to(source).as_posix()})
    for user_id, items in sorted(groups.items()):
        write_json(output / user_id / "annotations.json",
                   {r["image_id"]: r["annotation"] for r in sorted(items, key=lambda r: r["image_id"])})
    # Publish complete metadata only after every image has been checked.
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "mode": "group_by_user_id", "source": str(source),
        "total_images": len(records), "total_users": len(groups),
        "images_per_class": dict(sorted(Counter(r["class_name"] for r in records).items())),
        "images_per_user": {user: len(items) for user, items in sorted(groups.items())},
        "labels": sorted(label_counts), "annotations_per_label": dict(sorted(label_counts.items())),
        "layout": "<user_id>/images/<class>/<image>; <user_id>/annotations.json",
        "manifest_paths_relative_to": output.name, "image_storage": "independent copies",
        "validation": {"all_source_images_included_once": True,
                       "image_hashes_match_source": True, "original_annotations_preserved": True,
                       "each_user_has_exactly_one_folder": True}}
    write_json(output / "summary.json", summary)
    print(f"{output.name}: {len(records)} images, {len(groups)} users, "
          f"{len(summary['images_per_class'])} classes, {len(label_counts)} labels", flush=True)
    return summary


def export_dataset(output, partitions, records, source, ratios, seed, mode, annotated_total,
                   extend=False):
    # Refuse to replace existing data so reruns cannot silently overwrite a dataset.
    if output.exists() and any(output.iterdir()) and not extend:
        raise ValueError(f"Output folder is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    classes = sorted({record["class_name"] for record in records})
    # Count actual box labels, including no_gesture, separately from image classes.
    label_counts = Counter(label for record in records for label in record["annotation"]["labels"])
    user_sets = {}
    image_sets = {}
    summary = {"source": str(source), "seed": seed, "mode": mode,
               "requested_ratios": dict(zip(SPLITS, ratios)),
               "total_images": len(records), "total_users": len({r["user_id"] for r in records}),
               "annotation_entries_in_source_classes": annotated_total,
               "annotation_entries_without_local_images": annotated_total - len(records),
               "classes": classes, "labels": sorted(label_counts),
               "annotations_per_label": dict(sorted(label_counts.items())),
               "image_storage": "independent copies",
               "layout": "<split>/images/<class>/<image>; <split>/annotations/<class>.json",
               "ratio_unit": "users" if mode == "subject_independent" else "images_per_class",
               "splits": {}}
    for split in SPLITS:
        items = sorted(partitions[split], key=lambda r: (r["class_name"], r["image_id"]))
        if not items:
            raise ValueError(f"Empty subset: {split}")
        annotations = {class_name: {} for class_name in classes}
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "class_name", "user_id", "source_image"])
            writer.writeheader()
            for record in items:
                relative = Path("images") / record["class_name"] / record["source"].name
                destination = split_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Existing images were checked before extending the partitions.
                if not destination.exists():
                    shutil.copy2(record["source"], destination)
                annotations[record["class_name"]][record["image_id"]] = record["annotation"]
                writer.writerow({"image_id": record["image_id"], "image_path": relative.as_posix(),
                                 "class_name": record["class_name"], "user_id": record["user_id"],
                                 "source_image": str(record["source"].relative_to(source))})
        for class_name, data in annotations.items():
            write_json(split_dir / "annotations" / f"{class_name}.json", data)
        user_sets[split] = {record["user_id"] for record in items}
        image_sets[split] = {record["image_id"] for record in items}
        summary["splits"][split] = {
            "images": len(items), "users": len(user_sets[split]),
            "image_fraction": len(items) / len(records),
            "images_per_class": dict(sorted(Counter(r["class_name"] for r in items).items()))}
        print(f"{output.name}/{split}: {len(items)} images, {len(user_sets[split])} users", flush=True)
    overlaps = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            if image_sets[left] & image_sets[right]:
                raise ValueError(f"Images overlap: {left}, {right}")
            overlaps[f"{left}_{right}"] = len(user_sets[left] & user_sets[right])
    if mode == "subject_independent" and any(overlaps.values()):
        raise ValueError("Users overlap in the subject-independent dataset.")
    if set.union(*image_sets.values()) != {record["image_id"] for record in records}:
        raise ValueError("The dataset does not contain all source images.")
    summary["user_overlap_counts"] = overlaps
    write_json(output / "summary.json", summary)
    return summary


def main():
    # Make source paths, split ratios, and random seed easy to reuse.
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=project / "data/raw_train_data/hagrid-sample-30k-384p")
    parser.add_argument("--output-root", type=Path, default=project / "data")
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocol", choices=("p1", "p2", "both"), default="both")
    parser.add_argument("--extend-p1", action="store_true",
                        help="Add missing P1 images while preserving existing split assignments.")
    args = parser.parse_args()
    if any(value <= 0 for value in args.ratios) or abs(sum(args.ratios) - 1) > 1e-9:
        parser.error("Ratios must be positive and add up to 1.")
    source = args.source.resolve()
    output_root = args.output_root.resolve()
    p1_output = output_root / "P1_random_image"
    if (args.protocol in {"p1", "both"} and p1_output.exists()
            and any(p1_output.iterdir()) and not args.extend_p1):
        parser.error(f"Output folder is not empty: {p1_output}")
    records, annotated_total = load_records(source)
    print(f"Found {len(records)} local images from {len({r['user_id'] for r in records})} users.", flush=True)
    # Allow P2 to be completed without rebuilding or changing P1.
    if args.protocol in {"p1", "both"}:
        existing = p1_output if args.extend_p1 and p1_output.exists() else None
        partition = make_image_split(records, args.ratios, args.seed, existing)
        export_dataset(p1_output, partition, records, source, args.ratios,
                       args.seed, "random_image", annotated_total, extend=args.extend_p1)
    if args.protocol in {"p2", "both"}:
        export_user_dataset(output_root / "P2_subject_independent", records, source)


if __name__ == "__main__":
    main()
