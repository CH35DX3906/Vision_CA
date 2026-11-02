"""Exercise migration, user isolation, and shared-source loading."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from dataset_protocols import (FIELDS, PROTOCOL_DIRECTORIES, SPLITS,
                               assignment_digest, load_split, read_manifest)
from prepare_datasets import (export_dataset, load_records, load_reference,
                              make_image_split, make_subject_split, validate_partitions)

ROOT = Path(__file__).resolve().parents[1]
RATIOS = (0.8, 0.1, 0.1)
ORIGINAL_A_DIGEST = "2558196cc12618e374a52cfda1c6a1f3ea3be6d78509f087210140f5d04bf288"


class SyntheticDatasetTests(unittest.TestCase):
    def setUp(self):
        # Keep temporary files inside the workspace, including on Windows.
        temporary_root = ROOT / "results" / "dataset_tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "raw"
        for class_name in ("fist", "palm"):
            directory = self.source / "hagrid_30k" / f"train_val_{class_name}"
            directory.mkdir(parents=True)
            annotations = {}
            for user in range(10):
                for index in range(2):
                    image_id = f"{class_name}-{user}-{index}"
                    (directory / f"{image_id}.jpg").write_bytes(b"fixture image")
                    label = "no_gesture" if image_id == "fist-0-0" else class_name
                    annotations[image_id] = {"user_id": f"user-{user}", "labels": [label],
                                             "bboxes": [[0.1, 0.2, 0.3, 0.4]], "extra": "preserve"}
            annotation_root = self.source / "ann_train_val"
            annotation_root.mkdir(exist_ok=True)
            (annotation_root / f"{class_name}.json").write_text(json.dumps(annotations))
        self.records, self.annotated = load_records(self.source)

    def export(self, protocol, partitions):
        # Use the public exporter for a small real filesystem fixture.
        return export_dataset(self.base / "evaluation_protocols" / PROTOCOL_DIRECTORIES[protocol], partitions, self.records,
                              self.source, RATIOS, 42, protocol, self.annotated)

    def test_legacy_a_preserves_arbitrary_saved_membership(self):
        # Deliberately use a split unrelated to the seeded shuffle.
        reference = self.base / "legacy"
        reference.mkdir()
        (reference / "summary.json").write_text(json.dumps({"mode": "random_image", "seed": 42,
                                                           "requested_ratios": dict(zip(SPLITS, RATIOS))}))
        expected = {split: [] for split in SPLITS}
        for index, record in enumerate(self.records):
            split = "val" if index % 10 == 0 else "test" if index % 10 == 1 else "train"
            expected[split].append(record)
        for split, items in expected.items():
            directory = reference / split
            directory.mkdir()
            with (directory / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS[:-1])
                writer.writeheader()
                for record in reversed(items):
                    row = {key: record[key] for key in FIELDS[:-1]}
                    row["source_image"] = row["source_image"].replace("/", "\\")
                    writer.writerow(row)
        actual = load_reference(self.records, reference, RATIOS, 42)
        self.assertEqual(assignment_digest(actual), assignment_digest(expected))
        with self.assertRaisesRegex(ValueError, "settings"):
            load_reference(self.records, reference, RATIOS, 43)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            load_reference(self.records[:-1], reference, RATIOS, 42)

    def test_b_keeps_users_together_and_is_order_independent(self):
        # Input ordering must not change which subset owns a user.
        partitions = make_subject_split(self.records, RATIOS, 42)
        reversed_input = make_subject_split(list(reversed(self.records)), RATIOS, 42)
        self.assertEqual(assignment_digest(partitions), assignment_digest(reversed_input))
        self.assertTrue(all(count == 0 for count in validate_partitions(
            partitions, self.records, True).values()))
        self.assertEqual([len({r["user_id"] for r in partitions[s]}) for s in SPLITS], [8, 1, 1])

    def test_validation_rejects_user_leaks_and_missing_images(self):
        # Swapping images preserves image counts while breaking user isolation.
        partitions = make_subject_split(self.records, RATIOS, 42)
        partitions["train"][0], partitions["val"][0] = partitions["val"][0], partitions["train"][0]
        with self.assertRaisesRegex(ValueError, "Users overlap"):
            validate_partitions(partitions, self.records, True)
        partitions = make_subject_split(self.records, RATIOS, 42)
        partitions["train"].pop()
        with self.assertRaisesRegex(ValueError, "cover all"):
            validate_partitions(partitions, self.records, True)

    def test_shared_loader_preserves_annotations_and_checks_a_checksum(self):
        # Both protocols share an API and never require copied images.
        for protocol, partitions in (("A", make_image_split(self.records, RATIOS, 42)),
                                     ("B", make_subject_split(self.records, RATIOS, 42))):
            summary = self.export(protocol, partitions)
            loaded = [row for split in SPLITS for row in load_split(
                PROTOCOL_DIRECTORIES[protocol], split, self.base / "evaluation_protocols")]
            self.assertEqual(len(loaded), len(self.records))
            self.assertTrue(all(row["image_path"].is_relative_to(self.source) for row in loaded))
            special = next(row for row in loaded if row["image_id"] == "fist-0-0")
            self.assertEqual(special["annotation"]["labels"], ["no_gesture"])
            self.assertEqual(special["annotation"]["extra"], "preserve")
            self.assertEqual(summary["folder_label_mismatches"][0]["image_id"], "fist-0-0")
        reference = self.base / "evaluation_protocols/A_random_image"
        summary = json.loads((reference / "summary.json").read_text())
        summary["assignments_sha256"] = "invalid"
        (reference / "summary.json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_reference(self.records, reference, RATIOS, 42)

    def test_manifest_rejects_path_traversal_and_duplicates(self):
        # Bad metadata must fail before any external path can be read.
        path = self.base / "invalid.csv"
        row = {key: self.records[0][key] for key in FIELDS}
        for rows, message in (([dict(row, source_image="../outside.jpg")], "relative path"),
                              ([row, row], "Duplicate")):
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, message):
                read_manifest(path)


class RepositoryDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Check the real dataset, including source metadata for every image.
        cls.source = ROOT / "data/raw_train_data/hagrid-sample-30k-384p"
        cls.records, _ = load_records(cls.source)
        cls.by_id = {row["image_id"]: row for row in cls.records}

    def test_a_matches_original_commit_and_b_has_no_user_leak(self):
        # This pinned checksum independently protects the pre-migration A split.
        for protocol in ("A", "B"):
            directory = ROOT / "data/evaluation_protocols" / PROTOCOL_DIRECTORIES[protocol]
            partitions = {s: read_manifest(directory / f"{s}.csv") for s in SPLITS}
            summary = json.loads((directory / "summary.json").read_text())
            overlaps = validate_partitions(partitions, self.records, protocol == "B")
            self.assertEqual(overlaps, summary["user_overlap_counts"])
            self.assertEqual(assignment_digest(partitions), summary["assignments_sha256"])
            for split, rows in partitions.items():
                self.assertEqual(len(rows), summary["splits"][split]["images"])
                for row in rows:
                    expected = self.by_id[row["image_id"]]
                    self.assertEqual(row, {key: expected[key] for key in FIELDS})
            if protocol == "A":
                self.assertEqual(assignment_digest(partitions), ORIGINAL_A_DIGEST)
            else:
                self.assertEqual(assignment_digest(partitions), assignment_digest(
                    make_subject_split(self.records, RATIOS, 42)))

    def test_both_protocols_load_every_source_image(self):
        # Verify the API independently of legacy image or annotation copies.
        for protocol in ("A", "B"):
            total = 0
            for split in SPLITS:
                rows = load_split(protocol, split)
                total += len(rows)
                for row in rows:
                    self.assertEqual(row["annotation"], self.by_id[row["image_id"]]["annotation"])
                    self.assertTrue(row["image_path"].is_relative_to(self.source))
            self.assertEqual(total, len(self.records))


if __name__ == "__main__":
    unittest.main()
