# Dataset protocols

The raw HaGRID Sample 30K images and original JSON annotations are stored once in
`data/raw_train_data/hagrid-sample-30k-384p/`. Both protocols use the same CSV
schema and the same Python reader. No image or annotation copies are required.

```text
data/
  raw_train_data/hagrid-sample-30k-384p/
    hagrid_30k/train_val_<class>/<image>.jpg
    ann_train_val/<class>.json
  evaluation_protocols/
    A_random_image/{train.csv,val.csv,test.csv,summary.json}
    B_subject_independent/{train.csv,val.csv,test.csv,summary.json}
```

## Existing assignments

Protocol A retains every image's subset from commit
`4272bddc11dbdbdf786fd542a12a1413fb1d4c64`. It uses a per-class random image split;
users may overlap. Migration imports the old manifests instead of rerunning the
shuffle. The membership SHA-256 is
`2558196cc12618e374a52cfda1c6a1f3ea3be6d78509f087210140f5d04bf288`.
The saved summary records this value to check A when its manifests are reused.

Protocol B shuffles sorted unique `user_id` values with seed 42 and allocates users
80/10/10 using largest-remainder rounding. Every image of a user stays in one
subset. Image proportions can differ from user proportions. Each subset contains
all 18 source folder classes. These counts are dataset images, not detected hands.

| Protocol | Subset | Images | Users |
| --- | --- | ---: | ---: |
| A | Train | 25,468 | 7,001 |
| A | Val | 3,185 | 1,931 |
| A | Test | 3,180 | 1,906 |
| B | Train | 25,274 | 6,379 |
| B | Val | 3,457 | 797 |
| B | Test | 3,102 | 797 |

The course explicitly names Train/Val/Test for A and requires user isolation for B.
This project also uses Train/Val/Test for B to support the same model-selection and
evaluation workflow. A user's multiple images are not independent observations.

## Read data

```python
from dataset_loader import load_split

# Change only the protocol when comparing the same recognition method.
samples = load_split("B_subject_independent", "train")
for sample in samples:
    image_path = sample["image_path"]
    annotation = sample["annotation"]
    # Use annotation['labels'] and annotation['bboxes'] for per-hand targets.
```

CSV fields are `image_id`, `class_name`, `user_id`, `source_image`, and
`source_annotation`. Source paths use forward slashes and are relative to the raw
dataset root. `summary.json` stores that root relative to its own directory, so a
clone has no machine-specific absolute paths. Pass `source=Path(...)` to
`load_split` when the raw dataset is relocated. For metadata-only access, use
`read_manifest(Path("data/evaluation_protocols/B_subject_independent/train.csv"))`.

The folder class is not necessarily a hand label. Image
`70970c12-ebf8-4021-8e85-6f9f70f9b829` is in `fist`, but its original annotation
contains only `no_gesture`. The original annotation is preserved and this mismatch
is recorded in both summaries. Summaries report folder counts separately from
annotated hand-label counts, including `no_gesture`.

## Regenerate manifests

Run with the project's Python interpreter from the repository root:

```text
python -B prepare_datasets.py
```

The default command regenerates both protocols. Existing A CSVs are used as the
reference and checked against their saved checksum. Changes to source membership,
seed, or ratios fail rather than silently reallocating A. `--protocol A_random_image`,
`--protocol B_subject_independent`, and `--protocol both` select outputs; short
names, course identifiers `A`/`B`, and legacy `p1`/`p2` remain aliases. `--output-root` now selects
the evaluation-protocol directory, not copied-image storage.
The old `--extend-p1` option and user-folder export have been removed.

For the initial conversion only, use
`python -B prepare_datasets.py --a-reference data/P1_random_image` while the old
manifests are available. On a genuinely new dataset with no A reference, the
per-class random splitter is used. The generator does not remove legacy folders;
the old copies are removed separately after validating the migration.

## Training and evaluation

Join existing MediaPipe cache paths to the manifests using their filename stem as
`image_id`. Multiple hands from one image inherit the same subset; never randomly
split hand-level feature rows. Frozen, per-image landmark extraction can be reused
by both protocols. Fit learned preprocessing and classifiers on Train only, select
models and rejection thresholds on Val, and reserve Test for final evaluation.

Do not discard images where no hand was detected from end-to-end evaluation.
Ground-truth boxes may assign training labels, but must not select hands during
validation/test inference. Report distributions for the chosen command mapping as
well as original gestures; command vocabulary decisions remain in model code.

Removing duplicate tracked paths does not rewrite Git history. Old copies remain
recoverable from the original commit, and cleaning them does not require uploading
replacement raw image contents.
