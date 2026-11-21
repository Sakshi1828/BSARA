# Data

BSARA uses three public, peer-reviewed laboratory datasets. **None of the images are
redistributed in this repository.** Download them from the original sources, respect
their licenses, and cite the original authors.

| Class | Dataset | Source | Images used |
|---|---|---|---|
| `impact_beating` | Impact beating spatter patterns (Attinger et al.) | CSAFE / ScienceDirect. TODO: add DOI + license | 61 scans |
| `gunshot` | Gunshot backspatter patterns | Iowa State University Figshare. TODO: add DOI + license | 8 scans |
| `drip` | Passive drip stains on paper, 25/50/75 cm drop heights | Ontario Tech University / ScienceDirect (2025). TODO: add DOI + license | 1,032 images of 300 drops |

> All blood used in these datasets is animal blood (bovine or swine) deposited under controlled
> laboratory conditions. None of it is crime-scene material.

## Expected layout

```
data/raw/
├── drip/paper25cm/<drop-id>/*.jpg     # as shipped: one folder per drop
├── drip/paper50cm/...
├── drip/paper75cm/...
├── gunshot/Hr*.jpg
└── impact_beating/{C*,HP_*}.jpg
```

Then run `python legacy/v1_detection/preprocess.py` to create `data/processed/` (640×640),
followed by `python -m bsara.build_index`.

## Things to know about this data

* **One drop, many files.** The drip dataset contains 3–5 versions of every drop (original photo,
  edited, cropped, red-channel, contrast-enhanced, Photoshop export). `data/index.csv` records the
  drop each file belongs to (`group`). Any split must keep groups together.
* **One class, one lab.** Each class comes from a different dataset, with different paper,
  scanner or camera, lighting and framing. See the README for how strongly this can be exploited.
* **Gunshot is tiny.** 8 scans are not enough to train or evaluate a gunshot class reliably.
* `data/index.csv` (committed) contains only file names, classes and group IDs, no image data.
