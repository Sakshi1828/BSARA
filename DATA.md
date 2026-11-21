# Data

BSARA uses three public, peer-reviewed laboratory datasets. **None of the images are
redistributed in this repository.** Download them from the original sources, respect
their licenses, and cite the original authors.

| Class | Dataset | License | Images used |
|---|---|---|---|
| `impact_beating` | Attinger D, Liu Y, Bybee T, De Brabanter K. *A data set of bloodstain patterns for teaching and research in bloodstain pattern analysis: Impact beating spatters.* Data in Brief 18 (2018) 648–654. [doi:10.1016/j.dib.2018.02.070](https://doi.org/10.1016/j.dib.2018.02.070) | CC BY 4.0 | 61 scans |
| `gunshot` | Attinger D, et al. *A data set of bloodstain patterns for teaching and research in bloodstain pattern analysis: Gunshot backspatters.* Data in Brief 22 (2019) 269–278. doi:10.1016/j.dib.2018.11.075 (TODO: confirm on ScienceDirect). Images: [Iowa State figshare collection](https://iastate.figshare.com/collections/Blood_stain_pattern_and_fluid_dynamics_datasets/5672977) | CC BY-NC-ND 4.0 per repository record (TODO: confirm) | 8 scans |
| `drip` | Kannan S, Pachong SM, Lewis PR, Stotesbury T. *A dataset of drip patterns for teaching and research purposes in forensic bloodstain pattern analysis.* Data in Brief 59 (2025) 111352. Images: [Ontario Tech Borealis dataverse](https://borealisdata.ca/dataverse/ontariotechu) (paper substrate, 25/50/75 cm subset) | TODO: check the Borealis dataset page | 1,032 images of 300 drops |

**License consequences.** If the gunshot data is NonCommercial, the models trained here, and
any released weights, are for **non-commercial research only**. Because of the NoDerivatives
clause, this repository shows no images derived from the gunshot or drip data. The example
figure in the README uses only the CC BY impact dataset (Attinger et al. 2018).

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
