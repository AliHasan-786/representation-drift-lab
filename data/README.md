# Data and artifact policy

Raw datasets, model checkpoints, and full embedding arrays are not committed. Each research run records dataset fingerprints, model revisions, and artifact checksums. Public web data is a compact, validated derivative containing only values and sampled coordinates needed by the portfolio experience.

The original course archive remains external and is registered through `data/legacy/manifest.json`. Set `DRIFTLAB_LEGACY_ROOT` to the `Deep Learning Final Project` directory when locally verifying checksums.

Synthetic smoke data is generated deterministically in memory and requires no download.
