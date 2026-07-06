# Published data

Files here are schema-versioned derivatives of validated experiment runs. Generate the smoke artifact with `make smoke`, the independent-seed CLIP benchmark with `make benchmark-local`, and the measured local method comparison with `make methods-local`.

`benchmark-local.json`, `method-comparison-local.json`, `interpolation-local.json`, and `early-warning-methodology.json` form the initial web payload. `reproduction-local.json` is a larger single-run diagnostic artifact loaded on demand. Every artifact points to a checksummed public manifest, and the web production build rejects mismatches.

Model checkpoints, raw datasets, and full embeddings do not belong here.
