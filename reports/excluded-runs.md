# Excluded and Invalidated Runs

## Oxford Pets to EuroSAT exploratory pair

The repository contains a completed three-seed exploratory artifact at `public/data/benchmark-pets-eurosat-local.json` (SHA-256 `016956b2d942cbc948f2b5bbed87b703627e113bbaee3e9477f629967609b501`). It is intentionally excluded from the primary domain comparison.

The retained EuroSAT zero-shot accuracy is exactly 0.0 at both baseline and step 20 for all three seeds. That makes the retained-accuracy change uninformative: a protocol that begins at a complete floor cannot measure additional forgetting. The run also came from an earlier configuration branch whose saved selections do not match the later stratified configuration currently in `configs/pets-eurosat-local.yaml`.

This artifact is preserved as a failed validity check, not a fourth scientific scenario. No causal or comparative conclusion is drawn from its 0% retained accuracy or its measured CKA movement. Re-admitting this pair requires a fresh clean run with pinned configuration, verified EuroSAT class names/prompts, non-floor baseline performance, and the same three-seed provenance gates as the published domain comparison.
