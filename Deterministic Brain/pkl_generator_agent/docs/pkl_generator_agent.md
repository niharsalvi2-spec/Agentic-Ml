# PKL Generator Agent — Reference

code: `code/pkl_utils.py`, `code/pkl_generator_agent.py`

## What a PKL File Is (Recap)

```
PKL = Pickle File (.pkl / .pickle)
Pickle = Python's built-in serialization module
Serialization = converting a Python object into a byte stream that can be
                saved to disk and reloaded later, identical to the original

Without it: retrain / reload everything from scratch every run
With it:    train once -> save -> load + predict in milliseconds
```

`joblib` (used by this package via `pkl_utils.save_pkl`) is preferred over
plain `pickle` for sklearn-style objects: it's more efficient for
numpy-array-heavy objects (tree ensembles, coefficient matrices), supports
compression, and supports memory-mapped loading for very large models.
`pkl_utils.save_pkl` uses joblib automatically and falls back to stdlib
`pickle` only if joblib isn't installed.

## Why a Bundle Instead of Just the Model

A bare `model.pkl` only gives you `.predict()`. To serve it correctly you
also need: the exact input columns and their order, how they were
preprocessed, what the model's validation metrics were, and provenance
(when/how it was produced). `PKLGeneratorAgent` packages the model together
with all of that into one dict, so the `.pkl` is self-contained and safe to
hand to a serving layer without tribal knowledge.

## `PKLGeneratorAgent` — API

```python
PKLGeneratorAgent(save_dir="pkl_output", registry_dir=None)
```
`registry_dir` defaults to `<save_dir>/registry` and backs `agent.version_manager`.

### `.generate(...)` — the main entry point

```python
agent.generate(
    pipeline_or_model,        # required: trained model or full sklearn Pipeline
    task,                     # required: "classification" | "regression" | "clustering"
    model_name,                # required: used for the filename and registry key
    feature_columns=None, numeric_cols=None, categorical_cols=None,
    target_column=None, classes=None,
    metrics=None,               # output of model_eval_toolkit's EvaluationAgent
    dataset_info=None,          # from Dataset_Collector_Agent / Data_Warehouse_tookit
    dimensionality_reduction=None,   # fitted PCA/UMAP object, if used upstream
    preprocessing_objects=None,       # {"imputer":..., "scaler":..., "encoder":...}
    description="", extra_metadata=None,
    compress=3, with_hash=True, register_version=False, verbose=True,
)
# -> {"filepath": ..., "sha256": ..., "size_bytes": ..., "model_name": ..., "task": ...,
#     "registry_version": ...}   # registry_version only present if register_version=True
```

Writes `<save_dir>/<model_name>.pkl` (+ `<model_name>.pkl.hash` if
`with_hash=True`). Pass `register_version=True` to also add this bundle as
the next version under `model_name` in `agent.version_manager`.

### `.build_bundle(...)` — same arguments, no save

Use this if you want to inspect or modify the bundle dict before writing it
yourself (e.g. with a different compression setting per environment).

### `PKLGeneratorAgent.load(filepath, verify_hash=True)` — static method

Loads a bundle (verifying its `.hash` companion file if present) and returns
a `PKLBundleLoader`. Raises `pkl_utils.PKLSecurityError` if the file was
modified after it was saved with a hash.

### `PKLGeneratorAgent.inspect(filepath, verbose=True)` — static method

Delegates to `pkl_utils.inspect_pkl` — prints (and returns) a summary of
what's inside a `.pkl` without needing to already know its structure. Works
on a bare model, a full Pipeline, or a bundle dict.

## `PKLBundleLoader` — API

Returned by `PKLGeneratorAgent.load(...)`.

```python
loader.predict(X)          # validates X has every column in bundle['feature_columns']
                            # (if X is a DataFrame), reorders them to match, then
                            # calls bundle['pipeline'].predict(X)
loader.predict_proba(X)    # same validation; raises AttributeError if the
                            # underlying model has no predict_proba
loader.summary()           # human-readable string: model_name, task, columns,
                            # classes, metrics, dataset_info
loader.bundle               # the raw bundle dict, for anything not wrapped above
loader.pipeline              # bundle['pipeline'] directly, if you need to bypass
                            # column validation (e.g. already-preprocessed numpy input)
```

Column validation only runs when `X` has a `.columns` attribute (i.e. a
pandas DataFrame) — raw numpy arrays are passed straight through, matching
how `bundle['pipeline']` itself expects input.

## `pkl_utils` — Low-Level Functions

```python
save_pkl(obj, filepath, compress=3, use_joblib=True) -> filepath
load_pkl(filepath) -> obj
compute_file_hash(filepath, algorithm="sha256") -> hex string
save_pkl_with_hash(obj, filepath, compress=3) -> (filepath, hash)
safe_load_pkl(filepath, verify_hash=True) -> obj      # raises PKLSecurityError on mismatch
inspect_pkl(filepath, verbose=True) -> summary dict
```

### `PKLVersionManager` (also `agent.version_manager`)

```python
vm = PKLVersionManager(registry_dir="model_registry")
vm.register(bundle, model_name, metrics=..., description=...) -> "v1", "v2", ...
vm.promote_to_production(model_name, version) -> production filepath
vm.load_production(model_name) -> bundle dict
vm.load_version(model_name, version) -> bundle dict
vm.compare_versions(model_name) -> list of {version, metrics, size_kb, created_at, is_production}
```
Registry state persists in `<registry_dir>/registry.json` and
`<registry_dir>/<model_name>/<model_name>_vN.pkl` files, so it survives
across process restarts.

## Pickle Protocol Notes

```
Protocol 0: ASCII text (oldest, slowest, largest)
Protocol 4: large-object support (>4GB), Python 3.4+
Protocol 5: out-of-band buffers for large arrays, Python 3.8+

pkl_utils.save_pkl always uses joblib (which picks an efficient protocol
automatically) or, as a fallback, pickle.HIGHEST_PROTOCOL — you don't need
to choose a protocol number yourself.
```

## PKL Security — Read Before Loading Anything

```
CRITICAL: pickle/joblib files can execute arbitrary code when loaded.
A malicious .pkl = a code execution attack, not just bad data.

NEVER load a .pkl file from a source you don't control, even if
verify_hash=True — a hash only proves the file wasn't modified AFTER you
(or your pipeline) produced it. It does not make an untrusted file safe.

Safe practices this package supports:
- save_pkl_with_hash() / PKLGeneratorAgent.generate(with_hash=True) — records
  a sha256 alongside the file so later tampering is detectable
- safe_load_pkl() / PKLGeneratorAgent.load() — verifies that hash before
  returning the object, raising PKLSecurityError on mismatch

For sharing model artifacts outside your own trusted pipeline, convert to
ONNX (via skl2onnx) instead of distributing the .pkl directly.
```

## Alternative Formats — When NOT to Use PKL

| Format | Use when |
|---|---|
| `.pkl` / `.joblib` (this package) | Python-only deployment, internal use, keeps preprocessing + model + metadata together |
| ONNX | Cross-language deployment (Java/C#/JS/C++), faster optimized inference, sharing outside a trusted Python pipeline |
| Framework-native (`model.save()` / `torch.save()`) | Deep learning models — never pickle these |
| PMML | Enterprise integration with Java/R/SAS systems |

This agent focuses on `.pkl`/`.joblib` because it's assembling the outputs of
scikit-learn-style upstream agents (classifier, regressor, clustering,
dimensionality reduction) into one Python-native artifact. If your project
later needs cross-language serving, convert `bundle['pipeline']` to ONNX as a
separate export step — the bundle metadata (columns, classes, metrics) still
tells you everything needed to build that conversion's input signature.
