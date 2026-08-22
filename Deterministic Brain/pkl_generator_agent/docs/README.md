# PKL File Generator Agent — Package Guide

AGENT-READABLE INDEX
=====================
This is the **final agent** in an 11-agent pipeline. It takes the outputs of
every upstream agent and bundles them into ONE self-describing `.pkl` file —
the main deliverable of the project.

```
/code
  pkl_utils.py             low-level save/load/hash/inspect/version-registry
                            functions — no agent logic, just the PKL mechanics
  pkl_generator_agent.py   PKLGeneratorAgent (builds + saves the bundle) and
                            PKLBundleLoader (validates columns, predicts)
/docs
  README.md                this file — decision guide + upstream mapping
  pkl_generator_agent.md    full API reference, bundle schema, security notes
/examples
  example_pkl_generator_agent.py   end-to-end: train -> evaluate -> bundle ->
                                    save -> load -> predict, wired to the
                                    Supervised_Classifier_Agent + model_eval_toolkit
                                    packages built earlier in this project
  example_load_and_serve.py        minimal "serving app" snippet: load a
                                    production .pkl and answer predict requests
  example_version_registry.py      register multiple versions, promote one to
                                    production, compare metrics across versions
```

## How This Agent Fits Your 11-Agent Pipeline

```
Dataset_Collector_Agent  ──┐
Data_Warehouse_tookit     ─┼──► dataset_info (source, row counts, lineage)
                            │
Dataset_Cleaner_Agent      ┼──► preprocessing_objects['imputer', ...]
Data_Encoding_Agent        ┼──► preprocessing_objects['encoder', ...], categorical_cols
Dimensionality_Reduction   ┼──► dimensionality_reduction (fitted PCA/UMAP object)
                            │
Supervised_Classifier_Agent┤
Supervised_Regression_Agent┼──► pipeline_or_model  (THE trained model/pipeline)
Unsupervised_Clustering    ┘
                            
model_eval_toolkit  ───────────► metrics (EvaluationAgent.evaluate_* report)

Visualizer_Agent  ─────────────► (not embedded — plots aren't a model artifact;
                                   reference them via extra_metadata if you
                                   want a pointer to where they're saved)
                            │
                            ▼
                 ┌─────────────────────┐
                 │ PKLGeneratorAgent    │  <-- THIS agent
                 │  .generate(...)      │
                 └──────────┬──────────┘
                            ▼
                  <model_name>.pkl  (+ .hash, + registry entry)
                            │
                            ▼
                 PKLGeneratorAgent.load(path)
                 -> PKLBundleLoader.predict(new_data)
```

**Two integration patterns**, depending on how your upstream cleaner/encoder
agents work:

1. **Preprocessing baked into the pipeline** (recommended — matches the
   uploaded reference doc's "Full Pipeline" pattern): `Dataset_Cleaner_Agent`
   and `Data_Encoding_Agent` build sklearn transformers, which get composed
   into a `ColumnTransformer` + model `Pipeline` *before* it reaches this
   agent. `pipeline_or_model` is then the single object that does everything
   from raw columns to a prediction. This is what `example_pkl_generator_agent.py`
   demonstrates.

2. **Preprocessing kept separate** (more flexible, matches the uploaded
   doc's "Saving Preprocessing Objects Separately" pattern): pass the fitted
   imputer/scaler/encoder objects via `preprocessing_objects={...}` alongside
   a `pipeline_or_model` that expects already-preprocessed input. A serving
   app must then apply `preprocessing_objects` in the right order before
   calling `.predict()` — `pkl_generator_agent.md` shows the exact order.

## Decision Guide

```
WHICH FUNCTION DO I CALL?
│
├── I just trained a model/pipeline and want the final .pkl
│   └── PKLGeneratorAgent(save_dir=...).generate(...)
│
├── I want to see what's inside an existing .pkl without loading it into
│   a full pipeline object
│   └── PKLGeneratorAgent.inspect(filepath)   (wraps pkl_utils.inspect_pkl)
│
├── I need to load a .pkl and make predictions in a serving app
│   └── PKLGeneratorAgent.load(filepath) -> PKLBundleLoader.predict(X)
│
├── I have multiple trained versions and need to track/promote the best one
│   └── agent.version_manager (a pkl_utils.PKLVersionManager)
│         .register(...) / .promote_to_production(...) / .load_production(...)
│
└── I only need the raw save/load mechanics (no bundle, no agent)
    └── pkl_utils.save_pkl(obj, path) / pkl_utils.load_pkl(path)

COMPRESSION LEVEL (passed straight to joblib):
│
├── Speed over size            → compress=0
├── Balanced (default)         → compress=3
└── Storage over speed         → compress=6-9

SECURITY:
│
├── You trust the file (you produced it)     → PKLGeneratorAgent.load(path)
│                                                (verifies hash automatically
│                                                 if a .hash file is present)
├── File came from an untrusted source        → DO NOT load it. See
│                                                 "PKL Security" in
│                                                 pkl_generator_agent.md
└── Need cross-language deployment             → convert to ONNX instead of
                                                   handing out the .pkl
                                                   (see pkl_generator_agent.md)
```

## Bundle Schema (what's inside every `.pkl` this agent produces)

```python
{
    "pipeline": <trained model or full sklearn Pipeline>,   # required
    "model_name": str,
    "task": "classification" | "regression" | "clustering",
    "description": str,

    "feature_columns": [str, ...] | None,   # raw input columns, in order
    "numeric_cols": [str, ...] | None,
    "categorical_cols": [str, ...] | None,
    "target_column": str | None,
    "classes": [label, ...] | None,          # classification only

    "dimensionality_reduction": <fitted PCA/UMAP object> | None,
    "preprocessing_objects": {"imputer": ..., "scaler": ..., "encoder": ...},
    "dataset_info": {"source": ..., "n_rows": ..., ...},

    "metrics": {<output of model_eval_toolkit's evaluate_*>},

    "created_at": ISO timestamp,
    "extra_metadata": {...},   # anything else you want attached
}
```
