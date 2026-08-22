"""
AGENT-READABLE MODULE
======================
name: pkl_generator_agent
purpose: The FINAL agent in the pipeline. Takes the outputs of every upstream
         agent — Dataset_Collector, Dataset_Cleaner, Data_Encoding,
         Dimensionality_Reduction, Unsupervised_Clustering,
         Supervised_Classifier/Regression, model_eval_toolkit, Visualizer,
         Data_Warehouse — and bundles them into ONE self-describing .pkl file:
         the main deliverable of the project. Also provides the loader an
         end user (or a serving app) uses to consume that .pkl file later.

WHY A BUNDLE, NOT JUST THE MODEL
----------------------------------
A bare model.pkl is not enough to serve predictions correctly: you also need
to know which columns it expects, in what order, how they were preprocessed,
what the model's validation metrics were, and when/how it was produced. This
agent packages the trained model/pipeline TOGETHER with all of that context
so the .pkl file is self-contained and safe to hand to a serving layer.

UPSTREAM AGENT -> BUNDLE FIELD MAPPING
-----------------------------------------
    Dataset_Collector_Agent        -> dataset_info (source, collected_at, n_rows_raw)
    Dataset_Cleaner_Agent           -> cleaning_steps, preprocessing_objects['imputer' etc.]
    Data_Encoding_Agent             -> preprocessing_objects['encoder'], categorical_cols
    Dimensionality_Reduction_Agent  -> dimensionality_reduction (fitted PCA/UMAP object + params)
    Unsupervised_Clustering_Agent   -> pipeline_or_model (if task == "clustering"), cluster metadata
    Supervised_Classifier_Agent /
    Supervised_Regression_Agent     -> pipeline_or_model (if task == "classification"/"regression")
    model_eval_toolkit               -> metrics (output of EvaluationAgent.evaluate_*)
    Data_Warehouse_tookit            -> dataset_info (table/source lineage)
    Visualizer_Agent                 -> not embedded in the pkl (plots aren't picklable
                                         artifacts you'd want in a model file) — referenced
                                         only via `extra_metadata['visualization_refs']` if given

USAGE (agent-facing)
---------------------
    from pkl_generator_agent import PKLGeneratorAgent

    agent = PKLGeneratorAgent(save_dir="pkl_output")

    result = agent.generate(
        pipeline_or_model=trained_pipeline,       # from a Supervised_*_Agent
        task="classification",
        model_name="loan_default_classifier",
        feature_columns=["age", "salary", "city", ...],
        numeric_cols=["age", "salary"],
        categorical_cols=["city"],
        target_column="target",
        metrics=eval_report,                      # from model_eval_toolkit
        dataset_info={"source": "loans.csv", "n_rows": 50000},
        dimensionality_reduction=pca_object,       # optional
        preprocessing_objects={"imputer": imputer, "scaler": scaler},  # optional
        description="Random Forest, tuned, from pipeline run #14",
        register_version=True,                    # also add to the version registry
    )
    print(result["filepath"], result["sha256"])

    # Later, in a serving app:
    loader = PKLGeneratorAgent.load(result["filepath"])
    predictions = loader.predict(new_raw_dataframe)
"""

import os
from datetime import datetime

import pkl_utils as pu


class PKLGeneratorAgent:
    """Assembles upstream agent outputs into one production .pkl bundle,
    saves it (with integrity hash + optional version registry entry), and
    hands back a loader for consuming it."""

    REQUIRED_TASKS = {"classification", "regression", "clustering"}

    def __init__(self, save_dir="pkl_output", registry_dir=None):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.version_manager = pu.PKLVersionManager(registry_dir or os.path.join(save_dir, "registry"))

    # ------------------------------------------------------------------ #
    # Bundle assembly
    # ------------------------------------------------------------------ #
    def build_bundle(self, pipeline_or_model, task, model_name,
                      feature_columns=None, numeric_cols=None, categorical_cols=None,
                      target_column=None, classes=None,
                      metrics=None, dataset_info=None,
                      dimensionality_reduction=None, preprocessing_objects=None,
                      description="", extra_metadata=None):
        """
        Build the bundle dict WITHOUT saving it — useful if the caller wants
        to inspect/modify it before writing to disk. `generate()` calls this
        internally.
        """
        if task not in self.REQUIRED_TASKS:
            raise ValueError(f"task must be one of {self.REQUIRED_TASKS}, got '{task}'")
        if pipeline_or_model is None:
            raise ValueError("pipeline_or_model is required — pass the trained model/pipeline "
                              "produced by the upstream Supervised_*/Unsupervised_*_Agent")

        bundle = {
            # --- core deliverable ---
            "pipeline": pipeline_or_model,
            "model_name": model_name,
            "task": task,
            "description": description,

            # --- schema needed for correct inference (from Dataset_Cleaner_Agent /
            #     Data_Encoding_Agent so a serving app knows what raw columns to expect) ---
            "feature_columns": list(feature_columns) if feature_columns is not None else None,
            "numeric_cols": list(numeric_cols) if numeric_cols is not None else None,
            "categorical_cols": list(categorical_cols) if categorical_cols is not None else None,
            "target_column": target_column,
            "classes": list(classes) if classes is not None else None,

            # --- optional upstream artifacts ---
            "dimensionality_reduction": dimensionality_reduction,   # e.g. fitted PCA/UMAP object
            "preprocessing_objects": preprocessing_objects or {},    # e.g. {'imputer':..., 'scaler':...}
            "dataset_info": dataset_info or {},                      # from Dataset_Collector/Data_Warehouse

            # --- evaluation (from model_eval_toolkit's EvaluationAgent) ---
            "metrics": metrics or {},

            # --- provenance ---
            "created_at": datetime.now().isoformat(),
            "extra_metadata": extra_metadata or {},
        }
        return bundle

    # ------------------------------------------------------------------ #
    # Save (the main entry point)
    # ------------------------------------------------------------------ #
    def generate(self, pipeline_or_model, task, model_name,
                 feature_columns=None, numeric_cols=None, categorical_cols=None,
                 target_column=None, classes=None,
                 metrics=None, dataset_info=None,
                 dimensionality_reduction=None, preprocessing_objects=None,
                 description="", extra_metadata=None,
                 compress=3, with_hash=True, register_version=False, verbose=True):
        """
        Build the bundle and write it to `<save_dir>/<model_name>.pkl`
        (plus a `.hash` file if with_hash=True, plus a version-registry entry
        if register_version=True). Returns a result dict with the filepath,
        hash, and size.
        """
        bundle = self.build_bundle(
            pipeline_or_model, task, model_name,
            feature_columns=feature_columns, numeric_cols=numeric_cols,
            categorical_cols=categorical_cols, target_column=target_column, classes=classes,
            metrics=metrics, dataset_info=dataset_info,
            dimensionality_reduction=dimensionality_reduction,
            preprocessing_objects=preprocessing_objects,
            description=description, extra_metadata=extra_metadata,
        )

        safe_name = model_name.lower().replace(" ", "_")
        filepath = os.path.join(self.save_dir, f"{safe_name}.pkl")

        if with_hash:
            filepath, file_hash = pu.save_pkl_with_hash(bundle, filepath, compress=compress)
        else:
            filepath = pu.save_pkl(bundle, filepath, compress=compress)
            file_hash = None

        result = {
            "filepath": filepath,
            "sha256": file_hash,
            "size_bytes": os.path.getsize(filepath),
            "model_name": model_name,
            "task": task,
        }

        if register_version:
            version = self.version_manager.register(
                bundle, model_name, metrics=metrics, description=description, compress=compress
            )
            result["registry_version"] = version

        if verbose:
            print(f"[pkl-generator] saved '{model_name}' ({task}) -> {filepath} "
                  f"({result['size_bytes']:,} bytes)")
            if metrics:
                print(f"[pkl-generator] metrics: {metrics}")

        return result

    # ------------------------------------------------------------------ #
    # Loading / inspection (class-level so a serving app doesn't need
    # to instantiate the whole agent just to consume a bundle)
    # ------------------------------------------------------------------ #
    @staticmethod
    def load(filepath, verify_hash=True):
        """Load a bundle and return a ready-to-use PKLBundleLoader."""
        bundle = pu.safe_load_pkl(filepath, verify_hash=verify_hash)
        return PKLBundleLoader(bundle, filepath)

    @staticmethod
    def inspect(filepath, verbose=True):
        return pu.inspect_pkl(filepath, verbose=verbose)


class PKLBundleLoader:
    """
    Wraps a loaded bundle dict with a convenient predict()/predict_proba()
    interface that validates input columns against what the bundle expects —
    this is what a serving app or downstream script should use.
    """

    def __init__(self, bundle, filepath=None):
        self.bundle = bundle
        self.filepath = filepath
        self.pipeline = bundle.get("pipeline")
        if self.pipeline is None:
            raise ValueError("Loaded bundle has no 'pipeline' key — is this a PKLGeneratorAgent bundle?")

    def _validate_and_order_columns(self, X):
        expected = self.bundle.get("feature_columns")
        if expected is None:
            return X
        # works for pandas DataFrame; leaves other array-likes untouched
        if hasattr(X, "columns"):
            missing = set(expected) - set(X.columns)
            if missing:
                raise ValueError(f"Input is missing expected columns: {missing}")
            return X[expected]
        return X

    def predict(self, X):
        X = self._validate_and_order_columns(X)
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError(f"'{self.bundle.get('model_name')}' does not support predict_proba "
                                  f"(task={self.bundle.get('task')})")
        X = self._validate_and_order_columns(X)
        return self.pipeline.predict_proba(X)

    def summary(self):
        """Human-readable summary of what this bundle contains."""
        b = self.bundle
        lines = [
            f"model_name      : {b.get('model_name')}",
            f"task            : {b.get('task')}",
            f"created_at      : {b.get('created_at')}",
            f"feature_columns : {b.get('feature_columns')}",
            f"target_column   : {b.get('target_column')}",
        ]
        if b.get("classes"):
            lines.append(f"classes         : {b['classes']}")
        if b.get("metrics"):
            lines.append(f"metrics         : {b['metrics']}")
        if b.get("dataset_info"):
            lines.append(f"dataset_info    : {b['dataset_info']}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Minimal smoke test: fake "upstream agent outputs" using sklearn directly
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X = np.random.RandomState(0).randn(300, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
    fake_metrics = {"test_accuracy": float(model.score(X_test, y_test))}

    agent = PKLGeneratorAgent(save_dir="/tmp/pkl_generator_smoke_test")
    result = agent.generate(
        pipeline_or_model=model,
        task="classification",
        model_name="smoke_test_model",
        feature_columns=["f0", "f1", "f2", "f3"],
        numeric_cols=["f0", "f1", "f2", "f3"],
        classes=list(model.classes_),
        metrics=fake_metrics,
        dataset_info={"source": "synthetic", "n_rows": 300},
        description="Smoke test bundle",
        register_version=True,
    )

    loader = PKLGeneratorAgent.load(result["filepath"])
    print(loader.summary())
    print("predictions on raw array:", loader.pipeline.predict(X_test[:5]))
