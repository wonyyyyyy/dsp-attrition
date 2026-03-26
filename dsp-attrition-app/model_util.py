import os
import re

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd


def _configure_tracking():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def _resolve_model_uri():
    model_uri = os.getenv("MLFLOW_MODEL_URI", "").strip()
    if model_uri:
        return model_uri

    run_id = os.getenv("MLFLOW_RUN_ID", "").strip()
    if run_id:
        return f"runs:/{run_id}/model"

    model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "").strip()
    model_version = os.getenv("MLFLOW_MODEL_VERSION", "").strip()
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "").strip()
    if model_name and model_version:
        return f"models:/{model_name}/{model_version}"
    if model_name and model_stage:
        return f"models:/{model_name}/{model_stage}"

    return ""


def _run_id_from_uri(model_uri):
    if model_uri.startswith("runs:/"):
        match = re.match(r"^runs:/([^/]+)/", model_uri)
        if match:
            return match.group(1)

    if model_uri.startswith("models:/"):
        client = mlflow.tracking.MlflowClient()
        parts = model_uri.replace("models:/", "", 1).split("/")
        if len(parts) < 2:
            return ""

        model_name, version_or_stage = parts[0], parts[1]
        if version_or_stage.isdigit():
            model_version = client.get_model_version(model_name, version_or_stage)
            return model_version.run_id

        latest = client.get_latest_versions(model_name, stages=[version_or_stage])
        if latest:
            return latest[0].run_id

    return os.getenv("MLFLOW_RUN_ID", "").strip()


def _download_run_artifact(run_id, relative_path):
    artifact_uri = f"runs:/{run_id}/artifacts/{relative_path}"
    return mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)


def _load_from_mlflow():
    _configure_tracking()

    model_uri = _resolve_model_uri()
    if not model_uri:
        raise RuntimeError(
            "MLFLOW model URI tidak ditemukan. "
            "Set salah satu: MLFLOW_MODEL_URI atau MLFLOW_RUN_ID."
        )

    run_id = _run_id_from_uri(model_uri)
    if not run_id:
        raise RuntimeError("Tidak bisa resolve run_id dari model URI MLflow.")

    model = mlflow.sklearn.load_model(model_uri)
    scaler = joblib.load(_download_run_artifact(run_id, "artifacts/scaler.pkl"))
    label_encoders = joblib.load(_download_run_artifact(run_id, "artifacts/label_encoders.pkl"))
    feature_names = joblib.load(_download_run_artifact(run_id, "artifacts/feature_names.pkl"))
    return model, scaler, label_encoders, feature_names


def _load_from_local():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_models_dir = os.getenv("LOCAL_MODELS_DIR", os.path.join(base_dir, "models"))

    model_path = os.path.join(local_models_dir, "best_model.pkl")
    scaler_path = os.path.join(local_models_dir, "scaler.pkl")
    encoders_path = os.path.join(local_models_dir, "label_encoders.pkl")
    features_path = os.path.join(local_models_dir, "feature_names.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(encoders_path)
    feature_names = joblib.load(features_path)
    return model, scaler, label_encoders, feature_names


def load_artifacts():
    source = os.getenv("MODEL_SOURCE", "auto").strip().lower()

    if source in {"auto", "mlflow"}:
        try:
            return _load_from_mlflow()
        except Exception as exc:
            if source == "mlflow":
                raise RuntimeError(f"Gagal load artifacts dari MLflow: {exc}") from exc
            print(f"Warning: MLflow load failed, fallback ke local models. Detail: {exc}")

    try:
        return _load_from_local()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Artifacts tidak ditemukan di MLflow maupun local folder models."
        ) from exc

def predict_attrition(df_input, model, scaler, label_encoders):
    df = df_input.copy()
    
    # Label Encoding
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handle unseen labels by mapping them to majority class or first class
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
    
    # Ensure numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Scale
    df_scaled = scaler.transform(df)
    
    prediction = model.predict(df_scaled)[0]
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_scaled)[0][1] # Probability of Class 1
    else:
        proba = float(prediction)
        
    return prediction, proba
