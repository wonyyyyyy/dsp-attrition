import os
import tempfile
import warnings

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def configure_mlflow():
    load_dotenv()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Attrition_Prediction_Experiment")
    run_name = os.getenv("MLFLOW_RUN_NAME", "XGBoost_Run")
    registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "").strip()

    mlflow.set_experiment(experiment_name)
    return run_name, registered_model_name


def preprocess_data(df):
    X = df.drop(columns=["Attrition"]).copy()
    y = df["Attrition"].astype(int)

    drop_cols = ["EmployeeCount", "StandardHours", "Over18", "EmployeeId"]
    X.drop(columns=[col for col in drop_cols if col in X.columns], inplace=True)

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])
        label_encoders[col] = encoder

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y, scaler, label_encoders, drop_cols, cat_cols


def log_preprocessing_artifacts(scaler, label_encoders, feature_names, drop_cols, cat_cols):
    with tempfile.TemporaryDirectory() as tmp_dir:
        scaler_path = os.path.join(tmp_dir, "scaler.pkl")
        encoders_path = os.path.join(tmp_dir, "label_encoders.pkl")
        features_path = os.path.join(tmp_dir, "feature_names.pkl")
        metadata_path = os.path.join(tmp_dir, "preprocessing_metadata.json")

        joblib.dump(scaler, scaler_path)
        joblib.dump(label_encoders, encoders_path)
        joblib.dump(feature_names, features_path)

        metadata = {
            "drop_columns": drop_cols,
            "categorical_columns": cat_cols,
            "feature_names": feature_names,
        }
        pd.Series(metadata).to_json(metadata_path, indent=2)

        mlflow.log_artifact(scaler_path, artifact_path="artifacts")
        mlflow.log_artifact(encoders_path, artifact_path="artifacts")
        mlflow.log_artifact(features_path, artifact_path="artifacts")
        mlflow.log_artifact(metadata_path, artifact_path="artifacts")


def main():
    run_name, registered_model_name = configure_mlflow()

    df = pd.read_csv("data/data_clean.csv")
    X, X_scaled, y, scaler, label_encoders, drop_cols, cat_cols = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.set_tag("pipeline", "xgboost + label_encoder + standard_scaler")

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model, artifact_path="model")
        log_preprocessing_artifacts(
            scaler=scaler,
            label_encoders=label_encoders,
            feature_names=X.columns.tolist(),
            drop_cols=drop_cols,
            cat_cols=cat_cols,
        )

        run_id = run.info.run_id
        print(f"Run completed: {run_id}")
        print(
            f"Metrics: Accuracy={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}"
        )
        print(f"Model URI: runs:/{run_id}/model")

        if registered_model_name:
            registered = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=registered_model_name,
            )
            print(
                f"Registered model: {registered_model_name} "
                f"(version {registered.version})"
            )


if __name__ == "__main__":
    main()
