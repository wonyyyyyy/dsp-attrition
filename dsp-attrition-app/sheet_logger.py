import json
import os
from datetime import datetime, timezone

import gspread


TARGET_HEADERS = [
    "EmployeeId",
    "Age",
    "Attrition",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeCount",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "Over18",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "Predicted_Row",
    "Attrition_Original",
    "Attrition_Predicted",
    "Attrition_Predicted_Yes",
    "Attrition_Source",
    "Attrition_Prediction_Probability",
    "Attrition_Final",
    "Attrition_Yes",
    "Attrition_No",
]


def _is_enabled():
    return os.getenv("GOOGLE_SHEETS_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _load_service_account_info():
    raw_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if raw_json:
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            # Allow fallback to file-based credentials when env value is not a JSON object.
            pass

    file_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    if file_path:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    raise RuntimeError(
        "Service account tidak ditemukan. Set GOOGLE_SERVICE_ACCOUNT_JSON atau GOOGLE_SERVICE_ACCOUNT_FILE."
    )


def _open_worksheet():
    spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "").strip()
    worksheet_name = os.getenv("GOOGLE_SHEETS_WORKSHEET_NAME", "Sheet1").strip()
    if not spreadsheet_id:
        raise RuntimeError("GOOGLE_SHEETS_SPREADSHEET_ID belum diset.")

    credentials_info = _load_service_account_info()
    client = gspread.service_account_from_dict(credentials_info)
    spreadsheet = client.open_by_key(spreadsheet_id)
    return spreadsheet.worksheet(worksheet_name)


def _ensure_headers(worksheet, feature_names):
    expected_headers = TARGET_HEADERS
    first_row = worksheet.row_values(1)
    if not first_row:
        worksheet.append_row(expected_headers, value_input_option="USER_ENTERED")
    elif first_row != expected_headers:
        raise RuntimeError(
            "Header worksheet tidak sesuai format target. "
            "Samakan urutan kolom sheet dengan format dataset prediksi."
        )
    return expected_headers


def _next_employee_id(worksheet):
    col_vals = worksheet.col_values(1)
    if len(col_vals) <= 1:
        return ""
    numeric_ids = []
    for value in col_vals[1:]:
        try:
            numeric_ids.append(int(float(str(value).strip())))
        except (TypeError, ValueError):
            continue
    return (max(numeric_ids) + 1) if numeric_ids else ""


def append_prediction(
    form_values,
    result,
    proba,
    feature_names,
    source="web_form",
    prediction_value=None,
):
    if not _is_enabled():
        return False, "Google Sheets logging nonaktif (GOOGLE_SHEETS_ENABLED=false)."

    worksheet = _open_worksheet()
    headers = _ensure_headers(worksheet, feature_names)
    pred = int(prediction_value) if prediction_value is not None else (1 if result == "Attrition" else 0)
    next_employee_id = _next_employee_id(worksheet)

    row_payload = {header: "" for header in headers}
    row_payload.update(form_values)

    # Default constants used in the training dataset.
    if not row_payload.get("EmployeeCount"):
        row_payload["EmployeeCount"] = 1
    if not row_payload.get("Over18"):
        row_payload["Over18"] = "Y"
    if not row_payload.get("StandardHours"):
        row_payload["StandardHours"] = 80

    row_payload["EmployeeId"] = next_employee_id
    row_payload["Attrition"] = pred
    row_payload["Predicted_Row"] = 1
    row_payload["Attrition_Original"] = ""
    row_payload["Attrition_Predicted"] = pred
    row_payload["Attrition_Predicted_Yes"] = pred
    row_payload["Attrition_Source"] = "Predicted"
    row_payload["Attrition_Prediction_Probability"] = round(float(proba), 6)
    row_payload["Attrition_Final"] = pred
    row_payload["Attrition_Yes"] = 1 if pred == 1 else 0
    row_payload["Attrition_No"] = 1 if pred == 0 else 0
    row_payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    row_payload["source"] = source

    worksheet.append_row([row_payload.get(col, "") for col in headers], value_input_option="USER_ENTERED")
    return True, "Hasil prediksi berhasil disimpan ke Google Spreadsheet."
