from flask import Flask, render_template, request
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from model_util import load_artifacts, predict_attrition
from sheet_logger import append_prediction

app = Flask(__name__)
model = None
scaler = None
label_encoders = {}
feature_names = []
MODEL_LOAD_ERROR = None
CATEGORICAL_OPTIONS = {}
IGNORED_COLUMNS = []
NUMERIC_FEATURES = set()
COLUMN_VARIANTS = []
FORM_FIELDS = []
NUMERIC_PLACEHOLDERS = {}
NUMERIC_MAX_VALUES = {}

IMPORTANCE_FEATURES = [
    'OverTime',               # Lembur: yes/no, sangat individual
    'YearsAtCompany',         # Lama di perusahaan ini
    'TotalWorkingYears',      # Total pengalaman kerja
    'YearsInCurrentRole',     # Lama di posisi sekarang
    'YearsSinceLastPromotion',# Lama sejak promosi terakhir
    'YearsWithCurrManager',   # Lama dengan manajer saat ini
    'EnvironmentSatisfaction',# Kepuasan lingkungan kerja (subjektif)
    'JobInvolvement',         # Tingkat keterlibatan kerja (subjektif)
    'MonthlyIncome',          # Gaji bulanan (sangat variatif)
    'MaritalStatus',          # Status pernikahan (kategorikal)
]

NUMERIC_MAX_OVERRIDES = {
    'Age': 65,
    'DistanceFromHome': 30,
    'TotalWorkingYears': 45,
    'YearsInCurrentRole': 20,
    'YearsWithCurrManager': 20,
    'NumCompaniesWorked': 10,
}


def model_is_loaded():
    return model is not None and scaler is not None and bool(feature_names)


def ensure_model_loaded():
    global model, scaler, label_encoders, feature_names, MODEL_LOAD_ERROR
    if model_is_loaded():
        return

    try:
        model, scaler, label_encoders, feature_names = load_artifacts()
        MODEL_LOAD_ERROR = None
        refresh_model_metadata()
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        model = None
        scaler = None
        label_encoders = {}
        feature_names = []
        refresh_model_metadata()


def refresh_model_metadata():
    global CATEGORICAL_OPTIONS, IGNORED_COLUMNS, NUMERIC_FEATURES, COLUMN_VARIANTS, FORM_FIELDS, NUMERIC_PLACEHOLDERS, NUMERIC_MAX_VALUES
    CATEGORICAL_OPTIONS = {
        name: list(encoder.classes_)
        for name, encoder in label_encoders.items()
    }
    IGNORED_COLUMNS = [col for col in DATASET_COLUMNS if col not in feature_names]
    NUMERIC_FEATURES = {name for name in feature_names if name not in label_encoders}
    NUMERIC_PLACEHOLDERS, NUMERIC_MAX_VALUES = load_numeric_field_metadata()
    COLUMN_VARIANTS = [
        ('dataset asli', DATASET_COLUMNS),
        ('tanpa EmployeeId', DATASET_COLUMNS[1:]),
        ('tanpa Attrition', [col for col in DATASET_COLUMNS if col != 'Attrition']),
        (
            'tanpa EmployeeId dan Attrition',
            [col for col in DATASET_COLUMNS if col not in {'EmployeeId', 'Attrition'}],
        ),
        ('field aktif model', feature_names),
    ]
    FORM_FIELDS = build_form_fields()


DATASET_COLUMNS = [
    'EmployeeId',
    'Age',
    'Attrition',
    'BusinessTravel',
    'DailyRate',
    'Department',
    'DistanceFromHome',
    'Education',
    'EducationField',
    'EmployeeCount',
    'EnvironmentSatisfaction',
    'Gender',
    'HourlyRate',
    'JobInvolvement',
    'JobLevel',
    'JobRole',
    'JobSatisfaction',
    'MaritalStatus',
    'MonthlyIncome',
    'MonthlyRate',
    'NumCompaniesWorked',
    'Over18',
    'OverTime',
    'PercentSalaryHike',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'StandardHours',
    'StockOptionLevel',
    'TotalWorkingYears',
    'TrainingTimesLastYear',
    'WorkLifeBalance',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager',
]


FIELD_DESCRIPTIONS = {
    'EmployeeId': 'Employee Identifier',
    'Attrition': 'Did the employee attrition? (0=no, 1=yes)',
    'Age': 'Age of the employee',
    'BusinessTravel': 'Travel commitments for the job',
    'DailyRate': 'Daily salary',
    'Department': 'Employee Department',
    'DistanceFromHome': 'Distance from work to home (in km)',
    'Education': '1-Below College, 2-College, 3-Bachelor, 4-Master, 5-Doctor',
    'EducationField': 'Field of Education',
    'EnvironmentSatisfaction': '1-Low, 2-Medium, 3-High, 4-Very High',
    'Gender': "Employee's gender",
    'HourlyRate': 'Hourly salary',
    'JobInvolvement': '1-Low, 2-Medium, 3-High, 4-Very High',
    'JobLevel': 'Level of job (1 to 5)',
    'JobRole': 'Job Roles',
    'JobSatisfaction': '1-Low, 2-Medium, 3-High, 4-Very High',
    'MaritalStatus': 'Marital Status',
    'MonthlyIncome': 'Monthly salary',
    'MonthlyRate': 'Monthly rate',
    'NumCompaniesWorked': 'Number of companies worked at',
    'Over18': 'Over 18 years of age?',
    'OverTime': 'Overtime?',
    'PercentSalaryHike': 'The percentage increase in salary last year',
    'PerformanceRating': '1-Low, 2-Good, 3-Excellent, 4-Outstanding',
    'RelationshipSatisfaction': '1-Low, 2-Medium, 3-High, 4-Very High',
    'StandardHours': 'Standard Hours',
    'StockOptionLevel': 'Stock Option Level',
    'TotalWorkingYears': 'Total years worked',
    'TrainingTimesLastYear': 'Number of training attended last year',
    'WorkLifeBalance': '1-Low, 2-Good, 3-Excellent, 4-Outstanding',
    'YearsAtCompany': 'Years at Company',
    'YearsInCurrentRole': 'Years in the current role',
    'YearsSinceLastPromotion': 'Years since the last promotion',
    'YearsWithCurrManager': 'Years with the current manager',
}


def resolve_dataset_path():
    app_dir = Path(__file__).resolve().parent
    candidate_paths = [
        app_dir.parent / 'employee_data.csv',
        app_dir / 'employee_data.csv',
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    return None


def format_numeric_placeholder(value):
    if pd.isna(value):
        return ''
    return str(int(round(value)))


def format_numeric_limit(value):
    if pd.isna(value):
        return ''
    if float(value).is_integer():
        return str(int(value))
    return f"{value}".rstrip('0').rstrip('.')


def load_numeric_field_metadata():
    numeric_columns = [
        name for name in feature_names
        if name in NUMERIC_FEATURES and name in DATASET_COLUMNS
    ]
    if not numeric_columns:
        return {}, {}

    dataset_path = resolve_dataset_path()
    if dataset_path is None:
        return {}, {}

    try:
        df = pd.read_csv(dataset_path, usecols=numeric_columns)
    except Exception:
        return {}, {}

    placeholders = {}
    max_values = {}
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors='coerce')
        if series.notna().any():
            placeholders[column] = format_numeric_placeholder(series.mean())
            max_value = NUMERIC_MAX_OVERRIDES.get(column, series.max())
            max_values[column] = format_numeric_limit(max_value)
    return placeholders, max_values


def build_form_fields():
    fields = []
    for name in feature_names:
        if name in CATEGORICAL_OPTIONS:
            fields.append({
                'name': name,
                'label': name,
                'type': 'select',
                'options': CATEGORICAL_OPTIONS[name],
                'description': FIELD_DESCRIPTIONS.get(name, ''),
                'placeholder': '',
                'max': '',
            })
        else:
            fields.append({
                'name': name,
                'label': name,
                'type': 'number',
                'options': [],
                'description': FIELD_DESCRIPTIONS.get(name, ''),
                'placeholder': NUMERIC_PLACEHOLDERS.get(name, ''),
                'max': NUMERIC_MAX_VALUES.get(name, ''),
            })
    return fields

def tokenize_pasted_row(raw_text):
    if not raw_text:
        return []

    normalized_text = raw_text.replace('\r\n', '\n').replace('\r', '\n').strip()
    if not normalized_text:
        return []

    if '\t' in normalized_text:
        separator = '\t'
    elif '\n' in normalized_text:
        separator = '\n'
    elif ';' in normalized_text:
        separator = ';'
    else:
        separator = ','

    return [value.strip() for value in normalized_text.split(separator)]


def score_column_variant(columns, values):
    if len(columns) != len(values):
        return None

    mapped_values = dict(zip(columns, values))
    score = 0
    for feature in feature_names:
        value = mapped_values.get(feature, '')
        if value == '':
            continue

        if feature in NUMERIC_FEATURES:
            try:
                float(value)
                score += 2
            except ValueError:
                score -= 2
        elif value in label_encoders[feature].classes_:
            score += 3
        else:
            score -= 3

    return score


def parse_pasted_row(raw_text):
    values = tokenize_pasted_row(raw_text)
    if not values:
        return {}

    best_variant = None
    best_score = None
    for label, columns in COLUMN_VARIANTS:
        score = score_column_variant(columns, values)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_variant = (label, columns)
            best_score = score

    if best_variant is None:
        accepted_lengths = ', '.join(str(len(columns)) for _, columns in COLUMN_VARIANTS)
        raise ValueError(
            "Jumlah nilai tidak sesuai. "
            f"Diterima {len(values)} nilai, format yang didukung memiliki {accepted_lengths} nilai."
        )

    _, matched_columns = best_variant
    mapped_values = dict(zip(matched_columns, values))
    return {feature: mapped_values.get(feature, '') for feature in feature_names}


def empty_form_values():
    return {field['name']: '' for field in FORM_FIELDS}


def build_template_context(**overrides):
    context = {
        'form_fields': FORM_FIELDS,
        'form_values': empty_form_values(),
        'pasted_row': '',
        'paste_error': None,
        'result': None,
        'proba': None,
        'save_status': None,
        'save_error': None,
        'dataset_columns': DATASET_COLUMNS,
        'active_columns': feature_names,
        'ignored_columns': IGNORED_COLUMNS,
        'categorical_options': CATEGORICAL_OPTIONS,
        'numeric_fields': sorted(NUMERIC_FEATURES),
        'importance_features': IMPORTANCE_FEATURES,
    }
    context.update(overrides)
    return context

@app.route('/')
def home():
    try:
        return render_template('home.html')
    except Exception:
        return "Service is running", 200


@app.route('/healthz')
@app.route('/healtz')
def healthz():
    if model_is_loaded():
        return {"status": "ok", "model_loaded": True}, 200
    if MODEL_LOAD_ERROR:
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": MODEL_LOAD_ERROR,
        }, 200
    return {"status": "ok", "model_loaded": False, "lazy_load": True}, 200

@app.route('/dashboard')
def dashboard():
    try:
        return render_template('dashboard_view.html')
    except Exception:
        return "Dashboard unavailable", 200

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    ensure_model_loaded()
    if model is None:
        return render_template(
            'predict_view.html',
            **build_template_context(
                paste_error=f"Model belum siap: {MODEL_LOAD_ERROR}",
            )
        )
    form_values = {
        field['name']: request.form.get(field['name'], '').strip()
        for field in FORM_FIELDS
    }
    pasted_row = request.form.get('pasted_row', '').strip()

    if request.method == 'POST':
        if pasted_row:
            try:
                parsed_values = parse_pasted_row(pasted_row)
            except ValueError as exc:
                return render_template(
                    'predict_view.html',
                    **build_template_context(
                        form_values=form_values,
                        pasted_row=pasted_row,
                        paste_error=str(exc),
                    )
                )

            for feature, value in parsed_values.items():
                form_values[feature] = value

        importance_mode = request.form.get('importance_mode') == 'true'
        if importance_mode:
            for field in FORM_FIELDS:
                name = field['name']
                if name not in IMPORTANCE_FEATURES and not form_values.get(name):
                    if name in CATEGORICAL_OPTIONS and CATEGORICAL_OPTIONS[name]:
                        form_values[name] = str(CATEGORICAL_OPTIONS[name][0])
                    else:
                        form_values[name] = str(NUMERIC_PLACEHOLDERS.get(name, '0') or '0')

        missing_fields = [
            field['label']
            for field in FORM_FIELDS
            if not form_values[field['name']]
        ]
        if missing_fields:
            return render_template(
                'predict_view.html',
                **build_template_context(
                    form_values=form_values,
                    pasted_row=pasted_row,
                    paste_error='Masih ada field kosong. Lengkapi semua input sebelum prediksi.',
                )
            )

        negative_fields = []
        for field in FORM_FIELDS:
            name = field['name']
            if name in NUMERIC_FEATURES:
                try:
                    if float(form_values[name]) < 0:
                        negative_fields.append(field['label'])
                except (ValueError, TypeError):
                    pass
        if negative_fields:
            return render_template(
                'predict_view.html',
                **build_template_context(
                    form_values=form_values,
                    pasted_row=pasted_row,
                    paste_error=f'Nilai tidak boleh negatif: {", ".join(negative_fields)}.',
                )
            )

        df_input = pd.DataFrame([form_values], columns=feature_names)
        prediction, proba = predict_attrition(df_input, model, scaler, label_encoders)
        result = "Attrition" if prediction == 1 else "No Attrition"
        save_status = None
        save_error = None
        try:
            saved, message = append_prediction(
                form_values=form_values,
                result=result,
                proba=proba,
                feature_names=feature_names,
                source='web_form',
                prediction_value=prediction,
            )
            if saved:
                save_status = message
            else:
                save_error = message
        except Exception as exc:
            save_error = f"Gagal menyimpan ke spreadsheet: {exc}"

        return render_template(
            'predict_view.html',
            **build_template_context(
                form_values=form_values,
                pasted_row=pasted_row,
                result=result,
                proba=proba,
                save_status=save_status,
                save_error=save_error,
            )
        )

    return render_template('predict_view.html', **build_template_context())

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
