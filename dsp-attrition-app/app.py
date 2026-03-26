from flask import Flask, render_template, request
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from model_util import load_artifacts, predict_attrition

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


def ensure_model_loaded():
    global model, scaler, label_encoders, feature_names, MODEL_LOAD_ERROR
    if model is not None and scaler is not None and feature_names:
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
    global CATEGORICAL_OPTIONS, IGNORED_COLUMNS, NUMERIC_FEATURES, COLUMN_VARIANTS, FORM_FIELDS
    CATEGORICAL_OPTIONS = {
        name: list(encoder.classes_)
        for name, encoder in label_encoders.items()
    }
    IGNORED_COLUMNS = [col for col in DATASET_COLUMNS if col not in feature_names]
    NUMERIC_FEATURES = {name for name in feature_names if name not in label_encoders}
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


def build_form_fields():
    fields = []
    for name in feature_names:
        if name in CATEGORICAL_OPTIONS:
            fields.append({
                'name': name,
                'label': name,
                'type': 'select',
                'options': CATEGORICAL_OPTIONS[name],
            })
        else:
            fields.append({
                'name': name,
                'label': name,
                'type': 'number',
                'options': [],
            })
    return fields


ensure_model_loaded()


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
        'dataset_columns': DATASET_COLUMNS,
        'active_columns': feature_names,
        'ignored_columns': IGNORED_COLUMNS,
        'categorical_options': CATEGORICAL_OPTIONS,
        'numeric_fields': sorted(NUMERIC_FEATURES),
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
def healthz():
    if MODEL_LOAD_ERROR:
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": MODEL_LOAD_ERROR,
        }, 200
    return {"status": "ok", "model_loaded": True}, 200

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

        df_input = pd.DataFrame([form_values], columns=feature_names)
        prediction, proba = predict_attrition(df_input, model, scaler, label_encoders)
        result = "Attrition" if prediction == 1 else "No Attrition"

        return render_template(
            'predict_view.html',
            **build_template_context(
                form_values=form_values,
                pasted_row=pasted_row,
                result=result,
                proba=proba,
            )
        )

    return render_template('predict_view.html', **build_template_context())

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
