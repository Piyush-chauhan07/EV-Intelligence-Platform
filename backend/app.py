import os
import pickle
import traceback
import math

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
ML_DIR = os.path.join(BASE_DIR, "ml")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

EV_FILE = os.path.join(DATA_DIR, "india_ev_range_clean.csv")
PM_FILE = os.path.join(DATA_DIR, "pm_edrive_ev_sales.csv")
TWO_THREE_FILE = os.path.join(DATA_DIR, "india_ev_2w_3w.csv")
MODEL_FILE = os.path.join(ML_DIR, "ev_price_model.pkl")


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# GLOBAL DATA
# ============================================================

ev_df = pd.DataFrame()
pm_df = pd.DataFrame()
two_three_df = pd.DataFrame()

model = None


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    """
    Convert pandas / numpy values into JSON-safe Python values.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def clean_records(df):
    """
    Convert DataFrame to JSON-safe records.
    """

    if df is None or df.empty:
        return []

    records = df.to_dict(orient="records")

    cleaned = []

    for record in records:
        cleaned_record = {}

        for key, value in record.items():
            cleaned_record[str(key)] = clean_value(value)

        cleaned.append(cleaned_record)

    return cleaned


def find_column(df, possible_names):
    """
    Find a column regardless of capitalization or minor naming differences.
    """

    if df is None or df.empty:
        return None

    normalized = {}

    for col in df.columns:
        key = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        normalized[key] = col

    for name in possible_names:
        key = (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        if key in normalized:
            return normalized[key]

    return None


def find_column_contains(df, keywords):
    """
    Find a column whose normalized name contains one of the keywords.
    """

    if df is None or df.empty:
        return None

    for col in df.columns:
        normalized = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

        for keyword in keywords:
            if keyword in normalized:
                return col

    return None


def dataframe_json(df):
    """
    JSON response helper.
    """

    return jsonify(clean_records(df))


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("=" * 60)
print("🚗 EV INTELLIGENCE PLATFORM")
print("=" * 60)
print()


# ------------------------------------------------------------
# 4-WHEELER DATASET
# ------------------------------------------------------------

print("Loading EV vehicle dataset...")
print(EV_FILE)

try:
    ev_df = pd.read_csv(EV_FILE)

    print("✅ EV vehicle dataset loaded")
    print("Rows:", len(ev_df))
    print("Columns:", len(ev_df.columns))
    print("Column names:", ev_df.columns.tolist())

except Exception as e:
    print("❌ Could not load EV dataset")
    print(e)
    ev_df = pd.DataFrame()


# ------------------------------------------------------------
# PM E-DRIVE DATASET
# ------------------------------------------------------------

print()
print("Loading PM E-DRIVE dataset...")
print(PM_FILE)

try:
    pm_df = pd.read_csv(PM_FILE)

    print("✅ PM E-DRIVE dataset loaded")
    print("Rows:", len(pm_df))
    print("Columns:", len(pm_df.columns))
    print("Column names:", pm_df.columns.tolist())

except Exception as e:
    print("❌ Could not load PM E-DRIVE dataset")
    print(e)
    pm_df = pd.DataFrame()


# ------------------------------------------------------------
# 2W / 3W DATASET
# ------------------------------------------------------------

print()
print("Loading 2W / 3W vehicle dataset...")
print(TWO_THREE_FILE)

try:
    two_three_df = pd.read_csv(TWO_THREE_FILE)

    print("✅ 2W / 3W dataset loaded")
    print("Rows:", len(two_three_df))
    print("Columns:", len(two_three_df.columns))
    print("Column names:", two_three_df.columns.tolist())

except Exception as e:
    print("❌ Could not load 2W / 3W dataset")
    print(e)
    two_three_df = pd.DataFrame()


# ============================================================
# PRICE COLUMN DETECTION
# ============================================================

price_column = find_column(
    ev_df,
    [
        "price_lakh_inr",
        "price_lakh_INR",
        "price",
        "price_inr",
    ],
)

if price_column:
    print()
    print("Detected price column:", price_column)


# ============================================================
# ML MODEL
# ============================================================

print()
print("Loading ML model...")
print("Model file:", MODEL_FILE)

try:

    if os.path.exists(MODEL_FILE):

        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)

        print("✅ ML model loaded")

    else:

        print("⚠️ ML model file not found")
        model = None

except Exception as e:

    print("⚠️ ML model could not be loaded")
    print(e)

    model = None


# ============================================================
# API INFORMATION
# ============================================================

API_INFO = {
    "service": "EV Intelligence Platform API",
    "status": "online",
    "server": "http://127.0.0.1:5050",
    "endpoints": [
        "/",
        "/status",
        "/vehicles",
        "/vehicles/<make>",
        "/vehicle/<make>/<model>",
        "/vehicles-2w-3w",
        "/vehicles-2w-3w/2w",
        "/vehicles-2w-3w/3w",
        "/predict",
        "/ev-adoption",
        "/ev-adoption/summary",
    ],
}


# ============================================================
# ROOT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        **API_INFO,
        "message": "EV Intelligence Platform backend is running.",
    })


# ============================================================
# STATUS
# ============================================================

@app.route("/status", methods=["GET"])
def status():

    return jsonify({
        "service": "EV Intelligence Platform API",
        "status": "online",
        "server": "http://127.0.0.1:5050",

        "ev_vehicle_rows": int(len(ev_df)),
        "pm_edrive_rows": int(len(pm_df)),
        "two_three_wheeler_rows": int(len(two_three_df)),

        "two_three_wheeler_columns": int(len(two_three_df.columns)),

        "ml_model_loaded": model is not None,

        "endpoints": API_INFO["endpoints"],
    })


# ============================================================
# 4-WHEELER VEHICLES
# ============================================================

@app.route("/vehicles", methods=["GET"])
def get_vehicles():

    try:

        if ev_df.empty:
            return jsonify({
                "error": "EV vehicle dataset is empty",
                "count": 0,
                "vehicles": [],
            }), 500

        data = clean_records(ev_df)

        return jsonify({
            "count": len(data),
            "vehicles": data,
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e),
        }), 500


# ============================================================
# VEHICLES BY MAKE
# ============================================================

@app.route("/vehicles/<make>", methods=["GET"])
def get_vehicles_by_make(make):

    try:

        if ev_df.empty:
            return jsonify({
                "count": 0,
                "vehicles": [],
            })

        make_col = find_column(
            ev_df,
            [
                "make",
                "brand",
                "manufacturer",
            ],
        )

        if make_col is None:

            return jsonify({
                "error": "Make/brand column not found",
                "available_columns": ev_df.columns.tolist(),
            }), 500

        mask = (
            ev_df[make_col]
            .astype(str)
            .str.strip()
            .str.lower()
            == make.strip().lower()
        )

        result = ev_df[mask]

        return jsonify({
            "make": make,
            "count": len(result),
            "vehicles": clean_records(result),
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e),
        }), 500


# ============================================================
# SINGLE VEHICLE
# ============================================================

@app.route("/vehicle/<make>/<model_name>", methods=["GET"])
def get_single_vehicle(make, model_name):

    try:

        if ev_df.empty:
            return jsonify({
                "error": "EV dataset is empty",
            }), 500

        make_col = find_column(
            ev_df,
            [
                "make",
                "brand",
                "manufacturer",
            ],
        )

        model_col = find_column(
            ev_df,
            [
                "model",
                "vehicle_model",
            ],
        )

        if make_col is None or model_col is None:

            return jsonify({
                "error": "Make or model column not found",
                "available_columns": ev_df.columns.tolist(),
            }), 500

        mask = (
            ev_df[make_col].astype(str).str.strip().str.lower()
            == make.strip().lower()
        ) & (
            ev_df[model_col].astype(str).str.strip().str.lower()
            == model_name.strip().lower()
        )

        result = ev_df[mask]

        return jsonify({
            "make": make,
            "model": model_name,
            "count": len(result),
            "vehicles": clean_records(result),
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e),
        }), 500


# ============================================================
# 2W / 3W ALL
# ============================================================

@app.route("/vehicles-2w-3w", methods=["GET"])
def get_two_three():

    try:

        if two_three_df.empty:
            return jsonify({
                "count": 0,
                "vehicles": [],
            })

        return jsonify({
            "count": len(two_three_df),
            "vehicles": clean_records(two_three_df),
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e),
        }), 500


# ============================================================
# 2W
# ============================================================

@app.route("/vehicles-2w-3w/2w", methods=["GET"])
def get_two_wheelers():

    try:

        if two_three_df.empty:
            return jsonify({
                "count": 0,
                "vehicles": [],
            })

        vehicle_type_col = find_column(
            two_three_df,
            [
                "vehicle_type",
                "vehicle type",
                "type",
            ],
        )

        if vehicle_type_col is None:

            return jsonify({
                "error": "vehicle_type column not found",
                "available_columns": two_three_df.columns.tolist(),
            }), 500

        values = (
            two_three_df[vehicle_type_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mask = (
            values.str.contains("2w")
            | values.str.contains("2-wheeler")
            | values.str.contains("2 wheeler")
            | values.str.contains("two")
        )

        result = two_three_df[mask]

        return jsonify({
            "count": len(result),
            "vehicles": clean_records(result),
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e),
        }), 500


# ============================================================
# 3W
# ============================================================

@app.route("/vehicles-2w-3w/3w", methods=["GET"])
def get_three_wheelers():

    try:

        if two_three_df.empty:
            return jsonify({
                "count": 0,
                "vehicles": [],
            })

        vehicle_type_col = find_column(
            two_three_df,
            [
                "vehicle_type",
                "vehicle type",
                "type",
            ],
        )

        if vehicle_type_col is None:

            return jsonify({
                "error": "vehicle_type column not found",
                "available_columns": two_three_df.columns.tolist(),
            }), 500

        values = (
            two_three_df[vehicle_type_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mask = (
            values.str.contains("3w")
            | values.str.contains("3-wheeler")
            | values.str.contains("3 wheeler")
            | values.str.contains("three")
        )

        result = two_three_df[mask]

        return jsonify({
            "count": len(result),
            "vehicles": clean_records(result),
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e),
        }), 500


# ============================================================
# PRICE PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict_price():

    try:

        data = request.get_json(silent=True) or {}

        brand = data.get("brand", "")
        range_value = data.get("range", data.get("range_km"))
        power_value = data.get("power", data.get("power_kw"))
        battery_value = data.get(
            "battery",
            data.get("battery_kwh")
        )

        if range_value is None:
            return jsonify({
                "error": "Range is required"
            }), 400

        if power_value is None:
            return jsonify({
                "error": "Power is required"
            }), 400

        if battery_value is None:
            return jsonify({
                "error": "Battery is required"
            }), 400

        try:
            range_value = float(range_value)
            power_value = float(power_value)
            battery_value = float(battery_value)
        except Exception:
            return jsonify({
                "error": "Range, power and battery must be numeric"
            }), 400


        # ----------------------------------------------------
        # TRY ML MODEL
        # ----------------------------------------------------

        if model is not None:

            try:

                # First try a dictionary/DataFrame-style prediction.
                input_df = pd.DataFrame([{
                    "range": range_value,
                    "power": power_value,
                    "battery": battery_value,
                }])

                prediction = model.predict(input_df)

                predicted = float(prediction[0])

                return jsonify({
                    "success": True,
                    "brand": brand,
                    "predicted_price_lakh_inr": round(predicted, 2),
                    "prediction_source": "machine_learning_model",
                })

            except Exception:

                # Some sklearn pipelines require a different feature set.
                # Fall through to a transparent fallback estimator.
                pass


        # ----------------------------------------------------
        # FALLBACK ESTIMATE
        # ----------------------------------------------------

        # This keeps the dashboard functional even if a saved model
        # was trained with a different feature schema.

        estimated_price = (
            1.8
            + (battery_value * 0.32)
            + (power_value * 0.025)
            + (range_value * 0.008)
        )

        if estimated_price < 2.0:
            estimated_price = 2.0

        return jsonify({
            "success": True,
            "brand": brand,
            "predicted_price_lakh_inr": round(
                estimated_price,
                2
            ),
            "prediction_source": "fallback_estimator",
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e),
        }), 500


# ============================================================
# EV ADOPTION
# ============================================================

@app.route("/ev-adoption", methods=["GET"])
def get_ev_adoption():

    try:

        if pm_df.empty:
            return jsonify([])

        return jsonify(clean_records(pm_df))

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e),
        }), 500


# ============================================================
# EV ADOPTION SUMMARY
# ============================================================

@app.route("/ev-adoption/summary", methods=["GET"])
def get_ev_adoption_summary():

    try:

        if pm_df.empty:

            return jsonify({
                "total_2w": 0,
                "total_3w": 0,
                "total_ev": 0,
            })

        two_w_col = find_column(
            pm_df,
            [
                "e-2w",
                "e_2w",
                "2w",
                "electric_2w",
            ],
        )

        three_w_col = find_column(
            pm_df,
            [
                "e-3w",
                "e_3w",
                "3w",
                "electric_3w",
            ],
        )

        total_col = find_column(
            pm_df,
            [
                "total",
                "total_ev",
                "ev_total",
            ],
        )

        def numeric_sum(column):

            if column is None:
                return 0

            return int(
                pd.to_numeric(
                    pm_df[column],
                    errors="coerce"
                )
                .fillna(0)
                .sum()
            )

        total_2w = numeric_sum(two_w_col)
        total_3w = numeric_sum(three_w_col)

        if total_col:
            total_ev = numeric_sum(total_col)
        else:
            total_ev = total_2w + total_3w

        state_col = find_column(
            pm_df,
            [
                "state_ut",
                "state",
                "state_name",
                "state/ut",
                "state_ut_name",
            ],
        )

        state_count = 0
        if state_col is not None:
            state_values = (
                pm_df[state_col]
                .dropna()
                .astype(str)
                .str.strip()
            )
            state_values = state_values[state_values != ""]
            state_count = int(state_values.nunique())

        return jsonify({
            "total_2w": total_2w,
            "total_3w": total_3w,
            "total_ev": total_ev,
            "states": state_count,
            "state_count": state_count,
            "total_records": int(len(pm_df)),
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e),
        }), 500


# ============================================================
# OPTIONAL FRONTEND FROM FLASK
# ============================================================

@app.route("/dashboard", methods=["GET"])
def dashboard():

    dashboard_file = os.path.join(
        FRONTEND_DIR,
        "dashboard.html"
    )

    if os.path.exists(dashboard_file):

        return send_from_directory(
            FRONTEND_DIR,
            "dashboard.html"
        )

    return jsonify({
        "error": "dashboard.html not found"
    }), 404


@app.route("/frontend/<path:filename>")
def frontend_files(filename):

    return send_from_directory(
        FRONTEND_DIR,
        filename
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "error": "Endpoint not found",
        "path": request.path,
    }), 404


@app.errorhandler(500)
def server_error(error):

    return jsonify({
        "error": "Internal server error",
        "details": str(error),
    }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("🚗 EV INTELLIGENCE PLATFORM API")
    print("=" * 60)
    print()
    print("Server: http://127.0.0.1:5050")
    print()
    print("Available endpoints:")
    print("GET  /")
    print("GET  /status")
    print("GET  /vehicles")
    print("GET  /vehicles/<make>")
    print("GET  /vehicle/<make>/<model>")
    print("GET  /vehicles-2w-3w")
    print("GET  /vehicles-2w-3w/2w")
    print("GET  /vehicles-2w-3w/3w")
    print("POST /predict")
    print("GET  /ev-adoption")
    print("GET  /ev-adoption/summary")
    print()
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=5050,
        debug=False,
    )