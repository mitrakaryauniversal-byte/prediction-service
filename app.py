import logging
from pathlib import Path
from datetime import date, datetime

import joblib
import numpy as np
import pymysql

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API Prediksi Stok Ban",
    description="API prediksi stok ban menggunakan model LSTM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model_lstm_stok_ban.keras"
SCALER_PATH = BASE_DIR / "scaler_stok_ban.pkl"


DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "ptmitrakarya",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "connect_timeout": 5,
    "read_timeout": 10,
    "write_timeout": 10,
}


try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"File scaler tidak ditemukan: {SCALER_PATH}"
        )

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError(
            "Scaler tidak memiliki atribut mean_ atau scale_."
        )

    if len(scaler.mean_) < 3 or len(scaler.scale_) < 3:
        raise ValueError(
            "Scaler harus memiliki minimal tiga fitur."
        )

    logger.info("Model LSTM berhasil dimuat.")
    logger.info("Scaler berhasil dimuat.")

except Exception:
    logger.exception("Gagal memuat model atau scaler.")
    raise


class PredictionInput(BaseModel):
    product_id: str = Field(
        ...,
        min_length=1,
        description="Kode produk ban, misalnya T001"
    )



def get_connection():
    """Membuat koneksi ke database MySQL."""
    return pymysql.connect(**DB_CONFIG)


def format_date(value):
    """
    Mengubah nilai tanggal menjadi teks YYYY-MM-DD.
    Mendukung objek date, datetime, string, dan None.
    """

    if value is None:
        return None

    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")

    # Jika tanggal dari database sudah berbentuk string.
    text_value = str(value).strip()

    if not text_value:
        return None

    # Mengambil bagian tanggal jika ada komponen waktu.
    return text_value[:10]


def clean_number(value):
    """Mengubah nilai numerik NumPy/Python menjadi float."""
    return float(value)


def get_product_history(product_id):
    """
    Mengambil tiga data terakhir untuk produk tertentu,
    kemudian mengurutkannya dari tanggal terlama ke terbaru.
    """

    connection = None

    try:
        connection = get_connection()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    tanggal,
                    product_id,
                    tyre_type,
                    opening_stock,
                    purchase_qty,
                    closing_stock,
                    kuartal,
                    tahun
                FROM riwayat_stok_ban
                WHERE product_id = %s
                ORDER BY tanggal DESC, id DESC
                LIMIT 3
                """,
                (product_id,)
            )

            rows = cursor.fetchall()

        # Hasil query berurutan terbaru ke terlama.
        # Balik urutannya agar menjadi kronologis.
        rows.reverse()

        return rows

    finally:
        if connection is not None:
            connection.close()



@app.get("/health")
def health():
    """Memeriksa status API, model, dan koneksi database."""

    database_status = "connected"
    connection = None

    try:
        connection = get_connection()

        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()

    except Exception:
        database_status = "disconnected"
        logger.exception("Pemeriksaan koneksi database gagal.")

    finally:
        if connection is not None:
            connection.close()

    return {
        "success": True,
        "status": "running",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "database": database_status,
    }



@app.get("/products")
def get_products():
    """Mengambil daftar product_id untuk dropdown halaman PHP."""

    connection = None

    try:
        connection = get_connection()

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT TRIM(product_id) AS product_id
                FROM riwayat_stok_ban
                WHERE product_id IS NOT NULL
                  AND TRIM(product_id) <> ''
                  AND LOWER(TRIM(product_id)) <> 'product_id'
                ORDER BY product_id ASC
                """
            )

            rows = cursor.fetchall()

        products = [
            row["product_id"]
            for row in rows
            if row.get("product_id")
        ]

        return {
            "success": True,
            "total_produk": len(products),
            "products": products,
        }

    except Exception as exc:
        logger.exception("Gagal mengambil daftar produk.")

        raise HTTPException(
            status_code=500,
            detail="Gagal mengambil daftar produk dari database."
        ) from exc

    finally:
        if connection is not None:
            connection.close()


@app.post("/predict")
def predict(data: PredictionInput):
    """
    Memprediksi closing_stock berdasarkan tiga catatan terakhir
    dari produk yang dipilih.
    """

    try:
        product_id = data.product_id.strip()

        if not product_id:
            raise HTTPException(
                status_code=400,
                detail="Kode produk tidak boleh kosong."
            )


        rows = get_product_history(product_id)

        if len(rows) < 3:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Produk {product_id} hanya memiliki "
                    f"{len(rows)} data. Minimal diperlukan "
                    "3 catatan stok untuk prediksi."
                )
            )


        feature_columns = [
            "opening_stock",
            "purchase_qty",
            "closing_stock",
        ]

        features = []

        for row in rows:
            feature_row = []

            for column in feature_columns:
                value = row.get(column)

                if value is None:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Data {column} untuk produk "
                            f"{product_id} masih kosong."
                        )
                    )

                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Nilai {column} pada produk "
                            f"{product_id} bukan angka yang valid."
                        )
                    ) from exc

                if not np.isfinite(number):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Nilai {column} pada produk "
                            f"{product_id} tidak valid."
                        )
                    )

                feature_row.append(number)

            features.append(feature_row)

        # Bentuk data: (1 sampel, 3 time steps, 3 fitur)
        input_array = np.array(
            features,
            dtype=np.float32
        ).reshape(1, 3, 3)


        original_shape = input_array.shape

        scaled_input = scaler.transform(
            input_array.reshape(-1, 3)
        ).reshape(original_shape)

        prediction_scaled = model.predict(
            scaled_input,
            verbose=0
        )

        prediction_scaled_value = float(
            np.asarray(prediction_scaled).reshape(-1)[0]
        )

        prediction_value = (
            prediction_scaled_value * float(scaler.scale_[2])
            + float(scaler.mean_[2])
        )

        if not np.isfinite(prediction_value):
            raise HTTPException(
                status_code=500,
                detail="Model menghasilkan prediksi yang tidak valid."
            )

        # Stok prediksi ditampilkan dua angka desimal.
        prediction_value = round(prediction_value, 2)

        history = []

        for row in rows:
            history.append({
                "tanggal": format_date(row.get("tanggal")),
                "product_id": str(row.get("product_id", "")),
                "tyre_type": str(row.get("tyre_type", "")),
                "opening_stock": clean_number(
                    row["opening_stock"]
                ),
                "purchase_qty": clean_number(
                    row["purchase_qty"]
                ),
                "closing_stock": clean_number(
                    row["closing_stock"]
                ),
                "kuartal": (
                    str(row["kuartal"])
                    if row.get("kuartal") is not None
                    else None
                ),
                "tahun": (
                    int(row["tahun"])
                    if row.get("tahun") is not None
                    else None
                ),
            })

        last_row = rows[-1]


        return {
            "success": True,
            "product_id": product_id,
            "tyre_type": str(last_row.get("tyre_type", "")),
            "prediksi_stok": prediction_value,
            "tanggal_data_terakhir": format_date(
                last_row.get("tanggal")
            ),
            "jumlah_data_digunakan": len(rows),
            "riwayat": history,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediksi gagal.")

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediksi gagal diproses. "
                "Periksa log server untuk informasi lebih lanjut."
            )
        ) from exc
