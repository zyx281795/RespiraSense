# =========================================================
# ResNet18 + Log-Mel Spectrogram.py
# 用途：
# 1. 可做模型訓練
# 2. 可做批次推論
# 3. 可做 Node-RED 單筆 wav 推論（CLI 模式）
# 4. 保留原本輸出資料夾
# 5. 輸出臨床 UI 需要的欄位
# =========================================================

import os
import sys
import json
import copy
import getpass
import re
import traceback
import wave
import time
from math import gcd
from collections import Counter
from typing import List, Dict, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # 不開視窗，直接存圖
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pymysql
from scipy.signal import resample_poly, spectrogram

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)

import torchvision
from torchvision import models

# OpenAI 套件改成可選，沒有也能跑
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================================================
# 0. 統一 log 函式
# 說明：
# - 一般狀態訊息全部印到 stderr
# - Node-RED 單筆模式時，stdout 只保留 JSON
# =========================================================
def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# =========================================================
# 1. 基本設定區
# =========================================================

DB_CONFIG = {
    "host": "127.0.0.1",                     # MySQL 主機
    "user": "root",                          # MySQL 帳號
    "database": "respiratory_audio_db",      # 資料庫名稱
    "charset": "utf8mb4"                     # 連線編碼
}
# 注意：密碼不寫在程式裡，執行時輸入

_DB_PASSWORD = None
# 暫存執行時輸入的 MySQL 密碼

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.join(BASE_DIR, "ResNet18_results")
# 所有輸出都維持原本這個資料夾

CHECKPOINT_DIR = os.path.join(RESULT_ROOT, "checkpoints")
REPORT_DIR = os.path.join(RESULT_ROOT, "reports")
SPECTROGRAM_DIR = os.path.join(RESULT_ROOT, "spectrograms")
SUMMARY_TXT_DIR = os.path.join(RESULT_ROOT, "summary_txt")
EXPORT_DIR = os.path.join(RESULT_ROOT, "prediction_exports")

DATASET_ROOT = os.path.join(BASE_DIR, "Asthma Detection Dataset Version 2", "Asthma Detection Dataset Version 2")
# 你的資料集根目錄

CLASS_FOLDER_MAP = {
    "asthma": "asthma",
    "Bronchial": "Bronchial",
    "copd": "copd",
    "healthy": "healthy",
    "pneumonia": "pneumonia"
}
# disease_type 對應資料夾

AUTO_REPAIR_DB_PATH = True
# 若資料庫中的 file_path 壞掉，就自動修復並可回寫 DB

# =========================================================
# 執行模式
# 說明：
# - 你現在要「重新跑」，所以這裡預設就是訓練 + 批次推論
# - 若之後只想跑推論，可把 RUN_TRAIN 改成 False
# =========================================================
RUN_TRAIN = True
RUN_BATCH_INFERENCE = True

AUTO_TRAIN_IF_CHECKPOINT_MISSING = True
# 若找不到 best_resnet18.pth，會自動先訓練

USE_PRETRAINED = False
# 是否使用 ImageNet 預訓練權重

NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
TEST_SIZE = 0.2
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["asthma", "Bronchial", "copd", "healthy", "pneumonia"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

TARGET_SR = 16000
TARGET_DURATION_SEC = 5
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
IMAGE_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.60

MODEL_NAME = "ResNet18_LogMel_ClinicalUI_Rerun_vFinal"

# =========================================================
# OpenAI 摘要設定
# 說明：
# - 預設關閉，因為你之後打算讓 Node-RED 去串接 API
# - 若你之後想讓 Python 直接呼叫 OpenAI，再改成 True
# =========================================================
ENABLE_LLM_SUMMARY = False
LLM_MODEL = "gpt-5.4-mini"
LLM_MAX_OUTPUT_TOKENS = 220
LLM_REASONING_EFFORT = "low"
_OPENAI_CLIENT = None


# =========================================================
# 2. 建立輸出資料夾
# =========================================================
os.makedirs(RESULT_ROOT, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(SPECTROGRAM_DIR, exist_ok=True)
os.makedirs(SUMMARY_TXT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)


# =========================================================
# 3. 資料庫連線與操作
# =========================================================

def get_db_password() -> str:
    """
    第一次連資料庫時，要求使用者輸入 MySQL 密碼。
    """
    global _DB_PASSWORD

    if _DB_PASSWORD is None:
        _DB_PASSWORD = getpass.getpass("請輸入 MySQL 密碼：")

    return _DB_PASSWORD


def get_db_connection():
    """
    建立 MySQL 連線。
    """
    return pymysql.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=get_db_password(),
        database=DB_CONFIG["database"],
        charset=DB_CONFIG["charset"],
        cursorclass=pymysql.cursors.DictCursor,
        init_command="SET NAMES utf8mb4"
    )


def load_audio_records() -> List[Dict]:
    """
    從 audio_files 讀取所有音檔資料。
    讀取欄位：
    - id
    - file_name
    - disease_type
    - file_path
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
            SELECT id, file_name, disease_type, file_path
            FROM audio_files
            WHERE disease_type IS NOT NULL
            ORDER BY id ASC
            """
            cursor.execute(sql)
            rows = cursor.fetchall()
        return rows
    finally:
        conn.close()


def update_audio_file_path(audio_id: int, new_path: str) -> None:
    """
    將修復後的 file_path 回寫到 audio_files。
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
            UPDATE audio_files
            SET file_path = %s
            WHERE id = %s
            """
            cursor.execute(sql, (new_path, audio_id))
        conn.commit()
    finally:
        conn.close()


def upsert_prediction_result(
    audio_id: int,
    file_name: str,
    true_class: str,
    predicted_class: str,
    confidence: float,
    top3_json: str,
    spectrogram_path: str,
    summary_text: str,
    model_name: str,
    remark: str
) -> None:
    """
    將推論結果寫入 prediction_results。
    若同一 audio_id + model_name 已存在，就更新；否則新增。
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            check_sql = """
            SELECT id
            FROM prediction_results
            WHERE audio_id = %s AND model_name = %s
            LIMIT 1
            """
            cursor.execute(check_sql, (audio_id, model_name))
            row = cursor.fetchone()

            if row:
                update_sql = """
                UPDATE prediction_results
                SET
                    file_name = %s,
                    true_class = %s,
                    predicted_class = %s,
                    confidence = %s,
                    top3_json = %s,
                    spectrogram_path = %s,
                    summary_text = %s,
                    inference_time = CURRENT_TIMESTAMP,
                    remark = %s
                WHERE id = %s
                """
                cursor.execute(
                    update_sql,
                    (
                        file_name,
                        true_class,
                        predicted_class,
                        confidence,
                        top3_json,
                        spectrogram_path,
                        summary_text,
                        remark,
                        row["id"]
                    )
                )
            else:
                insert_sql = """
                INSERT INTO prediction_results
                (
                    audio_id,
                    file_name,
                    true_class,
                    predicted_class,
                    confidence,
                    top3_json,
                    spectrogram_path,
                    summary_text,
                    model_name,
                    remark
                )
                VALUES
                (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """
                cursor.execute(
                    insert_sql,
                    (
                        audio_id,
                        file_name,
                        true_class,
                        predicted_class,
                        confidence,
                        top3_json,
                        spectrogram_path,
                        summary_text,
                        model_name,
                        remark
                    )
                )

        conn.commit()
    finally:
        conn.close()


# =========================================================
# 4. 路徑修復函式
# =========================================================

def normalize_windows_path(path_str: str) -> str:
    """
    清理與正規化 Windows 路徑字串。
    """
    path_str = str(path_str).strip().strip('"').strip("'")
    path_str = path_str.replace("/", "\\")
    path_str = re.sub(r"\\+", r"\\", path_str)

    drive_match = re.match(r"^([A-Za-z]):(?!\\)(.*)$", path_str)
    if drive_match:
        drive = drive_match.group(1)
        rest = drive_match.group(2)
        path_str = f"{drive}:\\{rest}"

    return path_str


def rebuild_path_from_root(file_name: str, disease_type: str) -> str:
    """
    依照資料集根目錄重建正確的完整路徑。
    """
    folder_name = CLASS_FOLDER_MAP[disease_type]
    return os.path.normpath(os.path.join(DATASET_ROOT, folder_name, file_name))


def repair_record_path(row: Dict, auto_update_db: bool = False) -> Optional[str]:
    """
    嘗試修復單筆資料的 file_path。
    優先順序：
    1. 原始路徑正規化後若存在，就用它
    2. 用 DATASET_ROOT + disease_type + file_name 重建
    3. 都不行就回傳 None
    """
    audio_id = row["id"]
    file_name = str(row["file_name"]).strip()
    disease_type = str(row["disease_type"]).strip()
    raw_path = str(row["file_path"]).strip()

    if disease_type not in CLASS_FOLDER_MAP:
        return None

    normalized_path = normalize_windows_path(raw_path)
    if os.path.exists(normalized_path):
        if auto_update_db and normalized_path != raw_path:
            update_audio_file_path(audio_id, normalized_path)
        return normalized_path

    rebuilt_path = rebuild_path_from_root(file_name, disease_type)
    if os.path.exists(rebuilt_path):
        if auto_update_db:
            update_audio_file_path(audio_id, rebuilt_path)
        return rebuilt_path

    return None


# =========================================================
# 5. WAV 讀檔與前處理
# =========================================================

def read_wav_with_wave(file_path: str) -> Tuple[np.ndarray, int]:
    """
    使用 Python 內建 wave 模組讀取 wav。
    避免依賴 soundfile / libsndfile.dll。
    """
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 1:
        audio = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0

    elif sampwidth == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        audio = audio / 32768.0

    elif sampwidth == 3:
        raw = np.frombuffer(raw_bytes, dtype=np.uint8)
        raw = raw.reshape(-1, 3)

        values = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )

        sign_mask = values & 0x800000
        values = values - (sign_mask > 0) * (1 << 24)
        audio = values.astype(np.float32) / float(1 << 23)

    elif sampwidth == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
        audio = audio / 2147483648.0

    else:
        raise ValueError(f"不支援的 WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio.astype(np.float32), sr


def can_read_wav(file_path: str) -> bool:
    """
    快速檢查 wav 是否可正常讀取。
    """
    try:
        _audio, _sr = read_wav_with_wave(file_path)
        return True
    except Exception:
        return False


def resample_audio_safely(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    用 scipy.signal.resample_poly 做重採樣。
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g

    audio = resample_poly(audio, up=up, down=down)
    return audio.astype(np.float32)


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """
    Hz 轉 Mel。
    """
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """
    Mel 轉 Hz。
    """
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None
) -> np.ndarray:
    """
    手動建立 Mel filter bank，不依賴 librosa。
    """
    if fmax is None:
        fmax = sr / 2

    mel_min = hz_to_mel(np.array([fmin], dtype=np.float32))[0]
    mel_max = hz_to_mel(np.array([fmax], dtype=np.float32))[0]

    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float32)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        if right > n_fft // 2 + 1:
            right = n_fft // 2 + 1

        for k in range(left, min(center, filterbank.shape[1])):
            filterbank[m - 1, k] = (k - left) / max(center - left, 1)

        for k in range(center, min(right, filterbank.shape[1])):
            filterbank[m - 1, k] = (right - k) / max(right - center, 1)

    return filterbank


def load_audio_fixed_length(file_path: str) -> np.ndarray:
    """
    讀 wav → 重採樣 → 固定長度。
    """
    audio, sr = read_wav_with_wave(file_path)
    audio = resample_audio_safely(audio, orig_sr=sr, target_sr=TARGET_SR)

    target_length = TARGET_SR * TARGET_DURATION_SEC

    if len(audio) < target_length:
        pad_len = target_length - len(audio)
        audio = np.pad(audio, (0, pad_len), mode="constant")
    elif len(audio) > target_length:
        audio = audio[:target_length]

    return audio.astype(np.float32)


def build_log_mel(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    建立 Log-Mel Spectrogram。
    回傳：
    1. normalized_spec
    2. log_mel_db
    """
    freqs, times, spec = spectrogram(
        audio,
        fs=TARGET_SR,
        window="hann",
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH,
        nfft=N_FFT,
        detrend=False,
        scaling="spectrum",
        mode="magnitude"
    )

    power_spec = spec ** 2

    mel_filterbank = create_mel_filterbank(
        sr=TARGET_SR,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=0.0,
        fmax=TARGET_SR / 2
    )

    mel_spec = np.dot(mel_filterbank, power_spec)
    mel_spec = np.maximum(mel_spec, 1e-10)

    log_mel_db = 10.0 * np.log10(mel_spec)

    min_val = np.min(log_mel_db)
    max_val = np.max(log_mel_db)
    normalized_spec = (log_mel_db - min_val) / (max_val - min_val + 1e-8)

    return normalized_spec.astype(np.float32), log_mel_db.astype(np.float32)


def spectrogram_to_tensor(spec_2d: np.ndarray) -> torch.Tensor:
    """
    將 2D 頻譜圖轉成 ResNet18 可接受的 tensor。
    """
    x = torch.from_numpy(spec_2d).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(
        x,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )
    x = x.squeeze(0)
    x = x.repeat(3, 1, 1)
    return x


def save_spectrogram(log_mel_db: np.ndarray, audio_id: int, file_name: str) -> str:
    """
    將 log-mel 頻譜圖存成 png。
    """
    base_name = os.path.splitext(file_name)[0]
    safe_name = base_name.replace(" ", "_")

    save_path = os.path.join(
        SPECTROGRAM_DIR,
        f"audio_{audio_id}_{safe_name}.png"
    )

    plt.figure(figsize=(8, 4))
    plt.imshow(log_mel_db, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title("Log-Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Bins")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path


# =========================================================
# 6. 臨床 UI 與指標 helper
# =========================================================

def safe_div(a: float, b: float) -> float:
    """
    安全除法，避免除以 0。
    """
    return float(a) / float(b) if b != 0 else 0.0


def get_wav_metadata(file_path: str) -> Dict:
    """
    取得原始 wav metadata，供 UI 顯示。
    """
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()

    duration_sec = n_frames / sr if sr > 0 else 0.0

    return {
        "original_sample_rate": sr,
        "duration_sec": float(duration_sec),
        "num_channels": int(n_channels),
        "sample_width_bytes": int(sampwidth)
    }


def compute_overall_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    計算整體 validation 指標。
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1)
    }


def compute_per_class_clinical_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> pd.DataFrame:
    """
    每類別輸出臨床可解讀指標：
    sensitivity, specificity, PPV, NPV, F1, support
    """
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    total = cm.sum()
    rows = []

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        sensitivity = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        ppv = safe_div(tp, tp + fp)
        npv = safe_div(tn, tn + fn)
        f1 = safe_div(2 * ppv * sensitivity, ppv + sensitivity)
        support = int(tp + fn)

        rows.append({
            "class_name": class_name,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "f1": f1,
            "support": support
        })

    return pd.DataFrame(rows)


def save_validation_metric_reports(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    report_dir: str
) -> Dict:
    """
    一次輸出 validation metrics：
    1. overall_validation_metrics.json
    2. per_class_clinical_metrics.csv
    3. classification_report.txt
    4. confusion_matrix.csv
    """
    overall_metrics = compute_overall_metrics(y_true, y_pred)
    per_class_df = compute_per_class_clinical_metrics(y_true, y_pred, class_names)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    with open(os.path.join(report_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(report_dir, "confusion_matrix.csv"),
        encoding="utf-8-sig"
    )

    per_class_df.to_csv(
        os.path.join(report_dir, "per_class_clinical_metrics.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    with open(os.path.join(report_dir, "overall_validation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(overall_metrics, f, ensure_ascii=False, indent=2)

    return overall_metrics


# =========================================================
# 7. 文本生成：模板 + 可選 LLM
# =========================================================

def get_openai_client():
    """
    建立 OpenAI client。
    若套件不存在或沒設 API key，就回傳 None。
    """
    global _OPENAI_CLIENT

    if OpenAI is None:
        return None

    if _OPENAI_CLIENT is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        _OPENAI_CLIENT = OpenAI(api_key=api_key)

    return _OPENAI_CLIENT


def build_fallback_summary_text(
    file_name: str,
    predicted_class: str,
    confidence: float,
    top3: List[Dict],
    abnormal_probability: float,
    review_flag: str
) -> str:
    """
    API 不可用時的保底模板摘要。
    """
    second_candidate = top3[1]["class"] if len(top3) > 1 else "N/A"
    third_candidate = top3[2]["class"] if len(top3) > 2 else "N/A"

    summary = (
        f"Primary model impression: {predicted_class} "
        f"(confidence {confidence:.2%}). "
        f"Estimated abnormal respiratory-pattern probability: {abnormal_probability:.2%}. "
        f"Differential considerations include {second_candidate} and {third_candidate}. "
    )

    if review_flag != "ok":
        summary += "Manual clinical review is recommended before final interpretation."
    else:
        summary += "No automatic review trigger was raised."

    summary += f" Source file: {file_name}."
    return summary


def generate_clinician_summary_with_llm(
    file_name: str,
    predicted_class: str,
    confidence: float,
    top3: List[Dict],
    review_flag: str,
    healthy_probability: Optional[float] = None,
    abnormal_probability: Optional[float] = None,
    duration_sec: Optional[float] = None,
    original_sample_rate: Optional[int] = None
) -> str:
    """
    若 ENABLE_LLM_SUMMARY=True 且 API 可用，就用 OpenAI 生成臨床摘要。
    否則退回模板摘要。
    """
    fallback_text = build_fallback_summary_text(
        file_name=file_name,
        predicted_class=predicted_class,
        confidence=confidence,
        top3=top3,
        abnormal_probability=abnormal_probability if abnormal_probability is not None else 0.0,
        review_flag=review_flag
    )

    if not ENABLE_LLM_SUMMARY:
        return fallback_text

    client = get_openai_client()
    if client is None:
        return fallback_text

    payload = {
        "file_name": file_name,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top3": top3,
        "review_flag": review_flag,
        "healthy_probability": healthy_probability,
        "abnormal_probability": abnormal_probability,
        "duration_sec": duration_sec,
        "original_sample_rate": original_sample_rate
    }

    try:
        response = client.responses.create(
            model=LLM_MODEL,
            reasoning={"effort": LLM_REASONING_EFFORT},
            instructions=(
                "You are a clinical documentation assistant for respiratory-audio screening. "
                "Use only the structured fields provided by the classification system. "
                "Do not claim a definitive diagnosis. "
                "Do not add symptoms, history, physical findings, treatment, or recommendations that were not provided. "
                "Write 3 to 5 concise sentences in English for clinicians. "
                "Mention the primary impression, confidence, differential considerations, and whether manual review is recommended."
            ),
            input=json.dumps(payload, ensure_ascii=False),
            max_output_tokens=LLM_MAX_OUTPUT_TOKENS
        )

        text = response.output_text.strip()
        if text:
            return text

        return fallback_text

    except Exception as e:
        log(f"[WARN] LLM summary generation failed: {e}")
        return fallback_text


def save_summary_txt(audio_id: int, file_name: str, summary_text: str) -> str:
    """
    將 summary_text 存成 txt。
    """
    base_name = os.path.splitext(file_name)[0]
    safe_name = base_name.replace(" ", "_")

    txt_path = os.path.join(
        SUMMARY_TXT_DIR,
        f"audio_{audio_id}_{safe_name}.txt"
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return txt_path


# =========================================================
# 8. Dataset 定義
# =========================================================

class RespiratoryAudioDataset(Dataset):
    """
    讀取單筆音檔 → 轉頻譜圖 → 回傳 label 與 metadata。
    """
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        row = self.records[idx]

        file_path = row["file_path"]
        true_class = row["disease_type"]

        audio = load_audio_fixed_length(file_path)
        spec_norm, _ = build_log_mel(audio)
        x = spectrogram_to_tensor(spec_norm)

        y = CLASS_TO_IDX[true_class]

        return (
            x,
            y,
            row["id"],
            row["file_name"],
            row["disease_type"],
            row["file_path"]
        )


# =========================================================
# 9. 模型建立
# =========================================================

def build_resnet18(num_classes: int) -> nn.Module:
    """
    建立 ResNet18。
    """
    if USE_PRETRAINED:
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        except Exception:
            log("[WARN] 預訓練權重不可用，改用隨機初始化。")
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)

    return model


# =========================================================
# 10. 訓練與驗證函式
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer):
    """
    單一 epoch 訓練。
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x, y, _, _, _, _ = batch
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion):
    """
    單一 epoch 驗證。
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in loader:
            x, y, _, _, _, _ = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            y_true_all.extend(y.cpu().numpy().tolist())
            y_pred_all.extend(preds.cpu().numpy().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, y_true_all, y_pred_all


# =========================================================
# 11. 資料切分與修復
# =========================================================

def filter_valid_records(records: List[Dict]) -> List[Dict]:
    """
    過濾並修復資料：
    - 類別不在五類中的資料略過
    - 路徑壞掉則嘗試修復
    - WAV 不可讀則略過
    """
    valid = []
    invalid_class_count = 0
    repaired_count = 0
    missing_count = 0
    unreadable_count = 0

    for row in records:
        row["file_name"] = str(row["file_name"]).strip()
        row["disease_type"] = str(row["disease_type"]).strip()
        row["file_path"] = str(row["file_path"]).strip()

        if row["disease_type"] not in CLASS_NAMES:
            invalid_class_count += 1
            continue

        repaired_path = repair_record_path(row, auto_update_db=AUTO_REPAIR_DB_PATH)

        if repaired_path is None:
            log(f"[SKIP] 找不到音檔：{row['file_path']}")
            missing_count += 1
            continue

        if not can_read_wav(repaired_path):
            log(f"[SKIP] WAV 無法讀取：{repaired_path}")
            unreadable_count += 1
            continue

        if repaired_path != row["file_path"]:
            repaired_count += 1
            row["file_path"] = repaired_path

        valid.append(row)

    log(f"[INFO] 無效類別筆數：{invalid_class_count}")
    log(f"[INFO] 修復路徑筆數：{repaired_count}")
    log(f"[INFO] 找不到實體檔案筆數：{missing_count}")
    log(f"[INFO] WAV 無法讀取筆數：{unreadable_count}")
    log(f"[INFO] 最終保留筆數：{len(valid)}")

    return valid


def split_records(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    train / val split。
    若某些類別太少，不適合 stratify，就退回一般切分。
    """
    if len(records) < 2:
        raise RuntimeError(f"可用資料只有 {len(records)} 筆，無法切分。")

    labels = [CLASS_TO_IDX[row["disease_type"]] for row in records]
    label_counts = Counter(labels)

    if min(label_counts.values()) < 2:
        log("[WARN] 某些類別樣本少於 2，改用非分層切分。")
        train_records, val_records = train_test_split(
            records,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True
        )
    else:
        train_records, val_records = train_test_split(
            records,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=labels
        )

    return train_records, val_records


# =========================================================
# 12. 訓練主流程
# =========================================================

def train_model() -> str:
    """
    完整訓練流程。
    回傳 checkpoint 路徑。
    """
    log("[INFO] 從資料庫讀取 audio_files ...")
    records = load_audio_records()
    log(f"[INFO] audio_files 原始筆數：{len(records)}")

    records = filter_valid_records(records)

    if len(records) == 0:
        raise RuntimeError("filter_valid_records 後沒有任何可用資料。")

    if len(records) < 2:
        raise RuntimeError(f"可用資料只有 {len(records)} 筆，無法切分。")

    log(f"[INFO] 有效資料共 {len(records)} 筆")

    train_records, val_records = split_records(records)

    train_dataset = RespiratoryAudioDataset(train_records)
    val_dataset = RespiratoryAudioDataset(val_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = build_resnet18(num_classes=len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    best_y_true = None
    best_y_pred = None
    no_improve_count = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, y_true, y_pred = validate_one_epoch(model, val_loader, criterion)

        epoch_metrics = compute_overall_metrics(y_true, y_pred)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_balanced_acc": epoch_metrics["balanced_accuracy"],
            "val_macro_precision": epoch_metrics["macro_precision"],
            "val_macro_recall": epoch_metrics["macro_recall"],
            "val_macro_f1": epoch_metrics["macro_f1"],
            "val_weighted_f1": epoch_metrics["weighted_f1"]
        })

        log(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"val_macro_f1={epoch_metrics['macro_f1']:.4f}, "
            f"val_bal_acc={epoch_metrics['balanced_accuracy']:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_y_true = copy.deepcopy(y_true)
            best_y_pred = copy.deepcopy(y_pred)
            no_improve_count = 0
            log(f"[INFO] 儲存最佳模型：epoch={epoch}, val_acc={val_acc:.4f}")
        else:
            no_improve_count += 1

        if no_improve_count >= EARLY_STOPPING_PATIENCE:
            log("[INFO] 觸發 Early Stopping")
            break

    if best_model_state is None:
        raise RuntimeError("沒有得到可用的最佳模型。")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_resnet18.pth")
    torch.save(best_model_state, checkpoint_path)

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(REPORT_DIR, "training_history.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    overall_metrics = save_validation_metric_reports(
        y_true=best_y_true,
        y_pred=best_y_pred,
        class_names=CLASS_NAMES,
        report_dir=REPORT_DIR
    )

    metadata = {
        "model_name": MODEL_NAME,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "best_val_macro_f1": float(overall_metrics["macro_f1"]),
        "best_val_balanced_accuracy": float(overall_metrics["balanced_accuracy"]),
        "num_train_records": len(train_records),
        "num_val_records": len(val_records),
        "target_sr": TARGET_SR,
        "target_duration_sec": TARGET_DURATION_SEC,
        "n_mels": N_MELS,
        "image_size": IMAGE_SIZE,
        "class_names": CLASS_NAMES,
        "dataset_root": DATASET_ROOT
    }

    with open(os.path.join(CHECKPOINT_DIR, "train_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log("=" * 60)
    log("[INFO] 訓練完成")
    log(f"[INFO] 最佳 epoch：{best_epoch}")
    log(f"[INFO] 最佳 val_acc：{best_val_acc:.4f}")
    log(f"[INFO] 最佳 val_macro_f1：{overall_metrics['macro_f1']:.4f}")
    log(f"[INFO] 最佳 val_balanced_accuracy：{overall_metrics['balanced_accuracy']:.4f}")
    log(f"[INFO] 模型輸出：{checkpoint_path}")
    log("=" * 60)

    return checkpoint_path


# =========================================================
# 13. 載入已訓練模型
# =========================================================

def load_trained_model() -> nn.Module:
    """
    載入已訓練的最佳 ResNet18 權重。
    若 checkpoint 不存在，且 AUTO_TRAIN_IF_CHECKPOINT_MISSING=True，
    就自動先訓練再載入。
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_resnet18.pth")

    if not os.path.exists(checkpoint_path):
        if AUTO_TRAIN_IF_CHECKPOINT_MISSING:
            log("[WARN] 找不到 checkpoint，開始自動訓練 ...")
            checkpoint_path = train_model()
        else:
            raise FileNotFoundError(f"Best model checkpoint not found: {checkpoint_path}")

    model = build_resnet18(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    return model


# =========================================================
# 14. 單筆推論（臨床 UI 欄位版）
# =========================================================

def infer_single_record(row: Dict, model: nn.Module) -> Dict:
    """
    對單筆音檔做完整推論，輸出臨床 UI 需要的欄位。
    """
    start_time = time.perf_counter()

    audio_id = row["id"]
    file_name = row["file_name"]
    true_class = row["disease_type"]
    file_path = row["file_path"]

    wav_meta = get_wav_metadata(file_path)

    audio = load_audio_fixed_length(file_path)
    spec_norm, log_mel_db = build_log_mel(audio)

    x = spectrogram_to_tensor(spec_norm).unsqueeze(0).to(DEVICE)
    spectrogram_path = save_spectrogram(log_mel_db, audio_id, file_name)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    predicted_class = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx])

    class_probabilities = {
        class_name: float(probs[idx])
        for idx, class_name in enumerate(CLASS_NAMES)
    }

    top_indices = np.argsort(probs)[::-1][:3]
    top3 = []
    for rank, idx in enumerate(top_indices, start=1):
        top3.append({
            "rank": rank,
            "class": IDX_TO_CLASS[int(idx)],
            "score": float(probs[int(idx)])
        })

    top3_json = json.dumps(top3, ensure_ascii=True)
    class_probabilities_json = json.dumps(class_probabilities, ensure_ascii=True)

    healthy_probability = class_probabilities["healthy"]
    abnormal_probability = 1.0 - healthy_probability

    abnormal_flag = "abnormal_suspected" if abnormal_probability >= 0.5 else "no_abnormal_pattern_detected"

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        review_flag = "manual_review_recommended"
    elif predicted_class in ["pneumonia", "copd"] and confidence >= 0.80:
        review_flag = "priority_review_recommended"
    else:
        review_flag = "ok"

    summary_text = generate_clinician_summary_with_llm(
        file_name=file_name,
        predicted_class=predicted_class,
        confidence=confidence,
        top3=top3,
        review_flag=review_flag,
        healthy_probability=healthy_probability,
        abnormal_probability=abnormal_probability,
        duration_sec=wav_meta["duration_sec"],
        original_sample_rate=wav_meta["original_sample_rate"]
    )

    summary_txt_path = save_summary_txt(
        audio_id=audio_id,
        file_name=file_name,
        summary_text=summary_text
    )

    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "audio_id": audio_id,
        "file_name": file_name,
        "true_class": true_class,
        "predicted_class": predicted_class,
        "primary_impression": predicted_class,
        "confidence": confidence,
        "top3_json": top3_json,
        "class_probabilities_json": class_probabilities_json,
        "healthy_probability": healthy_probability,
        "abnormal_probability": abnormal_probability,
        "abnormal_flag": abnormal_flag,
        "review_flag": review_flag,
        "spectrogram_path": spectrogram_path,
        "summary_text": summary_text,
        "summary_txt_path": summary_txt_path,
        "duration_sec": wav_meta["duration_sec"],
        "original_sample_rate": wav_meta["original_sample_rate"],
        "num_channels": wav_meta["num_channels"],
        "sample_width_bytes": wav_meta["sample_width_bytes"],
        "inference_time_ms": inference_time_ms,
        "model_name": MODEL_NAME,
        "remark": review_flag
    }


def infer_uploaded_wav(upload_wav_path: str) -> Dict:
    """
    給 Node-RED / 單筆 API 用：
    不依賴 audio_files，直接對一個上傳的 wav 做推論。
    """
    if not os.path.exists(upload_wav_path):
        raise FileNotFoundError(f"Uploaded wav not found: {upload_wav_path}")

    if not can_read_wav(upload_wav_path):
        raise ValueError(f"Uploaded wav is unreadable: {upload_wav_path}")

    model = load_trained_model()

    row = {
        "id": 0,
        "file_name": os.path.basename(upload_wav_path),
        "disease_type": "",
        "file_path": upload_wav_path
    }

    return infer_single_record(row, model)


# =========================================================
# 15. 批次推論主流程
# =========================================================

def run_batch_inference():
    """
    對 audio_files 全部音檔做批次推論。
    """
    log("[INFO] 載入已訓練模型 ...")
    model = load_trained_model()

    log("[INFO] 從資料庫讀取 audio_files ...")
    records = load_audio_records()
    log(f"[INFO] audio_files 原始筆數：{len(records)}")

    records = filter_valid_records(records)

    if len(records) == 0:
        raise RuntimeError("沒有任何可用資料可做推論。")

    results_for_export = []
    success_count = 0
    fail_count = 0

    for row in records:
        try:
            result = infer_single_record(row, model)

            upsert_prediction_result(
                audio_id=result["audio_id"],
                file_name=result["file_name"],
                true_class=result["true_class"],
                predicted_class=result["predicted_class"],
                confidence=result["confidence"],
                top3_json=result["top3_json"],
                spectrogram_path=result["spectrogram_path"],
                summary_text=result["summary_text"],
                model_name=result["model_name"],
                remark=result["remark"]
            )

            results_for_export.append(result)
            success_count += 1

            log(
                f"[DONE] audio_id={result['audio_id']} | "
                f"{result['predicted_class']} | "
                f"{result['confidence']:.4f}"
            )

        except Exception as e:
            fail_count += 1
            log(f"[FAIL] audio_id={row['id']} | {row['file_name']} | {e}")
            log(traceback.format_exc())

    df = pd.DataFrame(results_for_export)

    csv_path = os.path.join(EXPORT_DIR, "batch_inference_results.csv")
    json_path = os.path.join(EXPORT_DIR, "batch_inference_results.json")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_for_export, f, ensure_ascii=False, indent=2)

    log("=" * 60)
    log(f"[SUMMARY] Successful inferences: {success_count}")
    log(f"[SUMMARY] Failed inferences: {fail_count}")
    log(f"[SUMMARY] CSV: {csv_path}")
    log(f"[SUMMARY] JSON: {json_path}")
    log(f"[SUMMARY] Spectrogram folder: {SPECTROGRAM_DIR}")
    log(f"[SUMMARY] Summary txt folder: {SUMMARY_TXT_DIR}")
    log("=" * 60)


# =========================================================
# 16. Node-RED / CLI 單筆推論入口
# 說明：
# - 若命令列有帶 wav 路徑，就只做單筆推論
# - stdout 只輸出 JSON
# - stderr 才輸出 log
# =========================================================

def run_single_wav_cli() -> bool:
    """
    給 Node-RED / 命令列用：
    python xxx.py "C:\\path\\to\\file.wav"
    直接輸出單筆推論 JSON。
    """
    if len(sys.argv) < 2:
        return False

    wav_path = sys.argv[1]

    if str(wav_path).startswith("--"):
        return False

    try:
        result = infer_uploaded_wav(wav_path)
        output = {
            "ok": True,
            **result
        }
        print(json.dumps(output, ensure_ascii=False))
        return True
    except Exception as e:
        output = {
            "ok": False,
            "error": str(e)
        }
        print(json.dumps(output, ensure_ascii=False))
        return True


# =========================================================
# 17. 主程式入口
# =========================================================

def main():
    """
    主入口：
    - 若 RUN_TRAIN=True，先訓練模型
    - 若 RUN_BATCH_INFERENCE=True，再做批次推論
    """
    if RUN_TRAIN:
        train_model()

    if RUN_BATCH_INFERENCE:
        run_batch_inference()


if __name__ == "__main__":
    handled = run_single_wav_cli()
    if not handled:
        main()