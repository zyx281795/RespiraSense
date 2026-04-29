# RespiraSense — 智慧呼吸音臨床輔助判讀系統

> 醫用行動軟體設計 期中報告  

---

## 目錄

1. [開發動機](#開發動機)
2. [系統簡介](#系統簡介)
3. [系統架構](#系統架構)
4. [系統流程](#系統流程)
5. [Node-RED 架構](#node-red-架構)
6. [模型效能](#模型效能)
7. [專案結構](#專案結構)
8. [安裝與設定](#安裝與設定)
9. [使用方式](#使用方式)

---

## 開發動機

近年來醫療資訊系統逐漸朝向智慧化與即時化發展，但在基層門診、遠距照護或臨床初步篩檢情境中，**呼吸音判讀仍高度依賴醫師或專業人員經驗**。

傳統呼吸音判讀容易受到主觀經驗、環境雜音與聽診品質影響，導致不同人員之間對於異常呼吸音的辨識可能存在差異。因此，若能建立一套可輸入肺部聲音檔案、快速完成模型推論，並同步產生頻譜圖與文字化摘要的系統，將有助於提升初步篩檢效率與判讀輔助能力。

本系統希望整合**人工智慧模型**、**Node-RED 流程自動化**、**MySQL 資料庫**與 **Google Gemini API**，建構一套智慧呼吸音輔助判讀系統。系統不僅能對呼吸音檔進行分類，也能輸出可信度、Top-3 預測結果、梅爾頻譜圖（Mel Spectrogram）與文字摘要，整合進完整的資訊系統流程中。

### DATASET：https://www.kaggle.com/datasets/mohammedtawfikmusaed/asthma-detection-dataset-version-2
---

## 系統簡介

本系統是一套**以呼吸音 .wav 檔案為輸入**的智慧輔助判讀平台，主要目標是提供使用者一個簡潔且具資訊整合能力的操作介面。

- 使用者可透過 HTML 前端上傳單筆呼吸音檔案，系統接收後會先由 **Node-RED** 進行流程控管與資料交換，再呼叫 **Python** 後端推論程式進行音訊前處理與分類推論。
- 模型以 **ResNet18** 為核心分類器，並將原始呼吸音轉換為 **Mel Spectrogram** 作為模型輸入。
- 系統可辨識五種呼吸音類別：`asthma`、`Bronchial`、`copd`、`healthy`、`pneumonia`。
- 推論完成後，系統會輸出**主要預測類別、可信度、Top-3 類別排序、異常機率、人工覆核建議與頻譜圖路徑**，並透過 **Google Gemini API** 將模型結果轉換為較符合臨床閱讀習慣的英文摘要。
- 所有推論結果會寫入 **MySQL** 資料庫，以支援歷史紀錄查詢、單筆結果回顧與後續資料管理。

---

## 系統架構

| 系統層級 | 使用技術 | 負責工作 |
|:--------:|:--------:|:-------:|
| 使用者介面層 | HTML、CSS、JavaScript | 提供音檔上傳、分析請求送出、結果顯示與歷史紀錄查詢介面 |
| 流程控制層 | Node-RED | 負責接收前端請求、暫存音檔、呼叫 Python 推論程式、串接 MySQL 與 Gemini API，並回傳結果 |
| 模型推論層 | Python、ResNet18 | 負責呼吸音分類，輸出預測類別、可信度、Top-3 結果與人工覆核建議 |
| 特徵轉換層 | Mel Spectrogram | 將原始 .wav 呼吸音檔轉換為頻譜圖，作為深度學習模型輸入特徵 |
| 資料儲存層 | MySQL | 儲存音檔資訊、推論結果、可信度、Top-3 預測、頻譜圖路徑、摘要與推論時間 |
| 文本生成層 | Google Gemini API | 根據模型輸出的結構化結果產生英文摘要，提升結果可讀性 |

---

## 系統流程

```
使用者上傳 .wav 呼吸音檔
        │
        ▼
HTML 前端接收檔案並送出分析請求
        │
        ▼
Node-RED API 接收請求
        │
        ▼
Node-RED 暫存音檔至伺服器資料夾
        │
        ▼
Node-RED 呼叫 Python 推論程式
        │
        ▼
Python 進行音訊前處理（讀檔、重採樣、固定長度）
        │
        ▼
產生 Mel Spectrogram
        │
        ▼
ResNet18 模型進行五分類推論
        │
        ▼
輸出 predicted_class、confidence、top3、review_flag
        │
        ├──────────────────────────────────────────►
        │                                           │
        ▼                                           ▼
Node-RED 將模型結果整理成 Prompt        呼叫 Google Gemini API 產生臨床風格摘要
                                                    │
                                                    ▼
                              推論結果與文字摘要寫入 MySQL prediction_results
                                                    │
                                                    ▼
                                       Node-RED 回傳結果給前端
                                                    │
                                                    ▼
                              前端顯示預測類別、可信度、Top-3、頻譜圖與摘要
```

---

## Node-RED 架構

Node-RED flow 包含以下主要路由：

| 端點 | 功能 |
|------|------|
| `POST /api/infer` | 接收前端上傳的 base64 WAV，呼叫 Python 推論 |
| `GET /api/history` | 查詢所有歷史推論紀錄 |
| `GET /api/result/:id` | 查詢單筆推論結果 |
| `DELETE /api/result/:id` | 刪除單筆推論紀錄 |
| `GET /api/spectrogram/:filename` | 回傳頻譜圖 PNG |
| `GET /api/audio-files` | 查詢音檔資料表 |
| `GET /api/db/stats` | 查詢資料庫統計數據 |
| `GET /demo` | 臨床大型展示頁 |

Flow 檔案：[`respirasense_v17_fix_choosewav_metric_text_full.json`](./respirasense_v17_fix_choosewav_metric_text_full.json)

---

## 模型效能

模型：**ResNet18 + Log-Mel Spectrogram**  
訓練集：971 筆 ｜ 驗證集：243 筆 ｜ 最佳 epoch：16

### 整體驗證指標

| 指標 | 數值 |
|------|------|
| Accuracy | **95.47%** |
| Balanced Accuracy | **95.29%** |
| Macro Precision | 95.04% |
| Macro Recall | 95.29% |
| Macro F1 | **95.05%** |
| Weighted F1 | 95.49% |

### 各類別效能

| 類別 | Precision | Recall | F1 | Support |
|------|-----------|--------|----|---------|
| asthma | 98.11% | 89.66% | 93.69% | 58 |
| Bronchial | 100.00% | 95.24% | 97.56% | 21 |
| copd | 98.75% | 98.75% | 98.75% | 80 |
| healthy | 86.67% | 96.30% | 91.23% | 27 |
| pneumonia | 91.67% | 96.49% | 94.02% | 57 |

---

## 專案結構

```
RespiraSense/
├── ResNet18+Log-MelSpectrogram.py          # 主要 ML 訓練與推論腳本
├── respirasense_v17_fix_choosewav_metric_text_full.json  # Node-RED flow
├── .gitignore
├── README.md
├── ResNet18_results/
│   ├── checkpoints/
│   │   ├── best_resnet18.pth               # 訓練完成的模型權重 (42.7 MB)
│   │   └── train_metadata.json             # 訓練元資料
│   ├── reports/
│   │   ├── classification_report.txt       # 分類報告
│   │   ├── confusion_matrix.csv            # 混淆矩陣
│   │   ├── overall_validation_metrics.json # 整體驗證指標
│   │   ├── per_class_clinical_metrics.csv  # 各類別臨床指標
│   │   └── training_history.csv            # 訓練歷程
│   ├── prediction_exports/
│   │   ├── batch_inference_results.csv     # 批次推論結果
│   │   └── batch_inference_results.json
│   ├── spectrograms/                       # 生成的頻譜圖 (.gitignore)
│   └── summary_txt/                        # 生成的文字摘要 (.gitignore)
├── uploads/                                # Node-RED 暫存上傳音檔 (.gitignore)
└── Asthma Detection Dataset Version 2/     # 訓練資料集 (.gitignore)
```

---

## 安裝與設定

### 1. 環境需求

- Python 3.8+（建議使用 Conda 虛擬環境）
- Node.js + Node-RED
- MySQL 8.0+

### 2. Python 套件安裝

```bash
pip install torch torchvision scipy numpy pandas pymysql matplotlib scikit-learn
```

### 3. MySQL 資料庫設定

建立資料庫 `respiratory_audio_db`，並建立以下兩張資料表：

```sql
CREATE DATABASE respiratory_audio_db DEFAULT CHARACTER SET utf8mb4;

USE respiratory_audio_db;

CREATE TABLE audio_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    disease_type VARCHAR(50),
    file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE prediction_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    audio_id INT,
    file_name VARCHAR(255),
    true_class VARCHAR(50),
    predicted_class VARCHAR(50),
    confidence FLOAT,
    top3_json TEXT,
    spectrogram_path TEXT,
    summary_text TEXT,
    model_name VARCHAR(100),
    remark VARCHAR(255),
    inference_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4. 設定環境變數

Node-RED flow 使用以下環境變數，請在啟動 Node-RED 前設定：

```bash
# 專案根目錄（絕對路徑）
set RESPIRASENSE_ROOT=C:\your\path\to\RespiraSense

# Python 執行檔路徑（對應到你的虛擬環境）
set PYTHON_EXE=C:\Users\YourName\anaconda3\envs\yourenv\python.exe
```

或在 `~/.node-red/settings.js` 中加入：

```js
process.env.RESPIRASENSE_ROOT = 'C:\\your\\path\\to\\RespiraSense';
process.env.PYTHON_EXE = 'C:\\path\\to\\python.exe';
```

### 5. 修改 Python 腳本的資料庫連線設定

開啟 `ResNet18+Log-MelSpectrogram.py`，確認以下設定符合你的環境：

```python
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "database": "respiratory_audio_db",
    "charset": "utf8mb4"
}
```

密碼在執行時由終端互動輸入，不會寫入程式碼中。

### 6. 匯入 Node-RED Flow

1. 開啟 Node-RED（通常為 `http://localhost:1880`）
2. 右上角選單 → Import → 選擇 `respirasense_v17_fix_choosewav_metric_text_full.json`
3. 確認 MySQL 節點設定（host / user / password / database）
4. Deploy

### 7. 資料集

本專案使用 **Asthma Detection Dataset Version 2**，包含五類呼吸音：
`asthma`、`Bronchial`、`copd`、`healthy`、`pneumonia`。

請將資料集解壓縮至以下位置：

```
RespiraSense/
└── Asthma Detection Dataset Version 2/
    └── Asthma Detection Dataset Version 2/
        ├── asthma/
        ├── Bronchial/
        ├── copd/
        ├── healthy/
        └── pneumonia/
```

---

## 使用方式

### 模型訓練

```bash
# 執行訓練 + 批次推論
python "ResNet18+Log-MelSpectrogram.py"
```

在 Python 腳本頂部設定控制旗標：

```python
RUN_TRAIN = True            # True = 執行訓練
RUN_BATCH_INFERENCE = True  # True = 執行批次推論
```

### Node-RED 單筆推論（CLI）

```bash
python "ResNet18+Log-MelSpectrogram.py" "path\to\audio.wav"
```

輸出為 JSON 格式，包含：

```json
{
  "ok": true,
  "predicted_class": "Bronchial",
  "confidence": 1.0,
  "top3_json": "[...]",
  "abnormal_probability": 1.0,
  "review_flag": "ok",
  "spectrogram_path": "...",
  "summary_text": "...",
  "inference_time_ms": 2264.33
}
```
---

## 授權

本專案為學術用途（醫用行動軟體設計課程期中報告），僅供學習與展示使用。
