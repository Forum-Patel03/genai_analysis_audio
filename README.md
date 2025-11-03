# 🎧 GenAI Audio Analysis (Google Cloud + Gemini + NLP + SQL)

This project performs intelligent **audio analysis** using **Google Cloud**, **BigQuery**, and **Vertex AI (Gemini)**. The analysis is done on audio files manually created using Gemini to be able to distinguish between Speaker 1 voice and Speaker 2 voice.

It has two major components:

1. **Batch Audio Processing** — analyzes pre-uploaded audios from Google Cloud Storage (GCS).  
2. **Upload Audio Processing** — allows users to upload audio files via a web UI for real-time analysis.

Both modules extract:
- 📜 Transcript  
- 📞 Phone Number  
- 💬 Problem Type & Resolution Status  
- 🙂 Sentiment Analysis

Both modules contain a file that has:

- 🧾 Auto-generated SQL Queries (via Natural Language Queries using NLP)

---

## 🧠 Project Overview

### 1️⃣ Batch Audio Processing
- Fetches audio files **already uploaded to GCS**.
- Processes them asynchronously through **Gemini API**.
- Extracts customer insights (transcript, problem type, sentiment, etc.).
- Generates random **Customer IDs** for processed records.
- Uses `nlp_sql.py` to convert **user’s natural language questions** into SQL queries (using rule-based NLP logic).

**📂 Folder Structure:**
```
batch_audio_procesing/
├── style2/
│ └── style2.css
├── templates2/
│ └── index2.html
├── app2.py                            # Flask frontend for NLP-SQL interface
├── batch_processing.py                # Core batch audio logic (GCS + Gemini + BigQuery)
├── nlp_sql.py                         # Rule-based NLP → SQL query generator
├── .env
├── .json

```

---

### 2️⃣ Upload Audio Processing
- Provides a web-based frontend to **upload audio files**.
- Sends them to Gemini API for transcription and sentiment analysis.
- Extracts same insights as batch mode.
- Stores data in **BigQuery** for further querying.
- Includes a separate interface for **NLP to SQL** queries.

**📂 Folder Structure:**
```
upload_audio_processing/
├── templates/
│ └── index.html                                                 # Upload audio frontend
├── templates2/
│ └── index2.html                                                # NLP-SQL frontend
├── style2/
│ └── style2.css
├── app.py                                                       # Flask upload + analysis app
├── app2.py                                                      # Flask NLP-SQL frontend
├── audio_processing.py                                          # Core Gemini-based analysis
├── audio_processing_using_cloud_speech_to_text.py               # Optional GCP STT alternative
├── nlp_sql.py                                                   # Shared rule-based NLP to SQL module
├── .env
├── .json

```

---

## ⚙️ Environment Setup

Before running the app, create a `.env` file in the root directory of the project. Make sure that batch processing and audio processing have different dataset, table and 2 different sub folders in one GCS Bucket.

### Example `.env` File

```bash
GCS_BUCKET=your-google-cloud-bucket-name
BIGQUERY_PROJECT_ID=your-project-id
BIGQUERY_DATASET=your-dataset-name
BIGQUERY_TABLE=your-table-name
GEMINI_API_KEY=your-vertex-ai-api-key
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
```

### ⚙️ Environment Setup

Check requirements.txt for the required pip files.

### 🌟 About
GenAI Audio Analysis Project — powered by Google Cloud, Gemini, and Flask.
Designed for scalable audio intelligence and data analytics.
