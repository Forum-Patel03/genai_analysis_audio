import os
import re
import random
import json
import mimetypes
import time
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from google.cloud import bigquery, storage
from vertexai import init
from vertexai.generative_models import GenerativeModel, Part
from pydantic import BaseModel, Field

load_dotenv()

# --- CONFIGURATION ---
BIGQUERY_PROJECT_ID = # your Google Cloud Project ID
BIGQUERY_DATASET = # your BigQuery Dataset name
BIGQUERY_TABLE = # your BigQuery Table name
GEMINI_MODEL = "gemini-2.5-flash"
GCS_BUCKET = os.environ.get("GCS_BUCKET", "your-gcs-bucket-name")

# --- CLIENT INITIALIZATION ---
try:
    bigquery_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
    storage_client = storage.Client()
    init(project=BIGQUERY_PROJECT_ID, location="us-central1")
    gemini_model = GenerativeModel(GEMINI_MODEL)
    print("✅ Clients initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing clients: {e}")
    raise SystemExit(1)


# --- STRUCTURE FOR OUTPUT ---
class CallAnalysis(BaseModel):
    phone_number: str = Field(description="10-digit number, or 'Incomplete phone number' / 'Missing phone number'")
    problem_solved: str = Field(description="Solved or Pending")
    problem_type: str = Field(description="Payment, Network, Recharge")
    sentiment: str = Field(description="Summary of customer’s emotional tone")
    full_transcript: str = Field(description="Full corrected transcript text")


# --- HELPERS ---
def generate_customer_id() -> int:
    return random.randint(10000, 99999)


def clean_phone_number(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) == 10:
        return digits
    elif 7 <= len(digits) < 10:
        return "Incomplete phone number"
    return "Missing phone number"


def upload_file_to_gcs(local_path: str, bucket_name: str, dest_blob_name: str) -> (str, str):
    mime_type, _ = mimetypes.guess_type(local_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_blob_name)
    print(f"⬆️ Uploading {local_path} → gs://{bucket_name}/{dest_blob_name}")
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{dest_blob_name}", mime_type


# --- MAIN PROCESSING FUNCTION ---
def transcribe_and_analyze_audio(gcs_uri: str, mime_type: str = None) -> Dict[str, Any]:
    """
    Single Gemini prompt that does BOTH:
    - Transcription
    - Analysis (phone number, problem solved, type, sentiment)
    """
    mime_type = mime_type or mimetypes.guess_type(gcs_uri)[0] or "audio/wav"
    print(f"🎧 Starting combined transcription + analysis for {gcs_uri} ({mime_type})")

    audio_part = Part.from_uri(gcs_uri, mime_type=mime_type)

    unified_prompt = """
    You are an expert Airtel call analyst.
    Carefully listen to this entire audio recording and produce a structured JSON response.

    Your tasks:
    1️⃣ Transcribe the full call clearly, correcting grammar and labeling speakers as "Customer" and "Support".
    2️⃣ Extract:
        - "phone_number": number heard from call (any length or missing).
        - "problem_solved": "Solved" or "Pending"
        - "problem_type": Choose from ["Payment", "Network", "Recharge"]
        - "sentiment": Describe the customer's emotion in 20 words max
    Return **valid JSON only**, formatted as:
    {
        "phone_number": "",
        "problem_solved": "",
        "problem_type": "",
        "sentiment": "",
        "full_transcript": ""
    }
    """

    try:
        response = gemini_model.generate_content([audio_part, unified_prompt])
        raw = response.text.strip()

        match = re.search(r"```json(.*?)```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            print(f"⚠️ Failed to parse JSON from Gemini output. Raw content:\n{raw}")
            return {"error": "Invalid JSON", "raw_output": raw}

        # Validate and clean phone number
        parsed["phone_number"] = clean_phone_number(parsed.get("phone_number", ""))
        return parsed

    except Exception as e:
        print(f"❌ Error during Gemini processing: {e}")
        return {"error": str(e)}


def insert_to_bigquery(data: dict, customer_id: int):
    """Inserts combined results into BigQuery."""
    table_id = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
    schema = [
        bigquery.SchemaField("customer_id", "INTEGER"),
        bigquery.SchemaField("phone_number", "STRING"),
        bigquery.SchemaField("full_transcript", "STRING"),
        bigquery.SchemaField("problem_solved", "STRING"),
        bigquery.SchemaField("problem_type", "STRING"),
        bigquery.SchemaField("sentiment", "STRING"),
    ]

    row = [{
        "customer_id": customer_id,
        "phone_number": data.get("phone_number", ""),
        "full_transcript": data.get("full_transcript", ""),
        "problem_solved": data.get("problem_solved", ""),
        "problem_type": data.get("problem_type", ""),
        "sentiment": data.get("sentiment", ""),
    }]

    print(f"📦 Uploading results for Customer ID {customer_id} to BigQuery...")
    job_config = bigquery.LoadJobConfig(schema=schema)
    job = bigquery_client.load_table_from_json(row, table_id, job_config=job_config)
    job.result()
    print("✅ Data inserted successfully.")


# --- MAIN EXECUTION ---
def process_local_file_and_upload(local_path: str, bucket_name: str = GCS_BUCKET, dest_blob_name: str = None):
    """
    Uploads a local audio file to GCS and runs the complete call analysis pipeline.
    Returns structured analysis data (dict) and gs:// URI.
    """
    if dest_blob_name is None:
        dest_blob_name = f"upload_audio/{Path(local_path).stem}_{random.randint(1000,9999)}{Path(local_path).suffix}" # sub folder in GCS Bucket

    gs_uri, mime_type = upload_file_to_gcs(local_path, bucket_name, dest_blob_name)
    
    # 🔥 FIXED: call unified transcribe+analyze
    result = transcribe_and_analyze_audio(gs_uri, mime_type)

    # Optional: insert to BigQuery directly if you want
    customer_id = generate_customer_id()
    insert_to_bigquery(result, customer_id)

    return {"gs_uri": gs_uri, "customer_id": customer_id, "result": result}

if __name__ == "__main__":
    local_file = "sample_audio.wav"
    if os.path.exists(local_file):
        print("🚀 Starting local audio processing pipeline...")
        process_local_file_and_upload(local_file)
    else:
        print("⚠️ sample_audio.wav not found. Please place an audio file in this directory.")
