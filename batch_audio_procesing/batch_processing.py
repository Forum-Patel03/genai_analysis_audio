import os
import re
import json
import asyncio
import time
import random
import mimetypes
from typing import Dict, Any, List, Optional
from asyncio import Semaphore

from google.cloud import bigquery, storage
from vertexai import init
from vertexai.generative_models import GenerativeModel, Part
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    RetryError,
)

load_dotenv()

BIGQUERY_PROJECT_ID = # your Google Cloud Project ID
BIGQUERY_DATASET = # your BigQuery Dataset name
BIGQUERY_TABLE = # your BigQuery Table name
GEMINI_MODEL = "gemini-2.5-flash"
GCS_BUCKET = os.getenv("GCS_BUCKET", "your-gcs-bucket-name")
MAX_CONCURRENT_TASKS = 10

try:
    bigquery_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
    storage_client = storage.Client()
    init(project=BIGQUERY_PROJECT_ID, location="us-central1")
    gemini_model = GenerativeModel(GEMINI_MODEL)
    print("✅ Clients initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing clients: {e}")
    raise SystemExit(1)

def generate_customer_id() -> int:
    return random.randint(10000, 99999)

def clean_phone_number(phone: str) -> str:
    if not phone:
        return "Missing phone number"

    phone = phone.strip().lower()

    if phone in ["missing phone number", "no number", "none", "null"]:
        return "Missing phone number"

    digits = re.sub(r"\D", "", phone)

    if len(digits) == 10:
        return digits

    if 7 <= len(digits) < 10:
        return "Incomplete phone number"

    return "Missing phone number"

def safe_json_parse(text: str) -> dict:
    text = text.strip()
    match = re.search(r"json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    text = text.replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"⚠️ Failed to parse JSON, returning raw text. Content: {text[:200]}...")
        return {"raw_text": text}

async def insert_batch_to_bigquery(rows: List[Dict[str, Any]]):
    if not rows:
        print("ℹ️ No rows to insert.")
        return
    table_id = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"
    schema = [
        bigquery.SchemaField("customer_id", "INTEGER"),
        bigquery.SchemaField("phone_number", "STRING"),
        bigquery.SchemaField("full_transcript", "STRING"),
        bigquery.SchemaField("problem_solved", "STRING"),
        bigquery.SchemaField("problem_type", "STRING"),
        bigquery.SchemaField("sentiment", "STRING"),
    ]
    job_config = bigquery.LoadJobConfig(schema=schema)
    try:
        print(f"📦 Inserting {len(rows)} rows into BigQuery...")
        job = bigquery_client.load_table_from_json(rows, table_id, job_config=job_config)
        await asyncio.to_thread(job.result)
        if job.errors:
            print(f"❌ BigQuery job finished with errors: {job.errors}")
        else:
            print(f"✅ Successfully inserted {len(rows)} rows.")
    except Exception as e:
        print(f"❌ Failed to insert batch into BigQuery: {e}")

RETRYABLE_EXCEPTIONS = (
    RetryError,
    google_exceptions.ResourceExhausted,
    google_exceptions.ServiceUnavailable,
    google_exceptions.InternalServerError,
)

@retry(
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
)
async def call_gemini_async(audio_part: Part, prompt: str) -> str:
    response = await gemini_model.generate_content_async(
        [audio_part, prompt],
        generation_config={
            "temperature": 1,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        },
    )
    return response.text.strip()

async def process_audio_file(gcs_uri: str) -> Optional[Dict[str, Any]]:
    mime_type, _ = mimetypes.guess_type(gcs_uri)
    mime_type = mime_type or "audio/wav"
    print(f"\n🎧 Processing {gcs_uri} ...")
    start_time = time.time()
    unified_prompt = """
    You are an expert call analyst. Listen to the call very very carefully and understand each and every words and numbers of the audio.
    1️⃣ Transcribe this customer care call for Airtel.
    2️⃣ Label speakers (Customer, Support).
    3️⃣ Correct grammar errors.
    4️⃣ Extract strictly in JSON:
    {
        "phone_number": 
    "Extract the phone number if mentioned in the call. 
     • If the number is exactly 10 digits → output only the 10 digits. 
     • If the number contains 7–9 digits → output those digits only (do NOT output null). 
     • If no number is spoken at all → output: \"Missing phone number\"",
        "problem_solved": "Solved/Pending",
        "problem_type": "Payment/Network/Recharge",
        "sentiment": - "Provide a short summary of the customer's emotional tone throughout the entire call, indicating how it started, how it progressed, and how it ended in maximum 20 words.",
        "full_transcript": "entire conversation text"
    }
    """
    try:
        audio_part = Part.from_uri(gcs_uri, mime_type=mime_type)
        text = await call_gemini_async(audio_part, unified_prompt)
        parsed = safe_json_parse(text)
        if "raw_text" in parsed:
            print(f"❌ Failed to parse JSON for {gcs_uri}. Skipping.")
            return None
        customer_id = generate_customer_id()
        def get_string_value(data: dict, key: str) -> str:
            value = data.get(key)
            if value is None:
                return ""
            if isinstance(value, (dict, list)):
                return json.dumps(value)
            return str(value)
        row_data = {
            "customer_id": customer_id,
            "phone_number": clean_phone_number(get_string_value(parsed, "phone_number")),
            "full_transcript": get_string_value(parsed, "full_transcript"),
            "problem_solved": get_string_value(parsed, "problem_solved"),
            "problem_type": get_string_value(parsed, "problem_type"),
            "sentiment": get_string_value(parsed, "sentiment")
        }
        total_time = round(time.time() - start_time, 2)
        print(f"✅ Completed {gcs_uri} in {total_time}s")
        return row_data
    except Exception as e:
        print(f"❌ Error processing {gcs_uri}: {e}")
        return None

def list_audio_files_from_gcs(bucket_name: str, prefix: str = "batch_audio/") -> List[str]: # here batch_audio is the sub folder in GCS Bucket containing the audio files already uploaded
    try:
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        audio_files = [
            f"gs://{bucket_name}/{blob.name}"
            for blob in blobs
            if blob.name.lower().endswith((".wav", ".mp3")) and blob.size > 0
        ]
        print(f"🎵 Found {len(audio_files)} audio files.")
        return audio_files
    except Exception as e:
        print(f"❌ Error listing GCS files: {e}")
        return []

async def process_with_limit(semaphore: Semaphore, uri: str):
    async with semaphore:
        return await process_audio_file(uri)

async def main():
    start_total = time.time()
    all_files = list_audio_files_from_gcs(GCS_BUCKET)
    if not all_files:
        print("❌ No audio files found in GCS.")
        return
    print(f"\n🚀 Starting async processing for {len(all_files)} files...")
    semaphore = Semaphore(MAX_CONCURRENT_TASKS)
    tasks = [asyncio.create_task(process_with_limit(semaphore, uri)) for uri in all_files]
    results = await asyncio.gather(*tasks)
    print("\n✅ ALL FILES PROCESSED.")
    successful_rows = [row for row in results if row is not None]
    failed_indices = [i for i, row in enumerate(results) if row is None]
    failed_uris = [all_files[i] for i in failed_indices]
    print(f"\n📊 Processing summary:")
    print(f"  Total files: {len(all_files)}")
    print(f"  Successful:  {len(successful_rows)}")
    print(f"  Failed:      {len(failed_uris)}")
    await insert_batch_to_bigquery(successful_rows)
    retry_success_rows: List[Dict[str, Any]] = []
    if failed_uris:
        print("\n🔁 Retrying failed files one by one...")
        for uri in failed_uris:
            attempt = 0
            max_attempts = 5
            success_row = None
            while attempt < max_attempts and success_row is None:
                attempt += 1
                print(f"🔁 Attempt {attempt} for {uri}")
                try:
                    result = await process_audio_file(uri)
                    if result is not None:
                        success_row = result
                        retry_success_rows.append(success_row)
                        print(f"✅ Retry succeeded for {uri} on attempt {attempt}")
                    else:
                        print(f"❌ Retry returned no data for {uri} on attempt {attempt}")
                except Exception as e:
                    print(f"❌ Exception retrying {uri} on attempt {attempt}: {e}")
                if success_row is None and attempt < max_attempts:
                    await asyncio.sleep(2 * attempt)
    if retry_success_rows:
        print(f"\n📦 Inserting {len(retry_success_rows)} retry-success rows into BigQuery...")
        await insert_batch_to_bigquery(retry_success_rows)
    final_failed_count = len(failed_uris) - len(retry_success_rows)
    print(f"\n📉 Retry summary:")
    print(f"  Retry successes: {len(retry_success_rows)}")
    print(f"  Final failed:    {final_failed_count}")
    total_time = round((time.time() - start_total) / 60, 2)
    print(f"\n⏰ Total time taken: {total_time} minutes")

if __name__ == "__main__":
    asyncio.run(main())
