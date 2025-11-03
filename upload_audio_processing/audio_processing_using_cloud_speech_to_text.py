# ENABLE CLOUD SPEECH TO TEXT IN GCP TO RUN THIS. BUT THE ACCURACY IS NOT GOOD
import os
import re
import random
import json
import time  # Added for waiting on async transcription
from pathlib import Path
from google import genai
from google.genai import types
from google.cloud import bigquery, storage  # storage added
from google.cloud import speech_v1p1beta1 as speech  # NEW IMPORT
from pydantic import BaseModel, Field
from vertexai import init
from vertexai.generative_models import GenerativeModel
import mimetypes
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURATION ---
BIGQUERY_PROJECT_ID = # your Google Cloud Project ID
BIGQUERY_DATASET = # your BigQuery Dataset name
BIGQUERY_TABLE = # your BigQuery Table name
GEMINI_MODEL = "gemini-2.5-pro"
GCS_BUCKET = os.environ.get("GCS_BUCKET", "your-gcs-bucket-name")  # NEW

# --- SPEECH-TO-TEXT CONFIG ---
# Adjust these based on your audio file properties (e.g., mono/stereo, sample rate)
AUDIO_SAMPLE_RATE_HERTZ = None  # Typical for phone calls
AUDIO_LANGUAGE_CODE = "en-US"
MAX_SPEAKER_COUNT = 2

# Initialize Clients
try:
    # Initialize Gemini Client (Requires GEMINI_API_KEY)
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    gemini_client = genai.Client(api_key=gemini_api_key)
    # Initialize BigQuery Client (Uses Application Default Credentials)
    bigquery_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)
    # Initialize Speech-to-Text Client
    speech_client = speech.SpeechClient()  # NEW CLIENT
    # Initialize GCS Client
    storage_client = storage.Client()
except Exception as e:
    print(f"Error initializing Google Clients: {e}")
    print("Ensure credentials (API Key for Gemini, ADC for GCP) are set.")
    exit()

# --- Pydantic Schema for Structured Output ---
class CallAnalysis(BaseModel):
    """Defines the structured output for the Gemini analysis."""
    phone_number: str = Field(description="The customer's mobile number found in the transcript.")
    problem_solved: str = Field(description="Whether the core issue was resolved during the call (e.g., 'Fully Solved', 'Partially Solved', 'Unsolved').")
    problem_type: str = Field(description="A concise summary of the core issue (e.g., 'Failed Recharge and Slow Internet').")
    sentiment: str = Field(description="A detailed, full analysis of the customer's emotional journey throughout the call (e.g., 'Started Frustrated, shifted to Relieved, ended Skeptical').")


# --- HELPER FUNCTIONS FOR UPLOAD ---
def upload_file_to_gcs(local_path: str, bucket_name: str, dest_blob_name: str) -> str:
    """
    Uploads a local file to Google Cloud Storage and returns its gs:// URI.
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(dest_blob_name)
        print(f"Uploading {local_path} to gs://{bucket_name}/{dest_blob_name} ...")
        blob.upload_from_filename(local_path)
        gs_uri = f"gs://{bucket_name}/{dest_blob_name}"
        print(f"Upload successful: {gs_uri}")
        return gs_uri
    except Exception as e:
        raise RuntimeError(f"GCS Upload failed: {e}")


def process_local_file_and_upload(local_path: str, bucket_name: str = GCS_BUCKET, dest_blob_name: str = None):
    """
    Uploads a local audio file to GCS and runs the complete call analysis pipeline.
    Returns structured analysis data (dict) and gs:// URI.
    """
    if dest_blob_name is None:
        dest_blob_name = f"upload_audio/{Path(local_path).stem}_{random.randint(1000,9999)}{Path(local_path).suffix}" #gcs bucket folder name

    gs_uri = upload_file_to_gcs(local_path, bucket_name, dest_blob_name)
    result = process_call_analysis(gs_uri)  # This now returns analysis_data dict
    return {"gs_uri": gs_uri, "result": result}



# --- CORE FUNCTIONS ---

def generate_customer_id():
    """Generates a random 4-5 digit number for the customer ID."""
    return random.randint(10000, 99999)


def get_audio_transcript(gcs_uri: str) -> str:
    print(f"Starting Speech-to-Text transcription for: {gcs_uri}")

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        alternative_language_codes=["en-IN", "hi-IN", "ta-IN", "mr-IN"],
        enable_speaker_diarization=True,
        diarization_speaker_count=MAX_SPEAKER_COUNT,
        model="latest_long",
    )

    operation = speech_client.long_running_recognize(config=config, audio=audio)
    print("Waiting for transcription operation to complete...")
    response = operation.result(timeout=10000)

    if not response.results:
        raise Exception("Transcription failed: No results returned.")

    words = response.results[-1].alternatives[0].words

    # Determine which speaker spoke first
    first_speaker_tag = words[0].speaker_tag
    if first_speaker_tag == 1:
        speaker_map = {1: "Support", 2: "Customer"}
    else:
        speaker_map = {1: "Customer", 2: "Support"}

    transcript_lines = []
    current_speaker = words[0].speaker_tag
    current_line = []

    for w in words:
        if w.speaker_tag != current_speaker:
            transcript_lines.append(f"{speaker_map[current_speaker]}: {' '.join(current_line)}")
            current_speaker = w.speaker_tag
            current_line = []
        current_line.append(w.word)

    # Add final line
    if current_line:
        transcript_lines.append(f"{speaker_map[current_speaker]}: {' '.join(current_line)}")

    reconstructed_transcript = "\n".join(transcript_lines)
    return reconstructed_transcript



# Initialize Vertex AI Gemini
init(project=BIGQUERY_PROJECT_ID, location="us-central1")  # or your region
gemini_model = GenerativeModel("gemini-2.5-pro")

def analyze_transcript_with_gemini(transcript: str) -> dict:
    """Uses Vertex AI Gemini (authenticated with ADC) to analyze the transcript and parse structured JSON."""
    prompt = f"""
    Analyze the following customer service call transcript from Airtel customer care. If any other company is mentioned then change it to Airtel as the audio calls are from airtel. Make the necessary corrections to the transcript first. Check the grammar, the sentences, if it makes sense or not.
    Speakers are labeled as Customer and Support. Figure out from the conversation which part is the customer and which part is from the support or airtel customer care person.
    TRANSCRIPT:
    ---
    {transcript}
    ---

    Extract the following:
    - Phone Number (IN THE FORM 9876543210)
    - Problem Solved (Solved OR Pending)
    - Problem Type (ANY ONE - Payment, Network, Recharge)
    - Sentiment (EMOTION THROUGHOUT THE AUDIO IN SUMMARISED FORM ONLY OF THE CUSTOMER)
    Return valid JSON only.
    """

    response = gemini_model.generate_content(prompt)
    raw = response.text.strip()

    # If Gemini wraps the JSON in ```json ... ```
    match = re.search(r"```json(.*?)```", raw, re.DOTALL)
    if match:
        raw = match.group(1).strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        # fallback if Gemini didn't produce clean JSON
        parsed = {"raw_text": raw}

    # Normalize keys to match your table fields
    return {
        "phone_number": parsed.get("Phone Number", ""),
        "problem_solved": str(parsed.get("Problem Solved", "")),
        "problem_type": parsed.get("Problem Type", ""),
        "sentiment": parsed.get("Sentiment", ""),  # store dict as string
    }


def insert_to_bigquery(data: dict, transcript: str, customer_id: int):
    """Inserts the analyzed data into the specified BigQuery table with explicit schema."""

    table_id = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"

    schema = [
        bigquery.SchemaField("customer_id", "INTEGER"),
        bigquery.SchemaField("phone_number", "STRING"),
        bigquery.SchemaField("full_transcript", "STRING"),
        bigquery.SchemaField("problem_solved", "STRING"),
        bigquery.SchemaField("problem_type", "STRING"),
        bigquery.SchemaField("sentiment", "STRING"),
    ]

    row_to_insert = [{
        "customer_id": str(customer_id),
        "phone_number": data.get("phone_number", ""),
        "full_transcript": transcript,
        "problem_solved": data.get("problem_solved", ""),
        "problem_type": data.get("problem_type", ""),
        "sentiment": data.get("sentiment", ""),
    }]

    print(f"Inserting data for Customer ID: {customer_id} into BigQuery...")

    job_config = bigquery.LoadJobConfig(schema=schema)
    job = bigquery_client.load_table_from_json(row_to_insert, table_id, job_config=job_config)
    job.result()  # Wait for job to finish

    print("✅ Data successfully loaded into BigQuery.")



def process_call_analysis(gcs_uri: str):
    """
    Orchestrates the entire process:
    - Speech-to-Text transcription
    - Gemini AI structured analysis
    - BigQuery storage
    Returns the structured analysis dict for further use (e.g., Flask API response)
    """

    # 1. Transcription
    try:
        transcript = get_audio_transcript(gcs_uri)
    except Exception as e:
        print(f"Transcription Error: {e}")
        return {"error": f"Transcription failed: {e}"}

    # 2. Gemini Analysis
    print("Sending transcript to Gemini for structured analysis...")
    try:
        analysis_data = analyze_transcript_with_gemini(transcript)
        print("Analysis successful.")
    except Exception as e:
        print(f"Error during Gemini analysis: {e}")
        return {"error": f"Gemini analysis failed: {e}"}

    # 3. BigQuery Insert
    customer_id = generate_customer_id()
    insert_to_bigquery(analysis_data, transcript, customer_id)

    summary = {
        "customer_id": customer_id,
        "transcript": transcript,
        "analysis_data": analysis_data,
    }

    print("\n--- Summary of Extracted Data ---")
    print(json.dumps(summary, indent=4))

    return summary


# --- MAIN EXECUTION (for direct testing) ---
if __name__ == "__main__":
    # Example local path test
    local_file = "sample_audio.wav"  # <-- Replace with your local file
    if os.path.exists(local_file):
        print("Processing local file and uploading to GCS...")
        output = process_local_file_and_upload(local_file)
        print(json.dumps(output, indent=4))
    else:
        print("No local file found. Using sample GCS URI...")
        GCS_AUDIO_PATH = "gs://your-bucket-name/path/to/audio.wav"
        process_call_analysis(GCS_AUDIO_PATH)
