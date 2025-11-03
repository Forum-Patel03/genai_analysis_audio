import os
import json
from google.cloud import bigquery
from vertexai import init
from vertexai.generative_models import GenerativeModel
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURATION ---
BIGQUERY_PROJECT_ID = # your Google Cloud Project ID
BIGQUERY_DATASET = # your BigQuery Dataset name
BIGQUERY_TABLE = # your BigQuery Table name
GEMINI_MODEL = "gemini-2.5-flash"  # Fast & cost-efficient

# --- INITIALIZE VERTEX AI CLIENTS ---
try:
    # Authenticate with Application Default Credentials (ADC)
    # Ensure you've run: gcloud auth application-default login
    init(project=BIGQUERY_PROJECT_ID, location="us-central1")

    gemini_model = GenerativeModel(GEMINI_MODEL)
    bigquery_client = bigquery.Client(project=BIGQUERY_PROJECT_ID)

    print("✅ Vertex AI Gemini and BigQuery initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing clients: {e}")
    print("Make sure you have run 'gcloud auth application-default login' and have the right project access.")
    exit()

# --- HELPER: FETCH SCHEMA ---
def get_table_schema(project_id: str, dataset_id: str, table_id: str) -> str:
    """Retrieves and formats the BigQuery table schema for Gemini prompt."""
    try:
        table_ref = bigquery_client.dataset(dataset_id).table(table_id)
        table = bigquery_client.get_table(table_ref)

        schema_info = [f"{field.name} ({field.field_type})" for field in table.schema]
        return f"Table Name: {table_id}\nSchema: {', '.join(schema_info)}"
    except Exception as e:
        print(f"Error fetching BigQuery schema: {e}")
        return "Error: Could not retrieve schema."

# --- 1️⃣ NL → SQL ---
def nl_to_sql(question: str, schema_info: str) -> str:
    """
    Converts a user's natural language question into a BigQuery SQL query.
    """
    prompt = f"""
    You are an expert BigQuery SQL translator.
    Convert the user's natural language question into a valid BigQuery Standard SQL query.

    Table & Schema:
    {schema_info}

    User Question: "{question}"

    Rules:
    - Return only the SQL query (no explanations or markdown).
    - Use table `{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}`.
    
    - The database contains only the following exact values (case-insensitive):
        • problem_solved: Pending, Solved
        • problem_type: Network, Recharge, Payment

    - When filtering problem_type or problem_solved, match exact category names using LOWER() for case-insensitive comparison.
      Example: LOWER(problem_type) = 'network'
               LOWER(problem_solved) = 'pending'

    - Do NOT use LIKE, partial matching, or any other words for problem_type or problem_solved.
      Only use the exact valid values listed above.

    - Do NOT generate conditions like LIKE '%network issue%' when user means Network.
      Instead, map user language to the closest existing problem_type category:
          • "network issue", "network problem", "network related to recharge" → Network
          • "recharge issue", "recharge problem" → Recharge
          • "payment issue", "payment failed", "payment problem" → Payment

    - If user includes "network issue", "network problem", or "network related to recharge":
        • Apply `LOWER(problem_type) = 'network'`
        • If user also mentions recharge, add transcript keyword search:
            AND LOWER(transcript) LIKE '%recharge%'

    - Map user phrases to problem_solved:
          • "unsolved", "not solved", "unresolved", "pending issue" → `LOWER(problem_solved) = 'pending'`
          • "solved", "resolved", "fixed", "completed" → `LOWER(problem_solved) = 'solved'`

    - When user mentions pending/solved along with network/recharge/payment, apply both filters.
      Example: "pending network issue" →
          LOWER(problem_type) = 'network'
          AND LOWER(problem_solved) = 'pending'

    - If asked for "most common" or "top", include LIMIT.

     - When the user asks for records with missing data (e.g., null fields, incomplete records, missing values), use AND between conditions, not OR.
      Example: phone_number IS NULL AND full_transcript IS NULL

        - Sentiment is stored as a summarized paragraph, not as a single word or fixed label. 
      Do NOT filter sentiment using direct string matching such as LOWER(sentiment) = 'good' or LIKE '%good%'.
      Instead, analyze the full sentiment text to determine whether it expresses positive, negative, or neutral sentiment.
      Then apply the filter based on the interpreted sentiment category.
      Example: If user asks for "good" or "positive" sentiment, return rows where the sentiment text indicates the customer ended satisfied, happy, relieved, appreciative, or with a positive resolution.

    - The column phone_number can contain 3 possible types of values:
      • A 10-digit phone number (e.g., "9876543210")
      • "Incomplete phone number" (phone provided but <10 digits)
      • "Missing phone number" (no phone number provided)

    - When user asks for:
        • "missing phone number" → filter using: phone_number = 'Missing phone number'
        • "incomplete phone number" → filter using: phone_number = 'Incomplete phone number'
        • "customer with no phone number", "no phone", "without phone" → treat as 'Missing phone number'

    - Do NOT use phone_number IS NULL unless data actually has SQL NULLs.

    - When checking phone_number equality, use LOWER() for case-insensitive matching.

    - If user says "invalid phone number", treat it same as "Incomplete phone number".
    - If user asks for "phone number present", return only rows where LENGTH(phone_number) = 10.
    - If user asks for "all phone numbers", return customer_id and phone_number only.



    """

    print("-> Converting NL to SQL using Gemini...")

    response = gemini_model.generate_content(prompt)
    sql_query = response.text.strip().replace("```sql", "").replace("```", "").strip()

    return sql_query

# --- 2️⃣ Execute SQL ---
def execute_query(sql_query: str) -> List[Dict[str, Any]]:
    """Executes SQL query in BigQuery and returns rows as list of dicts."""
    print(f"-> Executing SQL: {sql_query}")
    query_job = bigquery_client.query(sql_query)
    return [dict(row) for row in query_job]

# --- 3️⃣ Interpret Results (SQL → Natural Language) ---
def interpret_results(question: str, raw_result: List[Dict[str, Any]]) -> str:
    """
    Converts raw query result into a conversational natural language response.
    
    UPDATED: Includes highly specific instructions to force line breaks and remove markdown.
    """
    raw_result_json = json.dumps(raw_result, indent=2)
    prompt = f"""
    A user asked: "{question}"

    The database returned:
    {raw_result_json}

    Please summarize this result in a clear, natural, and conversational tone.
    
    CRITICAL INSTRUCTION: You MUST format the output for multiple records using plain text, colons, and forced line breaks. 
    
    Do NOT use any markdown characters, including asterisks (**).
    
    The format for EACH customer record MUST be:
    
    [Field Name]: [Value]\n\n 
    
    Use a single line break after the colon and value, and then a second line break (i.e., a blank line) before the next field name. Use a double line break (i.e., one blank line) to separate each customer's complete block of details.

    Example Output MUST look exactly like this:
    
    Here are the details for the first client:
    
    Customer ID: 
    20462
    
    Sentiment: 
    Initially frustrated due to recurring failed recharge transactions, the customer's sentiment improved significantly after the support agent provided an effective alternative solution using the Airtel Thanks app, leading to a successful recharge. The support agent was helpful and proactive.
    
    Problem Type: 
    Recharge
    
    Transcript:
    (it should be in this format)
    Support: Good evening. Thank you for calling Airtel International Support. I'm Vikram. How may I help you?
    Customer: Hi Vikram, I'm traveling to Singapore tomorrow and my international roaming isn't working, even though I activated it.
    Support: I understand this is urgent for your travel. Let me check your roaming status. May I have your Airtel number?
    Customer: It's 8876543210. I activated the 1299 plan yesterday as recommended.
    Support: Thank you. Checking your roaming activation. I see the plan is active but needs manual provisioning. Let me do that now.
    Customer: How long will this take? My flight is in eight hours.
    Support: It should activate within 30 minutes. I'm prioritizing your request. Done. Your roaming will be active before your flight.
    Customer: Thank goodness. Will I get confirmation?
    Support: Yes, you'll receive an SMS confirmation shortly. Is there anything else you need for your travel?
    Customer: No, that covers it. Thanks for the quick help.
    Support: Safe travels. Enjoy your trip with Airtel.
    

    Do NOT include SQL or JSON structure. Just give the answer directly.
    """

    print("-> Interpreting results using Gemini...")
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# --- 4️⃣ INTERACTIVE CONSOLE LOOP ---
def interactive_nl2sql_analysis():
    """Interactive prompt to ask natural language questions about BigQuery data."""
    schema = get_table_schema(BIGQUERY_PROJECT_ID, BIGQUERY_DATASET, BIGQUERY_TABLE)
    if schema.startswith("Error"):
        print("❌ Cannot start without valid schema access.")
        return

    print("\n--- Vertex AI + BigQuery NL2SQL Interactive Analyzer ---")
    print(f"Connected to: {BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}")
    print("Ask questions about customer calls, e.g., 'What is the most common problem type?'")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("Your Question > ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Session ended. Goodbye!")
            break
        if not user_input:
            continue

        try:
            sql_query = nl_to_sql(user_input, schema)
            if not sql_query.lower().startswith("select"):
                print("⚠️ Gemini did not produce a SELECT query. Try rephrasing your question.")
                continue

            query_results = execute_query(sql_query)
            if not query_results:
                print("ℹ️ Query executed successfully but returned no results.")
                continue

            final_answer = interpret_results(user_input, query_results)
            print("\n--- Answer ---")
            print(final_answer)
            print("--------------\n")

        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            print("Please verify your BigQuery permissions and query correctness.\n")

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    interactive_nl2sql_analysis()