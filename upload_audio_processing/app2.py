from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
from flask import Flask, render_template, request, jsonify
from nlp_sql import nl_to_sql, execute_query, interpret_results, get_table_schema
from nlp_sql import BIGQUERY_PROJECT_ID, BIGQUERY_DATASET, BIGQUERY_TABLE

app = Flask(__name__, template_folder='templates2', static_folder='style2')

# Load schema once
schema = get_table_schema(BIGQUERY_PROJECT_ID, BIGQUERY_DATASET, BIGQUERY_TABLE)

@app.route("/")
def home():
    return render_template("index2.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_question = request.json.get("question", "")
        if not user_question:
            return jsonify({"response": "Please enter a question."})

        sql_query = nl_to_sql(user_question, schema)
        results = execute_query(sql_query)
        answer = interpret_results(user_question, results)
        return jsonify({"response": answer})

    except Exception as e:
        print(e)
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
