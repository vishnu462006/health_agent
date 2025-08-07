import os
import re
import json
import nltk
import docx
import pdfplumber
import extract_msg
import requests
from email import message_from_bytes
from nltk.tokenize import sent_tokenize
from langdetect import detect
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
from typing import List, Union

# === CONFIG ===
PINECONE_API_KEY = "pcsk_7RNs5X_MyuECusJWqmjF7jzfWvxFSSYhtQ4zN4XzXCiCyibVhNBJejiZo5HaNtYaaEEmMU"
PINECONE_INDEX = "query-retrival-system"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_DIMENSION = 384
PINECONE_METRIC = "cosine"

OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "gpt-4o-mini"

nltk.download("punkt")

# === INIT MODELS & CLIENTS ===
model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=PINECONE_DIMENSION,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

index = pc.Index(PINECONE_INDEX)

# === FILE PROCESSING FUNCTIONS ===
def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_msg(file_path):
    msg = extract_msg.Message(file_path)
    return msg.body if msg.body else ""

def extract_text_from_eml(file_path):
    with open(file_path, 'rb') as f:
        msg = message_from_bytes(f.read())
    return msg.get_payload() if msg.get_payload() else ""

def chunk_text_paragraphs(text, max_words=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def ingest_file(file_path):
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif ext.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif ext.endswith(".msg"):
        text = extract_text_from_msg(file_path)
    elif ext.endswith(".eml"):
        text = extract_text_from_eml(file_path)
    else:
        return 0

    chunks = chunk_text_paragraphs(text)
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        vectors.append({
            "id": f"{os.path.basename(file_path)}_{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })

    if vectors:
        index.upsert(vectors)
        return len(vectors)
    return 0

def semantic_search(query, top_k=5):
    query_embedding = model.encode(query).tolist()
    res = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [{"score": match["score"], "text": match["metadata"].get("text", "[No data]")} for match in res["matches"] if match["metadata"].get("text")]

def regex_parse_query(text: str):
    # age+gender e.g. "25F", "46M"
    age_gender_match = re.search(r"(\d{1,3})\s*([MFmf])", text)
    age = int(age_gender_match.group(1)) if age_gender_match else None
    gender = None
    if age_gender_match:
        gender = "male" if age_gender_match.group(2).lower() == "m" else "female"

    # location (simple heuristic: look for "in <Location>")
    location_match = re.search(r"in ([A-Za-z\s]+)", text)
    location = location_match.group(1).strip() if location_match else None

    # duration like "3-month", "9 months", "1 year"
    duration_match = re.search(r"(\d+\s*[-]?\s*(month|months|year|years))", text, re.I)
    duration = duration_match.group(1).replace(" ", "") if duration_match else None

    # disease/treatment - heuristic: between age/gender and location/duration
    # Remove age/gender and location/duration parts to isolate disease phrase:
    temp_text = text
    if age_gender_match:
        temp_text = temp_text.replace(age_gender_match.group(0), "")
    if location_match:
        temp_text = temp_text.replace(location_match.group(0), "")
    if duration_match:
        temp_text = temp_text.replace(duration_match.group(0), "")

    # Clean commas and extra spaces
    disease = temp_text.replace(",", "").strip() or None

    return {
        "age": age,
        "gender": gender,
        "disease": disease if disease else None,
        "location": location,
        "duration": duration
    }

def parse_query_with_llm(query):
    # Support both str and list inputs
    queries = query if isinstance(query, list) else [query]
    results = []

    system_prompt = """
You are a health insurance assistant. Extract key data from the query.
Respond as JSON with keys: age, gender, disease, location, duration (or null if not found).
Example:
{
  "age": 38,
  "gender": "male",
  "disease": "kidney transplant",
  "location": "Nagpur",
  "duration": "9-month"
}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    for q in queries:
        data = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q}
            ]
        }

        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=data)
            response.raise_for_status()
            content = response.json()
            parsed = json.loads(content["choices"][0]["message"]["content"])
            # Basic validation of parsed keys
            if not all(k in parsed for k in ["age", "gender", "disease", "location", "duration"]):
                raise ValueError("Missing keys in LLM response")
            results.append(parsed)
        except Exception:
            # Fallback to regex parsing if LLM fails
            parsed = regex_parse_query(q)
            results.append(parsed)

    return results[0] if len(results) == 1 else results

# === FASTAPI APP ===
app = FastAPI(title="Bajaj Hackathon Insurance Query System")

class QueryRequest(BaseModel):
    query: Union[str, List[str]]

class IngestResponse(BaseModel):
    ingested_files: int

class ParseResponse(BaseModel):
    parsed_data: dict

class SearchResult(BaseModel):
    score: float
    text: str

class SearchResponse(BaseModel):
    results: List[SearchResult]

class AnswerResponse(BaseModel):
    answer: dict

@app.post("/ingest", response_model=IngestResponse)
def ingest_docs():
    folder = "docs"
    if not os.path.exists(folder):
        raise HTTPException(status_code=400, detail=f"Folder '{folder}' not found")
    files = [f for f in os.listdir(folder) if f.lower().endswith((".pdf", ".docx", ".eml", ".msg"))]
    total = 0
    for f in files:
        total += ingest_file(os.path.join(folder, f))
    return IngestResponse(ingested_files=total)

@app.post("/parse", response_model=ParseResponse)
def parse_query(req: QueryRequest):
    try:
        parsed = parse_query_with_llm(req.query)
        return ParseResponse(parsed_data=parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
def search(req: QueryRequest):
    try:
        results = semantic_search(req.query)
        return SearchResponse(results=[SearchResult(score=r["score"], text=r["text"]) for r in results])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answer")
def get_answer(req: QueryRequest):
    try:
        queries = req.query if isinstance(req.query, list) else [req.query]
        all_answers = []

        for query in queries:
            results = semantic_search(query, top_k=1)
            if not results:
                all_answers.append({
                    "question": query,
                    "answer": {
                        "policy": None,
                        "benefit": None,
                        "details": {}
                    },
                    "message": "Sorry, no relevant information found in the policy."
                })
                continue

            top_chunk = results[0]["text"]

            prompt = f"""
You are an expert health insurance assistant. Based on the policy content below, answer the user's question with a detailed JSON object in this format:

{{
  "policy": "Policy Name here",
  "benefit": "Benefit name here",
  "details": {{
    "waiting_period": "value if relevant",
    "description": "detailed description",
    "notes": [
      "Note 1",
      "Note 2"
    ]
  }}
}}

Policy Content:
{top_chunk}

User's Question: {query}

Provide only the JSON object as the answer.
"""

            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": "You answer clearly with structured JSON as instructed."},
                    {"role": "user", "content": prompt}
                ]
            }
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }

            llm_response = requests.post(OPENROUTER_URL, headers=headers, json=payload)

            print("LLM response status:", llm_response.status_code)  # Debug print
            print("LLM response raw:", llm_response.text)           # Debug print

            if llm_response.status_code != 200:
                raise Exception(f"OpenRouter API error: {llm_response.status_code} - {llm_response.text}")

            response_json = llm_response.json()

            if "choices" not in response_json:
                raise Exception(f"OpenRouter response missing 'choices': {response_json}")

            answer_text = response_json["choices"][0]["message"]["content"]

            try:
                answer_json = json.loads(answer_text)
            except Exception:
                answer_json = {"answer_raw": answer_text}

            all_answers.append({
                "question": query,
                "answer": answer_json
            })

        return {"answers": all_answers if len(all_answers) > 1 else all_answers[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# === WEB UI ROUTE ===
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Health Insurance Query Assistant</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
<style>
    body { padding: 20px; }
    .result { margin-top: 20px; white-space: pre-wrap; background: #f8f9fa; padding: 15px; border-radius: 5px; }
</style>
</head>
<body>
<div class="container">
    <h1 class="mb-4">Health Insurance Query Assistant</h1>

    <div class="mb-4">
        <label for="queryInput" class="form-label">Enter your Query:</label>
        <input type="text" class="form-control" id="queryInput" placeholder="e.g. 38M, kidney transplant in Nagpur, 9-month insurance" />
    </div>

    <div class="d-flex gap-2">
        <button class="btn btn-primary" onclick="parseQuery()">Parse Query</button>
        <button class="btn btn-secondary" onclick="searchQuery()">Search Docs</button>
        <button class="btn btn-success" onclick="getAnswer()">Get Answer</button>
    </div>

    <div id="result" class="result mt-4"></div>
</div>

<script>
async function parseQuery() {
    const query = document.getElementById('queryInput').value;
    if (!query) {
        alert('Please enter a query');
        return;
    }
    const res = await fetch('/parse', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query})
    });
    const data = await res.json();
    document.getElementById('result').textContent = JSON.stringify(data, null, 2);
}

async function searchQuery() {
    const query = document.getElementById('queryInput').value;
    if (!query) {
        alert('Please enter a query');
        return;
    }
    const res = await fetch('/search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query})
    });
    const data = await res.json();
    document.getElementById('result').textContent = JSON.stringify(data, null, 2);
}

async function getAnswer() {
    const query = document.getElementById('queryInput').value;
    if (!query) {
        alert('Please enter a query');
        return;
    }
    const res = await fetch('/answer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query})
    });
    const data = await res.json();
    document.getElementById('result').textContent = JSON.stringify(data, null, 2);
}
</script>

</body>
</html>
"""

if __name__ == "__main__":
    print("Starting API server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
