import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def post_ingest():
    print("Calling /ingest ...")
    response = requests.post(f"{BASE_URL}/ingest")
    if response.status_code == 200:
        print("Ingested files/chunks:", response.json().get("ingested_files"))
    else:
        print("Ingest failed:", response.text)

def post_parse(queries):
    print("\nCalling /parse ...")
    payload = {"query": queries}
    response = requests.post(f"{BASE_URL}/parse", json=payload)
    if response.status_code == 200:
        print("Parse results:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Parse failed:", response.text)

def post_search(query):
    print("\nCalling /search ...")
    payload = {"query": query}
    response = requests.post(f"{BASE_URL}/search", json=payload)
    if response.status_code == 200:
        print("Search results:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Search failed:", response.text)

def post_answer(queries):
    print("\nCalling /answer ...")
    payload = {"query": queries}
    response = requests.post(f"{BASE_URL}/answer", json=payload)
    if response.status_code == 200:
        print("Answer results:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Answer failed:", response.text)

if __name__ == "__main__":
    # Step 1: Ingest docs
    post_ingest()

    # Step 2: Parse queries
    test_queries = [
        "46M, knee surgery in Pune, 3-month policy",
        "25F, maternity coverage in Delhi for 9 months",
    ]
    post_parse(test_queries)

    # Step 3: Search a query
    post_search("knee surgery in Pune")

    # Step 4: Get detailed answer for queries
    post_answer(test_queries)
