import requests
import json

# Set your OpenRouter API key here
API_KEY = "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"

# Define the OpenRouter endpoint
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def query(user_input):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    else:
        return f"Error {response.status_code}: {response.text}"

# Example usage
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    response = query(user_query)
    print("Response:", response)