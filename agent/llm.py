import requests


def call_llm(prompt, system="You are a helpful AI"):
    payload = {
        "model": "llama3:8b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    r = requests.post(
        "http://localhost:11434/api/chat",
        json=payload
    )
    return r.json()["message"]["content"]
