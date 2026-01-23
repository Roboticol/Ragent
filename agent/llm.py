import requests


def call_llm(prompt, system="You are a helpful AI"):
    payload = {
        "model": "llama3.2:3b",
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

    # print("returned message: ", r.json())
    return r.json()["message"]["content"]
