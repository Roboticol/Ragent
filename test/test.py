from ragas.metrics import (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall)
from ragas import evaluate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import requests
from datasets import Dataset
from ragas.run_config import RunConfig
import asyncio
import json
import time

evaluator_llm = LangchainLLMWrapper(OllamaLLM(model="mashriram/sarvam-m"))
indic_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="mashriram/sarvam-m"))
eval_langs = ["english", "hindi", "bengali", "marathi"]

metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=indic_embeddings),
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
]

async def prepare_metrics():
    for metric in metrics:
        metric.llm = evaluator_llm

def query(query_text, lang="english"):
    res = requests.post(
        "http://localhost:8000/query", 
        json={"query": query_text, "top_k": 5, "language": lang}, 
        headers={"x-api-key": "SECRET"}
    )
    data = res.json()
    return data["answer"], data["sources"]

# Example Multilingual Test Set
test_set = [];
print("Test set loading...")
with open("./test/test_set.json", "r", encoding='utf-8') as f:
    test_set = json.load(f)["data"];
print("Test set loaded!")
print(test_set)

for i in test_set:
    if i["lang"] in eval_langs:
        print(i["lang"])
        i["answer"], i["contexts"] = query(i["user_input"], lang=i["lang"])
        time.sleep(2)
    else:
        test_set.remove(i)


dataset = Dataset.from_list(test_set)

run_config = RunConfig(
    timeout=600,
    max_workers=1,
    max_retries=3,
)

results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=indic_embeddings,
    run_config=run_config
)

print(results)