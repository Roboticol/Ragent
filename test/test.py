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

evaluator_llm = LangchainLLMWrapper(OllamaLLM(model="llama3.1"))
indic_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="mashriram/sarvam-1"))

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
with open("./test/test_set.json", "r", encoding='utf-8') as f:
    test_set = json.load(f)["data"];

for i in test_set:
    print(i["user_input"], i["lang"])
    i["answer"], i["contexts"] = query(i["user_input"], lang=i["lang"])

print(test_set)

dataset = Dataset.from_list(test_set)

run_config = RunConfig(
    timeout=600,  # bump to 10 min, llama3.1 is slow
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