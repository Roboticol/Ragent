from ragas.metrics import (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall)
from ragas import evaluate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_ollama import OllamaLLM
import re
from datasets import Dataset
from ragas.run_config import RunConfig
import asyncio
import json
import time

# class StripThinkingOllamaLLM(OllamaLLM):
#     def _generate(self, prompts, stop=None, **kwargs):
#         result = super()._generate(prompts, stop=stop, **kwargs)
#         cleaned = []
#         for gen in result.generations:
#             for g in gen:
#                 text = re.sub(r"<think>.*?</think>", "", g.text, flags=re.DOTALL).strip()
#                 cleaned.append([type(g)(text=text, generation_info=g.generation_info)])
#         result.generations = cleaned
#         return result

evaluator_llm = LangchainLLMWrapper(
    OllamaLLM(model="aya:8b")
)
indic_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings( model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", model_kwargs={"device": "cpu"}))
eval_langs = ["english", "hindi", "bengali", "marathi", "tamil", "marathi"]

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
    if i["lang"] not in eval_langs:
        test_set.remove(i)

for i in test_set:
    if i["lang"] in eval_langs:
        print(i["lang"])
        i["response"], i["retrieved_contexts"] = query(i["user_input"], lang=i["lang"])
        print(i["retrieved_contexts"])
        time.sleep(2)


dataset = Dataset.from_list(test_set)

run_config = RunConfig(
    timeout=1800,
    max_workers=1,
    max_retries=1,
)

results = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=indic_embeddings,
    run_config=run_config
)

print(results)