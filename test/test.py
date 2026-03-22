from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from ragas.llms import LangchainLLMWrapper
import requests

llm = LangchainLLMWrapper(OllamaLLM(model="mashriram/sarvam-1"))
embeddings = OllamaEmbeddings(model="mashriram/sarvam-1")

# results = evaluate(
#     dataset=dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
#     llm=llm,
#     embeddings=embeddings,
# )

res = requests.post("http://localhost:8000/query", json={"query": "bla bla bla", "top_k": 5, "language": "english"}, headers={"x-api-key": "SECRET"})
print(res.text)

# test_set = [
#     {
#         "question": "...",
#         "ground_truth": "...",
#         "answer": query("..."),
#         "contexts": chromadb.retrieve("...")
#     }
# ]

# dataset = Dataset.from_list(samples)

# results = evaluate(
#     dataset=dataset,
#     metrics=[
#         faithfulness,
#         answer_relevancy,
#         context_precision,
#         context_recall,
#     ]
# )

# print(results)