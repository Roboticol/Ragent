import chromadb
from chromadb.config import Settings
from agent.orchestrator import run_agent

client = chromadb.Client(
    Settings(persist_directory="./chroma_db")
)

collection = client.get_or_create_collection(
    name="research_rag"
)

goal = input("Enter goal: ")

result = run_agent(goal, mode="research")
