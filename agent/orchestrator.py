from agent.planner import create_plan
from agent.reasoner import reason_over_content
from agent.critic import critique
from tools.file_reader import read_text_files

from tools.pdf_reader import read_pdfs
from tools.chunker import chunk_text
from tools.code_parser import load_codebase


def run_agent(goal, mode="research"):
    print("[ORCHESTRATOR RUNNING]")
    print("[PLANNER RUNNING]")
    plan = create_plan(goal)
    print("[PLAN CREATED]")
    print(plan)

    collected_knowledge = ""

    if mode == "research":
        docs = read_text_files("data/papers")
        pdfs = read_pdfs("data/pdfs")
        print(f"[READ {len(docs) + len(pdfs)} FILES IN RESEARCH MODE]")

        for name, text in pdfs:
            chunks = chunk_text(text)
            print(f"[PDF] {name} split into {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                collected_knowledge += reason_over_content(
                    chunk, goal, f"{name} (chunk {i+1})")
                collected_knowledge += "\n"

        for name, text in docs:
            print("[REASONING]")
            collected_knowledge += reason_over_content(text, goal)

    if mode == "code":
        code = load_codebase("data/codebase")
        print(f"[READ {len(code)} FILES IN CODING MODE]")
        for name, src in code:
            print("[REASONING]")
            collected_knowledge += reason_over_content(src, goal)

    print("[CRITIC RUNNING]")
    feedback = critique(collected_knowledge, goal)

    return {
        "plan": plan,
        "analysis": collected_knowledge,
        "critique": feedback
    }
