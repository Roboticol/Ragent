from agent.llm import call_llm


def critique(answer, goal):
    prompt = f"""
Goal:
{goal}

Draft Answer:
{answer}

Critique this answer.
Identify missing points, errors, or improvements.
"""
    return call_llm(prompt, system="You are a critical reviewer.")
