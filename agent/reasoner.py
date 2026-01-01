from agent.llm import call_llm


def reason_over_content(content, goal):
    prompt = f"""
Goal:
{goal}

Content:
{content}

Explain the important ideas clearly.
"""
    return call_llm(prompt)
