from agent.llm import call_llm


def reason_over_content(content, goal):
    prompt = f"""
Goal:
{goal}

Make sure to infer the most important points when reasoning, store them in a list.
Make sure you do all this within 500 words.

Content:
{content}

Explain the important ideas clearly.
"""
    return call_llm(prompt)
