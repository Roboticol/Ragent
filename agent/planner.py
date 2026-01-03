from agent.llm import call_llm


def create_plan(goal):
    prompt = f"""
You are an autonomous research agent.
Break the following goal into clear, ordered steps.
If the goal is asking you to read a file, then you will be fed the file later. Have patience.

Goal:
{goal}

Return a numbered list.
"""
    response = call_llm(prompt)
    steps = [line for line in response.split("\n") if line.strip()]
    return steps
