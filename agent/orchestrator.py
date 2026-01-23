from agent.prompt import prompt

def run_agent(query, sources, mode="research"):
    print("Agent running...")
    response = prompt(query, sources)
    print("Returned response!")

    return response