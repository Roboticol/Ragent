from agent.orchestrator import run_agent

goal = input("Enter goal: ")

result = run_agent(goal, mode="research")
