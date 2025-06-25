from graph import graph
from state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

def init_state() -> AgentState:
    return {
        "input": "",
        "chat_history": [],
        "output": "",
        "current_date": ""
    }

def run_loop():
    state = init_state()
    print("Agent running—type ‘quit’ to exit.")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        # Update and invoke graph
        state["input"] = user_input
        state = graph.invoke(state)

        # Print and append to history
        print("\nAssistant:", state["output"])

    print("\nGoodbye!")

if __name__ == "__main__":
    run_loop()