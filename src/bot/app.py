import os
import uuid

os.environ["GRPC_ARG_KEEPALIVE_TIME_MS"] = "60000"
os.environ["GRPC_ARG_KEEPALIVE_TIMEOUT_MS"] = "20000"
os.environ["GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS"] = "0"
os.environ["GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA"] = "0"

from langchain_core.messages import HumanMessage
from src.bot.graph import graph

def run_cli():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("Parking Assistant Started. Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break

        if not user_input.strip():
            continue

        graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        current_state = graph.get_state(config)
        if current_state.values.get("messages"):
            last_msg = current_state.values["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"\nAssistant: {last_msg.content}\n")

if __name__ == "__main__":
    run_cli()