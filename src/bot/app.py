import uuid
from langchain_core.messages import HumanMessage, ToolMessage
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

        state = graph.get_state(config)
        
        if state.next and state.next[0] == "admin_node":
            last_msg = state.values["messages"][-1]
            tool_call_id = last_msg.tool_calls[0]["id"]
            
            res_data = last_msg.tool_calls[0]["args"]
            print(f"\n[HITL] Administrator acceptance is needed:")
            print(f"       Data: {res_data}")
            
            approval = input("Proceed with reservation and save to PostgreSQL? (Y/N): ")
            
            if approval.strip().upper() in ["T", "Y"]:
                print("[SYSTEM] Administrator approved. Saving to database...")
                graph.invoke(None, config=config)
            else:
                print("[SYSTEM] Administrator rejected. Informing the bot...")
                
                rejection_msg = ToolMessage(
                    content="SYSTEM: The administrator rejected the reservation. Tell the user that unfortunately the administrator did not approve the reservation and you cannot do anything about it.",
                    tool_call_id=tool_call_id
                )
                
                graph.update_state(config, {"messages": [rejection_msg]}, as_node="admin_node")
                
                graph.invoke(None, config=config)

        current_state = graph.get_state(config)
        if current_state.values.get("messages"):
            last_msg = current_state.values["messages"][-1]
            if last_msg.type == "ai" and last_msg.content:
                print(f"\nAssistant: {last_msg.content}\n")

if __name__ == "__main__":
    run_cli()