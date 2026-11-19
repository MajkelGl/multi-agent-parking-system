import os
import psycopg2
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, ToolMessage
from src.utils.llm_config import load_llm
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from src.bot.sql_chain import run_sql_query
from src.bot.retriever import run_chain as run_milvus_query
from src.bot.guardrails import guardrails

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    reservation_data: dict

sys_msg = SystemMessage(
    content="""
    You are a helpful parking assistant. 
    Your job is to answer user questions accurately and help them with parking spot reservations.

    RULES FOR GENERAL & PRICING INQUIRIES:
    1. For questions about prices, rules, regulations, or policies, ALWAYS call `check_rules`.
    2. Respond to the user using ONLY the retrieved facts from `check_rules`.

    RULES FOR AVAILABILITY:
    1. Use `check_database` to check spot availability or specific spot features.

    RULES FOR RESERVATIONS:
    1. Collect ALL 6 pieces of details before calling `submit_reservation`:
       - Full Name
       - Phone number (+48 format)
       - License plate number (eg. DW02359, needs to be without spaces)
       - Desired sector (Only B or VIP)
       - Start date and time
       - End date and time

    2. Once you have all 6 pieces of information, call `submit_reservation` IMMEDIATELY.
    3. After calling `submit_reservation`, tell the user that their request was saved as PENDING and they must wait for Administrator approval.

    CRITICAL PRIVACY RULE:
    - Internal placeholders like [TOKEN_...] may appear in tool arguments.
    - NEVER display or mention any placeholder like [TOKEN_...] in your final response to the user.
    """
)

@tool
def submit_reservation(name: str, license_plate: str, sector: str, start_time: str, end_time: str, phone: str = "Missing phone number") -> str:
    """
    Use this tool ONLY when the user wants to make a reservation and provides all 6 required pieces of information: 
    name, phone number, license plate, sector, start time and end time.
    Never guess these pieces of information.
    """
    return "Sending request to database as PENDING."

@tool
def check_database(query: str):
    """Use this to check parking spot availability, EV chargers, and disabled spots."""
    return run_sql_query(query)

@tool
def check_rules(query: str):
    """Use this to check parking rules, pricing, penalties, and general information."""
    return run_milvus_query(query)


def guardrail_node(state: AgentState):
    last_message = state["messages"][-1]
    safe_text = guardrails.tokenize(last_message.content)
    
    if safe_text != last_message.content:
        print("\n[GUARDRAILS] Secured sensitive data!")
    
    last_message.content = safe_text
    return {"messages": [last_message]}


def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        raw_query = tool_call["args"].get("query", "")
        
        print(f"\n[SYSTEM] Agent is using: {tool_name}...")
        
        if tool_name == "check_database":
            res = run_sql_query(raw_query)
            tool_messages.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
            
        elif tool_name == "check_rules":
            res = run_milvus_query(raw_query)
            tool_messages.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
            
    return {"messages": tool_messages}

def receptionist_node(state: AgentState):
    llm = load_llm()
    llm_with_tools = llm.bind_tools([submit_reservation, check_database, check_rules])
    
    messages = state.get("messages", [])
    messages = [sys_msg] + messages
    
    answer = llm_with_tools.invoke(messages)

    if answer.tool_calls:
        for tool_call in answer.tool_calls:
            if tool_call["name"] == "submit_reservation":
                return {
                    "messages": [answer],
                    "reservation_data": tool_call["args"] 
                }
        return {"messages": [answer]}
    else:
        return {"messages": [answer]}


def database_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_call = next((tc for tc in last_message.tool_calls if tc["name"] == "submit_reservation"), last_message.tool_calls[0])
    tool_call_id = tool_call["id"]
    
    res_data = tool_call["args"] 
    real_res_data = guardrails.detokenize(res_data)

    raw_sector = real_res_data.get('sector', 'B')
    clean_sector = raw_sector.upper().replace("SECTOR", "").strip()
    
    print("\n[Database Node] Connecting to PostgreSQL to save PENDING reservation...")
    
    try:
        conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
        cur = conn.cursor()

        cur.execute("""
            SELECT id, spot_number FROM parking_spots 
            WHERE sector = %s 
            AND id NOT IN (
                SELECT spot_id FROM reservations 
                WHERE status IN ('APPROVED', 'PENDING')
                AND start_time < %s AND end_time > %s
            )
            LIMIT 1 FOR UPDATE
        """, (clean_sector, real_res_data.get('end_time'), real_res_data.get('start_time')))
        
        spot = cur.fetchone()
        
        if not spot:
            conn.rollback()
            cur.close()
            conn.close()
            guardrails.clear_vault()
            msg = f"SYSTEM: Unfortunately, in sector {clean_sector} there are no available spots for this time period."
            return {"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
            
        spot_id, spot_number = spot
        
        cur.execute("""
            INSERT INTO reservations 
            (spot_id, guest_name, phone_number, license_plate, start_time, end_time, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'PENDING')
        """, (
            spot_id, 
            real_res_data.get('name', 'Missing name'), 
            real_res_data.get('phone', 'Missing phone number'), 
            real_res_data.get('license_plate', 'No license plate'), 
            real_res_data.get('start_time', 'Missing start time'), 
            real_res_data.get('end_time', 'Missing end time')
        ))
        
        conn.commit()
        cur.close()
        conn.close()

        final_msg = f"SYSTEM: Reservation successfully saved as PENDING for spot {spot_number}. Inform the user that it awaits administrator approval."
        print(f"[Database Node] {final_msg}")
        
        guardrails.clear_vault()
        
        return {"messages": [ToolMessage(content=final_msg, tool_call_id=tool_call_id)]}
        
    except Exception as e:
        print(f"\n[Error DB] {e}")
        guardrails.clear_vault() 
        return {"messages": [ToolMessage(content="SYSTEM: An error occurred while saving to the database.", tool_call_id=tool_call_id)]}

def route_receptionist(state: AgentState):
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        if any(tc["name"] == "submit_reservation" for tc in last_message.tool_calls):
            return "database_node"
        else:
            return "tool_node"
            
    return END

builder = StateGraph(AgentState)

builder.add_node("guardrail_node", guardrail_node)
builder.add_node("receptionist_node", receptionist_node)
builder.add_node("database_node", database_node)
builder.add_node("tool_node", tool_node)

builder.set_entry_point("guardrail_node")

builder.add_edge("guardrail_node", "receptionist_node")
builder.add_conditional_edges("receptionist_node", route_receptionist)
builder.add_edge("database_node", "receptionist_node")
builder.add_edge("tool_node", "receptionist_node")

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)