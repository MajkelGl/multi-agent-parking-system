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
import requests
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    reservation_data: dict
    admin_decision: str

sys_msg = SystemMessage(
    content="""
    You are a helpful parking assistant. 
    Your job is to answer user questions and help them with parking spot reservations.

    ALWAYS use check_rules for policy/pricing queries and if it is posible to do the reservation on user terms then use
    check_database for availability queries. If they are not possible explain the rules for the user.
    
    If the client asks for a reservation in an allowed sector(and additionally terms are okay, for example you can't park 
    normal vehicle in Sector B[everything u need is in check_rules]), you must collect ALL of the following information before proceeding:
    - Full Name
    - Phone number (for emergency contact needs to be with +48 format)
    - License plate number
    - Desired parking sector (Only B or VIP)
    - Start date and time
    - End date and time

    Don't second guess input data. Just proceed what u have got.
    
    IMPORTANT: If there is a placeholder like [TOKEN_PHONE_NUMBER_0], treat it as a valid, 
    real value and pass it EXACTLY as it is to the reservation tool, this is used to preserve PII from getting compromised. 
    Additionally don't print anything with this [TOKEN____], because user has no idea what it is. It is internal in system.
    
    Ask for the missing information in a polite and natural way. 
    Do not use the reservation tool until you have all 6 pieces of information.

    Don't ask about picking spots. Assign first that is available.

    CRITICAL TOOL USAGE RULES:
    1. Before proceeded with reservation ALWAYS use 'check_rules' ONCE per user prompt, because you need to verify is expectations can be met.
    2. Do NOT call `check_rules` and 'check_database more than ONCE per user turn during doing the reservation.
    3. If you already have all required user details (Name, Phone, Plate, Sector, Dates) and you checked spot availability via `check_database`, immediately call `submit_reservation`. 
    4. Do not over-query the vector store (`check_rules`) if the user simply wants to make a reservation.
    5. If user is asking about other things than reservation, don't gather his personal information. Only do it when he want's to reserve a spot.
    """
)

@tool
def submit_reservation(name: str, license_plate: str, sector: str, parking_type: str, start_time: str, end_time: str, phone: str = "Missing phone number") -> str:
    """
    Use this tool ONLY when the user wants to make a reservation and provides all 6 required pieces of information: 
    name, phone number, license plate, sector, start time and end time.
    Never guess these pieces of information.
    """
    return "Sended approval for administrators."

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
        tool_name = tool_call["name"]
        raw_query = tool_call["args"].get("query", "")

        clean_query = guardrails.detokenize(raw_query)
        
        print(f"\n[SYSTEM] Agent is using: {tool_name}...")
        
        if tool_name == "check_database":
            res = run_sql_query(clean_query)
            tool_messages.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
            
        elif tool_name == "check_rules":
            res = run_milvus_query(clean_query)
            tool_messages.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
            
    return {"messages": tool_messages}

def receptionist_node(state: AgentState):
    llm = load_llm()
    llm_with_tools = llm.bind_tools([submit_reservation, check_database, check_rules])
    
    messages = state.get("messages", [])
    messages = [sys_msg] + messages
    
    answer = llm_with_tools.invoke(messages)

    if answer.tool_calls:
        tool_name = answer.tool_calls[0]["name"]
        if tool_name == "submit_reservation":
            return {
                "messages": [answer],
                "reservation_data": answer.tool_calls[0]["args"] 
            }
        else:
            return {"messages": [answer]}
    else:
        return {"messages": [answer]}

def admin_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    tool_call_id = tool_call["id"]
    
    res_data = tool_call["args"] 
    
    real_res_data = guardrails.detokenize(res_data)

    raw_sector = real_res_data.get('sector', 'B')
    clean_sector = raw_sector.upper().replace("SECTOR", "").strip()
    
    print("\n[Admin Node] Connecting to PostgreSQL database to save reservation...")
    
    try:
        conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
        cur = conn.cursor()

        cur.execute("""
            SELECT id, spot_number FROM parking_spots 
            WHERE sector = %s AND is_occupied = FALSE 
            LIMIT 1 FOR UPDATE
        """, (clean_sector,))
        
        spot = cur.fetchone()
        
        if not spot:
            conn.rollback()
            cur.close()
            conn.close()
            msg = f"Unfortunatelly, in sector {real_res_data['sector']} there are no available spots."
            return {"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]}
            
        spot_id, spot_number = spot
        
        cur.execute("""
            UPDATE parking_spots 
            SET is_occupied = TRUE 
            WHERE id = %s
        """, (spot_id,))
        
        cur.execute("""
            INSERT INTO reservations 
            (spot_id, guest_name, phone_number, license_plate, start_time, end_time)
            VALUES (%s, %s, %s, %s, %s, %s)
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
        success_msg = f"Reservation saved in the database! Assigned spot: {spot_number}."

        api_data = {
            "full_name": real_res_data.get("name"),
            "registration_number": real_res_data.get("license_plate"),
            "start_time": real_res_data.get("start_time"),
            "end_time": real_res_data.get("end_time")
        }

        headers = {"X-API-Key": os.getenv("X_API_KEY")}

        try:
            response = requests.post("http://localhost:8000/api/reservations", json=api_data, headers=headers)
            if response.status_code == 200:
                print("[MCP Server] Saved data in confirmed_reservation.txt")
        except Exception as e:
            print(f"[MCP Server Error] {e}")
        
        print(f"[Admin Node] {success_msg}")
        
        return {"messages": [ToolMessage(content=success_msg, tool_call_id=tool_call_id)]}
        
    except Exception as e:
        print(f"\n[Error DB] {e}")
        error_msg = "An error occurred while saving to the database."
        return {"messages": [ToolMessage(content=error_msg, tool_call_id=tool_call_id)]}

def route_receptionist(state: AgentState):
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        if last_message.tool_calls[0]["name"] == "submit_reservation":
            return "admin_node"
        else:
            return "tool_node"
            
    return END

builder = StateGraph(AgentState)

builder.add_node("guardrail_node", guardrail_node)
builder.add_node("receptionist_node", receptionist_node)
builder.add_node("admin_node", admin_node)
builder.add_node("tool_node", tool_node)

builder.set_entry_point("guardrail_node")

builder.add_edge("guardrail_node", "receptionist_node")

builder.add_conditional_edges("receptionist_node", route_receptionist)
builder.add_edge("admin_node", "receptionist_node")
builder.add_edge("tool_node", "receptionist_node")

memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["admin_node"] 
)