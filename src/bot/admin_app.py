import os
import time
import json
import psycopg2
import requests
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.llm_config import load_llm

load_dotenv()

def fetch_pending_reservation():
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id, spot_id, guest_name, phone_number, license_plate, start_time, end_time, status 
        FROM reservations 
        WHERE status = 'PENDING' 
        ORDER BY id ASC 
        LIMIT 1
    """)
    
    reservation = cur.fetchone()
    cur.close()
    conn.close()
    
    if reservation:
        return {
            "id": reservation[0],
            "spot_id": reservation[1],
            "full_name": reservation[2],
            "phone_number": reservation[3],
            "license_plate": reservation[4],
            "start_time": str(reservation[5]),
            "end_time": str(reservation[6]),
            "status": reservation[7]
        }
    return None

def check_spot_availability(spot_id: int, start_time: str, end_time: str, current_res_id: int) -> bool:
    """Checks, if the spot is physically available and there is no conflict with APPROVED reservations."""
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id FROM reservations 
        WHERE spot_id = %s 
          AND status = 'APPROVED'
          AND id != %s
          AND start_time < %s 
          AND end_time > %s
    """, (spot_id, current_res_id, end_time, start_time))
    
    conflict = cur.fetchone()
    cur.close()
    conn.close()
    
    return conflict is None

def check_user_has_overlapping_reservation(license_plate: str, start_time: str, end_time: str, current_res_id: int) -> bool:
    """Checks, if the given vehicle (license_plate) already has a PENDING or APPROVED reservation in this time window."""
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id FROM reservations 
        WHERE license_plate = %s 
          AND status IN ('PENDING', 'APPROVED')
          AND id != %s
          AND start_time < %s 
          AND end_time > %s
    """, (license_plate, current_res_id, end_time, start_time))
    
    existing = cur.fetchone()
    cur.close()
    conn.close()
    
    return existing is not None

def update_reservation_status(res_id: int, new_status: str):
    conn = psycopg2.connect(os.getenv("POSTGRES_URL"))
    cur = conn.cursor()
    
    cur.execute("""
        UPDATE reservations 
        SET status = %s 
        WHERE id = %s
    """, (new_status, res_id))
    
    conn.commit()
    cur.close()
    conn.close()

def send_to_fastapi(reservation_data: dict) -> bool:
    api_data = {
        "full_name": reservation_data["full_name"],
        "registration_number": reservation_data["license_plate"],
        "start_time": reservation_data["start_time"],
        "end_time": reservation_data["end_time"]
    }
    
    headers = {"X-API-Key": os.getenv("X_API_KEY")}
    
    try:
        response = requests.post(
            "http://localhost:8000/api/reservations", 
            json=api_data, 
            headers=headers,
            timeout=5.0
        )
        if response.status_code == 200:
            print("[FastAPI Server] Data successfully saved to confirmed_reservation.txt")
            return True
        else:
            print(f"[FastAPI Server Error] Status Code: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("[FastAPI Server Error] Request timed out after 5 seconds.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[FastAPI Server Error] Connection failed: {e}")
        return False

def run_admin_agent():
    print("Admin Initialized. Waiting for PENDING reservations...\n")
    llm = load_llm()
    
    admin_template = """
    You are an AI assistant for a Facility Administrator.
    Analyze the following parking reservation request.
    
    Reservation Details:
    Name: {full_name}
    Phone: {phone_number}
    License Plate: {license_plate}
    Start Time: {start_time}
    End Time: {end_time}
    
    Task:
    1. Summarize the request in one short sentence.
    2. Verify if the provided data looks like genuine, real-world data (e.g., phone number length, license plate format).
    3. Point out any red flags or confirm if it looks perfectly fine.
    
    Output ONLY a valid JSON object in this exact format, with no markdown:
    {{
        "summary": "Short summary.",
        "data_looks_valid": true,
        "warnings": "List any red flags here, or write 'None. Everything looks correct.'"
    }}
    """
    
    prompt = PromptTemplate.from_template(admin_template)
    admin_chain = prompt | llm | StrOutputParser()

    while True:
        pending_res = fetch_pending_reservation()
        
        if pending_res:
            print(f"[NEW REQUEST] Reservation ID: {pending_res['id']} | Guest: {pending_res['full_name']}")
            print("Running database checks (collisions & duplicate user bookings)...")
            
            is_duplicate = check_user_has_overlapping_reservation(
                pending_res["license_plate"],
                pending_res["start_time"],
                pending_res["end_time"],
                pending_res["id"]
            )
            
            if is_duplicate:
                print(f"[System] Duplicate detected! License plate {pending_res['license_plate']} already has an active/pending reservation for this timeframe.")
                print(f"[System] Automatically rejecting reservation ID {pending_res['id']}...")
                update_reservation_status(pending_res["id"], "REJECTED")
                print("[Database] Status updated to REJECTED automatically due to duplicate.")
                time.sleep(2)
                continue

            is_available = check_spot_availability(
                pending_res["spot_id"], 
                pending_res["start_time"], 
                pending_res["end_time"], 
                pending_res["id"]
            )
            
            if not is_available:
                print(f"[System] Database conflict detected! Spot ID {pending_res['spot_id']} is already occupied for this time window.")
                print(f"[System] Automatically rejecting reservation ID {pending_res['id']}...")
                update_reservation_status(pending_res["id"], "REJECTED")
                print("[Database] Status updated to REJECTED automatically.")
                time.sleep(2)
                continue

            print("Checks passed. Running AI analysis...")
            
            raw_analysis = admin_chain.invoke(pending_res)
            clean_analysis = raw_analysis.replace("```json", "").replace("```", "").strip()
            
            try:
                analysis_data = json.loads(clean_analysis)
                summary = analysis_data.get("summary", "N/A")
                is_valid = analysis_data.get("data_looks_valid", False)
                warnings = analysis_data.get("warnings", "N/A")
            except Exception as e:
                summary = "Failed to generate AI summary."
                is_valid = False
                warnings = f"JSON Parsing Error: {clean_analysis}"

            print("\n AI ADMIN REPORT ")
            print(f"Spot Available: YES")
            print(f"Summary:        {summary}")
            print(f"Data Valid:     {is_valid}")
            print(f"Warnings:       {warnings}\n")
            
            decision = input("\nDo you approve this reservation? (Y/N): ").strip().upper()
            
            if decision == 'Y':
                print("[System] Attempting to issue ticket via FastAPI Server...")
                FastAPI_success = send_to_fastapi(pending_res)
                
                if FastAPI_success:
                    update_reservation_status(pending_res["id"], "APPROVED")
                    print("[Database] Success! Status updated to APPROVED.")
                else:
                    update_reservation_status(pending_res["id"], "FastAPI_ERROR")
                    print("[Database] Aborted! Status updated to FastAPI_ERROR due to API failure.")
            else:
                update_reservation_status(pending_res["id"], "REJECTED")
                print("[Database] Reservation REJECTED.\n\n")
                
            
        time.sleep(5)

if __name__ == "__main__":
    run_admin_agent()