import hmac
import os
from threading import Lock
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
write_lock = Lock()

class ReservationPayload(BaseModel):
    full_name: str
    registration_number: str
    start_time: str
    end_time: str


def authenticate(x_api_key: str = Header(None)):
    real_api_key = os.getenv("X_API_KEY")
    
    if not x_api_key or not real_api_key or not hmac.compare_digest(
        x_api_key.encode("utf-8"), 
        real_api_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.post("/api/reservations")
async def read_data(payload: ReservationPayload, authorized: bool = Depends(authenticate)):
    approval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saving_data = f"{payload.full_name} | {payload.registration_number} | {payload.start_time} - {payload.end_time} | {approval_time}"
    
    with write_lock:
        try:
            with open("confirmed_reservation.txt", "a", encoding="utf-8") as f:
                f.write(saving_data + "\n")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to write audit log: {str(e)}"
            )

    return {"status": "success", "message": "Reservation logged"}