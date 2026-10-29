from fastapi import FastAPI, Depends, HTTPException, Header
from typing import Annotated
import uvicorn 
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from datetime import datetime

app = FastAPI()

class ReservationPayload(BaseModel):
    full_name: str
    registration_number: str
    start_time: str
    end_time: str


def authenticate(x_api_key: str = Header()):
    load_dotenv()
    real_api_key = os.getenv("X_API_KEY")
    if x_api_key != real_api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    else:
        pass

@app.post("/api/reservations")
async def read_data(payload: ReservationPayload, authorized: bool = Depends(authenticate)):
    approval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saving_data = f"{payload.full_name} | {payload.registration_number} | {payload.start_time} - {payload.end_time} | {approval_time}"
    with open("confirmed_reservation.txt", "a", encoding="utf-8") as f:
        f.write(saving_data + "\n")

    return {"status": "success", "message": "Reservation logged"}