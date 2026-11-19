import os
from dotenv import load_dotenv
import psycopg2
import random
from datetime import datetime, timedelta

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from src.utils.llm_config import load_llm, load_embeddings

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
MILVUS_FILE_PATH = os.getenv("MILVUS_FILE_PATH", "./milvus_parking.db")

CREATE_TABLE_QUERY = """
DROP TABLE IF EXISTS reservations;
DROP TABLE IF EXISTS parking_spots;

CREATE TABLE parking_spots (
    id INTEGER PRIMARY KEY,
    spot_number VARCHAR(10) UNIQUE NOT NULL,
    sector VARCHAR(10) NOT NULL,
    has_ev_charger BOOLEAN DEFAULT FALSE,
    is_disabled_only BOOLEAN DEFAULT FALSE
);

CREATE TABLE reservations (
    id SERIAL PRIMARY KEY,
    spot_id INTEGER REFERENCES parking_spots(id),
    guest_name VARCHAR(100),
    phone_number VARCHAR(20),
    license_plate VARCHAR(20),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20) DEFAULT 'APPROVED'
);
"""

first_names = ["Jan", "Piotr", "Michal", "Tomasz", "Maciej", "Anna", "Katarzyna", "Magdalena", "Agnieszka", "Julia"]
last_names = ["Kowalski", "Nowak", "Wisniewski", "Wojcik", "Kowalczyk", "Kaminski", "Lewandowski", "Zielinski", "Szymanski"]

def gen_name():
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def gen_phone():
    return f"+48 {random.randint(500, 899)} {random.randint(100, 999)} {random.randint(100, 999)}"

def gen_plate():
    return f"DW {random.randint(10000, 99999)}"

def seed_database():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(POSTGRES_URL)
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_QUERY)

    spots = []
    reservations = []
    now = datetime.now()
    spot_id = 1

    for i in range(1, 101):
        spots.append((spot_id, f"A-{i:03d}", "A", False, False))
        spot_id += 1

    for i in range(1, 51):
        is_ev = True if 21 <= i <= 35 else False
        is_disabled = True if 36 <= i <= 50 else False
        spots.append((spot_id, f"B-{i:03d}", "B", is_ev, is_disabled))
        
        for _ in range(random.randint(0, 2)):
            start_offset = random.randint(-48, 48)
            start = (now + timedelta(hours=start_offset)).strftime("%Y-%m-%d %H:00:00")
            end = (now + timedelta(hours=start_offset + random.randint(2, 12))).strftime("%Y-%m-%d %H:00:00")
            reservations.append((spot_id, gen_name(), gen_phone(), gen_plate(), start, end))
            
        spot_id += 1

    for i in range(1, 26):
        spots.append((spot_id, f"VIP-{i:03d}", "VIP", False, False))
        if random.choice([True, False]):
            start = (now - timedelta(hours=random.randint(1, 3))).strftime("%Y-%m-%d %H:00:00")
            end = (now + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d %H:00:00")
            reservations.append((spot_id, gen_name(), gen_phone(), gen_plate(), start, end))
        spot_id += 1

    insert_spots_query = """
        INSERT INTO parking_spots (id, spot_number, sector, has_ev_charger, is_disabled_only)
        VALUES (%s, %s, %s, %s, %s)
    """
    cur.executemany(insert_spots_query, spots)

    insert_reservations_query = """
        INSERT INTO reservations (spot_id, guest_name, phone_number, license_plate, start_time, end_time)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cur.executemany(insert_reservations_query, reservations)

    conn.commit()
    cur.close()
    conn.close()

def seed_vector_database():
    print("1. Configuration of loader for .md files...")
    loader = DirectoryLoader('data/docs/', glob="**/*.md", loader_cls=TextLoader)
    
    print("2. Loading files from disk...")
    documents = loader.load()
    print(f" -> Loaded {len(documents)} files.")

    print("3. Splitting text into chunks (text_splitter)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f" -> Created {len(docs)} text chunks.")

    print("4. Initializing OpenAI Embeddings model...")
    embeddings = load_embeddings()

    print("5. Attempting to connect to Milvus Lite and save data...")
    vector_db = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        connection_args={"uri": MILVUS_FILE_PATH},
        collection_name="parking_rules",
        drop_old=True
    )
    print("6. Vector database seeded successfully!")

if __name__ == "__main__":
    seed_database()
    seed_vector_database()
    print("Seeding completed successfully!")