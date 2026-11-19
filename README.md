# Multi-Agent Parking Reservation System (LangGraph & FastAPI)

## DS Part: System Architecture & AI Decision Making

### Overview
This project implements an intelligent, multi-agent parking reservation system designed to automate customer inquiries while maintaining strict data security and human oversight. The solution leverages Large Language Models (LLMs) orchestrated via LangGraph, integrating conversational AI with external knowledge bases and deterministic execution environments.

### Deep Dive: How the System Works (End-to-End Flow)
The architecture is designed to handle a full reservation lifecycle securely and autonomously until human approval is required. Here is the step-by-step data flow:

1. **User Input & PII Guardrails (Presidio):** 
   When a user sends a message (e.g., "Book a spot for John Doe, +48 123 456 789"), the message first passes through a custom `GuardrailsManager`. Microsoft Presidio detects sensitive entities (PERSON, PHONE_NUMBER, LICENSE_PLATE) and replaces them with secure tokens (e.g., `[TOKEN_PERSON_0]`). The LLM never sees the real personal data.
   
2. **Agent Reasoning & Routing (LangGraph):**
   The tokenized prompt is passed to the main Conversational Agent (`agent_node`). Using a ReAct (Reasoning and Acting) prompt loop, the LLM determines what it needs to do next. It has access to two main tools:
   * `check_rules`: Queries the **Milvus Vector Database** (RAG) to understand static parking regulations (e.g., VIP sector rules, EV charging policies).
   * `check_database`: Generates and executes a SQL query (Text-to-SQL) against the **PostgreSQL Database** to find real-time availability, protected by a strict **Read-Only SQL Validator** preventing any unauthorized modifications (`DROP`, `UPDATE`, `DELETE`, etc.).

3. **Data Collection & Escalation:**
   Once the agent has gathered all 6 required reservation details (Name, Phone, Plate, Sector, Start Time, End Time) and verified availability, it attempts to submit the reservation, saving it initially as `PENDING` in PostgreSQL.

4. **Human-In-The-Loop (HITL) Checkpoint & Admin Copilot:**
   An independent background process (`admin_app.py`) acts as an AI Admin, performing time-overlap and duplicate checks per vehicle/license plate.
   * If a time conflict or vehicle duplicate is detected, the system automatically rejects the request.
   * Otherwise, it prompts the Administrator via CLI (`Y/N`). If approved, it proceeds to finalization.

5. **Dual-Write Audit (FastAPI):**
   Upon approval, the system performs a secure dual-write operation:
   * **Step A:** Updates the PostgreSQL database status from `PENDING` to `APPROVED`.
   * **Step B:** Sends a secure, authenticated POST request (with an API key) to the standalone **FastAPI server**, which independently writes an unalterable audit log to `confirmed_reservation.txt`.

### Solutions & Reasoning
* **Multi-Agent Orchestration (LangGraph):** Designed a robust architecture featuring conversational RAG agents and an independent Admin Copilot with automated checks and Human-In-The-Loop (HITL) validation.
* **PII Vault Tokenization:** Guardrails are applied in-flight, ensuring 0% PII leakage to the LLM.
* **Read-Only SQL Guard:** Enforced strict regex checks on LLM-generated SQL queries to ensure only safe `SELECT` statements are executed.
* **Duplicate Prevention Logic:** Implemented time-window overlap validation checking existing `PENDING` and `APPROVED` statuses against license plates.

### Outcomes & Insights
The final system successfully bridges generative AI with deterministic business logic. The dual-write transaction ensures that a reservation is only finalized when a human administrator approves it, resulting in a 100% safe, auditable, and privacy-compliant AI pipeline.

---

## MLE Part: Engineering & Deployment

### Repository Overview
The repository is structured to separate the AI orchestration logic from the execution environment. It features a conversational bot module, an automated Admin Copilot, and audit server built with FastAPI.

### Project Structure
```text
.
├── data/
│   └── docs/
│       └── regulations.md        # Markdown file containing parking rules for RAG
├── infrastructure/
│   └── docker-compose.yml        # Docker configuration for PostgreSQL
├── src/
│   ├── bot/
│   │   ├── admin_app.py          # Administrator Agent background process
│   │   ├── app.py                # Main entry point and CLI interface for users
│   │   ├── evaluate_rag.py       # RAG evaluation script
│   │   ├── graph.py              # LangGraph nodes, routing, and HITL logic
│   │   ├── guardrails.py         # PII tokenization (Presidio) integration
│   │   ├── retriever.py          # Vector database (Milvus) retrieval logic
│   │   ├── router.py             # Query routing logic for the agent
│   │   └── sql_chain.py          # Text-to-SQL logic with read-only security guards
│   ├── database/
│   │   └── seed.py               # DB initialization and synthetic data generator
│   ├── FastAPI/
│   │   └── server.py             # FastAPI microservice for secure audit logging
│   ├── tests/
│   │   └── test_parking_system.py# Pytest unit and integration test suite
│   └── utils/
│       ├── llm_config.py         # Centralized LLM and Embeddings configuration
│       └── test_connections.py   # Script to test DB and API connections
├── .env.example                  # Template for environment variables
├── .gitignore                    # Git ignored files configuration
├── confirmed_reservation.txt     # Output log file generated by FastAPI (created when run)
├── evaluation_report.txt         # RAG evaluation results (created when run)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

### Step-by-step Instructions

#### 0. Prerequisites & Local Environment Setup
Before running the system, ensure you have Python 3.10+ and Docker Desktop installed.

1. **Clone the repository and set up a virtual environment:**
   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   python -m venv venv
   ```
2. **Activate the virtual environment:**
   * On Windows: `venv\Scripts\activate`
   * On macOS/Linux: `source venv/bin/activate`
3. **Install dependencies and the required NLP model:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
4. **Configure Environment Variables:**
   Copy the example `.env` file and fill in your Azure OpenAI keys and API configurations.
   ```bash
   cp .env.example .env
   ```

#### 1. How to run training/setup with Docker
*Note: This project relies on initializing the local Milvus vector store and the PostgreSQL relational database.*

1. **Start the infrastructure using Docker Compose:**
   Ensure your local Docker Desktop is running, then execute:
   ```bash
   docker-compose -f infrastructure/docker-compose.yml up -d
   ```
2. **Initialize and Seed the Databases:** 
   Run the `seed.py` script to set up tables, generate synthetic parking data, and embed regulations into Milvus.
   ```bash
   python -m src.database.seed
   ```

#### 2. How to run tests
   To run the automated test suite using `pytest`:
   ```bash
   python -m pytest -v
   ```

#### 3. How to run inference
The system requires running three concurrent terminal processes:

**Terminal 1: Start the FastAPI Server**
```bash
uvicorn src.FastAPI.server:app --port 8000 --reload
```

**Terminal 2: Start the AI Agent (Chat and declaring reservation)**
```bash
python -m src.bot.app
```

**Terminal 3: Start the AI Admin Agent (Admin panel with summary and acceptance of reservation)**
```bash
python -m src.bot.admin_app
```

#### Description of Outputs
* **PostgreSQL (`reservations` & `parking_spots` tables):** Updates spot availability (`is_occupied = TRUE`) and stores the detokenized reservation details.
* **Audit Log (`confirmed_reservation.txt`):** The FastAPI server generates a physical log entry upon successful database transaction. Format: `Name | License Plate | Start Time - End Time | Approval Timestamp`.
* **Standard Output (CLI):** Real-time conversational interface with system logs detailing tool usage, PII tokenization, and HITL requests (acting as the Administrator interface).