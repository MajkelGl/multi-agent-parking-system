# Multi-Agent Parking Reservation System (LangGraph & MCP)

## DS Part: System Architecture & AI Decision Making

### Overview
This project implements an intelligent, multi-agent parking reservation system designed to automate customer inquiries while maintaining strict data security and human oversight. The solution leverages Large Language Models (LLMs) orchestrated via LangGraph, integrating conversational AI with external knowledge bases and deterministic execution environments.

### Deep Dive: How the System Works (End-to-End Flow)
The architecture is designed to handle a full reservation lifecycle securely and autonomously until human approval is required. Here is the step-by-step data flow:

1. **User Input & PII Guardrails (Presidio):** 
   When a user sends a message (e.g., "Book a spot for John Doe, +48 123 456 789"), the message first passes through a custom `GuardrailsManager`. Microsoft Presidio detects sensitive entities (PERSON, PHONE_NUMBER) and replaces them with secure tokens (e.g., `[TOKEN_PERSON_0]`). The LLM never sees the real personal data.
   
2. **Agent Reasoning & Routing (LangGraph):**
   The tokenized prompt is passed to the main Conversational Agent (`agent_node`). Using a ReAct (Reasoning and Acting) prompt loop, the LLM determines what it needs to do next. It has access to two main tools:
   * `check_rules`: Queries the **Milvus Vector Database** (RAG) to understand static parking regulations (e.g., VIP sector rules, EV charging policies).
   * `check_database`: Generates and executes a SQL query (Text-to-SQL) against the **PostgreSQL Database** to find real-time availability.

3. **Data Collection & Escalation:**
   Once the agent has gathered all 6 required reservation details (Name, Phone, Plate, Sector, Start Time, End Time) and verified availability, it attempts to call the `submit_reservation` tool. 

4. **Human-In-The-Loop (HITL) Checkpoint:**
   LangGraph intercepts the `submit_reservation` call using an `interrupt_before` mechanism. The system pauses the AI's execution and escalates the request to the Administrator via the CLI (`admin_node`). 
   * If the Admin rejects (presses 'N'), a `ToolMessage` is injected back into the graph, and the AI politely informs the user of the rejection.
   * If the Admin approves (presses 'Y'), the system proceeds to finalization.

5. **Detokenization & Dual-Write Audit (FastAPI MCP):**
   Upon approval, the `admin_node` retrieves the real personal data from the Presidio Vault (`detokenize`). It then performs a secure dual-write operation:
   * **Step A:** Updates the PostgreSQL database (marks the spot as occupied and saves the reservation).
   * **Step B:** Sends a secure, authenticated POST request (with an API key) to the standalone **FastAPI (MCP) server**, which independently writes an unalterable audit log to `confirmed_reservation.txt`.

### Key Problems & Challenges Encountered
1. **Data Privacy (PII):** Handling sensitive user data (names, phone numbers, license plates) during LLM processing posed a security risk.
2. **Infinite Tool Loops (ReAct):** The agent occasionally entered recursive loops when querying rule databases, attempting to validate multiple constraints simultaneously.
3. **gRPC Connection Instability:** Long-running conversational pauses led to `ENHANCE_YOUR_CALM (too_many_pings)` errors from the vector database.
4. **Execution Safety:** Autonomous database modifications required human validation before execution.

### Solutions & Reasoning
* **Multi-Agent Orchestration (LangGraph):** Designed a two-agent architecture consisting of a Conversational RAG Agent (interacting with users) and an Execution Agent (Administrator) with a Human-In-The-Loop (HITL) checkpoint.
* **PII Vault Tokenization:** Guardrails are applied in-flight. Data is dynamically detokenized only after HITL approval, right before database insertion.
* **Hybrid Knowledge Retrieval:** Combines dynamic Text-to-SQL (PostgreSQL) with semantic search (Milvus).
* **Loop & Stability Handling:** Enforced strict system prompt constraints, applied LangGraph recursion limits (`recursion_limit=10`), and configured gRPC C-core environment variables (`GRPC_ARG_KEEPALIVE_TIME_MS`) to ensure connection stability during idle chat states.

### Outcomes & Insights
The final system successfully bridges generative AI with deterministic business logic. The dual-write transaction ensures that a reservation is only finalized when a human administrator approves it, resulting in a 100% safe, auditable, and privacy-compliant AI pipeline.

---

## MLE Part: Engineering & Deployment

### Repository Overview
The repository is structured to separate the AI orchestration logic from the execution environment. It features a conversational bot module and a standalone Model Context Protocol (MCP) audit server built with FastAPI.

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
│   │   ├── app.py                # Main entry point and CLI interface
│   │   ├── evaluate_rag.py       # RAG evaluation script
│   │   ├── graph.py              # LangGraph nodes, routing, and HITL logic
│   │   ├── guardrails.py         # PII tokenization (Presidio) integration
│   │   ├── retriever.py          # Vector database (Milvus) retrieval logic
│   │   ├── router.py             # Query routing logic for the agent
│   │   └── sql_chain.py          # Text-to-SQL logic for PostgreSQL
│   ├── database/
│   │   └── seed.py               # DB initialization and synthetic data generator
│   ├── mcp/
│   │   └── server.py             # FastAPI microservice for secure audit logging
│   └── utils/
│       ├── llm_config.py         # Centralized LLM and Embeddings configuration
│       └── test_connections.py   # Script to test DB and API connections
├── .env.example                  # Template for environment variables
├── .gitignore                    # Git ignored files configuration
├── confirmed_reservation.txt     # Output log file generated by MCP (created when run)
├── evaluation_report.txt         # RAG evaluation results (created when run, evaluate_rag.py)
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
*Note: This project does not require traditional model training, but relies on setting up the vector store (Milvus) and relational database (PostgreSQL) for context retrieval.*

1. **Start the infrastructure using Docker Compose:**
   Ensure your local Docker Desktop is running, then execute:
   ```bash
   docker-compose -f infrastructure/docker-compose.yml up -d
   ```
2. **Initialize and Seed the Databases:** 
   Run the `seed.py` script. This script drops existing tables (if any), creates the necessary schema, and **generates realistic synthetic data** (randomized names, phone numbers, license plates, and parking spot occupancies) so the PostgreSQL database is populated for immediate testing. It also embeds the Markdown documentation into the Milvus vector store.
   ```bash
   python -m src.database.seed
   ```

#### 2. How to run inference
The system requires running two concurrent services: the MCP Audit Server and the Bot interface.

**Terminal 1: Start the MCP Server (FastAPI)**
```bash
uvicorn src.mcp.server:app --port 8000 --reload
```

**Terminal 2: Start the AI Agent (Inference & Chat)**
```bash
python -m src.bot.app
```

#### Description of Outputs
* **PostgreSQL (`reservations` & `parking_spots` tables):** Updates spot availability (`is_occupied = TRUE`) and stores the detokenized reservation details.
* **Audit Log (`confirmed_reservation.txt`):** The MCP server generates a physical log entry upon successful database transaction. Format: `Name | License Plate | Start Time - End Time | Approval Timestamp`.
* **Standard Output (CLI):** Real-time conversational interface with system logs detailing tool usage, PII tokenization, and HITL requests (acting as the Administrator interface).

#### Reported Metrics
* **E2E Dual-Write Success Rate:** 100% (The pipeline ensures the MCP audit log is only appended if the PostgreSQL transaction and HITL approval succeed).
* **PII Leakage Rate:** 0% (All sensitive entities are successfully tokenized prior to LLM processing).