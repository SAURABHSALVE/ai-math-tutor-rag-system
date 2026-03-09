# Math Mentor - AI-Powered JEE Math Tutor

A multimodal AI application that solves JEE-style math problems using **GPT-4o**, **LangGraph** multi-agent orchestration, **LangChain + FAISS** RAG, human-in-the-loop verification, and **SQLite**-backed memory.

## Architecture

```mermaid
graph TD
    A["🎓 Student"] --> B["Streamlit UI"]

    subgraph INPUT["📥 Multimodal Input Layer"]
        B --> C{Input Mode?}
        C -->|"✏️ Text"| D["Text Input (passthrough)"]
        C -->|"📷 Image"| E["Mistral OCR (primary)"]
        C -->|"🎤 Audio"| F["OpenAI Whisper API"]
        E -->|fallback| E2["EasyOCR → GPT-4o Vision"]
    end

    E --> G{"Confidence ≥ threshold?"}
    E2 --> G
    F --> G
    D --> H["Raw Math Text"]
    G -->|"✅ High"| H
    G -->|"⚠️ Low"| I["HITL: Student Reviews & Edits Text"]
    I --> H

    H --> PIPELINE

    subgraph PIPELINE["🤖 LangGraph Multi-Agent Pipeline (6 Agents)"]
        direction TB
        AG1["🛡️ Agent 0: Guardrail
        Validates input is a math problem
        (gpt-4o-mini)"]
        --> AG2["📝 Agent 1: Parser
        Cleans text, extracts equation,
        detects topic & ambiguity
        (gpt-4o-mini)"]
        --> AGQ{Needs Clarification?}
        AGQ -->|"Yes"| HITL2["HITL: Ask Student"]
        AGQ -->|"No"| AG3
        AG3["🔀 Agent 2: Intent Router
        Keyword + regex topic detection,
        picks solve strategy, generates RAG queries
        (gpt-4o-mini)"]
        --> AG4["🧮 Agent 3: Solver
        Uses RAG context + memory +
        SymPy calculator to solve
        (gpt-4o)"]
        --> AG5["✅ Agent 4: Verifier
        Dual-run SymPy check, confidence scoring,
        triggers HITL if uncertain
        (gpt-4o)"]
        --> AG6["📖 Agent 5: Explainer
        Step-by-step student-friendly explanation,
        key concepts, tips, common mistakes
        (gpt-4o)"]
    end

    subgraph RAG["📚 RAG Pipeline"]
        R1["Knowledge Base (6 MD docs)
        Algebra · Calculus · Probability
        Linear Algebra · Mistakes · Templates"]
        --> R2["LangChain Text Splitter
        (500 chars, 50 overlap)"]
        --> R3["OpenAI text-embedding-3-small"]
        --> R4["FAISS Vector Store
        (Top-5 retrieval)"]
    end

    subgraph MEMORY["🧠 Memory & Self-Learning"]
        M1["SQLite Database"]
        M2["Similar Problem Retrieval
        (cosine similarity on embeddings)"]
        M3["Correction Patterns
        (learned from student feedback)"]
        M1 --- M2
        M1 --- M3
    end

    AG3 -.->|"query"| R4
    R4 -.->|"relevant context"| AG3
    AG3 -.->|"query"| M2
    M2 -.->|"past solutions"| AG3
    M3 -.->|"correction hints"| AG3

    AG6 --> OUTPUT

    subgraph OUTPUT["📊 Output & Feedback"]
        O1["Final Answer
        (formatted with math symbols)"]
        O2["Step-by-Step Solution"]
        O3["Verification Details + Confidence"]
        O4["Key Concepts · Tips · Common Mistakes"]
        O5["Difficulty Rating"]
        O6["Agent Trace (full pipeline visibility)"]
        O1 --- O2 --- O3 --- O4 --- O5 --- O6
    end

    OUTPUT --> FB{Student Feedback}
    FB -->|"✅ Correct"| S1["Store in SQLite Memory"]
    FB -->|"❌ Incorrect + Correction"| S2["Store Correction Pattern"]
    S1 --> M1
    S2 --> M1

    style INPUT fill:#1e1e2e,stroke:#818cf8,color:#cdd6f4
    style PIPELINE fill:#1e1e2e,stroke:#c084fc,color:#cdd6f4
    style RAG fill:#1e1e2e,stroke:#f9e2af,color:#cdd6f4
    style MEMORY fill:#1e1e2e,stroke:#a6e3a1,color:#cdd6f4
    style OUTPUT fill:#1e1e2e,stroke:#89b4fa,color:#cdd6f4
```

## Core Stack

| Layer | Tool | Why |
|-------|------|-----|
| LLM | GPT-4o + GPT-4o-mini | Heavy agents (Solver, Verifier, Explainer) use GPT-4o; lightweight agents (Guardrail, Parser, Router) use GPT-4o-mini for speed |
| OCR | Mistral OCR (primary) + EasyOCR + GPT-4o Vision (fallbacks) | Best math OCR accuracy with multi-engine fallback |
| Agent Framework | LangGraph | Typed state, conditional routing, HITL support, agent trace |
| RAG | LangChain + FAISS | Fast similarity search, no infra needed |
| Embeddings | text-embedding-3-small | Cheap, fast, OpenAI-native |
| SymPy | SymPy Calculator | Dual-run symbolic verification of solutions |
| Audio (ASR) | OpenAI Whisper API (gpt-4o-transcribe) | One-liner, math-aware transcription |
| UI | Streamlit | Dark-themed dashboard with topic cards, memory stats, agent trace |
| Memory | SQLite + OpenAI Embeddings | Similar problem retrieval, correction pattern learning, survives restarts |
| Deployment | Streamlit Cloud | Free, instant, reviewer gets a live link |

## Features

- **Multimodal Input**: Text, Image (GPT-4o Vision), Audio (Whisper API)
- **5 LangGraph Agents**: Parser, Intent Router, Solver, Verifier, Explainer
- **RAG Pipeline**: LangChain text splitter + FAISS + OpenAI embeddings
- **Human-in-the-Loop**: Triggers on low confidence, ambiguity, verification failures
- **Memory & Self-Learning**: SQLite-backed store; retrieves similar problems, learns from corrections
- **Agent Trace**: Full visibility into each agent's output
- **Confidence Indicators**: Visual confidence scores throughout

## Supported Topics

- Algebra (equations, identities, sequences, inequalities)
- Probability (counting, distributions, Bayes' theorem)
- Calculus (limits, derivatives, optimization)
- Linear Algebra (matrices, determinants, vectors, eigenvalues)

## Setup & Run

### Prerequisites

- Python 3.9+
- OpenAI API Key (GPT-4o access)

### Installation

```bash
git clone <repo-url>
cd ai-math

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Project Structure

```
ai-math/
├── app.py                  # Streamlit UI
├── agents.py               # LangGraph multi-agent pipeline (5 agents)
├── rag_pipeline.py         # LangChain + FAISS RAG
├── memory_layer.py         # SQLite memory store
├── input_handlers.py       # GPT-4o Vision + Whisper API
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── .env.example            # Environment template
├── knowledge_base/         # Curated math knowledge (6 docs)
│   ├── algebra_formulas.md
│   ├── probability_formulas.md
│   ├── calculus_formulas.md
│   ├── linear_algebra_formulas.md
│   ├── common_mistakes.md
│   └── solution_templates.md
├── memory_store/           # SQLite DB (auto-created)
└── vector_store/           # FAISS index (auto-created)
```

## How It Works

1. **Input** - Student provides a math problem via text, image, or audio
2. **Extraction** - GPT-4o Vision (images) or Whisper API (audio) extracts text; user can edit (HITL)
3. **LangGraph Pipeline** runs 5 agents sequentially with typed state:
   - **Parser** - Cleans input, identifies topic, detects ambiguity
   - **Intent Router** - Selects strategy, generates RAG queries
   - **Solver** - Uses RAG context + memory + SymPy to solve
   - **Verifier** - Checks correctness, triggers HITL if unsure
   - **Explainer** - Creates student-friendly step-by-step explanation
4. **Feedback** - Student marks correct/incorrect; corrections stored in SQLite for future learning

## UI Screenshots

### Home Page — Text Input & Solution
The main interface features a dark-themed dashboard with topic selection, memory stats, pipeline trace, and a clean input area supporting Text, Image, and Audio modes.

![Home - Text Input & Answer](UI/Screenshot%20(41).png)

### Step-by-Step Solution & Verification
After solving, the app displays a detailed step-by-step solution with confidence score, verification status, difficulty rating, and agent trace.

![Step-by-Step Solution](UI/Screenshot%20(42).png)

### Verification, Common Mistakes & Input Corrections
The verifier agent checks correctness independently. Common mistakes and input corrections are highlighted to help students learn.

![Verification & Common Mistakes](UI/Screenshot%20(43).png)

### Difficulty Rating, Key Concepts & Feedback
Each solution ends with a difficulty rating, key concepts, tips, common mistakes summary, and a feedback section for self-learning.

![Difficulty Rating & Feedback](UI/Screenshot%20(44).png)

---

## Example Solution — Image Input (Linear Algebra)

End-to-end walkthrough of solving `Find det([[-4,-2],[5,4]])` from an uploaded image.

### 1. Image Upload & OCR Extraction
A math problem image is uploaded. Mistral OCR extracts the text with 95% confidence. The student can edit the extracted text before solving (HITL).

![Image Upload & OCR](solutions/math_39-sol/Screenshot%20(36).png)

### 2. Final Answer & Agent Trace
The solver returns the final answer (`-6`) with a full step-by-step explanation. The Agent Trace panel shows each agent's status (Guardrail, Parser, Router, Solver, Verifier, Explainer). Retrieved RAG context is displayed alongside.

![Solution & Agent Trace](solutions/math_39-sol/Screenshot%20(37).png)

### 3. Solution Steps, SymPy Verification & Confidence
Detailed arithmetic steps are shown. SymPy solver independently verifies the result. Retrieved context from the knowledge base and overall confidence score are visible.

![Steps & SymPy Verification](solutions/math_39-sol/Screenshot%20(38).png)

### 4. Key Concepts, Tips & Verification Details
Key concepts, tips for similar problems, and common mistakes are listed. The verification panel confirms correctness with a detailed checklist.

![Key Concepts & Verification](solutions/math_39-sol/Screenshot%20(39).png)

### 5. Difficulty, Expandable Details & Feedback
Difficulty rating, expandable sections (Parsed Problem, Solution Raw, Verification, Routing Strategy), and Correct/Incorrect feedback buttons for self-learning.

![Difficulty & Feedback](solutions/math_39-sol/Screenshot%20(40).png)

---

## Deployment (Streamlit Cloud)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, set `app.py` as main file
4. Add `OPENAI_API_KEY` in Secrets
5. Deploy - reviewer gets a live link
