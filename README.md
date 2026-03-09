# Math Mentor - AI-Powered JEE Math Tutor

**[Try it live here](https://ai-math-wise.streamlit.app/)** — for more details and understanding, read below.

A multimodal AI application that solves JEE-style math problems using **GPT-4o**, **LangGraph** multi-agent orchestration, **LangChain + FAISS** RAG, human-in-the-loop verification, and **SQLite**-backed memory.

## Architecture

### High-Level System Flow

```mermaid
graph LR
    A[" Student "] ==> B[" Streamlit UI "]
    B ==> C[" Input Layer — Text / Image / Audio "]
    C ==> D[" LangGraph Pipeline — 6 AI Agents "]
    D <-.-> E[" RAG — FAISS + LangChain "]
    D <-.-> F[" Memory — SQLite + Embeddings "]
    D ==> G[" Output — Answer + Explanation "]
    G ==> H[" Feedback Loop "]
    H ==> F
```

---

### Multimodal Input Layer

```mermaid
graph TD
    A[" Student uploads problem "] --> B{" Select Input Mode "}

    B -->|" Text "| C[" Direct Text Input — passthrough "]

    B -->|" Image "| D[" Mistral OCR — primary engine "]
    D -->|" if fails "| D2[" EasyOCR — fallback 1 "]
    D2 -->|" if fails "| D3[" GPT-4o Vision — fallback 2 "]

    B -->|" Audio "| E[" OpenAI Whisper API — gpt-4o-transcribe "]

    D --> F{" Confidence >= threshold? "}
    D2 --> F
    D3 --> F
    E --> F

    C --> G[" Raw Math Text "]
    F -->|" High confidence "| G
    F -->|" Low confidence "| H[" HITL — Student Reviews and Edits Text "]
    H --> G

    G --> I[" Send to LangGraph Pipeline "]
```

---

### LangGraph Multi-Agent Pipeline — 6 Agents

```mermaid
graph TD
    START[" Raw Math Text "] --> AG0

    AG0[" Agent 0 — GUARDRAIL\nValidates input is a math problem\nBlocks non-math queries\nModel: gpt-4o-mini "]
    --> AG1

    AG1[" Agent 1 — PARSER\nCleans text and extracts equation\nIdentifies topic and detects ambiguity\nModel: gpt-4o-mini "]
    --> CHECK{" Needs Clarification? "}

    CHECK -->|" No "| AG2
    CHECK -->|" Yes "| HITL[" HITL — Ask Student for Clarification "]
    HITL --> AG2

    AG2[" Agent 2 — INTENT ROUTER\nKeyword + regex topic detection\nPicks solve strategy\nGenerates RAG search queries\nModel: gpt-4o-mini "]
    --> AG3

    AG3[" Agent 3 — SOLVER\nUses RAG context + memory\nSymPy symbolic calculator\nProduces step-by-step solution\nModel: gpt-4o "]
    --> AG4

    AG4[" Agent 4 — VERIFIER\nDual-run SymPy verification\nConfidence scoring 0 to 100 percent\nTriggers HITL if uncertain\nModel: gpt-4o "]
    --> AG5

    AG5[" Agent 5 — EXPLAINER\nStudent-friendly explanation\nKey concepts and tips and common mistakes\nDifficulty rating\nModel: gpt-4o "]
    --> DONE[" Final Output "]

    AG2 -.->|" search queries "| RAG[" FAISS Vector Store "]
    RAG -.->|" relevant context "| AG3

    AG3 -.->|" find similar problems "| MEM[" SQLite Memory "]
    MEM -.->|" past solutions and corrections "| AG3
```

---

### RAG Pipeline and Memory System

```mermaid
graph LR
    subgraph RAG [" RAG Pipeline "]
        direction TB
        KB[" Knowledge Base — 6 Markdown docs\nAlgebra Formulas\nCalculus Formulas\nProbability Formulas\nLinear Algebra Formulas\nCommon Mistakes\nSolution Templates "]
        --> SPLIT[" LangChain Text Splitter\n500 chars — 50 overlap "]
        --> EMBED[" OpenAI Embeddings\ntext-embedding-3-small "]
        --> FAISS[" FAISS Vector Store\nTop-5 retrieval\nSimilarity search "]
    end

    subgraph MEMORY [" Memory and Self-Learning "]
        direction TB
        DB[" SQLite Database\nStores every solved problem\nwith full pipeline state "]
        --> SIM[" Similar Problem Retrieval\nCosine similarity on embeddings "]
        DB --> CORR[" Correction Patterns\nLearned from student feedback over time "]
    end

    FAISS -->|" relevant formulas and templates "| SOLVER[" Solver Agent "]
    SIM -->|" past solutions "| SOLVER
    CORR -->|" correction hints "| SOLVER
```

---

### Output and Feedback Loop

```mermaid
graph TD
    EXP[" Explainer Agent "] ==> OUT[" Final Output "]

    OUT --> A1[" Final Answer — formatted with math symbols like x squared and square root "]
    OUT --> A2[" Step-by-Step Solution — detailed walkthrough "]
    OUT --> A3[" Verification Details — SymPy check + confidence percent "]
    OUT --> A4[" Key Concepts and Tips and Common Mistakes "]
    OUT --> A5[" Difficulty Rating — Easy or Medium or Hard "]
    OUT --> A6[" Agent Trace — full pipeline visibility "]

    A1 --> FB{" Student Feedback "}
    A2 --> FB
    A3 --> FB
    A4 --> FB
    A5 --> FB
    A6 --> FB

    FB -->|" Correct "| STORE[" Store solution in SQLite Memory "]
    FB -->|" Incorrect + Student Correction "| LEARN[" Store correction pattern for learning "]

    STORE --> LOOP[" Improves future problem solving "]
    LEARN --> LOOP
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
- **6 LangGraph Agents**: Guardrail, Parser, Intent Router, Solver, Verifier, Explainer
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
├── agents.py               # LangGraph multi-agent pipeline (6 agents)
├── rag_pipeline.py         # LangChain + FAISS RAG
├── memory_layer.py         # SQLite memory store
├── input_handlers.py       # Mistral OCR + EasyOCR + GPT-4o Vision + Whisper API
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
2. **Extraction** - Mistral OCR (images) or Whisper API (audio) extracts text; user can edit (HITL)
3. **LangGraph Pipeline** runs 6 agents sequentially with typed state:
   - **Guardrail** - Validates input is a math problem
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
