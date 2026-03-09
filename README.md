# Math Mentor - AI-Powered JEE Math Tutor

A multimodal AI application that solves JEE-style math problems using **GPT-4o**, **LangGraph** multi-agent orchestration, **LangChain + FAISS** RAG, human-in-the-loop verification, and **SQLite**-backed memory.

## Architecture

### High-Level System Flow

```mermaid
graph LR
    A["🎓 Student"] ==> B["🖥️ Streamlit UI"]
    B ==> C["📥 Input Layer\n(Text / Image / Audio)"]
    C ==> D["🤖 LangGraph Pipeline\n(6 AI Agents)"]
    D <-.-> E["📚 RAG\n(FAISS + LangChain)"]
    D <-.-> F["🧠 Memory\n(SQLite + Embeddings)"]
    D ==> G["📊 Output\n(Answer + Explanation)"]
    G ==> H["🔄 Feedback Loop"]
    H ==> F

    style A fill:#6c5ce7,stroke:#a29bfe,color:#fff,stroke-width:2px
    style B fill:#0984e3,stroke:#74b9ff,color:#fff,stroke-width:2px
    style C fill:#00b894,stroke:#55efc4,color:#fff,stroke-width:2px
    style D fill:#e17055,stroke:#fab1a0,color:#fff,stroke-width:2px
    style E fill:#fdcb6e,stroke:#ffeaa7,color:#333,stroke-width:2px
    style F fill:#00cec9,stroke:#81ecec,color:#fff,stroke-width:2px
    style G fill:#6c5ce7,stroke:#a29bfe,color:#fff,stroke-width:2px
    style H fill:#d63031,stroke:#ff7675,color:#fff,stroke-width:2px
```

---

### 📥 Multimodal Input Layer

```mermaid
graph TD
    A["🎓 Student uploads\nproblem"] --> B{"Select Input Mode"}

    B -->|"✏️ Text"| C["Direct Text Input\n(passthrough)"]

    B -->|"📷 Image"| D["Mistral OCR\n(primary engine)"]
    D -->|"if fails"| D2["EasyOCR\n(fallback #1)"]
    D2 -->|"if fails"| D3["GPT-4o Vision\n(fallback #2)"]

    B -->|"🎤 Audio"| E["OpenAI Whisper API\n(gpt-4o-transcribe)"]

    D --> F{"Confidence\n≥ threshold?"}
    D2 --> F
    D3 --> F
    E --> F
    C --> G["✅ Raw Math Text"]
    F -->|"✅ High\nconfidence"| G
    F -->|"⚠️ Low\nconfidence"| H["👤 HITL:\nStudent Reviews\n& Edits Text"]
    H --> G

    G --> I["➡️ Send to\nLangGraph Pipeline"]

    style A fill:#6c5ce7,stroke:#a29bfe,color:#fff,stroke-width:2px
    style B fill:#0984e3,stroke:#74b9ff,color:#fff,stroke-width:2px
    style C fill:#00b894,stroke:#55efc4,color:#fff,stroke-width:2px
    style D fill:#fdcb6e,stroke:#ffeaa7,color:#333,stroke-width:2px
    style D2 fill:#fdcb6e,stroke:#ffeaa7,color:#333,stroke-width:2px
    style D3 fill:#fdcb6e,stroke:#ffeaa7,color:#333,stroke-width:2px
    style E fill:#00b894,stroke:#55efc4,color:#fff,stroke-width:2px
    style F fill:#e17055,stroke:#fab1a0,color:#fff,stroke-width:2px
    style G fill:#00cec9,stroke:#81ecec,color:#fff,stroke-width:2px
    style H fill:#d63031,stroke:#ff7675,color:#fff,stroke-width:2px
    style I fill:#6c5ce7,stroke:#a29bfe,color:#fff,stroke-width:2px
```

---

### 🤖 LangGraph Multi-Agent Pipeline (6 Agents)

```mermaid
graph TD
    START["📥 Raw Math Text"] --> AG0

    AG0["🛡️ Agent 0: GUARDRAIL\n─────────────────\nValidates input is a math problem\nBlocks non-math queries\n\n🏷️ Model: gpt-4o-mini"]
    --> AG1

    AG1["📝 Agent 1: PARSER\n─────────────────\nCleans text, extracts equation\nIdentifies topic & detects ambiguity\n\n🏷️ Model: gpt-4o-mini"]
    --> CHECK{"❓ Needs\nClarification?"}

    CHECK -->|"✅ No"| AG2
    CHECK -->|"⚠️ Yes"| HITL["👤 HITL:\nAsk Student\nfor Clarification"]
    HITL --> AG2

    AG2["🔀 Agent 2: INTENT ROUTER\n─────────────────\nKeyword + regex topic detection\nPicks solve strategy\nGenerates RAG search queries\n\n🏷️ Model: gpt-4o-mini"]
    --> AG3

    AG3["🧮 Agent 3: SOLVER\n─────────────────\nUses RAG context + memory\nSymPy symbolic calculator\nProduces step-by-step solution\n\n🏷️ Model: gpt-4o"]
    --> AG4

    AG4["✅ Agent 4: VERIFIER\n─────────────────\nDual-run SymPy verification\nConfidence scoring (0-100%)\nTriggers HITL if uncertain\n\n🏷️ Model: gpt-4o"]
    --> AG5

    AG5["📖 Agent 5: EXPLAINER\n─────────────────\nStudent-friendly explanation\nKey concepts, tips, common mistakes\nDifficulty rating\n\n🏷️ Model: gpt-4o"]
    --> DONE["📊 Final Output"]

    AG2 -.->|"search\nqueries"| RAG["📚 FAISS\nVector Store"]
    RAG -.->|"relevant\ncontext"| AG3

    AG3 -.->|"find similar\nproblems"| MEM["🧠 SQLite\nMemory"]
    MEM -.->|"past solutions\n& corrections"| AG3

    style START fill:#6c5ce7,stroke:#a29bfe,color:#fff,stroke-width:2px
    style AG0 fill:#636e72,stroke:#b2bec3,color:#fff,stroke-width:2px,text-align:left
    style AG1 fill:#0984e3,stroke:#74b9ff,color:#fff,stroke-width:2px
    style CHECK fill:#e17055,stroke:#fab1a0,color:#fff,stroke-width:2px
    style HITL fill:#d63031,stroke:#ff7675,color:#fff,stroke-width:2px
    style AG2 fill:#6c5ce7,stroke:#a29bfe,color:#fff,stroke-width:2px
    style AG3 fill:#00b894,stroke:#55efc4,color:#fff,stroke-width:2px
    style AG4 fill:#fdcb6e,stroke:#ffeaa7,color:#333,stroke-width:2px
    style AG5 fill:#e84393,stroke:#fd79a8,color:#fff,stroke-width:2px
    style DONE fill:#00cec9,stroke:#81ecec,color:#fff,stroke-width:2px
    style RAG fill:#fdcb6e,stroke:#ffeaa7,color:#333,stroke-width:2px
    style MEM fill:#00cec9,stroke:#81ecec,color:#fff,stroke-width:2px
```

---

### 📚 RAG Pipeline & 🧠 Memory System

```mermaid
graph LR
    subgraph RAG["📚 RAG Pipeline"]
        direction TB
        KB["📄 Knowledge Base\n(6 Markdown docs)\n─────────────\nAlgebra Formulas\nCalculus Formulas\nProbability Formulas\nLinear Algebra Formulas\nCommon Mistakes\nSolution Templates"]
        --> SPLIT["✂️ LangChain\nText Splitter\n─────────────\n500 chars\n50 overlap"]
        --> EMBED["🔢 OpenAI\nEmbeddings\n─────────────\ntext-embedding-\n3-small"]
        --> FAISS["🔍 FAISS\nVector Store\n─────────────\nTop-5 retrieval\nSimilarity search"]
    end

    subgraph MEMORY["🧠 Memory & Self-Learning"]
        direction TB
        DB["🗄️ SQLite Database\n─────────────\nStores every solved problem\nwith full pipeline state"]
        --> SIM["🔎 Similar Problem\nRetrieval\n─────────────\nCosine similarity\non embeddings"]
        DB --> CORR["📝 Correction\nPatterns\n─────────────\nLearned from student\nfeedback over time"]
    end

    FAISS -->|"relevant\nformulas &\ntemplates"| SOLVER["🧮 Solver Agent"]
    SIM -->|"past\nsolutions"| SOLVER
    CORR -->|"correction\nhints"| SOLVER

    style KB fill:#fdcb6e,stroke:#ffeaa7,color:#333,stroke-width:2px
    style SPLIT fill:#fdcb6e,stroke:#ffeaa7,color:#333,stroke-width:2px
    style EMBED fill:#fdcb6e,stroke:#ffeaa7,color:#333,stroke-width:2px
    style FAISS fill:#e17055,stroke:#fab1a0,color:#fff,stroke-width:2px
    style DB fill:#00cec9,stroke:#81ecec,color:#fff,stroke-width:2px
    style SIM fill:#00cec9,stroke:#81ecec,color:#fff,stroke-width:2px
    style CORR fill:#00cec9,stroke:#81ecec,color:#fff,stroke-width:2px
    style SOLVER fill:#00b894,stroke:#55efc4,color:#fff,stroke-width:2px
```

---

### 📊 Output & Feedback Loop

```mermaid
graph TD
    EXP["📖 Explainer Agent"] ==> OUT["📊 Final Output"]

    OUT --> A1["✨ Final Answer\n(formatted with math symbols\nlike x², √, π)"]
    OUT --> A2["📝 Step-by-Step Solution\n(detailed walkthrough)"]
    OUT --> A3["✅ Verification Details\n(SymPy check + confidence %)"]
    OUT --> A4["💡 Key Concepts · Tips\n· Common Mistakes"]
    OUT --> A5["📈 Difficulty Rating\n(Easy / Medium / Hard)"]
    OUT --> A6["🔍 Agent Trace\n(full pipeline visibility)"]

    A1 --> FB{"🎓 Student Feedback"}
    A2 --> FB
    A3 --> FB
    A4 --> FB
    A5 --> FB
    A6 --> FB

    FB -->|"✅ Correct"| STORE["🧠 Store solution\nin SQLite Memory"]
    FB -->|"❌ Incorrect +\nStudent Correction"| LEARN["🧠 Store correction\npattern for learning"]

    STORE --> LOOP["🔄 Improves future\nproblem solving"]
    LEARN --> LOOP

    style EXP fill:#e84393,stroke:#fd79a8,color:#fff,stroke-width:2px
    style OUT fill:#6c5ce7,stroke:#a29bfe,color:#fff,stroke-width:2px
    style A1 fill:#0984e3,stroke:#74b9ff,color:#fff,stroke-width:2px
    style A2 fill:#0984e3,stroke:#74b9ff,color:#fff,stroke-width:2px
    style A3 fill:#0984e3,stroke:#74b9ff,color:#fff,stroke-width:2px
    style A4 fill:#0984e3,stroke:#74b9ff,color:#fff,stroke-width:2px
    style A5 fill:#0984e3,stroke:#74b9ff,color:#fff,stroke-width:2px
    style A6 fill:#0984e3,stroke:#74b9ff,color:#fff,stroke-width:2px
    style FB fill:#e17055,stroke:#fab1a0,color:#fff,stroke-width:2px
    style STORE fill:#00b894,stroke:#55efc4,color:#fff,stroke-width:2px
    style LEARN fill:#00b894,stroke:#55efc4,color:#fff,stroke-width:2px
    style LOOP fill:#00cec9,stroke:#81ecec,color:#fff,stroke-width:2px
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
