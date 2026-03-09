"""Math Mentor - Streamlit Application.

Multimodal AI math tutor using GPT-4o, LangGraph agents, LangChain RAG,
HITL verification, and SQLite-backed memory.
"""

import re
import streamlit as st
import os

import config
from agents import run_pipeline
from input_handlers import extract_text_from_image, transcribe_audio, save_uploaded_file, create_sample_math_image
from memory_layer import store_problem, update_feedback, get_all_memories, retrieve_similar


def _format_math_answer(text: str) -> str:
    """Convert SymPy notation to clean math display.

    Examples:
        2*x**2 + x  →  2x² + x
        x**3 - 4*x  →  x³ - 4x
        sqrt(2)/2    →  √2/2
    """
    if not text:
        return text
    result = str(text)

    # Superscript map for common exponents
    _sup = {"0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
            "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}

    # Replace **n with superscript digits (handles multi-digit like **12)
    def _to_super(m):
        return "".join(_sup.get(c, c) for c in m.group(1))
    result = re.sub(r'\*\*(\d+)', _to_super, result)

    # Remove multiplication signs between number and variable: 2*x → 2x
    result = re.sub(r'(\d)\*([a-zA-Z])', r'\1\2', result)
    # Remove multiplication signs between variable and variable: x*y → xy
    result = re.sub(r'([a-zA-Z])\*([a-zA-Z])', r'\1\2', result)
    # Remove multiplication signs between closing paren and variable: )*x → )x
    result = re.sub(r'\)\*([a-zA-Z])', r')\1', result)

    # sqrt() → √()
    result = result.replace('sqrt(', '√(')

    return result


# ── Page Config ──────────────────────────────────────────────

st.set_page_config(
    page_title="Math Mentor - AI Math Tutor",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Global ── */
    .block-container { padding-top: 1rem; max-width: 1120px; }

    /* ── Header ── */
    .app-header {
        text-align: center; padding: 1.2rem 0 0.4rem;
    }
    .app-header h1 {
        font-size: 2.6rem; font-weight: 900; margin: 0; letter-spacing: -1px;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .app-header .tagline {
        opacity: 0.5; font-size: 0.78rem; margin: 0.25rem 0 0;
        letter-spacing: 0.6px; font-weight: 500;
    }

    /* ── Answer highlight ── */
    .answer-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.10), rgba(52, 211, 153, 0.06));
        border: 2px solid #10b981; border-radius: 18px;
        padding: 1.6rem 1.2rem; text-align: center; margin: 0.8rem 0 1.2rem;
        box-shadow: 0 6px 32px rgba(16, 185, 129, 0.10);
        position: relative; overflow: hidden;
    }
    .answer-box::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
    }
    .answer-box .answer-label {
        font-size: 0.68rem; font-weight: 700; color: #10b981;
        text-transform: uppercase; letter-spacing: 2.5px; margin-bottom: 8px;
    }
    .answer-box .answer-text {
        font-size: 2rem; font-weight: 800;
        font-family: 'Georgia', 'Times New Roman', serif;
    }

    /* ── Agent trace ── */
    .trace-step {
        display: flex; align-items: center; gap: 0.5rem;
        padding: 8px 12px; margin: 4px 0; border-radius: 10px;
        background: rgba(34, 197, 94, 0.08);
        border-left: 3px solid #22c55e;
        font-size: 0.83rem; transition: transform 0.15s;
    }
    .trace-step:hover { transform: translateX(3px); }
    .trace-step.pending {
        background: rgba(234, 179, 8, 0.1);
        border-left-color: #eab308;
    }

    /* ── Confidence badge ── */
    .conf-badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.3px;
    }
    .conf-high   { background: rgba(16, 185, 129, 0.18); color: #34d399; }
    .conf-medium { background: rgba(245, 158, 11, 0.18); color: #fbbf24; }
    .conf-low    { background: rgba(239, 68, 68, 0.18);  color: #f87171; }

    /* ── Source chips ── */
    .source-chip {
        display: inline-block; border-radius: 20px;
        padding: 4px 14px; margin: 3px; font-size: 0.78rem;
        background: rgba(139, 92, 246, 0.15);
        color: #a78bfa; font-weight: 600;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }

    /* ── Verification checks ── */
    .verif-item {
        padding: 6px 0; font-size: 0.83rem; line-height: 1.5;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }
    .verif-item:last-child { border-bottom: none; }

    /* ── Sidebar topic grid ── */
    .topic-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
        margin-bottom: 4px;
    }
    .topic-card {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.12); border-radius: 12px;
        padding: 12px 10px; text-align: center;
        transition: transform 0.15s, border-color 0.15s;
    }
    .topic-card:hover {
        transform: translateY(-1px);
        border-color: rgba(139, 92, 246, 0.4);
    }
    .topic-card .topic-icon { font-size: 1.3rem; margin-bottom: 2px; }
    .topic-card .topic-name {
        font-weight: 700; font-size: 0.78rem; margin: 2px 0 0;
    }
    .topic-card .topic-desc {
        font-size: 0.68rem; opacity: 0.5;
    }

    /* ── Sidebar stat cards ── */
    .stat-row {
        display: flex; gap: 8px; margin-bottom: 4px;
    }
    .stat-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.10), rgba(192, 132, 252, 0.06));
        border-radius: 14px; padding: 14px 10px; text-align: center;
        border: 1px solid rgba(139, 92, 246, 0.20);
        transition: transform 0.15s;
    }
    .stat-card:hover { transform: translateY(-1px); }
    .stat-card .stat-num {
        font-size: 1.7rem; font-weight: 800; color: #a78bfa;
        line-height: 1.1;
    }
    .stat-card .stat-label {
        font-size: 0.68rem; color: #a78bfa; text-transform: uppercase;
        letter-spacing: 1.2px; font-weight: 600; opacity: 0.75;
        margin-top: 2px;
    }

    /* ── Learning section cards ── */
    .learn-card {
        background: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px; padding: 1rem; height: 100%;
    }
    .learn-card h4 {
        font-size: 0.85rem; font-weight: 700; opacity: 0.8;
        margin: 0 0 0.5rem; padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(128, 128, 128, 0.15);
    }
    .learn-card ul { color: inherit; }
    .learn-card.concepts { border-top: 3px solid #818cf8; }
    .learn-card.tips     { border-top: 3px solid #fbbf24; }
    .learn-card.mistakes { border-top: 3px solid #f87171; }

    /* ── Input tabs styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: rgba(128, 128, 128, 0.06);
        border-radius: 14px; padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 8px 24px; font-weight: 600;
        font-size: 0.88rem;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(139, 92, 246, 0.15);
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }

    /* ── Input section header ── */
    .input-header {
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 1.5px; opacity: 0.45; margin-bottom: 0.6rem;
    }

    /* ── Sidebar section labels ── */
    .sidebar-label {
        font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1.5px; opacity: 0.45; margin: 0.8rem 0 0.5rem;
    }

    /* ── Sidebar polish ── */
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.02);
    }
    section[data-testid="stSidebar"] .stMetric label { font-size: 0.78rem; }
    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }

    /* ── Progress bar color override ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #818cf8, #c084fc, #34d399);
    }

    /* ── Primary button override ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: #fff; border: none; font-weight: 700; letter-spacing: 0.3px;
        border-radius: 10px; padding: 0.55rem 1.2rem;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.35);
    }

    /* ── Secondary button override ── */
    .stButton > button[kind="secondary"] {
        border-radius: 10px; font-weight: 600;
        transition: transform 0.15s;
    }
    .stButton > button[kind="secondary"]:hover {
        transform: translateY(-1px);
    }

    /* ── Expander styling ── */
    .streamlit-expanderHeader {
        font-weight: 600; font-size: 0.88rem;
    }

    /* ── Feedback section ── */
    .feedback-section {
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 14px; padding: 1rem 1.2rem; margin-top: 0.5rem;
    }
    .feedback-section .feedback-q {
        font-size: 0.85rem; font-weight: 600; margin-bottom: 0.6rem;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center; font-size: 0.72rem; padding: 2rem 0 1rem;
        opacity: 0.35; letter-spacing: 0.3px;
    }

    /* ── Hide default streamlit menu & footer ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────

for key, default in {
    "result": None,
    "extracted_text": "",
    "extraction_confidence": 1.0,
    "current_problem_id": None,
    "hitl_active": False,
    "feedback_given": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Header ───────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1>Math Mentor</h1>
    <p class="tagline">Mistral OCR  &bull;  SymPy Solver  &bull;  GPT-4o  &bull;  LangGraph  &bull;  RAG  &bull;  Memory</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-label">Topics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="topic-grid">'
        '<div class="topic-card"><div class="topic-icon">📊</div>'
        '<div class="topic-name">Algebra</div>'
        '<div class="topic-desc">Equations, identities</div></div>'
        '<div class="topic-card"><div class="topic-icon">📈</div>'
        '<div class="topic-name">Calculus</div>'
        '<div class="topic-desc">Limits, derivatives</div></div>'
        '<div class="topic-card"><div class="topic-icon">🎲</div>'
        '<div class="topic-name">Probability</div>'
        '<div class="topic-desc">Bayes, distributions</div></div>'
        '<div class="topic-card"><div class="topic-icon">🔢</div>'
        '<div class="topic-name">Linear Algebra</div>'
        '<div class="topic-desc">Matrices, determinants</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">Memory</div>', unsafe_allow_html=True)
    memories = get_all_memories()
    correct_count = sum(1 for m in memories if m.get("user_feedback") == "correct")
    accuracy = f"{correct_count / len(memories):.0%}" if memories else "—"
    st.markdown(
        '<div class="stat-row">'
        f'<div class="stat-card"><div class="stat-num">{len(memories)}</div>'
        f'<div class="stat-label">Solved</div></div>'
        f'<div class="stat-card"><div class="stat-num">{correct_count}</div>'
        f'<div class="stat-label">Correct</div></div>'
        f'<div class="stat-card"><div class="stat-num">{accuracy}</div>'
        f'<div class="stat-label">Accuracy</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")  # spacer
    if st.button("Clear Memory", use_container_width=True, type="secondary"):
        from memory_layer import clear_memory
        clear_memory()
        st.success("Memory cleared!")
        st.rerun()

    st.markdown('<div class="sidebar-label">Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.75rem;opacity:0.6;line-height:1.7;">'
        'Guardrail &rarr; Parser &rarr; Router<br>'
        '&rarr; SymPy Solver &rarr; Verifier<br>'
        '&rarr; Explainer'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Input Section ────────────────────────────────────────────

st.markdown('<div class="input-header">Enter your problem</div>', unsafe_allow_html=True)

raw_text = ""
input_type = "text"

tab_text, tab_image, tab_audio = st.tabs(["  Type  ", "  Image  ", "  Audio  "])

# ── Text Tab ──
with tab_text:
    input_type_text = "text"
    raw_text_input = st.text_area(
        "Enter your math problem:",
        height=100,
        placeholder="e.g., Find the roots of x^2 - 5x + 6 = 0",
        label_visibility="collapsed",
    )
    if raw_text_input:
        raw_text = raw_text_input
        input_type = "text"
        st.session_state.extracted_text = raw_text
        st.session_state.extraction_confidence = 1.0

# ── Image Tab ──
with tab_image:
    img_cols = st.columns([3, 1])
    with img_cols[0]:
        uploaded_image = st.file_uploader(
            "Upload a photo/screenshot of the math problem",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    with img_cols[1]:
        st.caption("")  # spacer
        use_sample = st.button("Use sample image", use_container_width=True)

    if use_sample:
        sample_path = create_sample_math_image(config.UPLOADS_DIR)
        st.session_state["_sample_image_path"] = sample_path
        st.session_state.extracted_text = ""

    # Determine which image to use
    _image_path = None
    _image_display = None
    if uploaded_image:
        _image_path = save_uploaded_file(uploaded_image, config.UPLOADS_DIR)
        _image_display = uploaded_image
    elif st.session_state.get("_sample_image_path"):
        _image_path = st.session_state["_sample_image_path"]
        _image_display = _image_path

    if _image_path:
        col_img, col_ocr = st.columns([1, 1])
        with col_img:
            st.image(_image_display, use_container_width=True)

        with col_ocr:
            extract_btn = st.button("Extract Text", type="primary", use_container_width=True)

            if extract_btn or st.session_state.extracted_text:
                if not st.session_state.extracted_text or extract_btn:
                    with st.spinner("Running OCR pipeline (Mistral → EasyOCR → GPT-4o)..."):
                        extracted, confidence = extract_text_from_image(_image_path)
                        st.session_state.extracted_text = extracted
                        st.session_state.extraction_confidence = confidence

                conf = st.session_state.extraction_confidence
                if conf >= 0.8:
                    conf_class = "conf-high"
                elif conf >= 0.5:
                    conf_class = "conf-medium"
                else:
                    conf_class = "conf-low"
                st.markdown(
                    f'<span class="conf-badge {conf_class}">OCR Confidence: {conf:.0%}</span>',
                    unsafe_allow_html=True,
                )
                if conf < 0.5:
                    st.warning("Low confidence — please review the text below.")

                raw_text = st.text_area(
                    "Extracted text (edit if needed):",
                    value=st.session_state.extracted_text,
                    height=80,
                )
                st.session_state.extracted_text = raw_text
                input_type = "image"

# ── Audio Tab ──
with tab_audio:
    # Microphone recording
    try:
        from audio_recorder_streamlit import audio_recorder

        st.caption("Click the mic to record, click again to stop.")
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e74c3c",
            neutral_color="#2c3e50",
            icon_name="microphone",
            icon_size="2x",
            pause_threshold=3.0,
            energy_threshold=0.01,
        )

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            os.makedirs(config.UPLOADS_DIR, exist_ok=True)
            mic_path = os.path.join(config.UPLOADS_DIR, "mic_recording.wav")
            with open(mic_path, "wb") as f:
                f.write(audio_bytes)
            st.session_state["_mic_audio_path"] = mic_path
            st.session_state.extracted_text = ""
            st.session_state.extraction_confidence = 1.0

        if st.session_state.get("_mic_audio_path") and audio_bytes:
            with st.spinner("Transcribing..."):
                transcript, confidence = transcribe_audio(st.session_state["_mic_audio_path"])
                st.session_state.extracted_text = transcript
                st.session_state.extraction_confidence = confidence

            conf = st.session_state.extraction_confidence
            conf_class = "conf-high" if conf >= 0.7 else "conf-medium"
            st.markdown(
                f'<span class="conf-badge {conf_class}">Transcription Confidence: {conf:.0%}</span>',
                unsafe_allow_html=True,
            )
            raw_text = st.text_area(
                "Transcript (edit if needed):",
                value=st.session_state.extracted_text,
                height=80,
                key="mic_transcript_edit",
            )
            st.session_state.extracted_text = raw_text
            input_type = "audio"

        st.divider()
        st.caption("Or upload an audio file:")
    except ImportError:
        st.info("Install `audio-recorder-streamlit` for mic recording.")

    uploaded_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg"],
                                      label_visibility="collapsed")
    if uploaded_audio:
        st.audio(uploaded_audio)
        audio_path = save_uploaded_file(uploaded_audio, config.UPLOADS_DIR)

        if st.button("Transcribe", type="primary") or st.session_state.extracted_text:
            if not st.session_state.extracted_text:
                with st.spinner("Transcribing..."):
                    transcript, confidence = transcribe_audio(audio_path)
                    st.session_state.extracted_text = transcript
                    st.session_state.extraction_confidence = confidence

            conf = st.session_state.extraction_confidence
            if conf >= 0.7:
                conf_class = "conf-high"
            elif conf >= 0.4:
                conf_class = "conf-medium"
            else:
                conf_class = "conf-low"
            st.markdown(
                f'<span class="conf-badge {conf_class}">Confidence: {conf:.0%}</span>',
                unsafe_allow_html=True,
            )
            if conf < 0.4:
                st.warning("Low confidence — please review.")

            raw_text = st.text_area(
                "Transcript (edit if needed):",
                value=st.session_state.extracted_text,
                height=80,
            )
            st.session_state.extracted_text = raw_text
            input_type = "audio"


# ── Solve ────────────────────────────────────────────────────

final_text = raw_text or st.session_state.extracted_text

if final_text:
    # Similar problems hint
    similar = retrieve_similar(final_text)
    if similar:
        with st.expander(f"Similar problems from memory ({len(similar)} found)", expanded=False):
            for sp in similar:
                st.markdown(f"**Q:** {sp.get('parsed_question', 'N/A')[:120]}")
                st.markdown(f"**A:** {sp.get('solution', 'N/A')[:120]}")
                fb = sp.get('user_feedback', 'none')
                if fb == "correct":
                    st.markdown("Feedback: ✅ Correct")
                elif fb == "incorrect":
                    st.markdown("Feedback: ❌ Incorrect")
                st.divider()

    if st.button("Solve", type="primary", use_container_width=True):
        st.session_state.result = None
        st.session_state.feedback_given = False
        st.session_state.hitl_active = False

        with st.spinner("Running LangGraph multi-agent pipeline..."):
            result = run_pipeline(final_text, input_type)
            st.session_state.result = result

            if result.get("status") in ("solved", "needs_human_review"):
                problem_id = store_problem({
                    "input_type": input_type,
                    "original_input": final_text,
                    "parsed_question": result["parsed"].get("problem_text", ""),
                    "topic": result["parsed"].get("topic", ""),
                    "retrieved_context": [
                        s.get("source", "") for s in result["solution"].get("retrieved_sources", [])
                    ],
                    "solution": result["solution"].get("solution", ""),
                    "explanation": result["explanation"].get("explanation", ""),
                    "verifier_result": result["verification"],
                    "user_feedback": "",
                })
                st.session_state.current_problem_id = problem_id


# ── Display Results ──────────────────────────────────────────

result = st.session_state.result

if result:
    st.markdown("---")

    if result.get("status") == "blocked":
        guardrail = result.get("guardrail", {})
        st.error(f"Input blocked: {guardrail.get('rejection_reason', 'Not a valid math problem.')}")
        st.info("Please enter a valid math problem (algebra, probability, calculus, or linear algebra).")

    elif result.get("status") == "needs_clarification":
        parsed = result.get("parsed", {})
        st.warning(f"Clarification needed: {parsed.get('clarification_reason', 'Ambiguous input')}")
        st.info("Please edit your input above and try again.")

    elif result.get("status") in ("solved", "needs_human_review"):

        # ── Answer Box (prominent) ──
        raw_answer = result['solution'].get('final_answer', 'N/A')
        formatted_answer = _format_math_answer(raw_answer)

        corrected = result["solution"].get("corrected_problem", "")
        original = result["parsed"].get("original_text", "")

        st.markdown(
            f'<div class="answer-box">'
            f'<div class="answer-label">ANSWER</div>'
            f'<div class="answer-text">{formatted_answer}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if corrected and corrected != original:
            st.caption(f"Interpreted as: *{corrected}*")

        # ── Corrections banner ──
        corrections = result["parsed"].get("corrections_applied", [])
        if corrections:
            with st.expander("Input Corrections Applied"):
                for c in corrections:
                    st.markdown(
                        f"- `{c.get('original', '')}` → `{c.get('corrected', '')}` "
                        f"*({c.get('reason', '')})*"
                    )

        # ── Main content: two columns ──
        col_main, col_side = st.columns([5, 3])

        with col_main:
            # Step-by-step explanation
            st.markdown("#### Step-by-Step Solution")
            st.markdown(result["explanation"].get("explanation", ""))

            # Key concepts, tips, mistakes — in compact expanders
            key_concepts = result["explanation"].get("key_concepts", [])
            tips = result["explanation"].get("tips", [])
            common_mistakes = result["explanation"].get("common_mistakes", [])

            if key_concepts or tips or common_mistakes:
                learn_cols = st.columns(3 if (key_concepts and tips and common_mistakes) else
                                        2 if sum(bool(x) for x in [key_concepts, tips, common_mistakes]) == 2 else 1)
                col_idx = 0
                if key_concepts:
                    with learn_cols[col_idx]:
                        items_html = "".join(f"<li>{c}</li>" for c in key_concepts)
                        st.markdown(
                            f'<div class="learn-card concepts"><h4>Key Concepts</h4>'
                            f'<ul style="padding-left:1.2rem;margin:0;font-size:0.85rem;">{items_html}</ul></div>',
                            unsafe_allow_html=True,
                        )
                    col_idx += 1
                if tips:
                    with learn_cols[col_idx]:
                        items_html = "".join(f"<li>{t}</li>" for t in tips)
                        st.markdown(
                            f'<div class="learn-card tips"><h4>Tips</h4>'
                            f'<ul style="padding-left:1.2rem;margin:0;font-size:0.85rem;">{items_html}</ul></div>',
                            unsafe_allow_html=True,
                        )
                    col_idx += 1
                if common_mistakes:
                    with learn_cols[col_idx]:
                        items_html = "".join(f"<li>{m}</li>" for m in common_mistakes)
                        st.markdown(
                            f'<div class="learn-card mistakes"><h4>Common Mistakes</h4>'
                            f'<ul style="padding-left:1.2rem;margin:0;font-size:0.85rem;">{items_html}</ul></div>',
                            unsafe_allow_html=True,
                        )

        with col_side:
            # ── Confidence + Verification combined ──
            solver_conf = result["solution"].get("confidence", 0)
            verifier_conf = result["verification"].get("confidence", 0)
            try:
                solver_conf = float(solver_conf)
            except (ValueError, TypeError):
                solver_conf = 0.5
            try:
                verifier_conf = float(verifier_conf)
            except (ValueError, TypeError):
                verifier_conf = 0.5

            verif_steps_list = result["verification"].get("verification_steps", [])
            all_verif_passed = (
                verif_steps_list
                and all(v.get("result") == "pass" for v in verif_steps_list)
            )
            if all_verif_passed and verifier_conf >= 0.9:
                avg_conf = verifier_conf * 0.8 + solver_conf * 0.2
            else:
                avg_conf = (solver_conf + verifier_conf) / 2
            avg_conf = min(avg_conf, 1.0)

            # Verification status
            verification = result["verification"]
            is_correct = verification.get("correct", verification.get("is_correct"))

            if is_correct:
                st.success(f"Verified Correct — {avg_conf:.0%} confidence")
            else:
                error_type = verification.get("error_type", "")
                st.error(f"Incorrect — {error_type or 'check below'}")
                correct_answer = verification.get("correct_answer", "")
                if correct_answer:
                    st.info(f"Corrected: **{_format_math_answer(correct_answer)}**")

            st.progress(avg_conf)

            # Verification steps
            verif_steps = verification.get("verification_steps", [])
            if verif_steps:
                with st.expander("Verification Details", expanded=False):
                    for v in verif_steps:
                        icon = "✅" if v.get("result") == "pass" else "❌"
                        st.markdown(
                            f'<div class="verif-item">{icon} <strong>{v.get("check", "")}</strong>: '
                            f'{v.get("detail", "")}</div>',
                            unsafe_allow_html=True,
                        )

            for issue in verification.get("issues", []):
                st.warning(issue)
            if verification.get("needs_human_review"):
                st.warning(f"Human review recommended: {verification.get('review_reason', '')}")
                st.session_state.hitl_active = True

            # ── SymPy Self-Consistency ──
            sympy_cons = result["solution"].get("sympy_consistency", {})
            if sympy_cons:
                if sympy_cons.get("consistent"):
                    st.markdown("**SymPy:** ✅ Self-consistent (2 runs agree)")
                else:
                    st.markdown("**SymPy:** ⚠️ Inconsistent")
                    st.caption(f"Run A: {sympy_cons.get('result_a', 'N/A')[:80]}")
                    st.caption(f"Run B: {sympy_cons.get('result_b', 'N/A')[:80]}")

            # ── Difficulty ──
            diff = result["explanation"].get("difficulty_rating", "Medium")
            diff_icons = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
            st.markdown(f"**Difficulty:** {diff_icons.get(diff, '🟡')} {diff}")

            # ── Agent Trace ──
            with st.expander("Agent Trace", expanded=False):
                for step in result.get("trace", []):
                    icon = "✅" if step["status"] == "completed" else "⏳"
                    status_class = "" if step["status"] == "completed" else " pending"
                    st.markdown(
                        f'<div class="trace-step{status_class}">'
                        f'{icon} <strong>{step["agent"]}</strong> — '
                        f'{step.get("output_summary", "")}</div>',
                        unsafe_allow_html=True,
                    )

            # ── Retrieved Sources ──
            sources = result["solution"].get("retrieved_sources", [])
            if sources:
                with st.expander("Knowledge Sources", expanded=False):
                    for src in sources:
                        st.markdown(
                            f'<span class="source-chip">{src["source"]} ({src["score"]:.2f})</span>',
                            unsafe_allow_html=True,
                        )

        # ── Raw details (collapsed) ──
        with st.expander("Raw Details (Parsed, Solution, Verification, Routing)"):
            detail_tabs = st.tabs(["Parsed", "Solution", "Verification", "Routing"])
            with detail_tabs[0]:
                st.json(result["parsed"])
            with detail_tabs[1]:
                st.json(result["solution"])
            with detail_tabs[2]:
                st.json(result["verification"])
            with detail_tabs[3]:
                st.json(result["route"])

        # ── Feedback (HITL) ──
        st.markdown(
            '<div class="feedback-section"><div class="feedback-q">Was this answer correct?</div></div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.feedback_given:
            fc = st.columns([1, 1, 4])
            with fc[0]:
                if st.button("Correct", use_container_width=True, type="primary"):
                    if st.session_state.current_problem_id:
                        update_feedback(st.session_state.current_problem_id, "correct")
                    st.session_state.feedback_given = True
                    st.rerun()
            with fc[1]:
                if st.button("Wrong", use_container_width=True):
                    st.session_state.hitl_active = True

            if st.session_state.hitl_active:
                correction = st.text_area("What was wrong? Provide the correct answer:",
                                          placeholder="e.g., The answer should be x = 3, not x = 2")
                if st.button("Submit Correction", type="primary"):
                    if st.session_state.current_problem_id and correction:
                        update_feedback(st.session_state.current_problem_id, "incorrect", correction)
                        st.session_state.feedback_given = True
                        st.success("Correction saved — the system will learn from this.")
                        st.rerun()
        else:
            st.success("Thanks for the feedback!")

    # ── New Problem button ──
    st.markdown("")
    if st.button("Start New Problem", use_container_width=True):
        for key in ("result", "extracted_text", "extraction_confidence",
                     "current_problem_id", "hitl_active", "feedback_given"):
            st.session_state[key] = {"result": None, "extracted_text": "",
                                      "extraction_confidence": 1.0, "current_problem_id": None,
                                      "hitl_active": False, "feedback_given": False}[key]
        st.rerun()


# ── Footer ───────────────────────────────────────────────────

st.markdown(
    '<div class="app-footer">'
    'Math Mentor v3.2 &mdash; Mistral OCR &bull; SymPy &bull; GPT-4o &bull; LangGraph &bull; RAG &bull; HITL'
    '</div>',
    unsafe_allow_html=True,
)
