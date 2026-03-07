"""Math Mentor - Streamlit Application.

Multimodal AI math tutor using GPT-4o, LangGraph agents, LangChain RAG,
HITL verification, and SQLite-backed memory.
"""

import streamlit as st
import os

import config
from agents import run_pipeline
from input_handlers import extract_text_from_image, transcribe_audio, save_uploaded_file, create_sample_math_image
from memory_layer import store_problem, update_feedback, get_all_memories, retrieve_similar


# ── Page Config ──────────────────────────────────────────────

st.set_page_config(
    page_title="Math Mentor - AI Math Tutor",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: .5rem; }
    .sub-header  { text-align: center; color: #666; margin-bottom: 2rem; }
    .agent-step  { padding: 8px 12px; margin: 4px 0; border-radius: 5px; border-left: 4px solid; }
    .agent-completed { border-left-color: #28a745; background-color: #d4edda; }
    .confidence-high   { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low    { color: #dc3545; font-weight: bold; }
    .source-chip { display: inline-block; background: #e9ecef; border-radius: 15px;
                   padding: 4px 12px; margin: 2px; font-size: .85rem; }
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

st.markdown('<div class="main-header">📐 Math Mentor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">GPT-4o + LangGraph Agents + LangChain RAG + HITL + Memory</div>', unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input("OpenAI API Key", type="password", value=config.OPENAI_API_KEY)
    if api_key:
        config.OPENAI_API_KEY = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.header("Supported Topics")
    st.markdown("""
    - **Algebra** (equations, identities, sequences)
    - **Probability** (counting, distributions, Bayes)
    - **Calculus** (limits, derivatives, optimization)
    - **Linear Algebra** (matrices, determinants, vectors)
    """)

    st.divider()
    st.header("Memory Store")
    memories = get_all_memories()
    st.metric("Problems Solved", len(memories))
    correct_count = sum(1 for m in memories if m.get("user_feedback") == "correct")
    st.metric("Confirmed Correct", correct_count)

    if st.button("Clear Memory"):
        from memory_layer import clear_memory
        clear_memory()
        st.success("Memory cleared!")
        st.rerun()


# ── Input Section ────────────────────────────────────────────

st.header("Input Your Math Problem")

input_mode = st.radio("Select input mode:", ["Text", "Image", "Audio"], horizontal=True)

raw_text = ""
input_type = "text"

if input_mode == "Text":
    input_type = "text"
    raw_text = st.text_area(
        "Type your math problem:",
        height=120,
        placeholder="e.g., Find the roots of x^2 - 5x + 6 = 0",
    )
    st.session_state.extracted_text = raw_text
    st.session_state.extraction_confidence = 1.0

elif input_mode == "Image":
    input_type = "image"

    # Sample image for quick testing
    if st.button("Use Sample Math Image (for testing)"):
        sample_path = create_sample_math_image(config.UPLOADS_DIR)
        st.session_state["_sample_image_path"] = sample_path
        st.session_state.extracted_text = ""

    uploaded_image = st.file_uploader(
        "Upload a photo/screenshot of the math problem", type=["jpg", "jpeg", "png"]
    )

    # Use sample image if no file uploaded
    if not uploaded_image and st.session_state.get("_sample_image_path"):
        image_path = st.session_state["_sample_image_path"]
        st.image(image_path, caption="Sample Math Image", use_container_width=True)
        if st.button("Extract Text from Sample (EasyOCR)") or st.session_state.extracted_text:
            if not st.session_state.extracted_text:
                with st.spinner("Running EasyOCR (+ GPT-4o refinement if low confidence)..."):
                    extracted, confidence = extract_text_from_image(image_path)
                    st.session_state.extracted_text = extracted
                    st.session_state.extraction_confidence = confidence
            conf = st.session_state.extraction_confidence
            if conf >= 0.8:
                st.markdown(f'<span class="confidence-high">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
            elif conf >= 0.5:
                st.markdown(f'<span class="confidence-medium">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="confidence-low">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
                st.warning("Low confidence! Please review and edit the extracted text.")
            raw_text = st.text_area("Edit extracted text if needed:", value=st.session_state.extracted_text, height=100)
            st.session_state.extracted_text = raw_text

    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        image_path = save_uploaded_file(uploaded_image, config.UPLOADS_DIR)

        if st.button("Extract Text (EasyOCR)") or st.session_state.extracted_text:
            if not st.session_state.extracted_text:
                with st.spinner("Running EasyOCR (+ GPT-4o refinement if low confidence)..."):
                    extracted, confidence = extract_text_from_image(image_path)
                    st.session_state.extracted_text = extracted
                    st.session_state.extraction_confidence = confidence

            with col2:
                st.subheader("Extracted Text")
                conf = st.session_state.extraction_confidence
                if conf >= 0.8:
                    st.markdown(f'<span class="confidence-high">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
                elif conf >= 0.5:
                    st.markdown(f'<span class="confidence-medium">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="confidence-low">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
                    st.warning("Low confidence! Please review and edit the extracted text.")

                raw_text = st.text_area(
                    "Edit extracted text if needed:",
                    value=st.session_state.extracted_text,
                    height=100,
                )
                st.session_state.extracted_text = raw_text

elif input_mode == "Audio":
    input_type = "audio"

    # ── Real-time microphone recording ──
    try:
        from audio_recorder_streamlit import audio_recorder

        st.markdown("**Record directly from your microphone:**")
        st.caption("Click the mic icon to start recording, click again to stop.")
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
            with st.spinner("Transcribing with gpt-4o-transcribe..."):
                transcript, confidence = transcribe_audio(st.session_state["_mic_audio_path"])
                st.session_state.extracted_text = transcript
                st.session_state.extraction_confidence = confidence

            conf = st.session_state.extraction_confidence
            st.subheader("Transcript")
            if conf >= 0.7:
                st.markdown(f'<span class="confidence-high">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="confidence-medium">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
            raw_text = st.text_area(
                "Edit transcript if needed:",
                value=st.session_state.extracted_text,
                height=100,
                key="mic_transcript_edit",
            )
            st.session_state.extracted_text = raw_text

        st.divider()
        st.markdown("**Or upload an audio file:**")
    except ImportError:
        st.info("Install `audio-recorder-streamlit` for real-time mic recording.")

    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])
    if uploaded_audio:
        st.audio(uploaded_audio)
        audio_path = save_uploaded_file(uploaded_audio, config.UPLOADS_DIR)

        if st.button("Transcribe (gpt-4o-transcribe)") or st.session_state.extracted_text:
            if not st.session_state.extracted_text:
                with st.spinner("Transcribing with gpt-4o-transcribe..."):
                    transcript, confidence = transcribe_audio(audio_path)
                    st.session_state.extracted_text = transcript
                    st.session_state.extraction_confidence = confidence

            conf = st.session_state.extraction_confidence
            st.subheader("Transcript")
            if conf >= 0.7:
                st.markdown(f'<span class="confidence-high">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
            elif conf >= 0.4:
                st.markdown(f'<span class="confidence-medium">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="confidence-low">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)
                st.warning("Low transcription confidence! Please review and edit.")

            raw_text = st.text_area(
                "Edit transcript if needed:",
                value=st.session_state.extracted_text,
                height=100,
            )
            st.session_state.extracted_text = raw_text


# ── Solve ────────────────────────────────────────────────────

final_text = raw_text or st.session_state.extracted_text

if final_text:
    similar = retrieve_similar(final_text)
    if similar:
        with st.expander(f"Similar problems from memory ({len(similar)} found)"):
            for sp in similar:
                st.markdown(f"**Q:** {sp.get('parsed_question', 'N/A')[:100]}...")
                st.markdown(f"**A:** {sp.get('solution', 'N/A')[:100]}...")
                st.markdown(f"**Feedback:** {sp.get('user_feedback', 'none')}")
                st.divider()

    if st.button("Solve Problem", type="primary", use_container_width=True):
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
    st.divider()

    if result.get("status") == "blocked":
        guardrail = result.get("guardrail", {})
        st.error(f"Input blocked by Guardrail Agent: {guardrail.get('rejection_reason', 'Not a valid math problem.')}")
        st.info("Please enter a valid math problem (algebra, probability, calculus, or linear algebra).")

    elif result.get("status") == "needs_clarification":
        parsed = result.get("parsed", {})
        st.warning(f"Clarification needed: {parsed.get('clarification_reason', 'Ambiguous input')}")
        st.info("Please edit your input above and try again.")

    elif result.get("status") in ("solved", "needs_human_review"):
        col_main, col_side = st.columns([2, 1])

        with col_main:
            # Show corrections if parser fixed OCR/transcription errors
            corrections = result["parsed"].get("corrections_applied", [])
            if corrections:
                with st.expander("Input Corrections Applied", expanded=True):
                    for c in corrections:
                        st.markdown(
                            f"- `{c.get('original', '')}` → `{c.get('corrected', '')}` "
                            f"*({c.get('reason', '')})*"
                        )

            st.subheader("Final Answer")
            st.success(f"**{result['solution'].get('final_answer', 'N/A')}**")

            # Show corrected problem if different from original
            corrected = result["solution"].get("corrected_problem", "")
            original = result["parsed"].get("original_text", "")
            if corrected and corrected != original:
                st.caption(f"**Corrected Problem:** {corrected}")

            st.subheader("Step-by-Step Explanation")
            st.markdown(result["explanation"].get("explanation", ""))

            key_concepts = result["explanation"].get("key_concepts", [])
            if key_concepts:
                st.subheader("Key Concepts")
                for c in key_concepts:
                    st.markdown(f"- {c}")

            tips = result["explanation"].get("tips", [])
            if tips:
                st.subheader("Tips for Similar Problems")
                for t in tips:
                    st.markdown(f"- {t}")

            common_mistakes = result["explanation"].get("common_mistakes", [])
            if common_mistakes:
                st.subheader("Common Mistakes to Avoid")
                for m in common_mistakes:
                    st.markdown(f"- {m}")

        with col_side:
            # Agent Trace
            st.subheader("Agent Trace")
            for step in result.get("trace", []):
                icon = "✅" if step["status"] == "completed" else "⏳"
                st.markdown(
                    f'<div class="agent-step agent-completed">'
                    f'{icon} <strong>{step["agent"]}</strong><br>'
                    f'<small>{step.get("output_summary", "")}</small></div>',
                    unsafe_allow_html=True,
                )

            # Retrieved Context
            st.subheader("Retrieved Context")
            sources = result["solution"].get("retrieved_sources", [])
            if sources:
                for src in sources:
                    st.markdown(
                        f'<span class="source-chip">{src["source"]} ({src["score"]:.2f})</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No sources retrieved.")

            # Confidence
            st.subheader("Confidence")
            solver_conf = result["solution"].get("confidence", 0)
            verifier_conf = result["verification"].get("confidence", 0)

            # Handle string confidence values
            try:
                solver_conf = float(solver_conf)
            except (ValueError, TypeError):
                solver_conf = 0.5
            try:
                verifier_conf = float(verifier_conf)
            except (ValueError, TypeError):
                verifier_conf = 0.5

            avg_conf = (solver_conf + verifier_conf) / 2
            st.progress(min(avg_conf, 1.0))

            if avg_conf >= 0.8:
                st.markdown(f'<span class="confidence-high">Overall: {avg_conf:.0%}</span>', unsafe_allow_html=True)
            elif avg_conf >= 0.5:
                st.markdown(f'<span class="confidence-medium">Overall: {avg_conf:.0%}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="confidence-low">Overall: {avg_conf:.0%}</span>', unsafe_allow_html=True)

            # Verification
            st.subheader("Verification")
            verification = result["verification"]
            if verification.get("correct", verification.get("is_correct")):
                st.success("Verified as correct")
            else:
                error_type = verification.get("error_type", "")
                st.error(f"Solution incorrect — Error: {error_type or 'unknown'}")

                # Show corrected solution if verifier provided one
                correct_answer = verification.get("correct_answer", "")
                correct_solution = verification.get("correct_solution", "")
                if correct_answer:
                    st.info(f"**Corrected Answer:** {correct_answer}")
                if correct_solution:
                    with st.expander("Corrected Solution (from Verifier)", expanded=True):
                        st.markdown(correct_solution)

            # Show detailed verification steps
            verif_steps = verification.get("verification_steps", [])
            if verif_steps:
                for v in verif_steps:
                    icon = "✅" if v.get("result") == "pass" else "❌"
                    st.markdown(f"{icon} **{v.get('check', '')}**: {v.get('detail', '')}")

            for issue in verification.get("issues", []):
                st.warning(issue)
            if verification.get("needs_human_review"):
                st.warning(f"Human review recommended: {verification.get('review_reason', '')}")
                st.session_state.hitl_active = True

            # Difficulty
            st.subheader("Difficulty")
            diff = result["explanation"].get("difficulty_rating", "Medium")
            diff_icons = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
            st.markdown(f"{diff_icons.get(diff, '🟡')} {diff}")

        # Expandable details
        with st.expander("Parsed Problem Details"):
            st.json(result["parsed"])
        with st.expander("Solution Details (Raw)"):
            st.json(result["solution"])
        with st.expander("Verification Details"):
            st.json(result["verification"])
        with st.expander("Routing Strategy"):
            st.json(result["route"])

        # ── Feedback (HITL) ──────────────────────────────────
        st.divider()
        st.subheader("Feedback")

        if not st.session_state.feedback_given:
            fc = st.columns([1, 1, 3])
            with fc[0]:
                if st.button("✅ Correct", use_container_width=True):
                    if st.session_state.current_problem_id:
                        update_feedback(st.session_state.current_problem_id, "correct")
                    st.session_state.feedback_given = True
                    st.success("Thanks! Feedback recorded.")
                    st.rerun()
            with fc[1]:
                if st.button("❌ Incorrect", use_container_width=True):
                    st.session_state.hitl_active = True

            if st.session_state.hitl_active:
                correction = st.text_area("What was wrong? Provide the correct answer/approach:")
                if st.button("Submit Correction"):
                    if st.session_state.current_problem_id and correction:
                        update_feedback(st.session_state.current_problem_id, "incorrect", correction)
                        st.session_state.feedback_given = True
                        st.success("Correction recorded! The system will learn from this.")
                        st.rerun()
        else:
            st.info("Feedback already recorded for this problem.")


# ── Reset ────────────────────────────────────────────────────

if st.session_state.result:
    if st.button("New Problem"):
        for key in ("result", "extracted_text", "extraction_confidence",
                     "current_problem_id", "hitl_active", "feedback_given"):
            st.session_state[key] = {"result": None, "extracted_text": "",
                                      "extraction_confidence": 1.0, "current_problem_id": None,
                                      "hitl_active": False, "feedback_given": False}[key]
        st.rerun()


# ── Footer ───────────────────────────────────────────────────

st.divider()
st.markdown(
    '<div style="text-align:center;color:#888;font-size:.85rem;">'
    'Math Mentor v2.0 | GPT-4o + LangGraph + LangChain RAG + FAISS + SQLite Memory'
    '</div>',
    unsafe_allow_html=True,
)
