"""Multimodal input handling.

Image  : EasyOCR (primary) with per-detection confidence scores.
         If OCR confidence is low, GPT-4o Vision refines the extraction.
Audio  : OpenAI Whisper API.
Text   : passthrough.
"""

import os
import json
import base64
from typing import Tuple

from openai import OpenAI

import config

# ── lazy-loaded EasyOCR reader (heavy import) ────────────────

_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def _get_openai() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


# ── Image → Text (EasyOCR + optional GPT-4o refinement) ─────

def extract_text_from_image(image_path: str) -> Tuple[str, float]:
    """Extract math problem text from an image using EasyOCR.

    Returns (extracted_text, avg_confidence 0-1).
    If avg confidence < 0.6 and an OpenAI key is available,
    GPT-4o Vision is used to refine the OCR output.
    """
    # --- Step 1: EasyOCR ---
    reader = _get_ocr_reader()
    detections = reader.readtext(image_path)

    if not detections:
        # No text detected at all – fall back to GPT-4o Vision directly
        if config.OPENAI_API_KEY:
            return _gpt4o_vision_extract(image_path)
        return "", 0.0

    texts = []
    confidences = []
    for _bbox, text, conf in detections:
        texts.append(text)
        confidences.append(conf)

    ocr_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences)

    # --- Step 2: If low confidence, refine with GPT-4o Vision ---
    if avg_confidence < 0.6 and config.OPENAI_API_KEY:
        refined_text, refined_conf = _gpt4o_vision_refine(image_path, ocr_text)
        if refined_text:
            return refined_text, refined_conf

    return ocr_text, round(avg_confidence, 3)


def _gpt4o_vision_extract(image_path: str) -> Tuple[str, float]:
    """Pure GPT-4o Vision extraction (used when EasyOCR finds nothing)."""
    client = _get_openai()
    b64, mime = _encode_image(image_path)

    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a math OCR expert. Extract the math problem from the image. "
                    'Return ONLY a JSON object: {"text": "...", "confidence": 0.0-1.0}. '
                    "No markdown fences."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract the math problem from this image."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return _parse_vision_json(resp.choices[0].message.content)


def _gpt4o_vision_refine(image_path: str, ocr_text: str) -> Tuple[str, float]:
    """Use GPT-4o Vision to fix / improve a low-confidence OCR result."""
    client = _get_openai()
    b64, mime = _encode_image(image_path)

    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a math OCR expert. The OCR engine produced the text below "
                    "but with LOW confidence. Look at the image and return a corrected "
                    "version of the math problem.\n"
                    'Return ONLY JSON: {"text": "...", "confidence": 0.0-1.0}. '
                    "No markdown fences."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"OCR output (low confidence):\n{ocr_text}\n\nPlease correct using the image.",
                    },
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return _parse_vision_json(resp.choices[0].message.content)


# ── helpers ──────────────────────────────────────────────────

def _encode_image(image_path: str) -> Tuple[str, str]:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")
    return b64, mime


def _parse_vision_json(raw: str) -> Tuple[str, float]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(raw)
        return parsed.get("text", raw), float(parsed.get("confidence", 0.7))
    except (json.JSONDecodeError, ValueError):
        return raw, 0.7


# ── Audio → Text (OpenAI gpt-4o-transcribe) ──────────────────

MATH_TRANSCRIPTION_PROMPT = (
    "The speaker is describing a math problem. "
    "Transcribe exactly, preserving all numbers, variables, operators, and equations. "
    "Common terms: integral, derivative, limit, matrix, determinant, eigenvalue, "
    "probability, factorial, sigma, pi, theta, sqrt, log, ln, sin, cos, tan."
)


def transcribe_audio(audio_path: str) -> Tuple[str, float]:
    """Transcribe audio using OpenAI gpt-4o-transcribe.

    Returns (transcript, confidence 0-1).
    """
    client = _get_openai()

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=config.WHISPER_MODEL,   # gpt-4o-transcribe
            file=f,
            response_format="text",
            prompt=MATH_TRANSCRIPTION_PROMPT,
        )

    # gpt-4o-transcribe with response_format="text" returns a plain string
    text = response.strip() if isinstance(response, str) else response.text.strip()
    text = _postprocess_math_transcript(text)

    confidence = 0.92 if len(text) > 10 else 0.5
    return text, confidence


def _postprocess_math_transcript(text: str) -> str:
    """Replace spoken math phrases with symbolic equivalents."""
    replacements = {
        "square root of": "sqrt(",
        "squared": "^2",
        "cubed": "^3",
        "raised to the power": "^",
        "raised to": "^",
        "to the power of": "^",
        "divided by": "/",
        "multiplied by": "*",
        "times": "*",
        "plus": "+",
        "minus": "-",
        "equals": "=",
        "is equal to": "=",
        "greater than or equal to": ">=",
        "less than or equal to": "<=",
        "greater than": ">",
        "less than": "<",
    }
    result = text
    for phrase, replacement in replacements.items():
        result = result.replace(phrase, replacement)
    return result


# ── Sample image generator ───────────────────────────────────

def create_sample_math_image(directory: str) -> str:
    """Create a sample math problem image for testing and return its path."""
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "sample_math_problem.png")
    if os.path.exists(path):
        return path

    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_body  = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font_title = ImageFont.load_default()
        font_body  = font_title

    lines = [
        ("Math Problem - Sample Test Image", (20, 20), font_title, (30, 30, 180)),
        ("Solve for x:", (20, 70), font_body, (0, 0, 0)),
        ("x^2 - 5x + 6 = 0", (20, 100), font_body, (0, 0, 0)),
        ("", (20, 130), font_body, (0, 0, 0)),
        ("Find the probability:", (20, 150), font_body, (0, 0, 0)),
        ("P(A) = 3/4, P(B) = 1/2, P(A and B) = 1/4", (20, 178), font_body, (0, 0, 0)),
        ("Find P(A or B).", (20, 206), font_body, (0, 0, 0)),
        ("", (20, 234), font_body, (0, 0, 0)),
        ("Calculate the derivative of f(x) = x^3 - 3x^2 + 2x", (20, 254), font_body, (0, 0, 0)),
    ]
    for text, pos, font, color in lines:
        if text:
            draw.text(pos, text, fill=color, font=font)

    draw.rectangle([10, 10, 590, 290], outline=(100, 100, 100), width=2)
    img.save(path)
    return path


# ── File upload helper ───────────────────────────────────────

def save_uploaded_file(uploaded_file, directory: str) -> str:
    """Save a Streamlit UploadedFile to disk and return the path."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
