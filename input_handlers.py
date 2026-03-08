"""Multimodal input handling.

Image  : Mistral OCR (primary) → EasyOCR (fallback) → GPT-4o Vision (final fallback).
         Best result is selected based on confidence and content quality.
Audio  : OpenAI Whisper API.
Text   : passthrough.
"""

import os
import json
import base64
import logging
from typing import Tuple

from openai import OpenAI

import config

logger = logging.getLogger(__name__)

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


# ── Mistral OCR (primary) ────────────────────────────────────

def _mistral_ocr_extract(image_path: str) -> Tuple[str, float]:
    """Extract text from image using Mistral OCR API.

    Returns (extracted_text, confidence 0-1).
    """
    from mistralai import Mistral

    client = Mistral(api_key=config.MISTRAL_API_KEY)
    b64, mime = _encode_image(image_path)
    data_url = f"data:{mime};base64,{b64}"

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "image_url",
            "image_url": data_url,
        },
        include_image_base64=False,
    )

    # Combine markdown from all pages
    all_text = []
    for page in ocr_response.pages:
        if page.markdown and page.markdown.strip():
            all_text.append(page.markdown.strip())

    combined = "\n".join(all_text).strip()
    if not combined:
        return "", 0.0

    # Clean up markdown artifacts for math usage
    combined = _postprocess_mistral_math(combined)

    # Mistral OCR is high quality; assign confidence based on content
    confidence = 0.95 if len(combined) > 5 else 0.6
    return combined, confidence


def _postprocess_mistral_math(text: str) -> str:
    """Clean Mistral OCR markdown output for math problem usage."""
    import re as _re

    result = text

    # Remove markdown image placeholders
    result = _re.sub(r'!\[.*?\]\(.*?\)', '', result)
    # Remove markdown headings (# Find → Find)
    result = _re.sub(r'^#+\s*', '', result, flags=_re.MULTILINE)
    # Remove markdown bold/italic but keep content
    result = result.replace('**', '').replace('__', '')

    # ── LaTeX commands with arguments ──
    # \frac{a}{b} → (a)/(b)
    result = _re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', result)
    # \sqrt{x} → sqrt(x)
    result = _re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', result)
    # \text{abc}, \mathrm{abc}, \mathbf{abc} → abc
    result = _re.sub(r'\\(?:text|mathrm|mathbf|operatorname)\{([^}]*)\}', r'\1', result)

    # ── LaTeX symbol commands → plain math ──
    _latex_symbols = {
        '\\int': '\u222b',     # ∫
        '\\sum': '\u2211',     # ∑
        '\\prod': '\u220f',    # ∏
        '\\lim': 'lim',
        '\\infty': '\u221e',   # ∞
        '\\pm': '\u00b1',      # ±
        '\\mp': '\u2213',      # ∓
        '\\times': '*',
        '\\cdot': '*',
        '\\div': '/',
        '\\neq': '!=',
        '\\leq': '<=',
        '\\geq': '>=',
        '\\le': '<=',
        '\\ge': '>=',
        '\\lt': '<',
        '\\gt': '>',
        '\\approx': '\u2248',  # ≈
        '\\pi': '\u03c0',      # π
        '\\theta': '\u03b8',   # θ
        '\\alpha': '\u03b1',   # α
        '\\beta': '\u03b2',    # β
        '\\gamma': '\u03b3',   # γ
        '\\delta': '\u03b4',   # δ
        '\\sigma': '\u03c3',   # σ
        '\\mu': '\u03bc',      # μ
        '\\lambda': '\u03bb',  # λ
        '\\cap': '\u2229',     # ∩
        '\\cup': '\u222a',     # ∪
        '\\in': '\u2208',      # ∈
        '\\to': '\u2192',      # →
        '\\rightarrow': '\u2192',
        '\\left': '',
        '\\right': '',
        '\\,': ' ',
        '\\;': ' ',
        '\\quad': ' ',
        '\\qquad': '  ',
    }
    # Sort by longest key first to avoid partial matches (e.g. \le matching inside \left)
    for cmd, repl in sorted(_latex_symbols.items(), key=lambda x: -len(x[0])):
        result = result.replace(cmd, repl)

    # Convert LaTeX superscript: x^{2} → x^2
    result = _re.sub(r'\^\{([^}]*)\}', r'^\1', result)
    # Convert LaTeX subscript: x_{i} → x_i
    result = _re.sub(r'_\{([^}]*)\}', r'_\1', result)

    # Remove dollar signs (inline math delimiters)
    result = result.replace('$', '')
    # Remove stray backslashes before common symbols
    result = _re.sub(r'\\([+\-*/=(){}])', r'\1', result)
    # Remove any remaining lone backslash commands: \foo → foo
    result = _re.sub(r'\\([a-zA-Z]+)', r'\1', result)

    # Normalize minus signs and dashes
    result = result.replace('\u2014', '-').replace('\u2013', '-').replace('\u2212', '-')

    # ── Clean up coefficient "1" artifacts ──
    # "1x" → "x" (but keep "10x", "11x", "1.5x", etc.)
    result = _re.sub(r'(?<![0-9.])1([a-zA-Z])', r'\1', result)

    # Collapse multiple spaces and blank lines
    result = _re.sub(r' +', ' ', result)
    result = _re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


# ── EasyOCR (fallback) ───────────────────────────────────────

def _has_math_symbols(text: str) -> bool:
    """Check if text contains math notation that EasyOCR often misreads."""
    import re as _re
    return bool(_re.search(r'[\^=+\-*/]|x\d|\d[a-zA-Z]', text))


def _postprocess_ocr_math(text: str) -> str:
    """Clean up common EasyOCR artifacts in math expressions."""
    import re as _re
    result = text

    # Fix common EasyOCR misreads of math symbols
    result = result.replace('}{', ')(')  # curly braces → parens
    result = result.replace('{', '(').replace('}', ')')
    result = result.replace('|', '1')  # pipe often misread for 1 in equations

    # Normalize minus signs and dashes
    result = result.replace('—', '-').replace('–', '-').replace('−', '-')

    # Fix spaces around = sign
    result = _re.sub(r'=(?=\S)', '= ', result)
    result = _re.sub(r'(?<=\S)=', ' =', result)

    # Collapse multiple spaces
    result = _re.sub(r' +', ' ', result).strip()

    return result


def _easyocr_extract(image_path: str) -> Tuple[str, float]:
    """Extract text using EasyOCR. Returns (text, confidence)."""
    reader = _get_ocr_reader()
    detections = reader.readtext(image_path)

    if not detections:
        return "", 0.0

    texts = []
    confidences = []
    for _bbox, text, conf in detections:
        texts.append(text)
        confidences.append(conf)

    ocr_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences)
    ocr_text = _postprocess_ocr_math(ocr_text)

    return ocr_text, round(avg_confidence, 3)


# ── Main OCR pipeline: Mistral → EasyOCR → GPT-4o ───────────

def extract_text_from_image(image_path: str) -> Tuple[str, float]:
    """Extract math problem text from an image using a 3-tier OCR pipeline.

    Pipeline order:
    1. Mistral OCR (best for structured math, LaTeX, complex layouts)
    2. EasyOCR (fallback if Mistral fails)
    3. GPT-4o Vision (final fallback, uses vision model to read image)

    Returns (extracted_text, confidence 0-1).
    """
    ocr_source = None

    # --- Tier 1: Mistral OCR ---
    if config.MISTRAL_API_KEY:
        try:
            text, conf = _mistral_ocr_extract(image_path)
            if text and len(text.strip()) > 2:
                logger.info(f"Mistral OCR succeeded (confidence={conf})")
                ocr_source = "mistral"
                return text, conf
            logger.warning("Mistral OCR returned empty/short text, trying EasyOCR")
        except Exception as e:
            logger.warning(f"Mistral OCR failed: {e}, falling back to EasyOCR")

    # --- Tier 2: EasyOCR ---
    try:
        text, conf = _easyocr_extract(image_path)
        if text and len(text.strip()) > 2:
            logger.info(f"EasyOCR succeeded (confidence={conf})")
            ocr_source = "easyocr"

            # If EasyOCR confidence is decent, return as-is
            # If low confidence + math symbols, try GPT-4o refinement
            needs_refinement = (
                (conf < 0.8 and _has_math_symbols(text))
                or conf < 0.6
            )
            if needs_refinement and config.OPENAI_API_KEY:
                refined_text, refined_conf = _gpt4o_vision_refine(image_path, text)
                if refined_text:
                    return refined_text, refined_conf

            return text, conf
        logger.warning("EasyOCR returned empty/short text, trying GPT-4o Vision")
    except Exception as e:
        logger.warning(f"EasyOCR failed: {e}, falling back to GPT-4o Vision")

    # --- Tier 3: GPT-4o Vision (pure extraction) ---
    if config.OPENAI_API_KEY:
        try:
            text, conf = _gpt4o_vision_extract(image_path)
            if text:
                logger.info(f"GPT-4o Vision succeeded (confidence={conf})")
                return text, conf
        except Exception as e:
            logger.error(f"GPT-4o Vision also failed: {e}")

    return "", 0.0


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
                    "version of the math problem.\n\n"
                    "CRITICAL RULES:\n"
                    "- Read EVERY digit, coefficient, and exponent directly from the image.\n"
                    "- Do NOT guess or 'fix' coefficients — copy them exactly as written.\n"
                    "- Pay special attention to: exponents (^2, ^3), minus vs plus signs, "
                    "multi-digit numbers (36 vs 6, 13 vs 3), and coefficients before variables.\n"
                    "- Use standard notation: x^2 for x squared, * for multiplication.\n\n"
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
    "Pay very careful attention to multi-digit numbers like 36, 25, 144 — do not drop digits. "
    "Use standard notation: x^2 for 'x squared', x^3 for 'x cubed', sqrt() for square root. "
    "Common terms: integral, derivative, limit, matrix, determinant, eigenvalue, "
    "probability, factorial, sigma, pi, theta, sqrt, log, ln, sin, cos, tan. "
    "Example equations: x^2 - 5x + 6 = 0, 3x^2 - 13x + 36 = 0, P(A|B) = 1/2."
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
    """Replace spoken math phrases with symbolic equivalents.

    Handles:
    - Spoken operators and relations ("plus" → "+", "equals" → "=")
    - Spoken exponents ("x square" → "x^2", "x cube" → "x^3")
    - Transcription artifacts ("= to 0" → "= 0", "equal to" → "=")
    """
    import re as _re

    # Longer phrases first to avoid partial matches
    replacements = {
        "square root of": "sqrt(",
        "is equal to": "=",
        "equal to": "=",
        "greater than or equal to": ">=",
        "less than or equal to": "<=",
        "greater than": ">",
        "less than": "<",
        "raised to the power of": "^",
        "raised to the power": "^",
        "to the power of": "^",
        "raised to": "^",
        "divided by": "/",
        "multiplied by": "*",
        "squared": "^2",
        "cubed": "^3",
        "times": "*",
        "plus": "+",
        "minus": "-",
        "equals": "=",
    }
    result = text
    for phrase, replacement in replacements.items():
        result = result.replace(phrase, replacement)

    # Fix spoken exponents: "x square" → "x^2", "x cube" → "x^3"
    result = _re.sub(r'(\b[a-zA-Z])\s+square\b', r'\1^2', result, flags=_re.IGNORECASE)
    result = _re.sub(r'(\b[a-zA-Z])\s+cube\b', r'\1^3', result, flags=_re.IGNORECASE)

    # Fix "= to" artifact: "= to 0" → "= 0" (common Whisper mishearing)
    result = _re.sub(r'=\s*to\s+', '= ', result)

    # Fix "where" before variable: "where x^2" → "x^2" (sometimes Whisper adds context words)
    result = _re.sub(r'\bwhere\s+', '', result, flags=_re.IGNORECASE)

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
