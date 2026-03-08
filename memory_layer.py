"""Memory & Self-Learning Layer using SQLite + JSON + OpenAI Embeddings."""

import json
import sqlite3
import time
import numpy as np
from typing import List

from openai import OpenAI

import config

DB_PATH = config.MEMORY_DB_PATH


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS solved_problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_type TEXT,
            original_input TEXT,
            parsed_question TEXT,
            topic TEXT,
            retrieved_context TEXT,
            solution TEXT,
            explanation TEXT,
            verifier_result TEXT,
            user_feedback TEXT DEFAULT '',
            user_correction TEXT DEFAULT '',
            timestamp REAL
        )
    """)
    conn.commit()
    return conn


def _to_str(value) -> str:
    """Ensure a value is a plain string for SQLite storage."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def store_problem(entry: dict) -> int:
    """Store a solved problem. Returns the row id."""
    conn = _get_conn()
    cur = conn.execute(
        """INSERT INTO solved_problems
           (input_type, original_input, parsed_question, topic,
            retrieved_context, solution, explanation, verifier_result,
            user_feedback, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            entry.get("input_type", "text"),
            entry.get("original_input", ""),
            entry.get("parsed_question", ""),
            entry.get("topic", ""),
            json.dumps(entry.get("retrieved_context", [])),
            _to_str(entry.get("solution", "")),
            _to_str(entry.get("explanation", "")),
            json.dumps(entry.get("verifier_result", {})),
            entry.get("user_feedback", ""),
            time.time(),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def update_feedback(problem_id: int, feedback: str, correction: str = ""):
    """Update feedback for a stored problem."""
    conn = _get_conn()
    conn.execute(
        "UPDATE solved_problems SET user_feedback = ?, user_correction = ? WHERE id = ?",
        (feedback, correction, problem_id),
    )
    conn.commit()
    conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    # Parse JSON fields back
    for field in ("retrieved_context", "verifier_result"):
        if field in d and isinstance(d[field], str):
            try:
                d[field] = json.loads(d[field])
            except json.JSONDecodeError:
                pass
    return d


def _get_embedding(text: str) -> List[float]:
    """Get embedding vector for a text string using OpenAI."""
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text[:2000],  # truncate to avoid token limits
        )
        return resp.data[0].embedding
    except Exception:
        return []


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def retrieve_similar(query: str, topic: str = "", top_k: int = 3) -> List[dict]:
    """Retrieve similar previously solved problems using embedding similarity.

    Falls back to keyword overlap if embedding fails (e.g., no API key).
    """
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM solved_problems ORDER BY timestamp DESC").fetchall()
    conn.close()

    if not rows:
        return []

    # Try embedding-based similarity first
    query_emb = _get_embedding(query)
    if query_emb:
        scored = []
        for row in rows:
            d = _row_to_dict(row)
            parsed = d.get("parsed_question", "")
            if not parsed:
                continue
            mem_emb = _get_embedding(parsed)
            if not mem_emb:
                continue
            sim = _cosine_similarity(query_emb, mem_emb)
            # Boost topic matches and user-confirmed correct answers
            topic_bonus = 0.1 if topic and d.get("topic", "").lower() == topic.lower() else 0
            feedback_bonus = 0.05 if d.get("user_feedback") == "correct" else 0
            final_score = sim + topic_bonus + feedback_bonus
            scored.append((final_score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Only return results with meaningful similarity (> 0.7)
        return [item[1] for item in scored[:top_k] if item[0] > 0.7]

    # Fallback: keyword overlap (when embeddings unavailable)
    query_words = set(query.lower().split())
    scored = []
    for row in rows:
        d = _row_to_dict(row)
        parsed = d.get("parsed_question", "").lower()
        mem_words = set(parsed.split())
        overlap = len(query_words & mem_words)
        topic_bonus = 5 if topic and d.get("topic", "").lower() == topic.lower() else 0
        feedback_bonus = 3 if d.get("user_feedback") == "correct" else 0
        score = overlap + topic_bonus + feedback_bonus
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def get_correction_patterns(topic: str = "") -> List[dict]:
    """Get patterns from corrected problems."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM solved_problems WHERE user_feedback = 'incorrect' AND user_correction != ''"
    ).fetchall()
    conn.close()

    corrections = []
    for row in rows:
        d = _row_to_dict(row)
        if not topic or d.get("topic", "").lower() == topic.lower():
            corrections.append({
                "original_question": d.get("parsed_question", ""),
                "wrong_answer": d.get("solution", ""),
                "correction": d.get("user_correction", ""),
                "topic": d.get("topic", ""),
            })
    return corrections


def get_all_memories() -> List[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM solved_problems ORDER BY timestamp DESC").fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def clear_memory():
    conn = _get_conn()
    conn.execute("DELETE FROM solved_problems")
    conn.commit()
    conn.close()
