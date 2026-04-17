import json
import os
import re
from difflib import SequenceMatcher
from typing import Any

from agent import config


def _safe_thread_id(thread_id: str) -> str:
    raw = (thread_id or "").strip()
    if not raw:
        return "default-thread"
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._")
    return normalized or "default-thread"


def _store_path(thread_id: str) -> str:
    safe_thread_id = _safe_thread_id(thread_id)
    return os.path.join(config.CONVERSATION_STORE_DIR_PATH, f"{safe_thread_id}.json")


def _empty_memory() -> dict[str, Any]:
    return {"summary": "", "turns": [], "qa_cache": []}


def load_thread_memory(thread_id: str) -> dict[str, Any]:
    os.makedirs(config.CONVERSATION_STORE_DIR_PATH, exist_ok=True)
    path = _store_path(thread_id)
    if not os.path.exists(path):
        return _empty_memory()
    with open(path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise RuntimeError("会话记忆文件格式错误：根节点必须为对象。")
    data.setdefault("summary", "")
    data.setdefault("turns", [])
    data.setdefault("qa_cache", [])
    return data


def save_thread_memory(thread_id: str, memory: dict[str, Any]) -> None:
    os.makedirs(config.CONVERSATION_STORE_DIR_PATH, exist_ok=True)
    path = _store_path(thread_id)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(memory, file_obj, ensure_ascii=False, indent=2)


def _extract_terms(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]+", (text or "").lower()))


def semantic_similarity(text_a: str, text_b: str) -> float:
    normalized_a = (text_a or "").strip()
    normalized_b = (text_b or "").strip()
    if not normalized_a or not normalized_b:
        return 0.0

    seq_ratio = SequenceMatcher(None, normalized_a, normalized_b).ratio()
    terms_a = _extract_terms(normalized_a)
    terms_b = _extract_terms(normalized_b)
    if not terms_a or not terms_b:
        token_jaccard = 0.0
    else:
        token_jaccard = len(terms_a & terms_b) / len(terms_a | terms_b)
    return (0.6 * seq_ratio) + (0.4 * token_jaccard)


def find_cached_reply(
    memory: dict[str, Any],
    user_prompt: str,
    similarity_threshold: float,
) -> tuple[str | None, float]:
    qa_cache = memory.get("qa_cache", [])
    best_reply = None
    best_score = 0.0
    for item in qa_cache:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", ""))
        reply = str(item.get("reply", ""))
        if not question or not reply:
            continue
        score = semantic_similarity(user_prompt, question)
        if score > best_score:
            best_score = score
            best_reply = reply
    if best_reply and best_score >= similarity_threshold:
        return best_reply, best_score
    return None, best_score


def build_conversation_context(memory: dict[str, Any], max_turns: int = 3) -> str:
    summary = str(memory.get("summary", "")).strip()
    turns = memory.get("turns", [])
    if not isinstance(turns, list):
        turns = []
    recent_turns = turns[-max_turns:]

    lines = []
    if summary:
        lines.append(f"历史摘要：{summary}")
    for item in recent_turns:
        if not isinstance(item, dict):
            continue
        user_msg = str(item.get("user", "")).strip()
        assistant_msg = str(item.get("assistant", "")).strip()
        if user_msg:
            lines.append(f"用户：{user_msg}")
        if assistant_msg:
            lines.append(f"客服：{assistant_msg}")
    return "\n".join(lines)


def append_conversation_turn(
    memory: dict[str, Any],
    user_prompt: str,
    reply: str,
    history_limit: int,
) -> dict[str, Any]:
    turns = memory.get("turns", [])
    qa_cache = memory.get("qa_cache", [])
    if not isinstance(turns, list):
        turns = []
    if not isinstance(qa_cache, list):
        qa_cache = []

    turns.append({"user": user_prompt, "assistant": reply})
    if history_limit > 0:
        turns = turns[-history_limit:]

    qa_cache.append({"question": user_prompt, "reply": reply})
    cache_limit = max(history_limit * 2, 10)
    qa_cache = qa_cache[-cache_limit:]

    recent_questions = [str(item.get("user", "")).strip() for item in turns if isinstance(item, dict)]
    recent_questions = [q for q in recent_questions if q]
    summary = "；".join(recent_questions[-3:])
    if summary:
        summary = f"最近用户关注：{summary}"

    memory["turns"] = turns
    memory["qa_cache"] = qa_cache
    memory["summary"] = summary
    return memory


__all__ = [
    "append_conversation_turn",
    "build_conversation_context",
    "find_cached_reply",
    "load_thread_memory",
    "save_thread_memory",
    "semantic_similarity",
]
