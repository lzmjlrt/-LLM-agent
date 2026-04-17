import hashlib
import json
import os
from datetime import UTC, datetime

from agent import config


def sanitize_filename(file_name: str) -> str:
    raw_name = (file_name or "uploaded_manual.pdf").strip()
    safe_chars = []
    for char in raw_name:
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars).strip("._")
    return sanitized or "uploaded_manual.pdf"


def persist_uploaded_manual(uploaded_file, thread_id: str) -> tuple[str, str]:
    file_bytes = bytes(uploaded_file.getbuffer())
    pdf_hash = hashlib.sha256(file_bytes).hexdigest()

    upload_dir = os.path.join(config.UPLOADS_DIR_PATH, thread_id)
    os.makedirs(upload_dir, exist_ok=True)

    safe_name = sanitize_filename(getattr(uploaded_file, "name", "manual.pdf"))
    pdf_name = f"{pdf_hash[:12]}_{safe_name}"
    pdf_path = os.path.join(upload_dir, pdf_name)
    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as file_obj:
            file_obj.write(file_bytes)

    metadata = {
        "thread_id": thread_id,
        "pdf_hash": pdf_hash,
        "filename": safe_name,
        "saved_path": pdf_path,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    metadata_path = os.path.join(upload_dir, "index_meta.json")
    with open(metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    return pdf_path, pdf_hash


__all__ = ["persist_uploaded_manual", "sanitize_filename"]
