import hashlib
import os

from agent import config


def file_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_obj:
        while True:
            chunk = file_obj.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def build_faiss_cache_path(pdf_path: str, embedding_provider: str, embedding_model_name: str) -> str:
    pdf_hash = file_sha256(pdf_path)[:16]
    provider_key = embedding_provider.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_key = embedding_model_name.lower().replace(" ", "_").replace("-", "_")
    cache_key = f"{provider_key}_{model_key}_{pdf_hash}"
    return os.path.join(config.FAISS_CACHE_DIR_PATH, cache_key)
