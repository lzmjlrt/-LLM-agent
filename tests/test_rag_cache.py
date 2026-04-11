import os
import tempfile
import unittest

from agent.rag.cache import build_faiss_cache_path


class TestRagCachePath(unittest.TestCase):
    def test_cache_path_is_deterministic_for_same_inputs(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(b"same-content")
            pdf_path = tmp_file.name

        try:
            path1 = build_faiss_cache_path(pdf_path, "OpenAI", "text-embedding-3-small")
            path2 = build_faiss_cache_path(pdf_path, "OpenAI", "text-embedding-3-small")
            self.assertEqual(path1, path2)
        finally:
            os.remove(pdf_path)

    def test_cache_path_changes_when_pdf_content_changes(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file_a:
            tmp_file_a.write(b"content-a")
            path_a = tmp_file_a.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file_b:
            tmp_file_b.write(b"content-b")
            path_b = tmp_file_b.name

        try:
            cache_a = build_faiss_cache_path(path_a, "OpenAI", "text-embedding-3-small")
            cache_b = build_faiss_cache_path(path_b, "OpenAI", "text-embedding-3-small")
            self.assertNotEqual(cache_a, cache_b)
        finally:
            os.remove(path_a)
            os.remove(path_b)


if __name__ == "__main__":
    unittest.main()
