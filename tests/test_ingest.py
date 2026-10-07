import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rag.ingest import _build_vectorstore, _source_manifest


class SourceManifestTest(unittest.TestCase):
    def test_rebuilds_when_pdf_changes_and_reuses_current_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            pdf = data_dir / "paper.pdf"
            pdf.write_bytes(b"original")
            store = Mock()

            def save_local(path):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "index.faiss").touch()

            store.save_local.side_effect = save_local
            with (
                patch("rag.ingest.DATA_DIR", str(data_dir)),
                patch("rag.ingest.PERSIST_DIR", str(root / "indexes")),
                patch("rag.ingest._load_pdf", return_value=([object()], None)),
                patch("rag.ingest._build_chunks", return_value=[object()]),
                patch("rag.ingest.get_embeddings", return_value=object()),
                patch("rag.ingest.FAISS.from_documents", return_value=store) as build,
                patch("rag.ingest.FAISS.load_local", return_value=store) as load,
            ):
                _build_vectorstore("candidate", "candidate", ["paper.pdf"])
                _build_vectorstore("candidate", "candidate", ["paper.pdf"])
                self.assertEqual(build.call_count, 1)
                self.assertEqual(load.call_count, 1)

                pdf.write_bytes(b"revised source")
                _build_vectorstore("candidate", "candidate", ["paper.pdf"])
                self.assertEqual(build.call_count, 2)

    def test_detects_new_and_changed_pdfs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.pdf"
            second = root / "second.pdf"
            first.write_bytes(b"first")

            with patch("rag.ingest.DATA_DIR", directory):
                original = _source_manifest(["first.pdf"])
                with_missing = _source_manifest(["first.pdf", "second.pdf"])
                second.write_bytes(b"second")
                with_added = _source_manifest(["first.pdf", "second.pdf"])
                first.write_bytes(b"first revised")
                with_changed = _source_manifest(["first.pdf", "second.pdf"])

        self.assertNotEqual(original, with_missing)
        self.assertNotEqual(with_missing, with_added)
        self.assertNotEqual(with_added, with_changed)


if __name__ == "__main__":
    unittest.main()
