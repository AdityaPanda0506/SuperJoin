"""
Layout-Aware PDF Page Parser & Chunker using PyMuPDF (pymupdf).
Extracts clean text, detects table structures, generates document hashes, and chunks content.
"""

import hashlib
import io
import re
from pathlib import Path
from typing import Any, Optional, Union

import pymupdf  # Modern PyMuPDF API


class PageChunk:
    """
    Represents a structured page-level text chunk extracted from a PDF document.
    Supports opportunistic multimodal visual chart fallback.
    """

    def __init__(
        self,
        doc_name: str,
        doc_hash: str,
        page_number: int,
        clean_text: str,
        has_tables: bool,
        token_estimate: int,
        has_images: bool = False,
        image_bytes: bytes | None = None,
        provenance_modality: str = "TEXT",
    ):
        self.doc_name = doc_name
        self.doc_hash = doc_hash
        self.page_number = page_number
        self.clean_text = clean_text
        self.has_tables = has_tables
        self.token_estimate = token_estimate
        self.has_images = has_images
        self.image_bytes = image_bytes
        self.provenance_modality = provenance_modality

    def __repr__(self) -> str:
        return (
            f"<PageChunk doc={self.doc_name} page={self.page_number} "
            f"tokens~={self.token_estimate} tables={self.has_tables} "
            f"images={self.has_images} modality={self.provenance_modality}>"
        )


class DocumentParser:
    """
    Layout-aware parser using PyMuPDF to extract text chunks, detect charts/images, and document metadata.
    """

    def __init__(self, max_tokens_per_chunk: int = 2048):
        """
        Initialize DocumentParser.

        Args:
            max_tokens_per_chunk: Soft limit for maximum tokens per chunk before splitting.
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk

    @staticmethod
    def compute_sha256(content_bytes: bytes) -> str:
        """Compute SHA-256 hash of PDF binary contents."""
        return hashlib.sha256(content_bytes).hexdigest()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count estimation (approx 4 chars per token or ~0.75 words per token)."""
        words = text.split() if text else []
        return max(1, int(len(words) * 1.3))

    @classmethod
    def clean_page_text(cls, raw_text: str) -> str:
        """
        Normalize extracted raw page text while retaining line breaks and layout flow.
        """
        if not raw_text:
            return ""

        # Normalize carriage returns and non-breaking spaces
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
        # Strip excessive trailing spaces on lines while keeping paragraph line breaks
        lines = [line.rstrip() for line in text.split("\n")]
        # Remove more than 3 consecutive blank lines
        cleaned = re.sub(r"\n{4,}", "\n\n\n", "\n".join(lines))
        return cleaned.strip()

    @classmethod
    def detect_tables(cls, page: pymupdf.Page) -> bool:
        """
        Detect table structures on a page using PyMuPDF table finder or layout heuristics.
        """
        try:
            # Native PyMuPDF table finder check
            tables = page.find_tables()
            if tables and len(tables.tables) > 0:
                return True
        except Exception:
            pass

        # Fallback heuristic: check for drawing lines or grid-like text blocks
        try:
            drawings = page.get_drawings()
            rect_count = sum(1 for d in drawings if d.get("type") in ("rect", "f", "s"))
            if rect_count > 4:
                return True
        except Exception:
            pass

        return False

    @classmethod
    def detect_visual_elements(cls, page: pymupdf.Page) -> tuple[bool, bool]:
        """
        Detect visual elements (embedded images, bar/line plots, infographics) and tables.
        Returns: Tuple of (has_visual_elements: bool, has_tables: bool)
        """
        has_tables = cls.detect_tables(page)
        has_visual = False
        try:
            images = page.get_images()
            if images and len(images) > 0:
                has_visual = True
            else:
                drawings = page.get_drawings()
                if drawings and len(drawings) > 4:
                    has_visual = True
        except Exception:
            pass

        return has_visual, has_tables

    @classmethod
    def render_page_to_image(cls, page: pymupdf.Page, dpi: int = 150) -> bytes:
        """
        Opportunistically render page to 150 DPI JPEG image bytes in memory for multimodal vision extraction.
        """
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("jpeg")

    @classmethod
    def parse_pdf(
        cls,
        source_input: Union[str, Path, bytes, io.BytesIO],
        doc_name: str | None = None,
    ) -> list[PageChunk]:
        """Classmethod helper to parse PDF and return list of PageChunk objects."""
        parser = cls()
        _, chunks, _ = parser.parse_document(source_input, doc_name=doc_name)
        return chunks

    @classmethod
    def parse_pdf_buffer(
        cls,
        buffer: Union[bytes, io.BytesIO, memoryview],
        doc_name: str,
    ) -> list[dict[str, Any]]:
        """
        Parse a PDF buffer into a list of page dictionaries containing clean text, page number, doc name, and doc hash.
        """
        parser = cls()
        if isinstance(buffer, memoryview):
            buffer = bytes(buffer)
        doc_hash, chunks, _ = parser.parse_document(buffer, doc_name=doc_name)
        pages = []
        for chunk in chunks:
            pages.append(
                {
                    "clean_text": chunk.clean_text,
                    "source_doc_name": chunk.doc_name,
                    "page_number": chunk.page_number,
                    "doc_hash": chunk.doc_hash,
                    "has_images": chunk.has_images,
                    "image_bytes": chunk.image_bytes,
                    "provenance_modality": chunk.provenance_modality,
                }
            )
        return pages

    def parse_document(
        self,
        source_input: Union[str, Path, bytes, io.BytesIO],
        doc_name: str | None = None,
    ) -> tuple[str, list[PageChunk], dict[int, str]]:
        """
        Parse a PDF document source into page chunks.

        Args:
            source_input: File path, Path object, raw bytes, or BytesIO buffer.
            doc_name: Optional custom document name.

        Returns:
            Tuple of (doc_hash, list of PageChunk objects, dictionary of page_number -> clean_text).
        """
        if isinstance(source_input, (str, Path)):
            path_obj = Path(source_input)
            filename = doc_name or path_obj.name
            with open(path_obj, "rb") as f:
                pdf_bytes = f.read()
        elif isinstance(source_input, io.BytesIO):
            pdf_bytes = source_input.getvalue()
            filename = doc_name or "document.pdf"
        elif isinstance(source_input, bytes):
            pdf_bytes = source_input
            filename = doc_name or "document.pdf"
        else:
            raise ValueError(f"Unsupported source_input type: {type(source_input)}")

        doc_hash = self.compute_sha256(pdf_bytes)
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        page_chunks: list[PageChunk] = []
        page_texts: dict[int, str] = {}

        for page_idx in range(len(doc)):
            page_num = page_idx + 1  # 1-indexed
            page = doc[page_idx]
            raw_text = page.get_text("text")
            clean_text = self.clean_page_text(raw_text)

            has_visual, has_tables = self.detect_visual_elements(page)
            if not clean_text and not has_visual:
                continue

            img_bytes = self.render_page_to_image(page, dpi=150) if has_visual else None
            modality = "VISUAL_CHART" if has_visual else "TEXT"

            token_estimate = self.estimate_tokens(clean_text) if clean_text else 50
            page_texts[page_num] = clean_text or "[Visual Page]"

            if token_estimate > self.max_tokens_per_chunk:
                sub_chunks = self._sub_chunk_text(clean_text, self.max_tokens_per_chunk)
                for sub_text in sub_chunks:
                    page_chunks.append(
                        PageChunk(
                            doc_name=filename,
                            doc_hash=doc_hash,
                            page_number=page_num,
                            clean_text=sub_text,
                            has_tables=has_tables,
                            token_estimate=self.estimate_tokens(sub_text),
                            has_images=has_visual,
                            image_bytes=img_bytes,
                            provenance_modality=modality,
                        )
                    )
            else:
                page_chunks.append(
                    PageChunk(
                        doc_name=filename,
                        doc_hash=doc_hash,
                        page_number=page_num,
                        clean_text=clean_text,
                        has_tables=has_tables,
                        token_estimate=token_estimate,
                        has_images=has_visual,
                        image_bytes=img_bytes,
                        provenance_modality=modality,
                    )
                )

        doc.close()
        return doc_hash, page_chunks, page_texts

    def _sub_chunk_text(self, text: str, max_tokens: int) -> list[str]:
        """Split page text into smaller paragraph-bounded chunks if oversized."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_tokens = 0

        for p in paragraphs:
            p_tokens = self.estimate_tokens(p)
            if current_tokens + p_tokens > max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [p]
                current_tokens = p_tokens
            else:
                current_chunk.append(p)
                current_tokens += p_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
