"""
Document loader for various file formats.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from ..core.config import settings
from ..schemas.document import DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """Exception raised when document loading fails."""

    pass


class DocumentLoader:
    """
    Loads documents from various formats and extracts text content.

    Supports PDF, images (with OCR), and text files.
    """

    def __init__(self, ocr_enabled: Optional[bool] = None, ocr_language: Optional[str] = None):
        """
        Initialize the document loader.

        Args:
            ocr_enabled: Whether to enable OCR for images
            ocr_language: Language for OCR (default: eng)
        """
        self.ocr_enabled = ocr_enabled if ocr_enabled is not None else settings.ocr_enabled
        self.ocr_language = ocr_language or settings.ocr_language
        self._tesseract_available = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            logger.warning("Tesseract OCR not available. Image OCR will be disabled.")
            return False

    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def load(self, file_path: str) -> Tuple[DocumentMetadata, str]:
        """
        Load a document and extract its text content.

        Args:
            file_path: Path to the document file

        Returns:
            Tuple of (DocumentMetadata, extracted_text)

        Raises:
            DocumentLoadError: If the document cannot be loaded
        """
        path = Path(file_path)

        if not path.exists():
            raise DocumentLoadError(f"File not found: {file_path}")

        file_size = path.stat().st_size
        max_size = settings.max_document_size_mb * 1024 * 1024

        if file_size > max_size:
            raise DocumentLoadError(
                f"File too large: {file_size / 1024 / 1024:.2f}MB "
                f"(max: {settings.max_document_size_mb}MB)"
            )

        file_type = path.suffix.lower().lstrip(".")
        metadata = DocumentMetadata(
            file_path=str(path.absolute()),
            file_name=path.name,
            file_type=file_type,
            file_size_bytes=file_size,
            source_hash=self._compute_hash(file_path),
        )

        # Load based on file type
        if file_type == "pdf":
            text, page_count = self._load_pdf(file_path)
            metadata.page_count = page_count
        elif file_type in ("jpg", "jpeg", "png", "gif", "bmp", "tiff"):
            text = self._load_image(file_path)
            metadata.ocr_applied = self.ocr_enabled and self._tesseract_available
        elif file_type in ("txt", "text", "md", "csv", "json"):
            text = self._load_text(file_path)
        else:
            raise DocumentLoadError(f"Unsupported file type: {file_type}")

        logger.info(f"Loaded document: {path.name} ({len(text)} characters)")
        return metadata, text

    def _load_pdf(self, file_path: str) -> Tuple[str, int]:
        """
        Load a PDF document.

        Args:
            file_path: Path to the PDF file

        Returns:
            Tuple of (extracted_text, page_count)
        """
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            page_count = len(reader.pages)

            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

            text = "\n\n".join(text_parts)

            # If PDF has no extractable text and OCR is enabled, try OCR
            if not text.strip() and self.ocr_enabled and self._tesseract_available:
                logger.info("PDF has no extractable text, attempting OCR...")
                text = self._ocr_pdf(file_path)

            return text, page_count

        except ImportError:
            raise DocumentLoadError("pypdf not installed. Cannot load PDF files.")
        except Exception as e:
            raise DocumentLoadError(f"Failed to load PDF: {e}")

    def _ocr_pdf(self, file_path: str) -> str:
        """
        Perform OCR on a PDF document.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text from OCR
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract

            images = convert_from_path(file_path)
            text_parts = []

            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang=self.ocr_language)
                if page_text.strip():
                    text_parts.append(f"--- Page {i + 1} (OCR) ---\n{page_text}")

            return "\n\n".join(text_parts)

        except ImportError:
            logger.warning("pdf2image or pytesseract not installed. OCR unavailable.")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed for PDF: {e}")
            return ""

    def _load_image(self, file_path: str) -> str:
        """
        Load an image and perform OCR if enabled.

        Args:
            file_path: Path to the image file

        Returns:
            Extracted text from OCR or empty string
        """
        if not self.ocr_enabled or not self._tesseract_available:
            logger.warning("OCR disabled or unavailable. Returning empty text for image.")
            return "[Image content - OCR not available]"

        try:
            import pytesseract

            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang=self.ocr_language)
            return text.strip()

        except ImportError:
            logger.warning("pytesseract not installed. OCR unavailable.")
            return "[Image content - OCR not available]"
        except Exception as e:
            logger.warning(f"OCR failed for image: {e}")
            return f"[Image content - OCR failed: {e}]"

    def _load_text(self, file_path: str) -> str:
        """
        Load a text file.

        Args:
            file_path: Path to the text file

        Returns:
            File contents as string
        """
        encodings = ["utf-8", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise DocumentLoadError(f"Failed to decode text file with any supported encoding")
