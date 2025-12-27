import pdfplumber
import docx
import pytesseract
from PIL import Image, UnidentifiedImageError
import magic     # Using this for MIME sniffing
import io

def extract_text_from_pdf(file_bytes):
    try:
        with pdfplumber.open(file_bytes) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception:
        raise ValueError("Failed to read PDF file")

def extract_text_from_docx(file_bytes):
    try:
        doc = docx.Document(file_bytes)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        raise ValueError("Failed to read DOCX file")

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.read().decode("utf-8", errors="ignore")
    except Exception:
        raise ValueError("Failed to read TXT file")

def extract_text_from_image(file_bytes):
    try:
        image = Image.open(file_bytes)
        return pytesseract.image_to_string(image)
    except UnidentifiedImageError:
        raise ValueError("Invalid image file")
    except Exception:
        raise ValueError("Failed to extract text via OCR")

def extract_text_from_file(upload_file):
    """
    Detect file type using BOTH:
    - MIME-Type sniffing (magic library)
    - filename extension 
    
    Extract text accordingly.
    """

    filename = upload_file.filename.lower()
    raw_bytes = upload_file.file.read()
    file_bytes = io.BytesIO(raw_bytes)
    file_bytes.seek(0)

    try:
        mime_type = magic.from_buffer(raw_bytes, mime=True)
    except:
        mime_type = None

    if mime_type:
        # PDF
        if mime_type == "application/pdf":
            return extract_text_from_pdf(file_bytes)
        # DOCX
        if mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            return extract_text_from_docx(file_bytes)
        # Plain text
        if mime_type.startswith("text/"):
            return extract_text_from_txt(io.BytesIO(raw_bytes))
        # Images
        if mime_type.startswith("image/"):
            return extract_text_from_image(io.BytesIO(raw_bytes))

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)

    if filename.endswith(".docx"):
        return extract_text_from_docx(file_bytes)

    if filename.endswith(".txt"):
        return extract_text_from_txt(io.BytesIO(raw_bytes))

    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        return extract_text_from_image(io.BytesIO(raw_bytes))

    raise ValueError(f"Unsupported file format: {filename}")

