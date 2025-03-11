import os
from PyPDF2 import PdfReader
from ebooklib import epub
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

def read_pdf(file_path):
    """
    Reads a PDF file and extracts its text content.
    :param file_path: Path to the PDF file.
    :return: Extracted text as a single string.
    """
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages)

def read_epub(file_path):
    """
    Reads an EPUB file and extracts its text content.
    :param file_path: Path to the EPUB file.
    :return: Extracted text as a single string.
    """
    book = epub.read_epub(file_path, options={'ignore_ncx': True}) # Set `ignore_ncx` to avoid the warning

    text_content = []
    for item in book.items:
        if isinstance(item, epub.EpubHtml):
            soup = BeautifulSoup(item.get_body_content(), 'html.parser')
            text_content.append(soup.get_text())
    return "\n".join(text_content)


def read_xml(file_path):
    """
    Reads an XML file and extracts its text content.
    :param file_path: Path to the XML file.
    :return: Extracted text as a single string.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    def recursively_extract_text(element):
        texts = []
        if element.text:
            texts.append(element.text.strip())
        for child in element:
            texts.extend(recursively_extract_text(child))
        if element.tail:
            texts.append(element.tail.strip())
        return texts

    text_content = recursively_extract_text(root)
    return "\n".join(text_content)

def ingest_file(file_path):
    """
    Determines the file type and extracts text content accordingly.
    :param file_path: Path to the file (PDF, EPUB, or XML).
    :return: Extracted text as a single string.
    """
    if file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.epub'):
        return read_epub(file_path)
    elif file_path.endswith('.xml'):
        return read_xml(file_path)
    else:
        raise ValueError(f"Unsupported file type: {os.path.splitext(file_path)[1]}")
