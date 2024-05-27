"""
This python file provides some tools needed for PDF file analysis.
"""
import re
import os

import pdfplumber
from pdf2image import convert_from_path


def get_pdf_page_count(pdf_path: str) -> int:
    """
    To Get the total page number of file
    :param pdf_path: file path
    :return: total page number
    """
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def extract_text_from_pdf(pdf_path: str, page_num: int = 1) -> str:
    """
    To extract the text of the specified page from the document
    :param pdf_path: file path
    :param page_num: The page number of the text you want to extract
    :return: text of specified page
    """
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        text += page.extract_text() + '\n'
    return text


def extract_table_from_pdf(pdf_path: str, page_num: int = 1) -> list:
    """
    To extract the table of the specified page from the document.
    :param pdf_path: file path
    :param page_num: The page number of the table you want to extract
    :return: table of specidied page
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        table = page.extract_table()
    return table


def extract_structure_from_text(text: str, only_sections: bool = False):
    """
    Retrieve the corresponding chapter content using the regex method
    :param text: The text you want to retrieve
    :param only_sections: True, if only sections needed; Flase, if chapter and section together
    :return:
    """
    # Set the regx pattern and get matches
    structure = {}
    order = []
    pattern = r"(\d+(?:\.\d+)?)\s+(.+?)\.{2,}\s+(\d+)"
    matches = re.findall(pattern, text)

    # Start iteration and get correct structure info with correct order info
    for i, match in enumerate(matches):
        section_id, section_name, page_number = match
        section_name = section_name.rstrip()
        if only_sections and section_id.isdigit():
            continue
        order.append(section_id)
        structure[section_id] = {
            "section_id": section_id,
            "section_name": section_name,
            "page_number": page_number
        }

    return structure, order


def pdf_to_images(pdf_path: str, output_folder: str, dpi: int = 300):
    """
    Convert each single page of PDF into pictures and store into output folder
    :param pdf_path: the file you want to convert
    :param output_folder: the output folder to store pictures
    :param dpi: the higher the dpi, the clear the picture
    :return: None
    """
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = os.path.join(output_folder, pdf_filename)
    os.makedirs(output_folder, exist_ok=True)

    # covert pdf to pages in picture format
    pages = convert_from_path(pdf_path, dpi)
    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f"page_{i+1}.jpg")
        page.save(image_path, "JPEG")


