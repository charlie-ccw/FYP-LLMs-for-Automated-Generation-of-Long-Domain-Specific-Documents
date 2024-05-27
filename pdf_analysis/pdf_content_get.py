"""
Through our attempts, we discovered that some PDF files contain incorrect table of contents.
Therefore, this code aims to obtain the correct and complete table of contents sequence and store it.
Note:
    Due to errors in the PDF files themselves, some parts of the table of contents may not be found.
    Manual correction based on the output printed by the program may be required.
"""
import os
import json
from pdf_analysis.utils import extract_structure_from_text, extract_text_from_pdf, get_pdf_page_count

# Set the file folder path
folder = "../file/Energy_demand/structure_1"
# Get all files inside file folder
files = os.listdir(folder)

# Iterate the files
for file in files:
    print(file)
    # Format the file path
    file_path = os.path.join(folder, file)
    # Get total num of pages of file
    total_page_num = get_pdf_page_count(pdf_path=file_path)

    # Get the content structure (this may be correct structure with wrong page number)
    content_text = ""
    content_text += extract_text_from_pdf(pdf_path=file_path, page_num=1)
    content_text += extract_text_from_pdf(pdf_path=file_path, page_num=2)
    structure_tmp, order = extract_structure_from_text(text=content_text, only_sections=True)

    # Start to build the structure with correct page number
    structure_final = {}
    current_page = int(structure_tmp[order[0]]['page_number'])
    # Check the page number
    if current_page < 3:
        print(f"+++++++++++++++ Please check the whole json of this pdf file")
    last_page_find = current_page

    # Iterate the sections
    for section_id in order:
        section_name = section_id + ' ' + structure_tmp[section_id]['section_name']
        find_page_number = False
        # start to find the correct page number
        while not find_page_number:
            # Check if we fing the last page
            if current_page > total_page_num:
                print(f">>>>>>>>>>>>>>> Current Section ID: {section_id}. Cannot find the page number of this section")
                current_page = last_page_find
                break
            # Get the text of current page
            text = extract_text_from_pdf(pdf_path=file_path, page_num=current_page-1)
            # Check if current page is the start of current section
            if section_name in text:
                find_page_number = True
                # Update the last page we find
                last_page_find = current_page
            else:
                current_page += 1
        # If we find the start page number of current page
        if find_page_number:
            # store the content info into structure with correct page number
            structure_final[section_id] = {
                "section_id": section_id,
                "section_name": structure_tmp[section_id]['section_name'],
                "page_number": str(current_page)
            }
        else:
            # store the content info into structure with None page number
            structure_final[section_id] = {
                "section_id": section_id,
                "section_name": structure_tmp[section_id]['section_name'],
                "page_number": ""
            }

    # Add END tag for extraction
    structure_final["END"] = {
        "section_id": "END",
        "section_name": "END",
        "page_number": ""
    }
    # Store the correct content for current file
    file_name_without_extension = os.path.splitext(file)[0]
    with open(f'../file/Energy_demand_content/structure_1/{file_name_without_extension}.json', 'w', encoding='utf-8') as f:
        json.dump(structure_final, f, indent=4)

