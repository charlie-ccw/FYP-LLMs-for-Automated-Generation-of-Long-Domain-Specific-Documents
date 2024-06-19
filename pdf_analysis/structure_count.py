"""
Since historical PDF documents cannot be 100% matched with the current target template,
we need to analyze the structure of the historical PDF documents to determine the final generated template.
"""
import json
import os

from globalParameter.parameters import DOMAIN
from pdf_analysis.utils import extract_text_from_pdf, extract_structure_from_text

# Set the folder we want to process
folder_70 = f"../file/FYP_Golden_Documents/{DOMAIN}/70"
folder_80 = f"../file/FYP_Golden_Documents/{DOMAIN}/80"
# Get all files
files = os.listdir(folder_70)
files += os.listdir(folder_80)

structures = {}
for file in files:
    # get the file path and extract contents text from each file
    file = os.path.join(f'../file/FYP_Documents/{DOMAIN}', file)
    text = extract_text_from_pdf(file, 1)
    text += extract_text_from_pdf(file, 2)
    text += extract_text_from_pdf(file, 3)

    structure, _ = extract_structure_from_text(text, only_sections=False)

    # Start iteration and count the structure info
    for key, value in structure.items():
        if value['section_name'] in structures.keys():
            structures[value['section_name']] += 1
        else:
            structures[value['section_name']] = 1

# Store the structure info
with open('analysis_json_data/structure_info.json', 'w', encoding='utf-8') as f:
    json.dump(structures, f, indent=4)

