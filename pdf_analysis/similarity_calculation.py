"""
The dataset includes various domains, and it is necessary to determine which domain's data is most suitable for research.
Therefore, this program aims to obtain basic information about the files in each domain
and perform similarity analysis with the target template.
"""
import copy
import re
import os
import shutil
import json

from pdf_analysis.utils import extract_text_from_pdf

# Initialise the Section similarity score dict
section_similarity = {
    "80": 0,
    "70": 0,
    "60": 0,
    "50": 0,
    "sum": 0,
    "count": 0,
}

# This is our target template structure
sections_info = {
    "PROJECT DETAILS": [1, 0, 0],
    "Summary Description of the Project": [1.1, 0, 0],
    "Audit History": [1.2, 0, 0],
    "Sectoral Scope and Project Type": [1.3, 0, 0],
    "Project Eligibility": [1.4, 0, 0],
    "Project Design": [1.5, 0, 0],
    "Project Proponent": [1.6, 0, 0],
    "Other Entities Involved in the Project": [1.7, 0, 0],
    "Ownership": [1.8, 0, 0],
    "Project Start Date": [1.9, 0, 0],
    "Project Crediting Period": [1.10, 0, 0],
    "Project Scale and Estimated GHG Emission Reductions or Removals": [1.11, 0, 0],
    "Description of the Project Activity": [1.12, 0, 0],
    "Project Location": [1.13, 0, 0],
    "Conditions Prior to Project Initiation": [1.14, 0, 0],
    "Compliance with Laws, Statutes and Other Regulatory Frameworks": [1.15, 0, 0],
    "Double Counting and Participation under Other GHG Programs": [1.16, 0, 0],
    "Double Claiming, Other Forms of Credit, and Scope 3 Emissions": [1.17, 0, 0],
    "Sustainable Development Contributions": [1.18, 0, 0],
    "Additional Information Relevant to the Project": [1.19, 0, 0],
    "SAFEGUARDS AND STAKEHOLDER ENGAGEMENT": [2, 0, 0],
    "Stakeholder Engagement and Consultation": [2.1, 0, 0],
    "Risks to Stakeholders and the Environment": [2.2, 0, 0],
    "Respect for Human Rights and Equity": [2.3, 0, 0],
    "Ecosystem Health": [2.4, 0, 0],
    "APPLICATION OF METHODOLOGY": [3, 0, 0],
    "Title and Reference of Methodology": [3.1, 0, 0],
    "Applicability of Methodology": [3.2, 0, 0],
    "Project Boundary": [3.3, 0, 0],
    "Baseline Scenario": [3.4, 0, 0],
    "Additionality": [3.5, 0, 0],
    "Methodology Deviations": [3.6, 0, 0],
    "QUANTIFICATION OF ESTIMATED GHG EMISSION REDUCTIONS AND REMOVALS": [4, 0, 0],
    "Baseline Emissions": [4.1, 0, 0],
    "Project Emissions": [4.2, 0, 0],
    "Leakage Emissions": [4.3, 0, 0],
    "Estimated GHG Emission Reductions and Carbon Dioxide Removals": [4.4, 0, 0],
    "MONITORING": [5, 0, 0],
    "Data and Parameters Available at Validation": [5.1, 0, 0],
    "Data and Parameters Monitored": [5.2, 0, 0],
    "Monitoring Plan": [5.3, 0, 0],
}

domain_info = {}
domain_similarity = {}

# Set the folder which contains files of all domains
base_path = '../file/FYP_Documents'
# Get all the domains' folder
folders = os.listdir(base_path)

# Start iteration and get info analysis for each domain
for folder in folders:
    print(folder)
    # Format the domain folder path
    folder_path = os.path.join(base_path, folder)
    # Get all PDF files of current domain
    files = os.listdir(folder_path)

    # initialise the domian info and similarity analysis for current domain
    domain_info[folder] = copy.deepcopy(sections_info)
    domain_similarity[folder] = copy.deepcopy(section_similarity)

    # Start the file iteration for current domain
    for file in files:
        # get the file path and extract contents text from each file
        file_path = os.path.join(folder_path, file)
        extracted_text1 = extract_text_from_pdf(file_path, 1)
        extracted_text2 = extract_text_from_pdf(file_path, 2)
        extracted_text3 = extract_text_from_pdf(file_path, 3)
        extracted_text = extracted_text1 + extracted_text2 + extracted_text3

        pattern = r"(\d+(?:\.\d+)?)\s+(.+?)\.{2,}\s+(\d+)"
        matches = re.findall(pattern, extracted_text)

        # for each matched section, analysis the similarity and info data
        similarity = 0
        for i, match in enumerate(matches[:-1]):
            # Get matched section id, name and page number
            section_id, section_name, page_number = match
            # Format the section name
            section_name = section_name.rstrip()
            # calculate the similarity score (0-100) and store the domain info (total page numbers and total count)
            if section_name in sections_info.keys():
                similarity += 2.5
                domain_info[folder][section_name][1] += int(matches[i + 1][2]) - int(page_number) + 1
                domain_info[folder][section_name][2] += 1

        # Classify the files based on the similarity score.
        if similarity >= 80:
            destination_file = os.path.join('file/FYP_Golden_Documents', folder, '80', file)
            shutil.copy(file_path, destination_file)
            domain_similarity[folder]['80'] += 1

        if similarity >= 70:
            destination_file = os.path.join('file/FYP_Golden_Documents', folder, '70', file)
            shutil.copy(file_path, destination_file)
            domain_similarity[folder]['70'] += 1

        if similarity >= 60:
            destination_file = os.path.join('file/FYP_Golden_Documents', folder, '60', file)
            shutil.copy(file_path, destination_file)
            domain_similarity[folder]['60'] += 1

        if similarity >= 50:
            destination_file = os.path.join('file/FYP_Golden_Documents', folder, '50', file)
            shutil.copy(file_path, destination_file)
            domain_similarity[folder]['50'] += 1

        # Calculate the total similarity score for current domain with file number count
        domain_similarity[folder]['sum'] += similarity
        domain_similarity[folder]['count'] += 1

# Store the domain info
with open('analysis_json_data/domain_info.json', 'w') as file:
    json.dump(domain_info, file, indent=4)

# Store the similarity analysis
with open('analysis_json_data/domain_similarity.json', 'w') as file:
    json.dump(domain_similarity, file, indent=4)


