import json
import os


def check_generated_structure(file: str):
    with open("project_template/template_version_1.json", 'r', encoding='utf-8') as f:
        doc_template = json.load(f)

    sections = []
    section_names = []
    for chapter_id, chapter_template in doc_template.items():
        sections_template = chapter_template["sections"]
        for section_id, section_template in sections_template.items():
            section_name = section_template['name']
            sections.append(section_id)
            section_names.append(section_name)

    with open(file, 'r', encoding='utf-8') as f:
        generated_sections = json.load(f)

    for section_id, generated_section in generated_sections.items():
        sections.remove(section_id)
        section_names.remove(generated_section['section_name'])
        if not isinstance(generated_section['generation'], str):
            print(f"{file} ---------- {section_id} NOT STRING")

    if len(sections) > 0 or len(section_names) > 0:
        print(f"{file} ---------- HAVEN NOT DONE")



generation_folder = "generated_file"
generation_version_folders = os.listdir(generation_folder)
for version in generation_version_folders:
    generated_files = os.listdir(os.path.join(generation_folder, version))
    if len(generated_files) != 25:
        print(f"---------- {version} ---------- HAVEN NOT DONE")
    for generated_file in generated_files:
        check_generated_structure(os.path.join(generation_folder, version, generated_file))