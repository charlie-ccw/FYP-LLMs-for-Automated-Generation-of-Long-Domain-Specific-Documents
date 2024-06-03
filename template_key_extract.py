import json
import os
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

from prompt.prompt_of_template_to_key_table import TEMPLATE_TO_KEY_TABLE_SYSTEM, TEMPLATE_TO_KEY_TABLE_PROMPT
from util.prompt_based_generation import prompt_based_generation


chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=TEMPLATE_TO_KEY_TABLE_SYSTEM),
        HumanMessagePromptTemplate.from_template(TEMPLATE_TO_KEY_TABLE_PROMPT),
    ]
)

template_file_path = "project_template/template_version_1.json"
with open(template_file_path, 'r', encoding='utf-8') as f:
    doc_template = json.load(f)


detailed_table = {}
for chapter_id, chapter_template in doc_template.items():
    chapter_name = chapter_template["name"]
    sections_template = chapter_template["sections"]
    for section_id, section_template in sections_template.items():
        section_name = section_template["name"]
        section_description = section_template["description"]

        messages = chat_template.format_messages(template_requirements=section_description)

        response = prompt_based_generation(prompt=messages, model='gpt-4o', temperature=0.5, json_format=True)

        if isinstance(response, dict):
            detailed_table[section_id] = {
                'section_name': section_name,
                'detailed_table': response
            }
        else:
            detailed_table[section_id] = {
                'section_name': section_name,
                'detailed_table': None
            }
            print(f">>>>>>>>>>>>>>> ERROR for section '{section_id}' <<<<<<<<<<<<<<<")


file_name_without_extension = os.path.splitext(os.path.basename(template_file_path))[0]
with open(f"project_template/{file_name_without_extension}_detailed_table.json", 'w', encoding='utf-8') as f:
    json.dump(detailed_table, f, indent=4)


