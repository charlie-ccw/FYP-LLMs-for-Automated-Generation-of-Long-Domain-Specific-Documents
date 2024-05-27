"""
This is version 1 of the system, which generates overall documents by using the analyzed template requirements and a simple summary of the files.
"""
import json
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from prompt.prompt_of_generation_with_template_only import GENERATION_WITH_TEMPLATE_ONLY_SYSTEM, GENERATION_WITH_TEMPLATE_ONLY_PROMPT
from util.prompt_based_generation import prompt_based_generation

if __name__ == "__main__":
    # Set up the Message Template for generation
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=GENERATION_WITH_TEMPLATE_ONLY_SYSTEM),
            HumanMessagePromptTemplate.from_template(GENERATION_WITH_TEMPLATE_ONLY_PROMPT),
        ]
    )

    # Load the train and test files with summary
    train_test_file_path = "project_template/template_version_1_summary.json"
    with open(train_test_file_path, 'r', encoding='utf-8') as f:
        train_test_datasets = json.load(f)

    # Load the Template structure
    template_file_path = "project_template/template_version_1.json"
    with open(template_file_path, 'r', encoding='utf-8') as f:
        doc_template = json.load(f)

    # Get the test files with summary and generate each one by one
    test_datasets = train_test_datasets['test']
    for idx, test_dataset in enumerate(test_datasets):
        file_name_without_extension = test_dataset['file_name']
        file_summary = test_dataset['summary']
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {file_name_without_extension} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<< {idx}/{len(test_datasets)}")

        # Get the Chapter and section info
        file_generation = {}
        for chapter_id, chapter_template in doc_template.items():
            chapter_name = chapter_template["name"]
            sections_template = chapter_template["sections"]
            for section_id, section_template in sections_template.items():
                section_name = section_template["name"]
                section_description = section_template["description"]

                # Format the whole messages with input variables using message template
                messages = chat_template.format_messages(section_requirement=section_description, knowledge=file_summary)

                # Call the model for generating
                while 1:
                    try:
                        response = prompt_based_generation(prompt=messages, model='gpt-3.5-turbo', temperature=0.5)
                        break
                    except Exception as e:
                        print(e)
                generation = response.content

                # Store the generation with section id and name info
                file_generation[section_id] = {
                    'section_name': section_name,
                    'generation': generation
                }

        # Write into correct JSON file
        with open(f"generated_file/version_1/{file_name_without_extension}.json", 'w', encoding='utf-8') as f:
            json.dump(file_generation, f, indent=4)


