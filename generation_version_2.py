"""
This is version 2 of the system. Building on version 1, it adds an external knowledge base.
During the generation of each chapter, it retrieves the necessary core content
to enhance the overall professionalism and coherence.
"""
import asyncio
import json
import os
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from prompt.prompt_of_generation_with_template_and_key_info import GENERATION_WITH_TEMPLATE_AND_KEY_INFO_SYSTEM, \
    GENERATION_WITH_TEMPLATE_AND_KEY_INFO_PROMPT
from tools.retrieval_qa_tool import RetrievalQATool
from util.chroma_db_util import ChromaDBUtil
from util.prompt_based_generation import prompt_based_generation


async def generation_version_2(file_name_without_extension: str):
    print(f">>>>>>>>>>>>>>> {file_name_without_extension} <<<<<<<<<<<<<<< version 2")
    # Set up the Message Template for generation
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=GENERATION_WITH_TEMPLATE_AND_KEY_INFO_SYSTEM),
            HumanMessagePromptTemplate.from_template(GENERATION_WITH_TEMPLATE_AND_KEY_INFO_PROMPT),
        ]
    )

    # Load the Template structure
    template_file_path = "project_template/template_version_1.json"
    with open(template_file_path, 'r', encoding='utf-8') as f:
        doc_template = json.load(f)

    # Load the detailed table for core info retrieval
    detailed_table_file_path = "project_template/template_version_1_detailed_table.json"
    with open(detailed_table_file_path, 'r', encoding='utf-8') as f:
        detailed_table = json.load(f)

    # Build the knowledge base for core info retrieval
    retrieval_qa_tool = RetrievalQATool()
    target_file_path = os.path.join("file/Energy_demand/structure_1", f"{file_name_without_extension}.pdf")
    chroma_util = ChromaDBUtil()
    chroma_util.initialise_vectorstore_with_files(persist_directory=f"version2/{file_name_without_extension}",
                                                  files=[target_file_path])

    # Generation start
    file_generation = {}
    for chapter_id, chapter_template in doc_template.items():
        chapter_name = chapter_template["name"]
        sections_template = chapter_template["sections"]
        for section_id, section_template in sections_template.items():
            section_name = section_template["name"]
            section_description = section_template["description"]

            # Get detailed table for section
            section_key_table = detailed_table[section_id]['detailed_table']
            key_infos = []

            # Set up the retrieval query and start to retrieve key info
            for key_name, key_description in section_key_table.items():
                question = f"find an answer for {key_name}, which has a description: {key_description}"
                query = f"{key_name} with description: {key_description}"

                # start to retrieve key info
                while 1:
                    try:
                        answer = await retrieval_qa_tool.acall(
                            question=question,
                            query=query,
                            knowledge_base=f"version2/{file_name_without_extension}"
                        )
                        break
                    except Exception as e:
                        print(e)

                # Check if the model can find correct answer
                if answer['find_answer_in_extracted_part'].lower() == 'YES'.lower():
                    key_answer = answer['answer']
                else:
                    key_answer = None

                # Store the key info
                key_infos.append({
                    'key_name': key_name,
                    'key_description': key_description,
                    'key_info/answer': key_answer
                })

            # Format the key info for section
            key_data = ""
            for key_info in key_infos:
                key_data += f"Key_name: {key_info['key_name']}\nKey_info: {key_info['key_info/answer']}\n\n"

            # Format the whole messages with input variables using message template
            messages = chat_template.format_messages(section_requirement=section_description, key_info=key_data)

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
    with open(f"generated_file/version_2/{file_name_without_extension}.json", 'w', encoding='utf-8') as f:
        json.dump(file_generation, f, indent=4)


async def main():
    # Load the train and test files with summary
    train_test_file_path = "project_template/template_version_1_summary.json"
    with open(train_test_file_path, 'r', encoding='utf-8') as f:
        train_test_datasets = json.load(f)

    # Get the test files with summary
    test_datasets = train_test_datasets['test']

    # Asynchronous call for generating tasks for 1 files each time
    files_list = []
    for idx, test_dataset in enumerate(test_datasets):
        file_name_without_extension = test_dataset['file_name']
        files_list.append(file_name_without_extension)

        if len(files_list) == 1:
            tasks = [generation_version_2(file_name) for file_name in files_list]
            await asyncio.gather(*tasks)
            files_list = []

    # Process any remaining files
    if files_list:
        tasks = [generation_version_2(file_name) for file_name in files_list]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())

