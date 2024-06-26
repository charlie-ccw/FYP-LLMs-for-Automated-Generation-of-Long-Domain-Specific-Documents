import asyncio
import json
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from globalParameter.parameters import DOMAIN
from prompt.prompt_of_generation_with_template_and_key_info import GENERATION_WITH_TEMPLATE_AND_KEY_INFO_SYSTEM, \
    GENERATION_WITH_TEMPLATE_AND_KEY_INFO_PROMPT
from tools.retrieval_qa_with_llm_and_resort_tool import RetrievalQAWithLLMAndResortTool
from util.chroma_db_util import ChromaDBUtil
from util.prompt_based_generation import aprompt_based_generation


async def generation_version_4(file_name_without_extension: str):
    print(f">>>>>>>>>>>>>>> {file_name_without_extension} <<<<<<<<<<<<<<< version 4")
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
    retrieval_qa_with_llm_and_resort_tool = RetrievalQAWithLLMAndResortTool()
    build_version_4_knowledge_base(file_name_without_extension=file_name_without_extension)

    # Get generation in the past
    if os.path.isfile(f"generated_file/{DOMAIN}/version_4/{file_name_without_extension}.json"):
        with open(f"generated_file/{DOMAIN}/version_4/{file_name_without_extension}.json", 'r', encoding='utf-8') as f:
            file_generation = json.load(f)
    else:
        file_generation = {}

    # Generation start
    for chapter_id, chapter_template in doc_template.items():
        chapter_name = chapter_template["name"]
        sections_template = chapter_template["sections"]
        for section_id, section_template in sections_template.items():
            print(f"{file_name_without_extension} --- {chapter_id} --- {section_id}")
            if section_id in file_generation.keys():
                continue
            section_name = section_template["name"]
            section_description = section_template["description"]

            # Get detailed table for section
            section_key_table = detailed_table[section_id]['detailed_table']
            key_infos = []

            # Set up the retrieval question and start to retrieve key info
            for key_name, key_description in section_key_table.items():
                question = f"find an answer for {key_name}, which has a description: {key_description}"

                # start to retrieve key info
                while 1:
                    try:
                        answer = await retrieval_qa_with_llm_and_resort_tool.acall(
                            question=question,
                            knowledge_base=f"version4/child/{file_name_without_extension}"
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

            # Format the whole messages with input varibales using template
            messages = chat_template.format_messages(section_requirement=section_description, key_info=key_data)

            # Call the model for generating
            while 1:
                try:
                    response = await aprompt_based_generation(prompt=messages, temperature=0.5)
                    break
                except Exception as e:
                    print(e)

            generation = response.content

            # Store the generation with section id and name info
            file_generation[section_id] = {
                'section_name': section_name,
                'generation': generation
            }

            # Write into correct JSON file every section
            with open(f"generated_file/{DOMAIN}/version_4/{file_name_without_extension}.json", 'w', encoding='utf-8') as f:
                json.dump(file_generation, f, indent=4)


def build_version_4_knowledge_base(file_name_without_extension: str):
    if os.path.isdir(f"vectorDB/chromaDB/version4/parent/{file_name_without_extension}"):
        return
    print(f"No knowledge base, start to build one for {file_name_without_extension}")
    # Set up the origin file path
    file_path = os.path.join(f"file/{DOMAIN}/structure_1", f"{file_name_without_extension}.pdf")
    # Create pdfloader to extract info from pdf
    loader = PyMuPDFLoader(file_path)
    # Load context
    datas = loader.load()
    # Format the texts
    text = ""
    for data in datas:
        text += data.page_content

    # Create text splitter for parent and child documents
    parent_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    # Get all parent documents and their child documents
    parent_documents = []
    child_documents = []
    for i, parent_page_content in enumerate(parent_text_splitter.split_text(text=text)):
        parent_documents.append(Document(
            page_content=parent_page_content,
            metadata={
                'source': file_name_without_extension,
                'index': i
            }
        ))
        for j, child_page_content in enumerate(child_text_splitter.split_text(text=parent_page_content)):
            child_documents.append(Document(
                page_content=child_page_content,
                metadata={
                    'source': file_name_without_extension,
                    'index': j,
                    'parent_index': i
                }
            ))

    # Set up the knowledge for parent and child documents of version4
    chroma_db_util = ChromaDBUtil()
    chroma_db_util.initialise_vectorstore_with_documents(persist_directory=f"version4/parent/{file_name_without_extension}",
                                                         documents=parent_documents)
    chroma_db_util.initialise_vectorstore_with_documents(persist_directory=f"version4/child/{file_name_without_extension}",
                                                         documents=child_documents)


async def main():
    os.makedirs(f"generated_file/{DOMAIN}/version_4", exist_ok=True)
    # Load  the train and test files with summary
    train_test_file_path = f"project_template/template_version_1_{DOMAIN}_summary.json"
    with open(train_test_file_path, 'r', encoding='utf-8') as f:
        train_test_datasets = json.load(f)

    # Get the test files with summary
    test_datasets = train_test_datasets['test']

    # Asynchronous call for generating tasks for 1 files each time
    files_list = []
    for idx, test_dataset in enumerate(test_datasets):
        file_name_without_extension = test_dataset['file_name']
        files_list.append(file_name_without_extension)

        if len(files_list) == 5:
            tasks = [generation_version_4(file_name) for file_name in files_list]
            await asyncio.gather(*tasks)
            files_list = []

    # Process any remaining files
    if files_list:
        tasks = [generation_version_4(file_name) for file_name in files_list]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
