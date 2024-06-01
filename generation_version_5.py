import asyncio
import json
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.key_info_retrieval_agent import get_key_info_retrieval_agent
from prompt.prompt_of_generation_with_template_and_key_info import GENERATION_WITH_TEMPLATE_AND_KEY_INFO_SYSTEM, \
    GENERATION_WITH_TEMPLATE_AND_KEY_INFO_PROMPT
from util.chroma_db_util import ChromaDBUtil
from util.prompt_based_generation import aprompt_based_generation


async def generation_version_5(file_name_without_extension: str):
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

    # Build the Agent for key info summary task
    key_info_retrieval_agent = get_key_info_retrieval_agent()

    # Build the knowledge base for core info retrieval
    build_version_5_knowledge_base(file_name_without_extension=file_name_without_extension)

    # Get generation in the past
    if os.path.isfile(f"generated_file/version_5/{file_name_without_extension}.json"):
        with open(f"generated_file/version_5/{file_name_without_extension}.json", 'r', encoding='utf-8') as f:
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
            section_name = section_template['name']
            section_description = section_template['description']

            # Build the agent input prompt
            agent_input_prompt = f"""Given the Template Requirement, your job is to collect core info for current section and return a summary of core infos.
Template Requirement:
{section_description}

NOTE: You need to extract key questions as many as possible and use tool to do retrieval question and answering for getting full key infos. There is only one knowledge base: 'version5/child/{file_name_without_extension}'"""

            # Get section key info using Agent
            while 1:
                try:
                    agent_result = await key_info_retrieval_agent.ainvoke({'input': agent_input_prompt})
                    section_key_info = agent_result['output']
                    break
                except Exception as e:
                    print(e)

            # Format the whole messages with input varibales using template
            messages = chat_template.format_messages(section_requirement=section_description, key_info=section_key_info)

            # Call the model for generating
            while 1:
                try:
                    response = await aprompt_based_generation(prompt=messages, model='gpt-3.5-turbo', temperature=0.5)
                    break
                except Exception as e:
                    print(e)

            generation = response.content

            # Store the generation with section id and name info
            file_generation[section_id] = {
                'section_name': section_name,
                'genetation': generation
            }

            # Write into correct JSON file every section
            with open(f"generated_file/version_5/{file_name_without_extension}.json", 'w', encoding='utf-8') as f:
                json.dump(file_generation, f, indent=4)



def build_version_5_knowledge_base(file_name_without_extension: str):
    if os.path.isdir(f"vectorDB/chromaDB/version5/parent/{file_name_without_extension}"):
        return
    print(f"No knowledge base, start to build one for {file_name_without_extension}")
    # Set up the origin file path
    file_path = os.path.join("file/Energy_demand/structure_1", f"{file_name_without_extension}.pdf")
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
    chroma_db_util.initialise_vectorstore_with_documents(persist_directory=f"version5/parent/{file_name_without_extension}",
                                                         documents=parent_documents)
    chroma_db_util.initialise_vectorstore_with_documents(persist_directory=f"version5/child/{file_name_without_extension}",
                                                         documents=child_documents)


async def main():
    # Load  the train and test files with summary
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

        if len(files_list) == 5:
            tasks = [generation_version_5(file_name) for file_name in files_list]
            await asyncio.gather(*tasks)
            files_list = []

    # Process any remaining files
    if files_list:
        tasks = [generation_version_5(file_name) for file_name in files_list]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())























