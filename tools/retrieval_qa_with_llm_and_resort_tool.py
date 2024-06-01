"""
This tool can help you answer relevant questions by retrieving from an external knowledge base.
"""
import asyncio
from collections import defaultdict
from typing import Type, Any, Union
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document

from globalParameter.parameters import MODEL
from prompt.prompt_of_contextual_compression import CONTEXTUAL_COMPRESSION_SYSTEM, CONTEXTUAL_COMPRESSION_PROMPT
from prompt.prompt_of_multi_query import MULTI_QUERY_SYSTEM, MULTI_QUERY_PROMPT
from prompt.prompt_of_retrieval_qa_tool import RETRIEVAL_QA_TOOL_PROMPT, RETRIEVAL_QA_TOOL_SYSTEM
from tools.retrieval_tool import RetrievalTool
from util.chroma_db_util import ChromaDBUtil
from util.prompt_based_generation import prompt_based_generation, aprompt_based_generation


class RetrievalQAWithLLMAndResortInput(BaseModel):
    """Pydantic model for RetrievalQATool"""

    question: str = Field(description="should be a question to be answered")
    knowledge_base: str = Field(description="should be a name of knowledge base for retrieving extra info")


class RetrievalQAWithLLMAndResortTool(BaseTool):
    """
    A tool to retrieve relevant documents and answer the question based on the info retrieved.
    """

    name = "RetrievalQAWithLLMAndResortTool"
    description = "useful for when you need to answer a question based on extra information retrieved from a specified knowledge base"
    args_schema: Type[BaseModel] = RetrievalQAWithLLMAndResortInput

    @staticmethod
    def get_common_documents(documents_list: Union[list[list[Document]], Any]) -> list[Document]:
        # Get the total list number
        list_num = len(documents_list)

        # Initialise a variable for page_content count
        count_dict = defaultdict(int)
        # Initialise a variable for storing the link between page_content and object
        objects_dict = {}

        # Start iteration and count the page_content
        for documents in documents_list:
            # Use a set to avoid counting duplicates within the same sublist.
            unique_objects = {document.page_content: document for document in documents}
            for page_content, obj in unique_objects.items():
                count_dict[page_content] += 1
                objects_dict[page_content] = obj

        # Find the common objects
        common_objects = [objects_dict[page_content] for page_content, count in count_dict.items()
                          if count >= list_num / 2]

        return common_objects

    @staticmethod
    def reciprocal_rank_fusion_calculation(doc: Document, documents_list: list[list[Document]]) -> float:
        # Initialise the rrf score
        rrf_score = 0.0

        # Start iteration and get score of each sub list
        for sublist in documents_list:
            # rrf of sub list calculation
            for rank, sub_doc in enumerate(sublist):
                # Check if we find the correct document in sub list
                if sub_doc.metadata['index'] == doc.metadata['index']:
                    # Calculate rrf of sub lis, RRF formula = 1 / (k + rank), here k = 1
                    rrf_score += 1 / (1 + rank)
                    break

        return rrf_score

    def calculate_parents_importance(self, common_documents: list[Document],
                                     documents_list: Union[list[list[Document]], Any]) -> dict:
        # Initialise the dict to store the importance of parent document
        importance_dict = {}

        # Start iteration and calculate importance of parent document
        for doc in common_documents:
            parent_index = doc.metadata.get('parent_index')
            parent_key = (doc.metadata['source'], parent_index)

            # Initialise the importance of parent document
            if parent_key not in importance_dict:
                importance_dict[parent_key] = 0

            # Calculate the importance and add it to specific parent document
            importance = self.reciprocal_rank_fusion_calculation(doc=doc, documents_list=documents_list)
            importance_dict[parent_key] += importance

        return importance_dict

    @staticmethod
    def get_multi_query(question: str) -> list[str]:
        # Set up the message template for multi query generation
        multi_query_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=MULTI_QUERY_SYSTEM),
                HumanMessagePromptTemplate.from_template(MULTI_QUERY_PROMPT),
            ]
        )
        # Build the whole multi query prompt
        multi_query_prompt = multi_query_template.format_messages(question=question)
        # Get multi queries using llm
        response = prompt_based_generation(prompt=multi_query_prompt, model=MODEL,
                                           temperature=0.5, json_format=True)
        # Format multi queries
        multi_query = []
        for query_key, query in response.items():
            multi_query.append(query)

        return multi_query

    @staticmethod
    def get_sorted_parent_documents(knowledge_base: str, parent_documents_with_importance: dict) -> list[Document]:
        # Set the chromaDB util to get all parent documents
        chroma_db = ChromaDBUtil().load_vectorstore(persist_directory=knowledge_base.replace("child", "parent"))
        parent_documents = []
        for (source, parent_index), importance in parent_documents_with_importance.items():
            results = chroma_db.get(where={"$and": [{'index': parent_index},
                                                    {'source': source}]})
            for i, metadata in enumerate(results['metadatas']):
                document = Document(page_content=results['documents'][i], metadata=metadata)
                parent_documents.append((document, importance))

        # Resort the parent document using their importance score
        sorted_parent_documents = [doc for doc, _ in sorted(parent_documents, key=lambda x: x[1], reverse=True)]

        return sorted_parent_documents

    @staticmethod
    def get_compressed_parent_documents(sorted_parent_documents: list[Document],
                                        question: str) -> list[Document]:
        # Set up the message template for contextual compression
        multi_query_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=CONTEXTUAL_COMPRESSION_SYSTEM),
                HumanMessagePromptTemplate.from_template(CONTEXTUAL_COMPRESSION_PROMPT),
            ]
        )
        # Start iteration and compress parent document using llm
        compressed_parent_documents = []
        for parent_document in sorted_parent_documents:
            # Build the whole multi query prompt
            multi_query_prompt = multi_query_template.format_messages(question=question,
                                                                      context=parent_document.page_content)
            # Get compressed content using llm
            compressed_doc_content = prompt_based_generation(prompt=multi_query_prompt,
                                                             model=MODEL,
                                                             temperature=0.5)

            # Check if llm find any relevant context
            if not compressed_doc_content.content.lower() == 'NO OUTPUT STRING'.lower():
                # Store compressed parent document
                compressed_parent_documents.append(Document(page_content=compressed_doc_content.content,
                                                            metadata=parent_document.metadata))
        return compressed_parent_documents

    @staticmethod
    async def aget_compressed_parent_documents(sorted_parent_documents: list[Document],
                                               question: str) -> list[Document]:
        # Set up the message template for contextual compression
        multi_query_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=CONTEXTUAL_COMPRESSION_SYSTEM),
                HumanMessagePromptTemplate.from_template(CONTEXTUAL_COMPRESSION_PROMPT),
            ]
        )

        compressed_parent_documents = []
        # Build the whole multi query prompt and Get compressed content using llm
        tasks = [aprompt_based_generation(prompt=multi_query_template.format_messages(question=question,
                                                                                      context=parent_document.page_content),
                                          model=MODEL,
                                          temperature=0.5) for parent_document in sorted_parent_documents]
        compressed_doc_contents = await asyncio.gather(*tasks)

        for i, compressed_doc_content in enumerate(compressed_doc_contents):
            # Check if llm find any relevant context
            if not compressed_doc_content.content.lower() == 'NO OUTPUT STRING'.lower():
                # Store compressed parent document
                compressed_parent_documents.append(Document(page_content=compressed_doc_content.content,
                                                            metadata=sorted_parent_documents[i].metadata))
        return compressed_parent_documents

    @staticmethod
    def reorder_documents(compressed_parent_documents: list[Document]) -> list[Document]:
        # Reverse the documents list
        compressed_parent_documents.reverse()
        # Initialize an empty list to store the reordered result
        reordered_parent_documents = []

        # Iterate over the reversed list of documents
        for i, value in enumerate(compressed_parent_documents):
            # If the index is odd, append the document to the end of reordered_result
            if i % 2 == 1:
                reordered_parent_documents.append(value)
            # If the index is even, insert the document at the beginning of reordered_result
            else:
                reordered_parent_documents.insert(0, value)

        return reordered_parent_documents

    def call(self, question: str, knowledge_base: str, k_num: int = 20, *args: Any, **kwargs: Any) -> Any:
        """
        call the _run() from outside
        """
        return self._run(question=question,
                         knowledge_base=knowledge_base,
                         k_num=k_num,
                         *args,
                         **kwargs)

    def _run(self, question: str, knowledge_base: str, k_num: int = 20,
             *args: Any, **kwargs: Any) -> Any:
        """
        Use the tool.
        To answer a question by retrieving extra knowledge
        """
        # Multi Query Set Up
        multi_query = self.get_multi_query(question=question)

        # Get the Retrival Tool to Extract document objects from knowledge base
        retrieval_tool = RetrievalTool()
        # Initialise the documents list and retrieve relevent documents for each query
        documents_list = []
        for query in multi_query:
            documents_list.append(retrieval_tool.call(query=query,
                                                      knowledge_base=knowledge_base,
                                                      k_num=k_num))

        # Get common documents using documents_list
        common_documents = self.get_common_documents(documents_list=documents_list)
        # Calculate Parent documents with their importance
        parent_documents_with_importance = self.calculate_parents_importance(common_documents=common_documents,
                                                                             documents_list=documents_list)

        # Get All Parent Documents and Resort Them Using RRF Algorithm
        sorted_parent_documents = self.get_sorted_parent_documents(knowledge_base=knowledge_base,
                                                                   parent_documents_with_importance=parent_documents_with_importance)

        # Contextual Compression for Parent Documents
        compressed_parent_documents = self.get_compressed_parent_documents(sorted_parent_documents=sorted_parent_documents,
                                                                           question=question)

        # Long-Context Reorder
        reordered_parent_documents = self.reorder_documents(compressed_parent_documents=compressed_parent_documents)

        # Format the relevant text for question answer
        documents_text = self.format_documents_info(documents=reordered_parent_documents)

        # Get Prompts and Model output mode, You can pass in custom template content through kwargs
        human_prompt = kwargs.get('prompt', RETRIEVAL_QA_TOOL_PROMPT)
        prompt_parameter = kwargs.get('prompt_parameter', {'question': question, 'summaries': documents_text})
        json_format = kwargs.get('json_format', True)

        # Set up the message template
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=RETRIEVAL_QA_TOOL_SYSTEM),
                HumanMessagePromptTemplate.from_template(human_prompt),
            ]
        )
        # Format the whole Message Prompt using input variables
        prompt = chat_template.format_messages(**prompt_parameter)
        # Call the Model API for Queation Answer Task
        response = prompt_based_generation(prompt=prompt, model=MODEL, temperature=0.5,
                                           json_format=json_format)
        return response

    async def acall(self, question: str, knowledge_base: str, k_num: int = 20, *args: Any,
                    **kwargs: Any) -> Any:
        """
        call the _arun() from outside in Async way
        """
        return await self._arun(question=question,
                                knowledge_base=knowledge_base,
                                k_num=k_num,
                                *args,
                                **kwargs)

    async def _arun(self, question: str, knowledge_base: str, k_num: int = 20,
                    *args: Any, **kwargs: Any) -> Any:
        """Use the tool asynchronously."""
        # Multi Query Set Up
        multi_query = self.get_multi_query(question=question)

        # Get the Retrival Tool to Extract document objects from knowledge base
        retrieval_tool = RetrievalTool()
        # Retrieve relevent documents for each query in Async way
        tasks = [retrieval_tool.acall(query=query,
                                      knowledge_base=knowledge_base,
                                      k_num=k_num,
                                      score_threshold=0.6) for query in multi_query]
        documents_list = await asyncio.gather(*tasks)

        # Get common documents using documents_list
        common_documents = self.get_common_documents(documents_list=documents_list)
        # Calculate Parent documents with their importance
        parent_documents_with_importance = self.calculate_parents_importance(common_documents=common_documents,
                                                                             documents_list=documents_list)

        # Get All Parent Documents and Resort Them Using RRF Algorithm
        sorted_parent_documents = self.get_sorted_parent_documents(knowledge_base=knowledge_base,
                                                                   parent_documents_with_importance=parent_documents_with_importance)

        # Contextual Compression for Parent Documents
        compressed_parent_documents = await self.aget_compressed_parent_documents(
            sorted_parent_documents=sorted_parent_documents,
            question=question)

        # Long-Context Reorder
        reordered_parent_documents = self.reorder_documents(compressed_parent_documents=compressed_parent_documents)

        # Format the relevant text for question answer
        documents_text = self.format_documents_info(documents=reordered_parent_documents)

        # Get Prompts and Model output mode, You can pass in custom template content through kwargs
        human_prompt = kwargs.get('prompt', RETRIEVAL_QA_TOOL_PROMPT)
        prompt_parameter = kwargs.get('prompt_parameter', {'question': question, 'summaries': documents_text})
        json_format = kwargs.get('json_format', True)

        # Set up the message template
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=RETRIEVAL_QA_TOOL_SYSTEM),
                HumanMessagePromptTemplate.from_template(human_prompt),
            ]
        )
        # Format the whole Message Prompt using input variables
        prompt = chat_template.format_messages(**prompt_parameter)
        # Call the Model API for Queation Answer Task in Async way
        response = await aprompt_based_generation(prompt=prompt, model=MODEL, temperature=0.5,
                                                  json_format=json_format)
        return response

    @staticmethod
    def format_documents_info(documents: list[Document]) -> str:
        format_text = ""
        for document in documents:
            format_text += f"Content: {document.page_content}\nSource: {document.metadata['source']}\n"
        return format_text
