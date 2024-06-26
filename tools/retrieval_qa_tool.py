"""
This tool can help you answer relevant questions by retrieving from an external knowledge base.
"""
from typing import Type, Any
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document

from globalParameter.parameters import MODEL
from prompt.prompt_of_retrieval_qa_tool import RETRIEVAL_QA_TOOL_PROMPT, RETRIEVAL_QA_TOOL_SYSTEM
from tools.retrieval_tool import RetrievalTool
from util.prompt_based_generation import prompt_based_generation, aprompt_based_generation


class RetrievalQAInput(BaseModel):
    """Pydantic model for RetrievalQATool"""

    question: str = Field(description="should be a question to be answered")
    query: str = Field(description="should be a search query to get extra info for answering the question")
    knowledge_base: str = Field(description="should be a name of knowledge base for retrieving extra info")


class RetrievalQATool(BaseTool):
    """
    A tool to retrieve relevant documents and answer the question based on the info retrieved.
    """

    name = "RetrievalQATool"
    description = "useful for when you need to answer a question based on extra information retrieved from a specified knowledge base"
    args_schema: Type[BaseModel] = RetrievalQAInput

    def call(self, question: str, query: str, knowledge_base: str, k_num: int = 4, *args: Any, **kwargs: Any) -> Any:
        """
        call the _run() from outside
        """
        return self._run(question=question,
                         query=query,
                         knowledge_base=knowledge_base,
                         k_num=k_num,
                         *args,
                         **kwargs)

    def _run(self, question: str, query: str, knowledge_base: str, k_num: int = 4, *args: Any, **kwargs: Any) -> Any:
        """
        Use the tool.
        To answer a question by retrieving extra knowledge
        """
        # Get the Retrival Tool to Extract document objects from knowledge base
        retrieval_tool = RetrievalTool()
        # Retrieve relevant content fragments.
        documents = retrieval_tool.call(query=query, knowledge_base=knowledge_base, k_num=k_num)
        # Format the relevant text for question answer
        documents_text = self.format_documents_info(documents=documents)

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

    async def acall(self, question: str, query: str, knowledge_base: str, k_num: int = 4, *args: Any,
                    **kwargs: Any) -> Any:
        """
        call the _arun() from outside in Async way
        """
        return await self._arun(question=question,
                                query=query,
                                knowledge_base=knowledge_base,
                                k_num=k_num,
                                *args,
                                **kwargs)

    async def _arun(self, question: str, query: str, knowledge_base: str, k_num: int = 4, *args: Any,
                    **kwargs: Any) -> Any:
        """Use the tool asynchronously."""
        # Get the Retrival Tool to Extract document objects from knowledge base
        retrieval_tool = RetrievalTool()
        # Retrieve relevant content fragments in Async way
        documents = await retrieval_tool.acall(
            query=query,
            knowledge_base=knowledge_base,
            k_num=k_num)
        # Format the relevant text for question answer
        documents_text = self.format_documents_info(documents=documents)

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
