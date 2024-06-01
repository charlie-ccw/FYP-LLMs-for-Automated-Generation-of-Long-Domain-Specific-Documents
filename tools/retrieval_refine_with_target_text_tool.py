from typing import Type, Any
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from globalParameter.parameters import MODEL
from prompt.prompt_of_retrieval_refine_with_target_text_tool import RETRIEVAL_REFINE_WITH_TARGET_TEXT_TOOL_PROMPT, \
    RETRIEVAL_REFINE_WITH_TARGET_TEXT_TOOL_SYSTEM
from tools.retrieval_tool import RetrievalTool
from util.prompt_based_generation import prompt_based_generation, aprompt_based_generation


class RetrievalRefineWithTargetTextInput(BaseModel):
    """Pydantic model for RetrievalRefineWithTargetTextTool"""

    target_text: str = Field(
        description="should be a model text that you hope to retrieve from the knowledge base, which is particularly similar to it.")
    draft_text: str = Field(description="should be a draft text that needs to be refined based on a model text.")
    knowledge_base: str = Field(description="should be a name of knowledge base for retrieving extra info")


class RetrievalRefineWithTargetTextTool(BaseTool):
    """
    A tool to refine a text based on example text retrieved from a specified knowledge base using target text.
    """

    name = "RetrievalRefineWithTargetTextTool"
    description = "useful for when you need to refine a text based on example text retrieved from a specified knowledge base using target text"
    args_schema: Type[BaseModel] = RetrievalRefineWithTargetTextInput

    def call(self, target_text: str, draft_text: str, knowledge_base: str, k_num: int = 1, *args: Any,
             **kwargs: Any) -> Any:
        """
        call the _run() from outside
        """
        return self._run(target_text=target_text,
                         draft_text=draft_text,
                         knowledge_base=knowledge_base,
                         k_num=k_num,
                         *args,
                         **kwargs)

    def _run(self, target_text: str, draft_text: str, knowledge_base: str, k_num: int = 1, *args: Any,
             **kwargs: Any) -> Any:
        """
        Use the tool.
        To answer a question by retrieving extra knowledge
        """
        retrieval_tool = RetrievalTool()
        documents = retrieval_tool.call(
            query=target_text,
            knowledge_base=knowledge_base,
            k_num=k_num)
        documents_text = self.format_documents_info(documents=documents)

        human_prompt = kwargs.get('prompt', RETRIEVAL_REFINE_WITH_TARGET_TEXT_TOOL_PROMPT)
        prompt_parameter = kwargs.get('prompt_parameter', {'example_text': documents_text, 'draft_text': draft_text})
        json_format = kwargs.get('json_format', True)

        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=RETRIEVAL_REFINE_WITH_TARGET_TEXT_TOOL_SYSTEM),
                HumanMessagePromptTemplate.from_template(human_prompt),
            ]
        )
        prompt = chat_template.format_messages(**prompt_parameter)

        response = prompt_based_generation(prompt=prompt, model=MODEL, temperature=0.5,
                                           json_format=json_format)
        return response

    async def acall(self, target_text: str, draft_text: str, knowledge_base: str, k_num: int = 1, *args: Any,
                    **kwargs: Any) -> Any:
        """
        call the _arun() from outside in Async way
        """
        return await self._arun(target_text=target_text,
                                draft_text=draft_text,
                                knowledge_base=knowledge_base,
                                k_num=k_num,
                                *args,
                                **kwargs)

    async def _arun(self, target_text: str, draft_text: str, knowledge_base: str, k_num: int = 1, *args: Any,
                    **kwargs: Any) -> Any:
        """Use the tool asynchronously."""
        retrieval_tool = RetrievalTool()
        documents = await retrieval_tool.acall(
            query=target_text,
            knowledge_base=knowledge_base,
            k_num=k_num)
        documents_text = self.format_documents_info(documents=documents)

        human_prompt = kwargs.get('prompt', RETRIEVAL_REFINE_WITH_TARGET_TEXT_TOOL_PROMPT)
        prompt_parameter = kwargs.get('prompt_parameter', {'example_text': documents_text, 'draft_text': draft_text})
        json_format = kwargs.get('json_format', True)

        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=RETRIEVAL_REFINE_WITH_TARGET_TEXT_TOOL_SYSTEM),
                HumanMessagePromptTemplate.from_template(human_prompt),
            ]
        )
        prompt = chat_template.format_messages(**prompt_parameter)

        response = await aprompt_based_generation(prompt=prompt, model=MODEL, temperature=0.5,
                                                  json_format=json_format)
        return response

    @staticmethod
    def format_documents_info(documents: list[Document]) -> str:
        format_text = ""
        for document in documents:
            format_text += f"{document.page_content}\n"
        return format_text
