"""
This tool can retrieve relevant fragments from an external knowledge base based on the query you input.
"""
from typing import Type, Any
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.documents import Document

from util.chroma_db_util import ChromaDBUtil


class RetrievalInput(BaseModel):
    """Pydantic model for RetrievalTool"""

    query: str = Field(description="should be a search query")
    knowledge_base: str = Field(description="should be a name of knowledge base")


class RetrievalTool(BaseTool):
    """
    A tool to retrieve relevant documents to query
    """

    name = "RetrivalTool"
    description = "useful for when you need to retrieve information from a specified knowledge base"
    args_schema: Type[BaseModel] = RetrievalInput

    def call(self, query: str, knowledge_base: str, k_num: int = 6, *args: Any, **kwargs: Any) -> list[Document]:
        """
        call the _run() from outside
        """
        return self._run(query=query,
                         knowledge_base=knowledge_base,
                         k_num=k_num,
                         *args,
                         **kwargs)

    def _run(self, query: str, knowledge_base: str, k_num: int = 6, *args: Any, **kwargs: Any) -> list[Document]:
        """
        Use the tool.
        To get K_num of relevant documents, which are chunks
        """
        # Set up the tool to process Chroma DB
        chroma_util = ChromaDBUtil()
        # Get Vector store
        vectorstore = chroma_util.load_vectorstore(persist_directory=knowledge_base)
        # Get retriever for retrieval task
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_num})
        # Start to retrieve
        documents = retriever.invoke(query)
        return documents

    async def acall(self, query: str, knowledge_base: str, k_num: int = 6, *args: Any, **kwargs: Any) -> list[Document]:
        """
        call the _arun() from outside in Async way
        """
        return await self._arun(query=query,
                                knowledge_base=knowledge_base,
                                k_num=k_num,
                                *args,
                                **kwargs)

    async def _arun(self, query: str, knowledge_base: str, k_num: int = 6, *args: Any, **kwargs: Any) -> list[Document]:
        """Use the tool asynchronously."""
        # Set up the tool to process Chroma DB
        chroma_util = ChromaDBUtil()
        # Get Vector store
        vectorstore = chroma_util.load_vectorstore(persist_directory=knowledge_base)
        # Get retriever for retrieval task
        retriever = vectorstore.as_retriever(search_kwargs={"k": k_num})
        # Start to retrieve
        documents = await retriever.ainvoke(query)
        return documents
