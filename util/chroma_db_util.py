"""
This tool is responsible for managing and retrieving the corresponding external knowledge base.
"""
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from globalParameter import parameters


class ChromaDBUtil:
    """
    Use this ChromaDBUtil to:
        1. load chroma vectorstore
        2. initialise chroma vectorstore using pdf files or Document class (A class defined by langchain)
        3. load more pdf files or Document class into existing vectorstore
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    embeddings_model = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    def load_vectorstore(self, persist_directory: str) -> Chroma:
        """
        To Get a vectorstore using name
        :param persist_directory: your vectorstore/knowledge name
        :return: Chroma vectorstore
        """
        # Get full vectorstore path
        persist_directory = "vectorDB/chromaDB/" + persist_directory
        # Create an object for your chroma vectorstore
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings_model)
        return vectorstore

    def initialise_vectorstore_with_documents(self, persist_directory: str, documents: list[Document]) -> Chroma:
        """
        To initialise a new vectorstore using Document class
        :param persist_directory: your vectorstore/knowledge name
        :param documents: a list of Document objects
        :return: Chroma vectorstore
        """
        # Get full vectorstore path
        persist_directory = "vectorDB/chromaDB/" + persist_directory
        # Create an object for your chroma vectorstore using documents
        vectorstore = Chroma.from_documents(documents=documents, embedding=self.embeddings_model,
                                            persist_directory=persist_directory)
        return vectorstore

    def initialise_vectorstore_with_files(self, persist_directory: str, files: list[str], need_resort: bool = False) -> Chroma:
        """
        To intialise a new vectorstore using files
        :param persist_directory: your vectorstore/knowledge name
        :param files: a list of file path (PDF)
        :param need_resort: Do you need to resort the pdf data by adding extra index, useful when you need to use Resort RetrievalTool
        :return: Chroma vectorstore
        """
        documents = []
        # Iterate the files
        for file in files:
            # Load the info from your pdf file
            loader = PyMuPDFLoader(file)
            data = loader.load()
            # Need to add extra index for your file data
            if need_resort:
                text = ""
                for document in data:
                    text += document.page_content
                texts = self.text_splitter.split_text(text)
                for i, text in enumerate(texts):
                    documents.append(Document(page_content=text, metadata={"source": file, "index": i}))
            else:
                documents.extend(data)
        # Create an object for your chroma vectorstore using documents created by files
        vectorstore = self.initialise_vectorstore_with_documents(persist_directory=persist_directory, documents=documents)

        return vectorstore

    def load_file_to_vectorstore(self, persist_directory: str, files: list[str], need_resort: bool = False):
        """
        To add more files into existing vectorstore
        :param persist_directory: your vectorstore/knowledge name
        :param files: a list of file path (PDF)
        :param need_resort: Do you need to resort the pdf data by adding extra index, useful when you need to use Resort RetrievalTool
        :return: Chroma vectorstore
        """
        documents = []
        # Iterate the files
        for file in files:
            # Load the info from your pdf file
            loader = PyMuPDFLoader(file)
            data = loader.load()
            # Need to add extra index for your file data
            if need_resort:
                text = ""
                for document in data:
                    text += document.page_content
                texts = self.text_splitter.split_text(text)
                for i, text in enumerate(texts):
                    documents.append(Document(page_content=text, metadata={"source": file, "index": i}))
            else:
                documents.extend(data)
        # Add the new Document objects into your vectorstore
        self.load_documents_to_vectorstore(persist_directory=persist_directory, documents=documents)

    def load_documents_to_vectorstore(self, persist_directory: str, documents: list[Document]):
        """
        To add more Document objects into existing vectorstore
        :param persist_directory: your vectorstore/knowledge name
        :param documents: a list of Document objects
        :return: Chroma vectorstore
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        vectorstore = self.load_vectorstore(persist_directory=persist_directory)
        vectorstore.add_texts(texts=texts, metadatas=metadatas)

