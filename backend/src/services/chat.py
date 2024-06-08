from chromadb.config import Settings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChatService:
    vectorstore = None
    retriver = None

    @classmethod
    async def embed_urls(cls, urls: list[str]):
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)
        cls.vectorstore = Chroma.from_documents(  # type: ignore
            documents=doc_splits,
            collection_name="chats",
            embedding=GPT4AllEmbeddings(),  # type: ignore
            persist_directory="./chroma",
            client_settings=Settings(allow_reset=True),
        )
        cls.retriver = cls.vectorstore.as_retriever()

    @classmethod
    async def query_db(cls, query: str):
        if cls.retriver:
            results = cls.retriver.invoke(query)
            print(results)
        else:
            print("please make sure to have inputted some urls.")

    @classmethod
    async def clear(cls):
        if cls.vectorstore:
            cls.vectorstore._client.reset()  # type: ignore
