from typing import Any, Optional, TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

from src.config import settings
from src.db import client
from src.models.chat import Chat, Url
from src.models.core import PyObjectId
from src.repositories.chat import ChatRepository


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    summary: str
    chat_id: str
    question: str
    generation: str
    web_search: str
    documents: list[Document]


class ChatService:
    def __init__(self, chat_repository: ChatRepository):
        self.chat_repository = chat_repository

    async def get_chats(self):
        docs = await self.chat_repository.find()  # type: ignore

        return [Chat(**doc) for doc in docs]  # type: ignore

    async def get_chat_summary(self, chat_id: str):
        doc = await self.chat_repository.find_one(chat_id)  # type: ignore

        return Chat(**doc)  # type: ignore

    async def add_message(self, chat_id: PyObjectId, message: dict[str, Any]):
        await self.chat_repository.add_message(chat_id, message)

    async def create_chat(
        self,
        name: Optional[str],
        urls: list[str],
        file: str,
    ):
        chat = Chat(
            name=name,
            urls=[],  # type: ignore
            file=file.split("/")[-1],
        )
        inserted_id = await self.chat_repository.create(chat)
        metadata = await self.embed_urls(chat_id=str(inserted_id), urls=urls)  # type: ignore
        chat.urls = [Url(**m) for m in metadata]  # type: ignore
        await self.load_and_embed_pdf(chat_id=str(inserted_id), file_path=file)
        await self.chat_repository.update(inserted_id, chat)
        return

    async def embed_urls(self, chat_id: str, urls: list[str]):  # type: ignore
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)
        model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
        Chroma.from_documents(  # type: ignore
            documents=doc_splits,
            collection_name=chat_id,
            client=client,
            embedding=GPT4AllEmbeddings(model_name=model_name),  # type: ignore
        )

        return [doc.metadata for doc in docs_list]  # type: ignore

    async def load_and_embed_pdf(self, chat_id: str, file_path: str):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
        Chroma.from_documents(  # type: ignore
            documents=pages,
            collection_name=chat_id,
            client=client,
            embedding=GPT4AllEmbeddings(model_name=model_name),  # type: ignore
        )


def query_db(chat_id: str, query: str):
    model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    vectorstore = Chroma(
        collection_name=chat_id,
        client=client,
        embedding_function=GPT4AllEmbeddings(model_name=model_name),  # type: ignore
    )
    retriver = vectorstore.as_retriever()
    results = retriver.invoke(query)

    return results


def grade_retrieval(question: str, result: Document):
    json_llm = ChatOllama(
        model="llama3", format="json", temperature=1, base_url="http://ollama:11434"
    )
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
of a retrieved document to a user question. If the document contains keywords related to the user question, 
grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
Here is the retrieved document: \n\n {document} \n\n
Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
        input_variables=["question", "document"],
    )

    retrieval_grader = prompt | json_llm | JsonOutputParser()  # type: ignore
    return retrieval_grader.invoke(  # type: ignore
        {"question": question, "document": result.page_content}
    )


def generate_answer(question: str, docs: list[Document]):
    llm = ChatOllama(model="llama3", temperature=1, base_url="http://ollama:11434")
    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )
    rag_chain = prompt | llm | StrOutputParser()  # type: ignore
    generation = rag_chain.invoke(  # type: ignore
        {
            "context": "\n\n".join(doc.page_content for doc in docs),
            "question": question,
        }
    )

    return generation


def hallucination_grader(generation: str, docs: list[Document]):
    json_llm = ChatOllama(
        model="llama3", format="json", temperature=1, base_url="http://ollama:11434"
    )
    # Prompt
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a  single key 'score' and no preamble or explanation. Make sure to exactly output a json with one key 'score' <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer you need to grade and don't forget, the output needs to be in json format with one and only one key called score (yes if the answer is grouned to the facts else no): {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )
    hallucination_grader = prompt | json_llm | JsonOutputParser()  # type: ignore

    return hallucination_grader.invoke(  # type: ignore
        {"documents": docs, "generation": generation}
    )


def answer_grader(question: str, generation: str):
    json_llm = ChatOllama(
        model="llama3", format="json", temperature=1, base_url="http://ollama:11434"
    )
    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )

    answer_grader = prompt | json_llm | JsonOutputParser()  # type: ignore

    return answer_grader.invoke({"question": question, "generation": generation})  # type: ignore


def router(state: GraphState):
    summary = state["summary"]
    question = state["question"]
    json_llm = ChatOllama(
        model="llama3", format="json", temperature=1, base_url="http://ollama:11434"
    )
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
        {summary}. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
        no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["summary", "question"],
    )
    question_router = prompt | json_llm | JsonOutputParser()  # type: ignore

    source = question_router.invoke({"summary": summary, "question": question})  # type: ignore

    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def retrieve(state: GraphState):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    chat_id = state["chat_id"]

    # Retrieval
    documents = query_db(chat_id, question)
    return {"documents": documents, "question": question, "chat_id": chat_id}


def generate(state: GraphState):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = generate_answer(question, documents)
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state: GraphState):  # type: ignore
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = grade_retrieval(question, d)
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)  # type: ignore
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }  # type: ignore


def web_search(state: GraphState):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    web_search_tool = TavilySearchResults(
        tavily_api_key=settings.TAVILY_API_KEY,  # type: ignore
        k=3,  # type: ignore
    )
    # Web search
    docs = web_search_tool.invoke({"query": question})  # type: ignore
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:  # type: ignore
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


def decide_to_generate(state: GraphState):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader(generation, documents)
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader(question, generation)
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        return "not supported"


def final(chat_id: str, question: str, summary: Any):
    workflow = StateGraph(GraphState)
    workflow.add_node("websearch", web_search)  # type: ignore # web search
    workflow.add_node("retrieve", retrieve)  # type: ignore # retrieve
    workflow.add_node("grade_documents", grade_documents)  # type: ignore # grade documents
    workflow.add_node("generate", generate)  # type: ignore # generatae
    workflow.set_conditional_entry_point(
        router,  # type: ignore
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )
    # Compile
    app = workflow.compile()
    inputs = {"chat_id": chat_id, "question": question, "summary": summary}
    outputs = app.invoke(inputs)
    return outputs
