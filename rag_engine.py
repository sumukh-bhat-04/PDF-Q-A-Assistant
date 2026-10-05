import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2

def extract_documents_from_pdf(pdf_file) -> list[Document]:
    """
    Extracts text page-by-page from an uploaded PDF file-like object 
    and wraps them into LangChain Document objects with page metadata.
    """
    reader = PyPDF2.PdfReader(pdf_file)
    documents = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        doc = Document(
            page_content=page_text,
            metadata={"page": i + 1}
        )
        documents.append(doc)
    return documents

def chunk_documents(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits larger documents into smaller, overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks: list[Document], api_key: str = None) -> Chroma:
    """
    Generates embeddings and builds an in-memory Chroma vector store from document chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    # In-memory transient ChromaDB
    return Chroma.from_documents(chunks, embeddings)

def get_rag_chain(vector_store: Chroma, api_key: str = None):
    """
    Constructs and returns the LangChain retrieval QA chain.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0, google_api_key=api_key)
    
    # Custom QA system prompt to minimize hallucinations and enforce context grounding
    system_prompt = (
        "You are a helpful assistant specialized in document intelligence.\n"
        "Answer the user's question using only the provided context. If the answer\n"
        "cannot be found or inferred from the context, state clearly that the answer\n"
        "is not present in the document. Do not invent any facts.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create combine documents chain and the final retrieval chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return retrieval_chain

def query_rag(chain, question: str) -> dict:
    """
    Executes a Q&A query against the retrieval chain.
    Returns a dictionary containing 'answer' and 'context' (retrieved chunks).
    """
    response = chain.invoke({"input": question})
    return {
        "answer": response["answer"],
        "context": response["context"]
    }
