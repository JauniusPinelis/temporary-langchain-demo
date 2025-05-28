import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTOR_STORE_PATH = "vector_store"

def load_pasvalys_document():
    """Load the Pasvalys document from the documents folder."""
    try:
        with open("documents/pasvalys.txt", "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print("Error: documents/pasvalys.txt not found!")
        return None

def create_vector_store(text_content):
    """Create a vector store from the text content or load existing one."""
    # Check if vector store already exists
    if os.path.exists(VECTOR_STORE_PATH):
        print("📁 Loading existing vector store...")
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    
    if not text_content or not text_content.strip():
        print("Error: No valid text content provided for vector store creation")
        return None
        
    print("🔨 Creating new vector store...")
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Create documents from the text
    docs = [Document(page_content=text_content, metadata={"source": "pasvalys.txt"})]
    texts = text_splitter.split_documents(docs)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # Save vector store to disk
    vector_store.save_local(VECTOR_STORE_PATH)
    print("💾 Vector store saved to disk")
    
    return vector_store

def format_docs(docs):
    """Format documents for the RAG chain."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_qa_chain(vector_store):
    """Create a modern RAG chain using the vector store."""
    if not vector_store:
        print("Error: No valid vector store provided for QA chain creation")
        return None
        
    # Initialize the language model with modern parameter
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks about Pasvalys. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}""")
    
    # Create the modern RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever 