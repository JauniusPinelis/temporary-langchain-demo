from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document

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
    """Create a vector store from the text content."""
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
    
    return vector_store

def create_qa_chain(vector_store):
    """Create a QA chain using the vector store."""
    # Initialize the language model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain 