import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables
load_dotenv()

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

def main():
    """Main function to demonstrate the RAG system."""
    print("🏛️ Pasvalys RAG System using LangChain")
    print("=" * 50)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("❌ Please set your OPENAI_API_KEY in the environment variables or .env file")
        print("You can create a .env file with: OPENAI_API_KEY=your_actual_api_key")
        return
    
    print("📚 Loading Pasvalys document...")
    text_content = load_pasvalys_document()
    
    if not text_content:
        return
    
    print("🔍 Creating vector embeddings...")
    try:
        vector_store = create_vector_store(text_content)
        print("✅ Vector store created successfully!")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return
    
    print("🤖 Setting up QA chain...")
    try:
        qa_chain = create_qa_chain(vector_store)
        print("✅ QA chain ready!")
    except Exception as e:
        print(f"❌ Error setting up QA chain: {e}")
        return
    
    # Interactive mode
    print("\n💬 Interactive Mode (type 'quit' to exit):")
    print("=" * 50)
    
    while True:
        user_question = input("\n🔮 Ask about Pasvalys: ").strip()
        
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_question:
            continue
            
        try:
            result = qa_chain.invoke({"query": user_question})
            print(f"🤖 Answer: {result['result']}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
