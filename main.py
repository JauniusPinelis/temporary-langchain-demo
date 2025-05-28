import os
from dotenv import load_dotenv
from rag_helpers import load_pasvalys_document, create_vector_store, create_qa_chain

# Load environment variables
load_dotenv()

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
            
            # Show source information
            if result.get('source_documents'):
                print(f"\n📖 Sources used ({len(result['source_documents'])} chunks):")
                print("-" * 40)
                for i, doc in enumerate(result['source_documents'], 1):
                    # Show the full source content
                    content = doc.page_content.strip()
                    
                    print(f"   {i}. Source: {doc.metadata.get('source', 'Unknown')}")
                    print(f"      Content: {content}")
                    print()
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
