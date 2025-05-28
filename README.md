# Pasvalys RAG System

A Retrieval-Augmented Generation (RAG) system built with LangChain that answers questions about the city of Pasvalys, Lithuania.

## Features

- 📚 Document loading and text splitting
- 🔍 Vector embeddings using OpenAI
- 🗃️ FAISS vector store for efficient similarity search
- 🤖 Question-answering using GPT-3.5-turbo
- 💬 Interactive chat interface
- 🎯 Pre-built demo questions

## Setup

1. **Install dependencies** (using uv):
   ```bash
   uv sync
   ```

2. **Set up OpenAI API Key**:
   Create a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```
   
   Or set it as an environment variable:
   ```bash
   export OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

3. **Run the application**:
   ```bash
   uv run python main.py
   ```

## How it Works

1. **Document Loading**: Reads the Pasvalys document from `documents/pasvalys.txt`
2. **Text Splitting**: Breaks the document into chunks of 500 characters with 50-character overlap
3. **Embeddings**: Creates vector embeddings using OpenAI's embedding model
4. **Vector Store**: Stores embeddings in FAISS for efficient retrieval
5. **QA Chain**: Uses RetrievalQA to answer questions based on relevant document chunks

## Demo Questions

The system includes pre-built demo questions:
- What is Pasvalys?
- What happened in 1557 in Pasvalys?
- What sports teams does Pasvalys have?
- What museums are in Pasvalys?
- Tell me about the history of Pasvalys during Soviet occupation.

## Interactive Mode

After the demo, you can ask your own questions about Pasvalys. Type 'quit' to exit.

## Dependencies

- `langchain` - Main framework for LLM applications
- `langchain-openai` - OpenAI integrations
- `langchain-community` - Community integrations including FAISS
- `faiss-cpu` - Vector similarity search
- `tiktoken` - Tokenizer for OpenAI models
- `python-dotenv` - Environment variable management

## Requirements

- Python 3.12+
- OpenAI API key
- Internet connection for API calls