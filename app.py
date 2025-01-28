import os
import chainlit as cl
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Initialize Model
def initialize_model():
    """
    Initializes the Azure OpenAI GPT-4 model via LangChain's wrapper.
    - Fetches the required Azure OpenAI configuration (endpoint, API key, version) from environment variables.
    - Sets the `AzureChatOpenAI` wrapper with the deployment name, temperature, and token limit.
    """
    load_dotenv()  # Load .env file for environment variables
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # Create the LLM instance (Azure GPT-4)
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,  # Azure endpoint
        api_key=api_key,                # API key for authentication
        api_version=api_version,        # API version
        azure_deployment="gpt-4o-mini", # Replace with correct deployment name
        temperature=0.7,                # Creativity vs. determinism
        max_tokens=500                  # Maximum token length for responses
    )
    return llm  # Return the initialized LLM

# Load Markdown Files
def load_markdown_files(directory):
    """
    Loads all Markdown (.md) files from the specified directory.
    - Uses LangChain's `TextLoader` to read and split the text.
    - Returns a list of document objects.
    """
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):  # Only process Markdown files
            loader = TextLoader(os.path.join(directory, filename), encoding="utf-8")
            docs.extend(loader.load_and_split())  # Split the loaded text into sections
    return docs  # Return loaded documents

# Split Documents into Chunks
def split_documents(documents):
    """
    Splits large documents into smaller chunks using `RecursiveCharacterTextSplitter`.
    - Ensures that chunks are of manageable size (e.g., 500 characters with 100-character overlap).
    - Adds start indices for better context tracking.
    """
    text_splitter = RecursiveCharacterTextSplitter( # CharacterTextSplitter, SentenceTextSplitter, or MarkdownHeaderTextSplitter
        chunk_size=500,  # Max size of each chunk
        chunk_overlap=100,  # Overlap to maintain context across chunks
        add_start_index=True  # Track where each chunk starts in the original document
    )
    chunks = text_splitter.split_documents(documents)  # Split the documents
    return chunks  # Return the resulting chunks

# Create Embeddings and Vector Store
def create_vector_store(chunks):
    """
    Creates a FAISS vector store for fast similarity search.
    - Uses `HuggingFaceEmbeddings` for text embeddings with a lightweight model.
    - Embeds the document chunks into a dense vector space for retrieval.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Lightweight embedding model # Better model: sentence-transformers/all-mpnet-base-v2, SciBERT
    vector_store = FAISS.from_documents(chunks, embedding_model)  # Create vector store with FAISS or Weaviate, Pinecone, Milvus, Chroma
    return vector_store  # Return vector store

def transform_files(path):
    """
    Helper function to load, split, and embed documents.
    - Combines multiple steps (load, split, create vector store) into one function.
    - Returns the vector store for retrieval.
    """
    documents = load_markdown_files(path)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    return vector_store  # Return final vector store

# Create QA Chain Function
def create_qa_chain(vector_store, llm):
    """
    Creates a QA system using LangChain's `RetrievalQA`.
    - Combines the LLM with a retriever (vector store) for context-aware question answering.
    - Uses `load_qa_chain` to set up the combination logic for retrieved documents.
    """
    retriever = vector_store.as_retriever()  # Convert vector store into a retriever

    # Load a default QA chain for combining documents
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"  # Use the "stuff" chain type for combining document chunks, "map_reduce" (summarization) or "refine" (iterative improvement).
    )
    return qa_chain

# Chainlit Handlers
@cl.on_chat_start
async def on_chat_start():
    """
    Chainlit event: Executes when a chat starts.
    - Initializes the LLM, loads and processes Markdown files, and sets up the RetrievalQA system.
    - Stores the QA chain in the user session for later use.
    """
    llm = initialize_model()  # Initialize the LLM
    directory = "data"  # Directory containing the Markdown files
    documents = load_markdown_files(directory)  # Load Markdown files
    chunks = split_documents(documents)  # Split into chunks
    vector_store = create_vector_store(chunks)  # Create vector store
    qa_chain = create_qa_chain(vector_store, llm)  # Set up QA chain
    cl.user_session.set("qa_chain", qa_chain)  # Save the QA chain in the user session

@cl.on_message
async def on_message(message: cl.Message):
    """
    Chainlit event: Handles user messages.
    - Retrieves the QA chain from the user session.
    - Passes the user question to the chain and sends back the answer.
    """
    qa_chain = cl.user_session.get("qa_chain")  # Get QA chain from the session
    response = qa_chain.run(message.content)  # Run the QA chain with the user's message
    await cl.Message(content=response).send()  # Send the response back to the user

#chainlit run app.py --port 8501 --host 0.0.0.0 --headless