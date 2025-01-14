import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Initialize Model
@st.cache_resource
def initialize_model():
    """
    Initializes the Azure OpenAI GPT-4 model via LangChain's wrapper.
    """
    load_dotenv()  # Load .env file for environment variables
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment="gpt-4o-mini",  # Replace with correct deployment name
        temperature=0.7,
        max_tokens=500
    )
    return llm

# Load Markdown Files
def load_markdown_files(directory):
    """
    Loads all Markdown (.md) files from the specified directory.
    """
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"): 
            loader = TextLoader(os.path.join(directory, filename), encoding="utf-8")
            docs.extend(loader.load_and_split())
    return docs  

# Split Documents into Chunks
def split_documents(documents):
    """
    Splits large documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    return chunks 

# Create Embeddings and Vector Store
def create_vector_store(chunks):
    """
    Creates a vector store for fast similarity search.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store

# Transform Files
@st.cache_resource
def transform_files(path):
    """
    Helper function to load, split, and embed documents.
    """
    documents = load_markdown_files(path)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    return vector_store

# Create QA Chain
@st.cache_resource
def create_qa_chain(_vector_store, _llm):
    """
    Creates a QA system using LangChain's `RetrievalQA`.
    """
    retriever = _vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=_llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain

# Main App
def main():
    st.title("Streamlit LangChain QA Bot")

    # Load model and data
    llm = initialize_model()
    directory = "data"
    vector_store = transform_files(directory)
    qa_chain = create_qa_chain(vector_store, llm)

    # User interaction
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.subheader("Chat with the Bot")

    # Display chat history
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            st.markdown(f"**User:** {message['content']}")
        else:
            st.markdown(f"**Bot:** {message['content']}")

    # User input
    user_input = st.text_input("Your question:", key="input", placeholder="Type your message here...")
    if st.button("Send"):
        if user_input:
            # Append user message to session
            st.session_state["messages"].append({"role": "user", "content": user_input})

            # Get response from QA chain
            response = qa_chain.run(user_input)

            # Append bot response to session
            st.session_state["messages"].append({"role": "bot", "content": response})

            # Rerun the app to display new messages
            st.rerun()

if __name__ == "__main__":
    main()

