import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

#0000FF 
#FFFFFF
#7000FF
#000000
#C0C0C0

FALLBACK_ENABLED = False  # Toggle for fallback mode


@st.cache_resource
def initialize_model():
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500,
    )
    return llm


def load_markdown_files(directory):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            loader = TextLoader(os.path.join(directory, filename), encoding="utf-8")
            docs.extend(loader.load_and_split())
    return docs


def split_documents(docs, headers_to_split_on=None, chunk_size=500, chunk_overlap=100):
    # Default header splitting rules, chunk size and ovelpat to be tested.
    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    all_splits = []
    for doc in docs:
        # 1: Split document into header-based sections with MarkdownHeaderTextSplitter
        md_header_splits = markdown_splitter.split_text(doc.page_content)
        # 2: Apply character-level splitting to each section with RecursiveCharacterTextSplitter
        splits = text_splitter.split_documents(md_header_splits)
        # 3: Store the results
        all_splits.extend(splits)

    return all_splits


@st.cache_resource
def create_vector_store(_chunks):
    """
    Creates a FAISS vector store for fast similarity search.
    - Uses `HuggingFaceEmbeddings` for text embeddings with a lightweight model.
    - Embeds the document chunks into a dense vector space for retrieval.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Create vector store with FAISS or Weaviate, Pinecone, Milvus, Chroma
    vector_store = FAISS.from_documents(_chunks, embedding_model)
    return vector_store


def create_qa_chain(llm, vector_store):
    """
    Creates a QA system using LangChain's `RetrievalQA`.
    - Combines the LLM with a retriever (vector store) for context-aware question answering.
    - Uses `load_qa_chain` to set up the combination logic for retrieved documents.
    """
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        # Use the "stuff" type to combine document chunks, "map_reduce"(summary) or "refine"(iterative improvement?).
        llm=llm, retriever=retriever, chain_type="stuff"
    )
    return qa_chain


def main():
    st.title("Streamlit Bot 🤖")

    ## Inject custom CSS for styling colors
    ## 1)
    #st.markdown(
    #    "<style>.reportview-container {background: #ffffff;}</style>", unsafe_allow_html=True
    #)

    ## 2)
    #st.markdown(
    #    """
    #    <style>
    #    /* Background color for the app */
    #    .main {
    #        background-color: #7000FF; /* purple */
    #    }
#
    #    /* User message styling */
    #    .user-message {
    #        background-color: ffffff; /* Light green */
    #        padding: 10px;
    #        border-radius: 8px;
    #        margin: 5px 0;
    #    }
#
    #    /* Bot message styling */
    #    .bot-message {
    #        background-color: #ffffff; /* Light pink */
    #        padding: 10px;
    #        border-radius: 8px;
    #        margin: 5px 0;
    #    }
    #    </style>
    #    """,
    #    unsafe_allow_html=True,
    #)
    #
#

    llm = initialize_model()
    directory = "data"
    documents = load_markdown_files(directory)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    qa_chain = create_qa_chain(llm, vector_store)

    # Manage chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Callback for processing input
    def process_input():
        user_input = st.session_state["user_input"].strip()
        if user_input:
            # Append user message
            st.session_state["messages"].append({"role": "user", "content": user_input})

            # Option 1: Try RAG-based retrieval
            response = qa_chain.invoke({"query": user_input})
            rag_text = response.get("result", "")

            # Custom response for "I don't know" cases
            if rag_text.strip().lower() in ["i don't know.", "i dont know.", "i do not know."]:
                rag_text = "Sorry, I do not have this information. Please try another question."

            # Option 2: Check fallback condition
            bot_reply = rag_text
            if FALLBACK_ENABLED:
                if not rag_text or "sorry" in rag_text.lower():
                    fallback_prompt = (
                        f"The user asked: \"{user_input}\". Answer from your own knowledge."
                    )
                    fallback_response = llm(fallback_prompt)
                    bot_reply = fallback_response.content

            # Append bot reply
            st.session_state["messages"].append({"role": "bot", "content": bot_reply})
            # Clear input field
            st.session_state["user_input"] = ""

    #st.sidebar.markdown("### Customize")
    #st.sidebar.selectbox("Theme", options=["Light", "Dark"])

    # Display chat history
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("ai").write(message["content"])

     # User input box for the user with placeholder text
    st.text_input(
        "",  # Remove the label entirely
        placeholder="Type your message here...",  # Placeholder text inside the input box
        key="user_input",
        on_change=process_input  # Trigger input processing when the user presses Enter
    )

if __name__ == "__main__":
    main()

