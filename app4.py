import os
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import chainlit as cl
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.llm import 

# Initialize Hugging Face Model
def initialize_model():
    load_dotenv()
    model_name = "tiiuae/falcon-7b-instruct"  # Replace with your desired Hugging Face model

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", low_cpu_mem_usage=True)

    # Create a text-generation pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=500, temperature=0.5)

    # Wrap in LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Load Markdown Files
def load_markdown_files(directory):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            loader = TextLoader(os.path.join(directory, filename), encoding="utf-8")
            docs.extend(loader.load_and_split())
    return docs

# Split Documents into Chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Create Embeddings and Vector Store
def create_vector_store(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store

def transform_files(path):
    documents = load_markdown_files(path)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    return vector_store

# Create a RetrievalQA Chain
def create_qa_chain(vector_store, llm):
    retriever = vector_store.as_retriever()

    prompt_template = """
    You are a helpful assistant. Use the following context to answer the question:
    {context}
    Question: {question}
    Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA(llm=llm, retriever=retriever, prompt=prompt)
    return qa_chain

# Chainlit Handlers
@cl.on_chat_start
async def on_chat_start():
    llm = initialize_model()
    directory = "Chatbot/data" 
    documents = load_markdown_files(directory)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    qa_chain = create_qa_chain(vector_store, llm)
    cl.user_session.set("qa_chain", qa_chain)

@cl.on_message
async def on_message(message: cl.Message):
    qa_chain = cl.user_session.get("qa_chain")
    response = qa_chain.run(message.content)
    await cl.Message(content=response).send()
