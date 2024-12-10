from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

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

# Initialize the model
llm = initialize_model()

# Test the model
response = llm("Explain the concept of AI in simple terms.")
print(response)






################### chat ############################

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Step 1: Load Markdown Files
# This function reads all `.md` (Markdown) files from a specified directory.
# It uses LangChain's `TextLoader` to load the content of the files and process them into documents.
def load_markdown_files(directory):
    docs = []  # List to hold all loaded documents
    for filename in os.listdir(directory):
        if filename.endswith(".md"):  # Only process Markdown files
            loader = TextLoader(os.path.join(directory, filename), encoding="utf-8")
            # Load and split the content of the Markdown file into a list of documents
            docs.extend(loader.load_and_split())
    return docs

# Step 2: Split Documents into Chunks
# Large documents can be difficult for models to process in one go. This function
# splits documents into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`.
def split_documents(documents):
    # Create a text splitter with a chunk size of 500 characters and a 50-character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=True) # track index in original document
    # Apply the text splitter to the loaded documents
    chunks = text_splitter.split_documents(documents)
    return chunks

# Step 3: Create Embeddings and Vector Store
# Embeddings convert text into numerical vectors that can be used to compare the similarity
# of texts. FAISS (Facebook AI Similarity Search) is used to store and retrieve these vectors efficiently.
def create_vector_store(chunks):
    # Load a pre-trained embedding model from Hugging Face
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Create a FAISS vector store from the text chunks and their embeddings
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store

# Load and process your .md files
directory = "path_to_your_md_files"  # Replace with the directory containing your Markdown files
# Step 1: Load the Markdown files into a list of documents
documents = load_markdown_files(directory)
# Step 2: Split the documents into smaller, manageable chunks
chunks = split_documents(documents)
# Step 3: Create a FAISS vector store to store the chunks and their embeddings
vector_store = create_vector_store(chunks)


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Step 4: Create a RetrievalQA Chain
# The RetrievalQA chain ties everything together: it retrieves relevant text chunks
# from the vector store and uses the language model to generate an answer.
def create_qa_chain(vector_store, llm):
    # Use the FAISS vector store as the retriever to find relevant chunks
    retriever = vector_store.as_retriever()
    
    # Define a prompt template for the chatbot
    # - `context` will hold the retrieved text chunks
    # - `question` will be the user's query
    prompt_template = """
    You are a helpful assistant. Use the following context to answer the question:
    {context}
    Question: {question}
    Answer:"""

    # Create a prompt object using the template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Create the RetrievalQA chain with the LLM, retriever, and prompt
    qa_chain = RetrievalQA(llm=llm, retriever=retriever, prompt=prompt)
    return qa_chain

# NOTE: Ensure you have initialized your language model (`llm`) before this step.
# For example, you can use HuggingFacePipeline with a model like Falcon or BLOOM.

# Step 4: Create the QA chain using the vector store and the language model
qa_chain = create_qa_chain(vector_store, llm)

# At this stage, you have:
# - Loaded Markdown files into documents
# - Split those documents into smaller chunks
# - Created embeddings for the chunks and stored them in a FAISS vector store
# - Created a RetrievalQA chain that uses the vector store and the language model to answer questions







################### chat with indexing ##########################


from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Step 1: Create an Index from Markdown Files
# This function loads Markdown files, converts them into embeddings, and stores them in a vector database (FAISS).
def create_index(directory="data"):
    # Load Markdown files from the specified directory.
    # `DirectoryLoader` automatically finds all `.md` files using the provided `glob` pattern.
    loader = DirectoryLoader(directory, glob="*.md")  # Match all files with the `.md` extension.

    # Initialize an embedding model for converting text into numerical representations (embeddings).
    # Here, we use a lightweight and efficient embedding model from Hugging Face.
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a vector store index with the following:
    # - FAISS: A fast and efficient vector similarity search engine.
    # - Embedding Model: Converts text chunks into embeddings that can be searched.
    # `VectorstoreIndexCreator` handles document splitting, embedding, and storing in the vector store.
    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,  # Use FAISS for the vector store.
        embedding=embedding_model  # Specify the embedding model to generate embeddings.
    ).from_loaders([loader])  # Process documents loaded by the `DirectoryLoader`.

    # Return the created index for further use.
    return index



index = create_index()
# Use the index to search for relevant information using natural language queries.
# The query is matched against the indexed Markdown files based on similarity.
query = "What is the main topic of file X?"
response = index.query(query)  # Perform the query and get the response.
print(response)  # Print the response to the console.


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Step 5: Create a RetrievalQA Chain
# This function connects the index (retriever) and a language model (LLM) to process queries.
def create_qa_chain(index, llm):
    # Use the vector store from the index as a retriever.
    # The retriever fetches the most relevant text chunks from the vector store based on the query.
    retriever = index.vectorstore.as_retriever()

    # Define a prompt template to guide the language model's behavior.
    # - `{context}`: Represents the retrieved text chunks.
    # - `{question}`: Represents the user's query.
    # The prompt ensures the language model uses the retrieved context to generate accurate answers.
    prompt_template = """
    You are a helpful assistant. Use the following context to answer the question:
    {context}
    Question: {question}
    Answer:"""

    # Initialize the prompt using the `PromptTemplate` class.
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create a RetrievalQA chain that ties together:
    # - The language model (LLM): Generates the final answer.
    # - The retriever: Finds the most relevant text chunks from the index.
    # - The prompt: Structures the query and context for the LLM.
    qa_chain = RetrievalQA(llm=llm, retriever=retriever, prompt=prompt)

    # Return the created QA chain for further use.
    return qa_chain

# Step 6: Initialize the QA chain.
# Ensure that you have already initialized the language model (LLM).
# For example, the LLM could be a Hugging Face pipeline wrapped in LangChain's `HuggingFacePipeline`.
qa_chain = create_qa_chain(index, llm)

# Step 7: Test the QA Chain
# Use the QA chain to answer a natural language query.
query = "What is the main topic of file X?"
response = qa_chain.run(query)  # Get the response from the QA chain.
print(response)  # Print the generated answer to the console.
