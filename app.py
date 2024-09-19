import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from appconfig import AppConfig
from azureai import AzureAI

# Create instances of AppConfig and AzureAI
config = AppConfig()
azure_ai = AzureAI(config)

# Initialize OpenAI embeddings
embeddings = azure_ai.get_embedding_client()

#Initialize the llm
llm = azure_ai.get_client()

# Streamlit app layout
st.title("SAP BTP Demo")

# Step 1: Input for URL
st.header("Step 1: Load Data from URL")
url = st.text_input("Enter URL to process", "")

if st.button("Load and Process URL"):
    if url:
        with st.spinner("Loading data from URL..."):
            # Load document from the provided URL
            doc = WebBaseLoader(url).load()
            st.success("Document loaded and processed.")

        # Step 2: Split the document
        with st.spinner("Splitting document into smaller chunks..."):
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=500, chunk_overlap=0
            )
            doc_splits = text_splitter.split_documents(doc)
            st.success(f"Document split into {len(doc_splits)} chunks.")

        # Step 3: Create and populate the Chroma vector store
        with st.spinner("Creating and populating Chroma vector store..."):
            vector_store = Chroma(
                collection_name="langgraph_demo",
                embedding_function=embeddings,
                persist_directory="./chroma_langchain_db",  # Directory to persist the DB
            )
            vector_store.add_documents(doc_splits)
            chroma_vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)
            st.success(f"Inserted {len(doc_splits)} document chunks into Chroma vector store.")

        # Step 4: Setup RAG pipeline using OpenAI LLM
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever
        )

        # Step 5: Ask a question and generate answer using RAG
        st.header("Step 5: Ask a Question")
        query = st.text_input("Enter your query", value="What is agent?")

        if st.button("Get Answer"):
            with st.spinner("Generating response..."):
                # Use the OpenAI LLM and Chroma retriever to generate an answer
                answer = qa_chain.run(query)
                st.write(f"Answer: {answer}")
    else:
        st.warning("Please enter a valid URL.")
