import streamlit as st
import os
from llama_index.core import Document
import openai
from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.llms.openai import OpenAI
from trulens_eval import Tru
from Code.rag_utils import get_prebuilt_trulens_recorder
import Code.rag_utils as rag_utils
from Code.rag_utils import get_sentence_window_query_engine
from Code.rag_utils import build_sentence_window_index

openai.api_key = rag_utils.get_openai_api_key()

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from Code.rag_utils import build_automerging_index
from Code.rag_utils import get_automerging_query_engine


# Initialize OpenAI
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

def read_doc(uploaded_file):
    if uploaded_file is not None:
        # List of encodings to try
        encodings = ["utf-8", "latin-1"]  # You can add more encodings if needed

        # Initialize document as None
        document = None

        # Try decoding using different encodings
        for encoding in encodings:
            try:
                # Read the content of the uploaded file
                file_contents = uploaded_file.read()

                # Convert bytes to string using the current encoding
                text = file_contents.decode(encoding)

                # Create a Document object
                document = Document(text=text)
                
                # Break the loop if decoding is successful
                break
            except UnicodeDecodeError:
                # If decoding fails, try the next encoding
                continue

        return document
    else:
        return None

def vector_indexing(document):
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
    )
    index = VectorStoreIndex.from_documents([document], service_context=service_context)
    return index

def query_index(index, query):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response

def main():
    st.title("RAG (Retrieval-Augmented Generation) System")
    
    uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'txt'])
    if uploaded_file is not None:
        document = read_doc(uploaded_file)
        st.write("Document uploaded successfully!")
        
        query = st.text_input("Enter your query:")
        if st.button("Generate Response"):
            index = vector_indexing(document)
            response = query_index(index, query)
            st.write("Response:")
            st.write(response)

if __name__ == "__main__":
    main()
