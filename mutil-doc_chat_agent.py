import streamlit as st
from utils.rag_agent_utils import get_doc_tools
from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core.objects import ObjectIndex
import os

st.title("RAG Chatbot")

# Input for OpenAI API key
api_key = st.text_input("Enter your OpenAI API key:", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key  # Set API key in the environment
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)  # Use the API key in the model initialization
else:
    st.warning("Please enter your OpenAI API key.")
    st.stop()


# Allow users to upload multiple PDF files
uploaded_files = st.file_uploader("Choose PDF files", type=['pdf'], accept_multiple_files=True)

# Assuming llm is already defined and configured, if not, define or import llm
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)  # Example placeholder

if uploaded_files:
    paper_to_tools_dict = {}
    for uploaded_file in uploaded_files:
        # Save uploaded file to disk
        with open(Path(uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Get document tools for each uploaded document
        vector_tool, summary_tool = get_doc_tools(Path(uploaded_file.name).as_posix(), Path(uploaded_file.name).stem)
        paper_to_tools_dict[uploaded_file.name] = [vector_tool, summary_tool]

    # Creating an object index and retriever from these tools
    all_tools = [tool for tools in paper_to_tools_dict.values() for tool in tools]
    obj_index = ObjectIndex.from_objects(all_tools, index_cls=VectorStoreIndex)
    obj_retriever = obj_index.as_retriever(similarity_top_k=3)

    # Initialize conversation history
    if 'conversations' not in st.session_state:
        st.session_state.conversations = []

    # Function to handle user question submission
    def submit_query():
        query = st.session_state.query_input
        tools = obj_retriever.retrieve(query)
        agent_worker = FunctionCallingAgentWorker.from_tools(
            tool_retriever=obj_retriever,
            llm=llm,
            system_prompt=""" \
            You are an agent designed to answer queries over a set of given papers.
            Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
            """,
            verbose=True
        )
        agent = AgentRunner(agent_worker)
        response = agent.query(query)
        st.session_state.conversations.append(f"User: {query}")
        st.session_state.conversations.append(f"Agent: {response}")
        st.session_state.query_input = ""

    # Text input for user queries
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""

    # Display all past conversations using markdown for automatic text wrapping
    for exchange in st.session_state.conversations:
        st.markdown(f"**{exchange}**")

    # Create a text input field that allows the user to enter their question
    user_input = st.text_input("Ask a question about the documents:", 
                               value=st.session_state.query_input, 
                               key="query_input",
                               on_change=submit_query)

    # Submit button
    submit_button = st.button("Submit", on_click=submit_query)


