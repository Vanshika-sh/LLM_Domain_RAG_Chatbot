import streamlit as st
from utils.rag_agent_utils import get_doc_tools
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
import os

st.title("Document Analysis App")

# Inject custom CSS to change the font in the app
st.markdown("""
<style>
    .markdown-text-container {
        font-family: 'Garamond', serif; /* Example: Garamond */
    }
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')
if uploaded_file is not None:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    vector_tool, summary_tool = get_doc_tools("temp_uploaded.pdf", "surveygpt")
    agent_worker = FunctionCallingAgentWorker.from_tools(
        [vector_tool, summary_tool], 
        llm=llm, 
        verbose=True
    )
    agent = AgentRunner(agent_worker)

    st.write("PDF uploaded successfully. You can now start asking questions about the document.")

    if 'conversations' not in st.session_state:
        st.session_state.conversations = []

    # Display all past conversations using markdown for automatic text wrapping
    for exchange in st.session_state.conversations:
        st.markdown(f"**{exchange}**", unsafe_allow_html=True)

    # Function to handle user question submission
    def submit_question():
        user_input = st.session_state.query_input
        if user_input.lower() == "exit":
            st.stop()
        else:
            response = agent.chat(user_input)
            formatted_user_input = f"User: {user_input}"
            formatted_response = f"Agent: {response}"
            st.session_state.conversations.append(formatted_user_input)
            st.session_state.conversations.append(formatted_response)
            st.session_state.query_input = ""  # Clear the input for a new query

    # Initialize the query input if it doesn't exist
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""

    # Create a text input field that allows the user to enter their question
    user_input = st.text_input("Ask a question about the document:", 
                               value=st.session_state.query_input, 
                               key="query_input",
                               on_change=submit_question)
                               

    # Submit button
    submit_button = st.button("Submit", on_click=submit_question)









