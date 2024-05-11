from Code.rag_agent_utils import get_doc_tools
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
import streamlit as st

st.title("Document Analysis App")
uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')

if uploaded_file is not None:
    # Save the uploaded PDF to a temporary file

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

    # response = agent.query(
    #     "Tell me about the projects Vanshika did, "
    #     "and then how she did them."
    # )
    # user_input = st.text_input("Ask a question about the document (type 'exit' to finish):")

    # print(response.source_nodes[0].get_content(metadata_mode="all"))

    # response = agent.chat(
    #     "Tell me about the evaluation datasets used."
    # )

    # response = agent.chat("Tell me the results over one of the above datasets.")

    while True:
        

        user_input = st.text_input("Ask a question about the document (type 'exit' to finish):", key="unique_key")
        # user_input = st.text_input("Ask a question about the document (type 'exit' to finish):")
        if user_input == "exit":
            break
        response = agent.chat(user_input)
        st.write(f"Agent: {response}")
        print(f"Agent: {response}")