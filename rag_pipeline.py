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
import streamlit as st
openai.api_key = rag_utils.get_openai_api_key()

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from Code.rag_utils import build_automerging_index
from Code.rag_utils import get_automerging_query_engine
from datasets import Dataset 
from ragas.metrics import *
from ragas import evaluate

import PyPDF2

def vector_indexing(document):
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model="local:BAAI/bge-small-en-v1.5",system_prompt="You are at answering questions about pdf documents. Assume that all questions are related to only the pdf document that has been uploaded. Keep your answers technical and based on facts – do not hallucinate features. If any detail is not present in the document, do not include that in your response, please say that this detail is not present in the given document"
    )
    index = VectorStoreIndex.from_documents([document],
                                            service_context=service_context)

    return index

def query_index(index,query):

    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    # in the given document, what is told about the LLAMA model in the Background section? elaborate answer with all the details
    # in the given document, generate 10 evaluation questions from different sections of the given paper
    return str(response)


def eval_basic_rag(query_engine,eval_questions):
    
    for line in eval_questions:
        # Remove newline character and convert to integer
        item = line.strip()
        print(item)
        eval_questions.append(item)
    tru = Tru()

    tru.reset_database()
    tru_recorder = get_prebuilt_trulens_recorder(query_engine,app_id="Direct Query Engine")

    with tru_recorder as recording:
        for question in eval_questions:
            response = query_engine.query(question)

    records, feedback = tru.get_records_and_feedback(app_ids=[])
    tru.get_leaderboard(app_ids=[])
    # launches on http://localhost:8501/
    # tru.run_dashboard()
    return records


#SENTENCE WINDOW RETRIEVAL TO IMPROVE CONTEXT RELEVANCE WINDOW

def sentence_window_retrieval(document):
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    sentence_index = build_sentence_window_index(
        document,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        save_dir="sentence_index"
    )
    return sentence_index



def sentence_window_response(sentence_index,query):
    sentence_window_engine = get_sentence_window_query_engine(sentence_index)

    window_response = sentence_window_engine.query(query)
    print(str(window_response))

    return str(window_response),sentence_window_engine


def eval_sentence_window_retrieval(sentence_window_engine,uploaded_evals):
    
    eval_questions = []
   
    for line in uploaded_evals:
        # Remove newline character and convert to integer
        item = line
        print(item)
        eval_questions.append(item)

    tru = Tru()
    tru.reset_database()

    tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
        sentence_window_engine,
        app_id = "Sentence Window Query Engine"
    )

    for question in eval_questions:
        with tru_recorder_sentence_window as recording:
            response = sentence_window_engine.query(question)
            print(question)
            print(str(response))
    records, feedback = tru.get_records_and_feedback(app_ids=[])
    tru.get_leaderboard(app_ids=[])

    # tru.run_dashboard()

    return records,feedback

def automerging_index_func(documents):
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    automerging_index = build_automerging_index(
        documents,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        save_dir="merging_index"
    )
    return automerging_index

def automerging_engine(automerging_index,query):
    automerging_query_engine = get_automerging_query_engine(
    automerging_index,
    )
    auto_merging_response = automerging_query_engine.query(query)
    print(str(auto_merging_response))

    return str(auto_merging_response),automerging_query_engine

def eval_automerging_retrieval(automerging_query_engine,uploaded_evals):
    eval_questions = []
    
    for line in uploaded_evals:
        # Remove newline character and convert to integer
        item = line
        print(item)
        eval_questions.append(item)

    tru = Tru()
    tru.reset_database()

    tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                                                         app_id="Automerging Query Engine")

    for question in eval_questions:
        with tru_recorder_automerging as recording:
            response = automerging_query_engine.query(question)
            print(question)
            print(response)
    records, feedback = tru.get_records_and_feedback(app_ids=[])
    tru.get_leaderboard(app_ids=[])

    # launches on http://localhost:8501/
    # tru.run_dashboard()

    return records,feedback

def convert_uploaded_pdf_to_document(uploaded_pdf):
    if uploaded_pdf is not None:
        try:
            # Initialize PyPDF2 reader
            pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
            pdf_text = ""

            # Read text from all pages
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                pdf_text += page.extract_text()

            # Create a Document object
            document = Document(text=pdf_text)
            return document
        except Exception as e:
            st.error("Error: {}".format(str(e)))  # Corrected line
            return None
    else:
        st.error("No PDF file uploaded yet.")
        return None


def main():
    
    # Read document
    st.title("RAG (Retrieval-Augmented Generation) System")
    uploaded_file = st.file_uploader("Upload Document", type=['pdf'])
    
    pdf = convert_uploaded_pdf_to_document(uploaded_file)
    pdf_auto = [pdf]
    if uploaded_file is not None:
        print('type_streamlit------',type(pdf))
        st.write("Document uploaded successfully!")
        # document = read_doc2(uploaded_file) 

        query = st.text_input("Enter your query:")
        query_context = ('Generate a short context for the question that was asked')
        eval_ques = st.text_area("Questions (one question per line)", height=200)
        questions_list = eval_ques.split('\n') if eval_ques else []
        # for i, question in enumerate(questions_list):
        #         st.write(f"{i+1}. {question}")
        
        print(type(query))
        if st.button("Generate Response and Evaluate RAG"):
            index = vector_indexing(pdf)
            basic_result = query_index(index,query)
            generated_context = query_index(index,query_context)
            st.write("Basic RAG Response:")
            st.info(basic_result)

            data_samples = {
            'question': [query],
            'answer': [basic_result],
            'contexts' : [[generated_context]],
            }
            dataset = Dataset.from_dict(data_samples)
            score = evaluate(dataset,metrics=[faithfulness])
            score.to_pandas()
            print('score---------------',score)
            st.write('Context')
            st.info(generated_context)
            st.write('RAGAS Faithfulness score: ')
            st.info(score)

            sentence_index = sentence_window_retrieval(pdf)
            sentence_window_result,sentence_window_engine = sentence_window_response(sentence_index,query)
            st.write("Sentence Window RAG Response:")
            st.info(sentence_window_result)
            automerging_index = automerging_index_func(pdf_auto)
            automerging_index_result, automerging_query_engine=automerging_engine(automerging_index,query)
            st.write("Automerging Index RAG Response:")
            st.info(automerging_index_result)

            st.write("Results for Basic RAG Retrieval:")

            st.write("Results for Sentence Window Retrieval:")
            rec_sent, feedback_sent = eval_sentence_window_retrieval(sentence_window_engine, questions_list)
            
            st.dataframe(rec_sent)#['input','output','Answer Relevance','Context Relevance','Groundedness'])
    
            st.write("Results for Automatic Merging Retrieval:")
            rec_auto, feedback_auto = eval_automerging_retrieval(automerging_query_engine, questions_list)
            st.dataframe(rec_auto)#['input','output','Answer Relevance','Context Relevance','Groundedness'])
            

        

            


            
            
            
            


    # # Perform vector indexing
    # index = vector_indexing(document)

    # # Query the index
    # query_result = query_index(index)
    # print("Query Result:", query_result)

    # Evaluate basic RAG
    
    
    # print("Basic RAG Records:", basic_rag_records)

    # Perform sentence window retrieval
    # sentence_index = sentence_window_retrieval(document,uploaded_evals)

    # # Query the sentence window index
    # sentence_window_result,sentence_window_engine = sentence_window_response(sentence_index,uploaded_evals)
    # # print("Sentence Window Result---------:", sentence_window_result)

    # automerging_index = automerging_index_func(documents)
    # response, automerging_query_engine=automerging_engine(automerging_index)
    # print("AutoMerging Result---------:", sentence_window_result)

    

    # Evaluate sentence window retrieval
    

    


if __name__ == "__main__":
    main()

