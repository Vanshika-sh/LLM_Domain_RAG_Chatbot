from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import FunctionTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# load documents


#ROUTER ENGINE QUERY SYSTEM
documents = SimpleDirectoryReader(input_files=["/Users/vanshikasharma/Desktop/LLM_UPSKILL/docs/RAG_ResearchPaper.pdf"]).load_data()

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()


summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to MetaGPT"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the MetaGPT paper."
    ),
)


query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

response = query_engine.query("What is the summary of the document?")
print(str(response))

print(len(response.source_nodes))

response = query_engine.query(
    "How do agents share information with other agents?"
)
print(str(response))

from Code.rag_agent_utils import get_router_query_engine

query_engine = get_router_query_engine("/Users/vanshikasharma/Desktop/LLM_UPSKILL/docs/RAG_ResearchPaper.pdf")

response = query_engine.query("Tell me about the ablation study results?")
print(str(response))


# TOOL CALLING QUERY SYSTEM


def add(x: int, y: int) -> int:
    """Adds two integers together."""
    return x + y

def mystery(x: int, y: int) -> int: 
    """Mystery function that operates on top of two numbers."""
    return (x + y) * (x + y)


add_tool = FunctionTool.from_defaults(fn=add)
mystery_tool = FunctionTool.from_defaults(fn=mystery)

from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
response = llm.predict_and_call(
    [add_tool, mystery_tool], 
    "Tell me the output of the mystery function on 2 and 9", 
    verbose=True
)
print(str(response))

#AUTO-RETRIEVAL TOOL

from llama_index.core import SimpleDirectoryReader
# load documents
documents = SimpleDirectoryReader(input_files=["/Users/vanshikasharma/Desktop/LLM_UPSKILL/docs/RAG_ResearchPaper.pdf"]).load_data()

from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

print(nodes[0].get_content(metadata_mode="all"))

from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine(similarity_top_k=2)

from llama_index.core.vector_stores import MetadataFilters

query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=MetadataFilters.from_dicts(
        [
            {"key": "page_label", "value": "2"}
        ]
    )
)

response = query_engine.query(
    "What are some high-level results of the research paper?", 
)

print(str(response))
for n in response.source_nodes:
    print(n.metadata)

from typing import List
from llama_index.core.vector_stores import FilterCondition


def vector_query(
    query: str, 
    page_numbers: List[str]
) -> str:
    """Perform a vector search over an index.
    
    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.
    
    """

    metadata_dicts = [
        {"key": "page_label", "value": p} for p in page_numbers
    ]
    
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(
            metadata_dicts,
            condition=FilterCondition.OR
        )
    )
    response = query_engine.query(query)
    return response
    

vector_query_tool = FunctionTool.from_defaults(
    name="vector_tool",
    fn=vector_query
)

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
response = llm.predict_and_call(
    [vector_query_tool], 
    "What are the high-level results of MetaGPT as described on page 2?", 
    verbose=True
)

for n in response.source_nodes:
    print(n.metadata)


import time
import backoff
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# Function to handle exponential backoff for rate limit errors
def backoff_hdlr(details):
    print("Backing off {wait:0.1f} seconds after {tries} tries calling function {target} with args {args} and kwargs {kwargs}".format(**details))

# Exponential backoff decorator for rate limit handling
@backoff.on_exception(backoff.expo,
                      Exception,  # Catch any Exception and then check for rate limit specifics
                      max_tries=8,
                      on_backoff=backoff_hdlr,
                      giveup=lambda e: 'rate limit' not in str(e).lower())
def query_openai_engine(query_engine, query):
    return query_engine.query(query)

# Configure OpenAI settings
Settings.llm = OpenAI(model="gpt-3.5-turbo", retry=3)  # Specifying retry behavior
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Load and process documents
documents = SimpleDirectoryReader(input_files=["/Users/vanshikasharma/Desktop/LLM_UPSKILL/docs/RAG_ResearchPaper.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# Create indices and query engines
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
vector_query_engine = vector_index.as_query_engine()

# Define query engine tools
summary_tool = QueryEngineTool.from_defaults(query_engine=summary_query_engine, description="Useful for summarization questions related to MetaGPT")
vector_tool = QueryEngineTool.from_defaults(query_engine=vector_query_engine, description="Useful for retrieving specific context from the MetaGPT paper.")

# Configure and use RouterQueryEngine
query_engine = RouterQueryEngine(selector=LLMSingleSelector.from_defaults(), query_engine_tools=[summary_tool, vector_tool], verbose=True)
response = query_openai_engine(query_engine, "What is the summary of the document?")
print(str(response))

# Additional queries and tool calls
response = query_openai_engine(query_engine, "How do agents share information with other agents?")
print(str(response))





