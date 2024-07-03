import helper
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.tools import FunctionTool
from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool
from typing import List

llm = OpenAI(model="gpt-3.5-turbo")
documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

"""Create Summary Index and retriever"""
summary_index = SummaryIndex(nodes)

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)

summary_tool = QueryEngineTool.from_defaults(
    name="summary_tool",
    query_engine=summary_query_engine,
    description=(
        "Useful if you want to get a summary of MetaGPT"
    ),
)

"""Create Vector index and retriever tool"""
vector_index = VectorStoreIndex(nodes)
def vector_query(query: str, page_numbers: List[str]):
    """Perform a vector search over an index.
    
    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.
    
    """
    metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]
    
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

response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "What is a summary of the paper?", 
    verbose=True
)

if __name__ == "__main__":
    print(str(response))
    for n in response.source_nodes:
        print(n.metadata)
