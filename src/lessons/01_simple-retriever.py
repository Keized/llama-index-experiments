import helper
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters
from llama_index.core import Settings

# Llama default settings: https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings/


llm = OpenAI(model="gpt-3.5-turbo") # Use model gpt3.5
documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data() # Read the metagpt.pdf file and load data
splitter = SentenceSplitter(chunk_size=1024) # chunk each sentences (max size 1024)
nodes = splitter.get_nodes_from_documents(documents) # transform documents into nodes
vector_index = VectorStoreIndex(nodes) # Create avector store (by default use ada-embedding-02)
query_engine = vector_index.as_query_engine( # treansform the index to retriever
    similarity_top_k=2, # provide only the mmoste 2 relevants documents
    filters=MetadataFilters.from_dicts( # provide only document with metadata page_label set to 2
        [
            {"key": "page_label", "value": "2"}
        ]
    )
)

response = query_engine.query("What are some high-level results of MetaGPT?") # Run the query

if __name__ == "__main__":
    print(str(response))
    for n in response.source_nodes:
        print(n.metadata)
