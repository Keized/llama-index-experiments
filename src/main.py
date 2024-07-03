from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, agent
from llama_index.core.agent import AgentRunner
from llama_index.readers.web import SimpleWebPageReader, BeautifulSoupWebReader

documents = BeautifulSoupWebReader().load_data(urls=['https://lespotevry.fr'])
vector_index = VectorStoreIndex.from_documents(documents)
query_engine = vector_index.as_query_engine()
response = query_engine.query("De quoi parle le site ?")
print(response)
for r in response.source_nodes:
    print(r)


"""
documents = SimpleDirectoryReader('./data').load_data(show_progress=True)

index = VectorStoreIndex.from_documents(documents)
queryEngine = index.as_query_engine()
chatEngine = index.as_query_engine()


response = queryEngine.query('resume moi le document en francais en quelques lignes')
response2 = chatEngine.query('quelle est la capital de la france')

print(response)
print('\n')
print('\n')

print('\n')
print('\n')

print(response2)

"""