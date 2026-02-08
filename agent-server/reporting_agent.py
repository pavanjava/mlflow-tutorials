from typing import Any
from dotenv import load_dotenv, find_dotenv

#Agno
from agno.agent import Agent
from agno.models.openai import OpenAILike
from mlflow.genai import scorer

#mlflow
from mlflow.genai.agent_server import invoke
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.genai.scorers import Completeness, Guidelines

#llamaindex
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import mlflow

# Specify the tracking URI for the MLflow server.
mlflow.set_tracking_uri("http://localhost:3000")

# Specify the experiment you just created for your GenAI application.
mlflow.set_experiment("RAG Agent")

# Enable automatic tracing for all Agno agents.
mlflow.agno.autolog()

load_dotenv(find_dotenv())

Settings.embed_model = FastEmbedEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

qdrant_connector = QdrantClient(url="http://localhost:6333", api_key="th3s3cr3tk3y")

def context_retriever(user_query: str) -> Any:
    """Always use this tool to fetch the information about the GERD queries"""
    if qdrant_connector.collection_exists(collection_name="gastroenterology"):
        vector_store = QdrantVectorStore(client=qdrant_connector, collection_name="gastroenterology") # collection name should match the collection name while ingesting
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        retriever = index.as_retriever(top_k=15)
        return retriever.retrieve(user_query)
    else:
        # handle the fallback if not qdrant
        return None


agent = Agent(
    model=OpenAILike(id="my-chat", base_url="http://localhost:3000/gateway/mlflow/v1", api_key="dummy"),
    tools=[context_retriever],
    instructions="You are a Surgical Gastroenterologist based on the user query and context you should write a doctor report. "
                 "Write a detailed report for the user query from the provided context. Output in diagnosis or user symptoms report format. "
                 "Never hallucinate out of the context provided and always validate your response against the context and user query."
                 "Answer very politely that you can not answer any query out of GERD domain. Call the tools if really required",
    markdown=True,
)

@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    msgs = [i.model_dump() for i in request.input]
    question = msgs[0]['content']

    # run the agent
    result = await agent.arun(input=question)
    custom_op: Any = result.to_dict()

    # response evals
    data = [
              {
                  "inputs": {"question": question},
                  "outputs": result.content,
              },
           ]
    mlflow.genai.evaluate(data=data, scorers=[Completeness()])

    # retun the final response
    return ResponsesAgentResponse(custom_outputs=custom_op, output=[])
