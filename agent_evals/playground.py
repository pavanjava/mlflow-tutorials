import mlflow
import pandas as pd
from mlflow.genai.scorers import Correctness, Safety
from agno.agent import Agent
from agno.models.openai import OpenAILike

def execute_agent(query: str):
    agent = Agent(
        model=OpenAILike(id="", base_url="")
    )
    agent.run(input=query)

# Create evaluation data as a Pandas DataFrame
eval_df = pd.DataFrame(
    [
        {
            "inputs": {"question": "What is MLflow?"},
            "expectations": {
                "expected_response": "MLflow is an open-source platform for ML lifecycle management"
            },
        },
        {
            "inputs": {"question": "How do I log metrics?"},
            "expectations": {
                "expected_response": "Use mlflow.log_metric() to log metrics"
            },
        },
    ]
)

# Run evaluation
results = mlflow.genai.evaluate(
    data=eval_df,
    predict_fn=execute_agent,
    scorers=[Correctness(), Safety()],
)
