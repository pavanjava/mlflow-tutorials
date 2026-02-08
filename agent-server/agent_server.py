# Need to import the agent to register the functions with the server
import reporting_agent  # noqa: F401
from mlflow.genai.agent_server import AgentServer

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app


def main():
    # To support multiple workers, pass the app as an import string
    agent_server.run(app_import_string="agent_server:app")


if __name__ == "__main__":
    main()