# mlflow-tutorials
this repository will focus on the mlflow framework and its capabilities

- before running the project please install mlflow as below

    -- `pip install --upgrade  mlflow`

    -- `pip install --upgrade mlflow[extras]` 
- to run the project navigate to src folder and run `python experiment.py`
- after that you should see **mlruns** folder getting created locally
- open terminal and navigate to src folder and run `mlflow ui`

# To make postgresql as backend store for mlflow registry

- run the following command `mlflow server --backend-store-uri postgresql://localhost:5432/mlflow --default-artifact-root ./mlruns --host 127.0.0.1 --port 3000`
- inorder to connect mlflow to postgres sql and internal dependency `psycopg2` is needed so install it using
    
    -- `pip install psycopg2`
