import mlflow
from mlflow import log_metric, log_param, log_artifact

if __name__ == '__main__':
    for i in range(1, 5):
        with mlflow.start_run(run_name="run:" + str(i)):
            # set the experiment name
            mlflow.set_experiment("experiment"+ str(i))

            # log the experiment params (key, value)
            log_param("param-"+str(1), i)

            # log metrics
            log_metric("mae", i)
            log_metric("rmse", i)
            log_metric("mse", i)

            # log the artifacts
            with open("output.txt", "w") as f:
                f.write("hellow to mlflow"+str(i))
            log_artifact("output.txt")


