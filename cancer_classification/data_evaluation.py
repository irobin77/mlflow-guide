# data_evaluation
import argparse
import warnings
import logging
import mlflow
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)


if __name__ == '__main__':

    logging.info('Evaluation started')

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dataset", type=str)
    eval_dataset = pd.read_csv(parser.parse_args().eval_dataset)
        
    with mlflow.start_run() as run:
        
        eval_dataset = mlflow.data.from_pandas(
            eval_dataset, targets="target"
        )
        last_version = mlflow.MlflowClient().get_registered_model('CancerModel').latest_versions[0].version
        mlflow.evaluate(
            data=eval_dataset, model_type="classifier", model=f'models:/CancerModel/{last_version}'
        )
        logging.info('Evaluation finished')
