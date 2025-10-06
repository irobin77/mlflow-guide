import os
import tempfile
import mlflow
import warnings
import logging
import xgboost as xgb
import pandas as pd


# set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)


if __name__ == '__main__':

    logging.info('Model training started')
 
    mlflow.xgboost.autolog()

    with mlflow.start_run() as run:

        experiment_id = run.info.experiment_id
        
        run_id = run.info.run_id
        logging.info(f'Start mlflow run: {run_id}')
        
        # get last finished run for data preprocessing
        last_data_run_id = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName = 'Data_Preprocessing' and status = 'FINISHED'",
            order_by=["start_time DESC"]
        ).loc[0, 'run_id']
    
        # download train and test data from last run
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.artifacts.download_artifacts(run_id=last_data_run_id, artifact_path='datasets/train.csv', dst_path=tmpdir)
            mlflow.artifacts.download_artifacts(run_id=last_data_run_id, artifact_path='datasets/test.csv', dst_path=tmpdir)
            train = pd.read_csv(os.path.join(tmpdir, 'train.csv'))
            test = pd.read_csv(os.path.join(tmpdir, 'test.csv'))

        # convert to DMatrix format
        features = [i for i in train.columns if i != 'target']
        dtrain = xgb.DMatrix(data=train.loc[:, features], label=train['target'])
        dtest = xgb.DMatrix(data=test.loc[:, features], label=test['target'])

        # get last finished run for hyperparameters tuning
        last_tuning_run = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName = 'Hyperparameters_Tuning' and status = 'FINISHED'",
            order_by=["start_time DESC"]
        ).loc[0, :]
        
        # get best params
        params = {col.split('.')[1]: last_tuning_run[col] for col in last_tuning_run.index if 'params' in col}
        params.update(eval_metric=['auc', 'error'])

        mlflow.log_params(params)
        
        model = xgb.train(
            dtrain=dtrain,
            num_boost_round=int(params["num_boost_round"]),
            params=params,
            evals=[(dtest, 'test')],
            verbose_eval=False,
            early_stopping_rounds=10
        )

        mlflow.log_metric("accuracy", 1 - model.best_score)
        
        # Log model as Booster
        input_example = test.loc[0:10, features]
        predictions_example = pd.DataFrame(model.predict(xgb.DMatrix(input_example)), columns=['predictions'])
        mlflow.xgboost.log_model(xgb_model=model, name="booster", input_example=input_example)
        mlflow.log_text(predictions_example.to_json(orient='split', index=False), 'booster/predictions_example.json')

        # Register model
        model_uri = f"runs:/{run.info.run_id}/booster"
        mlflow.register_model(model_uri, 'CancerModelBooster')
        
        # Log model as sklearn completable XGBClassifier
        params.update(num_boost_round=model.best_iteration)
        model = xgb.XGBClassifier(**params)
        model.fit(train.loc[:, features], train['target'])
        mlflow.xgboost.log_model(xgb_model=model, name="model", input_example=input_example)

        # log datasets
        mlflow.log_text(train.to_csv(index=False), 'datasets/train.csv')
        mlflow.log_text(test.to_csv(index=False),'datasets/test.csv')

        logging.info('Model training finished')

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, 'CancerModel')
        
        logging.info('Model registered')
