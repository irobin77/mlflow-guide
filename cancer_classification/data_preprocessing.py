import argparse
import mlflow
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings('ignore')


def get_cancer_df():
    cancer = datasets.load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target)
    logging.info(f'Cancer data downloaded')
    
    return X, y


def main():
    TEST_SIZE = 0.3
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", default=TEST_SIZE, type=float)
    TEST_SIZE = parser.parse_args().test_size
        
    logging.info(f'Data preprocessing started with test size: {TEST_SIZE}')
    
    # create or use an experiment
    experiment_id = mlflow.set_experiment('Cancer_Classification').experiment_id
    
    with mlflow.start_run(run_name='Data_Preprocessing'):
            
        # download cancer dataset
        X, y = get_cancer_df()
    
        # add additional features
        X['additional_feature'] = X['mean symmetry'] / X['mean texture']
        logging.info('Additional features added')
    
        # log dataset size and features count
        mlflow.log_metric('full_data_size', X.shape[0])
        mlflow.log_metric('features_count', X.shape[1])
    
        # split dataset to train and test part and log sizes to mlflow
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
        mlflow.log_metric('train_size', X_train.shape[0])
        mlflow.log_metric('test_size', X_test.shape[0])
        
        # log and register datasets
        train = X_train.assign(target=y_train)
        mlflow.log_text(train.to_csv(index=False),'datasets/train.csv')
        dataset_source_link = mlflow.get_artifact_uri('datasets/train.csv')
        dataset = mlflow.data.from_pandas(train, name='train', targets="target", source=dataset_source_link)
        mlflow.log_input(dataset)

        test = X_test.assign(target=y_test)
        mlflow.log_text(test.to_csv(index=False),'datasets/test.csv')
        dataset_source_link = mlflow.get_artifact_uri('datasets/test.csv')
        dataset = mlflow.data.from_pandas(train, name='test', targets="target", source=dataset_source_link)
        mlflow.log_input(dataset)
        
        logging.info('Data preprocessing finished')


if __name__ == '__main__':
    main()
    