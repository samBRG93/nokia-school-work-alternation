from feature_engineering import execute_feat_engineering
from gx_data_validation import run_great_expectations_raw, run_great_expectations_baseline
from model_xgboost_training import execute_training
from serving.model_xgboost_serving import execute_serving


def execute():
    run_great_expectations_raw()
    df = execute_feat_engineering()
    df.to_csv('baseline_london_weather.csv')
    run_great_expectations_baseline()
    execute_training()
    execute_serving()


execute()