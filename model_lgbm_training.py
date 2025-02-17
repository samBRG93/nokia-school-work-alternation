from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import logging
import mlflow
from mlflow.models import infer_signature
from datetime import datetime
from feature_engineering import execute_feat_engineering, FEAT_COLUMNS, get_feature_importance, TARGET_COLUMNS
from plots import plot_scatter, plot_temperature
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)


def register_ml_flow_model(model, X_train, mse, mae, params):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    mlflow.set_experiment("LGBM Regression Model London Weather")

    with mlflow.start_run():
        mlflow.log_params(params)

        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("Mean Squared Error", mse)

        mlflow.log_table(data=get_feature_importance(X_train, model), artifact_file="feature_importance.json")

        mlflow.set_tag("Training Info", "LGBM Regression Model for London weather")

        signature = infer_signature(X_train, model.predict(X_train))

        _ = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="LGBM_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="LGBM_model",
        )


def main():
    df = execute_feat_engineering()

    X = df.drop(FEAT_COLUMNS, axis=1)
    y = df.drop(TARGET_COLUMNS, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    X_test_dates = X_test['date']
    X_train = X_train[FEAT_COLUMNS]
    X_test = X_test[FEAT_COLUMNS]

    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42,
        "eval_metric": "rmse"
    }

    model = lgb.LGBMRegressor(**params)

    model.fit(X_train,
              y_train,
              eval_set=
              [
                  (X_train, y_train),
                  (X_test, y_test)],
              )

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    register_ml_flow_model(model, X_train, mse, mae, params)

    plot_scatter(y_pred=y_pred, y_test=y_test)
    plot_temperature(x_axis=X_test_dates, y_test=y_test, y_pred=y_pred)

    run_id = mlflow.last_active_run().info.run_id
    run_name = mlflow.last_active_run().info.run_name

    print(f"Logged data and model in runId: {run_id} and runName: {run_name}"
          f"With Mean Square Error: {mse} and Mean Absolute Error: {mae} "
          f"from date: {datetime.strptime(str(X_test_dates.iloc[0]), '%Y-%m-%d %H:%M:%S')} "
          f"to date: {datetime.strptime(str(X_test_dates.iloc[-1]), '%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
