from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature
from datetime import datetime

from sklearn.neural_network import MLPRegressor

from feature_engineering import execute_feat_engineering, FEAT_COLUMNS, normalize_data, TARGET_COLUMNS
from plots import plot_temperature, plot_scatter


def register_ml_flow_model(model, X_train, mse, mae, params):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    mlflow.set_experiment("MLP Regression Model London Weather")

    with mlflow.start_run():
        mlflow.log_params(params)

        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("Mean Squared Error", mse)

        mlflow.sklearn.log_model(model, "model")

        mlflow.set_tag("Training Info", "MLP Regressor Model for London weather")

        signature = infer_signature(X_train, model.predict(X_train))

        _ = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="mlp_regressor_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="mlp_regressor_model",
        )


def main():
    df = execute_feat_engineering(date_upper_bound=2013)

    X = df.drop(FEAT_COLUMNS, axis=1)
    y = df.drop(TARGET_COLUMNS, axis=1)

    X, _ = normalize_data(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    mlflow.xgboost.autolog()

    X_test_dates = X_test['date']
    X_train = X_train[FEAT_COLUMNS]
    X_test = X_test[FEAT_COLUMNS]

    params = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 500,
        "random_state": 42
    }

    model = MLPRegressor(**params)

    model.fit(X_train,
              y_train,
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
