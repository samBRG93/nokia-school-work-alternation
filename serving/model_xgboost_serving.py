import json
import requests
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from feature_engineering import FEAT_COLUMNS, TARGET_COLUMNS
from plots import plot_temperature


def simulate_global_warming(df):
    df["years_since_1979"] = df["year"] - 1979

    warming_factor = 0.02

    df["max_temp"] += df["years_since_1979"] * warming_factor
    df["mean_temp"] += df["years_since_1979"] * (warming_factor * 0.8)
    df["min_temp"] += df["years_since_1979"] * (warming_factor * 0.5)
    df["precipitation"] *= (1 + df["years_since_1979"] * 0.002)
    df["snow_depth"] *= (1 - df["years_since_1979"] * 0.001)

    df.drop(columns=["years_since_1979"], inplace=True)
    return df


def get_production_data():
    df = pd.read_csv("baseline_london_weather.csv")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    df = simulate_global_warming(df)

    df = df[df["date"].dt.year >= 2019]

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    X = df[FEAT_COLUMNS]
    y = df[TARGET_COLUMNS]

    return X, y

def execute_serving():
    df, y = get_production_data()
    x_axis = pd.to_datetime(df["date"])

    payload = json.dumps({"dataframe_split": df.to_dict(orient="split")})
    response = requests.post(
        url=f"http://127.0.0.1:5000/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    y_pred = response.json()['predictions']
    plot_temperature(x_axis, y, y_pred)

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    print(f"MSE: {mse} --- MAE: {mae}")


if __name__ == '__main__':
    execute_serving()