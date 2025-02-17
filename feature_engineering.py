import pandas
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)

NUMERICAL_FEAT_COLUMNS = ['cloud_cover', 'sunshine', 'global_radiation', 'precipitation', 'pressure', 'snow_depth']
TARGET_COLUMNS = ["mean_temp"]

FEAT_COLUMNS = [
    "date",
    "year",
    "month",
    "day",
    "cloud_cover",
    "sunshine",
    "global_radiation",
    "precipitation",
    "pressure",
    "snow_depth",
    "rolling_5days_cloud_cover",
    "rolling_5days_sunshine",
    "rolling_5days_global_radiation",
    "rolling_5days_precipitation",
    "rolling_5days_pressure",
    "rolling_5days_snow_depth",
    "rolling_2days_cloud_cover",
    "rolling_2days_sunshine",
    "rolling_2days_global_radiation",
    "rolling_2days_precipitation",
    "rolling_2days_pressure",
    "rolling_2days_snow_depth"
]


def normalize_data(X, scaler: MinMaxScaler = None):
    try:
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(X[NUMERICAL_FEAT_COLUMNS])

        X[NUMERICAL_FEAT_COLUMNS] = scaler.transform(X[NUMERICAL_FEAT_COLUMNS])
    except Exception:
        logging.exception("Error while normalizing data")
        raise
    return X, scaler


def get_feature_importance(data: pandas.DataFrame, model):
    importance = model.feature_importances_
    features = data.columns
    df_feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
    print(f'Feature Importance: {df_feature_importance.sort_values(by='Importance', ascending=False)}')

    return df_feature_importance


def execute_feat_engineering(date_upper_bound=None, date_lower_bound=None) -> pd.DataFrame:
    df = pd.read_csv("london_weather.csv")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    if date_lower_bound:
        df = df[df["date"].dt.year >= date_lower_bound]
    if date_upper_bound:
        df = df[df["date"].dt.year < date_upper_bound]

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df["mean_temp"] = df[df["mean_temp"].isna()].apply(lambda x: (df["max_temp"] + df["min_temp"]) / 2)['mean_temp']
    df["mean_temp"] = df["mean_temp"].fillna(df["min_temp"])
    df["mean_temp"] = df["mean_temp"].fillna(df["max_temp"])

    df['cloud_cover'] = df['cloud_cover'].fillna(df["cloud_cover"].mean())
    df['global_radiation'] = df['global_radiation'].fillna(df["global_radiation"].mean())
    df['pressure'] = df['pressure'].fillna(df["pressure"].mean())
    df['snow_depth'] = df['snow_depth'].fillna(df["snow_depth"].mean())

    df = df.dropna()
    df.reset_index(inplace=True, drop=True)

    df['rolling_5days_cloud_cover'] = df['cloud_cover'].rolling(window=5).mean()
    df['rolling_5days_sunshine'] = df['sunshine'].rolling(window=5).mean()
    df['rolling_5days_global_radiation'] = df['global_radiation'].rolling(window=5).mean()
    df['rolling_5days_precipitation'] = df['precipitation'].rolling(window=5).mean()
    df['rolling_5days_pressure'] = df['pressure'].rolling(window=5).mean()
    df['rolling_5days_snow_depth'] = df['snow_depth'].rolling(window=5).mean()

    df['rolling_2days_cloud_cover'] = df['cloud_cover'].rolling(window=2).mean()
    df['rolling_2days_sunshine'] = df['sunshine'].rolling(window=2).mean()
    df['rolling_2days_global_radiation'] = df['global_radiation'].rolling(window=2).mean()
    df['rolling_2days_precipitation'] = df['precipitation'].rolling(window=2).mean()
    df['rolling_2days_pressure'] = df['pressure'].rolling(window=2).mean()
    df['rolling_2days_snow_depth'] = df['snow_depth'].rolling(window=2).mean()

    df.fillna(df.mean(), inplace=True)

    return df
