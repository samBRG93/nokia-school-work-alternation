import great_expectations as gx
import pandas as pd


def validate_data(df: pd.DataFrame, data_source_name, data_asset_name, data_batch_name, suite_name):
    context = gx.data_context.get_context()

    data_source = context.get_datasource(name=data_source_name)
    data_asset = data_source.get_asset(name=data_asset_name)
    batch_definition = data_asset.get_batch_definition(name=data_batch_name)

    batch_parameters = {"dataframe": df}
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    suite = context._get_expectation_suite_from_inputs(expectation_suite_name=suite_name)
    validation_results = batch.validate(suite)

    if not validation_results["success"]:
        raise ValueError("Data validation failed!")

    print(f"Validation results: {validation_results}")


def get_raw_expectations():
    return [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="date", mostly=1),
        gx.expectations.ExpectColumnValuesToBeIncreasing(column="date", mostly=1),

        gx.expectations.ExpectColumnValuesToNotBeNull(column="max_temp", mostly=1),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="mean_temp", mostly=1),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="min_temp", mostly=1),
        gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(column_A="max_temp", column_B="min_temp", mostly=1,
                                                                or_equal=True),
        gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(column_A="mean_temp", column_B="min_temp", mostly=1,
                                                                or_equal=True),
        gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(column_A="max_temp", column_B="mean_temp", mostly=1,
                                                                or_equal=True),

        gx.expectations.ExpectColumnValuesToBeBetween(column="mean_temp", min_value=-50, max_value=60),
        gx.expectations.ExpectColumnValuesToBeBetween(column="precipitation", min_value=0, max_value=100),
        gx.expectations.ExpectColumnValuesToBeBetween(column="snow_depth", min_value=0, max_value=50),
        gx.expectations.ExpectColumnValuesToBeBetween(column="sunshine", min_value=0, max_value=24),
        gx.expectations.ExpectColumnValuesToBeBetween(column="cloud_cover", min_value=0, max_value=24),

        gx.expectations.ExpectColumnValuesToNotBeNull(column="precipitation", mostly=1),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="snow_depth", mostly=1),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="sunshine", mostly=1),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="cloud_cover", mostly=1),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="pressure", mostly=1),

        gx.expectations.ExpectColumnValuesToBeBetween(column="pressure",
                                                      min_value=95000,
                                                      max_value=105000,
                                                      mostly=1),
    ]


def get_baseline_expectations():
    return [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="date", mostly=1),
        gx.expectations.ExpectColumnValuesToBeBetween(column="mean_temp", min_value=-50, max_value=60),
        gx.expectations.ExpectColumnValuesToBeBetween(column="precipitation", min_value=0, max_value=100),
        gx.expectations.ExpectColumnValuesToBeBetween(column="snow_depth", min_value=0, max_value=50),
        gx.expectations.ExpectColumnValuesToBeBetween(column="sunshine", min_value=0, max_value=24),
        gx.expectations.ExpectColumnValuesToBeBetween(column="cloud_cover", min_value=0, max_value=24),
        gx.expectations.ExpectColumnToExist(column="rolling_5days_cloud_cover"),
        gx.expectations.ExpectColumnToExist(column="rolling_5days_sunshine"),
        gx.expectations.ExpectColumnToExist(column="rolling_5days_global_radiation"),
        gx.expectations.ExpectColumnToExist(column="rolling_5days_precipitation"),
        gx.expectations.ExpectColumnToExist(column="rolling_5days_pressure"),
        gx.expectations.ExpectColumnToExist(column="rolling_5days_snow_depth"),
        gx.expectations.ExpectColumnToExist(column="rolling_2days_cloud_cover"),
        gx.expectations.ExpectColumnToExist(column="rolling_2days_sunshine"),
        gx.expectations.ExpectColumnToExist(column="rolling_2days_global_radiation"),
        gx.expectations.ExpectColumnToExist(column="rolling_2days_precipitation"),
        gx.expectations.ExpectColumnToExist(column="rolling_2days_global_radiation"),
        gx.expectations.ExpectColumnToExist(column="rolling_2days_snow_depth")
    ]


def init_great_expectations(df: pd.DataFrame,
                            expectations: list,
                            data_source_name: str,
                            data_asset_name: str,
                            data_batch_name: str,
                            suite_name: str
                            ):
    context = gx.get_context(mode='file')

    data_source = context.data_sources.add_or_update_pandas(name_or_datasource=data_source_name)
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)
    batch_definition = data_asset.add_batch_definition_whole_dataframe(data_batch_name)

    expectation_suite = gx.ExpectationSuite(
        name=suite_name,
        expectations=expectations,
    )

    batch_parameters = {"dataframe": df}
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    print(batch.head())

    validation_definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name="vd", data=batch_definition, suite=expectation_suite))

    cp = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name="checkpoint",
            validation_definitions=[validation_definition],
            actions=[gx.checkpoint.actions.UpdateDataDocsAction(name="action")],
        )
    )

    cp.run(batch_parameters=batch_parameters)
    context.open_data_docs()


def run_great_expectations(df, data_source_name, data_asset_name, data_batch_name, suite_name, expectations):
    init_great_expectations(
        expectations=expectations,
        df=df,
        data_source_name=data_source_name,
        data_asset_name=data_asset_name,
        data_batch_name=data_batch_name,
        suite_name=suite_name
    )

    validate_data(
        df=df,
        data_source_name=data_source_name,
        data_asset_name=data_asset_name,
        data_batch_name=data_batch_name,
        suite_name=suite_name
    )


def run_great_expectations_raw():
    raw_data_source = 'RawCSVFile'
    raw_data_asset = "RawCSVFile"
    raw_data_batch = "RawCSVFile"
    raw_suite = "RawSuite"
    raw_csv_file = "london_weather.csv"

    df = pd.read_csv(raw_csv_file)
    expectations = get_raw_expectations()
    run_great_expectations(expectations=expectations, df=df, data_source_name=raw_data_source,
                           data_asset_name=raw_data_asset,
                           data_batch_name=raw_data_batch, suite_name=raw_suite)


def run_great_expectations_baseline():
    baseline_data_source = 'BaselineCSVFile'
    baseline_data_asset = "BaselineCSVFile"
    baseline_data_batch = "BaselineCSVFile"
    baseline_suite = "BaselineSuite"
    baseline_csv_file = "baseline_london_weather.csv"

    df = pd.read_csv(baseline_csv_file)
    expectations = get_baseline_expectations()
    run_great_expectations(expectations=expectations, df=df, data_source_name=baseline_data_source,
                           data_asset_name=baseline_data_asset,
                           data_batch_name=baseline_data_batch, suite_name=baseline_suite)


if __name__ == "__main__":
    run_great_expectations_raw()

    # df = pd.read_csv("london_weather.csv")

    # init_great_expectations(
    #     data_source_name=RAW_DATA_SOURCE,
    #     data_asset_name=RAW_DATA_ASSET,
    #     data_batch_name=RAW_DATA_BATCH,
    #     suite_name=RAW_SUITE
    # )
    #
    # validate_data(
    #     data_source_name=RAW_DATA_SOURCE,
    #     data_asset_name=RAW_DATA_ASSET,
    #     data_batch_name=RAW_DATA_BATCH,
    #     suite_name=RAW_SUITE
    # )
