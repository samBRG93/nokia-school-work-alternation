# Predicting London’s Weather: ML Model Lifecycle & Deployment

This project focuses on building a complete Machine Learning (ML) pipeline to 
predict the daily mean temperature in London. 
It incorporates DataOps (via Great Expectations) and MLOps (via MLflow), 
ensuring data quality, model tracking, and deployment.

 
## Source Data

We are using the London Weather Dataset available on Kaggle:
https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data
This dataset spans from 1979 to 2021 and has been pre-cleaned for ease of use.

## Data Operations (DataOps)

We utilize Great Expectations to validate data quality at different stages:
Raw Data Validation: Ensuring integrity before processing.
Baseline Data Validation: Checking cleaned data post-ETL (Extract, Transform, Load)  

Reference: https://docs.greatexpectations.io/docs/reference/learn/data_quality_use_cases/dq_use_cases_lp

## Machine Learning Model

Our primary model is XGBoost (Extreme Gradient Boosting), a high-performance model for tabular data. We will compare its performance against other models, such as Random Forest.

Reference: https://www.qwak.com/post/xgboost-versus-random-forest


## Machine Learning Operations (MlOps)

We use Mlflow to manage the ML lifecycle:
Track ML experiments and compare model parameters.
Version control to identify the best-performing models.

Reference: https://mlflow.org/docs/2.7.0/what-is-mlflow.html

## Mlflow Deployment

Mlflow will be used as a local inference server:
Reference: Reference: https://mlflow.org/docs/2.7.0/what-is-mlflow.html#:~:text=MLflow%20is%20a%20versatile%2C%20expandable,%2C%20algorithm%2C%20or%20deployment%20tool

## Data Visualization (Optional)
Using Matplotlib, we will create visualizations to gain insights from the dataset.

## Considerations & documentation

This is a regression problem, as we aim to predict a continuous variable (temperature).
You can learn more about regression from this link:
https://www.geeksforgeeks.org/regression-in-machine-learning/

Here are some key metrics we will use to evaluate the error of the XGBoost model:
https://dev.to/mondal_sabbha/understanding-mae-mse-and-rmse-key-metrics-in-machine-learning-4la2

If you're interested in learning more about MLOps, check out this resource:
https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/

Mlflow model as a local inference server: 
https://mlflow.org/docs/latest/deployment/deploy-model-locally.html

### Data challenges in Real-world Applications
While this dataset is pre-cleaned, real-world data is often:  
Noisy (contains errors or inconsistencies)
Dynamic (changes over time)  
Subject to data drift, which can degrade model performance over time.

Reference to data drift phenomenon:
https://towardsdatascience.com/machine-learning-in-production-why-you-should-care-about-data-and-concept-drift-d96d0bc907fb/

