import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns

import math
import scipy.stats
from scipy.stats import norm
from scipy.stats import chisquare
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score

import xlrd
from collections import Counter



################################################################################################################################################################################################
#####################################################                     data prep                           ##################################################################################
################################################################################################################################################################################################



def load_data(file_path, sheet_name=0):
    """
    Load data from an Excel or CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the file.
    sheet_name (str or int): The name or index of the sheet to load (for Excel files). Default is 0 (the first sheet).
    
    Returns:
    pd.DataFrame: The loaded DataFrame, or None if an error occurred.
    """
    try:
        if file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print("Excel file imported successfully.")
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, index_col=0)
            print("file imported successfully.")
        else:
            print("Unsupported file format. Please provide a .xls, .xlsx, or .csv file.")
            return None
        return df
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None




def impute_monthly_income(df):
    """
    Impute missing values in the MonthlyIncome column using a Random Forest Regressor with log transformation.
    
    This function trains a Random Forest Regressor on the rows with known MonthlyIncome values
    using the specified features. It then predicts and imputes the missing MonthlyIncome values 
    for the rows where it is missing, while preserving the original MonthlyIncome column and creating
    a new column called ImputedMonthlyIncome.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the MonthlyIncome column and other features.
    
    Returns:
    pd.DataFrame: The DataFrame with an additional column ImputedMonthlyIncome containing the imputed values.
    """
    
    # Separate the data into rows with and without missing MonthlyIncome
    df_train = df[df['MonthlyIncome'].notna()].copy()
    df_missing = df[df['MonthlyIncome'].isna()].copy()
    
    # Check if there are any missing values to impute
    if df_missing.empty:
        print("No missing values to impute.")
        # Create the ImputedMonthlyIncome column as a copy of MonthlyIncome
        df['ImputedMonthlyIncome'] = df['MonthlyIncome']
        return df
    
    # Apply log transformation to MonthlyIncome to handle skewness
    df_train.loc[:, 'LogMonthlyIncome'] = np.log1p(df_train['MonthlyIncome'])
    
    # Predictors
    features = ['age', 'NumberOfDependents', 'NumberRealEstateLoansOrLines']
    
    # Train model to predict LogMonthlyIncome
    X_train = df_train[features]
    y_train = df_train['LogMonthlyIncome']
    
    # Initialize the Random Forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict missing LogMonthlyIncome values
    X_missing = df_missing[features]
    y_missing_pred_log = model.predict(X_missing)
    
    # Ensure predicted log values are within a reasonable range to prevent overflow
    max_log_value = np.log1p(df['MonthlyIncome'].max())
    y_missing_pred_log = np.clip(y_missing_pred_log, None, max_log_value)
    
    # Reverse log transformation to get imputed MonthlyIncome values
    y_missing_pred = np.expm1(y_missing_pred_log)

    # Determine the number of decimal places in the original MonthlyIncome column
    num_decimals = df['MonthlyIncome'].apply(lambda x: len(str(x).split('.')[1]) if pd.notnull(x) else 0).max()
    
    # Round the imputed values to match the number of decimal places in the original column
    y_missing_pred = np.round(y_missing_pred, num_decimals)
    
    # Create new ImputedMonthlyIncome column
    df['ImputedMonthlyIncome'] = df['MonthlyIncome']
    df.loc[df['MonthlyIncome'].isna(), 'ImputedMonthlyIncome'] = y_missing_pred
    
    return df


def metric_characterisation(data, var, confidence):
    """
    Performs statistical calculations to check general behavior of the metric.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.
    - var (str): The column name in the DataFrame containing the metric.
    - confidence (float): The desired confidence level (e.g., 0.95 for 95% confidence).

    Outputs:
    - Prints the expectation (mean), standard deviation, confidence intervals, and effect sizes.
    - Displays time series and distribution plots of the metric.
    """
    # Metric array
    a = 1.0 * np.array(data[var].dropna())
    # Sample size
    n = len(a)
    # Expectation (mean) and its standard error
    m, se = np.mean(a), scipy.stats.sem(a)
    # Confidence interval
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    
    # Check if the variable is binary
    unique_values = data[var].dropna().unique()
    is_binary = len(unique_values) == 2

    if is_binary:
        # Print formatted output for binary variable
        print("\n" + "="*50)
        print(f"{'Metric Characterization for':^50}")
        print(f"{var:^50}")
        print("="*50 + "\n")

        print(f"{'Baseline (average):':<35} {m:.2f}")
        print(f"{'Standard Deviation:':<35} {a.std():.2f}")
        print(f"{confidence*100:.0f}% CI Lower: {m - h:.2f}")
        print(f"{confidence*100:.0f}% CI Upper: {m + h:.2f}")

        # Visualization for binary variable
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data[var], kde=False, ax=ax, bins='auto')
        ax.set_title(f'Distribution of {var}', pad=20)
        ax.set_xlabel(var)
        ax.set_ylabel('Frequency')
    else:
        # Quartile coefficient of dispersion
        q_1 = np.percentile(a, 25)
        q_3 = np.percentile(a, 75)
        qcd = (q_3 - q_1) / (q_3 + q_1) if (q_3 + q_1) != 0 else np.nan
        # Skewness and kurtosis
        skewness = scipy.stats.skew(a)
        kurtosis = scipy.stats.kurtosis(a)
        # 10% increase/decrease
        ten_up = m * 1.1
        ten_down = m * 0.9

        # Print formatted output for non-binary variable
        print("\n" + "="*50)
        print(f"{'Metric Characterization for':^50}")
        print(f"{var:^50}")
        print("="*50 + "\n")

        print(f"{'Baseline (average):':<35} {m:.2f}")
        print(f"{'Standard Deviation:':<35} {a.std():.2f}")
        print(f"{'Quartile Coefficient of Dispersion (QCD):':<35} {qcd:.2f}")
        print(f"{'Coefficient of Variation:':<35} {a.std() / m:.2f}")
        print(f"{'Skewness:':<35} {skewness:.2f}")
        print(f"{'Kurtosis:':<35} {kurtosis:.2f}")
        print(f"{confidence*100:.0f}% CI Lower: {m - h:.2f}")
        print(f"{confidence*100:.0f}% CI Upper: {m + h:.2f}")
        print(f"{'10% increase:':<35} {ten_up:.2f}")
        print(f"{'10% decrease:':<35} {ten_down:.2f}")

        # Visualization for non-binary variable
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Time series plot
        ax1.plot(data[var], marker='o', linestyle='-', markersize=3)
        ax1.set_title(f'Time Series of {var}', pad=20)
        ax1.set_xlabel('Index')
        ax1.set_ylabel(var)
        ax1.tick_params(axis='x', rotation=45)
        
        # Distribution plot
        sns.histplot(data[var], kde=True, ax=ax2, bins='auto')
        ax2.set_title(f'Distribution of {var}', pad=20)
        ax2.set_xlabel(var)
        ax2.set_ylabel('Frequency')
    
    plt.suptitle(f'Metric Characterization for {var}', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show();
    



def cap_outliers(df, columns, alpha=1.5):
    """
    Cap outliers in specified columns of a DataFrame.
    
    This function caps the outliers in the specified columns by setting values below 
    the lower bound to the lower bound and values above the upper bound to the upper bound.
    The bounds are determined using the Interquartile Range (IQR) method.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    columns (list): A list of column names in which to cap outliers.
    factor (float): The multiplier for the IQR to determine the bounds for capping outliers.
                    Default is 1.5, which is a common choice for identifying outliers.
                    
    Returns:
    pd.DataFrame: The DataFrame with outliers capped in the specified columns.
    """
    for col in columns:
        # calculate Q1 and Q3
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        # compute IQR
        IQR = Q3 - Q1
        # calculate lower adn upper bounds
        lower_bound = Q1 - alpha * IQR
        upper_bound = Q3 + alpha * IQR
        # cap outliers
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df



################################################################################################################################################################################################
#####################################################                     modeling                           ###################################################################################
################################################################################################################################################################################################



def evaluate_model(model, X, y):
    """
    Evaluates a classification model using cross-validation and calculates the Area Under the Curve (AUC) score.

    Parameters:
    model : estimator object
        The classification model to be evaluated. This must be an object that implements the scikit-learn estimator interface.
    X : array-like of shape (n_samples, n_features)
        The input samples to train the model.
    y : array-like of shape (n_samples,)
        The target values (class labels) as integers or strings.

    Returns:
    mean_auc : float
        The mean AUC score from the cross-validation.
    std_auc : float
        The standard deviation of the AUC scores from the cross-validation.

    Prints:
    Model name and its mean AUC score along with its standard deviation across the cross-validation folds.

    Example:
    >>> from sklearn.linear_model import LogisticRegression
    >>> model = LogisticRegression()
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 1, 1, 0]
    >>> evaluate_model(model, X, y)
    Model: LogisticRegression, AUC: 0.7500 (+/- 0.1000)
    (0.75, 0.1)
    """
    # perform 5-fold cross validation with AUC scoring
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    # print model name and its score
    print(f"Model: {model.__class__.__name__}, AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    # return overall average and standard deviation
    return np.mean(cv_scores), np.std(cv_scores)



