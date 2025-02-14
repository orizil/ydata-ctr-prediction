import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import os
import holidays
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import wandb

from tqdm import tqdm
from scipy import stats
from itertools import product
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def clean_test_data(df):
    # remove entirely empty rows and fully duplicate rows
    df = df.dropna(how="all").drop_duplicates()
    
    # ensure the DateTime column is in datetime format
    df["DateTime"] = pd.to_datetime(df["DateTime"])

    # ensure values that are supposed to be ints are indeed so (and not unnecessarily floats)
    int_columns = ['campaign_id', 'webpage_id', 'product_category_1',
                   'age_level', 'user_depth', 'city_development_index']
    for col in int_columns:
        df[col] = df[col].apply(lambda x: int(x) if pd.notnull(x) else x)

    # ensure the df is sorted by "_order_tracking" when returned
    return df.sort_values("_order_tracking")


def impute_test_data(df, imputation_params):
    """
    Applies imputation pipeline to test data using saved parameters.
    """
    df = df.copy()
   
    # hndle demographics using saved defaults
    demographics = ['gender', 'age_level', 'user_depth', 'user_group_id']
    for demo in demographics:
        df[demo] = df[demo].fillna(imputation_params['demographics_defaults'][demo])
    
    # hndle product_category_2
    df['product_category_2_missing'] = df['product_category_2'].isna().astype(int)
    
    # handle city_development_index
    df['city_development_missing'] = df['city_development_index'].isna().astype(int)
    df['city_development_index'] = df['city_development_index'].fillna(
        imputation_params['city_development_mode']
    )
    
    # handle low missing features if any
    for col in df.columns:
        if df[col].isnull().mean() < 0.01:  # threshold for low missing
            if pd.api.types.is_object_dtype(df[col]) or df[col].nunique() <= 10:
                # categorical variables
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                # numerical variables
                df[col] = df[col].fillna(df[col].median())

    # ensure the df is sorted by "_order_tracking" when returned
    return df.sort_values("_order_tracking")


def create_temporal_features(df):
    """Create temporal features while preserving order"""
    df = df.copy()
    
    # time features
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    
    # binary features
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_early_morning'] = df['hour'].between(2, 5).astype(int)
    
    # holiday features
    us_holidays = holidays.US()
    df['date'] = df['DateTime'].dt.date
    df['is_holiday'] = df['date'].apply(
        lambda x: bool(us_holidays.get(x)) if pd.notna(x) else 0
    ).astype(int)
    
    df['is_near_holiday'] = df.apply(
        lambda row: any(
            abs((holiday - row['date']).days) <= 2
            for holiday in us_holidays.keys()
            if holiday <= row['date']
        ),
        axis=1
    ).astype(int)
    
    # time of day
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[-np.inf, 6, 12, 18, np.inf],
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    
    # hour bins
    df['hour_bin'] = pd.cut(
        df['hour'],
        bins=[0, 4, 8, 12, 16, 20, 24],
        labels=['dawn', 'early_morning', 'morning', 'afternoon', 'evening', 'night'],
        include_lowest=True,
        right=False
    )
    
    df = df.drop('date', axis=1)

    # ensure the df is sorted by "_order_tracking" when returned
    return df.sort_values("_order_tracking")


def create_test_engagement_features(df, engagement_defaults):
    """Create engagement features using default values"""
    df = df.copy()

    # apply default values from our saved parameters
    for feature, default_value in engagement_defaults.items():
        df[feature] = default_value

    # ensure the df is sorted by "_order_tracking" when returned
    return df.sort_values("_order_tracking")


def create_test_campaign_features(df, campaign_defaults):
    """Create campaign features using saved campaign-specific values when available"""
    df = df.copy()

    # use campaign-specific CTR if available, otherwise use global default
    df['campaign_historical_ctr_log'] = df['campaign_id'].astype(str).map(
        campaign_defaults['campaign_ctrs']
    ).fillna(campaign_defaults['campaign_historical_ctr_log'])
    
    # use campaign-hour specific performance if available
    df['campaign_hour_key'] = df.apply(
        lambda x: f"{x['campaign_id']}_{x['hour']}", axis=1
    )
    df['campaign_hour_relative'] = df['campaign_hour_key'].map(
        campaign_defaults['campaign_hour_performance']
    ).fillna(campaign_defaults['campaign_hour_relative'])
    
    # other campaign features
    df['campaign_success_percentile'] = campaign_defaults['campaign_success_percentile']
    df['campaign_webpage_relative'] = campaign_defaults['campaign_webpage_relative']
    
    df = df.drop('campaign_hour_key', axis=1)

    # ensure the df is sorted by "_order_tracking" when returned
    return df.sort_values("_order_tracking")


def create_interaction_features(df):
    """Create interaction features while preserving order"""
    df = df.copy()

    # time-based interactions
    df['campaign_hour_bin'] = df['campaign_id'].astype(str) + '_' + df['hour_bin'].astype(str)
    df['campaign_early_morning'] = df['campaign_id'] * df['is_early_morning']
    
    # user depth interactions
    df['user_depth_time'] = df['user_depth'].astype(str) + '_' + df['time_of_day'].astype(str)
    df['age_weekend'] = df['age_level'] * df['is_weekend']
    df['user_depth_age'] = df['user_depth'] * df['age_level']
    
    # session-time interaction
    df['session_count_bin'] = pd.cut(
        df['session_count_log'],
        bins=5,
        labels=['VL', 'L', 'M', 'H', 'VH']
    )
    df['user_sessions_time'] = df['session_count_bin'].astype(str) + '_' + df['time_of_day'].astype(str)
    
    # demographic interactions
    df['gender_age'] = df['gender'].astype(str) + '_' + df['age_level'].astype(str)
    
    # geographic interactions
    df['city_business'] = df['city_development_index'] * df['is_business_hours']

    # ensure the df is sorted by "_order_tracking" when returned
    return df.sort_values("_order_tracking")


def prepare_test_features_for_modeling(df, categorical_features, features_to_scale, keep_as_is):
    """
    Prepares features specifically for test data modeling, ensuring essential features are preserved.
    Similar to prepare_features_for_modeling but modified for test data needs.
    """
    df = df.copy()

    # convert boolean columns to integers if any exist
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    # handle ordinal features explicitly
    ordinal_features = {
        'age_level': range(7),
        'user_depth': range(1, 4),
        'city_development_index': range(1, 5)
    }
    
    ordinal_feature_names = []
    for col, categories in ordinal_features.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories, ordered=True).codes
            ordinal_feature_names.append(col)

    # one-hot encoding for categorical features
    # remove from the categorical features list features that are also in "keep_as_is"
    categorical_features = [col for col in categorical_features if col not in keep_as_is]
    encoded_feature_names = []
    for col in categorical_features:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
            encoded_feature_names.extend(dummies.columns.tolist())
    
    # update categorical_features to include both ordinal and one-hot encoded features
    categorical_features = ordinal_feature_names + encoded_feature_names

    # remove only specific non-modeling features, preserving essential ones
    features_to_drop = ['session_id', 'user_id', 'DateTime']
    df = df.drop([col for col in features_to_drop if col in df.columns], axis=1)

    # ensure the df is sorted by "_order_tracking" when returned
    return df.sort_values("_order_tracking"), categorical_features


def create_test_features(df, feature_params):
    """Create all features for test data while preserving order"""
    df = df.copy()
    
    # ensure DateTime is in datetime format
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # create features in sequence while maintaining order
    df = create_temporal_features(df)
    df = create_test_engagement_features(df, feature_params['engagement_defaults'])
    df = create_test_campaign_features(df, feature_params['campaign_defaults'])
    df = create_interaction_features(df)
    
    # get transformed dataframe
    df_transformed, _ = prepare_test_features_for_modeling(
        df,
        categorical_features=feature_params['categorical_features'],
        features_to_scale=feature_params['features_to_scale'],
        keep_as_is=feature_params['keep_as_is'] + ['_order_tracking']
    )
    
    # ensure all required dummy variables exist
    for feature in feature_params['selected_features']:
        if feature not in df_transformed.columns:
            # if it's a dummy variable that wasn't created because the category wasn't present
            if ('product_' in feature or 'gender_' in feature or 
                'time_of_day_' in feature or 'hour_bin_' in feature or
                'session_count_bin_' in feature or 'user_depth_time_' in feature or
                'user_sessions_time_' in feature or 'gender_age_' in feature):
                df_transformed[feature] = 0
    
    # verify all required features are present
    missing_features = set(feature_params['selected_features']) - set(df_transformed.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    features_to_return = feature_params['selected_features'] + ['_order_tracking']
    # ensure the df is sorted by "_order_tracking" when returned
    ret = df_transformed[features_to_return].sort_values("_order_tracking")

    return ret


def predict_with_model(X_test, model, target_positive_rate=0.067124):
    """
    Generate predictions using saved model, targeting training CTR
    
    Parameters:
        X_test: Features for prediction
        model: Trained XGBoost model
        target_positive_rate: Target CTR from training (default=6.71%)
    """
    dtest = xgb.DMatrix(X_test)
    probabilities = model.predict(dtest)
    
    # find threshold that gives similar positive rate to training
    threshold = np.percentile(probabilities, 100 - (target_positive_rate * 100))
    
    print(f"\nProbability distribution in test:")
    print(f"Min: {probabilities.min():.4f}")
    print(f"Max: {probabilities.max():.4f}")
    print(f"Mean: {probabilities.mean():.4f}")
    print(f"Std: {probabilities.std():.4f}")
    print(f"\nUsing threshold {threshold:.4f} to target {target_positive_rate:.2%} positive rate")
    
    binary_predictions = (probabilities > threshold).astype(int)
    actual_positive_rate = binary_predictions.mean()
    print(f"Achieved positive rate: {actual_positive_rate:.2%}")
    
    # Log metrics to wandb
    wandb.log({
        "prediction_min": probabilities.min(),
        "prediction_max": probabilities.max(),
        "prediction_mean": probabilities.mean(),
        "prediction_std": probabilities.std(),
        "threshold": threshold,
        "target_positive_rate": target_positive_rate,
        "achieved_positive_rate": actual_positive_rate,
    })
    
    # Create and log probability distribution histogram
    fig, ax = plt.subplots()
    ax.hist(probabilities, bins=50)
    ax.set_title("Prediction Probability Distribution")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Count")
    wandb.log({"probability_distribution": wandb.Image(fig)})
    plt.close()
    
    return binary_predictions

script_dir = os.path.dirname(os.path.realpath(__file__))
imputation_params_path = os.path.join(script_dir, 'imputation_params.json')
feature_params_path = os.path.join(script_dir, 'feature_params.json')
scaler_path = os.path.join(script_dir, 'feature_scaler.joblib')
model_path = os.path.join(script_dir, 'final_model.joblib')
test_file_path = os.path.join(script_dir, 'X_test_1st.csv')


def process_test_data(test_filepath=test_file_path, df=None, imputation_params_path=imputation_params_path, feature_params_path=feature_params_path, 
                     scaler_path=scaler_path, model_path=model_path):
    """Process test data while maintaining original order"""
    
    # Initialize wandb
    wandb.init(
        project="ctr-prediction-kaggle",
        config={
            "model_path": model_path,
            "feature_params_path": feature_params_path,
            "imputation_params_path": imputation_params_path,
            "scaler_path": scaler_path
        }
    )

    # load and preserve order
    if df is not None:
        test_df = df
    else:
        test_df = pd.read_csv(test_filepath)
    test_df['_order_tracking'] = np.arange(len(test_df))

    # Log dataset info
    wandb.log({
        "test_dataset_size": len(test_df),
        "num_features": len(test_df.columns)
    })

    # load parameters
    with open(imputation_params_path, 'r') as f:
        imputation_params = json.load(f)
    with open(feature_params_path, 'r') as f:
        feature_params = json.load(f)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # add the "_order_tracking" feature to the feature_params "keep_as_is"
    feature_params['keep_as_is'].append('_order_tracking')

    # basic cleaning
    test_df = clean_test_data(test_df)

    # process features while maintaining order
    test_df = impute_test_data(test_df, imputation_params)
    test_df = create_test_features(test_df, feature_params)

    # scale features (which does not affect order of rows as it's column-wise)
    X_test = test_df[feature_params['selected_features']]
    X_test[feature_params['features_to_scale']] = scaler.transform(
        X_test[feature_params['features_to_scale']]
    )

    # predict :-)
    predictions = predict_with_model(X_test, model)
    
    # Log final prediction stats
    wandb.log({
        "final_positive_predictions": predictions.mean(),
        "num_positive_predictions": predictions.sum(),
        "num_negative_predictions": len(predictions) - predictions.sum()
    })
    
    # Close wandb run
    wandb.finish()

    return predictions
