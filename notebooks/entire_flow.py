import json
import warnings
from datetime import datetime, timedelta

import holidays
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

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

print("here")
df = pd.read_csv("train_dataset_full.csv")
print("here after")

# clean the data
def clean_data(df):
    # remove entirely empty rows and fully duplicate rows
    df = df.dropna(how="all").drop_duplicates()
    
    # ensure the DateTime column is in datetime format
    df["DateTime"] = pd.to_datetime(df["DateTime"])

    # ensure values that are supposed to be ints are indeed so (and not unnecessarily floats)
    int_columns = ['campaign_id', 'webpage_id', 'product_category_1',
                   'age_level', 'user_depth', 'city_development_index']
    for col in int_columns:
        df[col] = df[col].apply(lambda x: int(x) if pd.notnull(x) else x)
    
    return df


df = clean_data(df)


# ## Data imputation
# 
# In the EDA, the missing values that stood out were in:
# 1. "product_category_2" (79.16% missing): we will creates a missing indicator feature (which might get dropped in future steps if deemed irrelevant) but won't impute values, as that could introduce misleading data, as there's simply too much missing data. 
# 2. "city_development_index" (27.60%): we will creates a missing indicator feature (which might get dropped in future steps if deemed irrelevant) and impute with the mode.
# 3. "gender", "age_level", "user_depth", "user_group_id": we will first attempt to fill from other sessions of the same user, but we'll fallback to global defaults as needed.
# 4. Else: we will automatically impute missing values for features with missing rates below 1%, using the mode for categorical variables (including those with low cardinality), and medians for numerical variables.


def impute_demographics_by_user(df):
    """
    Imputes demographic features (gender, age_level, user_depth) using other 
    sessions from the same user when available, then falls back to global defaults.
    
    The EDA showed these features have about 4.7% missing values and are generally
    consistent within users, making this a reliable approach.
    """
    df = df.copy()
    
    # first, attempt to fill missing demographics using other sessions from same user
    demographics = ['gender', 'age_level', 'user_depth', 'user_group_id']
    
    for demo in demographics:
        # group by user_id and apply forward fill and backward fill
        df[demo] = df.groupby('user_id')[demo].transform(
            lambda x: x.ffill().bfill()
        )
    
    # if there are any remaining missing values, fill with mode values:
    defaults = {"gender": df["gender"].mode().iloc[0],
                "age_level": df["age_level"].mode().iloc[0],
                "user_depth": df["user_depth"].mode().iloc[0],
                "user_group_id": df["user_group_id"].mode().iloc[0]}

    
    for column, default in defaults.items():
        df[column] = df[column].fillna(default)
    
    return df


def handle_product_category2(df):
    """
    Handles product_category_2 which has 79.16% missing values.
    Creates a binary indicator for missingness and leaves original values as is.
    """
    df = df.copy()
    
    # create missing indicator
    df['product_category_2_missing'] = df['product_category_2'].isna().astype(int)
    
    # return without imputing due to extremely high missingness
    return df


def handle_city_development(df):
    """
    Handles city_development_index which has 27.60% missing values.
    Creates a missing indicator and imputes with mode.
    """
    df = df.copy()
    
    # create missing indicator
    df['city_development_missing'] = df['city_development_index'].isna().astype(int)
    
    # impute with mode
    city_development_mode = df['city_development_index'].mode()[0]
    df['city_development_index'] = df['city_development_index'].fillna(city_development_mode)
    
    return df


def handle_low_missing(df, threshold: float=0.01):
    """
    Handles features with low missing rates (< threshold).
    Uses mode for categorical variables and median for numerical ones.
    """
    df = df.copy()
    
    # identify columns with low missing rates
    missing_rates = df.isnull().mean()
    low_missing_cols = missing_rates[missing_rates > 0][missing_rates < threshold].index
    
    for col in low_missing_cols:
        # determine if column should be treated as categorical
        unique_vals = df[col].nunique()
        is_categorical = pd.api.types.is_object_dtype(df[col]) or unique_vals <= 10
        
        if is_categorical:
            # for categorical variables, use mode
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            # for numerical variables, use median
            df[col] = df[col].fillna(df[col].median())
    
    return df


def impute_dataset(df):
    """
    Applies the complete imputation pipeline to the dataset.
    Returns a new dataframe with all imputation strategies applied.
    """
    df = df.copy()
    
    # apply each imputation step in sequence
    df = impute_demographics_by_user(df)
    df = handle_product_category2(df)
    df = handle_city_development(df)
    df = handle_low_missing(df)
    
    return df



df_imputed = impute_dataset(df)


# ### validation of the imputation: looking at column distributions changes


def validate_imputation(df_original, df_imputed):
    print("validate_imputation")
    """Validates that imputation maintained reasonable distributions"""
    for col in df_original.columns:
        if col == 'product_category_2':  # skip intentionally non-imputed column
            continue
            
        if pd.api.types.is_numeric_dtype(df_original[col]):
            # check that means and stds are similar for numeric columns
            orig_mean = df_original[col].mean()
            imp_mean = df_imputed[col].mean()
            orig_std = df_original[col].std()
            imp_std = df_imputed[col].std()
            
            print(f"\n{col}:")
            print(f"Mean - Original: {orig_mean:.2f}, Imputed: {imp_mean:.2f}")
            print(f"Std  - Original: {orig_std:.2f}, Imputed: {imp_std:.2f}")
        
        elif pd.api.types.is_object_dtype(df_original[col]):
            # check value distributions for categorical columns
            orig_dist = df_original[col].value_counts(normalize=True)
            imp_dist = df_imputed[col].value_counts(normalize=True)
            
            print(f"\n{col} distribution:")
            print("Original vs Imputed:")
            print(pd.concat([orig_dist, imp_dist], axis=1, 
                          keys=['Original', 'Imputed']).head())
            
            
            

imputation_params = {
    'demographics_defaults': {
        'gender': 'Male',
        'age_level': 3.0,
        'user_depth': 3.0,
        'user_group_id': 3.0
    },
    'city_development_mode': 2.0
}

# save for later use
with open('imputation_params.json', 'w') as f:
    json.dump(imputation_params, f)


def print_imputation_summary(df_original, df_imputed):
    """Prints summary of imputation changes"""
    total_missing_before = df_original.isnull().sum().sum()
    total_missing_after = df_imputed.isnull().sum().sum()
    
    print("\nImputation Summary:")
    print(f"Total missing values before: {total_missing_before:,}")
    print(f"Total missing values after: {total_missing_after:,}")
    print(f"Total values imputed: {total_missing_before - total_missing_after:,}")

validate_imputation(df, df_imputed)


# Looking at the above validation, we can see that:
# 1. Our target feature "is_click" maintained exactly same means and std, meaning its integrity is preserved.
# 2. All features maintain very similar proportions apart from the below; "gender" ratio slightly shifted but reasonably (Male: 88.3% → 88.9%) but these small shifts are acceptable given the missingness rates.
# 3. "city_development_index" shows the largest shift, but it's expected given its high missingness rate (27.6%) and our mode imputation strategy.


# ## Feature Engineering: creating new features

def create_temporal_features(df):
    """
    Creates temporal features from DateTime column without leakage.
    """
    df = df.copy()
    
    # basic time features (these are OK as they don't use future data)
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    
    # binary features
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_early_morning'] = df['hour'].between(2, 5).astype(int)
    
    # time of day as categorical
    df['time_of_day'] = pd.cut(
        df['hour'],
        bins=[-np.inf, 6, 12, 18, np.inf],
        labels=['night', 'morning', 'afternoon', 'evening']
    )
    
    # add hour bins
    df['hour_bin'] = pd.cut(
        df['hour'],
        bins=[0, 4, 8, 12, 16, 20, 24],
        labels=['dawn', 'early_morning', 'morning', 'afternoon', 'evening', 'night'],
        include_lowest=True,
        right=False
    )
    
    # holiday features
    us_holidays = holidays.US()
    df['date'] = df['DateTime'].dt.date
    df['is_holiday'] = df['date'].apply(
        lambda x: bool(us_holidays.get(x)) if pd.notna(x) else 0
    ).astype(int)
    
    # near holiday feature (only using past holidays to prevent leakage)
    df['is_near_holiday'] = df.apply(
        lambda row: any(
            abs((holiday - row['date']).days) <= 2
            for holiday in us_holidays.keys()
            if holiday <= row['date']  # only consider past and current holidays
        ),
        axis=1
    ).astype(int)
    
    df = df.drop('date', axis=1)
    return df


import numpy as np
import pandas as pd

def create_user_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates user engagement features without leakage by using expanding windows.
    """
    df = df.copy()
    
    # Ensure the DataFrame is sorted by user and time
    df.sort_values(['user_id', 'DateTime'], inplace=True)
    
    # -------------------------------
    # Historical CTR calculation
    # -------------------------------
    # Use expanding window over past sessions (excluding current row) for cumulative clicks
    df['cumulative_clicks'] = df.groupby('user_id')['is_click'].transform(
        lambda x: x.shift().expanding().sum()).fillna(0)
    df['cumulative_impressions'] = df.groupby('user_id').cumcount()
    # Laplace smoothing to avoid division by zero
    df['historical_user_ctr'] = (df['cumulative_clicks'] + 1) / (df['cumulative_impressions'] + 2)
    
    # -------------------------------
    # Session counts and log-transform
    # -------------------------------
    df['session_count_user'] = df.groupby('user_id').cumcount() + 1
    df['session_count_log'] = np.log1p(df['session_count_user'])
    
    # -------------------------------
    # Sessions per day (vectorized)
    # -------------------------------
    # Instead of a slow Python loop, compute cumulative unique days per user.
    df['date'] = df['DateTime'].dt.date
    df['sessions_per_day_mean'] = df.groupby('user_id')['date'].transform(
        lambda x: np.arange(1, len(x) + 1) / x.ne(x.shift()).cumsum()
    )
    df['sessions_per_day_mean_log'] = np.log1p(df['sessions_per_day_mean'])
    
    # -------------------------------
    # Time since last click (in hours)
    # -------------------------------
    df['time_since_last_click'] = (
        df.groupby('user_id')['DateTime']
          .diff()
          .dt.total_seconds() / 3600.0
    ).fillna(168)  # Default to 1 week (168 hours) for the first session
    
    # -------------------------------
    # Click frequency in past 24 hours (vectorized per group)
    # -------------------------------
    def compute_click_frequency(group: pd.DataFrame) -> pd.Series:
        # Convert timestamps and clicks to numpy arrays
        times = group['DateTime'].values.astype('datetime64[ns]')
        clicks = group['is_click'].values.astype(int)
        # Build cumulative sum with a zero prepended so that for each row i, 
        # cumsum[i] equals sum(clicks[:i]) i.e. only past clicks.
        cumsum = np.concatenate(([0], np.cumsum(clicks)))
        # For each row, find the index of the first timestamp within the last 24 hours.
        left_idx = np.searchsorted(times, times - np.timedelta64(24, 'h'), side='left')
        # Count of clicks in the window = cumulative clicks up to i (excl. current) minus cumulative clicks at left_idx.
        freq = cumsum[np.arange(len(clicks))] - cumsum[left_idx]
        return pd.Series(freq, index=group.index)
    
    df['click_frequency_24h'] = df.groupby('user_id', group_keys=False).apply(compute_click_frequency)
    df['click_frequency_24h_log'] = np.log1p(df['click_frequency_24h'])
    
    # -------------------------------
    # Expanding percentile ranks for engagement features
    # -------------------------------
    engagement_features = [
        'historical_user_ctr',
        'session_count_log',
        'sessions_per_day_mean_log',
        'click_frequency_24h_log'
    ]
    
    # For each feature, rank using only past data within each user group.
    # The first session (with no history) gets a rank of 0.
    for feature in engagement_features:
        df[f'{feature}_rank'] = df.groupby('user_id')[feature].transform(
            lambda x: x.shift().expanding().rank(pct=True)
        ).fillna(0)
    
    # -------------------------------
    # Weighted combination of ranks for engagement score
    # -------------------------------
    df['user_engagement_score'] = (
        0.4 * df['historical_user_ctr_rank'] +
        0.25 * df['session_count_log_rank'] +
        0.25 * df['sessions_per_day_mean_log_rank'] +
        0.1 * df['click_frequency_24h_log_rank']
    )
    
    # Drop intermediate columns that are no longer needed
    df.drop(['date', 'cumulative_clicks', 'cumulative_impressions'], axis=1, inplace=True)
    
    return df

    """
    Creates user engagement features without leakage by using expanding windows.
    """
    df = df.copy()
    
    # sort by user and time
    df = df.sort_values(['user_id', 'DateTime'])
    
    # calculate historical CTR using expanding window
    df['cumulative_clicks'] = df.groupby('user_id')['is_click'].transform(
        lambda x: x.shift().expanding().sum()).fillna(0)
    df['cumulative_impressions'] = df.groupby('user_id').cumcount()
    df['historical_user_ctr'] = (df['cumulative_clicks'] + 1) / (df['cumulative_impressions'] + 2)  # laplace smoothing
    
    # session counts (using only past data)
    df['session_count_user'] = df.groupby('user_id').cumcount() + 1
    df['session_count_log'] = np.log1p(df['session_count_user'])
    
    # sessions per day using a custom cumulative computation
    df['date'] = df['DateTime'].dt.date
    
    def compute_sessions_per_day_mean(dates):
        """For each row in the dates series, compute cumulative sessions per unique day."""
        seen = set()
        ratios = []
        for i, d in enumerate(dates):
            seen.add(d)
            # (i+1) is the cumulative count; len(seen) is the number of unique days so far
            ratios.append((i + 1) / len(seen))
        return pd.Series(ratios, index=dates.index)
    
    df['sessions_per_day_mean'] = df.groupby('user_id')['date'].transform(compute_sessions_per_day_mean)
    df['sessions_per_day_mean_log'] = np.log1p(df['sessions_per_day_mean'])
    
    # time since last click (only using past data)
    df['time_since_last_click'] = df.groupby('user_id')['DateTime'].diff().dt.total_seconds() / 3600
    df['time_since_last_click'] = df['time_since_last_click'].fillna(168)  # 1 week for first session
    
    # click frequency in past 24h
    df['prev_24h'] = df['DateTime'] - pd.Timedelta(hours=24)
    df['click_frequency_24h'] = df.groupby('user_id').apply(
        lambda group: group.apply(
            lambda row: group[
                (group['DateTime'] < row['DateTime']) &  # only use past data
                (group['DateTime'] >= row['prev_24h']) & 
                (group['is_click'] == 1)
            ].shape[0],
            axis=1
        )
    ).reset_index(level=0, drop=True)
    
    df['click_frequency_24h_log'] = np.log1p(df['click_frequency_24h'])
    
    # user engagement score using percentile ranks of past data only
    engagement_features = [
        'historical_user_ctr',
        'session_count_log',
        'sessions_per_day_mean_log',
        'click_frequency_24h_log'
    ]
    
    # calculate expanding ranks for each feature
    # which means : for each user, calculate the percentile rank of each feature at each time point using only past data of that user
    # example: if a user has 10 sessions, the first session will have a rank of 0, the second session will have a rank of 0.1, and so on up to 1
    for feature in engagement_features:
        df[f'{feature}_rank'] = df.groupby('user_id')[feature].transform(
            lambda x: x.shift().expanding().rank(pct=True)).fillna(0)  # first value gets rank 0
    
    # weighted combination of ranks
    df['user_engagement_score'] = (
        0.4 * df['historical_user_ctr_rank'] +
        0.25 * df['session_count_log_rank'] +
        0.25 * df['sessions_per_day_mean_log_rank'] +
        0.1 * df['click_frequency_24h_log_rank']
    )
    
    df = df.drop(['prev_24h', 'date', 'cumulative_clicks', 'cumulative_impressions'], axis=1)
    return df


def create_campaign_performance_features(df):
    """
    Creates campaign performance features using only past data.
    """
    df = df.copy()
    
    # sort by time to ensure we only use past data
    df = df.sort_values('DateTime')
    
    # calculate expanding window statistics for campaigns
    df['campaign_impressions'] = df.groupby('campaign_id').cumcount() + 1
    df['campaign_clicks'] = df.groupby('campaign_id')['is_click'].expanding().sum().reset_index(0, drop=True)
    df['campaign_historical_ctr'] = (df['campaign_clicks'] + 1) / (df['campaign_impressions'] + 2)  # laplace smoothing
    df['campaign_historical_ctr_log'] = np.log1p(df['campaign_historical_ctr'] * 100)
    
    # campaign-webpage combination with relative performance
    df['campaign_webpage_clicks'] = (
    df.groupby(['campaign_id', 'webpage_id'])['is_click']
        .expanding()
        .sum()
        .reset_index(level=[0, 1], drop=True)  # reset both groupby indices
        )
    df['campaign_webpage_impressions'] = df.groupby(['campaign_id', 'webpage_id']).cumcount() + 1
    df['campaign_webpage_ctr'] = (df['campaign_webpage_clicks'] + 1) / (df['campaign_webpage_impressions'] + 2)
    df['campaign_webpage_relative'] = df['campaign_webpage_ctr'] / df['campaign_historical_ctr']
    
    # hourly performance
    df['campaign_hour_clicks'] = (
        df.groupby(['campaign_id', 'hour'])['is_click']
        .cumsum()
        .shift(fill_value=0))
    df['campaign_hour_impressions'] = df.groupby(['campaign_id', 'hour']).cumcount() + 1
    df['campaign_hour_ctr'] = (df['campaign_hour_clicks'] + 1) / (df['campaign_hour_impressions'] + 2)
    df['campaign_hour_relative'] = df['campaign_hour_ctr'] / df['campaign_historical_ctr']
    
    # campaign success as percentile rank (using expanding window)
    df['campaign_success_percentile'] = df.groupby('DateTime')['campaign_historical_ctr'].rank(pct=True)
    
    # clean up intermediate columns
    df = df.drop(['campaign_clicks', 'campaign_webpage_clicks', 'campaign_webpage_impressions',
                  'campaign_hour_clicks', 'campaign_hour_impressions', 'campaign_hour_ctr',
                  'campaign_webpage_ctr'], axis=1)
    
    return df


def create_interaction_features(df):
    """
    Creates interaction features without leakage.
    """
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
        bins=5,  # Fixed 5 bins
        labels=['VL', 'L', 'M', 'H', 'VH'])
    df['user_sessions_time'] = df['session_count_bin'].astype(str) + '_' + df['time_of_day'].astype(str)
    
    # demographic interactions
    df['gender_age'] = df['gender'].astype(str) + '_' + df['age_level'].astype(str)
    
    # geographic interactions
    df['city_business'] = df['city_development_index'] * df['is_business_hours']
    
    return df


def prepare_dataset_for_inference(df, imputation_params, scaler, selected_features):
    """
    Prepares a dataset for inference using saved parameters.
    """
    df = df.copy()
    
    # first, impute missing values using saved parameters
    df['gender'] = df['gender'].fillna(imputation_params['demographics_defaults']['gender'])
    df['age_level'] = df['age_level'].fillna(imputation_params['demographics_defaults']['age_level'])
    df['user_depth'] = df['user_depth'].fillna(imputation_params['demographics_defaults']['user_depth'])
    df['user_group_id'] = df['user_group_id'].fillna(imputation_params['demographics_defaults']['user_group_id'])
    df['city_development_index'] = df['city_development_index'].fillna(imputation_params['city_development_mode'])
    
    # create features
    df = create_temporal_features(df)
    df = create_user_engagement_features(df)
    df = create_campaign_performance_features(df)
    df = create_interaction_features(df)
    
    # scale features using saved scaler
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_features] = scaler.transform(df[numeric_features])
    
    # select only the required features in the correct order
    return df[selected_features]


def create_all_features(df):
    print("create_all_features")
    """
    Applies all feature engineering steps and returns both engineered features
    and a list of features requiring scaling.
    """
    df = df.copy()
    
    # apply each feature engineering step
    print("create_temporal_features")
    df = create_temporal_features(df)
    print("create_user_engagement_features")
    df = create_user_engagement_features(df)
    print("create_campaign_performance_features")
    df = create_campaign_performance_features(df)
    print("create_interaction_features")
    df = create_interaction_features(df)
    
    # identify features that need scaling
    features_to_scale = [
        'historical_user_ctr',
        'session_count_log',
        'sessions_per_day_mean',
        'sessions_per_day_std',
        'session_count_user',
        'time_since_last_click',
        'click_frequency_24h',
        'click_frequency_24h_log',
        'user_engagement_score',
        'campaign_historical_ctr_log',
        'campaign_webpage_relative',
        'campaign_hour_relative',
        'campaign_success_percentile',
        'user_depth_age',
        'age_weekend',
        'city_business'
        ]
    
    return df, features_to_scale


df_engineered, features_to_scale = create_all_features(df_imputed)


# ### Validate the completness of the new features, etc.:


def validate_engineered_features(df):
    """
    Performs sanity checks on engineered features.
    """
    results = {}
    
    # check for nulls
    null_counts = df.isnull().sum()
    results['features_with_nulls'] = null_counts[null_counts > 0].to_dict()
    
    # check for infinite values
    inf_counts = df.isin([np.inf, -np.inf]).sum()
    results['features_with_inf'] = inf_counts[inf_counts > 0].to_dict()
    
    # validate binary features are actually binary
    binary_features = [col for col in df.columns if col.startswith('is_')]
    non_binary = {col: sorted(df[col].unique()) 
                 for col in binary_features 
                 if not df[col].isin([0, 1, np.nan]).all()}
    results['non_binary_features'] = non_binary
    
    # check for low variance features
    variances = df.select_dtypes(include=np.number).var()
    low_variance = variances[variances < 0.01].to_dict()
    results['low_variance_features'] = low_variance
    
    # validate time-based features
    if 'hour' in df.columns:
        results['hour_range_valid'] = df['hour'].between(0, 23).all()
    if 'day_of_week' in df.columns:
        results['day_range_valid'] = df['day_of_week'].between(0, 6).all()
        
    # check for correct CTR ranges (between 0 and 1)
    ctr_features = [col for col in df.columns if 'ctr' in col.lower()]
    invalid_ctr = {col: (df[col].min(), df[col].max()) 
                  for col in ctr_features 
                  if not df[col].between(0, 1, inclusive='both').all()}
    results['invalid_ctr_ranges'] = invalid_ctr
    
    # cardinality check for categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    high_cardinality = {col: df[col].nunique() 
                       for col in categorical_cols 
                       if df[col].nunique() > 100}
    results['high_cardinality_features'] = high_cardinality
    
    return results


def print_validation_results(results: dict):
    """
    Prints validation results in a readable format.
    """
    print("Feature Validation Results:")
    print("-" * 50)
    
    if results['features_with_nulls']:
        print("\nFeatures with null values:")
        for feat, count in results['features_with_nulls'].items():
            print(f"  {feat}: {count} nulls")
    
    if results['features_with_inf']:
        print("\nFeatures with infinite values:")
        for feat, count in results['features_with_inf'].items():
            print(f"  {feat}: {count} infinities")
    
    if results['non_binary_features']:
        print("\nNon-binary 'is_' features:")
        for feat, values in results['non_binary_features'].items():
            print(f"  {feat}: {values}")
    
    if results['low_variance_features']:
        print("\nFeatures with very low variance:")
        for feat, var in results['low_variance_features'].items():
            print(f"  {feat}: {var:.6f}")
    
    if results['invalid_ctr_ranges']:
        print("\nCTR features with invalid ranges:")
        for feat, (min_val, max_val) in results['invalid_ctr_ranges'].items():
            print(f"  {feat}: range [{min_val:.3f}, {max_val:.3f}]")
    
    if results['high_cardinality_features']:
        print("\nHigh cardinality categorical features:")
        for feat, count in results['high_cardinality_features'].items():
            print(f"  {feat}: {count} unique values")


print("validate_engineered_features")
validation_results = validate_engineered_features(df_engineered)
print_validation_results(validation_results)


def prepare_features_for_modeling(df, categorical_features, features_to_scale, keep_as_is):
    """
    Prepares features for modeling by handling categorical and numerical variables.
    """
    df = df.copy()
    
    # verify all features are accounted for
    all_features = set(categorical_features + features_to_scale + keep_as_is)
    missing_features = set(df.columns) - all_features
    extra_features = all_features - set(df.columns)
    
    if missing_features:
        print(f"Warning: These columns are not categorized: {missing_features}")
    if extra_features:
        print(f"Warning: These categorized features are not in dataframe: {extra_features}")

    # convert boolean columns to integers
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    # ordinal encoding for ordinal features
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
    encoded_feature_names = []
    for col in categorical_features:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
            encoded_feature_names.extend(dummies.columns.tolist())

    # update categorical_features to include both ordinal and one-hot encoded features
    categorical_features = ordinal_feature_names + encoded_feature_names

    # remove features not needed for modeling
    # true id features (that are not basically categories); datetime; "product_category_2" (due to high missingness)
    df = df.drop(['session_id', 'user_id', 'DateTime', 'product_category_2'], axis=1)            
    
    return df, categorical_features


categorical_features = [
    # base categorical features
    'gender', 
    'product',
    
    # temporal categoricals
    'time_of_day',    # night/morning/afternoon/evening
    'hour_bin',       # dawn/early_morning/morning/afternoon/evening/night
    
    # binned features
    'session_count_bin',  # VL/L/M/H/VH
    
    # interaction categoricals
    'campaign_hour_bin',  # campaign_id + hour_bin
    'user_depth_time',    # user_depth + time_of_day
    'user_sessions_time', # session_count_bin + time_of_day
    'gender_age'         # gender + age_level
]

features_to_scale = [
    # user engagement metrics
    'historical_user_ctr',
    'session_count_log',
    'sessions_per_day_mean',
    'sessions_per_day_mean_log',
    'session_count_user',
    'time_since_last_click',
    'click_frequency_24h',
    'click_frequency_24h_log',
    'user_engagement_score',
    
    # ranking features
    'historical_user_ctr_rank',
    'session_count_log_rank',
    'sessions_per_day_mean_log_rank',
    'click_frequency_24h_log_rank',
    
    # campaign performance metrics
    'campaign_historical_ctr',
    'campaign_historical_ctr_log',
    'campaign_webpage_relative',
    'campaign_hour_relative',
    'campaign_success_percentile', 
        
    # interaction numerics
    'user_depth_age',
    'age_weekend',
    'city_business',
    'campaign_early_morning'
]

keep_as_is = [
    'campaign_id',
    'webpage_id',
    'is_click',  # target, will be later separated
    
    # binary features
    'is_business_hours',
    'is_weekend',
    'is_early_morning',
    'is_holiday',
    'is_near_holiday',
    'product_category_2_missing',
    'city_development_missing',
    'var_1',
    
    # time features in fixed ranges
    'hour',         # 0-23
    'day_of_week',  # 0-6
    
    # ordinal features
    'age_level',           # 0-6
    'user_depth',          # 1-3
    'city_development_index', # 1-4
    
    # other features
    'product_category_1',  # 1-5
    'user_group_id',
    'campaign_impressions'
]


# prepare the features for modeling by encoding categorical variables
print("prepare_features_for_modeling")
df_prepared, categorical_features = prepare_features_for_modeling(
    df_engineered, 
    categorical_features, 
    features_to_scale, 
    keep_as_is
)


def prefilter_features(df: pd.DataFrame, target_col: str = 'is_click', verbose: bool = True):
    """
    Multi-stage feature pre-filtering process.
    Returns a list of features that pass all filtering stages.
    """
    df = df.copy()
    initial_features = len(df.columns) - 1  # exclude target
    if verbose:
        print(f"Starting with {initial_features} features")
    
    # remove constant and quasi-constant features
    def remove_low_variance_features(df, threshold=0.01):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variances = df[numeric_cols].var()
        low_var_features = variances[variances < threshold].index.tolist()
        
        if verbose:
            print(f"\nLow variance features removed ({len(low_var_features)}):")
            for f in low_var_features:
                print(f"- {f} (variance: {variances[f]:.6f})")
        
        return df.drop(columns=low_var_features)

    # remove highly correlated features
    def remove_correlated_features(df, threshold=0.95):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        high_corr_features = []
        seen_pairs = set()
        
        # find features to remove
        for col in upper.columns:
            # get correlations above threshold
            high_corr = upper[col][upper[col] > threshold].index.tolist()
            for feat in high_corr:
                if (col, feat) not in seen_pairs and (feat, col) not in seen_pairs:
                    # keep the one with higher correlation with target
                    corr_with_target = abs(df[[col, feat, target_col]].corr()[target_col])
                    if corr_with_target[col] < corr_with_target[feat]:
                        high_corr_features.append(col)
                    else:
                        high_corr_features.append(feat)
                    seen_pairs.add((col, feat))
        
        high_corr_features = list(set(high_corr_features))
        
        if verbose:
            print(f"\nHighly correlated features removed ({len(high_corr_features)}):")
            for f in high_corr_features:
                print(f"- {f}")
        
        return df.drop(columns=high_corr_features)

    # remove features with low mutual information
    def remove_low_mi_features(df, target_col, threshold=0.001):
        # separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove(target_col)
        
        # calculate MI scores for numeric features
        mi_scores = mutual_info_classif(
            StandardScaler().fit_transform(df[numeric_cols]), 
            df[target_col],
            random_state=42
        )
        mi_series = pd.Series(mi_scores, index=numeric_cols)
        
        # identify features with low MI scores
        low_mi_features = mi_series[mi_series < threshold].index.tolist()
        
        if verbose:
            print(f"\nLow mutual information features removed ({len(low_mi_features)}):")
            for f in low_mi_features:
                print(f"- {f} (MI score: {mi_series[f]:.6f})")
        
        return df.drop(columns=low_mi_features)


    # remove features with high p-value in univariate testing
    def remove_insignificant_features(df, target_col, p_threshold=0.05):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove(target_col)
        
        insignificant_features = []
        for col in numeric_cols:
            _, p_value = stats.mannwhitneyu(
                df[df[target_col] == 1][col],
                df[df[target_col] == 0][col],
                alternative='two-sided'
            )
            if p_value > p_threshold:
                insignificant_features.append(col)
        
        if verbose:
            print(f"\nStatistically insignificant features removed ({len(insignificant_features)}):")
            for f in insignificant_features:
                print(f"- {f}")
        
        return df.drop(columns=insignificant_features)


    # apply all filtering stages
    df = remove_low_variance_features(df)
    df = remove_correlated_features(df)
    df = remove_low_mi_features(df, target_col)
    df = remove_insignificant_features(df, target_col)
    
    selected_features = [col for col in df.columns if col != target_col]
    
    if verbose:
        print(f"\nFinal feature set: {len(selected_features)} features")
        print("\nRemaining features:")
        for f in selected_features:
            print(f"- {f}")
    
    return selected_features


# run through the pre-filtering process to see its suggestions
print("prefilter_features")
selected_features = prefilter_features(df_prepared)


# remove all of the non-selected features from the dataframe
X = df_prepared[selected_features]
y = df_prepared['is_click']

if 'is_click' in X.columns:
    X = X.drop('is_click', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# remove all of the non-selected features from the lists: "categorical_features", "features_to_scale", and "keep_as_is"
categorical_features = [col for col in categorical_features if col in selected_features]
features_to_scale = [col for col in features_to_scale if col in selected_features]
keep_as_is = [col for col in keep_as_is if col in selected_features]


# scale the features that need it, by training the scaler on the training set
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])


# ## Feature Selection


def evaluate_feature_importance_optimized(X, y, n_bootstraps=100, sample_frac=0.8):
    """
    Evaluate feature importance using bootstrapped samples with debug prints.
    """
    
    # initialize score arrays
    mi_scores = np.zeros((n_bootstraps, X.shape[1]), dtype=np.float32)
    rf_importance = np.zeros((n_bootstraps, X.shape[1]), dtype=np.float32)
    
    # combine features and target for sampling
    data = pd.concat([X, y.rename('target')], axis=1)
    
    # initialize random forest
    rf = RandomForestClassifier(
        n_estimators=30,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    
    for i in range(n_bootstraps):
        if i % 10 == 0:
            print(f"\nBootstrap iteration {i}/{n_bootstraps}")
        
        try:
            bootstrap_sample = data.sample(frac=sample_frac, replace=True, random_state=i)
            
            if i == 0:  # detailed diagnostics for first iteration
                print(f"Bootstrap {i} details:")
                print(f"Sample shape: {bootstrap_sample.shape}")
                print(f"Sample NaN in features: {bootstrap_sample.drop('target', axis=1).isna().sum().sum()}")
                print(f"Sample NaN in target: {bootstrap_sample['target'].isna().sum()}")
            
            X_boot = bootstrap_sample.drop('target', axis=1)
            y_boot = bootstrap_sample['target']
            
            # check for NaN values before MI calculation
            if X_boot.isna().any().any() or y_boot.isna().any():
                print(f"Bootstrap {i} has NaN values:")
                print(f"X_boot NaN count: {X_boot.isna().sum().sum()}")
                print(f"y_boot NaN count: {y_boot.isna().sum()}")
                raise ValueError("NaN values detected in bootstrap sample")
            
            # compute mutual information scores
            mi_scores[i] = mutual_info_classif(X_boot, y_boot, random_state=42)
            
            # fit random forest and get feature importances
            rf.fit(X_boot, y_boot)
            rf_importance[i] = rf.feature_importances_
            
        except Exception as e:
            print(f"Error in bootstrap {i}: {str(e)}")
            continue

    results = []
    for j, feature in enumerate(X.columns):
        valid_mi = mi_scores[:, j][~np.isnan(mi_scores[:, j])]
        valid_rf = rf_importance[:, j][~np.isnan(rf_importance[:, j])]
        
        if len(valid_mi) > 0 and len(valid_rf) > 0:
            mi_mean = valid_mi.mean()
            mi_std = valid_mi.std()
            mi_ci = stats.norm.interval(0.95, mi_mean, mi_std)
            
            rf_mean = valid_rf.mean()
            rf_std = valid_rf.std()
            rf_ci = stats.norm.interval(0.95, rf_mean, rf_std)
            
            results.append({
                'feature': feature,
                'mi_score': mi_mean,
                'mi_ci_lower': mi_ci[0],
                'mi_ci_upper': mi_ci[1],
                'rf_importance': rf_mean,
                'rf_ci_lower': rf_ci[0],
                'rf_ci_upper': rf_ci[1]
            })
    
    return pd.DataFrame(results)


def select_features_cv_optimized(X, y, base_features, threshold=0.01):
    """
    Perform forward feature selection using cross-validation.
    """
    print(f"\nStarting CV selection with {len(base_features)} base features")
    
    selected_features = base_features.copy()
    remaining_features = [f for f in X.columns if f not in selected_features]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    estimator = RandomForestClassifier(
        n_estimators=30,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )

    try:
        base_score = cross_val_score(
            estimator,
            X[selected_features],
            y,
            cv=cv,
            scoring='f1'
        ).mean()
        
        print(f"Base CV score: {base_score:.4f}")
        
        improved = True
        while improved and remaining_features:
            improved = False
            scores = {}
            
            for feature in remaining_features:
                current_features = selected_features + [feature]
                try:
                    score = cross_val_score(
                        estimator,
                        X[current_features],
                        y,
                        cv=cv,
                        scoring='f1',
                        n_jobs=-1
                    ).mean()
                    scores[feature] = score
                except Exception as e:
                    print(f"Error evaluating feature {feature}: {str(e)}")
                    continue

            if scores:
                best_feature, best_score = max(scores.items(), key=lambda x: x[1])
                if best_score > base_score + threshold:
                    selected_features.append(best_feature)
                    remaining_features.remove(best_feature)
                    base_score = best_score
                    print(f"Added {best_feature} (new f1: {best_score:.4f})")
                    improved = True
    
    except Exception as e:
        print(f"Error in CV selection: {str(e)}")
        return base_features

    return selected_features


def select_features_optimized(X_train_scaled, y_train, X_test_scaled=None):
    """
    Main feature selection function combining importance evaluation and CV selection.
    """
    print("Starting feature selection process...")
    print(f"Input shape: {X_train_scaled.shape}")

    try:
        # evaluate feature importance with bootstrapping
        importance_df = evaluate_feature_importance_optimized(
            X_train_scaled,
            y_train,
            n_bootstraps=100,
            sample_frac=0.8
        )
        
        print("\nFeature importance evaluation complete.")
        print(f"Evaluated {len(importance_df)} features")

        # identify stable features (positive confidence intervals)
        stable_features = importance_df[
            (importance_df['mi_ci_lower'] > 0) &
            (importance_df['rf_ci_lower'] > 0)
        ]['feature'].tolist()

        print(f"\nFound {len(stable_features)} stable features")

        # select base features (above median importance)
        base_features = importance_df[
            (importance_df['mi_score'] > importance_df['mi_score'].median()) &
            (importance_df['rf_importance'] > importance_df['rf_importance'].median())
        ]['feature'].tolist()

        # perform forward selection on stable features
        selected_features = select_features_cv_optimized(
            X_train_scaled[stable_features],
            y_train,
            base_features
        )
        print("\nFeature Selection Summary:")
        print(f"Initial features: {X_train_scaled.shape[1]}")
        print(f"Stable features: {len(stable_features)}")
        print(f"Final selected features: {len(selected_features)}")

        return selected_features, importance_df
        
        
    except Exception as e:
        print(f"Error in main feature selection: {str(e)}")
        print("Falling back to all features")
        return list(X_train_scaled.columns), pd.DataFrame()

# call the optimized feature selection function
print("select_features_optimized")
final_selected_features, importance_df = select_features_optimized(X_train_scaled, y_train, X_test_scaled)

# review results
print("\nSelected Features:")
for f in final_selected_features:
    print(f"- {f}")

print("\nTop 10 Features by Importance:")
print(importance_df.sort_values(by='rf_importance', ascending=False).head(10))


# keep only selected features in train and test sets
X_train_final_latest = X_train_scaled[final_selected_features]
X_test_final_latest = X_test_scaled[final_selected_features]



def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # find optimal threshold for F1 score
    thresholds = np.arange(0.2, 0.8, 0.05)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        f1_scores.append(f1_score(y_test, y_pred))
    
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (y_pred_proba > optimal_threshold).astype(int)
    
    # CV with custom F1 scoring
    def f1_score_with_threshold(estimator, X, y):
        probs = estimator.predict_proba(X)[:, 1]
        preds = (probs > optimal_threshold).astype(int)
        return f1_score(y, preds)
    
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=5, 
        scoring=make_scorer(f1_score_with_threshold)
    )
    
    print(f"\n{model_name} Results:")
    print("-" * 50)
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.2f}")
    print(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("\nAdditional Metrics:")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Average Precision Score: {average_precision_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, optimal_threshold


# Logistic Regression
print("evaluate_model")
# lr_model = evaluate_model(
#     LogisticRegression(max_iter=1000, class_weight='balanced'),
#     X_train_final_latest, X_test_final_latest, y_train, y_test,
#     "Logistic Regression"
# )

# # Random Forest
# rf_model = evaluate_model(
#     RandomForestClassifier(
#         n_estimators=200,
#         max_depth=10,
#         class_weight='balanced',
#         n_jobs=-1,
#         random_state=42
#     ),
#     X_train_final_latest, X_test_final_latest, y_train, y_test,
#     "Random Forest"
# )

# # Gradient Boosting
# gb_model = evaluate_model(
#     GradientBoostingClassifier(
#         n_estimators=200,
#         max_depth=5,
#         learning_rate=0.1,
#         random_state=42
#     ),
#     X_train_final_latest, X_test_final_latest, y_train, y_test,
#     "Gradient Boosting"
# )


dtrain = xgb.DMatrix(X_train_final_latest, label=y_train)
dtest = xgb.DMatrix(X_test_final_latest, label=y_test)

params = {
    'max_depth': 10,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'seed': 42,
    'min_child_weight': 3,  # Helps prevent overfitting
    'subsample': 0.9,      # Slight randomness
    'colsample_bytree': 0.9,  # Slight feature sampling
    'scale_pos_weight': 68969/4962  # Balance of positive and negative weights
}


# xgboost model
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=20,
    verbose_eval=50
)

# cross-validation
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    early_stopping_rounds=20,
    metrics=['auc', 'logloss'],
    seed=42
)


y_pred_proba = xgb_model.predict(dtest)  # predict probabilities

# find the optimal threshold for F1 score
thresholds = np.arange(0.2, 0.8, 0.05)
f1_scores = []
for threshold in thresholds:
    y_pred_thresh = (y_pred_proba > threshold).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh))

optimal_threshold = thresholds[np.argmax(f1_scores)]
y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)

y_pred = (y_pred_proba > optimal_threshold).astype(int)

# feature importance
importance_dict = xgb_model.get_score(importance_type='gain')
importance_df = pd.DataFrame(
    [(k, v) for k, v in importance_dict.items()],
    columns=['feature', 'importance']
).sort_values('importance', ascending=False)

# print evaluation metrics
print("\nXGBoost Results:")
print("-" * 50)
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))


# define our training and validation data as DMatrix objects
dtrain = xgb.DMatrix(X_train_final_latest, label=y_train)
dval = xgb.DMatrix(X_test_final_latest, label=y_test)

# define parameter grid
param_grid = {
    'max_depth': [3, 5],           
    'eta': [0.1, 0.2],            
    'min_child_weight': [1, 3],    
    'subsample': [0.9],            
    'colsample_bytree': [0.9],     
    'scale_pos_weight': [5, 10]    
}

# function to calculate F1 score for XGBoost predictions
def f1_score_xgb(predt: np.ndarray, dtrain: xgb.DMatrix):
    y_true = dtrain.get_label()
    thresholds = np.arange(0.2, 0.8, 0.05)
    best_f1, best_threshold = 0, None
    for threshold in thresholds:
        y_pred = (predt > threshold).astype(int)
        # explicitly specify pos_label=1 to focus on click predictions
        f1 = f1_score(y_true, y_pred, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    global optimal_threshold
    optimal_threshold = best_threshold
    
    return 'f1', best_f1


# function to perform k-fold cross validation for a set of parameters
def xgb_cv_score(params, dtrain, num_boost_round=50, nfold=2):
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=nfold,
        feval=f1_score_xgb,
        maximize=True,
        early_stopping_rounds=10,
        verbose_eval=False
    )
    return cv_results['test-f1-mean'].max()


# initialize best parameters and score
best_params = None
best_score = 0

# base, constant parameters
base_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'seed': 42
}

print("Starting grid search...")

# Calculate total number of combinations for tqdm
param_combinations = list(product(
    param_grid['max_depth'],
    param_grid['eta'],
    param_grid['min_child_weight'],
    param_grid['subsample'],
    param_grid['colsample_bytree'],
    param_grid['scale_pos_weight']
))

# manual grid search with tqdm
for max_depth, eta, min_child_weight, subsample, colsample_bytree, scale_pos_weight in tqdm(param_combinations, desc="Grid Search Progress"):

    params = {
        **base_params,
        'max_depth': max_depth,
        'eta': eta,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'scale_pos_weight': scale_pos_weight
    }
    
    score = xgb_cv_score(params, dtrain)
    
    # update best parameters if score is better
    if score > best_score:
        best_score = score
        best_params = params.copy()
        print(f"\nNew best F1 score: {best_score:.4f}")
        print("Parameters:", best_params)

print("\nBest parameters found:")
print(best_params)
print(f"Best CV F1 score: {best_score:.4f}")


# # train final model with best parameters
# final_model = xgb.train(
#     best_params,
#     dtrain,
#     num_boost_round=100,
#     evals=[(dtrain, 'train'), (dval, 'val')],
#     feval=f1_score_xgb,
#     early_stopping_rounds=20,
#     verbose_eval=False
# )


# calculate training metrics
training_metrics = {
    'positive_rate': (y_train == 1).mean(),
    'probability_distribution': {
        'train_probabilities': xgb_model.predict(dtrain),
        'val_probabilities': xgb_model.predict(dval)
    },
    'best_f1': max(f1_scores),
    'optimal_threshold': optimal_threshold,
    'class_balance': len(y_train[y_train == 0]) / len(y_train[y_train == 1])
}

# probability distribution stats
for data_split in ['train_probabilities', 'val_probabilities']:
    probs = training_metrics['probability_distribution'][data_split]
    training_metrics['probability_distribution'][f'{data_split}_stats'] = {
        'min': probs.min(),
        'max': probs.max(),
        'mean': probs.mean(),
        'std': probs.std()
    }

# metrics summary
print("\nTraining Data Metrics:")
print("-" * 50)
print(f"Positive rate (CTR): {training_metrics['positive_rate']:.4%}")
print(f"Class imbalance ratio: {training_metrics['class_balance']:.2f}:1")
print(f"Optimal threshold: {training_metrics['optimal_threshold']:.4f}")
print(f"Best F1 score: {training_metrics['best_f1']:.4f}")

print("\nProbability Distributions:")
print("-" * 50)
print("Training set:")
for k, v in training_metrics['probability_distribution']['train_probabilities_stats'].items():
    print(f"{k}: {v:.4f}")
print("\nValidation set:")
for k, v in training_metrics['probability_distribution']['val_probabilities_stats'].items():
    print(f"{k}: {v:.4f}")


# ## Pre-Processing the Test Batch for Prediction


# load the test data
test_data = pd.read_csv('X_test_1st.csv')

print(test_data.shape)


# saveing both the scaler and model:
# joblib.dump(scaler, 'feature_scaler.joblib')
# joblib.dump(final_model, 'final_model_amir.joblib')


def calculate_empirical_defaults(df):
    """
    Calculate empirical default values from training data.
    """
    engagement_defaults = {
        "historical_user_ctr": df['historical_user_ctr'].mean(),
        "session_count_log": df['session_count_log'].median(),
        "sessions_per_day_mean": df['sessions_per_day_mean'].mean(),
        "sessions_per_day_mean_log": df['sessions_per_day_mean_log'].median(),
        "time_since_last_click": df['time_since_last_click'].median(),
        "click_frequency_24h": df['click_frequency_24h'].mean(),
        "click_frequency_24h_log": df['click_frequency_24h_log'].median(),
        "historical_user_ctr_rank": 0.5  # Use middle rank for new users
    }
    
    # convert campaign-hour tuples to strings for JSON serialization
    campaign_hour_perf = df.groupby(['campaign_id', 'hour'])['campaign_hour_relative'].mean()
    campaign_hour_dict = {
        f"{campaign_id}_{hour}": value 
        for (campaign_id, hour), value in campaign_hour_perf.items()
    }
    
    campaign_defaults = {
        # global campaign metrics
        "campaign_historical_ctr_log": df.groupby('campaign_id')['campaign_historical_ctr_log'].mean().mean(),
        "campaign_success_percentile": 0.5,
        "campaign_webpage_relative": 1.0,
        "campaign_hour_relative": 1.0,
        
        # campaign-specific metrics
        "campaign_ctrs": df.groupby('campaign_id')['campaign_historical_ctr_log'].mean().to_dict(),
        "campaign_hour_performance": campaign_hour_dict
    }
    
    feature_statistics = {
        "historical_user_ctr_percentiles": {
            "p25": df['historical_user_ctr'].quantile(0.25),
            "p75": df['historical_user_ctr'].quantile(0.75)
        },
        "session_count_log_percentiles": {
            "p25": df['session_count_log'].quantile(0.25),
            "p75": df['session_count_log'].quantile(0.75)
        }
    }
    
    return {
        "engagement_defaults": engagement_defaults,
        "campaign_defaults": campaign_defaults,
        "feature_statistics": feature_statistics
    }

def create_feature_params(df_train, features_to_scale, selected_features, output_path='feature_params.json'):
    """
    Creates complete feature parameters and saves to JSON.
    """
    # calculate empirical defaults
    empirical_defaults = calculate_empirical_defaults(df_train)
    
    # combine all parameters
    feature_params = {
        "features_to_scale": features_to_scale,
        "selected_features": selected_features,
        "engagement_defaults": empirical_defaults["engagement_defaults"],
        "campaign_defaults": empirical_defaults["campaign_defaults"],
        "feature_statistics": empirical_defaults["feature_statistics"],
        "categorical_features": categorical_features,
        "keep_as_is": keep_as_is
    }
    
    save_feature_params(feature_params, output_path)
    
    return feature_params

def save_feature_params(params, output_path):
    """Save feature parameters to JSON file with proper rounding."""
    def round_nested_dict(d, decimals=6):
        if isinstance(d, dict):
            return {k: round_nested_dict(v, decimals) for k, v in d.items()}
        elif isinstance(d, float):
            return round(d, decimals)
        return d
    
    rounded_params = round_nested_dict(params)
    
    # save to JSON with proper formatting
    with open(output_path, 'w') as f:
        json.dump(rounded_params, f, indent=4)

def load_feature_params(params_path):
    """Load feature parameters from JSON file."""
    with open(params_path, 'r') as f:
        return json.load(f)



# create and save feature parameters
feature_params = create_feature_params(
    df_train=df_engineered,
    features_to_scale=features_to_scale,
    selected_features=final_selected_features,
    output_path='feature_params.json'
)



#%%


# load parameters for test set feature creation
loaded_params = load_feature_params('feature_params.json')


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
    
    return binary_predictions


def process_test_data(test_filepath, df, imputation_params_path, feature_params_path, 
                     scaler_path, model_path):
    """Process test data while maintaining original order"""
    # load and preserve order
    
    if df is not None:
        test_df = df
    else:
        test_df = pd.read_csv(test_filepath)
        test_df['_order_tracking'] = np.arange(len(test_df))

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

    # write predictions to CSV
    output_df = pd.DataFrame({'prediction': predictions})
    output_df.to_csv('predictions.csv', index=False)

    return


# generate predictions for test data


#run cell 


predictions = process_test_data(
    test_filepath='X_test_1st.csv',
    imputation_params_path='imputation_params.json',
    feature_params_path='feature_params.json',
    scaler_path='feature_scaler.joblib',
    model_path='final_model.joblib'
)



# %%
