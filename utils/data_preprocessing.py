"""
Data Preprocessing Utilities for Fraud Detection
Handles data loading, cleaning, scaling, and synthetic data generation
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=FutureWarning)


class DataPreprocessor:
    """
    Comprehensive data preprocessing for credit card fraud detection
    """

    def __init__(self, config_path=None, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.imputer = None
        self.config = self._load_config(config_path)

        # Initialize components
        self._init_scaler()
        self._init_imputer()

    def _load_config(self, config_path):
        """Load configuration from file or use defaults"""
        default_config = {
            'data': {
                'test_size': 0.2,
                'random_state': 42,
                'scale_features': True,
                'handle_missing': True,
                'remove_outliers': False,
                'balance_data': False
            },
            'preprocessing': {
                'scaler_type': 'standard',
                'imputer_type': 'median',
                'outlier_method': 'iqr',
                'balance_method': 'smote'
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge with defaults
            default_config.update(user_config)

        return default_config

    def _init_scaler(self):
        """Initialize scaler based on configuration"""
        scaler_type = self.config.get('preprocessing', {}).get('scaler_type', 'standard')

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}, using StandardScaler")
            self.scaler = StandardScaler()

    def _init_imputer(self):
        """Initialize imputer based on configuration"""
        imputer_type = self.config.get('preprocessing', {}).get('imputer_type', 'median')

        if imputer_type == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        elif imputer_type == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif imputer_type == 'mode':
            self.imputer = SimpleImputer(strategy='most_frequent')
        elif imputer_type == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            logger.warning(f"Unknown imputer type: {imputer_type}, using median")
            self.imputer = SimpleImputer(strategy='median')

    def load_real_data(self, filepath):
        """Load real credit card fraud dataset"""
        try:
            logger.info(f"Loading data from: {filepath}")
            df = pd.read_csv(filepath)

            # Validate required columns
            required_columns = ['Class']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing}")

            logger.info(f"Data loaded successfully: {df.shape}")
            logger.info(f"Fraud ratio: {df['Class'].mean():.4f}")

            return df

        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            logger.info("Generating synthetic data instead...")
            return self.generate_synthetic_data()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def generate_synthetic_data(self, n_normal=50000, n_fraud=1000, n_features=29,
                               complexity='medium', noise_level=0.1):
        """
        Generate synthetic credit card fraud dataset with various complexity levels
        """
        logger.info(f"Generating synthetic data:")
        logger.info(f"  Normal samples: {n_normal}")
        logger.info(f"  Fraud samples: {n_fraud}")
        logger.info(f"  Features: {n_features}")
        logger.info(f"  Complexity: {complexity}")

        np.random.seed(self.random_state)

        if complexity == 'simple':
            # Simple separation between normal and fraud
            X_normal = np.random.normal(0, 1, (n_normal, n_features))
            X_fraud = np.random.normal(2, 1.5, (n_fraud, n_features))

        elif complexity == 'medium':
            # Medium complexity with some overlap
            X_normal = np.random.multivariate_normal(
                mean=np.zeros(n_features),
                cov=np.eye(n_features) * 0.5,
                size=n_normal
            )

            # Multiple fraud patterns
            n_fraud_patterns = 3
            fraud_samples_per_pattern = n_fraud // n_fraud_patterns
            X_fraud_list = []

            for i in range(n_fraud_patterns):
                fraud_mean = np.random.uniform(-2, 2, n_features)
                fraud_cov = np.eye(n_features) * np.random.uniform(1.0, 2.0)

                X_fraud_pattern = np.random.multivariate_normal(
                    mean=fraud_mean,
                    cov=fraud_cov,
                    size=fraud_samples_per_pattern
                )
                X_fraud_list.append(X_fraud_pattern)

            # Handle remaining samples
            remaining_samples = n_fraud - len(X_fraud_list) * fraud_samples_per_pattern
            if remaining_samples > 0:
                X_fraud_remaining = np.random.multivariate_normal(
                    mean=np.random.uniform(-2, 2, n_features),
                    cov=np.eye(n_features) * 1.5,
                    size=remaining_samples
                )
                X_fraud_list.append(X_fraud_remaining)

            X_fraud = np.vstack(X_fraud_list)

        elif complexity == 'hard':
            # Complex patterns with significant overlap
            # Normal data with multiple clusters
            normal_clusters = 5
            normal_samples_per_cluster = n_normal // normal_clusters
            X_normal_list = []

            for i in range(normal_clusters):
                cluster_center = np.random.uniform(-1, 1, n_features)
                cluster_cov = np.eye(n_features) * np.random.uniform(0.3, 0.8)

                X_normal_cluster = np.random.multivariate_normal(
                    mean=cluster_center,
                    cov=cluster_cov,
                    size=normal_samples_per_cluster
                )
                X_normal_list.append(X_normal_cluster)

            # Handle remaining samples
            remaining_normal = n_normal - len(X_normal_list) * normal_samples_per_cluster
            if remaining_normal > 0:
                X_normal_remaining = np.random.multivariate_normal(
                    mean=np.zeros(n_features),
                    cov=np.eye(n_features) * 0.5,
                    size=remaining_normal
                )
                X_normal_list.append(X_normal_remaining)

            X_normal = np.vstack(X_normal_list)

            # Fraud data with subtle differences
            fraud_patterns = 4
            fraud_samples_per_pattern = n_fraud // fraud_patterns
            X_fraud_list = []

            for i in range(fraud_patterns):
                # Base pattern on normal clusters but with modifications
                base_center = np.random.uniform(-1, 1, n_features)
                fraud_center = base_center + np.random.uniform(-0.5, 0.5, n_features)
                fraud_cov = np.eye(n_features) * np.random.uniform(0.8, 1.5)

                X_fraud_pattern = np.random.multivariate_normal(
                    mean=fraud_center,
                    cov=fraud_cov,
                    size=fraud_samples_per_pattern
                )
                X_fraud_list.append(X_fraud_pattern)

            # Handle remaining fraud samples
            remaining_fraud = n_fraud - len(X_fraud_list) * fraud_samples_per_pattern
            if remaining_fraud > 0:
                X_fraud_remaining = np.random.multivariate_normal(
                    mean=np.random.uniform(-1, 1, n_features),
                    cov=np.eye(n_features) * 1.2,
                    size=remaining_fraud
                )
                X_fraud_list.append(X_fraud_remaining)

            X_fraud = np.vstack(X_fraud_list)

        # Add noise if specified
        if noise_level > 0:
            X_normal += np.random.normal(0, noise_level, X_normal.shape)
            X_fraud += np.random.normal(0, noise_level, X_fraud.shape)

        # Combine data
        X = np.vstack([X_normal, X_fraud])
        y = np.hstack([np.zeros(n_normal), np.ones(n_fraud)])

        # Create feature names (수정된 부분)
        feature_names = [f'V{i}' for i in range(1, n_features + 1)]

        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['Class'] = y

        # Add realistic Time and Amount distributions
        self._add_realistic_features(df, n_normal, n_fraud)

        logger.info(f"Synthetic data generated successfully")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Fraud ratio: {df['Class'].mean():.4f}")

        return df

    def _add_realistic_features(self, df, n_normal, n_fraud):
        """Add realistic Time and Amount features to synthetic data"""
        # Time: 48 hours in seconds with some patterns
        df['Time'] = np.random.uniform(0, 172800, len(df))

        # Amount: Log-normal distribution with different patterns for normal vs fraud
        normal_amounts = np.random.lognormal(mean=3.0, sigma=1.2, size=n_normal)
        fraud_amounts = np.random.lognormal(mean=3.5, sigma=1.5, size=n_fraud)

        # Clip extreme values
        normal_amounts = np.clip(normal_amounts, 0.01, 10000)
        fraud_amounts = np.clip(fraud_amounts, 0.01, 25000)

        # Assign amounts based on class
        df.loc[df['Class'] == 0, 'Amount'] = normal_amounts
        df.loc[df['Class'] == 1, 'Amount'] = fraud_amounts

    def preprocess_data(self, df):
        """
        Complete data preprocessing pipeline
        """
        logger.info("Preprocessing data...")
        df_processed = df.copy()

        # Handle missing values
        if self.config['data']['handle_missing']:
            missing_counts = df_processed.isnull().sum()
            if missing_counts.sum() > 0:
                logger.info(f"Found {missing_counts.sum()} missing values")

                # Separate numerical and categorical columns
                numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
                numerical_cols = [col for col in numerical_cols if col != 'Class']

                if len(numerical_cols) > 0:
                    df_processed[numerical_cols] = self.imputer.fit_transform(df_processed[numerical_cols])
                    logger.info("Missing values imputed")

        # Remove outliers if specified
        if self.config['data']['remove_outliers']:
            df_processed = self._remove_outliers(df_processed)

        # Handle infinite values
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        if df_processed.isnull().sum().sum() > 0:
            df_processed = df_processed.fillna(df_processed.median(numeric_only=True))

        # Remove Time column (usually not useful for fraud detection)
        if 'Time' in df_processed.columns:
            df_processed = df_processed.drop(['Time'], axis=1)
            logger.info("Time column removed")

        # Scale Amount column if present
        if 'Amount' in df_processed.columns and self.config['data']['scale_features']:
            amount_scaled = self.scaler.fit_transform(df_processed[['Amount']])
            df_processed['Amount'] = amount_scaled.flatten()
            logger.info("Amount column scaled")

        logger.info(f"Data preprocessing completed. Shape: {df_processed.shape}")
        return df_processed

    def _remove_outliers(self, df):
        """Remove outliers using IQR or Z-score method"""
        method = self.config.get('preprocessing', {}).get('outlier_method', 'iqr')

        df_no_outliers = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['Class', 'Time']]

        outlier_indices = set()

        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index

            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = df[z_scores > 3].index

            outlier_indices.update(col_outliers)

        df_no_outliers = df_no_outliers.drop(outlier_indices)
        logger.info(f"Removed {len(outlier_indices)} outliers")

        return df_no_outliers

    def split_data(self, df, test_size=None, validation_split=False, stratify=True):
        """
        Split data into train/test/validation sets
        """
        if test_size is None:
            test_size = self.config['data']['test_size']

        logger.info(f"Splitting data (test_size={test_size}, validation={validation_split})")

        # Separate features and target
        if 'Class' in df.columns:
            X = df.drop(['Class'], axis=1)
            y = df['Class']
        else:
            logger.warning("No 'Class' column found, returning features only")
            return df.values, None

        stratify_param = y if stratify else None

        if validation_split:
            # Three-way split
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self.random_state
            )
            train_val_idx, test_idx = next(splitter.split(X, y))

            X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
            y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]

            # Split train_val into train and validation
            val_size = 0.25  # 25% of remaining data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size,
                random_state=self.random_state, stratify=y_train_val
            )

            logger.info(f"Data split completed:")
            logger.info(f"  Train: {X_train.shape[0]} samples")
            logger.info(f"  Validation: {X_val.shape[0]} samples")
            logger.info(f"  Test: {X_test.shape[0]} samples")

            return (X_train.values, X_val.values, X_test.values,
                   y_train.values, y_val.values, y_test.values)
        else:
            # Two-way split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size,
                random_state=self.random_state, stratify=stratify_param
            )

            logger.info(f"Data split completed:")
            logger.info(f"  Train: {X_train.shape[0]} samples")
            logger.info(f"  Test: {X_test.shape[0]} samples")

            return X_train.values, X_test.values, y_train.values, y_test.values

    def balance_dataset(self, X, y):
        """
        Balance the dataset using various techniques
        """
        if not self.config['data']['balance_data']:
            return X, y

        method = self.config.get('preprocessing', {}).get('balance_method', 'smote')
        logger.info(f"Balancing dataset using {method}")

        original_ratio = np.mean(y)
        logger.info(f"Original fraud ratio: {original_ratio:.4f}")

        try:
            if method == 'smote':
                balancer = SMOTE(random_state=self.random_state)
            elif method == 'adasyn':
                balancer = ADASYN(random_state=self.random_state)
            elif method == 'undersample':
                balancer = RandomUnderSampler(random_state=self.random_state)
            elif method == 'edited_nn':
                balancer = EditedNearestNeighbours()
            else:
                logger.warning(f"Unknown balance method: {method}, skipping balancing")
                return X, y

            X_balanced, y_balanced = balancer.fit_resample(X, y)

            balanced_ratio = np.mean(y_balanced)
            logger.info(f"Balanced fraud ratio: {balanced_ratio:.4f}")
            logger.info(f"Shape change: {X.shape} -> {X_balanced.shape}")

            return X_balanced, y_balanced

        except Exception as e:
            logger.warning(f"Balancing failed: {e}, using original data")
            return X, y

    def get_preprocessing_info(self):
        """Get information about preprocessing steps"""
        return {
            'scaler_type': type(self.scaler).__name__,
            'imputer_type': type(self.imputer).__name__,
            'config': self.config,
            'random_state': self.random_state
        }


class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for fraud detection
    """

    def __init__(self):
        self.feature_history = {}

    def create_time_features(self, df):
        """Create time-based features if Time column exists"""
        if 'Time' not in df.columns:
            return df

        df_enhanced = df.copy()

        # Convert time to hours, days, etc.
        df_enhanced['Hour'] = (df_enhanced['Time'] // 3600) % 24
        df_enhanced['Day'] = df_enhanced['Time'] // (24 * 3600)
        df_enhanced['Hour_sin'] = np.sin(2 * np.pi * df_enhanced['Hour'] / 24)
        df_enhanced['Hour_cos'] = np.cos(2 * np.pi * df_enhanced['Hour'] / 24)

        logger.info("Time-based features created")
        return df_enhanced

    def create_amount_features(self, df):
        """Create amount-based features"""
        if 'Amount' not in df.columns:
            return df

        df_enhanced = df.copy()

        # Log transformation
        df_enhanced['Amount_log'] = np.log1p(df_enhanced['Amount'])

        # Amount bins
        df_enhanced['Amount_bin'] = pd.cut(df_enhanced['Amount'],
                                         bins=5, labels=False)

        # Amount standardized by customer (if customer info available)
        # This is a placeholder - would need customer grouping
        df_enhanced['Amount_normalized'] = (df_enhanced['Amount'] - df_enhanced['Amount'].mean()) / df_enhanced['Amount'].std()

        logger.info("Amount-based features created")
        return df_enhanced

    def create_statistical_features(self, df):
        """Create statistical features from existing features"""
        df_enhanced = df.copy()

        # Get numerical columns (exclude target and ID columns)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col not in ['Class', 'Time', 'Amount']]

        if len(feature_cols) > 0:
            feature_data = df_enhanced[feature_cols]

            # Statistical aggregations
            df_enhanced['Feature_sum'] = feature_data.sum(axis=1)
            df_enhanced['Feature_mean'] = feature_data.mean(axis=1)
            df_enhanced['Feature_std'] = feature_data.std(axis=1)
            df_enhanced['Feature_min'] = feature_data.min(axis=1)
            df_enhanced['Feature_max'] = feature_data.max(axis=1)
            df_enhanced['Feature_range'] = df_enhanced['Feature_max'] - df_enhanced['Feature_min']

            # Number of zero values
            df_enhanced['Zero_count'] = (feature_data == 0).sum(axis=1)

            # Number of negative values
            df_enhanced['Negative_count'] = (feature_data < 0).sum(axis=1)

            logger.info(f"Statistical features created from {len(feature_cols)} base features")

        return df_enhanced


class DataValidator:
    """
    Validate data quality and detect issues
    """

    @staticmethod
    def validate_fraud_dataset(df):
        """Validate fraud detection dataset"""
        issues = []

        # Check required columns
        if 'Class' not in df.columns:
            issues.append("Missing 'Class' target column")

        # Check data types
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        non_target_non_numeric = [col for col in non_numeric_cols if col != 'Class']
        if len(non_target_non_numeric) > 0:
            issues.append(f"Non-numeric feature columns: {non_target_non_numeric}")

        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            issues.append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")

        # Check class distribution
        if 'Class' in df.columns:
            class_dist = df['Class'].value_counts(normalize=True)
            fraud_ratio = class_dist.get(1, 0)

            if fraud_ratio == 0:
                issues.append("No fraud cases found in dataset")
            elif fraud_ratio > 0.5:
                issues.append(f"Unusual fraud ratio: {fraud_ratio:.3f} (typically < 0.05)")
            elif fraud_ratio < 0.001:
                issues.append(f"Very low fraud ratio: {fraud_ratio:.4f} (may cause training issues)")

        # Check for infinite values
        inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
        if inf_counts.sum() > 0:
            issues.append(f"Infinite values found: {inf_counts[inf_counts > 0].to_dict()}")

        # Check for constant columns
        numeric_df = df.select_dtypes(include=[np.number])
        constant_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"Constant columns found: {constant_cols}")

        return issues

    @staticmethod
    def print_data_summary(df):
        """Print comprehensive data summary"""
        print("=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)

        print(f"Dataset Shape: {df.shape}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Target distribution
        if 'Class' in df.columns:
            class_counts = df['Class'].value_counts()
            print(f"\nClass Distribution:")
            print(f"  Normal (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(df)*100:.2f}%)")
            print(f"  Fraud (1):  {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(df)*100:.2f}%)")

        # Data types
        print(f"\nData Types:")
        for dtype in df.dtypes.value_counts().index:
            count = df.dtypes.value_counts()[dtype]
            print(f"  {dtype}: {count} columns")

        # Missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"\nMissing Values:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
        else:
            print(f"\nMissing Values: None")

        print("=" * 60)


# Utility functions
def load_and_preprocess_data(filepath=None, config_path=None, **kwargs):
    """
    Convenience function for loading and preprocessing data
    """
    preprocessor = DataPreprocessor(config_path=config_path)

    if filepath:
        df = preprocessor.load_real_data(filepath)
    else:
        df = preprocessor.generate_synthetic_data(**kwargs)

    df_processed = preprocessor.preprocess_data(df)
    return preprocessor.split_data(df_processed)


# Export main classes and functions
__all__ = [
    'DataPreprocessor',
    'AdvancedFeatureEngineer',
    'DataValidator',
    'load_and_preprocess_data'
]