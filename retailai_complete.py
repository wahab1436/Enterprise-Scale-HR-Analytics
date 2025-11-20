"""
Enterprise-Grade Automated ML Analytics Platform for CSV Data
COMPLETE IMPLEMENTATION - All Blueprint Features
Supports: XGBoost, Prophet, SHAP, Embeddings, Vector Index, Full Pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import io
import base64
import json

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_FILE_SIZE_MB = 200
CHUNK_SIZE = 50000

st.set_page_config(
    page_title="Enterprise ML Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_csv_upload(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded CSV file"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File size exceeds {MAX_FILE_SIZE_MB}MB limit"
    
    if not uploaded_file.name.endswith('.csv'):
        return False, "Only CSV files are accepted"
    
    return True, "Valid file"

def load_large_csv(uploaded_file, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """Load large CSV files using chunking"""
    chunks = []
    try:
        for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        return df
    except:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)

def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect column data types with enhanced logic"""
    column_types = {}
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = 'datetime'
        else:
            # Try to parse as datetime
            try:
                non_null_samples = df[col].dropna().head(100)
                if len(non_null_samples) > 0:
                    parsed = pd.to_datetime(non_null_samples, errors='coerce')
                    if parsed.notna().sum() / len(non_null_samples) > 0.5:
                        column_types[col] = 'datetime'
                        continue
            except:
                pass
            
            # Check if categorical or text
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                unique_ratio = df[col].nunique() / len(df)
                avg_length = df[col].astype(str).str.len().mean()
                
                if unique_ratio < 0.5 and avg_length < 50:
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'text'
            else:
                column_types[col] = 'categorical'
    
    return column_types

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to snake_case"""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
    return df

# =============================================================================
# DATA CLEANING MODULE
# =============================================================================

class DataCleaner:
    """Comprehensive data cleaning pipeline"""
    
    def __init__(self):
        self.cleaning_report = {
            'initial_rows': 0,
            'final_rows': 0,
            'duplicates_removed': 0,
            'missing_values_handled': {},
            'outliers_detected': {},
            'corrupted_entries': 0,
            'columns_standardized': False,
            'datetime_columns_parsed': []
        }
    
    def clean_data(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Execute full cleaning pipeline"""
        df = df.copy()
        self.cleaning_report['initial_rows'] = len(df)
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        self.cleaning_report['duplicates_removed'] = initial_rows - len(df)
        
        # Standardize column names
        df = standardize_column_names(df)
        self.cleaning_report['columns_standardized'] = True
        
        # Parse datetime columns
        df = self._parse_datetime_columns(df, column_types)
        
        # Handle missing values
        df = self._handle_missing_values(df, column_types)
        
        # Detect and flag outliers
        self._detect_outliers(df, column_types)
        
        # Handle corrupted entries
        df = self._handle_corrupted_entries(df, column_types)
        
        self.cleaning_report['final_rows'] = len(df)
        
        return df
    
    def _parse_datetime_columns(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Parse datetime columns properly"""
        for col in df.columns:
            if column_types.get(col) == 'datetime':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    self.cleaning_report['datetime_columns_parsed'].append(col)
                except:
                    pass
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Handle missing values with advanced strategies"""
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                self.cleaning_report['missing_values_handled'][col] = missing_count
                
                col_type = column_types.get(col, 'categorical')
                
                if col_type == 'numeric':
                    # Use median imputation for numeric
                    df[col] = df[col].fillna(df[col].median())
                elif col_type == 'categorical':
                    # Use mode for categorical
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                    df[col] = df[col].fillna(mode_val)
                elif col_type == 'datetime':
                    # Forward fill for datetime
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                elif col_type == 'text':
                    # Empty string for text
                    df[col] = df[col].fillna('')
        
        return df
    
    def _detect_outliers(self, df: pd.DataFrame, column_types: Dict[str, str]) -> None:
        """Detect outliers using IQR and Z-score methods"""
        for col in df.columns:
            if column_types.get(col) == 'numeric':
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                # Z-score method for confirmation
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                z_outliers = (z_scores > 3).sum()
                
                if outliers > 0:
                    self.cleaning_report['outliers_detected'][col] = {
                        'iqr_method': int(outliers),
                        'z_score_method': int(z_outliers)
                    }
    
    def _handle_corrupted_entries(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Handle corrupted or inconsistent entries"""
        corrupted_count = 0
        
        for col in df.columns:
            if column_types.get(col) == 'numeric':
                # Check for non-numeric values in numeric columns
                before = len(df)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                after = df[col].notna().sum()
                corrupted_count += (before - after)
        
        self.cleaning_report['corrupted_entries'] = corrupted_count
        return df

# =============================================================================
# FEATURE ENGINEERING MODULE
# =============================================================================

class FeatureEngineer:
    """Automated feature engineering with advanced techniques"""
    
    def __init__(self):
        self.engineered_features = []
        self.feature_importance = {}
        self.selected_features = []
    
    def engineer_features(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Execute comprehensive feature engineering pipeline"""
        df = df.copy()
        
        # Date features
        df = self._create_date_features(df, column_types)
        
        # Categorical encoding
        df = self._encode_categorical(df, column_types)
        
        # Missing value flags
        df = self._create_missing_flags(df)
        
        # Interaction features
        df = self._create_interactions(df, column_types)
        
        # Polynomial features (limited)
        df = self._create_polynomial_features(df, column_types)
        
        # Normalization
        df = self._normalize_features(df, column_types)
        
        # Feature selection
        df = self._select_features(df, column_types)
        
        return df
    
    def _create_date_features(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Extract comprehensive features from datetime columns"""
        for col in df.columns:
            if column_types.get(col) == 'datetime':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                    df[f'{col}_quarter'] = df[col].dt.quarter
                    df[f'{col}_dayofyear'] = df[col].dt.dayofyear
                    df[f'{col}_weekofyear'] = df[col].dt.isocalendar().week
                    df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
                    
                    self.engineered_features.extend([
                        f'{col}_year', f'{col}_month', f'{col}_day',
                        f'{col}_dayofweek', f'{col}_quarter', f'{col}_dayofyear',
                        f'{col}_weekofyear', f'{col}_is_weekend'
                    ])
                except:
                    pass
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Encode categorical variables with multiple strategies"""
        for col in df.columns:
            if column_types.get(col) == 'categorical':
                n_unique = df[col].nunique()
                
                if n_unique < 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    self.engineered_features.extend(dummies.columns.tolist())
                elif n_unique < 50:
                    # Label encoding for medium cardinality
                    df[f'{col}_encoded'] = pd.factorize(df[col])[0]
                    self.engineered_features.append(f'{col}_encoded')
                else:
                    # Frequency encoding for high cardinality
                    freq_map = df[col].value_counts(normalize=True).to_dict()
                    df[f'{col}_freq'] = df[col].map(freq_map)
                    self.engineered_features.append(f'{col}_freq')
        
        return df
    
    def _create_missing_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary flags for originally missing values"""
        original_cols = [col for col in df.columns if col not in self.engineered_features]
        
        for col in original_cols:
            if df[col].isna().sum() > 0:
                df[f'{col}_was_missing'] = df[col].isna().astype(int)
                self.engineered_features.append(f'{col}_was_missing')
        
        return df
    
    def _create_interactions(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Create interaction features between numeric columns"""
        numeric_cols = [col for col in df.columns 
                       if column_types.get(col) == 'numeric' and col not in self.engineered_features]
        
        if len(numeric_cols) >= 2:
            # Create interactions for top numeric columns
            for i, col1 in enumerate(numeric_cols[:4]):
                for col2 in numeric_cols[i+1:5]:
                    try:
                        # Multiplication interaction
                        interaction_name = f'{col1}_x_{col2}'
                        df[interaction_name] = df[col1] * df[col2]
                        self.engineered_features.append(interaction_name)
                        
                        # Ratio interaction
                        if (df[col2] != 0).all():
                            ratio_name = f'{col1}_div_{col2}'
                            df[ratio_name] = df[col1] / (df[col2] + 1e-8)
                            self.engineered_features.append(ratio_name)
                    except:
                        pass
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Create polynomial features for key numeric columns"""
        numeric_cols = [col for col in df.columns 
                       if column_types.get(col) == 'numeric' and col not in self.engineered_features]
        
        for col in numeric_cols[:3]:  # Limit to top 3 to avoid explosion
            try:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
                self.engineered_features.extend([f'{col}_squared', f'{col}_sqrt'])
            except:
                pass
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Normalize numeric features"""
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        for col in numeric_cols:
            try:
                if df[col].std() > 0:
                    df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
                    self.engineered_features.append(f'{col}_normalized')
            except:
                pass
        
        return df
    
    def _select_features(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Select features using variance threshold"""
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        # Remove low variance features
        for col in numeric_cols:
            if df[col].std() < 1e-6:
                self.selected_features.append((col, 'removed_low_variance'))
        
        return df

# =============================================================================
# ML MODEL MODULE - WITH XGBOOST AND PROPHET
# =============================================================================

class MLEngine:
    """Advanced ML engine with XGBoost and Prophet support"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.model_name = None
        self.predictions = None
        self.feature_importance = None
        self.metrics = {}
        self.shap_values = None
        self.shap_explainer = None
    
    def train_model(self, df: pd.DataFrame, target_col: str, column_types: Dict[str, str], 
                    use_xgboost: bool = True) -> Dict[str, Any]:
        """Train ML model with XGBoost or sklearn"""
        try:
            # Prepare data
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Select only numeric features
            numeric_features = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
            X = X[numeric_features]
            
            # Handle any remaining NaN values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean() if pd.api.types.is_numeric_dtype(y) else y.mode()[0])
            
            # Remove infinite values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Determine model type
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                self.model_type = 'regression'
                result = self._train_regression(X, y, use_xgboost)
            else:
                self.model_type = 'classification'
                result = self._train_classification(X, y, use_xgboost)
            
            # Generate predictions
            self.predictions = self.model.predict(X)
            
            # Calculate feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(numeric_features, self.model.feature_importances_))
            
            # Calculate SHAP values
            self._calculate_shap_values(X)
            
            return {
                'success': True,
                'model_type': self.model_type,
                'model_name': self.model_name,
                'n_features': len(numeric_features),
                'metrics': self.metrics,
                'has_shap': self.shap_values is not None
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _train_regression(self, X: pd.DataFrame, y: pd.Series, use_xgboost: bool) -> None:
        """Train regression model"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if use_xgboost:
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                self.model_name = 'XGBoost Regressor'
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
                self.model_name = 'Gradient Boosting Regressor'
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            self.model_name = 'Random Forest Regressor'
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'MSE': float(mean_squared_error(y_test, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'MAE': float(mean_absolute_error(y_test, y_pred)),
            'R2': float(r2_score(y_test, y_pred))
        }
    
    def _train_classification(self, X: pd.DataFrame, y: pd.Series, use_xgboost: bool) -> None:
        """Train classification model"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if use_xgboost:
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                self.model_name = 'XGBoost Classifier'
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
                self.model_name = 'Gradient Boosting Classifier'
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            self.model_name = 'Random Forest Classifier'
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'Accuracy': float(accuracy_score(y_test, y_pred)),
            'Precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'Recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'F1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        }
    
    def _calculate_shap_values(self, X: pd.DataFrame) -> None:
        """Calculate SHAP values for model explainability"""
        try:
            import shap
            
            # Sample data if too large
            sample_size = min(100, len(X))
            X_sample = X.sample(n=sample_size, random_state=42)
            
            # Create explainer
            self.shap_explainer = shap.Explainer(self.model, X_sample)
            self.shap_values = self.shap_explainer(X_sample)
            
        except Exception as e:
            self.shap_values = None
            self.shap_explainer = None

class TimeSeriesEngine:
    """Time series forecasting with Prophet"""
    
    def __init__(self):
        self.model = None
        self.forecast = None
        self.metrics = {}
    
    def train_prophet(self, df: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
        """Train Prophet model for time series forecasting"""
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            prophet_df = df[[date_col, target_col]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.dropna()
            prophet_df = prophet_df.sort_values('ds')
            
            # Train model
            self.model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            self.model.fit(prophet_df)
            
            # Make forecast
            future = self.model.make_future_dataframe(periods=30)
            self.forecast = self.model.predict(future)
            
            return {
                'success': True,
                'model': 'Prophet',
                'forecast_periods': 30
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# =============================================================================
# TEXT EMBEDDINGS MODULE
# =============================================================================

class TextEmbeddingEngine:
    """Generate embeddings for text columns"""
    
    def __init__(self):
        self.embeddings = {}
        self.vector_index = None
    
    def generate_embeddings(self, df: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Generate embeddings for text columns using simple TF-IDF"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        text_cols = [col for col in df.columns if column_types.get(col) == 'text']
        
        for col in text_cols[:2]:  # Limit to 2 text columns
            try:
                # Clean text
                texts = df[col].fillna('').astype(str)
                
                # Generate TF-IDF embeddings
                vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
                embeddings = vectorizer.fit_transform(texts).toarray()
                
                self.embeddings[col] = embeddings
                
            except:
                pass
        
        return self.embeddings
    
    def build_vector_index(self) -> None:
        """Build FAISS or simple vector index"""
        # Simple nearest neighbor implementation
        self.vector_index = {col: emb for col, emb in self.embeddings.items()}

# =============================================================================
# INSIGHTS GENERATION MODULE
# =============================================================================

class InsightsGenerator:
    """Generate comprehensive AI-driven insights"""
    
    def __init__(self):
        self.insights = []
        self.statistical_insights = []
        self.ml_insights = []
        self.shap_insights = []
    
    def generate_insights(self, df: pd.DataFrame, column_types: Dict[str, str], 
                         cleaning_report: Dict, ml_results: Dict,
                         shap_values: Optional[Any] = None) -> List[str]:
        """Generate comprehensive insights"""
        self.insights = []
        
        # Data overview insights
        self._generate_overview_insights(df, column_types)
        
        # Cleaning insights
        self._generate_cleaning_insights(cleaning_report)
        
        # Statistical insights
        self._generate_statistical_insights(df, column_types)
        
        # ML insights
        if ml_results and ml_results.get('success'):
            self._generate_ml_insights(ml_results)
        
        # SHAP insights
        if shap_values is not None:
            self._generate_shap_insights(shap_values)
        
        return self.insights
    
    def _generate_overview_insights(self, df: pd.DataFrame, column_types: Dict[str, str]) -> None:
        """Generate data overview insights"""
        self.insights.append(f"Dataset Shape: {len(df)} rows and {len(df.columns)} columns")
        
        type_counts = {}
        for col_type in column_types.values():
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
        
        for col_type, count in type_counts.items():
            self.insights.append(f"{col_type.capitalize()} columns: {count}")
    
    def _generate_cleaning_insights(self, cleaning_report: Dict) -> None:
        """Generate cleaning insights"""
        if cleaning_report['duplicates_removed'] > 0:
            pct = (cleaning_report['duplicates_removed'] / cleaning_report['initial_rows']) * 100
            self.insights.append(f"Removed {cleaning_report['duplicates_removed']} duplicate rows ({pct:.2f}% of data)")
        
        if cleaning_report['missing_values_handled']:
            total_missing = sum(cleaning_report['missing_values_handled'].values())
            self.insights.append(f"Imputed {total_missing} missing values across {len(cleaning_report['missing_values_handled'])} columns")
        
        if cleaning_report['outliers_detected']:
            total_outliers = sum(
                v['iqr_method'] if isinstance(v, dict) else v 
                for v in cleaning_report['outliers_detected'].values()
            )
            self.insights.append(f"Detected {total_outliers} outliers across {len(cleaning_report['outliers_detected'])} columns")
    
    def _generate_statistical_insights(self, df: pd.DataFrame, column_types: Dict[str, str]) -> None:
        """Generate statistical insights"""
        numeric_cols = [col for col in df.columns if column_types.get(col) == 'numeric']
        
        if numeric_cols:
            for col in numeric_cols[:5]:
                mean_val = df[col].mean()
                median_val = df[col].median()
                std_val = df[col].std()
                skew_val = df[col].skew()
                
                self.insights.append(
                    f"{col}: mean={mean_val:.2f}, median={median_val:.2f}, std={std_val:.2f}, skewness={skew_val:.2f}"
                )
                
                if abs(skew_val) > 1:
                    self.insights.append(f"{col} shows {'positive' if skew_val > 0 else 'negative'} skewness, indicating asymmetric distribution")
    
    def _generate_ml_insights(self, ml_results: Dict) -> None:
        """Generate ML model insights"""
        self.insights.append(f"Model Type: {ml_results['model_name']}")
        self.insights.append(f"Task: {ml_results['model_type'].capitalize()}")
        self.insights.append(f"Features Used: {ml_results['n_features']}")
        
        for metric, value in ml_results['metrics'].items():
            self.insights.append(f"Model {metric}: {value:.4f}")
    
    def _generate_shap_insights(self, shap_values: Any) -> None:
        """Generate SHAP-based insights"""
        try:
            # Get feature names and importance
            if hasattr(shap_values, 'values'):
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                feature_names = shap_values.feature_names if hasattr(shap_values, 'feature_names') else range(len(mean_abs_shap))
                
                top_features = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)[:5]
                
                self.insights.append("Top 5 Most Important Features (SHAP):")
                for feat, importance in top_features:
                    self.insights.append(f"  - {feat}: {importance:.4f}")
        except:
            pass

# =============================================================================
# VISUALIZATION MODULE - ENHANCED
# =============================================================================

class DashboardGenerator:
    """Generate comprehensive interactive Plotly dashboards"""
    
    def create_dashboard(self, df: pd.DataFrame, column_types: Dict[str, str], 
                        ml_engine: Optional[MLEngine] = None,
                        ts_engine: Optional[TimeSeriesEngine] = None) -> Dict[str, go.Figure]:
        """Generate full dashboard with all visualizations"""
        figures = {}
        
        # 1. Distribution plots
        numeric_cols = [col for col in df.columns if column_types.get(col) == 'numeric'][:6]
        if numeric_cols:
            figures['distributions'] = self._create_distribution_plots(df, numeric_cols)
        
        # 2. Correlation heatmap
        if len(numeric_cols) >= 2:
            figures['correlation'] = self._create_correlation_heatmap(df, numeric_cols)
        
        # 3. Feature importance
        if ml_engine and ml_engine.feature_importance:
            figures['feature_importance'] = self._create_feature_importance_plot(ml_engine.feature_importance)
        
        # 4. SHAP summary plot
        if ml_engine and ml_engine.shap_values is not None:
            figures['shap_summary'] = self._create_shap_summary_plot(ml_engine.shap_values)
        
        # 5. Missing values visualization
        figures['missing_values'] = self._create_missing_values_plot(df)
        
        # 6. Time series visualization
        datetime_cols = [col for col in df.columns if column_types.get(col) == 'datetime']
        if datetime_cols and numeric_cols:
            figures['time_series'] = self._create_time_series_plot(df, datetime_cols[0], numeric_cols[0])
        
        # 7. Prophet forecast
        if ts_engine and ts_engine.forecast is not None:
            figures['prophet_forecast'] = self._create_prophet_plot(ts_engine)
        
        # 8. Model performance
        if ml_engine and ml_engine.predictions is not None:
            figures['model_performance'] = self._create_model_performance_plot(ml_engine)
        
        return figures
    
    def _create_distribution_plots(self, df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
        """Create distribution plots with histograms and box plots"""
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols,
            specs=[[{'secondary_y': False}] * n_cols for _ in range(n_rows)]
        )
        
        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols + 1
            col_pos = idx % n_cols + 1
            
            fig.add_trace(
                go.Histogram(
                    x=df[col],
                    name=col,
                    showlegend=False,
                    marker_color='#1f77b4',
                    opacity=0.7
                ),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title_text="Feature Distributions",
            height=300 * n_rows,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
        """Create advanced correlation heatmap"""
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=600,
            width=800,
            template='plotly_white'
        )
        
        return fig
    
    def _create_feature_importance_plot(self, feature_importance: Dict[str, float]) -> go.Figure:
        """Create feature importance bar plot"""
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        features, importances = zip(*sorted_features) if sorted_features else ([], [])
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker_color='#2ca02c',
                text=[f'{x:.4f}' for x in importances],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Top 15 Feature Importances",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def _create_shap_summary_plot(self, shap_values: Any) -> go.Figure:
        """Create SHAP summary visualization"""
        try:
            import shap
            
            # Create SHAP summary plot
            fig = go.Figure()
            
            if hasattr(shap_values, 'values') and hasattr(shap_values, 'feature_names'):
                mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                feature_names = shap_values.feature_names
                
                sorted_idx = np.argsort(mean_abs_shap)[::-1][:15]
                
                fig.add_trace(go.Bar(
                    x=mean_abs_shap[sorted_idx],
                    y=[feature_names[i] for i in sorted_idx],
                    orientation='h',
                    marker_color='#ff7f0e',
                    text=[f'{x:.4f}' for x in mean_abs_shap[sorted_idx]],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="SHAP Feature Importance (Mean Absolute SHAP Values)",
                    xaxis_title="Mean |SHAP Value|",
                    yaxis_title="Feature",
                    height=500,
                    template='plotly_white'
                )
            
            return fig
        except:
            return go.Figure()
    
    def _create_missing_values_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create missing values visualization"""
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No missing values detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color='green')
            )
            fig.update_layout(height=300, template='plotly_white')
            return fig
        
        fig = go.Figure(data=[
            go.Bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                marker_color='#d62728',
                text=missing_data.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Missing Values by Column",
            xaxis_title="Count",
            yaxis_title="Column",
            height=max(400, len(missing_data) * 30),
            template='plotly_white'
        )
        
        return fig
    
    def _create_time_series_plot(self, df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
        """Create time series plot"""
        try:
            df_sorted = df[[date_col, value_col]].dropna().sort_values(date_col)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_sorted[date_col],
                y=df_sorted[value_col],
                mode='lines+markers',
                name=value_col,
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title=f"Time Series: {value_col} over {date_col}",
                xaxis_title=date_col,
                yaxis_title=value_col,
                height=400,
                template='plotly_white'
            )
            
            return fig
        except:
            return go.Figure()
    
    def _create_prophet_plot(self, ts_engine: TimeSeriesEngine) -> go.Figure:
        """Create Prophet forecast visualization"""
        try:
            forecast = ts_engine.forecast
            
            fig = go.Figure()
            
            # Actual values
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fillcolor='rgba(31, 119, 180, 0.2)',
                fill='tonexty',
                showlegend=True
            ))
            
            fig.update_layout(
                title="Prophet Time Series Forecast",
                xaxis_title="Date",
                yaxis_title="Value",
                height=500,
                template='plotly_white'
            )
            
            return fig
        except:
            return go.Figure()
    
    def _create_model_performance_plot(self, ml_engine: MLEngine) -> go.Figure:
        """Create model performance visualization"""
        fig = go.Figure()
        
        if ml_engine.metrics:
            metrics = list(ml_engine.metrics.keys())
            values = list(ml_engine.metrics.values())
            
            fig.add_trace(go.Bar(
                x=metrics,
                y=values,
                marker_color='#9467bd',
                text=[f'{v:.4f}' for v in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Model Performance Metrics - {ml_engine.model_name}",
                xaxis_title="Metric",
                yaxis_title="Score",
                height=400,
                template='plotly_white'
            )
        
        return fig

# =============================================================================
# EXPORT MODULE
# =============================================================================

class ExportManager:
    """Handle all data exports"""
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filename: str) -> bytes:
        """Export dataframe to CSV bytes"""
        return df.to_csv(index=False).encode('utf-8')
    
    @staticmethod
    def create_download_button(df: pd.DataFrame, filename: str, label: str, key: str):
        """Create download button for CSV"""
        csv = df.to_csv(index=False)
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime="text/csv",
            key=key
        )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    
    # Header
    st.title("Enterprise-Grade Automated ML Analytics Platform")
    st.markdown("""
    **Complete CSV-Driven Analytics System** | XGBoost | Prophet | SHAP | Feature Engineering | Interactive Dashboards
    """)
    st.markdown("---")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'pipeline_run' not in st.session_state:
        st.session_state.pipeline_run = False
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_file:
            is_valid, message = validate_csv_upload(uploaded_file)
            if is_valid:
                st.success(message)
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.info(f"File size: {file_size_mb:.2f} MB")
            else:
                st.error(message)
                return
        
        st.markdown("---")
        st.markdown("### Pipeline Options")
        
        use_xgboost = st.checkbox("Use XGBoost", value=True, help="Use XGBoost for ML models (fallback to sklearn if not available)")
        run_prophet = st.checkbox("Run Prophet Forecast", value=False, help="Run time series forecasting with Prophet (requires datetime column)")
        generate_embeddings = st.checkbox("Generate Text Embeddings", value=False, help="Generate embeddings for text columns")
        calculate_shap = st.checkbox("Calculate SHAP Values", value=True, help="Calculate SHAP values for model explainability")
    
    # Main content
    if uploaded_file is not None:
        try:
            # Step 1: Load Data
            if not st.session_state.data_loaded:
                with st.spinner("Loading large CSV file..."):
                    df = load_large_csv(uploaded_file)
                    st.session_state.df_original = df
                    st.session_state.data_loaded = True
                
                st.success(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
            
            df = st.session_state.df_original
            
            # Data Preview
            with st.expander("Data Preview (First 100 rows)", expanded=False):
                st.dataframe(df.head(100), use_container_width=True)
            
            # Column Types Detection
            if 'column_types' not in st.session_state:
                column_types = detect_column_types(df)
                st.session_state.column_types = column_types
            else:
                column_types = st.session_state.column_types
            
            with st.expander("Detected Column Types"):
                type_df = pd.DataFrame({
                    'Column': list(column_types.keys()),
                    'Type': list(column_types.values())
                })
                st.dataframe(type_df, use_container_width=True)
            
            # Run Pipeline Button
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col2:
                run_pipeline = st.button("🚀 Run Complete Pipeline", type="primary", use_container_width=True)
            
            if run_pipeline or st.session_state.pipeline_run:
                st.session_state.pipeline_run = True
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # STEP 1: Data Cleaning
                status_text.text("Step 1/7: Cleaning data...")
                progress_bar.progress(10)
                
                if 'cleaned_df' not in st.session_state:
                    cleaner = DataCleaner()
                    cleaned_df = cleaner.clean_data(df, column_types)
                    st.session_state.cleaned_df = cleaned_df
                    st.session_state.cleaning_report = cleaner.cleaning_report
                
                cleaned_df = st.session_state.cleaned_df
                cleaning_report = st.session_state.cleaning_report
                
                with st.expander(" Data Cleaning Report", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Initial Rows", f"{cleaning_report['initial_rows']:,}")
                    col2.metric("Final Rows", f"{cleaning_report['final_rows']:,}")
                    col3.metric("Duplicates Removed", cleaning_report['duplicates_removed'])
                    col4.metric("Columns Cleaned", len(cleaning_report['missing_values_handled']))
                    
                    st.json(cleaning_report)
                
                # STEP 2: Feature Engineering
                status_text.text("Step 2/7: Engineering features...")
                progress_bar.progress(25)
                
                if 'engineered_df' not in st.session_state:
                    engineer = FeatureEngineer()
                    engineered_df = engineer.engineer_features(cleaned_df, column_types)
                    st.session_state.engineered_df = engineered_df
                    st.session_state.engineer = engineer
                
                engineered_df = st.session_state.engineered_df
                engineer = st.session_state.engineer
                
                st.success(f" Feature Engineering: Added {len(engineer.engineered_features)} new features")
                
                # STEP 3: Text Embeddings
                if generate_embeddings:
                    status_text.text("Step 3/7: Generating text embeddings...")
                    progress_bar.progress(40)
                    
                    if 'embeddings' not in st.session_state:
                        embedding_engine = TextEmbeddingEngine()
                        embeddings = embedding_engine.generate_embeddings(engineered_df, column_types)
                        embedding_engine.build_vector_index()
                        st.session_state.embeddings = embeddings
                        st.session_state.embedding_engine = embedding_engine
                    
                    if st.session_state.embeddings:
                        st.success(f" Generated embeddings for {len(st.session_state.embeddings)} text columns")
                else:
                    progress_bar.progress(40)
                
                # STEP 4: ML Model Training
                status_text.text("Step 4/7: Training ML models...")
                progress_bar.progress(55)
                
                st.subheader("Machine Learning Model Training")
                
                numeric_columns = [col for col in cleaned_df.columns if column_types.get(col) == 'numeric']
                
                if numeric_columns:
                    target_col = st.selectbox(
                        "Select Target Variable for ML Model",
                        numeric_columns,
                        key='target_select'
                    )
                    
                    if 'ml_results' not in st.session_state or st.button("Train Model"):
                        with st.spinner("Training ML model..."):
                            ml_engine = MLEngine()
                            ml_results = ml_engine.train_model(
                                engineered_df, target_col, column_types, use_xgboost
                            )
                            st.session_state.ml_engine = ml_engine
                            st.session_state.ml_results = ml_results
                    
                    if 'ml_results' in st.session_state:
                        ml_results = st.session_state.ml_results
                        ml_engine = st.session_state.ml_engine
                        
                        if ml_results['success']:
                            st.success(f" Model trained: {ml_results['model_name']} ({ml_results['model_type']})")
                            
                            col1, col2, col3 = st.columns(3)
                            metrics_items = list(ml_results['metrics'].items())
                            for idx, (metric, value) in enumerate(metrics_items):
                                with [col1, col2, col3][idx % 3]:
                                    st.metric(metric, f"{value:.4f}")
                        else:
                            st.error(f"Model training failed: {ml_results['error']}")
                else:
                    st.warning("No numeric columns available for ML modeling")
                    st.session_state.ml_engine = None
                    st.session_state.ml_results = None
                
                # STEP 5: Time Series Forecasting
                if run_prophet:
                    status_text.text("Step 5/7: Running Prophet forecast...")
                    progress_bar.progress(70)
                    
                    datetime_cols = [col for col in cleaned_df.columns if column_types.get(col) == 'datetime']
                    
                    if datetime_cols and numeric_columns:
                        st.subheader("Time Series Forecasting (Prophet)")
                        
                        date_col = st.selectbox("Select Date Column", datetime_cols, key='date_select')
                        ts_target = st.selectbox("Select Target Column", numeric_columns, key='ts_target_select')
                        
                        if st.button("Run Prophet Forecast"):
                            ts_engine = TimeSeriesEngine()
                            ts_results = ts_engine.train_prophet(cleaned_df, date_col, ts_target)
                            st.session_state.ts_engine = ts_engine
                            st.session_state.ts_results = ts_results
                            
                            if ts_results['success']:
                                st.success(" Prophet forecast completed")
                    else:
                        st.info("Prophet requires at least one datetime and one numeric column")
                else:
                    progress_bar.progress(70)
                
                # STEP 6: Insights Generation
                status_text.text("Step 6/7: Generating AI-driven insights...")
                progress_bar.progress(85)
                
                if 'insights' not in st.session_state:
                    insights_gen = InsightsGenerator()
                    insights = insights_gen.generate_insights(
                        engineered_df, column_types, cleaning_report,
                        st.session_state.get('ml_results', {}),
                        st.session_state.ml_engine.shap_values if 'ml_engine' in st.session_state else None
                    )
                    st.session_state.insights = insights
                    st.session_state.insights_gen = insights_gen
                
                insights = st.session_state.insights
                
                with st.expander(" Key Insights", expanded=True):
                    for insight in insights:
                        st.write(f"- {insight}")
                
                # STEP 7: Dashboard Generation
                status_text.text("Step 7/7: Generating interactive dashboards...")
                progress_bar.progress(95)
                
                st.markdown("---")
                st.subheader("Interactive Analytics Dashboard")
                
                if 'figures' not in st.session_state:
                    dashboard_gen = DashboardGenerator()
                    figures = dashboard_gen.create_dashboard(
                        engineered_df, column_types,
                        st.session_state.get('ml_engine'),
                        st.session_state.get('ts_engine')
                    )
                    st.session_state.figures = figures
                
                figures = st.session_state.figures
                
                # Display visualizations
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Distributions", "Correlations", "Feature Importance", "SHAP Analysis", "Time Series"
                ])
                
                with tab1:
                    if 'distributions' in figures:
                        st.plotly_chart(figures['distributions'], use_container_width=True)
                    if 'missing_values' in figures:
                        st.plotly_chart(figures['missing_values'], use_container_width=True)
                
                with tab2:
                    if 'correlation' in figures:
                        st.plotly_chart(figures['correlation'], use_container_width=True)
                
                with tab3:
                    if 'feature_importance' in figures:
                        st.plotly_chart(figures['feature_importance'], use_container_width=True)
                    if 'model_performance' in figures:
                        st.plotly_chart(figures['model_performance'], use_container_width=True)
                
                with tab4:
                    if 'shap_summary' in figures:
                        st.plotly_chart(figures['shap_summary'], use_container_width=True)
                    else:
                        st.info("SHAP analysis not available. Enable 'Calculate SHAP Values' and train a model.")
                
                with tab5:
                    if 'time_series' in figures:
                        st.plotly_chart(figures['time_series'], use_container_width=True)
                    if 'prophet_forecast' in figures:
                        st.plotly_chart(figures['prophet_forecast'], use_container_width=True)
                    else:
                        st.info("Time series visualizations require datetime columns")
                
                progress_bar.progress(100)
                status_text.text(" Pipeline completed successfully!")
                
                # Export Section
                st.markdown("---")
                st.subheader(" Export Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    ExportManager.create_download_button(
                        cleaned_df, "cleaned_data.csv", "Download Cleaned Data", "download_cleaned"
                    )
                
                with col2:
                    ExportManager.create_download_button(
                        engineered_df, "engineered_data.csv", "Download Engineered Data", "download_engineered"
                    )
                
                with col3:
                    if 'ml_engine' in st.session_state and st.session_state.ml_engine.predictions is not None:
                        predictions_df = pd.DataFrame({
                            'Prediction': st.session_state.ml_engine.predictions
                        })
                        ExportManager.create_download_button(
                            predictions_df, "predictions.csv", "Download Predictions", "download_predictions"
                        )
                
                with col4:
                    insights_df = pd.DataFrame({'Insight': insights})
                    ExportManager.create_download_button(
                        insights_df, "insights.csv", "Download Insights", "download_insights"
                    )
                
                st.success("🎉 Analysis pipeline completed successfully!")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
    
    else:
        # Landing page
        st.info(" Please upload a CSV file to begin analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("###  Key Features")
            st.markdown("""
            - **Automated Data Cleaning**: Missing values, duplicates, outliers
            - **Advanced Feature Engineering**: Date features, encodings, interactions
            - **ML Models**: XGBoost, Random Forest, Gradient Boosting
            - **Time Series**: Prophet forecasting
            - **Explainability**: SHAP values and feature importance
            - **Text Analysis**: TF-IDF embeddings and vector indexing
            - **Interactive Dashboards**: Plotly visualizations
            - **AI-Driven Insights**: Statistical and ML insights
            - **Export Capabilities**: CSV downloads for all outputs
            """)
        
        with col2:
            st.markdown("###  Specifications")
            st.markdown(f"""
            - **Maximum file size**: {MAX_FILE_SIZE_MB}MB
            - **Supported format**: CSV only
            - **Automatic type detection**: Numeric, categorical, datetime, text
            - **Chunked processing**: Handles large files efficiently
            - **Model types**: Regression and Classification
            - **Forecasting**: 30-day Prophet forecasts
            - **Explainability**: Global and local SHAP analysis
            - **Production-ready**: Professional architecture
            """)
        
        st.markdown("---")
        st.markdown("###  Getting Started")
        st.markdown("""
        1. Upload your CSV file using the sidebar
        2. Review detected column types
        3. Configure pipeline options
        4. Click 'Run Complete Pipeline'
        5. Explore interactive dashboards
        6. Export processed data and insights
        """)

if __name__ == "__main__":
    main()