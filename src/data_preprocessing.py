import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def clean_data(self, df):
        df_clean = df.copy()
        
        print("Starting data cleaning...")
        print(f"Original shape: {df_clean.shape}")
        
        df_clean = self._remove_duplicates(df_clean)
        df_clean = self._fix_temperature_anomalies(df_clean)
        df_clean = self._fix_negative_sensors(df_clean)
        df_clean = self._standardize_target(df_clean)
        df_clean = self._normalize_hvac(df_clean)
        df_clean = self._encode_categoricals(df_clean)
        df_clean = self._handle_missing_values(df_clean)
        df_clean = df_clean.drop('Session ID', axis=1)
        
        print(f"Final shape: {df_clean.shape}")
        print("Data cleaning completed!")
        
        return df_clean
    
    def _remove_duplicates(self, df):
        initial_count = len(df)
        df_clean = df.drop_duplicates()
        removed = initial_count - len(df_clean)
        print(f"Removed {removed} duplicate rows")
        return df_clean
    
    def _fix_temperature_anomalies(self, df):
        df_clean = df.copy()
        high_temp_mask = df_clean['Temperature'] > 100
        high_temp_count = high_temp_mask.sum()
        
        if high_temp_count > 0:
            print(f"Fixing {high_temp_count} temperature anomalies (values > 100°C)")
            df_clean.loc[high_temp_mask, 'Temperature'] = df_clean.loc[high_temp_mask, 'Temperature'] / 10
        
        return df_clean
    
    def _fix_negative_sensors(self, df):
        df_clean = df.copy()
        sensor_cols = ['Humidity', 'CO2_InfraredSensor', 'CO2_ElectroChemicalSensor']
        
        for col in sensor_cols:
            if col in df_clean.columns:
                negative_mask = df_clean[col] < 0
                negative_count = negative_mask.sum()
                
                if negative_count > 0:
                    print(f"Fixing {negative_count} negative values in {col}")
                    df_clean.loc[negative_mask, col] = np.nan
        
        return df_clean
    
    def _standardize_target(self, df):
        df_clean = df.copy()
        target_mapping = {
            'Low Activity': 'Low Activity',
            'LowActivity': 'Low Activity', 
            'Low_Activity': 'Low Activity',
            'Moderate Activity': 'Moderate Activity',
            'ModerateActivity': 'Moderate Activity',
            'High Activity': 'High Activity'
        }
        
        df_clean['Activity Level'] = df_clean['Activity Level'].map(target_mapping)
        print(f"Standardized target categories: {df_clean['Activity Level'].unique()}")
        
        return df_clean
    
    def _normalize_hvac(self, df):
        df_clean = df.copy()
        
        df_clean['HVAC Operation Mode'] = (df_clean['HVAC Operation Mode']
                                         .str.lower()
                                         .str.replace('_', ' ')
                                         .str.strip())
        
        return df_clean
    
    def _encode_categoricals(self, df):
        df_clean = df.copy()
        
        ordinal_mappings = {
            'Time of Day': {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3},
            'CO_GasSensor': {'extremely low': 0, 'low': 1, 'medium': 2, 'high': 3, 'extremely high': 4},
            'Ambient Light Level': {'very_dim': 0, 'moderate': 1, 'bright': 2, 'very_bright': 3}
        }
        
        for col, mapping in ordinal_mappings.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map(mapping)
        
        if 'HVAC Operation Mode' in df_clean.columns:
            hvac_dummies = pd.get_dummies(df_clean['HVAC Operation Mode'], prefix='HVAC', dummy_na=True)
            df_clean = pd.concat([df_clean, hvac_dummies], axis=1)
            df_clean = df_clean.drop('HVAC Operation Mode', axis=1)
        
        return df_clean
    
    def _handle_missing_values(self, df):
        df_clean = df.copy()
        
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Activity Level' and df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else 'unknown'
                df_clean[col] = df_clean[col].fillna(mode_val)
        
        return df_clean
        
    def fit_transform(self, df):
        df_clean = self.clean_data(df)
        self.fitted = True
        return df_clean
    
    def transform(self, df):
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        return self.clean_data(df)

def load_and_clean_data(db_path='data/gas_monitoring.db'):
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM gas_monitoring", conn)
    conn.close()
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.fit_transform(df)
    
    return df_clean, preprocessor