#!/usr/bin/env python3
"""
Vertex AI Training Job for LSTM Models
This script runs on Vertex AI to train LSTM models for stock price prediction
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import aiplatform

# Add the project root to the path
sys.path.append('/app')

from ml.historical_data_fetcher import HistoricalDataFetcher
from ml.feature_engineering import FeatureEngineer
from ml.lstm_model import StockLSTM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VertexTrainingJob:
    """Vertex AI training job for LSTM models"""
    
    def __init__(self, args):
        self.project_id = args.project_id
        self.region = args.region
        self.bucket_name = args.bucket_name
        self.dataset_id = args.dataset_id
        self.symbols = args.symbols.split(',')
        self.lookback_days = args.lookback_days
        self.prediction_horizon = args.prediction_horizon
        
        # Initialize clients
        self.bq_client = bigquery.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)
        
        # Initialize ML components
        self.data_fetcher = HistoricalDataFetcher()
        self.feature_engineer = FeatureEngineer()
        
        logger.info(f"◈ Initialized training job for symbols: {self.symbols}")
    
    def fetch_training_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch training data from BigQuery"""
        logger.info("◉ Fetching training data from BigQuery...")
        
        data = {}
        for symbol in self.symbols:
            query = f"""
            SELECT 
                timestamp,
                open, high, low, close, volume
            FROM `{self.project_id}.{self.dataset_id}.historical_prices`
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
            """
            
            df = self.bq_client.query(query).to_dataframe()
            if len(df) > 0:
                data[symbol] = df
                logger.info(f"◊ Fetched {len(df)} records for {symbol}")
            else:
                logger.warning(f"◊ No data found for {symbol}")
        
        return data
    
    def prepare_training_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare training data with features and labels"""
        logger.info("◉ Preparing training data...")
        
        training_data = {}
        
        for symbol, df in data.items():
            if len(df) < self.lookback_days + self.prediction_horizon:
                logger.warning(f"◊ Insufficient data for {symbol}, skipping...")
                continue
            
            # Engineer features
            df_with_features = self.feature_engineer.add_all_features(df)
            
            # Create sequences
            X, y = self.feature_engineer.create_sequences(
                df_with_features, 
                lookback_days=self.lookback_days,
                prediction_horizon=self.prediction_horizon
            )
            
            if len(X) > 0:
                training_data[symbol] = (X, y)
                logger.info(f"◊ Prepared {len(X)} sequences for {symbol}")
            else:
                logger.warning(f"◊ No sequences created for {symbol}")
        
        return training_data
    
    def train_models(self, training_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, str]:
        """Train LSTM models for each symbol"""
        logger.info("◉ Training LSTM models...")
        
        model_paths = {}
        
        for symbol, (X, y) in training_data.items():
            logger.info(f"◊ Training model for {symbol}...")
            
            # Initialize model
            model = StockLSTM(
                input_shape=(X.shape[1], X.shape[2]),
                lstm_units=50,
                lstm_layers=2,
                dropout=0.2
            )
            
            # Train model
            history = model.train(
                X, y,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                early_stopping_patience=10
            )
            
            # Save model
            model_filename = f"lstm_model_{symbol}.h5"
            model_path = f"/app/outputs/models/{model_filename}"
            model.save_model(model_path)
            
            # Upload to Cloud Storage
            bucket = self.storage_client.bucket(self.bucket_name)
            blob_name = f"lstm-models/{symbol}/{model_filename}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(model_path)
            
            model_paths[symbol] = f"gs://{self.bucket_name}/{blob_name}"
            
            # Log metrics
            self.log_training_metrics(symbol, history, len(X))
            
            logger.info(f"◊ Model trained and saved for {symbol}")
        
        return model_paths
    
    def log_training_metrics(self, symbol: str, history, num_samples: int):
        """Log training metrics to BigQuery"""
        logger.info(f"◊ Logging metrics for {symbol}...")
        
        # Extract metrics from history
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        train_mae = history.history['mae'][-1]
        val_mae = history.history['val_mae'][-1]
        
        # Calculate R² (simplified)
        train_r2 = 1 - (train_loss / np.var(history.history['loss']))
        val_r2 = 1 - (val_loss / np.var(history.history['val_loss']))
        
        # Insert metrics into BigQuery
        metrics_data = [{
            'timestamp': datetime.now(),
            'symbol': symbol,
            'model_version': f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'training_rmse': np.sqrt(train_loss),
            'validation_rmse': np.sqrt(val_loss),
            'training_mae': train_mae,
            'validation_mae': val_mae,
            'training_r2': train_r2,
            'validation_r2': val_r2,
            'training_samples': int(num_samples * 0.8),
            'validation_samples': int(num_samples * 0.2),
            'created_at': datetime.now()
        }]
        
        table_id = f"{self.project_id}.{self.dataset_id}.model_metrics"
        self.bq_client.insert_rows_json(table_id, metrics_data)
        
        logger.info(f"◊ Metrics logged for {symbol}")
    
    def run(self):
        """Run the complete training pipeline"""
        logger.info("◈ Starting Vertex AI training job...")
        
        try:
            # Fetch data
            data = self.fetch_training_data()
            
            if not data:
                logger.error("◊ No training data available")
                return
            
            # Prepare data
            training_data = self.prepare_training_data(data)
            
            if not training_data:
                logger.error("◊ No training data prepared")
                return
            
            # Train models
            model_paths = self.train_models(training_data)
            
            # Save model paths
            with open('/app/outputs/logs/model_paths.json', 'w') as f:
                json.dump(model_paths, f, indent=2)
            
            logger.info("◈ Training job completed successfully!")
            logger.info(f"◊ Models saved to: {list(model_paths.values())}")
            
        except Exception as e:
            logger.error(f"◊ Training job failed: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Vertex AI LSTM Training Job')
    parser.add_argument('--project_id', required=True, help='GCP Project ID')
    parser.add_argument('--region', required=True, help='GCP Region')
    parser.add_argument('--bucket_name', required=True, help='Cloud Storage bucket name')
    parser.add_argument('--dataset_id', required=True, help='BigQuery dataset ID')
    parser.add_argument('--symbols', required=True, help='Comma-separated list of symbols')
    parser.add_argument('--lookback_days', type=int, default=7, help='Lookback days for sequences')
    parser.add_argument('--prediction_horizon', type=int, default=7, help='Prediction horizon in days')
    
    args = parser.parse_args()
    
    # Run training job
    job = VertexTrainingJob(args)
    job.run()

if __name__ == "__main__":
    main()
