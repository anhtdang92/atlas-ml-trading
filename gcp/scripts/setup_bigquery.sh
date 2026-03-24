#!/bin/bash

# Set up BigQuery datasets and tables for ML predictions
# This script creates the necessary datasets with proper schemas

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-stock-ml-trading-487}"
DATASET_ID="stock_data"

echo "◈ Setting up BigQuery for project: $PROJECT_ID"

# Set the project
gcloud config set project $PROJECT_ID

# Create dataset
echo "◉ Creating dataset: $DATASET_ID"
bq mk --dataset --location=US $PROJECT_ID:$DATASET_ID

# Create historical prices table
echo "◉ Creating historical_prices table..."
bq mk --table \
    --schema="timestamp:TIMESTAMP,symbol:STRING,open:FLOAT64,high:FLOAT64,low:FLOAT64,close:FLOAT64,volume:FLOAT64,data_source:STRING,created_at:TIMESTAMP" \
    --time_partitioning_field=timestamp \
    --time_partitioning_type=DAY \
    $PROJECT_ID:$DATASET_ID.historical_prices

# Create predictions table
echo "◉ Creating predictions table..."
bq mk --table \
    --schema="timestamp:TIMESTAMP,symbol:STRING,predicted_price:FLOAT64,confidence:FLOAT64,model_version:STRING,created_at:TIMESTAMP" \
    --time_partitioning_field=timestamp \
    --time_partitioning_type=DAY \
    $PROJECT_ID:$DATASET_ID.predictions

# Create trades table
echo "◉ Creating trades table..."
bq mk --table \
    --schema="timestamp:TIMESTAMP,symbol:STRING,side:STRING,quantity:FLOAT64,price:FLOAT64,value:FLOAT64,fee:FLOAT64,order_id:STRING,status:STRING,created_at:TIMESTAMP" \
    --time_partitioning_field=timestamp \
    --time_partitioning_type=DAY \
    $PROJECT_ID:$DATASET_ID.trades

# Create model_metrics table
echo "◉ Creating model_metrics table..."
bq mk --table \
    --schema="timestamp:TIMESTAMP,symbol:STRING,model_version:STRING,training_rmse:FLOAT64,validation_rmse:FLOAT64,training_mae:FLOAT64,validation_mae:FLOAT64,training_r2:FLOAT64,validation_r2:FLOAT64,training_samples:INTEGER,validation_samples:INTEGER,created_at:TIMESTAMP" \
    --time_partitioning_field=timestamp \
    --time_partitioning_type=DAY \
    $PROJECT_ID:$DATASET_ID.model_metrics

# Create portfolio_snapshots table
echo "◉ Creating portfolio_snapshots table..."
bq mk --table \
    --schema="timestamp:TIMESTAMP,symbol:STRING,quantity:FLOAT64,price:FLOAT64,value:FLOAT64,weight:FLOAT64,created_at:TIMESTAMP" \
    --time_partitioning_field=timestamp \
    --time_partitioning_type=DAY \
    $PROJECT_ID:$DATASET_ID.portfolio_snapshots

# Create rebalancing_events table
echo "◉ Creating rebalancing_events table..."
bq mk --table \
    --schema="timestamp:TIMESTAMP,event_type:STRING,trigger:STRING,old_allocation:STRING,new_allocation:STRING,total_trades:INTEGER,total_fees:FLOAT64,confidence_score:FLOAT64,created_at:TIMESTAMP" \
    --time_partitioning_field=timestamp \
    --time_partitioning_type=DAY \
    $PROJECT_ID:$DATASET_ID.rebalancing_events

# Set dataset permissions
echo "◉ Setting dataset permissions..."

# Grant access to service accounts
bq update --source_format=NEWLINE_DELIMITED_JSON \
    --schema="[{\"role\":\"WRITER\",\"userByEmail\":\"ml-training-sa@$PROJECT_ID.iam.gserviceaccount.com\"},{\"role\":\"READER\",\"userByEmail\":\"ml-prediction-sa@$PROJECT_ID.iam.gserviceaccount.com\"},{\"role\":\"READER\",\"userByEmail\":\"stock-app-sa@$PROJECT_ID.iam.gserviceaccount.com\"}]" \
    $PROJECT_ID:$DATASET_ID

# Create views for common queries
echo "◉ Creating views..."

# Latest prices view
bq mk --view="
SELECT 
  symbol,
  close as price,
  volume,
  timestamp
FROM \`$PROJECT_ID.$DATASET_ID.historical_prices\`
WHERE timestamp = (
  SELECT MAX(timestamp) 
  FROM \`$PROJECT_ID.$DATASET_ID.historical_prices\` h2 
  WHERE h2.symbol = historical_prices.symbol
)
" $PROJECT_ID:$DATASET_ID.latest_prices

# Portfolio performance view
bq mk --view="
SELECT 
  DATE(timestamp) as date,
  SUM(value) as total_value,
  COUNT(DISTINCT symbol) as num_holdings,
  AVG(weight) as avg_weight
FROM \`$PROJECT_ID.$DATASET_ID.portfolio_snapshots\`
GROUP BY DATE(timestamp)
ORDER BY date DESC
" $PROJECT_ID:$DATASET_ID.portfolio_performance

# Model performance view
bq mk --view="
SELECT 
  symbol,
  model_version,
  AVG(validation_rmse) as avg_rmse,
  AVG(validation_r2) as avg_r2,
  MAX(timestamp) as last_trained
FROM \`$PROJECT_ID.$DATASET_ID.model_metrics\`
GROUP BY symbol, model_version
ORDER BY symbol, last_trained DESC
" $PROJECT_ID:$DATASET_ID.model_performance

echo "◈ BigQuery setup complete!"
echo "◊ Dataset: $PROJECT_ID:$DATASET_ID"
echo "◊ Tables created:"
echo "  - historical_prices (partitioned by timestamp)"
echo "  - predictions (partitioned by timestamp)"
echo "  - trades (partitioned by timestamp)"
echo "  - model_metrics (partitioned by timestamp)"
echo "  - portfolio_snapshots (partitioned by timestamp)"
echo "  - rebalancing_events (partitioned by timestamp)"
echo "◊ Views created:"
echo "  - latest_prices"
echo "  - portfolio_performance"
echo "  - model_performance"
echo "◊ Next steps:"
echo "  1. Run deploy_vertex_training.sh to deploy training job"
echo "  2. Run deploy_vertex_endpoint.sh to deploy prediction endpoint"
