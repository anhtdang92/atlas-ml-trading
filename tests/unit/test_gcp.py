#!/usr/bin/env python3
"""
Test Google Cloud Platform Connection and Setup

Verifies:
- GCP authentication
- BigQuery access
- Cloud Storage access
- Creates initial tables
"""

import sys
from datetime import datetime

try:
    from google.cloud import bigquery
    from google.cloud import storage
except ImportError:
    print("Error: Google Cloud libraries not found.")
    print("Install them with: pip install google-cloud-bigquery google-cloud-storage")
    sys.exit(1)


def test_bigquery():
    """Test BigQuery connection and create tables."""
    print("\n" + "=" * 60)
    print("Testing BigQuery Connection...")
    print("=" * 60)

    try:
        client = bigquery.Client(project='stock-ml-trading-487')

        query = "SELECT 1 as test"
        result = client.query(query).result()

        print("BigQuery connection successful!")
        print(f"   Project: {client.project}")

        # Create historical_prices table
        print("\nCreating BigQuery tables...")

        table_id = "stock-ml-trading-487.stock_data.historical_prices"

        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("open", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("high", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("low", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("close", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("volume", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("data_source", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="NULLABLE"),
        ]

        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp",
        )

        table = client.create_table(table, exists_ok=True)
        print(f"   Table created: {table_id}")

        # Create predictions table
        pred_table_id = "stock-ml-trading-487.stock_data.predictions"
        pred_schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("current_price", "FLOAT64"),
            bigquery.SchemaField("predicted_price", "FLOAT64"),
            bigquery.SchemaField("predicted_return", "FLOAT64"),
            bigquery.SchemaField("confidence", "FLOAT64"),
            bigquery.SchemaField("model_version", "STRING"),
            bigquery.SchemaField("prediction_source", "STRING"),
        ]

        pred_table = bigquery.Table(pred_table_id, schema=pred_schema)
        pred_table = client.create_table(pred_table, exists_ok=True)
        print(f"   Table created: {pred_table_id}")

        # Create trades table
        trades_table_id = "stock-ml-trading-487.stock_data.trades"
        trades_schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("action", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("quantity", "FLOAT64"),
            bigquery.SchemaField("price", "FLOAT64"),
            bigquery.SchemaField("total_value", "FLOAT64"),
            bigquery.SchemaField("fee", "FLOAT64"),
            bigquery.SchemaField("paper_trading", "BOOLEAN"),
        ]

        trades_table = bigquery.Table(trades_table_id, schema=trades_schema)
        trades_table = client.create_table(trades_table, exists_ok=True)
        print(f"   Table created: {trades_table_id}")

        return True

    except Exception as e:
        print(f"BigQuery test failed: {e}")
        print("\nTo fix:")
        print("1. Set GOOGLE_APPLICATION_CREDENTIALS to your service account key")
        print("2. Ensure the service account has BigQuery access")
        return False


def test_cloud_storage():
    """Test Cloud Storage connection."""
    print("\n" + "=" * 60)
    print("Testing Cloud Storage Connection...")
    print("=" * 60)

    try:
        client = storage.Client(project='stock-ml-trading-487')

        bucket_name = "stock-ml-models-487"

        bucket = client.bucket(bucket_name)
        if bucket.exists():
            print(f"Bucket exists: {bucket_name}")
        else:
            print(f"Bucket not found: {bucket_name}")
            print("Creating bucket...")
            bucket = client.create_bucket(bucket_name, location="US")
            print(f"Bucket created: {bucket_name}")

        # Test upload
        test_blob = bucket.blob("test/connection_test.txt")
        test_blob.upload_from_string(f"Connection test at {datetime.now()}")
        print(f"Test file uploaded to {bucket_name}/test/connection_test.txt")

        # Clean up
        test_blob.delete()
        print("Test file cleaned up")

        return True

    except Exception as e:
        print(f"Cloud Storage test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("GCP SERVICES TEST SUITE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: stock-ml-trading-487\n")

    results = []
    results.append(("BigQuery", test_bigquery()))
    results.append(("Cloud Storage", test_cloud_storage()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"   {name:<25} {status}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\n   Total: {passed_count}/{len(results)} tests passed")

    if passed_count == len(results):
        print("\nAll GCP services working!")
    else:
        print("\nSome GCP tests failed. Check authentication and permissions.")


if __name__ == "__main__":
    main()
