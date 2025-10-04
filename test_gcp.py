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
    print("❌ Error: Google Cloud libraries not found.")
    print("Install them with: pip install google-cloud-bigquery google-cloud-storage")
    sys.exit(1)


def test_bigquery():
    """Test BigQuery connection and create tables."""
    print("\n" + "="*60)
    print("🔍 Testing BigQuery Connection...")
    print("="*60)
    
    try:
        client = bigquery.Client(project='crypto-ml-trading-487')
        
        # Test query
        query = "SELECT 1 as test"
        result = client.query(query).result()
        
        print("✅ BigQuery connection successful!")
        print(f"   Project: {client.project}")
        print(f"   Location: US")
        
        # Create historical_prices table
        print("\n📊 Creating BigQuery tables...")
        
        table_id = "crypto-ml-trading-487.crypto_data.historical_prices"
        
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("open", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("high", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("low", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("close", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("volume", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("data_source", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        
        # Partition by timestamp for cost optimization
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp"
        )
        
        try:
            table = client.create_table(table)
            print(f"✅ Created table: {table.table_id}")
        except Exception as e:
            if "Already Exists" in str(e):
                print(f"ℹ️  Table already exists: {table_id}")
            else:
                raise
        
        # Create predictions table
        table_id = "crypto-ml-trading-487.crypto_data.predictions"
        
        schema = [
            bigquery.SchemaField("prediction_date", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("predicted_price", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("predicted_return", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("confidence", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("model_version", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="prediction_date"
        )
        
        try:
            table = client.create_table(table)
            print(f"✅ Created table: {table.table_id}")
        except Exception as e:
            if "Already Exists" in str(e):
                print(f"ℹ️  Table already exists: {table_id}")
            else:
                raise
        
        # Create trades table
        table_id = "crypto-ml-trading-487.crypto_data.trades"
        
        schema = [
            bigquery.SchemaField("trade_date", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("action", "STRING", mode="REQUIRED"),  # BUY or SELL
            bigquery.SchemaField("quantity", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("price", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("total_value", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("fees", "FLOAT64", mode="REQUIRED"),
            bigquery.SchemaField("paper_trading", "BOOLEAN", mode="REQUIRED"),
            bigquery.SchemaField("order_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="trade_date"
        )
        
        try:
            table = client.create_table(table)
            print(f"✅ Created table: {table.table_id}")
        except Exception as e:
            if "Already Exists" in str(e):
                print(f"ℹ️  Table already exists: {table_id}")
            else:
                raise
        
        print("\n✅ BigQuery setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ BigQuery error: {e}")
        return False


def test_cloud_storage():
    """Test Cloud Storage connection."""
    print("\n" + "="*60)
    print("🔍 Testing Cloud Storage Connection...")
    print("="*60)
    
    try:
        client = storage.Client(project='crypto-ml-trading-487')
        
        bucket_name = 'crypto-ml-models-487'
        bucket = client.bucket(bucket_name)
        
        # Test by listing (even if empty)
        blobs = list(bucket.list_blobs(max_results=1))
        
        print("✅ Cloud Storage connection successful!")
        print(f"   Bucket: gs://{bucket_name}/")
        print(f"   Location: {bucket.location}")
        
        # Create a test file
        blob = bucket.blob('test.txt')
        blob.upload_from_string(f'GCP connection test - {datetime.now()}')
        print(f"✅ Successfully uploaded test file")
        
        # Delete test file
        blob.delete()
        print(f"✅ Successfully cleaned up test file")
        
        return True
        
    except Exception as e:
        print(f"❌ Cloud Storage error: {e}")
        return False


def main():
    """Run all GCP tests."""
    print("="*60)
    print("🚀 Google Cloud Platform Connection Test")
    print("="*60)
    print(f"Project: crypto-ml-trading-487")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    results.append(("BigQuery", test_bigquery()))
    results.append(("Cloud Storage", test_cloud_storage()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for service, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {service:<20} {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\n   Total: {passed_count}/{total_count} tests passed")
    print("="*60)
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Google Cloud Platform is ready!")
        print("\n📝 Next steps:")
        print("   1. Start collecting historical crypto data")
        print("   2. Build and train ML models")
        print("   3. Deploy dashboard to Cloud Run")
        print("\n💰 Your $50 credit is active and ready to use!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

