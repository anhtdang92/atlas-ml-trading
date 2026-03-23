#!/usr/bin/env python3
"""
Test script for Vertex AI prediction endpoint
"""

import argparse
import json
import logging
import numpy as np
from datetime import datetime

from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_endpoint(project_id: str, region: str, endpoint_id: str):
    """Test the Vertex AI prediction endpoint"""
    
    logger.info(f"Testing endpoint: {endpoint_id}")
    logger.info(f"Project: {project_id}, Region: {region}")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Get endpoint
    endpoint = aiplatform.Endpoint(endpoint_id)
    
    # Create test data (7 days of 11 features)
    test_data = np.random.rand(1, 7, 11).tolist()
    
    logger.info("Making test prediction...")
    
    try:
        # Make prediction
        response = endpoint.predict(
            instances=[test_data],
            parameters={"confidence_threshold": 0.5}
        )
        
        predictions = response.predictions[0]
        
        logger.info("✅ Prediction successful!")
        logger.info(f"Response: {predictions}")
        
        # Test with multiple instances
        batch_data = [test_data] * 3
        batch_response = endpoint.predict(
            instances=batch_data,
            parameters={"confidence_threshold": 0.5}
        )
        
        logger.info("✅ Batch prediction successful!")
        logger.info(f"Batch response count: {len(batch_response.predictions)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return False

def get_endpoint_info(project_id: str, region: str, endpoint_id: str):
    """Get endpoint information"""
    
    aiplatform.init(project=project_id, location=region)
    endpoint = aiplatform.Endpoint(endpoint_id)
    
    info = {
        "endpoint_id": endpoint_id,
        "display_name": endpoint.display_name,
        "state": getattr(endpoint, 'state', 'ACTIVE'),
        "created_at": getattr(endpoint, 'create_time', 'Unknown'),
        "deployed_models": len(endpoint.list_models()),
        "region": region,
        "project": project_id
    }
    
    return info

def main():
    parser = argparse.ArgumentParser(description='Test Vertex AI Endpoint')
    parser.add_argument('--project_id', default='stock-ml-trading-487', help='GCP Project ID')
    parser.add_argument('--region', default='us-central1', help='GCP Region')
    parser.add_argument('--endpoint_id', required=True, help='Vertex AI Endpoint ID')
    parser.add_argument('--info_only', action='store_true', help='Only show endpoint info')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧪 Testing Vertex AI Prediction Endpoint")
    print("="*60)
    
    # Get endpoint info
    try:
        info = get_endpoint_info(args.project_id, args.region, args.endpoint_id)
        
        print(f"\n📊 Endpoint Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        if args.info_only:
            return
        
        # Test endpoint
        print(f"\n🔮 Testing predictions...")
        success = test_endpoint(args.project_id, args.region, args.endpoint_id)
        
        if success:
            print(f"\n✅ Endpoint test successful!")
            print(f"   Ready for production use")
        else:
            print(f"\n❌ Endpoint test failed!")
            print(f"   Check endpoint status and configuration")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    main()
