#!/usr/bin/env python3
"""
Simple progress checker for Google Cloud ML training
Run this to monitor your training job status
"""

import subprocess
import time
import json
from datetime import datetime

def get_training_status():
    """Get the current status of the training job."""
    try:
        result = subprocess.run([
            'gcloud', 'ai', 'custom-jobs', 'describe',
            'projects/64620033647/locations/us-central1/customJobs/3995218109618192384',
            '--format=json'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return {
                'state': data.get('state', 'UNKNOWN'),
                'display_name': data.get('displayName', 'Unknown'),
                'create_time': data.get('createTime', ''),
                'start_time': data.get('startTime', ''),
                'end_time': data.get('endTime', ''),
                'labels': data.get('labels', {})
            }
        else:
            return {'error': result.stderr}
    except Exception as e:
        return {'error': str(e)}

def get_training_logs():
    """Get recent logs from the training job."""
    try:
        result = subprocess.run([
            'gcloud', 'ai', 'custom-jobs', 'stream-logs',
            'projects/64620033647/locations/us-central1/customJobs/3995218109618192384'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0 or result.stdout:
            logs = result.stdout.strip().split('\n')
            non_empty_logs = [log for log in logs if log.strip()]
            return non_empty_logs[-5:] if non_empty_logs else ["No logs available yet"]
        else:
            return [f"Log error: {result.stderr}"]
    except subprocess.TimeoutExpired:
        return ["Waiting for training logs... (Job may still be starting)"]
    except Exception as e:
        return [f"Log error: {str(e)}"]

def main():
    print("🚀 Google Cloud ML Training Progress Monitor")
    print("=" * 50)
    print(f"📅 Started monitoring at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    job_id = "crypto-final-20251006-220955"
    print(f"🔍 Monitoring job: {job_id}")
    print()
    
    while True:
        try:
            # Get status
            status = get_training_status()
            
            if 'error' in status:
                print(f"❌ Error: {status['error']}")
                time.sleep(10)
                continue
            
            # Display status
            state = status['state']
            state_emoji = {
                'JOB_STATE_PENDING': '⏳',
                'JOB_STATE_RUNNING': '🔄',
                'JOB_STATE_SUCCEEDED': '✅',
                'JOB_STATE_FAILED': '❌',
                'JOB_STATE_CANCELLED': '🛑'
            }.get(state, '❓')
            
            print(f"{state_emoji} Status: {state.replace('JOB_STATE_', '').title()}")
            print(f"📝 Job: {status.get('display_name', 'Unknown')}")
            
            if status.get('create_time'):
                print(f"🕐 Created: {status['create_time']}")
            if status.get('start_time'):
                print(f"🚀 Started: {status['start_time']}")
            if status.get('end_time'):
                print(f"🏁 Ended: {status['end_time']}")
            
            # Get and display logs if running
            if state == 'JOB_STATE_RUNNING':
                print("\n📋 Recent Logs:")
                logs = get_training_logs()
                for log in logs[-3:]:  # Last 3 lines
                    if log.strip():
                        print(f"   {log}")
            
            print("\n" + "-" * 50)
            
            # Check if completed
            if state in ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED']:
                print(f"\n🎯 Training job completed with status: {state}")
                if state == 'JOB_STATE_SUCCEEDED':
                    print("✅ Success! Your ML model is ready.")
                    print("🚀 Next steps:")
                    print("   1. Deploy prediction endpoint")
                    print("   2. Test predictions")
                    print("   3. Integrate with trading system")
                break
            
            # Wait before next check
            print("⏱️  Checking again in 30 seconds...")
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
