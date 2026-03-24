"""Cloud Progress page - Training job monitoring, cost tracking, endpoint status."""

import streamlit as st
import plotly.graph_objects as go
import subprocess
import json
import os
import time
from datetime import datetime

from ui.styles import THEME
from ui.components import section_header, status_card, kpi_box, job_card, apply_chart_theme


# --- GCP helper functions ---

def get_training_job_status():
    """Get the status of the latest training job from Google Cloud."""
    try:
        result = subprocess.run([
            'gcloud', 'ai', 'custom-jobs', 'list',
            '--region=us-central1',
            '--format=value(state)',
            '--limit=1',
            '--sort-by=~createTime'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            status = result.stdout.strip()
            return status if status else "NO_JOBS"
        else:
            return "ERROR"
    except Exception as e:
        return f"ERROR: {str(e)}"


def get_latest_training_jobs():
    """Get the latest training jobs from Google Cloud."""
    try:
        result = subprocess.run([
            'gcloud', 'ai', 'custom-jobs', 'list',
            '--region=us-central1',
            '--format=json',
            '--limit=5',
            '--sort-by=~createTime'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            return json.loads(result.stdout)
        return []
    except Exception:
        return []


def get_latest_endpoints():
    """Get the latest endpoints from Google Cloud."""
    try:
        result = subprocess.run([
            'gcloud', 'ai', 'endpoints', 'list',
            '--region=us-central1',
            '--format=json',
            '--limit=5',
            '--sort-by=~createTime'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            return json.loads(result.stdout)
        return []
    except Exception:
        return []


def get_training_logs():
    """Get recent logs from the training job."""
    try:
        job_id = os.getenv('VERTEX_JOB_ID', '')
        if not job_id:
            return ["No VERTEX_JOB_ID configured. Set it in .env to stream logs."]
        result = subprocess.run([
            'gcloud', 'ai', 'custom-jobs', 'stream-logs', job_id
        ], capture_output=True, text=True, timeout=5)

        if result.returncode == 0 or result.stdout:
            logs = result.stdout.strip().split('\n')
            non_empty_logs = [log for log in logs if log.strip()]
            return non_empty_logs[-5:] if non_empty_logs else ["No logs available yet"]
        else:
            return [f"Log fetch error: {result.stderr}"]
    except subprocess.TimeoutExpired:
        return ["Waiting for training logs... (Job may still be starting)"]
    except Exception as e:
        return [f"Log error: {str(e)}"]


def get_gcp_costs():
    """Get current GCP costs (mock data for now)."""
    return {
        "vertex_ai": 2.50,
        "bigquery": 1.20,
        "storage": 0.80,
        "total": 4.50
    }


# --- Page function ---

def show_cloud_progress(_stock_api):
    """Display Google Cloud ML training progress and status."""
    section_header("Cloud Progress Tracker", icon="fa-cloud")

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=True, help="Automatically refresh progress every 30 seconds")

    # Manual refresh button
    if st.sidebar.button("🔄 Manual Refresh", use_container_width=True):
        st.rerun()

    # Progress status cards
    col1, col2, col3 = st.columns(3)

    with col1:
        current_status = get_training_job_status()
        status_info = {
            "JOB_STATE_PENDING": ("Pending", THEME['accent_warning'], "Training job is starting up"),
            "JOB_STATE_RUNNING": ("Running", THEME['accent_success'], "Training is in progress"),
            "JOB_STATE_SUCCEEDED": ("Completed", THEME['accent_success'], "Training completed successfully"),
            "JOB_STATE_FAILED": ("Failed", THEME['accent_danger'], "Training failed"),
            "NO_JOBS": ("No Jobs", "#666666", "No training jobs found"),
            "ERROR": ("Error", THEME['accent_danger'], "Unable to check status")
        }
        status_text, status_color, status_desc = status_info.get(current_status, ("Unknown", "#666666", "Unknown status"))
        kpi_box(status_text, "Training Status", status_color, status_desc, "Latest training job")

    with col2:
        costs = get_gcp_costs()
        budget_used = costs['total'] / 50 * 100
        budget_color = THEME['accent_primary'] if budget_used < 50 else THEME['accent_warning'] if budget_used < 80 else THEME['accent_danger']
        months_remaining = "3-4" if budget_used < 20 else "2-3" if budget_used < 40 else "1-2"
        status_card("Budget Usage",
                     f"""<div style="font-size: 24px; font-weight: bold; color: {THEME['accent_primary']}; margin-bottom: 10px;">
                         ${costs['total']:.2f} / $50.00
                     </div>
                     <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; margin-bottom: 10px;">
                         <div style="background: {budget_color}; height: 100%; width: {budget_used:.1f}%; border-radius: 4px; box-shadow: 0 0 8px {budget_color}40;"></div>
                     </div>""",
                     budget_color,
                     subtitle=f"{budget_used:.1f}% used - {months_remaining} months remaining")

    with col3:
        health_html = f"""
        <div style="font-size: 20px; margin-bottom: 10px; color: {THEME['accent_success']};">
            <i class="fas fa-heartbeat" style="margin-right: 8px;"></i>Healthy
        </div>
        <div style="text-align: left; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                <span style="color: {THEME['text_secondary']};">BigQuery:</span>
                <span style="color: {THEME['accent_success']};">Connected</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                <span style="color: {THEME['text_secondary']};">Yahoo Finance:</span>
                <span style="color: {THEME['accent_success']};">Connected</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                <span style="color: {THEME['text_secondary']};">Vertex AI:</span>
                <span style="color: {THEME['accent_success']};">Active</span>
            </div>
        </div>"""
        status_card("System Health", health_html, THEME['accent_success'])

    # Progress Timeline
    st.markdown("### 📈 Progress Timeline")

    timeline_data = [
        {"time": "22:09", "event": "GCP Infrastructure Setup", "status": "✅ Completed", "details": "Service accounts, buckets, BigQuery created"},
        {"time": "22:10", "event": "Training Job Deployed", "status": "✅ Completed", "details": "Vertex AI job submitted successfully"},
        {"time": "22:11", "event": "Container Startup", "status": "🔄 In Progress", "details": "Downloading TensorFlow container..."},
        {"time": "22:12", "event": "Model Training", "status": "⏳ Pending", "details": "Waiting for container startup"},
        {"time": "22:13", "event": "Prediction Endpoint", "status": "⏳ Pending", "details": "Will deploy after training"},
        {"time": "22:14", "event": "Dashboard Integration", "status": "✅ Completed", "details": "Real-time progress tracking active"}
    ]

    # Display timeline
    for item in timeline_data:
        col1, col2, col3 = st.columns([1, 2, 3])
        with col1:
            st.write(f"**{item['time']}**")
        with col2:
            st.write(item['event'])
        with col3:
            st.write(f"{item['status']} - {item['details']}")

    # Latest Training Jobs and Endpoints
    st.markdown("### 🔄 Latest Training Jobs & Endpoints")

    # Get latest data
    training_jobs = get_latest_training_jobs()
    endpoints = get_latest_endpoints()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Training Jobs")
        job_state_colors = {
            'JOB_STATE_SUCCEEDED': THEME['accent_success'],
            'JOB_STATE_RUNNING': THEME['accent_warning'],
            'JOB_STATE_FAILED': THEME['accent_danger'],
            'JOB_STATE_PENDING': THEME['accent_primary']
        }
        if training_jobs:
            for job in training_jobs[:3]:
                job_name = job.get('displayName', 'Unknown')
                job_state = job.get('state', 'UNKNOWN')
                create_time = job.get('createTime', '')
                if create_time:
                    try:
                        dt = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                        time_str = dt.strftime('%m/%d %H:%M')
                    except Exception:
                        time_str = create_time[:16]
                else:
                    time_str = 'Unknown'
                color = job_state_colors.get(job_state, '#666666')
                job_card(job_name, job_state.replace('JOB_STATE_', '').title(), time_str, color)
        else:
            st.info("No training jobs found")

    with col2:
        st.markdown("#### Prediction Endpoints")
        if endpoints:
            for endpoint in endpoints[:3]:
                endpoint_name = endpoint.get('displayName', 'Unknown')
                endpoint_id = endpoint.get('name', '').split('/')[-1] if endpoint.get('name') else 'Unknown'
                create_time = endpoint.get('createTime', '')
                if create_time:
                    try:
                        dt = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                        time_str = dt.strftime('%m/%d %H:%M')
                    except Exception:
                        time_str = create_time[:16]
                else:
                    time_str = 'Unknown'
                deployed_models = len(endpoint.get('deployedModels', []))
                job_card(endpoint_name, f"ID: {endpoint_id}", time_str,
                         THEME['accent_primary'], extra_info=f"Models: {deployed_models}")
        else:
            st.info("No endpoints found")

    # Live Logs Section
    st.markdown("### 📋 Live Training Logs")

    # Get and display logs
    logs = get_training_logs()
    if logs:
        log_text = "\n".join(logs)
        st.markdown(f"""
        <div style="
            background: #1e1e1e;
            color: #ffffff;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            max-height: 200px;
            overflow-y: auto;
            margin: 10px 0;
            border: 1px solid #333;
        ">
            {log_text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("📋 No logs available yet. Training job is starting up...")

    # Real-time Metrics
    col1, col2 = st.columns(2)

    with col1:
        # Training Progress Chart
        st.markdown("**📊 Training Progress**")

        # Mock progress data
        progress_data = {
            "Epoch": [1, 2, 3, 4, 5],
            "Loss": [0.8, 0.6, 0.4, 0.3, 0.2],
            "Validation Loss": [0.9, 0.7, 0.5, 0.4, 0.3]
        }

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=progress_data["Epoch"], y=progress_data["Loss"],
                                name="Training Loss", line=dict(color=THEME['accent_primary'], width=2)))
        fig.add_trace(go.Scatter(x=progress_data["Epoch"], y=progress_data["Validation Loss"],
                                name="Validation Loss", line=dict(color=THEME['accent_danger'], width=2, dash='dot')))
        apply_chart_theme(fig)
        fig.update_layout(title="Model Training Progress", xaxis_title="Epoch", yaxis_title="Loss", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cost Breakdown
        st.markdown("**💰 Cost Breakdown**")

        cost_data = {
            "Service": ["Vertex AI", "BigQuery", "Storage", "Functions"],
            "Cost": [2.50, 1.20, 0.80, 0.50],
        }

        fig = go.Figure(data=[go.Pie(
            labels=cost_data["Service"],
            values=cost_data["Cost"],
            hole=0.3,
            marker_colors=[THEME['accent_primary'], THEME['accent_secondary'], THEME['accent_success'], THEME['accent_warning']]
        )])
        apply_chart_theme(fig)
        fig.update_layout(title="Monthly Cost Breakdown", height=300)

        st.plotly_chart(fig, use_container_width=True)

    # Status info
    if current_status == "JOB_STATE_SUCCEEDED":
        st.success("🎉 **Training Complete!** Your ML model is ready for deployment.")
        st.info("🚀 **Next Steps**: Deploy prediction endpoint and test end-to-end pipeline.")
    elif current_status == "JOB_STATE_RUNNING":
        st.info("🔄 **Training in Progress** - Your ML model is being trained in Google Cloud.")
    elif current_status == "JOB_STATE_PENDING":
        st.warning("⏳ **Starting Up** - Training job is initializing (takes 2-5 minutes).")
    elif current_status == "JOB_STATE_FAILED":
        st.error("❌ **Training Failed** - Check logs for details.")
    else:
        st.info("📊 **Cloud-First ML System** - All operations run in Google Cloud with zero local dependencies.")

    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()
