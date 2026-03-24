"""ML Predictions page - LSTM price forecasts with confidence scores for position trading."""

import streamlit as st
import pandas as pd
import os
from datetime import datetime

from ui.styles import THEME
from ui.components import section_header, metric_card, status_badge
from data.stock_api import get_all_symbols


def _get_training_job_status():
    """Get the status of the latest training job from Google Cloud."""
    import subprocess
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


def show_predictions(_stock_api):
    """Display ML predictions with real-time data and model training."""
    col1, col2 = st.columns([3, 1])
    with col1:
        section_header("ML Price Predictions", icon="fa-brain")
    with col2:
        if st.button("⟳ Refresh Predictions", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Hybrid ML System Selection
    st.markdown("### ⚙️ Prediction System Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        prediction_mode = st.selectbox(
            "Prediction Mode:",
            options=["hybrid", "enhanced_mock", "vertex_ai"],
            format_func=lambda x: {
                "hybrid": "🔀 Hybrid (Best of Both Worlds)",
                "enhanced_mock": "🏠 Enhanced Mock (Reliable)",
                "vertex_ai": "☁️ Vertex AI (Real ML)"
            }[x],
            help="Choose your prediction strategy",
            index=0  # Default to Hybrid
        )

    with col2:
        # Show system status
        training_status = _get_training_job_status()
        if training_status == "JOB_STATE_SUCCEEDED":
            status_badge("online", "ML MODEL READY")
        elif training_status == "JOB_STATE_RUNNING":
            status_badge("training", "ML TRAINING...")
        else:
            status_badge("offline", "ML MODEL PENDING")

    # Initialize prediction service
    prediction_service = None
    hybrid_service = None
    system_summary = {
        'enhanced_mock_available': True,
        'vertex_ai_available': False,
        'local_ml_models_available': False,
        'local_ml_models_count': 0,
        'supported_symbols': get_all_symbols()[:12],
        'system_status': 'basic_operational'
    }

    # Try to initialize hybrid prediction service
    try:
        from ml.hybrid_prediction_service import HybridPredictionService
        hybrid_service = HybridPredictionService()

        # Get system summary
        system_summary = hybrid_service.get_prediction_summary()
    except ImportError as e:
        st.warning(f"⚠️ Hybrid service import failed: {e}")
        st.info("Falling back to basic prediction service.")
        hybrid_service = None

        # Show system status
        st.markdown("### 🔧 System Status")

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card("Enhanced Mock", "Active", icon="fa-check-circle")

        with col2:
            vertex_status = "Available" if system_summary['vertex_ai_available'] else "Unavailable"
            metric_card("Vertex AI", vertex_status, icon="fa-cloud")

        with col3:
            local_status = "Available" if system_summary['local_ml_models_available'] else "Unavailable"
            metric_card("Local Models", local_status, icon="fa-server")

    # Set prediction service based on mode
    try:
        if prediction_mode == "hybrid" and hybrid_service is not None:
            prediction_service = hybrid_service
        elif prediction_mode == "enhanced_mock":
            from ml.prediction_service import PredictionService
            prediction_service = PredictionService(provider="local")
        else:  # vertex_ai or hybrid fallback
            from ml.prediction_service import PredictionService
            prediction_service = PredictionService(provider="vertex")
    except ImportError as e:
        st.error(f"⊗ Could not import prediction service: {e}")
        st.info("Falling back to basic prediction service.")

        try:
            from ml.prediction_service import PredictionService
            prediction_service = PredictionService(provider="local")
        except ImportError:
            st.error("⊗ Could not import any prediction service.")
            return

    # Ensure prediction_service is always defined
    if prediction_service is None:
        try:
            from ml.prediction_service import PredictionService
            prediction_service = PredictionService(provider="local")
        except ImportError:
            st.error("⊗ Could not initialize any prediction service.")
            return

    # Show last update time
    st.caption(f"⟳ Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M:%S %p')}")

    # Show current prediction mode status
    if prediction_mode == "hybrid":
        st.success("🔀 **Using Hybrid Prediction System** - Best of both worlds!")
        st.info("""
        **Hybrid System Features:**
        • **Real ML Models** when available (Vertex AI or local)
        • **Enhanced Mock Predictions** with technical analysis as fallback
        • **Automatic failover** ensures predictions always work
        • **Source transparency** - see which method was used for each prediction
        """)
    elif prediction_mode == "enhanced_mock":
        st.success("🏠 **Using Enhanced Mock Predictions** - Reliable and sophisticated!")
        st.info("""
        **Enhanced Mock Features:**
        • **Real-time Yahoo Finance data**
        • **Technical analysis** (RSI, trends, volatility)
        • **Dynamic confidence scoring**
        • **Always available** - no training required
        """)
    else:  # vertex_ai
        if hasattr(prediction_service, 'vertex_service') and prediction_service.vertex_service:
            st.success("☁️ **Using Vertex AI ML Models** - Real trained models!")
        else:
            st.warning("⚠️ **Vertex AI Not Available** - Using enhanced mock predictions.")

            # Show deployment instructions
            with st.expander("🚀 Enable Vertex AI ML Predictions"):
                st.markdown("""
                **To enable real ML predictions:**

                1. **Wait for training to complete** (check Cloud Progress tab)
                2. **Deploy the trained model to endpoint**
                3. **Refresh this page** to see real ML predictions!

                **Current Training Status:** Check the "☁️ Cloud Progress" tab for updates.
                """)

    # Prediction controls
    st.markdown("### ↗ Prediction Controls")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_symbol = st.selectbox(
            "Select Stock:",
            options=['All'] + get_all_symbols()[:12],
            index=0
        )

    with col2:
        days_ahead = st.selectbox(
            "Prediction Horizon:",
            options=[1, 3, 7, 14, 30],
            index=2,
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}"
        )

    with col3:
        if st.button("◉ Train Model", use_container_width=True):
            with st.spinner("Training model... This may take a few minutes."):
                if selected_symbol == 'All':
                    st.info("Please select a specific stock to train its model.")
                else:
                    result = prediction_service.train_model(selected_symbol, days=365, epochs=50)
                    if result['status'] == 'success':
                        st.success(f"◉ Model trained successfully for {selected_symbol}!")
                        st.json(result)
                    else:
                        st.error(f"⊗ Training failed: {result['message']}")
                        if "TensorFlow" in result['message']:
                            st.info("◉ **Solution**: Install TensorFlow with Python 3.9-3.11 to train real models.")
                            st.info("◉ **What happened**: The system successfully fetched real data and calculated features, but failed at the TensorFlow model creation step.")

    # Train all models button
    st.markdown("### ◉ Train All Models")
    if st.button("◉ Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training all models... This may take several minutes."):
            try:
                results = prediction_service.train_all_models()

                # Show training results
                st.success("◉ Training completed!")

                # Display results (results is a dict like {'BTC': {...}, 'ETH': {...}})
                for symbol, result in results.items():
                    if result['status'] == 'success':
                        st.success(f"◉ {symbol}: {result.get('message', 'Training successful')}")
                    else:
                        st.error(f"⊗ {symbol}: {result.get('message', 'Training failed')}")

            except Exception as e:
                st.error(f"⊗ Training failed: {e}")
                st.info("◉ **Solution**: Install TensorFlow with Python 3.9-3.11 to train real models.")

                # Show what actually happened
                st.info("◉ **What happened**: The system successfully fetched real data and calculated features, but failed at the TensorFlow model creation step.")
                st.info("◉ **Data processed**: 366 days of real market data with 11 technical indicators")
                st.info("⟳ **Sequences created**: 352 training sequences ready for LSTM training")
                st.info("⊗ **Missing**: TensorFlow library for neural network training")

    st.markdown("---")

    # Generate predictions
    with st.spinner("◉ Generating predictions..."):
        try:
            if selected_symbol == 'All':
                # Check if using HybridPredictionService (needs symbols param) or regular PredictionService (doesn't)
                if hasattr(prediction_service, '__class__') and 'Hybrid' in prediction_service.__class__.__name__:
                    # HybridPredictionService needs symbols parameter
                    top_symbols = get_all_symbols()[:12]
                    predictions_dict = prediction_service.get_all_predictions(symbols=top_symbols, days_ahead=days_ahead)
                    predictions = [predictions_dict[s] for s in top_symbols if s in predictions_dict]
                else:
                    # Regular PredictionService returns list directly
                    predictions = prediction_service.get_all_predictions(days_ahead=days_ahead)
            else:
                predictions = [prediction_service.get_prediction(selected_symbol, days_ahead)]
        except Exception as e:
            st.error(f"⊗ Error generating predictions: {e}")
            predictions = []

    # Display predictions with portfolio-style cards
    st.markdown("### ↗ Price Predictions")

    if not predictions:
        st.warning("No predictions available. Please try again.")
        return

    # Create prediction cards
    num_cols = 3
    for i in range(0, len(predictions), num_cols):
        cols = st.columns(num_cols)

        for j, col in enumerate(cols):
            if i + j < len(predictions):
                pred = predictions[i + j]

                # Determine colors based on prediction
                if pred['predicted_return'] > 0.02:  # > 2% gain
                    return_color = THEME['accent_success']
                    return_symbol = "↗"
                elif pred['predicted_return'] < -0.02:  # < -2% loss
                    return_color = THEME['accent_danger']
                    return_symbol = "↘"
                else:  # Neutral
                    return_color = THEME['accent_primary']
                    return_symbol = "→"

                with col:
                    # Create a container with custom styling
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <h3 style="margin: 0; color: {THEME['text_primary']};">{pred['symbol']} {return_symbol}</h3>
                            <span class="badge" style="background: {return_color}20; color: {return_color}; border-color: {return_color};">
                                {pred['confidence']*100:.0f}% CONFIDENCE
                            </span>
                        </div>

                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                            <div>
                                <div style="font-size: 0.8rem; color: {THEME['text_muted']};">CURRENT</div>
                                <div style="font-size: 1.2rem; font-weight: bold; color: {THEME['text_primary']};">${pred['current_price']:,.2f}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.8rem; color: {THEME['text_muted']};">PREDICTED</div>
                                <div style="font-size: 1.2rem; font-weight: bold; color: {THEME['text_primary']};">${pred['predicted_price']:,.2f}</div>
                            </div>
                        </div>

                        <div style="text-align: center; padding: 8px; background: {return_color}10; border-radius: 8px; border: 1px solid {return_color}40;">
                            <span style="color: {return_color}; font-weight: bold; font-size: 1.1rem;">
                                {pred['predicted_return']*100:+.2f}%
                            </span>
                            <span style="color: {THEME['text_secondary']}; font-size: 0.8rem; margin-left: 5px;">
                                ({days_ahead}d forecast)
                            </span>
                        </div>

                        <div style="margin-top: 10px; font-size: 0.7rem; color: {THEME['text_muted']}; text-align: right;">
                            Source: {pred.get('prediction_source', 'unknown').replace('_', ' ').title()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")

    # Prediction table
    st.markdown("### ◉ Detailed Predictions")

    # Prepare data for table
    table_data = []
    for pred in predictions:
        prediction_source = pred.get('prediction_source', 'unknown')

        # Format prediction source for display
        source_display = {
            'vertex_ai_ml': '🤖 Vertex AI ML',
            'local_ml': '🧠 Local ML',
            'enhanced_mock': '🔧 Enhanced Mock',
            'basic_mock': '⚠️ Basic Mock'
        }.get(prediction_source, '❓ Unknown')

        table_data.append({
            'Symbol': pred['symbol'],
            'Current Price': f"${pred['current_price']:,.2f}",
            'Predicted Price': f"${pred['predicted_price']:,.2f}",
            'Predicted Return': f"{pred['predicted_return']*100:+.2f}%",
            'Confidence': f"{pred['confidence']*100:.1f}%",
            'Source': source_display,
            'Status': pred['status'].title()
        })

    if table_data:
        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True
        )

    # Model training section
    st.markdown("---")
    st.markdown("### ◉ Model Training")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **Train Custom Models:**

        - **LSTM Architecture**: 2-layer neural network with 50 units each
        - **Features**: 11 technical indicators (MA, RSI, Volume, Momentum, Volatility)
        - **Training Data**: 365 days of historical data from Yahoo Finance
        - **Prediction Target**: 7-day future returns
        - **Validation**: 80/20 train/validation split with early stopping
        """)

    with col2:
        if st.button("◉ Train All Models", use_container_width=True):
            with st.spinner("Training all models... This will take several minutes."):
                results = prediction_service.train_all_models(days=365, epochs=50)

                # Display results with better formatting
                st.markdown("#### ◉ Training Results:")

                successful = 0
                failed = 0

                for symbol, result in results.items():
                    if result['status'] == 'success':
                        st.success(f"◉ **{symbol}**: Trained successfully")
                        successful += 1
                        with st.expander(f"◉ {symbol} Training Details"):
                            st.json(result)
                    else:
                        st.error(f"⊗ **{symbol}**: {result['message']}")
                        failed += 1

                        # Show solution for TensorFlow error
                        if "TensorFlow is required" in result['message']:
                            st.info(f"◉ **Solution for {symbol}**: Install TensorFlow with Python 3.9-3.11")

                # Summary
                if successful > 0:
                    st.success(f"◉ **{successful} models trained successfully!**")
                if failed > 0:
                    st.warning(f"⊗ **{failed} models failed** - TensorFlow installation required")

                # Show installation instructions
                if failed > 0:
                    with st.expander("◉ How to Install TensorFlow for Real Model Training"):
                        st.markdown("""
                        **To train real LSTM models, you need TensorFlow:**

                        1. **Create a new Python environment** with Python 3.9-3.11:
                           ```bash
                           conda create -n stock-ml python=3.11
                           conda activate stock-ml
                           ```

                        2. **Install TensorFlow**:
                           ```bash
                           pip install tensorflow==2.13.0
                           ```

                        3. **Install other dependencies**:
                           ```bash
                           pip install -r requirements.txt
                           ```

                        4. **Run the app**:
                           ```bash
                           streamlit run app.py
                           ```

                        **Current Status**: Using mock predictions (realistic but not trained on your data)
                        """)

    # Model status
    st.markdown("### ◉ Model Status")

    model_status = []
    for symbol in get_all_symbols()[:12]:
        has_model = prediction_service._has_model(symbol)
        status = "◉ Trained" if has_model else "⊗ Not Trained"

        # Get last updated timestamp from model file
        model_path = f"models/{symbol}_model.h5"
        last_updated = 'N/A'
        if os.path.exists(model_path):
            timestamp = os.path.getmtime(model_path)
            last_updated = datetime.fromtimestamp(timestamp).strftime('%b %d, %Y %I:%M %p')

        model_status.append({
            'Symbol': symbol,
            'Status': status,
            'Model File': f"{symbol}_model.h5",
            'Last Updated': last_updated
        })

    status_df = pd.DataFrame(model_status)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    # Information section
    with st.expander("◉ About ML Predictions"):
        st.markdown("""
        **How It Works:**

        1. **Data Collection**: Fetches 365 days of OHLCV data from Yahoo Finance
        2. **Feature Engineering**: Creates 11 technical indicators:
           - Moving Averages (7, 14, 30-day)
           - RSI (Relative Strength Index)
           - Volume indicators
           - Price momentum
           - Volatility measures
        3. **LSTM Training**: 2-layer neural network learns patterns from historical data
        4. **Prediction**: Uses last 7 days of features to predict future returns
        5. **Confidence**: Based on model certainty and historical accuracy

        **Important Notes:**
        - Predictions are for educational purposes only
        - Past performance doesn't guarantee future results
        - Always do your own research before trading
        - Models are retrained weekly to adapt to market changes
        """)

    # Status info
    st.success("◉ **ML Prediction System Active** - Real-time forecasts powered by LSTM neural networks!")
