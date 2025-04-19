# visualization-service/app.py

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configuration
API_SERVICE_URL = "http://api-service:8000"
MONITORING_SERVICE_URL = "http://monitoring-service:5000"
LOG_PATH = os.getenv("LOG_PATH", "/app/logs")
MODEL_PATH = "/app/models"

# Page configuration
st.set_page_config(
    page_title="DQN CartPole Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4285F4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #0F9D58;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #F1F3F4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stPlotlyChart {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Functions to load data
@st.cache_data(ttl=60)
def load_training_metrics():
    """Load training metrics from file."""
    try:
        metrics_file = os.path.join(LOG_PATH, "training_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading training metrics: {str(e)}")
        return None

@st.cache_data(ttl=60)
def get_available_models():
    """Get available models from API."""
    try:
        response = requests.get(f"{API_SERVICE_URL}/models", timeout=5)
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []

@st.cache_data(ttl=300)
def get_experiment_runs(experiment_name="DQN-CartPole"):
    """Get runs for an experiment."""
    try:
        # Get experiment ID
        response = requests.get(f"{MONITORING_SERVICE_URL}/experiments", timeout=5)
        if response.status_code != 200:
            return []
        
        experiments = response.json().get("experiments", [])
        experiment_id = None
        for exp in experiments:
            if exp.get("name") == experiment_name:
                experiment_id = exp.get("experiment_id")
                break
        
        if experiment_id is None:
            return []
        
        # Get runs
        response = requests.get(f"{MONITORING_SERVICE_URL}/runs/{experiment_id}", timeout=5)
        if response.status_code == 200:
            return response.json().get("runs", [])
        return []
    except Exception as e:
        st.error(f"Error fetching experiment runs: {str(e)}")
        return []

@st.cache_data(ttl=60)
def evaluate_model(episodes=10):
    """Evaluate model using API."""
    try:
        response = requests.get(f"{API_SERVICE_URL}/evaluate?episodes={episodes}", timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")
        return None

@st.cache_data(ttl=300)
def simulate_episode():
    """Simulate episode using API."""
    try:
        response = requests.get(f"{API_SERVICE_URL}/simulate_episode", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error simulating episode: {str(e)}")
        return None

# Sidebar
st.sidebar.markdown("<h1 class='main-header'>DQN CartPole</h1>", unsafe_allow_html=True)
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Select a page", ["Training Metrics", "Model Evaluation", "Run Comparison", "Agent Simulation"])

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Training Metrics Page
if page == "Training Metrics":
    st.markdown("<h1 class='main-header'>Training Metrics</h1>", unsafe_allow_html=True)
    
    # Load metrics
    metrics = load_training_metrics()
    
    if metrics is None:
        st.info("No training metrics available yet. Wait for the training service to generate data.")
    else:
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Episode Rewards Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=metrics["episodes"], 
                y=metrics["scores"],
                mode='lines',
                name='Episode Score',
                line=dict(color='#4285F4', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=metrics["episodes"], 
                y=metrics["average_scores"],
                mode='lines',
                name='Average Score (last 100)',
                line=dict(color='#0F9D58', width=2)
            ))
            fig.update_layout(
                title='Episode Rewards',
                xaxis_title='Episode',
                yaxis_title='Reward',
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Epsilon and Loss Plot
            fig = go.Figure()
            
            # Create a secondary y-axis for epsilon
            fig.add_trace(go.Scatter(
                x=metrics["episodes"], 
                y=metrics["losses"],
                mode='lines',
                name='Loss',
                line=dict(color='#DB4437', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=metrics["episodes"], 
                y=metrics["epsilons"],
                mode='lines',
                name='Epsilon',
                line=dict(color='#F4B400', width=1),
                yaxis="y2"
            ))
            
            fig.update_layout(
                title='Training Loss and Epsilon',
                xaxis_title='Episode',
                yaxis_title='Loss',
                yaxis2=dict(
                    title='Epsilon',
                    titlefont=dict(color='#F4B400'),
                    tickfont=dict(color='#F4B400'),
                    anchor="x",
                    overlaying="y",
                    side="right"
                ),
                height=400,
                margin=dict(l=20, r=30, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Training Stats
        st.markdown("<h2 class='section-header'>Training Statistics</h2>", unsafe_allow_html=True)
        
        # Create metrics in a row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Episodes Completed", len(metrics["scores"]))
        
        with col2:
            if len(metrics["scores"]) > 0:
                st.metric("Latest Score", metrics["scores"][-1])
        
        with col3:
            if len(metrics["average_scores"]) > 0:
                st.metric("Latest Avg Score (100 ep)", round(metrics["average_scores"][-1], 2))
        
        with col4:
            if len(metrics["epsilons"]) > 0:
                st.metric("Current Epsilon", round(metrics["epsilons"][-1], 4))
        
        # Model Files
        st.markdown("<h2 class='section-header'>Available Models</h2>", unsafe_allow_html=True)
        
        models = get_available_models()
        if models:
            # Convert to DataFrame
            models_df = pd.DataFrame(models)
            models_df['created'] = pd.to_datetime(models_df['created'], unit='s')
            models_df = models_df.sort_values(by='created', ascending=False)
            
            # Format columns
            models_df['created'] = models_df['created'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Display as table
            st.dataframe(models_df, use_container_width=True)
        else:
            st.info("No models available yet.")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.markdown("<h1 class='main-header'>Model Evaluation</h1>", unsafe_allow_html=True)
    
    # Model selection
    models = get_available_models()
    if not models:
        st.info("No models available yet for evaluation.")
    else:
        # Convert models to options
        model_options = {m['name']: m['path'] for m in models}
        selected_model = st.selectbox("Select a model to evaluate", list(model_options.keys()))
        model_path = model_options[selected_model]
        
        # Set active model
        if st.button("Set as Active Model"):
            try:
                response = requests.post(
                    f"{API_SERVICE_URL}/set_model",
                    json={"model_path": model_path},
                    timeout=5
                )
                if response.status_code == 200:
                    st.success(f"Model {selected_model} set as active!")
                else:
                    st.error(f"Failed to set model: {response.json().get('error')}")
            except Exception as e:
                st.error(f"Error setting model: {str(e)}")
        
        # Evaluation parameters
        st.markdown("<h2 class='section-header'>Evaluation Settings</h2>", unsafe_allow_html=True)
        eval_episodes = st.slider("Number of episodes for evaluation", 10, 100, 20)
        
        # Run evaluation
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating model..."):
                evaluation = evaluate_model(episodes=eval_episodes)
                
                if evaluation:
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean Score", round(evaluation["mean_score"], 2))
                    
                    with col2:
                        st.metric("Median Score", round(evaluation["median_score"], 2))
                    
                    with col3:
                        st.metric("Max Score", round(evaluation["max_score"], 2))
                    
                    with col4:
                        st.metric("Min Score", round(evaluation["min_score"], 2))
                    
                    # Score Distribution
                    fig = px.histogram(
                        x=evaluation["scores"],
                        nbins=20,
                        labels={"x": "Score"},
                        title="Score Distribution",
                        color_discrete_sequence=['#4285F4']
                    )
                    fig.add_vline(x=evaluation["mean_score"], line_dash="dash", line_color="#DB4437")
                    fig.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Evaluation failed. Please check if the API service is running properly.")

# Run Comparison Page
elif page == "Run Comparison":
    st.markdown("<h1 class='main-header'>Run Comparison</h1>", unsafe_allow_html=True)
    
    # Get experiment runs
    runs = get_experiment_runs()
    
    if not runs:
        st.info("No experiment runs available yet.")
    else:
        # Display runs in a table
        st.markdown("<h2 class='section-header'>Available Runs</h2>", unsafe_allow_html=True)
        
        # Convert to DataFrame
        runs_df = []
        for run in runs:
            run_data = {
                "run_id": run["run_id"],
                "status": run["status"],
                "start_time": datetime.fromtimestamp(run["start_time"]/1000).strftime('%Y-%m-%d %H:%M:%S'),
                "duration": round((run["end_time"] - run["start_time"])/1000/60, 2) if run["end_time"] else "-"
            }
            
            # Add metrics
            for key, value in run["metrics"].items():
                run_data[f"metric_{key}"] = value
                
            # Add parameters
            for key, value in run["params"].items():
                run_data[f"param_{key}"] = value
                
            runs_df.append(run_data)
        
        runs_df = pd.DataFrame(runs_df)
        
        # Display as table
        st.dataframe(runs_df, use_container_width=True)
        
        # Select runs to compare
        st.markdown("<h2 class='section-header'>Compare Runs</h2>", unsafe_allow_html=True)
        
        # Get run IDs
        run_ids = [run["run_id"] for run in runs]
        selected_runs = st.multiselect("Select runs to compare", run_ids, max_selections=3)
        
        if selected_runs:
            # Prepare data for comparison
            comparison_data = {
                "run_id": [],
                "episode": [],
                "metric": [],
                "value": []
            }
            
            for run_id in selected_runs:
                for run in runs:
                    if run["run_id"] == run_id:
                        # Request detailed metrics
                        try:
                            response = requests.get(f"{MONITORING_SERVICE_URL}/metrics/{run_id}", timeout=10)
                            if response.status_code == 200:
                                metrics_history = response.json().get("metrics_history", {})
                                
                                for metric_name, metric_values in metrics_history.items():
                                    for entry in metric_values:
                                        comparison_data["run_id"].append(run_id[:8])  # Short ID
                                        comparison_data["episode"].append(entry["step"])
                                        comparison_data["metric"].append(metric_name)
                                        comparison_data["value"].append(entry["value"])
                        except Exception as e:
                            st.error(f"Error fetching metrics for run {run_id}: {str(e)}")
            
            if comparison_data["run_id"]:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create plots
                metrics = comparison_df["metric"].unique()
                
                for metric in metrics:
                    metric_df = comparison_df[comparison_df["metric"] == metric]
                    
                    fig = px.line(
                        metric_df,
                        x="episode",
                        y="value",
                        color="run_id",
                        title=f"Comparison of {metric}"
                    )
                    fig.update_layout(
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis_title="Episode",
                        yaxis_title=metric
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No comparison data available for selected runs.")

# Agent Simulation Page
elif page == "Agent Simulation":
    st.markdown("<h1 class='main-header'>Agent Simulation</h1>", unsafe_allow_html=True)
    
    # Run simulation
    if st.button("Run New Simulation"):
        with st.spinner("Simulating episode..."):
            simulation = simulate_episode()
            
            if simulation:
                # Display overall metrics
                st.markdown("<h2 class='section-header'>Simulation Results</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Reward", simulation["total_reward"])
                
                with col2:
                    st.metric("Steps", simulation["steps"])
                
                # Extract trajectory data
                trajectory = simulation["trajectory"]
                df = pd.DataFrame(trajectory)
                
                # Show state evolution
                st.markdown("<h2 class='section-header'>State Evolution</h2>", unsafe_allow_html=True)
                
                # Extract state components
                states = np.array([t["state"] for t in trajectory])
                
                # Cart Position
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.line(
                        x=list(range(len(states))),
                        y=states[:, 0],
                        title="Cart Position",
                        labels={"x": "Step", "y": "Position"}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.line(
                        x=list(range(len(states))),
                        y=states[:, 1],
                        title="Cart Velocity",
                        labels={"x": "Step", "y": "Velocity"}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.line(
                        x=list(range(len(states))),
                        y=states[:, 2],
                        title="Pole Angle",
                        labels={"x": "Step", "y": "Angle (rad)"}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.line(
                        x=list(range(len(states))),
                        y=states[:, 3],
                        title="Pole Angular Velocity",
                        labels={"x": "Step", "y": "Angular Velocity"}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Q-values and Actions
                st.markdown("<h2 class='section-header'>Q-values and Actions</h2>", unsafe_allow_html=True)
                
                # Extract Q-values
                q_values = np.array([t["q_values"] for t in trajectory])
                actions = np.array([t["action"] for t in trajectory])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(q_values))),
                    y=q_values[:, 0],
                    mode='lines',
                    name='Q(s,0) - Move Left',
                    line=dict(color='#DB4437')
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(len(q_values))),
                    y=q_values[:, 1],
                    mode='lines',
                    name='Q(s,1) - Move Right',
                    line=dict(color='#4285F4')
                ))
                
                # Add action markers
                for i, action in enumerate(actions):
                    color = '#DB4437' if action == 0 else '#4285F4'
                    fig.add_trace(go.Scatter(
                        x=[i],
                        y=[q_values[i, action]],
                        mode='markers',
                        marker=dict(color=color, size=10),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title='Q-values Over Time',
                    xaxis_title='Step',
                    yaxis_title='Q-value',
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Action distribution
                action_counts = np.bincount(actions.astype(int))
                action_names = ["Move Left", "Move Right"]
                
                fig = px.bar(
                    x=action_names[:len(action_counts)],
                    y=action_counts,
                    title="Action Distribution",
                    labels={"x": "Action", "y": "Count"},
                    color=action_names[:len(action_counts)],
                    color_discrete_map={"Move Left": "#DB4437", "Move Right": "#4285F4"}
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Simulation failed. Please check if the API service is running properly.")
    else:
        st.info("Click 'Run New Simulation' to simulate an episode with the current model.")