# simple_main.py
import sys
sys.path.append('.')

import streamlit as st
import torch
import numpy as np
from environment.garden_env import GardenEnv
from agents.dqn_agent import DQNAgent
from visualization.streamlit_app import GardenVisualizer

st.set_page_config(page_title="AI Gardening Puzzle", page_icon="🌱", layout="wide")

st.title("🌱 AI Gardening Puzzle")

# Initialize
env = GardenEnv(grid_size=3)
visualizer = GardenVisualizer()

# Display garden
garden_fig = visualizer.render_garden(env.plants, env.grid_size)
st.plotly_chart(garden_fig, use_container_width=True)

# Show controls
st.sidebar.title("Controls")
mode = st.sidebar.selectbox("Mode", ["View Garden", "Random Actions"])

if mode == "Random Actions":
    if st.button("Take Random Action"):
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        st.success(f"Action taken! Reward: {reward:.2f}")
        st.rerun()

# Show instructions
visualizer.render_instructions()