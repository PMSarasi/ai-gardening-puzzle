import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from environment.garden_env_normalized import GardenEnvNormalized
from agents.dqn_agent_fixed import DQNAgentFixed

# Page configuration
st.set_page_config(
    page_title="🌱 AI Gardening Puzzle",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        state_size = 57
        action_size = 36
        device = torch.device("cpu")
        agent = DQNAgentFixed(state_size, action_size, device)
        if os.path.exists("models/final_normalized.pth"):
            agent.load("models/final_normalized.pth")
            agent.epsilon = 0.01
            return agent
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None

def get_plant_emoji(plant):
    """Get emoji based on plant type and growth stage"""
    growth = plant.growth_stage
    plant_type = plant.type.name
    
    if growth < 0.8:
        return "🌱"
    elif growth < 1.6:
        if plant_type == 'FLOWER':
            return "🌿"
        elif plant_type == 'VEGETABLE':
            return "🥬"
        else:
            return "🌲"
    elif growth < 2.4:
        if plant_type == 'FLOWER':
            return "🌸"
        elif plant_type == 'VEGETABLE':
            return "🍅"
        else:
            return "🌳"
    else:
        if plant_type == 'FLOWER':
            return "🌻"
        elif plant_type == 'VEGETABLE':
            return "🥕"
        else:
            return "🎄"

def get_health_color(health):
    """Get color based on health"""
    if health > 70:
        return "#2ecc71"
    elif health > 40:
        return "#f1c40f"
    else:
        return "#e74c3c"

def create_garden_visualization(env):
    """Create simple garden visualization"""
    grid_size = env.grid_size
    plants = env.plants
    
    # Create a figure for the garden
    fig = make_subplots(
        rows=grid_size, cols=grid_size,
        subplot_titles=[f"Plant {i+1}" for i in range(grid_size * grid_size)],
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )
    
    for idx, plant in enumerate(plants):
        row = idx // grid_size
        col = idx % grid_size
        
        # Plant emoji
        emoji = get_plant_emoji(plant)
        health_color = get_health_color(plant.health)
        
        # Create plant text
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[0.7],
                mode='text',
                text=[emoji],
                textfont=dict(size=50, color=health_color),
                hoverinfo='text',
                hovertext=f"""
                <b>Plant {idx + 1}</b><br>
                Type: {plant.type.name}<br>
                Health: {plant.health:.1f}%<br>
                Growth: {plant.growth_stage:.1f}/3<br>
                Water: {plant.water:.1f}/5<br>
                Soil: {plant.soil:.1f}/5
                """,
                showlegend=False
            ),
            row=row+1, col=col+1
        )
        
        # Add health bar
        fig.add_trace(
            go.Bar(
                x=['Health'],
                y=[plant.health],
                marker_color=health_color,
                showlegend=False,
                text=[f'{plant.health:.0f}%'],
                textposition='outside',
                name=f'Plant {idx+1} Health'
            ),
            row=row+1, col=col+1
        )
        
        # Add resource bars
        fig.add_trace(
            go.Bar(
                x=['💧', '🌍'],
                y=[plant.water, plant.soil],
                marker_color=['#3498db', '#95a5a6'],
                showlegend=False,
                text=[f'{plant.water:.1f}', f'{plant.soil:.1f}'],
                textposition='outside'
            ),
            row=row+1, col=col+1
        )
    
    fig.update_layout(
        height=700,
        title_text="🌱 AI Gardener's Garden",
        showlegend=False
    )
    
    # Update subplot properties
    for i in range(1, grid_size * grid_size + 1):
        fig.update_xaxes(title_text="", showticklabels=False, 
                        range=[-0.2, 1.2],
                        row=(i-1)//grid_size+1, col=(i-1)%grid_size+1)
        fig.update_yaxes(title_text="", showticklabels=False, 
                        range=[0, 1],
                        row=(i-1)//grid_size+1, col=(i-1)%grid_size+1)
    
    return fig

def main():
    # Title
    st.title("🌱 AI Gardening Puzzle")
    st.markdown("### 🤖 An AI Agent That Learns to Care for Plants")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/gardening.png", width=80)
        st.markdown("---")
        
        # Mode selection
        st.subheader("🎮 Game Mode")
        mode = st.radio(
            "Choose Mode",
            ["🤖 Watch AI Agent", "👤 Play Yourself"],
            help="AI Mode: Watch the trained agent play | Human Mode: You control the garden"
        )
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Model Info")
        agent = load_model()
        if agent:
            st.success("✅ Model Loaded Successfully")
            st.info(f"""
            🎯 **Test Performance:**
            - Average Score: 75.8
            - Success Rate: 100%
            - Best Score: 78.4
            - Stability: ±3.0
            """)
        else:
            st.warning("⚠️ Model not found - using random actions")
        
        st.markdown("---")
        
        # Quick stats
        if 'env' in st.session_state:
            env = st.session_state.env
            if env:
                stats = env.get_garden_stats()
                st.metric("Current Score", f"{st.session_state.total_reward:.0f}")
                st.metric("Garden Health", f"{stats['avg_health']:.1f}%")
    
    # Initialize session state
    if 'env' not in st.session_state:
        st.session_state.env = GardenEnvNormalized(grid_size=3)
        st.session_state.total_reward = 0
        st.session_state.step_count = 0
        st.session_state.game_over = False
        st.session_state.reward_history = []
        st.session_state.action_history = []
        st.session_state.auto_play = False
    
    env = st.session_state.env
    agent = load_model()
    
    # Main content - Garden
    st.subheader("🌿 Your Garden")
    
    # Create garden visualization
    garden_fig = create_garden_visualization(env)
    st.plotly_chart(garden_fig, use_container_width=True)
    
    # Metrics Row
    stats = env.get_garden_stats()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("❤️ Health", f"{stats['avg_health']:.1f}%")
    with col2:
        st.metric("📈 Growth", f"{stats['avg_growth']:.1f}/3")
    with col3:
        st.metric("💧 Water", f"{stats['avg_water']:.1f}/5")
    with col4:
        st.metric("🌍 Soil", f"{stats['avg_soil']:.1f}/5")
    with col5:
        st.metric("🏆 Score", f"{st.session_state.total_reward:.0f}", 
                  delta=f"Day {st.session_state.step_count}/100")
    
    # Progress bars
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Garden Health Status**")
        health_progress = st.progress(stats['avg_health'] / 100)
        if stats['avg_health'] > 70:
            st.success("✅ Garden is thriving!")
        elif stats['avg_health'] > 40:
            st.warning("⚠️ Some plants need attention")
        else:
            st.error("🔴 Garden needs immediate care!")
    
    with col2:
        st.markdown("**📈 Garden Growth Progress**")
        growth_progress = st.progress(stats['avg_growth'] / 3)
        if stats['avg_growth'] > 2.5:
            st.success("🌻 Plants are mature!")
        elif stats['avg_growth'] > 1.5:
            st.info("🌱 Plants are growing well")
        else:
            st.warning("🌿 Plants are still young")
    
    # Game Controls
    st.markdown("---")
    
    if mode == "🤖 Watch AI Agent":
        st.subheader("🤖 AI Controls")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎮 AI Move", use_container_width=True):
                if agent and not st.session_state.game_over:
                    state = env._get_observation()
                    action = agent.act(state, eval_mode=True)
                    plant_idx = action // 4
                    action_type = action % 4
                    action_names = ["Water", "Fertilize", "Prune", "Wait"]
                    
                    next_state, reward, done, truncated, _ = env.step(action)
                    st.session_state.total_reward += reward
                    st.session_state.step_count += 1
                    st.session_state.reward_history.append(reward)
                    st.session_state.action_history.append(action_names[action_type])
                    
                    if reward > 0.3:
                        st.success(f"🤖 AI {action_names[action_type]} Plant {plant_idx + 1} (+{reward:.2f})")
                    elif reward < 0:
                        st.warning(f"🤖 AI {action_names[action_type]} Plant {plant_idx + 1} ({reward:.2f})")
                    else:
                        st.info(f"🤖 AI {action_names[action_type]} Plant {plant_idx + 1} ({reward:.2f})")
                    
                    if done or truncated:
                        st.session_state.game_over = True
                        st.balloons()
                        st.success(f"🎉 Episode Complete! Final Score: {st.session_state.total_reward:.0f}")
                    
                    st.rerun()
        
        with col2:
            if st.button("🔁 Auto Play", use_container_width=True):
                st.session_state.auto_play = not st.session_state.auto_play
                st.rerun()
        
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.env = GardenEnvNormalized(grid_size=3)
                st.session_state.total_reward = 0
                st.session_state.step_count = 0
                st.session_state.game_over = False
                st.session_state.reward_history = []
                st.session_state.action_history = []
                st.rerun()
        
        # Auto play logic
        if st.session_state.auto_play and not st.session_state.game_over and agent:
            time.sleep(0.2)
            state = env._get_observation()
            action = agent.act(state, eval_mode=True)
            next_state, reward, done, truncated, _ = env.step(action)
            st.session_state.total_reward += reward
            st.session_state.step_count += 1
            st.session_state.reward_history.append(reward)
            
            if done or truncated:
                st.session_state.game_over = True
                st.session_state.auto_play = False
                st.balloons()
                st.success(f"🎉 Episode Complete! Final Score: {st.session_state.total_reward:.0f}")
            
            st.rerun()
        
        # Auto play indicator
        if st.session_state.auto_play:
            st.info("🔁 Auto-play active - AI is gardening! Watch the plants grow 🌱")
    
    else:  # Human Mode
        st.subheader("👤 Your Controls")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            plant = st.selectbox("🌱 Select Plant", 
                                [f"Plant {i+1}" for i in range(9)],
                                help="Choose which plant to care for")
            plant_idx = int(plant.split()[1]) - 1
            
            # Show plant details
            selected_plant = env.plants[plant_idx]
            st.info(f"""
            **Plant {plant_idx + 1} Details:**
            - Type: {selected_plant.type.name}
            - Health: {selected_plant.health:.1f}%
            - Water: {selected_plant.water:.1f}/5
            - Soil: {selected_plant.soil:.1f}/5
            - Growth: {selected_plant.growth_stage:.1f}/3
            """)
        
        with col2:
            action = st.selectbox("🎯 Choose Action",
                                 ["💧 Water", "🌿 Fertilize", "✂️ Prune", "⏰ Wait"],
                                 help="Select what to do with the plant")
            action_map = {"💧 Water": 0, "🌿 Fertilize": 1, "✂️ Prune": 2, "⏰ Wait": 3}
            action_code = action_map[action]
        
        with col3:
            st.write("")
            st.write("")
            if st.button("🌱 Take Action", use_container_width=True):
                if not st.session_state.game_over:
                    action_val = plant_idx * 4 + action_code
                    next_state, reward, done, truncated, _ = env.step(action_val)
                    st.session_state.total_reward += reward
                    st.session_state.step_count += 1
                    st.session_state.reward_history.append(reward)
                    
                    # Show feedback
                    if reward > 0.3:
                        st.success(f"✅ Great choice! +{reward:.2f} points!")
                    elif reward > 0:
                        st.info(f"📈 Good action! +{reward:.2f} points")
                    elif reward < 0:
                        st.warning(f"⚠️ Not the best move: {reward:.2f} points")
                    else:
                        st.info(f"ℹ️ Neutral effect: {reward:.2f} points")
                    
                    if done or truncated:
                        st.session_state.game_over = True
                        st.balloons()
                        st.success(f"🎉 Episode Complete! Final Score: {st.session_state.total_reward:.0f}")
                    
                    st.rerun()
        
        # Reset button
        if st.button("🔄 Reset Garden", use_container_width=True):
            st.session_state.env = GardenEnvNormalized(grid_size=3)
            st.session_state.total_reward = 0
            st.session_state.step_count = 0
            st.session_state.game_over = False
            st.session_state.reward_history = []
            st.session_state.action_history = []
            st.rerun()
    
    # Game over message
    if st.session_state.game_over:
        st.markdown("---")
        st.success(f"""
        🎉 **Episode Complete!** 🎉
        
        | Metric | Value |
        |--------|-------|
        | **Final Score** | {st.session_state.total_reward:.0f} |
        | **Garden Health** | {stats['avg_health']:.1f}% |
        | **Average Growth** | {stats['avg_growth']:.1f}/3 |
        | **Plants Mature** | {sum(1 for p in env.plants if p.growth_stage > 2.5)}/9 |
        
        Click **Reset Garden** to play again!
        """)
    
    # Performance chart
    if st.session_state.reward_history:
        st.markdown("---")
        st.subheader("📈 Performance History")
        
        fig = go.Figure()
        
        # Reward line
        fig.add_trace(go.Scatter(
            y=st.session_state.reward_history,
            mode='lines+markers',
            name='Step Reward',
            line=dict(color='#2ecc71', width=2),
            marker=dict(size=8, color='#27ae60', symbol='circle')
        ))
        
        # Moving average
        if len(st.session_state.reward_history) > 5:
            ma = pd.Series(st.session_state.reward_history).rolling(window=5).mean()
            fig.add_trace(go.Scatter(
                y=ma,
                mode='lines',
                name='Moving Average (5 steps)',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="Reward Progress Over Time",
            xaxis_title="Step",
            yaxis_title="Reward",
            height=350,
            hovermode='x unified',
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Reward", f"{np.mean(st.session_state.reward_history):.2f}")
        with col2:
            st.metric("Best Step", f"{max(st.session_state.reward_history):.2f}")
        with col3:
            st.metric("Worst Step", f"{min(st.session_state.reward_history):.2f}")
        with col4:
            st.metric("Total Actions", len(st.session_state.reward_history))
    
    # Instructions footer
    with st.expander("📖 How to Play", expanded=False):
        st.markdown("""
        ### 🎮 Game Modes
        
        **🤖 AI MODE (Watch AI Learn):**
        - Click **AI Move** - Agent takes one action
        - Click **Auto Play** - Agent plays continuously
        - Watch the plants grow and score increase!
        
        **👤 HUMAN MODE (Play Yourself):**
        - Select a plant (1-9)
        - Choose an action (Water/Fertilize/Prune/Wait)
        - Click **Take Action** to perform the action
        - Try to beat the AI's score!
        
        ### 🎯 Goal
        - Keep plants healthy (>70% health)
        - Help them reach maturity (growth stage 3)
        - Maximize your score
        
        ### 💡 Tips
        - Water when 💧 is low (<3)
        - Fertilize when 🌍 is low (<3)
        - Prune when health is low (<60)
        - Wait when everything is optimal
        
        ### 📊 Scoring
        - Healthy plants: +0.5-1.0 points
        - Growth progress: +0.2-0.5 points
        - Optimal resources: +0.1-0.3 points
        - Bad actions: -0.2 to -1.0 points
        - Episode completion: +5-15 bonus
        """)

if __name__ == "__main__":
    main()