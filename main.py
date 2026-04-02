# main.py - FIXED VERSION
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch
import numpy as np
from environment.garden_env import GardenEnv
from agents.dqn_agent import DQNAgent
from visualization.streamlit_app import GardenVisualizer

# Set page config
st.set_page_config(page_title="AI Gardening Puzzle", page_icon="🌱", layout="wide")

def main():
    st.title("🌱 AI Gardening Puzzle")
    
    # Initialize session state for persistence
    if 'env' not in st.session_state:
        st.session_state.env = GardenEnv(grid_size=3)  # Start with smaller grid for speed
        st.session_state.agent = None
        st.session_state.trained = False
        st.session_state.visualizer = GardenVisualizer()
    
    env = st.session_state.env
    visualizer = st.session_state.visualizer
    
    # Sidebar controls
    st.sidebar.title("🎮 Controls")
    mode = st.sidebar.selectbox(
        "Mode",
        ["Watch AI Agent", "Play Yourself", "Train Agent"],
        help="Choose how to interact with the garden"
    )
    
    # Train Agent Mode
    if mode == "Train Agent":
        st.subheader("🤖 Training the AI Gardener")
        
        episodes = st.sidebar.slider("Training Episodes", 10, 200, 50)
        
        if st.button("🚀 Start Training", type="primary"):
            # Create agent
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            device = torch.device("cpu")
            agent = DQNAgent(state_size, action_size, device)
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            reward_placeholder = st.empty()
            
            rewards_history = []
            
            for episode in range(episodes):
                state, _ = env.reset()
                total_reward = 0
                
                for step in range(env.max_steps):
                    action = agent.act(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay()
                    total_reward += reward
                    state = next_state
                    
                    if done or truncated:
                        break
                
                rewards_history.append(total_reward)
                
                # Update progress
                progress_bar.progress((episode + 1) / episodes)
                status_text.text(f"Episode {episode + 1}/{episodes}")
                reward_placeholder.metric("Latest Reward", f"{total_reward:.2f}")
                
                # Show moving average
                if len(rewards_history) > 10:
                    avg_reward = np.mean(rewards_history[-10:])
                    st.sidebar.metric("Avg Reward (last 10)", f"{avg_reward:.2f}")
            
            # Save trained agent
            agent.save("models/trained_gardener.pth")
            st.session_state.agent = agent
            st.session_state.trained = True
            
            st.success("✅ Training complete! Agent saved.")
            
            # Show final results
            st.write("### Training Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Reward", f"{rewards_history[-1]:.2f}")
            with col2:
                st.metric("Best Reward", f"{max(rewards_history):.2f}")
            with col3:
                st.metric("Average Reward", f"{np.mean(rewards_history):.2f}")
    
    # Watch AI Agent Mode
    elif mode == "Watch AI Agent":
        st.subheader("🤖 AI Gardener in Action")
        
        # Check if we have a trained agent
        if not st.session_state.trained:
            # Try to load pre-trained model
            try:
                state_size = env.observation_space.shape[0]
                action_size = env.action_space.n
                device = torch.device("cpu")
                agent = DQNAgent(state_size, action_size, device)
                
                if os.path.exists("models/trained_gardener.pth"):
                    agent.load("models/trained_gardener.pth")
                    st.session_state.agent = agent
                    st.session_state.trained = True
                    st.info("✅ Loaded pre-trained gardener!")
                else:
                    st.warning("⚠️ No pre-trained model found. Train the agent first in 'Train Agent' mode!")
                    st.stop()
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.info("Please go to 'Train Agent' mode and train a model first.")
                st.stop()
        
        # Run AI episode button
        if st.button("🌱 Run AI Episode", type="primary"):
            # Reset environment
            state, _ = env.reset()
            total_reward = 0
            
            # Create placeholders for live updates
            garden_placeholder = st.empty()
            stats_placeholder = st.empty()
            reward_placeholder = st.empty()
            
            for step in range(env.max_steps):
                # Get AI action
                action = st.session_state.agent.act(state, eval_mode=True)
                
                # Take step
                next_state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                state = next_state
                
                # Update displays
                garden_fig = visualizer.render_garden(env.plants, env.grid_size)
                garden_placeholder.plotly_chart(garden_fig, use_container_width=True)
                
                stats = env.get_garden_stats()
                stats_placeholder.write(f"""
                **Step {step + 1}/{env.max_steps}**
                - Health: {stats['avg_health']:.1f}%
                - Growth: {stats['avg_growth']:.1f}/3
                - Water: {stats['avg_water']:.1f}/5
                - Soil: {stats['avg_soil']:.1f}/5
                """)
                
                reward_placeholder.metric("Step Reward", f"{reward:.2f}", delta=f"Total: {total_reward:.2f}")
                
                if done or truncated:
                    break
            
            st.success(f"🎉 Episode complete! Total reward: {total_reward:.2f}")
            
            # Show final garden stats
            final_stats = env.get_garden_stats()
            st.write("### Final Garden Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Health", f"{final_stats['avg_health']:.1f}%")
            with col2:
                st.metric("Average Growth", f"{final_stats['avg_growth']:.1f}/3")
            with col3:
                st.metric("Average Water", f"{final_stats['avg_water']:.1f}/5")
            with col4:
                st.metric("Average Soil", f"{final_stats['avg_soil']:.1f}/5")
        
        # Show current garden if not running episode
        else:
            garden_fig = visualizer.render_garden(env.plants, env.grid_size)
            st.plotly_chart(garden_fig, use_container_width=True)
            
            # Show garden stats
            stats = env.get_garden_stats()
            st.write("### Current Garden Status")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Health", f"{stats['avg_health']:.1f}%")
            with col2:
                st.metric("Growth", f"{stats['avg_growth']:.1f}/3")
            with col3:
                st.metric("Water", f"{stats['avg_water']:.1f}/5")
            with col4:
                st.metric("Soil", f"{stats['avg_soil']:.1f}/5")
    
    # Play Yourself Mode
    elif mode == "Play Yourself":
        st.subheader("👨‍🌾 You're the Gardener!")
        
        # Initialize game state
        if 'user_game' not in st.session_state:
            st.session_state.user_env = GardenEnv(grid_size=3)
            st.session_state.user_total_reward = 0
            st.session_state.user_step = 0
        
        user_env = st.session_state.user_env
        
        # Show garden
        garden_fig = visualizer.render_garden(user_env.plants, user_env.grid_size)
        st.plotly_chart(garden_fig, use_container_width=True)
        
        # Show stats
        stats = user_env.get_garden_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Garden Health", f"{stats['avg_health']:.1f}%")
        with col2:
            st.metric("Garden Growth", f"{stats['avg_growth']:.1f}/3")
        with col3:
            st.metric("Your Score", f"{st.session_state.user_total_reward:.2f}")
        
        # Plant selection
        plant_names = [f"Plant ({p.x},{p.y})" for p in user_env.plants]
        selected_plant = st.selectbox("Select a plant", range(len(plant_names)), format_func=lambda x: plant_names[x])
        
        # Action buttons
        st.write("### Choose Your Action")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💧 Water", use_container_width=True):
                action_code = 0  # water
                action = selected_plant * 4 + action_code
                next_state, reward, done, truncated, _ = user_env.step(action)
                st.session_state.user_total_reward += reward
                st.session_state.user_step += 1
                st.rerun()
        
        with col2:
            if st.button("🌿 Fertilize", use_container_width=True):
                action_code = 1  # fertilize
                action = selected_plant * 4 + action_code
                next_state, reward, done, truncated, _ = user_env.step(action)
                st.session_state.user_total_reward += reward
                st.session_state.user_step += 1
                st.rerun()
        
        with col3:
            if st.button("✂️ Prune", use_container_width=True):
                action_code = 2  # prune
                action = selected_plant * 4 + action_code
                next_state, reward, done, truncated, _ = user_env.step(action)
                st.session_state.user_total_reward += reward
                st.session_state.user_step += 1
                st.rerun()
        
        with col4:
            if st.button("⏰ Wait", use_container_width=True):
                action_code = 3  # wait
                action = selected_plant * 4 + action_code
                next_state, reward, done, truncated, _ = user_env.step(action)
                st.session_state.user_total_reward += reward
                st.session_state.user_step += 1
                st.rerun()
        
        # Reset button
        if st.button("🔄 Reset Garden", type="secondary"):
            st.session_state.user_env = GardenEnv(grid_size=3)
            st.session_state.user_total_reward = 0
            st.session_state.user_step = 0
            st.rerun()
        
        # Show step info
        if st.session_state.user_step > 0:
            st.info(f"Day {st.session_state.user_step}/{user_env.max_steps}")
            if st.session_state.user_step >= user_env.max_steps:
                st.success(f"🎉 Game complete! Final score: {st.session_state.user_total_reward:.2f}")
    
    # Instructions
    with st.expander("📖 How It Works", expanded=False):
        st.markdown("""
        ### 🤖 AI Gardener Learning to Care for Plants
        
        **🌱 Plants have 3 key resources:**
        - 💧 Water (0-5)
        - 🌞 Sunlight (0-5) 
        - 🌍 Soil Quality (0-5)
        
        **🎯 Actions:**
        - 💦 Water: Increases water level
        - 🌿 Fertilize: Improves soil quality
        - ✂️ Prune: Boosts health when damaged
        - ⏰ Wait: Do nothing
        
        **🏆 Rewards:**
        - +0.5 for healthy plants (>80% health)
        - +0.2 for growth progress
        - Bonus for mature plants
        - Penalty for neglect
        
        **Three Modes:**
        1. **Train Agent**: Train AI from scratch
        2. **Watch AI Agent**: See trained AI in action
        3. **Play Yourself**: Control the garden manually
        
        Watch the plants grow and the AI learn! 🌱
        """)

if __name__ == "__main__":
    main()