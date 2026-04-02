import streamlit as st
import torch
import numpy as np
import time
import os
import pandas as pd
import json
import random
import hashlib
from datetime import datetime
from environment.garden_env_normalized import GardenEnvNormalized
from agents.dqn_agent_fixed import DQNAgentFixed

# Page config
st.set_page_config(
    page_title="🌱 AI Gardening Game",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== CSS WITH FIXED VISIBILITY ==========
st.markdown("""
<style>
    /* Main container */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Login box */
    .login-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 50px;
        border-radius: 25px;
        text-align: center;
        color: white;
        margin: 50px 0;
    }
    
    /* Game header */
    .game-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
    
    /* Plant card - IMPROVED VISIBILITY */
    .plant-card {
        background: white;
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 2px solid #e0e0e0;
        position: relative;
    }
    
    .plant-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        border-color: #4CAF50;
    }
    
    .plant-card.selected {
        border: 3px solid #ffd700;
        background: #fffef5;
        box-shadow: 0 0 15px rgba(255,215,0,0.3);
    }
    
    .plant-emoji {
        font-size: 56px;
        margin-bottom: 10px;
    }
    
    .plant-name {
        font-size: 16px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    
    /* Health bar */
    .health-bar-container {
        background: #e0e0e0;
        border-radius: 10px;
        height: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .health-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s;
    }
    
    /* Stats row - FIXED VISIBILITY - Dark text on light background */
    .stats-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin: 10px 0;
        padding: 5px;
        background: #f0f2f5;
        border-radius: 10px;
    }
    
    .stat-item {
        flex: 1;
        text-align: center;
        font-size: 13px;
        font-weight: bold;
        color: #2c3e50 !important;
        background: white;
        padding: 4px 6px;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Popup action panel - NEAR THE PLANT */
    .action-popup {
        position: relative;
        margin-top: 15px;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        animation: fadeIn 0.3s ease;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .action-buttons-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 10px;
    }
    
    .action-button {
        padding: 12px;
        border: none;
        border-radius: 12px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.2s;
        color: white;
        text-align: center;
    }
    
    .action-water { background: #3498db; }
    .action-water:hover { background: #2980b9; transform: scale(1.02); }
    .action-fertilize { background: #2ecc71; }
    .action-fertilize:hover { background: #27ae60; transform: scale(1.02); }
    .action-prune { background: #e67e22; }
    .action-prune:hover { background: #d35400; transform: scale(1.02); }
    .action-wait { background: #95a5a6; }
    .action-wait:hover { background: #7f8c8d; transform: scale(1.02); }
    
    /* Combo card */
    .combo-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        margin: 10px 0;
        color: white;
        font-weight: bold;
    }
    
    /* Timer card */
    .timer-card {
        background: #e74c3c;
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    
    /* Alerts */
    .alert-success { background: #d4edda; color: #155724; padding: 10px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #28a745; }
    .alert-warning { background: #fff3cd; color: #856404; padding: 10px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }
    .alert-danger { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #dc3545; }
    .alert-info { background: #d1ecf1; color: #0c5460; padding: 10px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #17a2b8; }
    
    /* Level badge */
    .level-badge {
        background: #f1c40f;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        color: #2c3e50;
        display: inline-block;
    }
    
    /* Sidebar */
    .sidebar-stats {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 12px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ========== DATABASE FUNCTIONS ==========
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_user_data(username):
    try:
        os.makedirs("users", exist_ok=True)
        with open(f"users/{username}.json", 'r') as f:
            return json.load(f)
    except:
        return None

def save_user_data(username, data):
    os.makedirs("users", exist_ok=True)
    with open(f"users/{username}.json", 'w') as f:
        json.dump(data, f, indent=2)

def register_user(username, password):
    if load_user_data(username):
        return False, "Username already exists!"
    
    user_data = {
        "username": username,
        "password": hash_password(password),
        "coins": 100,
        "level": 1,
        "exp": 0,
        "achievements": [],
        "inventory": {"water_boost": 0, "fertilizer_boost": 0},
        "stats": {"games_played": 0, "best_score": 0, "total_score": 0, "perfect_moves": 0},
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_user_data(username, user_data)
    return True, "User created successfully!"

def login_user(username, password):
    user_data = load_user_data(username)
    if not user_data:
        return False, "User not found!"
    if user_data["password"] != hash_password(password):
        return False, "Wrong password!"
    return True, user_data

# ========== LOAD MODEL ==========
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
    except:
        pass
    return None

# ========== HELPER FUNCTIONS ==========
def get_plant_emoji(plant):
    growth = plant.growth_stage
    if growth < 0.8:
        return "🌱"
    elif growth < 1.6:
        return "🌿"
    elif growth < 2.4:
        return "🌳"
    else:
        return "🌻"

def get_health_color(health):
    if health > 70:
        return "#2ecc71"
    elif health > 40:
        return "#f1c40f"
    else:
        return "#e74c3c"

def get_smart_hint(plant, combo):
    if plant.health < 30:
        return "⚠️ CRITICAL! Prune immediately!"
    elif plant.health < 50:
        return f"⚠️ Unhealthy plant. Prune to recover! Combo: x{combo}"
    elif plant.water < 2:
        return "💧 Thirsty! Water this plant!"
    elif plant.water > 4.5:
        return "💧 Too much water! Let it dry."
    elif plant.soil < 2:
        return "🌿 Poor soil! Fertilize!"
    elif plant.soil > 4.5:
        return "🌿 Too much fertilizer! Wait."
    elif plant.growth_stage > 2.5:
        return "🌟 Mature plant! Maintain its health!"
    elif plant.health > 80 and combo > 5:
        return f"🔥 ON FIRE! x{combo} combo! Keep going!"
    else:
        return "💪 Take care of your plants! Good actions build combo!"

def random_event(env):
    events = [
        {"name": "🌧️ RAIN!", "effect": "water", "value": 0.8, "message": "Rain watered all plants! +0.8 water"},
        {"name": "☀️ HEATWAVE!", "effect": "water", "value": -0.5, "message": "Heatwave! Water decreased by 0.5"},
        {"name": "🦋 BENEFICIAL INSECTS!", "effect": "soil", "value": 0.5, "message": "Insects improved soil! +0.5 soil"},
    ]
    
    if random.random() < 0.12:
        event = random.choice(events)
        for plant in env.plants:
            if event["effect"] == "water":
                plant.water = np.clip(plant.water + event["value"], 0, 5)
            elif event["effect"] == "soil":
                plant.soil = np.clip(plant.soil + event["value"], 0, 5)
        return event["message"]
    return None

def ai_trash_talk():
    messages = [
        "🤖 You call that gardening?",
        "🤖 My algorithms are superior!",
        "🤖 Is that all you've got?",
        "🤖 Your plants are suffering!",
        "🤖 Humans are obsolete!",
    ]
    return random.choice(messages)

def display_plant_card(plant, idx, is_selected):
    emoji = get_plant_emoji(plant)
    health_color = get_health_color(plant.health)
    health_percent = plant.health / 100
    selected_class = "plant-card selected" if is_selected else "plant-card"
    
    # Determine health text color based on health level
    if plant.health > 70:
        health_text_color = "#2ecc71"
    elif plant.health > 40:
        health_text_color = "#f1c40f"
    else:
        health_text_color = "#e74c3c"
    
    return f"""
    <div class="{selected_class}">
        <div class="plant-emoji">{emoji}</div>
        <div class="plant-name">🌿 Plant {idx + 1}</div>
        <div class="health-bar-container">
            <div class="health-bar" style="width: {health_percent * 100}%; background: {health_color};"></div>
        </div>
        <div class="stats-row">
            <div class="stat-item">❤️ <span style="color: {health_text_color}; font-weight: bold;">{plant.health:.0f}%</span></div>
            <div class="stat-item">📈 <span style="color: #2c3e50; font-weight: bold;">{plant.growth_stage:.1f}</span></div>
        </div>
        <div class="stats-row">
            <div class="stat-item">💧 <span style="color: #2980b9; font-weight: bold;">{plant.water:.1f}</span></div>
            <div class="stat-item">🌍 <span style="color: #8e44ad; font-weight: bold;">{plant.soil:.1f}</span></div>
        </div>
    </div>
    """

def display_action_popup(plant, idx, combo):
    emoji = get_plant_emoji(plant)
    hint = get_smart_hint(plant, combo)
    
    return f"""
    <div class="action-popup">
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
            <div style="font-size: 48px;">{emoji}</div>
            <div>
                <div style="font-size: 18px; font-weight: bold; color: white;">Plant {idx + 1}</div>
                <div style="font-size: 12px; color: rgba(255,255,255,0.9);">❤️ {plant.health:.0f}% | 📈 {plant.growth_stage:.1f} | 💧 {plant.water:.1f} | 🌍 {plant.soil:.1f}</div>
            </div>
        </div>
        <div style="font-size: 13px; color: rgba(255,255,255,0.95); margin-bottom: 15px; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 10px;">
            💡 {hint}
        </div>
        <div class="action-buttons-grid">
            <div class="action-button action-water" data-action="water">💧 WATER</div>
            <div class="action-button action-fertilize" data-action="fertilize">🌿 FERTILIZE</div>
            <div class="action-button action-prune" data-action="prune">✂️ PRUNE</div>
            <div class="action-button action-wait" data-action="wait">⏰ WAIT</div>
        </div>
    </div>
    """

def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # ========== LOGIN SYSTEM ==========
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        st.markdown("""
        <div class="login-box">
            <div style="font-size: 48px;">🌱</div>
            <div style="font-size: 28px; font-weight: bold;">AI Gardening Game</div>
            <div style="margin: 20px 0;">Create your gardener profile and save your progress!</div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", use_container_width=True):
                success, result = login_user(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.coins = result["coins"]
                    st.session_state.level = result["level"]
                    st.session_state.exp = result["exp"]
                    st.session_state.achievements = result["achievements"]
                    st.session_state.inventory = result["inventory"]
                    st.session_state.stats = result["stats"]
                    st.rerun()
                else:
                    st.error(result)
        
        with tab2:
            new_username = st.text_input("Choose Username", key="reg_user")
            new_password = st.text_input("Choose Password", type="password", key="reg_pass")
            if st.button("Register", use_container_width=True):
                success, message = register_user(new_username, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        st.stop()
    
    # ========== LOAD AGENT ==========
    agent = load_model()
    
    # ========== INITIALIZE GAME ==========
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
    
    # ========== SIDEBAR - PLAYER INFO ==========
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 48px;">👤</div>
            <div style="font-size: 20px; font-weight: bold;">{st.session_state.username}</div>
            <div><span class="level-badge">Level {st.session_state.level}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f1c40f 0%, #e67e22 100%); padding: 15px; border-radius: 15px; text-align: center; margin-bottom: 20px;">
            <div style="font-size: 24px;">💰 {st.session_state.coins}</div>
            <div>Coins</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Stats
        st.subheader("📊 Stats")
        st.markdown(f"""
        <div class="sidebar-stats">
            🎮 Games: {st.session_state.stats['games_played']}<br>
            🏆 Best Score: {st.session_state.stats['best_score']}<br>
            ⭐ Perfect Moves: {st.session_state.stats['perfect_moves']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # ========== MAIN MENU ==========
    if not st.session_state.game_active:
        st.markdown("""
        <div class="login-box" style="padding: 40px;">
            <div style="font-size: 48px;">🎮</div>
            <div style="font-size: 28px; font-weight: bold;">Choose Your Adventure</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌱 PLAY GAME", use_container_width=True):
                st.session_state.game_active = True
                st.session_state.game_mode = "normal"
                st.session_state.human_env = GardenEnvNormalized(grid_size=3)
                st.session_state.human_score = 0
                st.session_state.human_steps = 0
                st.session_state.human_game_over = False
                st.session_state.combo = 0
                st.session_state.perfect_moves = 0
                st.session_state.score_history = []
                st.session_state.selected_plant = None
                st.rerun()
        
        with col2:
            if st.button("🤖 CHALLENGE AI", use_container_width=True):
                st.session_state.game_active = True
                st.session_state.game_mode = "vs_ai"
                st.session_state.human_env = GardenEnvNormalized(grid_size=3)
                st.session_state.human_score = 0
                st.session_state.human_steps = 0
                st.session_state.human_game_over = False
                st.session_state.combo = 0
                st.session_state.perfect_moves = 0
                st.session_state.score_history = []
                st.session_state.ai_score = None
                st.session_state.ai_message = None
                st.session_state.selected_plant = None
                st.rerun()
        
        st.stop()
    
    # ========== GAME PLAY ==========
    env = st.session_state.human_env
    stats = env.get_garden_stats()
    
    # Event
    event_msg = random_event(env)
    if event_msg:
        st.info(f"✨ {event_msg}")
    
    # Header
    mode_title = "CHALLENGE AI" if st.session_state.game_mode == "vs_ai" else "PLAY MODE"
    st.markdown(f"""
    <div class="game-header">
        <div style="font-size: 24px;">{mode_title}</div>
        <div>🏆 Score: <span style="font-size: 20px; font-weight: bold;">{st.session_state.human_score:.0f}</span> | 🔥 Combo: x{st.session_state.combo} | 📅 Day: {st.session_state.human_steps}/100</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Timer
    time_left = 100 - st.session_state.human_steps
    st.markdown(f"""
    <div class="timer-card">
        ⏳ Time Left: {time_left} days
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bars
    col1, col2 = st.columns(2)
    with col1:
        st.progress(stats['avg_health'] / 100, text="🌿 Garden Health")
    with col2:
        st.progress(st.session_state.human_steps / 100, text="📅 Day Progress")
    
    # Combo display
    if st.session_state.combo > 0:
        st.markdown(f"""
        <div class="combo-card">
            🔥 COMBO x{st.session_state.combo} 🔥
        </div>
        """, unsafe_allow_html=True)
    
    # ========== VS AI MODE ==========
    if st.session_state.game_mode == "vs_ai":
        col_left, col_mid, col_right = st.columns([1, 0.2, 1])
        
        with col_left:
            if st.button("🚀 Run AI", use_container_width=True):
                with st.spinner("🤖 AI is thinking..."):
                    time.sleep(0.5)
                    ai_env = GardenEnvNormalized(grid_size=3)
                    state, _ = ai_env.reset()
                    total = 0
                    for _ in range(ai_env.max_steps):
                        action = agent.act(state, eval_mode=True)
                        next_state, reward, done, truncated, _ = ai_env.step(action)
                        total += reward
                        state = next_state
                        if done or truncated:
                            break
                    st.session_state.ai_score = total
                    st.session_state.ai_message = ai_trash_talk()
                    st.rerun()
            
            if st.session_state.get('ai_score') is not None:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 20px; text-align: center; color: white;">
                    <div>🤖 AI SCORE</div>
                    <div style="font-size: 48px; font-weight: bold;">{st.session_state.ai_score:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_mid:
            st.markdown("""
            <div style="text-align: center; padding: 50px 0;">
                <div style="font-size: 48px;">⚔️</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_right:
            if st.session_state.get('ai_message'):
                st.warning(st.session_state.ai_message)
    
    # ========== GARDEN DISPLAY WITH POPUP ACTIONS ==========
    st.markdown("### 🌿 Your Garden")
    st.caption("💡 Click on any plant to reveal action buttons right below it")
    
    plants = env.plants
    cols = st.columns(3)
    
    def take_action(action_type):
        action_val = st.session_state.selected_plant * 4 + action_type
        _, reward, done, truncated, _ = env.step(action_val)
        st.session_state.human_score += reward
        st.session_state.human_steps += 1
        st.session_state.score_history.append(st.session_state.human_score)
        
        action_names = ["Watered", "Fertilized", "Pruned", "Waited"]
        
        if reward > 0.2:
            st.session_state.combo += 1
            st.session_state.perfect_moves += 1
            st.toast(f"✅ PERFECT! {action_names[action_type]} +{reward:.1f} | Combo x{st.session_state.combo}", icon="✨")
        else:
            st.session_state.combo = 0
            st.toast(f"⚠️ {action_names[action_type]} {reward:.1f} points | Combo broken!", icon="💔")
        
        if done or truncated:
            st.session_state.human_game_over = True
            st.session_state.stats['games_played'] += 1
            if st.session_state.human_score > st.session_state.stats['best_score']:
                st.session_state.stats['best_score'] = st.session_state.human_score
            st.session_state.stats['perfect_moves'] += st.session_state.perfect_moves
            st.session_state.coins += int(st.session_state.human_score / 5)
            save_user_data(st.session_state.username, {
                "username": st.session_state.username,
                "coins": st.session_state.coins,
                "level": st.session_state.level,
                "exp": st.session_state.exp,
                "achievements": st.session_state.achievements,
                "inventory": st.session_state.inventory,
                "stats": st.session_state.stats
            })
            st.balloons()
        
        st.session_state.selected_plant = None
        st.rerun()
    
    # Display plants
    for idx, plant in enumerate(plants):
        col = cols[idx % 3]
        with col:
            is_selected = (idx == st.session_state.get('selected_plant'))
            plant_html = display_plant_card(plant, idx, is_selected)
            st.markdown(plant_html, unsafe_allow_html=True)
            
            # Select plant button
            if st.button(f"🌿 Select Plant {idx+1}", key=f"select_{idx}", use_container_width=True):
                if st.session_state.get('selected_plant') == idx:
                    st.session_state.selected_plant = None
                else:
                    st.session_state.selected_plant = idx
                st.rerun()
            
            # Show action popup right below the selected plant
            if st.session_state.get('selected_plant') == idx:
                st.markdown(f"""
                <div class="action-popup">
                    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                        <div style="font-size: 48px;">{get_plant_emoji(plant)}</div>
                        <div>
                            <div style="font-size: 18px; font-weight: bold; color: white;">Plant {idx + 1}</div>
                            <div style="font-size: 12px; color: rgba(255,255,255,0.9);">❤️ {plant.health:.0f}% | 📈 {plant.growth_stage:.1f} | 💧 {plant.water:.1f} | 🌍 {plant.soil:.1f}</div>
                        </div>
                    </div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.95); margin-bottom: 15px; padding: 8px; background: rgba(0,0,0,0.2); border-radius: 10px;">
                        💡 {get_smart_hint(plant, st.session_state.combo)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    if st.button("💧 WATER", key=f"water_{idx}", use_container_width=True):
                        take_action(0)
                with col_b:
                    if st.button("🌿 FERTILIZE", key=f"fert_{idx}", use_container_width=True):
                        take_action(1)
                with col_c:
                    if st.button("✂️ PRUNE", key=f"prune_{idx}", use_container_width=True):
                        take_action(2)
                with col_d:
                    if st.button("⏰ WAIT", key=f"wait_{idx}", use_container_width=True):
                        take_action(3)
    
    # ========== SCORE HISTORY ==========
    if len(st.session_state.score_history) > 1:
        st.markdown("---")
        st.subheader("📈 Score Progress")
        chart_data = pd.DataFrame({
            'Step': range(len(st.session_state.score_history)),
            'Score': st.session_state.score_history
        })
        st.line_chart(chart_data.set_index('Step'), use_container_width=True)
    
    # ========== RESET BUTTONS ==========
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset Game", use_container_width=True):
            st.session_state.game_active = False
            st.rerun()
    with col2:
        if st.button("🏠 Main Menu", use_container_width=True):
            st.session_state.game_active = False
            st.rerun()
    
    # ========== GAME OVER ==========
    if st.session_state.get('human_game_over', False):
        st.markdown("---")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 20px; text-align: center; color: white;">
            <div style="font-size: 64px;">🎉</div>
            <div style="font-size: 32px; font-weight: bold;">GAME COMPLETE!</div>
            <div style="font-size: 48px; font-weight: bold;">{st.session_state.human_score:.0f}</div>
            <div>🔥 Best Combo: x{st.session_state.combo}</div>
            <div>⭐ Perfect Moves: {st.session_state.perfect_moves}</div>
            <div>💰 Coins Earned: {int(st.session_state.human_score / 5)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Play Again", use_container_width=True):
                st.session_state.game_active = False
                st.rerun()
        with col2:
            if st.button("🏠 Main Menu", use_container_width=True):
                st.session_state.game_active = False
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()