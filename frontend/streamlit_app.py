import streamlit as st
import requests
import json
from typing import Any, Dict, List

# Safe loading of API URL
try:
    API_URL = st.secrets.get("api_url", "http://localhost:8000/api")
except Exception:
    API_URL = "http://localhost:8000/api"

# Session state initialization
if "team" not in st.session_state:
    st.session_state.team = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Page config
st.set_page_config(
    page_title="Probabilistic Pundit",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for FPL-style pitch
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    /* Football Pitch Styling */
    .pitch-container {
        background: linear-gradient(180deg, #00a843 0%, #008a37 100%);
        border-radius: 12px;
        padding: 2rem 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        position: relative;
        min-height: 600px;
    }
    
    .pitch-lines {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        opacity: 0.3;
    }
    
    .position-row {
        display: flex;
        justify-content: space-around;
        align-items: center;
        margin: 1.5rem 0;
        position: relative;
        z-index: 1;
    }
    
    .player-card {
        background: white;
        border-radius: 8px;
        padding: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        min-width: 120px;
        max-width: 140px;
        transition: transform 0.2s;
    }
    
    .player-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .player-shirt {
        width: 60px;
        height: 60px;
        margin: 0 auto 0.5rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .player-name {
        font-weight: 600;
        font-size: 0.9rem;
        color: #111827;
        margin: 0.25rem 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .player-team {
        font-size: 0.75rem;
        color: #6b7280;
        margin: 0.15rem 0;
    }
    
    .player-price {
        font-size: 0.85rem;
        font-weight: 600;
        color: #059669;
        margin: 0.15rem 0;
    }
    
    .position-label {
        text-align: center;
        color: white;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 1rem 0 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .stChatMessage {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def render_pitch_view(team_data):
    """Render FPL-style pitch with players positioned by formation"""
    
    if team_data is None:
        st.warning("Team data is None")
        return
    
    # Handle dict with 'selected' key
    players_list = team_data
    if isinstance(team_data, dict):
        if "selected" in team_data:
            players_list = team_data["selected"]
        else:
            st.warning("Team data is a dict but has no 'selected' key")
            st.json(team_data)
            return
    
    if not isinstance(players_list, list):
        st.warning(f"Players data is not a list, it's a {type(players_list)}")
        return
        
    if len(players_list) == 0:
        st.warning("Players list is empty")
        return
    
    # Group players by position
    positions = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
    position_mapping = {
        "GKP": "GKP", "GK": "GKP", "Goalkeeper": "GKP",
        "DEF": "DEF", "Defender": "DEF",
        "MID": "MID", "Midfielder": "MID",
        "FWD": "FWD", "Forward": "FWD"
    }
    
    for player in players_list:
        if isinstance(player, dict):
            pos = (player.get("position") or player.get("pos") or 
                   player.get("element_type") or "Unknown")
            pos_key = position_mapping.get(pos, pos)
            if pos_key in positions:
                positions[pos_key].append(player)
    
    # Render each position row
    position_order = [
        ("GKP", "🧤 GOALKEEPER"),
        ("DEF", "🛡️ DEFENDERS"),
        ("MID", "⚡ MIDFIELDERS"),
        ("FWD", "🎯 FORWARDS")
    ]
    
    for pos_key, pos_label in position_order:
        if positions[pos_key]:
            # Position label
            st.markdown(f"**{pos_label}**")
            
            # Create equal-width columns for players
            num_players = len(positions[pos_key])
            # Add empty columns on sides for centering
            total_cols = 11  # Use 11 columns for flexible spacing
            
            if num_players == 1:
                cols = st.columns([3, 5, 3])
                player_cols = [cols[1]]
            elif num_players == 2:
                cols = st.columns([2, 3, 1, 3, 2])
                player_cols = [cols[1], cols[3]]
            elif num_players == 3:
                cols = st.columns([1, 3, 1, 3, 1, 3, 1])
                player_cols = [cols[1], cols[3], cols[5]]
            elif num_players == 4:
                cols = st.columns([0.5, 2, 1, 2, 1, 2, 1, 2, 0.5])
                player_cols = [cols[1], cols[3], cols[5], cols[7]]
            elif num_players == 5:
                cols = st.columns([2, 2, 2, 2, 2])
                player_cols = cols
            else:
                cols = st.columns(num_players)
                player_cols = cols
            
            for idx, player in enumerate(positions[pos_key]):
                with player_cols[idx]:
                    # Extract player data
                    name = (player.get("name") or player.get("web_name") or 
                           player.get("player_name") or "Unknown")
                    club = (player.get("club") or player.get("team") or 
                           player.get("team_name") or "")
                    price = (player.get("price") or player.get("now_cost") or 
                            player.get("cost") or 0)
                    
                    # Handle price formatting
                    try:
                        price_float = float(price)
                        if price_float > 100:
                            price_float = price_float / 10
                    except:
                        price_float = 0
                    
                    # Get initials for shirt
                    initials = "".join([n[0] for n in name.split()[:2]]).upper()
                    if len(initials) == 0:
                        initials = "?"
                    
                    # Render player card
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 0.5rem;">
                        <div style="width: 60px; height: 60px; margin: 0 auto 0.5rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; font-weight: bold; color: white; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                            {initials}
                        </div>
                        <div style="font-weight: 600; font-size: 0.95rem; color: #111827; margin: 0.3rem 0;">{name}</div>
                        <div style="font-size: 0.8rem; color: #6b7280; margin: 0.2rem 0;">{club}</div>
                        <div style="font-size: 0.9rem; font-weight: 600; color: #059669; margin: 0.3rem 0;">£{price_float:.1f}m</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add spacing between position rows
            st.markdown("<br>", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">⚽ Probabilistic Pundit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Fantasy Football Team Generator & Assistant</div>', unsafe_allow_html=True)

# Layout
col_main, col_sidebar = st.columns([2.5, 1.5])

with col_sidebar:
    st.markdown("### ⚙️ Team Configuration")
    
    with st.container():
        # Season selection
        season = st.selectbox(
            "Season",
            options=["2025-26","2024-25", "2023-24", "2022-23", "2021-22"],
            index=0,
            help="Select the FPL season"
        )
        
        # Gameweek selection
        gameweek = st.number_input(
            "Gameweek",
            min_value=1,
            max_value=38,
            value=1,
            step=1,
            help="Select the gameweek for predictions"
        )
        
        # Budget input
        budget = st.number_input(
            "Budget (£M)",
            min_value=50.0,
            max_value=200.0,
            value=100.0,
            step=0.5,
            help="Set your fantasy team budget"
        )
        
        generate_btn = st.button("🎯 Generate Team", type="primary", use_container_width=True)
        
        if generate_btn:
            with st.spinner("🤖 AI is analyzing players..."):
                try:
                    payload = {
                        "budget": budget,
                        "season": season,
                        "gameweek": gameweek
                    }
                    r = requests.post(
                        f"{API_URL}/generate_team",
                        json=payload,
                        timeout=60
                    )
                    if r.ok:
                        payload = r.json()
                        st.session_state.team = payload.get("team")
                        st.session_state.explanation = payload.get("explanation")
                        st.session_state.chat_history = [{
                            "role": "assistant",
                            "text": f"✅ Team generated successfully for {season} GW{gameweek}! Ask me why I picked any player or for tactical advice."
                        }]
                        st.success("✅ Team ready!")
                        st.rerun()
                    else:
                        st.error(f"❌ API error: {r.status_code}")
                except Exception as e:
                    st.error(f"❌ Backend unavailable: {e}")
    
    st.markdown("---")
    
    # Team stats
    if st.session_state.team:
        st.markdown("### 📊 Team Stats")
        
        team_data = st.session_state.team
        
        # Extract players list from dict if needed
        players_list = team_data
        if isinstance(team_data, dict) and "selected" in team_data:
            players_list = team_data["selected"]
        
        if isinstance(players_list, list):
            total_cost = 0
            total_ev = 0
            num_players = len(players_list)
            
            for p in players_list:
                if isinstance(p, dict):
                    price = p.get("price") or p.get("now_cost") or p.get("cost") or 0
                    try:
                        price_float = float(price)
                        if price_float > 100:
                            price_float = price_float / 10
                        total_cost += price_float
                    except:
                        pass
                    
                    ev = p.get("ev") or p.get("expected_value") or p.get("ep_next") or 0
                    try:
                        if ev is not None:
                            total_ev += float(ev)
                    except:
                        pass
            
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-value">£{total_cost:.1f}M</div>
                <div class="stat-label">Total Cost</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Players", num_players)
            with col2:
                st.metric("Avg EV", f"{total_ev/num_players:.1f}" if num_players > 0 else "0")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### 🔧 Actions")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.team = None
            st.session_state.explanation = None
            st.session_state.chat_history = []
            st.rerun()
    
    else:
        st.info("💡 Generate a team to get started!")
    
    # Debug info (collapsible)
    with st.expander("🔍 Debug Info"):
        st.code(API_URL, language="text")
        st.caption("Backend should be running on port 8000")
        if st.session_state.team:
            st.json(st.session_state.team)

with col_main:
    # Team Display - FPL Pitch Style
    if st.session_state.team:
        st.markdown("### 👥 Your Recommended Team")
        render_pitch_view(st.session_state.team)
        st.markdown("---")
    
    # Chat Interface
    st.markdown("### 💬 Chat with AI Assistant")
    
    if not st.session_state.team:
        st.info("👆 Generate a team first to start chatting with the AI assistant")
    else:
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for entry in st.session_state.chat_history:
                role = entry.get("role")
                text = entry.get("text")
                
                if role == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(text)
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(text)
        
        # Chat input
        user_input = st.chat_input("Ask about player selections, tactics, or strategy...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "text": user_input
            })
            
            # Show user message immediately
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            
            # Get AI response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    try:
                        payload = {
                            "question": user_input,
                            "team": st.session_state.team,
                            "context": st.session_state.explanation or {}
                        }
                        r = requests.post(
                            f"{API_URL}/explain_team",
                            json=payload,
                            timeout=60
                        )
                        
                        if r.ok:
                            resp = r.json()
                            answer = (
                                resp.get("answer") or 
                                resp.get("response") or 
                                resp.get("explanation") or 
                                json.dumps(resp)
                            )
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "text": answer
                            })
                            st.markdown(answer)
                        else:
                            error_msg = f"⚠️ API returned error {r.status_code}"
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "text": error_msg
                            })
                            st.error(error_msg)
                    except Exception as e:
                        error_msg = f"⚠️ Could not reach backend: {str(e)}"
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": error_msg
                        })
                        st.error(error_msg)
            
            st.rerun()