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

# Custom CSS for better styling
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
    .team-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .position-header {
        font-weight: 600;
        color: #374151;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #d1d5db;
    }
    .player-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem;
        margin: 0.25rem 0;
        background: white;
        border-radius: 4px;
        border: 1px solid #e5e7eb;
    }
    .player-name {
        font-weight: 500;
        color: #111827;
    }
    .player-club {
        color: #6b7280;
        font-size: 0.9rem;
    }
    .player-price {
        color: #059669;
        font-weight: 600;
    }
    .player-ev {
        color: #7c3aed;
        font-weight: 500;
        font-size: 0.9rem;
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

# Header
st.markdown('<div class="main-header">⚽ Probabilistic Pundit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Fantasy Football Team Generator & Assistant</div>', unsafe_allow_html=True)

# Layout
col_main, col_sidebar = st.columns([2.5, 1.5])

with col_sidebar:
    st.markdown("### ⚙️ Team Configuration")
    
    with st.container():
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
                    r = requests.post(
                        f"{API_URL}/generate_team",
                        json={"budget": budget},
                        timeout=60
                    )
                    if r.ok:
                        payload = r.json()
                        st.session_state.team = payload.get("team")
                        st.session_state.explanation = payload.get("explanation")
                        st.session_state.chat_history = [{
                            "role": "assistant",
                            "text": "✅ Team generated successfully! Ask me why I picked any player or for tactical advice."
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
        if isinstance(team_data, list):
            total_cost = sum(float(p.get("price", 0)) for p in team_data if isinstance(p, dict))
            total_ev = sum(float(p.get("ev", 0)) for p in team_data if isinstance(p, dict))
            num_players = len(team_data)
            
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

with col_main:
    # Team Display
    if st.session_state.team:
        st.markdown("### 👥 Your Recommended Team")
        
        team_data = st.session_state.team
        
        # Debug: Show raw structure
        with st.expander("🔍 Debug: Raw Team Data"):
            st.json(team_data)
        
        if isinstance(team_data, list) and len(team_data) > 0:
            # Check if it's a list of dicts
            if isinstance(team_data[0], dict):
                # Group players by position
                positions = {}
                for p in team_data:
                    # Try multiple possible field names
                    pos = (p.get("position") or p.get("pos") or 
                           p.get("element_type") or "Unknown")
                    
                    if pos not in positions:
                        positions[pos] = []
                    positions[pos].append(p)
                
                # Display by position
                position_order = ["GKP", "DEF", "MID", "FWD", "Goalkeeper", "Defender", "Midfielder", "Forward"]
                position_names = {
                    "GKP": "🧤 Goalkeepers",
                    "DEF": "🛡️ Defenders", 
                    "MID": "⚡ Midfielders",
                    "FWD": "🎯 Forwards",
                    "Goalkeeper": "🧤 Goalkeepers",
                    "Defender": "🛡️ Defenders",
                    "Midfielder": "⚡ Midfielders",
                    "Forward": "🎯 Forwards",
                }
                
                displayed_positions = set()
                
                for pos_code in position_order:
                    if pos_code in positions and pos_code not in displayed_positions:
                        st.markdown(f'<div class="position-header">{position_names.get(pos_code, pos_code)}</div>', unsafe_allow_html=True)
                        displayed_positions.add(pos_code)
                        
                        for player in positions[pos_code]:
                            # Try multiple possible field names for each attribute
                            name = (player.get("name") or player.get("web_name") or 
                                   player.get("player_name") or "Unknown Player")
                            club = (player.get("club") or player.get("team") or 
                                   player.get("team_name") or "")
                            price = (player.get("price") or player.get("now_cost") or 
                                    player.get("cost") or 0)
                            ev = (player.get("ev") or player.get("expected_value") or 
                                 player.get("ep_next") or 0)
                            
                            # Handle price formatting (might be in 10s)
                            try:
                                price_float = float(price)
                                if price_float > 100:  # Likely in tenths (e.g., 105 = 10.5)
                                    price_float = price_float / 10
                            except:
                                price_float = 0
                            
                            col_a, col_b, col_c, col_d = st.columns([3, 2, 1, 1])
                            with col_a:
                                st.markdown(f'**{name}**')
                            with col_b:
                                st.markdown(f'<span style="color: #6b7280;">{club}</span>', unsafe_allow_html=True)
                            with col_c:
                                st.markdown(f'<span style="color: #059669;">£{price_float:.1f}M</span>', unsafe_allow_html=True)
                            with col_d:
                                st.markdown(f'<span style="color: #7c3aed;">EV: {ev}</span>', unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                
                # Handle any unknown positions
                for pos, players in positions.items():
                    if pos not in displayed_positions:
                        st.markdown(f'<div class="position-header">{pos}</div>', unsafe_allow_html=True)
                        for player in players:
                            st.json(player)
            else:
                # Not a list of dicts, show as-is
                st.warning("⚠️ Unexpected team data format")
                for item in team_data:
                    st.write(item)
        else:
            st.warning("⚠️ Team data is not in expected format")
            st.json(team_data)
        
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