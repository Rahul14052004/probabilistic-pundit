import streamlit as st
import requests
import json
from typing import Any, Dict, List

# Safe loading of API URL
try:
    API_URL = st.secrets.get("api_url", "http://localhost:8000/api")
except Exception:
    API_URL = "http://localhost:8000/api"

# Ensure Streamlit chat primitives exist
_have_chat = hasattr(st, "chat_message") and hasattr(st, "chat_input")

# Session state initialization
if "team" not in st.session_state:
    st.session_state.team = None
if "explanation" not in st.session_state:
    st.session_state.explanation = None
if "chat_history" not in st.session_state:
    # chat entries: dicts {role: "user"|"assistant", "text": "..."}
    st.session_state.chat_history = []

st.set_page_config(page_title="Probabilistic Pundit", layout="wide")
st.title("Probabilistic Pundit — Chat & Explain (chat UI)")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.header("Team Generation")
    budget = st.number_input("Budget (M)", value=100.0, step=0.5)

    generate_btn = st.button("Generate Team")
    if generate_btn:
        with st.spinner("Generating team..."):
            try:
                r = requests.post(f"{API_URL}/generate_team", json={"budget": budget}, timeout=60)
                if r.ok:
                    payload = r.json()
                    st.session_state.team = payload.get("team")
                    st.session_state.explanation = payload.get("explanation")
                    st.session_state.chat_history = []
                    if st.session_state.explanation:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "text": "Team generated. Ask me why I picked any player or for tactical advice."
                        })
                    st.success("Team generated — ask questions below.")
                else:
                    st.error(f"API error: {r.status_code} {r.text}")
            except Exception as e:
                st.error(f"Backend unavailable: {e}")

    st.markdown("---")

    # Team view
    if st.session_state.team:
        st.subheader("Recommended Team")
        try:
            if isinstance(st.session_state.team, list) and len(st.session_state.team) > 0 and isinstance(st.session_state.team[0], dict):
                rows = []
                for p in st.session_state.team:
                    # safe extraction in case backend returns strings
                    if isinstance(p, dict):
                        pid = p.get("player_id") or p.get("id") or ""
                        name = p.get("name") or ""
                        pos = p.get("position") or ""
                        club = p.get("club") or ""
                        price = p.get("price") or ""
                        ev = p.get("ev", "")
                    else:
                        pid = str(p)
                        name = str(p)
                        pos = club = price = ev = ""
                    rows.append({"id": pid, "name": name, "position": pos, "club": club, "price": price, "ev": ev})
                st.table(rows)
            else:
                st.write(st.session_state.team)
        except Exception:
            st.write(st.session_state.team)

        # quick select player dropdown
        player_options = []
        if isinstance(st.session_state.team, list):
            for p in st.session_state.team:
                if isinstance(p, dict):
                    pid = p.get("player_id") or p.get("id") or ""
                    name = p.get("name") or str(pid)
                else:
                    pid = str(p)
                    name = str(p)
                player_options.append(f"{pid} - {name}")
        selected_player = st.selectbox("Ask about a specific player (optional)", options=["(none)"] + player_options)

    else:
        st.info("No team yet — click 'Generate Team' to create one (or ensure backend is running).")

    st.markdown("---")
    st.subheader("Chat with the Meta LLM")

    # Chat UI: use st.chat_input if available, else fallback to text_area + button
    if _have_chat:
        # Display existing history as chat bubbles
        chat_box = st.empty()
        with chat_box.container():
            for entry in st.session_state.chat_history:
                role = entry.get("role")
                text = entry.get("text")
                if role == "user":
                    st.chat_message("user").markdown(text)
                else:
                    st.chat_message("assistant").markdown(text)

        # Chat input (press Enter to send)
        user_input = st.chat_input("Enter your question (e.g. Why was Player X selected?)")
        if user_input:
            if not st.session_state.team:
                st.warning("Please generate a team first.")
            else:
                st.session_state.chat_history.append({"role": "user", "text": user_input})
                # display the new user message immediately
                st.chat_message("user").markdown(user_input)
                payload = {"question": user_input, "team": st.session_state.team, "context": st.session_state.explanation or {}}
                with st.chat_message("assistant"):
                    with st.spinner("Asking meta agent..."):
                        try:
                            r = requests.post(f"{API_URL}/explain_team", json=payload, timeout=60)
                            if r.ok:
                                resp = r.json()
                                answer = resp.get("answer") or resp.get("response") or resp.get("explanation") or json.dumps(resp)
                                st.session_state.chat_history.append({"role": "assistant", "text": answer})
                                st.write(answer)
                            else:
                                msg = f"(API error {r.status_code})"
                                st.session_state.chat_history.append({"role": "assistant", "text": msg})
                                st.write(msg)
                        except Exception as e:
                            mock_answer = f"(Mock) Could not reach backend. Your question: {user_input[:200]}"
                            st.session_state.chat_history.append({"role": "assistant", "text": mock_answer})
                            st.write(mock_answer)
    else:
        # Fallback older UI: text_area + Ask button
        question = st.text_area("Enter your question", height=80)
        ask_btn = st.button("Ask")
        if ask_btn:
            if not question or not question.strip():
                st.warning("Please enter a question.")
            elif not st.session_state.team:
                st.warning("Please generate a team first.")
            else:
                user_msg = question.strip()
                st.session_state.chat_history.append({"role": "user", "text": user_msg})
                payload = {"question": user_msg, "team": st.session_state.team, "context": st.session_state.explanation or {}}
                with st.spinner("Asking meta agent..."):
                    try:
                        r = requests.post(f"{API_URL}/explain_team", json=payload, timeout=60)
                        if r.ok:
                            resp = r.json()
                            answer = resp.get("answer") or resp.get("response") or resp.get("explanation") or json.dumps(resp)
                            st.session_state.chat_history.append({"role": "assistant", "text": answer})
                        else:
                            st.session_state.chat_history.append({"role": "assistant", "text": f"(API error {r.status_code})"})
                            st.error(f"API error: {r.status_code} {r.text}")
                    except Exception as e:
                        mock_answer = f"(Mock) I couldn't reach the backend. Your question was: {user_msg[:200]}"
                        st.session_state.chat_history.append({"role": "assistant", "text": mock_answer})
                        st.error(f"Backend request failed: {e}")

    st.markdown("---")
    st.subheader("Conversation (raw)")
    # Show raw history for debug
    st.write(st.session_state.chat_history)

with col_right:
    st.header("Meta Explanation (raw)")
    if st.session_state.explanation:
        st.json(st.session_state.explanation)
    else:
        st.info("No explanation available yet.")

    st.markdown("---")
    st.header("Quick actions")
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared.")

    st.markdown("---")
    st.header("Debug / Backend")
    st.code(API_URL)
    st.write("Tips:")
    st.markdown("- Start backend with `uvicorn backend.app.main:app --reload --port 8000`")
    st.markdown("- If no response, check backend logs and ensure `/api/generate_team` works")
