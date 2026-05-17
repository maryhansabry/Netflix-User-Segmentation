"""
gemini_utils.py — Gemini AI context building and chat initialization
"""
import streamlit as st
import google.generativeai as genai

from config import (
    CLUSTER_DESC, CLUSTER_RECOS,
    COUNTRY_TO_REGION, REGION_CONTEXT,
)


def build_system_context(user_data: dict, label: str, engagement: float, confidence: float) -> str:
    country = user_data.get("Country", "Unknown")
    region  = COUNTRY_TO_REGION.get(country, "Unknown")
    region_note = REGION_CONTEXT.get(region, "no additional regional context available")

    return f"""You are a warm, insightful Netflix data analyst who uses storytelling.
You analyzed a user and here is everything you know:

USER PROFILE:
- Age: {user_data['Age']} | Country: {country} | Region: {region}
- Subscription: {user_data['Subscription_Type']} | Favorite Genre: {user_data['Favorite_Genre']}
- Avg Watch Hours/Day: {user_data['Watch_Time_Hours']} | Days Since Last Login: {user_data['Days_Since_Last_Login']}

REGIONAL MARKET CONTEXT ({region}):
{region_note}

SEGMENT RESULT:
- Segment: {label}
- Engagement Score: {engagement:.2f}
- Confidence: {confidence:.1f}%
- Description: {CLUSTER_DESC[label]}
- Recommendations: {'; '.join(CLUSTER_RECOS[label])}

REGIONAL BUSINESS IMPLICATIONS:
- Consider how the regional market context above shapes the recommendations
- Factor in local pricing sensitivity, content preferences, and competitive landscape
- Tailor retention/upsell language to what works in {region} markets

RULES:
- Talk TO the business analyst, not the Netflix user
- Be warm, insightful, concise (under 200 words unless asked for more)
- Reference specific numbers AND regional context when relevant
- When discussing recommendations, frame them in the context of the {region} market
- Only discuss this user's data and regional context; politely decline off-topic questions
"""


def init_chat(user_data: dict, label: str, eng: float, conf: float, api_key: str):
    """Initialize Gemini chat and return (chat_obj, intro_text) or raise."""
    genai.configure(api_key=api_key)
    model   = genai.GenerativeModel("gemini-2.5-flash")
    context = build_system_context(user_data, label, eng, conf)
    chat    = model.start_chat(history=[])
    intro   = chat.send_message(
        f"{context}\n\nGreet the analyst, summarize this user in 2-3 warm sentences, "
        "and add one sentence about what makes their regional market interesting for this segment."
    ).text
    return chat, intro


def render_chat(session_key: str, user_data: dict, label: str, eng: float, conf: float, api_key: str):
    """Full chat UI component — initializes on first call, renders history, handles input."""

    if not api_key:
        st.info("Add GEMINI_API_KEY to your .env file to enable AI storytelling.", icon="🔑")
        return

    # Init chat once per profile
    if session_key not in st.session_state:
        try:
            chat, intro = init_chat(user_data, label, eng, conf, api_key)
            st.session_state[session_key] = {"chat": chat, "messages": [("ai", intro)]}
        except Exception as e:
            st.error(f"Gemini error: {e}")
            return

    chat_state = st.session_state[session_key]

    # Render all messages
    for role, msg in chat_state["messages"]:
        if role == "ai":
            st.markdown(
                f'<div class="chat-ai"><div class="chat-label ai">🤖 AI Analyst</div>{msg}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-user"><div class="chat-label user">👤 You</div>{msg}</div>',
                unsafe_allow_html=True,
            )

    user_q = st.chat_input("Ask about this user…")
    if user_q:
        chat_state["messages"].append(("user", user_q))
        with st.spinner("Thinking…"):
            try:
                reply = chat_state["chat"].send_message(user_q).text
                chat_state["messages"].append(("ai", reply))
            except Exception as e:
                st.error(f"Gemini error: {e}")
        st.rerun()
