import streamlit as st
from chatbot_model import smart_predict

st.set_page_config(page_title="Career Chatbot", page_icon="🤖")

st.title("🤖 Ask Me About My Career")
st.markdown("Curious about my work, skills, or journey? Just ask!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask a question", placeholder="e.g. Tell me about your skills")

if user_input:
    response = smart_predict(user_input)
    st.session_state.history.append((user_input, response))

for user_msg, bot_msg in reversed(st.session_state.history):
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Bot:** {bot_msg}")
