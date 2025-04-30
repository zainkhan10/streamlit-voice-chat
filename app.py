import os
import streamlit as st
import speech_recognition as sr
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# ─── Helper: speech → text ──────────────────────────────────────
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Say something…")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.success(f"🗣 You said: {text}")
        return text
    except Exception as e:
        st.error(f"⚠️ Could not recognize audio: {e}")
        return None

# ─── Main App ──────────────────────────────────────────────────
def main():
    st.title("🗣️ + 🤖 Audio‐to‐Chat Demo")

    # 1️⃣ Initialize history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2️⃣ Record audio button
    if st.button("Record Audio"):
        transcript = speech_to_text()
        if transcript:
            st.session_state.messages.append(HumanMessage(transcript))

    # 3️⃣ Count & disable after 10 prompts
    user_prompt_count = sum(isinstance(m, HumanMessage) for m in st.session_state.messages)
    max_prompts = 10
    disabled = user_prompt_count >= max_prompts

    st.markdown(f"**Prompts used:** {user_prompt_count}/{max_prompts}")

    # 4️⃣ Typed input fallback
    prompt = st.text_input("Or type your prompt", disabled=disabled)
    if prompt and not disabled:
        st.session_state.messages.append(HumanMessage(prompt))

    # 5️⃣ Call the LLM if there's any new human message
    if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage):
        try:
            llm = ChatOllama(model="mistral")
            ai_resp = llm.invoke(st.session_state.messages)
            st.session_state.messages.append(ai_resp)
        except Exception as e:
            st.error(f"Error calling LLM: {e}")

    # 6️⃣ Display the convo
    st.header("💬 Conversation")
    for msg in st.session_state.messages:
        role = "👤 You" if isinstance(msg, HumanMessage) else "🤖 AI"
        st.markdown(f"**{role}:** {msg.content}")

if __name__ == "__main__":
    main()

