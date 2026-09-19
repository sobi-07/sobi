import os
import requests
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(page_title="Day 5 GenAI", page_icon="🤖")
st.title("🤖 Day 5 GenAI Application")
st.caption("Streamlit + FastAPI + Gemini")

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.getenv("GOOGLE_API_KEY")

st.sidebar.header("Settings")
use_fastapi = st.sidebar.checkbox("Use FastAPI backend", value=False)

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_area(
    "Ask a question",
    placeholder="Explain RAG in simple words.",
    height=120,
)

if st.button("Ask AI", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Thinking..."):
        try:
            if use_fastapi:
                api_url = st.sidebar.text_input(
                    "FastAPI URL",
                    value="http://127.0.0.1:8000",
                )
                response = requests.post(
                    f"{api_url.rstrip('/')}/chat",
                    json={"question": question},
                    timeout=60,
                )
                response.raise_for_status()
                answer = response.json()["answer"]
            else:
                if not api_key:
                    st.error("GOOGLE_API_KEY is not configured.")
                    st.stop()

                result = ChatGoogleGenerativeAI(
                    model="gemini-3.6-flash",
                    temperature=0,
                    google_api_key=api_key,
                ).invoke(question)

                content = result.content
                if isinstance(content, str):
                    answer = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, str):
                            parts.append(block)
                        elif isinstance(block, dict) and "text" in block:
                            parts.append(block["text"])
                    answer = "".join(parts)
                else:
                    answer = str(content)

            st.session_state.history.append(
                {"question": question, "answer": answer}
            )

        except requests.RequestException as exc:
            st.error(f"FastAPI request failed: {exc}")
        except Exception as exc:
            st.error(f"AI request failed: {type(exc).__name__}: {exc}")

st.divider()
st.subheader("Conversation")

for item in reversed(st.session_state.history):
    st.markdown(f"**You:** {item['question']}")
    st.markdown(f"**AI:** {item['answer']}")
    st.divider()
