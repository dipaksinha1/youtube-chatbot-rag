import streamlit as st
from backend.rag_backend import YouTubeRAGChatbot
import re

st.set_page_config(page_title="YouTube RAG Chatbot")


# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = YouTubeRAGChatbot()

if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_id" not in st.session_state:
    st.session_state.video_id = None

st.title("🎥 YouTube Transcript Chatbot")


def get_video_id_from_url(url):
    """
    Returns video_id if valid YouTube URL, else None
    """
    pattern = (
        r"(?:https?:\/\/)?"
        r"(?:www\.)?"
        r"(?:youtube\.com\/(?:watch\?v=|embed\/|shorts\/)|youtu\.be\/)"
        r"([a-zA-Z0-9_-]{11})"
    )

    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None


# Step 1: Load video
if not st.session_state.video_loaded:
    # video_id
    video_url = st.text_input("Please Enter YouTube Video URL")

    # if video_url:
    if video_url is not None and st.button("Check Youtube URL"):
        video_id = get_video_id_from_url(video_url)

        if video_id:
            st.session_state.video_id = video_id
            # Thumbnail URL
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

            st.image(thumbnail_url, caption="YouTube Video Thumbnail", use_container_width=True)

        else:
            st.error("Invalid YouTube URL")

    
    if st.session_state.video_id:
        if st.button("Load Transcript"):
            with st.spinner("Indexing transcript..."):
                try:
                    st.session_state.chatbot.ingest_youtube_video(st.session_state.video_id)
                    st.session_state.video_loaded = True
                    st.success("Transcript indexed successfully!")
                    st.rerun()  # Rerun to show chat interface
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Step 2: Chat (only show if video is loaded)
else:
    st.subheader("Chat with the video")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about the video")

    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.chatbot.chat(user_input)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
                st.session_state.messages.pop()  # Remove user message if error
                st.stop()

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
