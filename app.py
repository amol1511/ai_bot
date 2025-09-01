import streamlit as st
import google.generativeai as genai
import tempfile
import docx
import fitz  # PyMuPDF for PDFs
from datetime import datetime

# ===============================
# CONFIGURE GEMINI API
# ===============================
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
genai.configure(api_key=API_KEY if API_KEY else None)
model = genai.GenerativeModel("gemini-1.5-flash")

# ===============================
# HELPER FUNCTIONS
# ===============================
def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT files."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    text = ""

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if file_type == "pdf":
        doc = fitz.open(tmp_path)
        for page in doc:
            text += page.get_text()
    elif file_type == "docx":
        doc = docx.Document(tmp_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_type == "txt":
        with open(tmp_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = "⚠️ Unsupported file type."

    return text


def chat_with_ai(prompt, context=""):
    """Send user prompt + context to Gemini."""
    try:
        full_prompt = f"{context}\n\nUser: {prompt}\nAI:"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {e}"

# ===============================
# STREAMLIT PAGE CONFIG
# ===============================
st.set_page_config(page_title="Gemini AI Chatbot", page_icon="🤖", layout="wide")

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1f1c2c, #928DAB);
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background: transparent;
    }
    .chat-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        max-width: 900px;
        margin: auto;
        height: 65vh;
        overflow-y: auto;
        scroll-behavior: smooth;
    }
    .chat-bubble-user {
        background: #4a90e2;
        color: white;
        padding: 12px 15px;
        border-radius: 18px;
        margin: 10px;
        text-align: right;
        max-width: 70%;
        margin-left: auto;
    }
    .chat-bubble-ai {
        background: #2ecc71;
        color: white;
        padding: 12px 15px;
        border-radius: 18px;
        margin: 10px;
        text-align: left;
        max-width: 70%;
        margin-right: auto;
    }
    .input-container {
        background: rgba(255, 255, 255, 0.2);
        padding: 15px;
        border-radius: 15px;
        margin-top: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 12px;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #ddd;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================
# TITLE
# ===============================
st.markdown('<div class="title">🤖 Gemini AI Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Chat naturally or upload a document for AI-powered insights!</div>', unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Settings")
api_key_input = st.sidebar.text_input("Enter Gemini API Key:", type="password")
if api_key_input:
    genai.configure(api_key=api_key_input)

# File Upload
uploaded_file = st.sidebar.file_uploader("📂 Upload a document", type=["pdf", "docx", "txt"])
doc_text = ""
if uploaded_file:
    file_details = {"Filename": uploaded_file.name, "Size (KB)": round(uploaded_file.size / 1024, 2)}
    st.sidebar.write("📄 **File Details:**", file_details)
    doc_text = extract_text_from_file(uploaded_file)
    st.sidebar.success("✅ Document uploaded successfully!")

# ===============================
# CHAT HISTORY
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ===============================
# CHAT UI
# ===============================
with st.container():
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-ai'>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# INPUT FIELD
# ===============================
user_input = st.text_input("💬 Type your message:", "")
if st.button("🚀 Send"):
    if user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        answer = chat_with_ai(user_input, doc_text)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.experimental_rerun()
    else:
        st.warning("⚠️ Please enter a message.")

# ===============================
# FOOTER
# ===============================
st.markdown("<hr><center style='color:white;'>✨ Built with ❤️ using Streamlit & Gemini</center>", unsafe_allow_html=True)
