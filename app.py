# ===============================
# RAG AI BOT WITH GEMINI + STREAMLIT
# ===============================
import streamlit as st
import google.generativeai as genai
import tempfile
import docx
import fitz  # PyMuPDF for PDFs
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ===============================
# CONFIGURE GEMINI
# ===============================
API_KEY = "AIzaSyDCdUOuFmdphaR-ODubf10LSov6_4Qv8Y8"  # 🔥 Replace with st.secrets["GEMINI_KEY"] or enter in sidebar
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ===============================
# EMBEDDING MODEL
# ===============================
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Small & Fast

# ===============================
# HELPERS
# ===============================
def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or TXT."""
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
        text = "Unsupported file type."
    return text

def create_vectorstore(chunks):
    """Create FAISS vector store for chunks."""
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_context(query, index, chunks, embeddings, top_k=3):
    """Retrieve top-k relevant chunks for query."""
    q_emb = embedder.encode([query], convert_to_numpy=True)
    distances, idxs = index.search(q_emb, top_k)
    return "\n".join([chunks[i] for i in idxs[0]])

def chat_with_rag(query, index, chunks, embeddings):
    """RAG-powered answer generation."""
    context = retrieve_context(query, index, chunks, embeddings)
    prompt = f"Answer the user's question using this context:\n{context}\n\nUser: {query}\nAI:"
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="RAG AI Bot", page_icon="🧠", layout="wide")
st.markdown("<h1 style='text-align:center;color:#4CAF50;'>🧠 RAG AI Chatbot</h1>", unsafe_allow_html=True)
st.sidebar.header("Settings ⚙️")
api_key_input = st.sidebar.text_input("Enter Gemini API Key:", type="password")
if api_key_input:
    genai.configure(api_key=api_key_input)

uploaded_file = st.file_uploader("📂 Upload PDF/DOCX/TXT", type=["pdf", "docx", "txt"])
doc_text, chunks, index, embeddings = "", [], None, None

if uploaded_file:
    doc_text = extract_text_from_file(uploaded_file)
    st.success("✅ Document uploaded and processed!")
    
    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(doc_text)
    
    # Vector Store
    index, embeddings = create_vectorstore(chunks)
    
    with st.expander("📖 Preview Extracted Text"):
        st.write(doc_text[:1000] + "..." if len(doc_text) > 1000 else doc_text)

st.markdown("---")
user_input = st.text_input("💬 Ask me anything:", "")

if st.button("🚀 Send"):
    if user_input.strip() and index:
        answer = chat_with_rag(user_input, index, chunks, embeddings)
        st.markdown(f"<div style='padding:10px;background:#DCF8C6;border-radius:10px;'><b>You:</b> {user_input}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding:10px;background:#F1F0F0;border-radius:10px;'><b>AI:</b> {answer}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please upload a document and enter a question.")

st.markdown("<hr><center>Built with ❤️ using Gemini + FAISS + Streamlit</center>", unsafe_allow_html=True)
