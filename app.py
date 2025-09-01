# ===============================
# RAG AI BOT WITH GEMINI + CHROMA
# ===============================
import streamlit as st
import google.generativeai as genai
import tempfile
import docx
import fitz  # PyMuPDF for PDFs
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# ===============================
# CONFIGURE GEMINI
# ===============================
API_KEY = ""  # 🔥 Replace with st.secrets["GEMINI_KEY"] or enter in sidebar
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ===============================
# EMBEDDING MODEL
# ===============================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

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


def create_chroma_collection(chunks):
    """Create Chroma vector DB collection."""
    client = chromadb.Client()
    collection = client.create_collection(
        name="docs",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    )
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[str(i)])
    return collection


def retrieve_context(query, collection, top_k=3):
    """Retrieve top-k relevant chunks."""
    results = collection.query(query_texts=[query], n_results=top_k)
    return "\n".join(results["documents"][0])


def chat_with_rag(query, collection):
    """RAG-powered answer generation."""
    context = retrieve_context(query, collection)
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
doc_text, chunks, collection = "", [], None

if uploaded_file:
    doc_text = extract_text_from_file(uploaded_file)
    st.success("✅ Document uploaded and processed!")
    
    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(doc_text)
    
    # Vector Store
    collection = create_chroma_collection(chunks)
    
    with st.expander("📖 Preview Extracted Text"):
        st.write(doc_text[:1000] + "..." if len(doc_text) > 1000 else doc_text)

st.markdown("---")
user_input = st.text_input("💬 Ask me anything:", "")

if st.button("🚀 Send"):
    if user_input.strip() and collection:
        answer = chat_with_rag(user_input, collection)
        st.markdown(f"<div style='padding:10px;background:#DCF8C6;border-radius:10px;'><b>You:</b> {user_input}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding:10px;background:#F1F0F0;border-radius:10px;'><b>AI:</b> {answer}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please upload a document and enter a question.")

st.markdown("<hr><center>Built with ❤️ using Gemini + Chroma + Streamlit</center>", unsafe_allow_html=True)
