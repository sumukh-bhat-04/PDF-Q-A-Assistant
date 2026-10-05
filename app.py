import os
import streamlit as st
from dotenv import load_dotenv

# Import our custom RAG engine
from rag_engine import (
    extract_documents_from_pdf,
    chunk_documents,
    create_vector_store,
    get_rag_chain,
    query_rag
)

# To Load environment variables from .env
load_dotenv()

# Constants (hides advanced configuration for clean UX)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieving API key
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Page config
st.set_page_config(
    page_title="PDF Q&A Assistant — Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Injection
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* Global Typography Reset */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, .main-title {
        font-family: 'Outfit', sans-serif;
    }

    /* Sidebar Custom Glassmorphic Dark Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        box-shadow: 4px 0 15px -3px rgba(0, 0, 0, 0.1);
        padding-top: 1.5rem;
    }
    
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #f1f5f9 !important;
    }

    /* Beautify Streamlit File Uploader inside sidebar */
    [data-testid="stFileUploader"] {
        background-color: rgba(255, 255, 255, 0.04);
        border: 2px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px;
        padding: 12px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6 !important;
        background-color: rgba(255, 255, 255, 0.08);
    }
    
    /* App Header Title Styling */
    .main-title {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 28px;
        font-weight: 400;
    }
    
    /* Stat cards */
    .stat-card {
        background: #1e293b;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.15), 0 2px 8px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.25), 0 4px 12px -2px rgba(0, 0, 0, 0.15);
    }
    
    .stat-number {
        font-family: 'Outfit', sans-serif;
        font-size: 32px;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 2px;
    }
    
    .stat-label {
        font-size: 11px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    /* Conversational Chat Interface Bubble Styling */
    .chat-label-user {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: right;
        margin-top: 20px;
        margin-bottom: 4px;
    }
    
    .chat-bubble-user {
        background-color: #1e3b8b;
        color: #f8fafc;
        font-weight: 500;
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        margin-left: auto;
        margin-right: 0;
        max-width: 80%;
        border: 1px solid #2563eb;
        box-shadow: 0 2px 8px 0 rgba(0, 0, 0, 0.2);
        line-height: 1.5;
    }
    
    .chat-label-assistant {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 700;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 20px;
        margin-bottom: 4px;
    }
    
    .chat-bubble-assistant {
        background-color: #1e293b;
        color: #f1f5f9;
        border-left: 4px solid #7c3aed;
        border-radius: 4px 16px 16px 16px;
        padding: 18px 22px;
        margin-right: auto;
        margin-left: 0;
        max-width: 90%;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.2), 0 2px 6px -1px rgba(0, 0, 0, 0.1);
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        line-height: 1.6;
    }

    /* Source Citation Cards */
    .source-card {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 13.5px;
        color: #cbd5e1;
        line-height: 1.5;
    }
    
    .source-badge {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 10.5px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 6px;
        border: 1px solid #2563eb;
    }
    
    /* Submit Form Buttons styling */
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Pre-flight Check: Ensure Gemini API key is configured
if not api_key:
    st.error("🔑 **Gemini API Key missing!** Please add your API key to the `.env` file in the project root directory (e.g. `GEMINI_API_KEY=your_key_here`).")
    st.stop()

# Initialize Session States
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "stats" not in st.session_state:
    st.session_state.stats = {}

# Sidebar Configuration
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 10px;'><img src='https://img.icons8.com/color/96/000000/pdf.png' width='70'/></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-bottom: 20px; font-family: Outfit;'>Document Hub</h3>", unsafe_allow_html=True)
    
    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF document to begin", type=["pdf"])
    
    st.markdown("<div style='margin-top: 40px; padding: 12px; border-radius: 8px; background-color: rgba(255,255,255,0.02); font-size: 12px; color: #94a3b8; border: 1px solid rgba(255,255,255,0.05);'>💡 <b>API status:</b> Permanent key successfully loaded from environment.</div>", unsafe_allow_html=True)

# Main Processing Flow
if uploaded_file:
    # Build unique file identifier using constants
    file_id = f"{uploaded_file.name}_{uploaded_file.size}_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
    
    if st.session_state.pdf_name != file_id:
        with st.spinner("Analyzing document structure & indexing content..."):
            try:
                # 1. Extract documents
                raw_docs = extract_documents_from_pdf(uploaded_file)
                
                # 2. Chunk documents
                chunks = chunk_documents(raw_docs, CHUNK_SIZE, CHUNK_OVERLAP)
                
                # 3. Create vector store
                vector_store = create_vector_store(chunks, api_key=api_key)
                
                # Cache references in state
                st.session_state.vector_store = vector_store
                st.session_state.rag_chain = get_rag_chain(vector_store, api_key=api_key)
                st.session_state.pdf_name = file_id
                st.session_state.chat_history = []  # Reset conversation for new document
                st.session_state.stats = {
                    "pages": len(raw_docs),
                    "chunks": len(chunks)
                }
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

# Header Content
st.markdown("<h1 class='main-title'>📄 PDF Q&A Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Extract insights, analyze structures, and get answers from your documents instantly</p>", unsafe_allow_html=True)

# If PDF is loaded, display layout
if st.session_state.vector_store:
    # Display Stats Row
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.stats['pages']}</div>
            <div class="stat-label">Total PDF Pages</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{st.session_state.stats['chunks']}</div>
            <div class="stat-label">Indexed Text Chunks</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<h3 style='margin-top: 30px; margin-bottom: 10px; font-family: Outfit;'>🧠 Conversation Workspace</h3>", unsafe_allow_html=True)
    
    # Query Form
    with st.form("query_form", clear_on_submit=True):
        user_query = st.text_input(
            "Enter your question about the uploaded document:",
            placeholder="e.g., What is the summary of section 2? What are the key deliverables?"
        )
        submit_button = st.form_submit_button("Ask Assistant")
        
    if submit_button and user_query:
        with st.spinner("Performing semantic search & generating answer..."):
            try:
                result = query_rag(st.session_state.rag_chain, user_query)
                
                # Prepend to chat history (newest first)
                st.session_state.chat_history.insert(0, {
                    "question": user_query,
                    "answer": result["answer"],
                    "context": result["context"]
                })
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                
    # Render Q&A chat history in professional bubbles
    if st.session_state.chat_history:
        for idx, chat in enumerate(st.session_state.chat_history):
            # User Message
            st.markdown(f"<div class='chat-label-user'>User Query</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble-user'>{chat['question']}</div>", unsafe_allow_html=True)
            
            # Assistant Message
            st.markdown(f"<div class='chat-label-assistant'>AI Assistant</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble-assistant'>{chat['answer']}</div>", unsafe_allow_html=True)
            
            # Sources Expander
            with st.expander("🔍 View Grounded Sources"):
                for s_idx, doc in enumerate(chat["context"]):
                    page = doc.metadata.get("page", "Unknown")
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-badge">Source Passage #{s_idx + 1} — Page {page}</div>
                        <div>{doc.page_content}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.divider()
else:
    st.markdown("""
    <div style="background-color: #1e293b; border-left: 4px solid #2563eb; padding: 20px; border-radius: 8px; margin-top: 20px; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155;">
        <h4 style="margin-top:0; color: #3b82f6; font-family: Outfit;">Welcome to Document Intelligence!</h4>
        <p style="margin: 0; color: #cbd5e1; font-size: 14.5px; line-height: 1.5;">
            To begin querying your documents, please <b>upload a PDF file in the sidebar</b>. 
            The pipeline will extract the text, partition it into semantic chunks, generate vector embeddings, 
            and stand up an in-memory vector database using ChromaDB automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)
