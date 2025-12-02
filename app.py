import os
import streamlit as st
import datetime
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import sqlite3
import bcrypt
import random
import string
import time
from io import BytesIO
import re 
import streamlit.components.v1 as components
from mutagen.mp3 import MP3
import base64

# --- IMPORTS ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from gtts import gTTS
import speech_recognition as sr

# --- URL MAPPINGS ---
PAGE_TO_URL = {
    "employee_workspace": "agento",
    "ai_call_mode": "call-mode",
    "admin_dashboard": "dashboard",
    "user_profile": "profile",
    "auth": "login"
}
URL_TO_PAGE = {v: k for k, v in PAGE_TO_URL.items()}

# --- SESSION STATE INIT ---
if "page" not in st.session_state: st.session_state.page = "auth"
if "user" not in st.session_state: st.session_state.user = None
if "auth_status" not in st.session_state: st.session_state.auth_status = False
if "theme" not in st.session_state: st.session_state.theme = "dark" 
if "popup_diagram" not in st.session_state: st.session_state.popup_diagram = None

# --- CONFIGURATION ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
load_dotenv()

# Determine layout based on page
page_layout = "wide" if st.session_state.page in ["ai_call_mode", "admin_dashboard", "auth"] else "centered"

st.set_page_config(page_title="AGENTO: Enterprise", layout=page_layout, initial_sidebar_state="expanded")

# --- STYLING ---
def get_theme_css():
    themes = {
        "dark": {
            "--background-color": "#161B22",
            "--bg-gradient-start": "#0D1117",
            "--bg-gradient-end": "#161B22",
            "--primary-text-color": "#E6EDF3",
            "--secondary-text-color": "#8B949E",
            "--accent-color": "#58A6FF",
            "--accent-color-hover": "#79C0FF",
            "--card-background-color": "rgba(33, 39, 48, 0.7)",
            "--border-color": "rgba(139, 148, 158, 0.3)",
            "--glow-color": "rgba(88, 166, 255, 0.5)"
        },
    }
    theme = themes[st.session_state.theme]
    
    return f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <style>
        @keyframes gradientAnimation {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
        :root {{ --background-color: {theme['--background-color']}; --primary-text-color: {theme['--primary-text-color']}; --accent-color: {theme['--accent-color']}; --card-background-color: {theme['--card-background-color']}; }}
        html, body, [class*="st-"] {{ font-family: 'Poppins', sans-serif; color: var(--primary-text-color); }}
        .main {{ background: linear-gradient(-45deg, {theme['--bg-gradient-start']}, {theme['--bg-gradient-end']}); background-size: 400% 400%; animation: gradientAnimation 15s ease infinite; }}
        
        /* SIDEBAR STYLING */
        [data-testid="stSidebar"] {{ 
            background-color: var(--card-background-color); 
            backdrop-filter: blur(10px); 
            border-right: 1px solid {theme['--border-color']}; 
        }}
        
        .stButton > button {{ border: 1px solid var(--accent-color); background-color: transparent; color: var(--accent-color) !important; border-radius: 8px; }}
        .stButton > button:hover {{ background-color: var(--accent-color); color: white !important; box-shadow: 0 0 15px {theme['--glow-color']}; }}
        .stTextInput > div > div > input, .stSelectbox > div > div {{ background-color: var(--card-background-color); color: var(--primary-text-color); border: 1px solid {theme['--border-color']}; border-radius: 8px; }}
        [data-testid="stChatMessage"] {{ background: var(--card-background-color); border: 1px solid {theme['--border-color']}; border-radius: 12px; }}
    </style>
    """

# --- DATABASE ---
# --- DATABASE ---
@st.cache_resource
def get_db():
    conn = sqlite3.connect('agento.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    
    # Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, password BLOB, role TEXT, company_id TEXT, company_name TEXT)''')
    
    # Companies Table
    c.execute('''CREATE TABLE IF NOT EXISTS companies
                 (company_id TEXT PRIMARY KEY, name TEXT, created_at TIMESTAMP, admin TEXT)''')
    
    # Documents Table
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, company_id TEXT, filename TEXT, category TEXT, 
                  uploaded_by TEXT, upload_date TIMESTAMP, full_text TEXT)''')
    
    # Chat History Table
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, company_id TEXT, user TEXT, query TEXT, 
                  response TEXT, timestamp TIMESTAMP, mode TEXT)''')
    
    conn.commit()

init_db()

# --- ROUTING HELPER ---
def navigate_to(page):
    url_path = PAGE_TO_URL.get(page, page)
    st.query_params["page"] = url_path
    st.session_state.page = page
    st.rerun()

# --- AUTH LOGIC ---
def generate_id(name):
    prefix = name[:3].upper()
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{prefix}_{suffix}"

def register_company(conn, name, email, password):
    c = conn.cursor()
    if c.execute("SELECT email FROM users WHERE email = ?", (email,)).fetchone(): return False, "Email exists."
    
    cid = generate_id(name)
    try:
        c.execute("INSERT INTO companies (company_id, name, created_at, admin) VALUES (?, ?, ?, ?)",
                  (cid, name, datetime.datetime.now().isoformat(), email))
        
        c.execute("INSERT INTO users (email, password, role, company_id, company_name) VALUES (?, ?, ?, ?, ?)",
                  (email, bcrypt.hashpw(password.encode(), bcrypt.gensalt()), "admin", cid, name))
        conn.commit()
        return True, cid
    except Exception as e:
        return False, str(e)

def join_company(conn, email, password, cid):
    c = conn.cursor()
    company = c.execute("SELECT * FROM companies WHERE company_id = ?", (cid,)).fetchone()
    if not company: return False, "Invalid ID."
    
    if c.execute("SELECT email FROM users WHERE email = ?", (email,)).fetchone(): return False, "Email exists."
    
    try:
        c.execute("INSERT INTO users (email, password, role, company_id, company_name) VALUES (?, ?, ?, ?, ?)",
                  (email, bcrypt.hashpw(password.encode(), bcrypt.gensalt()), "employee", cid, company["name"]))
        conn.commit()
        return True, "Joined!"
    except Exception as e:
        return False, str(e)

def login(conn, email, password):
    c = conn.cursor()
    user = c.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if user and bcrypt.checkpw(password.encode(), user["password"]):
        st.session_state.user = dict(user) # Convert Row to dict
        st.session_state.auth_status = True
        return True
    return False

# --- MERMAID RENDERER ---
def render_mermaid(code, height=400):
    html_code = f"""
    <div id="mermaid-container" style="width: 100%; overflow-x: auto;">
        <div class="mermaid">
            {code}
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'dark',
            securityLevel: 'loose',
            suppressErrorRendering: true,
        }});
    </script>
    """
    return components.html(html_code, height=height, scrolling=True)

def parse_and_display_response(response_text):
    pattern = r"```mermaid(.*?)```"
    matches = re.split(pattern, response_text, flags=re.DOTALL | re.IGNORECASE)
    if len(matches) > 1:
        for i, part in enumerate(matches):
            if i % 2 == 0:
                if part.strip(): st.markdown(part)
            else:
                clean_code = part.strip().replace("`", "")
                if clean_code:
                    st.caption("📊 Process Flow")
                    render_mermaid(clean_code)
    else:
        st.markdown(response_text)

# --- RAG LOGIC ---
def ingest_file(conn, pdf, category, user):
    text = "".join([p.extract_text() for p in PdfReader(pdf).pages])
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(text)
    
    c = conn.cursor()
    c.execute("INSERT INTO documents (company_id, filename, category, uploaded_by, upload_date, full_text) VALUES (?, ?, ?, ?, ?, ?)",
              (user["company_id"], pdf.name, category, user["email"], datetime.datetime.now().isoformat(), text))
    conn.commit()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection_name = f"collection_{user['company_id']}"
    
    try:
        QdrantVectorStore.from_texts(
            texts=chunks,
            embedding=embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=collection_name,
            force_recreate=False
        )
    except Exception as e: st.error(f"Qdrant Error: {e}")

def chat_with_jarvis(user, query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection_name = f"collection_{user['company_id']}"
    
    try:
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        template = """You are AGENTO, a professional internal knowledge assistant for {company_name}.
        GUIDELINES:
        1. **Tone:** Professional, concise, and direct.
        2. **Accuracy:** Answer ONLY based on the Context.
        VISUALIZATION RULES:
        1. If the answer describes a workflow/process, generate a Mermaid.js diagram.
        2. **Syntax Safety:** No brackets () in node labels. Use A[Start] --> B[Process].
        3. Place mermaid code at the end.

        Context: {context}
        Question: {question}
        Professional Answer:"""
        
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.2)
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
                "question": RunnablePassthrough(),
                "company_name": lambda x: user['company_name']
            }
            | prompt_template 
            | model 
            | StrOutputParser()
        )
        return chain.invoke(query)
    except Exception:
        return "⚠️ I am unable to access the company knowledge base at this moment."

# --- AUDIO HELPERS ---
def speak_text(text):
    try:
        clean_text = re.sub(r"```.*?```", " I have displayed the diagram.", text, flags=re.DOTALL)
        tts = gTTS(text=clean_text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        return fp
    except: return None

def transcribe_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        with open("temp_input.wav", "wb") as f: f.write(audio_bytes.read())
        with sr.AudioFile("temp_input.wav") as source:
            return r.recognize_google(r.record(source))
    except: return None

def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes.read()).decode()
    md = f"""<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
    st.markdown(md, unsafe_allow_html=True)

# --- SIDEBAR ---
def render_sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.title(f"🏢 {user['company_name']}")
        st.markdown(f"**Agent:** {user['email']}")
        st.markdown("---")
        pages = {
            "employee_workspace": {"label": "Go to Chat", "icon": "💬"},
            "ai_call_mode": {"label": "AI Call Mode", "icon": "📞"},
            "admin_dashboard": {"label": "Admin Dashboard", "icon": "📊", "admin_only": True},
            "user_profile": {"label": "User Profile", "icon": "👤"}
        }
        for page_id, page_info in pages.items():
            if page_info.get("admin_only") and user['role'] != 'admin': continue
            if st.session_state.get('page') != page_id:
                if st.button(page_info["label"], width='stretch'): navigate_to(page_id)
        st.markdown("---")
        if st.button("Logout", width='stretch'):
            for key in st.session_state.keys(): del st.session_state[key]
            st.session_state.page = "auth"
            st.query_params.clear()
            st.rerun()

# --- PAGES ---
def page_landing():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 style='text-align: center; margin-top: 50px;'>Welcome Back!</h1>", unsafe_allow_html=True)
        if st.button("Go to Workspace", width='stretch'): navigate_to("employee_workspace")
        if st.button("Logout", width='stretch'):
            st.session_state.auth_status = False; st.session_state.page = "auth"; st.rerun()

def page_auth(db):
    st.markdown("<h1 style='text-align: center; margin-bottom: 50px;'><i class='bi bi-shield-lock'></i> AGENTO: Secure Access</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_register, tab_join = st.tabs(["Login", "Register Company", "Join Team"])
        with tab_login:
            email = st.text_input("Email", key="l_email")
            pwd = st.text_input("Password", type="password", key="l_pwd")
            if st.button("Login"):
                if login(db, email, pwd): navigate_to("admin_dashboard" if st.session_state.user["role"] == "admin" else "employee_workspace")
                else: st.error("Invalid Credentials")
        with tab_register:
            c_name = st.text_input("Company Name")
            a_email = st.text_input("Admin Email")
            a_pwd = st.text_input("Admin Password", type="password")
            if st.button("Register Company"):
                success, msg = register_company(db, c_name, a_email, a_pwd)
                if success: 
                    st.success("Company Registered Successfully!")
                    st.markdown("**Workspace ID (Copy this):**")
                    st.code(msg, language="text")
                    st.info("Please switch to the 'Login' tab to sign in.")
                else: st.error(msg)
        with tab_join:
            j_id = st.text_input("Workspace ID")
            j_email = st.text_input("Your Email")
            j_pwd = st.text_input("Password", type="password")
            if st.button("Join Team"):
                success, msg = join_company(db, j_email, j_pwd, j_id)
                if success: st.success(msg)
                else: st.error(msg)

def page_admin_dashboard(conn):
    user = st.session_state.user
    render_sidebar()
    st.title("Company Command Center")
    col1, col2, col3 = st.columns(3)
    
    c = conn.cursor()
    doc_count = c.execute("SELECT COUNT(*) FROM documents WHERE company_id = ?", (user["company_id"],)).fetchone()[0]
    user_count = c.execute("SELECT COUNT(*) FROM users WHERE company_id = ?", (user["company_id"],)).fetchone()[0]
    col1.metric("Total Documents", doc_count); col2.metric("Team Members", user_count); col3.metric("System Status", "Online")
    st.markdown("---")
    col_upload, col_list = st.columns([1, 2])
    with col_upload:
        st.subheader("Upload Knowledge")
        predefined_cats = ["HR", "Engineering", "Sales", "Legal", "General"]
        selected = st.selectbox("Category", predefined_cats + ["➕ Create New"])
        cat = st.text_input("New Category:").strip() if selected == "➕ Create New" else selected
        files = st.file_uploader("Select PDF", type="pdf", accept_multiple_files=True)
        if st.button("Index Documents"):
            if files and cat:
                with st.spinner("Pushing to Qdrant..."):
                    for f in files: ingest_file(conn, f, cat, user)
                st.success("Indexed!"); time.sleep(1); st.rerun()
    with col_list:
        st.subheader("Recent Uploads")
        c = conn.cursor()
        docs = c.execute("SELECT filename, category, upload_date FROM documents WHERE company_id = ? ORDER BY upload_date DESC LIMIT 10", (user["company_id"],)).fetchall()
        docs = [dict(row) for row in docs]
        if docs: st.dataframe(pd.DataFrame(docs), width='stretch')

def page_employee_workspace(conn):
    user = st.session_state.user
    render_sidebar()
    st.markdown("<h1 style='text-align: center; padding-top: 20px;'>AGENTO AI Assistant</h1>", unsafe_allow_html=True)
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): 
            if msg["role"] == "assistant": parse_and_display_response(msg["content"])
            else: st.markdown(msg["content"])
    if prompt := st.chat_input("How can I help you?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            response = chat_with_jarvis(user, prompt)
        with st.chat_message("assistant"): parse_and_display_response(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        c = conn.cursor()
        c.execute("INSERT INTO chat_history (company_id, user, query, response, timestamp, mode) VALUES (?, ?, ?, ?, ?, ?)",
                  (user["company_id"], user["email"], prompt, response, datetime.datetime.now().isoformat(), "text"))
        conn.commit()
# ==============
# ============================
# 📞 PAGE: AI CALL MODE (Fixed Sidebar CSS)
# ==========================================
def page_ai_call_mode(conn):
    user = st.session_state.user
    render_sidebar()
    
    if "audio_key" not in st.session_state: st.session_state.audio_key = 0
    if "active_diagram" not in st.session_state: st.session_state.active_diagram = None

    # --- ADVANCED CSS (Modal) ---
    st.markdown("""
    <style>
        .main { background: #000000 !important; } 
        .block-container { padding-top: 2rem; } 
        audio { display: none !important; }
        
        /* 1. THE MODAL CONTAINER */
        div[data-testid="stVerticalBlock"]:has(div.modal-marker) {
            position: fixed;
            top: 50%; left: 50%; transform: translate(-50%, -50%);
            width: 70vw; max-height: 80vh;
            background-color: #0D1117; border: 1px solid #58A6FF;
            border-radius: 12px; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.9);
            z-index: 99999; padding: 20px; overflow-y: auto;
        }

        /* 2. CLOSE BUTTON STYLING (SCOPED STRICTLY TO MODAL) */
        /* This prevents it from breaking the Sidebar buttons */
        div[data-testid="stVerticalBlock"]:has(div.modal-marker) button {
            float: right; 
            border: none; 
            background: transparent; 
            color: #ff4b4b; 
            font-size: 20px;
        }
        div[data-testid="stVerticalBlock"]:has(div.modal-marker) button:hover {
            color: #ff0000;
            background: rgba(255, 0, 0, 0.1);
            box-shadow: none;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- MODAL LOGIC ---
    modal_placeholder = st.empty()

    if st.session_state.active_diagram:
        with modal_placeholder.container():
            # The Marker allows CSS to find this specific box
            st.markdown('<div class="modal-marker"></div>', unsafe_allow_html=True)
            
            c_head, c_close = st.columns([9, 1])
            with c_head: st.markdown("### 🧠 Workflow Visualization")
            with c_close: 
                if st.button("✕", key="close_modal"):
                    st.session_state.active_diagram = None
                    st.rerun()
            st.markdown("---")
            render_mermaid(st.session_state.active_diagram, height=450)

    # --- MAIN SCREEN ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
    <style>
        .gradient-text {{
            /* Gradient from Pastel Pink/Lavender to Pastel Blue */
            background: linear-gradient(to right, #E0C3FC, #8EC5FC);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3em;
            font-weight: bold;
            display: inline-block;
        }}
    </style>
    <div style='text-align: center; margin-bottom: 2rem;'>
        <span class="gradient-text">AGENTO VOICE ASSISTANT</span>
    </div>
    """, unsafe_allow_html=True)
        st.image("public/logo.gif", width='stretch')
        status = "🟢 LISTENING..." if not st.session_state.get("processing_voice") else "🟣 PROCESSING..."
        st.markdown(f"<p style='text-align: center; color: #888; letter-spacing: 2px;'>{status}</p>", unsafe_allow_html=True)
    
    st.write("---") 
    _, c2, _ = st.columns([2,1,2])
    with c2: 
        audio_value = st.audio_input("Tap to Speak", label_visibility="collapsed", key=f"audio_input_{st.session_state.audio_key}")

    # --- PROCESSING LOOP ---
    if audio_value:
        st.session_state.processing_voice = True
        user_text = transcribe_audio(audio_value)
        
        if user_text:
            ai_response = chat_with_jarvis(user, user_text)
            
            # --- INSTANT DIAGRAM POPUP ---
            pattern = r"```mermaid(.*?)```"
            matches = re.split(pattern, ai_response, flags=re.DOTALL | re.IGNORECASE)
            
            if len(matches) > 1:
                diagram_code = matches[1].strip()
                st.session_state.active_diagram = diagram_code
                
                with modal_placeholder.container():
                    st.markdown('<div class="modal-marker"></div>', unsafe_allow_html=True)
                    c_head, c_close = st.columns([9, 1])
                    with c_head: st.markdown("### 🧠 Workflow Visualization")
                    with c_close: st.button("✕", disabled=True) 
                    st.markdown("---")
                    render_mermaid(diagram_code, height=450)
            
            # --- AUDIO ---
            audio_fp = speak_text(ai_response)
            if audio_fp:
                audio_fp.seek(0); audio_meta = MP3(audio_fp); duration = audio_meta.info.length 
                audio_fp.seek(0); autoplay_audio(audio_fp)
                audio_fp.seek(0); autoplay_audio(audio_fp)
                c = conn.cursor()
                c.execute("INSERT INTO chat_history (company_id, user, query, response, timestamp, mode) VALUES (?, ?, ?, ?, ?, ?)",
                          (user["company_id"], user["email"], user_text, ai_response, datetime.datetime.now().isoformat(), "voice_call"))
                conn.commit()
                time.sleep(duration + 1)
                st.session_state.audio_key += 1
                st.rerun()
        else: st.warning("Could not understand audio.")
        st.session_state.processing_voice = False
# ==========================================
# 👤 PAGE: USER PROFILE (Dynamic)
# ==========================================
def page_user_profile(conn):
    user = st.session_state.user
    render_sidebar()
    
    st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>User Profile</h1>", unsafe_allow_html=True)
    
    # Fetch fresh company data (incase stats changed)
    c = conn.cursor()
    company_data = c.execute("SELECT * FROM companies WHERE company_id = ?", (user["company_id"],)).fetchone()
    company_data = dict(company_data) if company_data else {}
    
    # --- LAYOUT GRID ---
    col1, col2 = st.columns([1, 1], gap="large")
    
    # --- LEFT COLUMN: PERSONAL INFO (Everyone sees this) ---
    with col1:
        st.markdown("### 👤 Personal Identity")
        st.markdown(f"""
        <div style="
            background-color: rgba(22, 27, 34, 0.8); 
            padding: 25px; 
            border-radius: 15px; 
            border: 1px solid #30363d;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <p style="font-size: 1.1rem; margin-bottom: 10px;"><strong>Email:</strong> <span style="color: #58A6FF;">{user['email']}</span></p>
            <p style="font-size: 1.1rem; margin-bottom: 10px;"><strong>Role:</strong> {user['role'].upper()}</p>
            <p style="font-size: 1.1rem; margin-bottom: 0px;"><strong>Status:</strong> <span style="color: #238636;">● Active</span></p>
        </div>
        """, unsafe_allow_html=True)

    # --- RIGHT COLUMN: COMPANY INFO (Admin gets extra details) ---
    with col2:
        if user['role'] == 'admin':
            st.markdown("### 🏢 Company Registry")
            
            # Format Date
            created_at = company_data.get('created_at')
            if isinstance(created_at, str):
                try: created_at = datetime.datetime.fromisoformat(created_at)
                except: created_at = datetime.datetime.now()
            elif not isinstance(created_at, datetime.datetime):
                created_at = datetime.datetime.now()
                
            reg_date = created_at.strftime("%B %d, %Y")
            
            st.markdown(f"""
            <div style="
                background-color: rgba(22, 27, 34, 0.8); 
                padding: 25px; 
                border-radius: 15px; 
                border: 1px solid #58A6FF; /* Blue border for Admin importance */
                box-shadow: 0 0 20px rgba(88, 166, 255, 0.1);
            ">
                <p style="font-size: 1.2rem; font-weight: bold; color: #E6EDF3;">{company_data['name']}</p>
                <hr style="border-color: #30363d; margin: 10px 0;">
                <p style="margin-bottom: 8px;"><strong>Registered On:</strong> {reg_date}</p>
                <p style="margin-bottom: 8px;"><strong>Admin:</strong> {company_data['admin']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer
            st.markdown("#### 🔑 Workspace Access Key")
            st.caption("Share this ID with your employees so they can join your workspace:")
            # Use st.code so it's easy to copy-paste
            st.code(user['company_id'], language="text")
            
        else:
            # EMPLOYEE VIEW
            st.markdown("### 🏢 Organization")
            st.markdown(f"""
            <div style="
                background-color: rgba(22, 27, 34, 0.8); 
                padding: 25px; 
                border-radius: 15px; 
                border: 1px solid #30363d;
            ">
                <p style="font-size: 1.1rem;">You are a member of:</p>
                <h2 style="color: #58A6FF; margin: 0;">{user['company_name']}</h2>
                <p style="margin-top: 10px; color: #888;">Workspace ID: {user['company_id']}</p>
            </div>
            """, unsafe_allow_html=True)

    # --- BOTTOM SECTION: ACCOUNT ACTIONS ---
    st.write("---")
    c_left, c_right = st.columns([6, 1])
    with c_right:
        if st.button("🔒 Secure Logout", type="primary", width='stretch'):
            for key in st.session_state.keys(): del st.session_state[key]
            st.session_state.page = "auth"
            st.rerun()
# --- MAIN ROUTER ---
def main():
    conn = get_db()
    if conn is None: st.error("Database Connection Failed."); return
    st.markdown(get_theme_css(), unsafe_allow_html=True)
    query_params = st.query_params.to_dict()
    page_in_url = query_params.get("page")
    
    if not st.session_state.auth_status: page_auth(conn)
    else:
        page_to_render = URL_TO_PAGE.get(page_in_url, 'landing')
        if page_to_render == 'landing': page_landing()
        elif page_to_render == 'auth': page_landing()
        elif page_to_render == 'admin_dashboard' and st.session_state.user.get('role') == 'admin': page_admin_dashboard(conn)
        elif page_to_render == 'employee_workspace': page_employee_workspace(conn)
        elif page_to_render == 'ai_call_mode': page_ai_call_mode(conn)
        elif page_to_render == 'user_profile': page_user_profile(conn)
        else: page_landing()

if __name__ == "__main__":
    main()


