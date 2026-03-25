import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Page configuration
st.set_page_config(
    page_title="AI Quiz Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PWA support
st.markdown("""
    <link rel="manifest" href="/app/static/manifest.json">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="QuizAI">
    <meta name="theme-color" content="#4CAF50">
    <link rel="apple-touch-icon" href="/app/static/icon-192.png">
    <script>
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/app/static/sw.js');
      }
    </script>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 1.1rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .quiz-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main { padding: 0.5rem; }
        .stButton>button { font-size: 1rem; padding: 0.75rem; }
        h1 { font-size: 1.5rem !important; }
        .block-container { padding: 1rem 0.5rem !important; }
    }
    </style>
""", unsafe_allow_html=True)

def load_document(file_path):
    """Load document based on file extension"""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext in ['.ppt', '.pptx']:
        loader = UnstructuredPowerPointLoader(file_path)
    elif ext in ['.doc', '.docx']:
        loader = UnstructuredWordDocumentLoader(file_path)
    elif ext == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    return loader.load()

@st.cache_resource
def get_embeddings():
    """Cache embeddings model - loads faster on subsequent uses"""
    return HuggingFaceEmbeddings(model_kwargs={'device': 'cpu'})

@st.cache_resource
def get_llm():
    """Cache LLM model"""
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, groq_api_key=api_key)

def parse_quiz_response(response, question_type):
    """Parse AI response into structured quiz data"""
    import re
    questions = []
    lines = response.strip().split('\n')

    current_q = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('Q') and ':' in line:
            if current_q:
                questions.append(current_q)
            current_q = {'question': line.split(':', 1)[1].strip(), 'options': []}

        elif question_type == "Multiple Choice":
            # Handle options all on one line: "A) opt B) opt C) opt D) opt"
            if re.search(r'[A-D]\)', line):
                # Split by A) B) C) D) pattern
                parts = re.split(r'(?=[A-D]\))', line)
                for part in parts:
                    part = part.strip()
                    if part and re.match(r'^[A-D]\)', part):
                        current_q.setdefault('options', []).append(part)
            elif line.startswith('Answer:'):
                current_q['answer'] = line.split(':', 1)[1].strip()

        else:  # True/False
            if line.startswith('Answer:'):
                current_q['answer'] = line.split(':', 1)[1].strip()

    if current_q:
        questions.append(current_q)

    return questions

def display_interactive_quiz(questions, question_type):
    """Display one question at a time for smooth mobile experience"""
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'current_q' not in st.session_state:
        st.session_state.current_q = 0

    total = len(questions)

    if st.session_state.show_results:
        # Results screen
        correct = sum(
            1 for i, q in enumerate(questions)
            if st.session_state.user_answers.get(i) and
            st.session_state.user_answers.get(i, '')[0] == q.get('answer', '').strip()[0]
        )
        score = (correct / total) * 100
        st.markdown(f"### 🎯 Score: {correct}/{total} ({score:.0f}%)")
        if score >= 80:
            st.success("Excellent work! 🏆")
        elif score >= 60:
            st.info("Good job! Keep practicing. 👍")
        else:
            st.warning("Keep studying, you'll get there! 💪")

        for i, q in enumerate(questions):
            ans = st.session_state.user_answers.get(i, '')
            correct_ans = q.get('answer', '').strip()
            is_correct = ans and ans[0] == correct_ans[0]
            icon = "✅" if is_correct else "❌"
            st.markdown(f"{icon} **Q{i+1}:** {q['question']}")
            if not is_correct:
                st.caption(f"Correct answer: {correct_ans}")

        st.markdown("---")
        if st.button("🔄 Try Again", use_container_width=True):
            st.session_state.user_answers = {}
            st.session_state.show_results = False
            st.session_state.current_q = 0
            st.rerun()
        return

    # Single question view
    i = st.session_state.current_q
    q = questions[i]

    # Progress
    st.progress((i) / total)
    st.caption(f"Question {i+1} of {total}")
    st.markdown(f"### {q['question']}")
    st.markdown("---")

    if question_type == "Multiple Choice":
        options = q.get('options', [])
        if options:
            prev = st.session_state.user_answers.get(i)
            prev_index = next((j for j, o in enumerate(options) if o.startswith(prev)), None) if prev else None
            choice = st.radio(
                "Select your answer:",
                options,
                key=f"q_{i}",
                index=prev_index,
                horizontal=False
            )
            st.session_state.user_answers[i] = choice[0] if choice else None
    else:
        prev = st.session_state.user_answers.get(i)
        prev_index = ["True", "False"].index(prev) if prev in ["True", "False"] else None
        choice = st.radio(
            "Select your answer:",
            ["True", "False"],
            key=f"q_{i}",
            index=prev_index,
            horizontal=False
        )
        st.session_state.user_answers[i] = choice

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if i > 0:
            if st.button("← Prev", use_container_width=True):
                st.session_state.current_q -= 1
                st.rerun()
    with col2:
        if i < total - 1:
            if st.button("Next →", use_container_width=True):
                st.session_state.current_q += 1
                st.rerun()
        else:
            if st.button("📝 Submit", use_container_width=True):
                st.session_state.show_results = True
                st.rerun()

def generate_quiz(file_path, num_questions, question_type):
    """Generate quiz questions from document - optimized for speed"""
    try:
        # Load document
        with st.spinner("📄 Loading document..."):
            docs = load_document(file_path)
            st.success(f"✅ Loaded {len(docs)} page(s)")
        
        # Get LLM (cached)
        llm = get_llm()
        
        # Extract text directly - skip embeddings for speed
        # Just use first few pages or limit text length
        all_text = "\n\n".join([doc.page_content for doc in docs[:5]])  # Only first 5 pages
        
        # Limit context to 3000 characters for faster processing
        context = all_text[:3000] if len(all_text) > 3000 else all_text
        
        # Create shorter, more efficient prompt
        if question_type == "Multiple Choice":
            prompt = f"""Create {num_questions} multiple choice questions from this text. Be concise.

Text: {context}

Format:
Q1: [question]
A) [option] B) [option] C) [option] D) [option]
Answer: [letter]

Generate all {num_questions} questions now:"""
        else:  # True/False
            prompt = f"""Create {num_questions} true/false questions from this text. Be concise.

Text: {context}

Format:
Q1: [statement]
Answer: True/False

Generate all {num_questions} questions now:"""
        
        # Generate questions with progress
        with st.spinner(f"✨ Generating {num_questions} questions... (this may take 30-60 seconds)"):
            response = llm.invoke(prompt)
            # ChatGroq returns an AIMessage, extract content
            result = response.content if hasattr(response, 'content') else str(response)
        
        return result, None
    
    except Exception as e:
        if "invalid_api_key" in str(e).lower() or "authentication" in str(e).lower():
            error_msg = """
            ❌ **Invalid Groq API Key!**
            
            Please make sure your GROQ_API_KEY is set correctly in Streamlit secrets.
            Get your free key at https://console.groq.com
            """
            return None, error_msg
        else:
            return None, f"❌ Error: {str(e)}"

def generate_pdf(quiz_text, title):
    """Generate a PDF from quiz text"""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Quiz: {title}", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", size=11)
    for line in quiz_text.split('\n'):
        line = line.strip()
        if not line:
            pdf.ln(4)
            continue
        # Encode to latin-1 safely
        safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, safe_line)
    return bytes(pdf.output())


def show_preview():
    """Show mobile phone simulator"""
    st.title("📱 Mobile Preview")
    st.markdown("Interactive phone simulator — see how the app looks on mobile")
    try:
        with open("static/preview.html", "r") as f:
            html_content = f.read()
        # Strip the outer html/body to embed cleanly, inject in iframe-like component
        st.components.v1.html(html_content, height=900, scrolling=False)
    except FileNotFoundError:
        st.error("Preview file not found.")


def main():
    # Sidebar
    with st.sidebar:
        page = st.radio("Navigate", ["🏠 Quiz Generator", "📱 Mobile Preview"], label_visibility="collapsed")
        st.divider()

    if page == "📱 Mobile Preview":
        show_preview()
        return

    # Header
    st.title("📝 AI Quiz Generator")
    st.markdown("Generate quiz questions from any document using AI")

    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'ppt', 'pptx', 'doc', 'docx', 'txt'],
            help="Supported formats: PDF, PPT, DOCX, TXT"
        )
        
        st.divider()
        
        # Number of questions
        num_questions = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=30,
            value=5,
            help="Select how many questions to generate"
        )
        
        # Question type
        question_type = st.selectbox(
            "Question Type",
            ["Multiple Choice", "True/False"],
            help="Choose the type of questions"
        )
        
        st.divider()
        
        # Generate button
        generate_btn = st.button("🚀 Generate Quiz", use_container_width=True)
        
        st.divider()
        
        # Info
        with st.expander("ℹ️ About"):
            st.markdown("""
            **AI Quiz Generator** uses advanced AI to create quiz questions from your documents.
            
            **Supported Formats:**
            - PDF (.pdf)
            - PowerPoint (.ppt, .pptx)
            - Word (.doc, .docx)
            - Text (.txt)
            
            **Requirements:**
            - Ollama installed and running
            - llama3 model downloaded
            """)
    
    # Main content
    if not uploaded_file:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 3rem;'>
                <h2>👋 Welcome!</h2>
                <p style='font-size: 1.2rem; color: #666;'>
                    Upload a document to get started
                </p>
                <p style='color: #999;'>
                    Supported formats: PDF, PPT, DOCX, TXT
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Display file info
        st.info(f"📄 **File:** {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
        
        if generate_btn:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Generate quiz
                result, error = generate_quiz(tmp_file_path, num_questions, question_type)
                
                if error:
                    st.error(error)
                else:
                    st.success("✅ Quiz generated successfully!")
                    
                    # Parse and display interactive quiz
                    questions = parse_quiz_response(result, question_type)
                    
                    if questions:
                        # Store in session state
                        st.session_state.quiz_questions = questions
                        st.session_state.quiz_type = question_type
                        st.session_state.quiz_raw = result
                    else:
                        st.warning("Could not parse quiz. Showing raw output:")
                        st.text(result)
            
            finally:
                # Clean up temp file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        # Display interactive quiz if available
        if 'quiz_questions' in st.session_state and st.session_state.quiz_questions:
            display_interactive_quiz(st.session_state.quiz_questions, st.session_state.quiz_type)
            
            # Download button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                pdf_data = generate_pdf(st.session_state.quiz_raw, Path(uploaded_file.name).stem)
                st.download_button(
                    label="📥 Download Quiz (PDF)",
                    data=pdf_data,
                    file_name=f"quiz_{Path(uploaded_file.name).stem}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
