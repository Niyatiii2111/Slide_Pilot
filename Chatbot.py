import streamlit as st
from datetime import datetime
from PyPDF2 import PdfReader

# LangChain / RAG Components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="centered", page_title="PDF AI Chatbot")
st.title("📄 PDF Intellectual Assistant")
st.markdown("---")

# ---------------- SESSION STATE ----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- PDF LOGIC ----------------
def extract_text_from_pdfs(pdf_files):
    """Extracts raw text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_files:
        pdf.seek(0)
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def build_vector_store(text):
    """Chunks text and creates a searchable FAISS vector database."""
    # 1. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)

    # 2. Embeddings (Small, fast model)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Create FAISS index
    return FAISS.from_texts(chunks, embeddings)

def get_ai_response(user_query, api_key):
    """Retrieves context and generates an answer using Groq."""
    # 1. Search for relevant chunks
    docs = st.session_state.vector_store.similarity_search(user_query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 2. Setup Chain
    template = """
    You are a helpful assistant. Use the following context to answer the user's question accurately.
    If the answer is not contained within the context, strictly say: 
    "I'm sorry, that information is not available in the uploaded documents."

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )

    chain = prompt | llm
    
    # 3. Invoke
    return chain.invoke({"context": context, "question": user_query})

# ---------------- SIDEBAR: CONFIG ----------------
with st.sidebar:
    st.header("🔑 Authentication")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    
    st.header("📂 Data Source")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("Initialize Knowledge Base"):
        if not groq_api_key:
            st.error("Please provide a Groq API Key first.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Analyzing documents..."):
                full_text = extract_text_from_pdfs(uploaded_files)
                st.session_state.vector_store = build_vector_store(full_text)
                st.success("Chatbot is ready!")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------- MAIN CHAT INTERFACE ----------------
# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate response
    if not st.session_state.vector_store:
        with st.chat_message("assistant"):
            st.warning("Please upload and initialize your PDFs in the sidebar first.")
    elif not groq_api_key:
        with st.chat_message("assistant"):
            st.error("API Key missing. Please enter it in the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    response = get_ai_response(prompt, groq_api_key)
                    ai_content = response.content
                    st.markdown(ai_content)
                    # Add to history
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_content})
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ---------------- FOOTER ----------------
if not st.session_state.vector_store:
    st.info("💡 **Tip:** Start by uploading your PDF files in the sidebar and clicking 'Initialize'.")