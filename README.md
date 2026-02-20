#  Smart Presentation Assistant & PDF Intellectual Assistant  
### Hand Gesture Control + AI PDF Chatbot (Streamlit Application)

**Author:** Niyati Bhandari   
**Collaborators:** Raghav Khandelwal, Nityam Kalal, Parikshit Bishnoi, Suhani Jangid  

---

## Introduction

This project combines **Computer Vision** and **AI-powered Document Retrieval (RAG)** into a unified intelligent system.

It enables:

- Real-time **hand gesture based human–computer interaction**
- Navigate PDF slides using simple **left/right hand gestures**
- An **AI-powered PDF chatbot**
- Document question answering using Large Language Models

The system demonstrates practical integration of:

- MediaPipe (Hand Tracking)
- OpenCV (Computer Vision)
- FAISS (Vector Database)
- LangChain (RAG Pipeline)
- Groq LLM (LLaMA 3.1)
- Streamlit (User Interface)

---
**Home Page:**  
Streamlit-based interface displaying PDF slide viewer, gesture control panel, and Groq-powered document assistant.
<p align="left">
  <img src="https://raw.githubusercontent.com/Niyatiii2111/SP0510---Major-Project/5bc771d89149149ac559ba111079aa279043476c/Home%20page.jpeg" width="500"/>
  <br>
  <em>Gesture Mapping for Slide Navigation</em>
</p>



## Features

### Hand Gesture Control

- Real-time webcam hand tracking
- Slide navigation using finger-count gestures  
- One finger / fist → Previous slide  
- Four or more fingers → Next slide 

---

### PDF Intellectual Assistant

- Upload multiple PDF documents
- Automatic text extraction (PyPDF2)
- Recursive chunking with overlap
- Embedding generation (HuggingFace)
- FAISS vector database creation
- Semantic similarity search
- Context-aware response using Groq LLM
- Streamlit-based conversational UI
- Session-based chat history

---

## Gesture Guide
           


<p align="left">
  <img src="https://github.com/Niyatiii2111/SP0510---Major-Project/blob/8c1e9b3e60fa7957d3939abb810d6981e883cad3/Hands%20image.jpeg?raw=true" width="200"/>
  <br>
  <em>Gesture Mapping for Slide Navigation</em>
</p>

| Gesture | Action |
|--------|--------|
| 0–1 Fingers / Fist | Previous Slide |
| 4–5 Fingers | Next Slide |

## Project Architecture

### Gesture Control Flow
Webcam → CVZone HandDetector → Gesture Logic → Streamlit State → Slide Navigation


### PDF Chatbot (RAG) Flow
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS →Similarity Search → Groq LLM → Response


---

## Local Installation and Requirements

### Software Requirements

- Python 3.10+
- Webcam
- Internet connection (for Groq API)

---



## Running the Application

###  Gesture Control Module

```bash
python gesture_control.py
```

### PDF AI Chatbot Module

```bash
streamlit run pdf_chatbot.py
```

## Groq API Key Setup

1. Create an API key at:
https://console.groq.com
2. Enter the API key inside the Streamlit sidebar  
3. Upload PDF files  
4. Click process pdf

---
## Resources, Tools and Packages

### Main Libraries

- OpenCV  
- MediaPipe  
- NumPy  
- Streamlit  
- LangChain  
- FAISS  
- HuggingFace Embeddings  
- Groq LLM  
- PyPDF2  

---

## Deployment

The application is now publicly accessible via Streamlit Cloud.

🔗 *Open Web App:* https://your-streamlit-link-here.streamlit.app

> No installation required — works directly in browser

---

### What You Can Do Online
- Upload PDF files and chat with them
- Ask document-based questions
- Experience AI powered semantic search
- Test the interface without local setup

---

## My TODO List: Things that can be improved or added

### Software Improvements

- Refactor codebase  
- Improve gesture accuracy  
- Optimize cursor smoothing  
- Improve error handling  

---

### Feature Enhancements

- Voice commands
- Mouse cursor control using hand tracking  
- Zoom in / zoom out gesture support   
- Persistent chat history  
- Document summarization  

---

## Disclaimer

This project is developed for **learning, demonstration, and academic purposes only**.

Not intended for production or commercial deployment.
