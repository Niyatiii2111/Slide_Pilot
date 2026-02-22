# SlidePilot — Smart Presentation Assistant  
### Gesture Navigation · Zoom Control · AI PDF Chatbot

**Author:** Niyati Bhandari   
**Collaborators:** Raghav Khandelwal, Nityam Kalal, Parikshit Bishnoi, Suhani Jangid  

---



## Introduction

SlidePilot is an intelligent PDF presentation assistant that combines **gesture-based slide navigation**, **gesture-controlled zooming**, and an **AI-powered PDF chatbot** in a single Streamlit application.

Users can upload PDF documents, move between slides using hand gestures, zoom in or out with finger gestures, and ask questions about document content using an integrated AI assistant powered by Retrieval Augmented Generation (RAG).

The system supports dual gesture modes (Navigation and Zoom), ensuring reliable interaction while minimizing false detections. This project demonstrates practical integration of Computer Vision and document-based AI for academic presentation automation.

The system demonstrates practical integration of:

- CVZone (MediaPipe-based Hand Tracking)
- OpenCV (Computer Vision)
- FAISS (Vector Database)
- LangChain (RAG Pipeline)
- Groq LLM (LLaMA 3.1)
- Streamlit (User Interface)

---
## Home Page
Streamlit-based interface displaying PDF slide viewer, gesture control panel, and Groq-powered document assistant.
<p align="left">
  <img src="https://raw.githubusercontent.com/Niyatiii2111/SP0510---Major-Project/7f7740a96539d775317be7165a83cc0b54c7ca36/Slidepilot%20homepage.jpeg
Project/5bc771d89149149ac559ba111079aa279043476c/Home%20page.jpeg" width="500"/>
  <br>



## Features

### Hand Gesture Control

- Real-time webcam hand tracking (CVZone + OpenCV)
- Dual gesture modes: Navigation and Zoom
- Fist hold (2 seconds) to switch modes
- Navigation Mode:
  - 1 finger → Previous slide
  - 4-5 fingers → Next slide
- Zoom Mode:
  - 1 finger → Zoom In
  - 4-5 fingers → Zoom Out
- Dead zone for 2–3 fingers to avoid false triggers
- Manual zoom controls as fallback
- Visual gesture feedback and mode indicators
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

| Mode | Gesture | Action |
|------|---------|--------|
| NAV | Fist (2sec) | Switch to Zoom Mode |
| NAV | 1 finger | Previous Slide |
| NAV | 4-5 fingers | Next Slide |
| ZOOM | Fist (2sec) | Switch to Nav Mode |
| ZOOM | 1 finger | Zoom In |
| ZOOM | 4-5 fingers | Zoom Out |
| Both | 2–3 fingers | Dead zone (no action) |



## Project Architecture

### Gesture Control Flow
Webcam → CVZone HandDetector → Gesture Logic → Streamlit State → Slide/Zoom Control


### PDF Chatbot (RAG) Flow
PDF Upload → Text Extraction → Chunking → Embeddings → FAISS → Similarity Search → Groq LLM → Response


---

## Local Installation and Requirements

### Software Requirements

- Python 3.10+
- Webcam
- Internet connection (for Groq API)

---


# GROQ API SETUP (update)

Your code now uses environment variable.

## Groq API Key Setup

Set your API key as an environment variable.

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

🔗 *Open Web App:* https://smart-presentation-bot.streamlit.app/

> No installation required — works directly in browser

---

### What You Can Do Online
- Upload PDF files and chat with them
- Ask document-based questions
- Experience AI powered semantic search
- Test the interface without local setup
---

### Future Enhancements

- Voice commands
- Mouse cursor control using hand tracking    
- Persistent chat history  
- Document summarization  

---

## Disclaimer

This project is developed for **learning, demonstration, and academic purposes only**.

Not intended for production or commercial deployment.
