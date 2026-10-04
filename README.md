# 📄 PDF Q&A Assistant — RAG-Based Document Intelligence

An intelligent document question-answering system that allows users to upload PDF documents and ask questions about their content. The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from documents and generate accurate, context-aware responses using an OpenAI language model.

---

## 🚀 Project Overview

The PDF Q&A Assistant is an end-to-end document intelligence application built using Python, Streamlit, LangChain, OpenAI GPT, and ChromaDB.

The system processes uploaded PDF documents by extracting their text, splitting the content into meaningful chunks, generating vector embeddings, and storing them in a vector database. When a user asks a question, the system performs similarity-based retrieval to find the most relevant document sections and passes the retrieved context to the language model to generate an answer.

---

## ✨ Features

- 📤 Upload PDF documents
- 📑 Extract text from PDF files
- ✂️ Split documents into manageable chunks
- 🔢 Generate embeddings for document chunks
- 🗄️ Store and retrieve embeddings using ChromaDB
- 🔍 Perform similarity-based document search
- 🤖 Generate context-aware answers using OpenAI GPT
- 💬 Interactive question-answering interface
- ⚡ Efficient document retrieval and query processing

---

## 🏗️ System Architecture

```text
                ┌─────────────────┐
                │   PDF Document  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Text Extraction│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Document Chunking│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Embeddings   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    ChromaDB     │
                │  Vector Store   │
                └────────┬────────┘
                         │
                  User Question
                         │
                         ▼
                ┌─────────────────┐
                │ Similarity Search│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Relevant Context│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   OpenAI GPT    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     Answer      │
                └─────────────────┘
