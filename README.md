# Advanced Offline RAG Assistant: The Uni-RAG Agent

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green?style=for-the-badge)
![Agentic AI](https://img.shields.io/badge/Agentic%20AI-ReAct%20MCP-blueviolet?style=for-the-badge)
![CUDA](https://img.shields.io/badge/NVIDIA%20CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia)

An advanced, **Zero-Trust**, and completely **offline Agentic Retrieval-Augmented Generation (RAG)** system built with Streamlit and LangChain. This application transforms conversational AI by handling complex, multi-step queries entirely on a secure local machine, providing **Grounded Answers**.

---

## 📖 Table of Contents
- [About The Project](#-about-the-project)
- [✨ Key Features](#-key-features)
- [🏛️ System Architecture (The Local Agentic MCP)](#️-system-architecture-the-local-agentic-mcp)
- [🛠️ Tech Stack](#-tech-stack)
- [🚀 Getting Started](#-getting-started)
- [💻 Usage & Demo](#-usage-&-demo)
- [🛣️ Future Work](#-future-work)

---

## 🏛️ About The Project

The **Uni-RAG Agent** solves the critical challenge of deploying powerful, conversational AI in environments that demand **Zero-Trust Data Privacy**. Unlike cloud-based RAG solutions, this system runs entirely on your local machine using local Large Language Models (LLMs) and GPU acceleration.

The core RAG pipeline has been upgraded to an **Agentic Control Plane (MCP)** that performs **Multi-Step Reasoning** (using the ReAct framework), allowing it to decompose complex user questions into a sequence of precise, verifiable searches across multiple documents.

---

## ✨ Key Features

* **Local Agentic Control Plane (MCP):** Uses a **ReAct Agent** to analyze multi-part questions, determine the necessary steps (Task Decomposition), call the RAG Tool multiple times if needed, and synthesize the final answer. This is the core innovation over standard RAG.
* **Zero-Trust Data Privacy (100% Offline):** After initial model downloads, the entire application runs without any internet connection. All components, including the LLM (**Ollama/phi3**), are local, guaranteeing absolute data privacy.
* **Grounded & Verifiable Answers:** The system enforces the LLM to use **embedded source citations** (`[File Name, Page X]`) for every factual claim based on the retrieved metadata. This provides full transparency and meets audit requirements.
* **GPU Acceleration (CUDA):** Leverages an NVIDIA GPU for fast document processing (`HuggingFaceEmbeddings`) and fast offline **Text-to-Speech (TTS)** generation (`Coqui TTS`).
* **Persistent Knowledge Base:** Supports indexing and searching an entire local folder, with incremental updates for new files.

---

## 🏛️ System Architecture (The Local Agentic MCP)

The architecture is centered on the **Agentic Workflow**:

1.  **User Input:** User asks a complex question (e.g., "What was the Q3 budget, and what is the risk mentioned on page 5 of the Q4 document?").
2.  **Local Agentic MCP (Executor):** The Agent (LLaMA model running on Ollama) takes control and initiates the ReAct Loop.
    * **Thought:** The Agent plans the steps (e.g., "First search Q3, then search Q4").
    * **Action:** The Agent calls the **`document_search_tool`** (custom RAG tool).
3.  **Tool (Custom RAG Retriever):** The tool searches the local **ChromaDB** and returns relevant text chunks **with File/Page Metadata**.
4.  **Iteration & Final Synthesis:** The Agent iterates the loop until all facts are retrieved, and then synthesizes a single, coherent, and **Grounded Answer**.

### Agentic Workflow Diagram



<img width="500px" height="500px" alt="Architecture diagram" src="https://github.com/user-attachments/assets/7e808cb1-96a6-4c5a-bc22-35cc4b81f75b" />



---

## 🛠️ Tech Stack

| Category | Tools Used |
| :--- | :--- |
| **Framework** | Streamlit |
| **AI Orchestration** | **LangChain** (`AgentExecutor`, `Tool`, `create_react_agent`) |
| **Local LLM (MCP)** | **Ollama** (serving the **`phi3`** model) |
| **Vector Database** | **ChromaDB** |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) on **CUDA** |
| **TTS Engine** | Coqui TTS on **CUDA** |

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.11**
* An **NVIDIA GPU** with at least 6GB of VRAM
* **Ollama** installed and running (`ollama serve`)

### Installation Guide

This process must be followed carefully to ensure the entire system runs **offline**.

**1. Setup Python Environment & Dependencies**
* Create and activate a new, clean Python virtual environment (e.g., `venv_agent`).
* **Install PyTorch for CUDA 12.1** (requires intermittent internet):
    ```sh
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```
* **Install Requirements & Agent Dependencies:** Ensure `langchain-agents` is installed along with your `requirements.txt`:
    ```sh
    pip install -r requirements.txt
    pip install langchain-agents
    ```

**2. Setup Local LLM & AI Models**
* Pull the required LLM:
    ```sh
    ollama pull phi3
    ```
* Run your pre-download script (`setup.py`) to cache all other models locally:
    ```sh
    python setup.py
    ```

---

## 💻 Usage & Demo

1.  Make sure your **Ollama server is running** (`ollama serve`) in the background.
2.  Start the Streamlit app from your terminal:
    ```sh
    streamlit run app.py
    ```
3.  **To Demo Agentic RAG:** Use the "Chat with a Folder" mode.
    * **Test the Agent:** Input a complex, multi-step query (e.g., "What was the revenue in the Q2 report, and which document lists the security risks?").
    * **Crucially:** The terminal running Streamlit will display the **Agent's Thought/Action/Observation** log (`verbose=True`), demonstrating the Agent calling the RAG Tool multiple times.
4.  **Verify Grounding:** The final answer in the UI will contain the required citations, demonstrating the **Grounded & Verifiable Answers** feature.

---

## 🛣️ Future Work

* **Agentic Data Automation:** Fully implement the **Local Data Connector Tool** for the Agent to securely synchronize files from services like Google Drive and Gmail to the local archive.
* **Multimodal RAG:** Integrate the originally planned Image Q&A and Audio Ingestion using Vision-Language Models (like `bakllava`) and speech-to-text (**Whisper**).
