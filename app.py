# # --- Force Hugging Face Transformers to be offline ---
# import os
# os.environ['HF_HUB_OFFLINE'] = '1'

# import streamlit as st
# import ollama
# import tempfile
# import hashlib
# from TTS.api import TTS

# # --- LANGCHAIN AND AI LIBRARIES (MODERN IMPORTS) ---
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import OllamaLLM

# # --- APP CONFIGURATION ---
# st.set_page_config(page_title="Advanced RAG Agent", layout="wide")
# st.title("📄 Advanced RAG Agent")

# # --- GLOBAL CONFIGURATION ---
# CHROMA_DB_PATH = "./chroma_db_folder"
# KNOWLEDGE_BASE_PATH = r"C:\Users\vijay\OneDrive\Desktop\RAG-Base"

# # --- CORE FUNCTIONS ---

# @st.cache_resource
# def get_tts_model():
#     return TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=True)

# @st.cache_data
# def generate_audio_file(text):
#     tts = get_tts_model()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
#         tts.tts_to_file(text=text, file_path=fp.name)
#         return fp.name

# def get_file_hash(file):
#     hasher = hashlib.md5()
#     for chunk in iter(lambda: file.read(4096), b""):
#         hasher.update(chunk)
#     file.seek(0)
#     return hasher.hexdigest()

# # Function for single, temporary uploads
# @st.cache_resource
# def create_single_file_vector_store(file_hash, file_path, file_name):
#     st.info(f"Processing {file_name} on GPU...")
#     loader = PyPDFLoader(file_path) if file_name.endswith('.pdf') else Docx2txtLoader(file_path)
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_documents(docs)
#     for split in splits:
#         split.metadata['source'] = file_name
#     model_kwargs = {'device': 'cuda'}
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs=model_kwargs)
#     vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
#     st.success(f"File '{file_name}' processed and ready!")
#     return vectorstore

# # Function for the persistent, multi-file folder
# @st.cache_resource
# def load_or_create_folder_vector_store(folder_path):
#     st.info(f"Loading knowledge base from '{folder_path}'...")
#     model_kwargs = {'device': 'cuda'}
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs=model_kwargs)
#     vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    
#     indexed_files = [metadata['source'] for metadata in vectorstore.get()['metadatas']]
#     current_files = [f for f in os.listdir(folder_path) if f.endswith(('.pdf', '.docx'))]
#     new_files = [f for f in current_files if f not in indexed_files]
    
#     if new_files:
#         st.info(f"New documents found: {', '.join(new_files)}. Processing...")
#         all_new_docs = []
#         for filename in new_files:
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 loader = PyPDFLoader(file_path) if filename.endswith('.pdf') else Docx2txtLoader(file_path)
#                 docs = loader.load()
#                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                 splits = text_splitter.split_documents(docs)
#                 for split in splits:
#                     split.metadata['source'] = filename
#                 all_new_docs.extend(splits)
#                 st.write(f"Processed '{filename}'.")
#             except Exception as e:
#                 st.error(f"Error processing file {filename}: {e}")
        
#         if all_new_docs:
#             st.info("Creating embeddings for new documents on GPU...")
#             vectorstore.add_documents(all_new_docs)
#             st.success("Knowledge base updated successfully!")
#     else:
#         st.success("Knowledge base is up-to-date.")

#     return vectorstore

# def chat_interface(vectorstore):
#     st.header("Ask a Question")
#     col1, col2 = st.columns([0.9, 0.1])
#     with col1:
#         question = st.text_input("Enter your question...", key="question_input", label_visibility="collapsed")
#     with col2:
#         search_button = st.button("🔍", use_container_width=True)

#     if search_button or question:
#         if not question:
#             st.warning("Please enter a question.")
#         else:
#             with st.spinner("Searching, thinking, and generating answer..."):
#                 retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})
#                 retrieved_docs = retriever.invoke(question)
#                 if not retrieved_docs:
#                     st.warning("Could not find any relevant information in the document(s).")
#                 else:
#                     context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
#                     try:
#                         llm = OllamaLLM(model="phi3")
#                         template = "Use the following context to answer the question concisely... \n\nContext: {context}\nQuestion: {question}\nAnswer:"
#                         prompt = PromptTemplate.from_template(template)
#                         rag_chain = prompt | llm | StrOutputParser()
                        
#                         st.success("Here is the answer:")
#                         response_stream = rag_chain.stream({"context": context, "question": question})
#                         response = st.write_stream(response_stream)

#                         with st.spinner("Generating audio..."):
#                             audio_file_path = generate_audio_file(response)
#                             if audio_file_path:
#                                 st.audio(audio_file_path)
                        
#                         # --- MODIFIED: Replaced the expander with a single source line ---
#                         if retrieved_docs:
#                             top_source_filename = retrieved_docs[0].metadata['source']
#                             st.markdown(f"**Source:** `{top_source_filename}`")

#                     except Exception as e:
#                         st.error(f"An error occurred with the LLM: {e}. Is the Ollama server running?")

# # --- MAIN APPLICATION LOGIC ---
# def main():
#     with st.sidebar:
#         st.header("Select Mode")
#         app_mode = st.radio(
#             "Choose your interaction mode:",
#             ("Upload a Single Document", "Chat with a Folder")
#         )
#         st.markdown("---")

#     if app_mode == "Upload a Single Document":
#         st.write("Upload a single PDF or DOCX file for a focused chat session.")
#         with st.sidebar:
#             st.header("Upload Your Document")
#             uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx"], label_visibility="collapsed")

#         if uploaded_file:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name
#             file_hash = get_file_hash(uploaded_file)
#             vectorstore = create_single_file_vector_store(file_hash, tmp_file_path, uploaded_file.name)
#             chat_interface(vectorstore)
#         else:
#             st.warning("Please upload a file to begin.")

#     elif app_mode == "Chat with a Folder":
#         st.write(f"Ask questions across all documents located in: `{KNOWLEDGE_BASE_PATH}`")
#         if not os.path.exists(KNOWLEDGE_BASE_PATH):
#             os.makedirs(KNOWLEDGE_BASE_PATH)
        
#         vectorstore = load_or_create_folder_vector_store(KNOWLEDGE_BASE_PATH)
        
#         if vectorstore and vectorstore._collection.count() > 0:
#             chat_interface(vectorstore)
#         else:
#             st.warning(f"The folder '{KNOWLEDGE_BASE_PATH}' is empty. Please add PDF or DOCX files and refresh the page.")

# if __name__ == "__main__":
#     main()

# --- Force Hugging Face Transformers to be offline ---
import os
os.environ['HF_HUB_OFFLINE'] = '1'

# --- API & UTILITY IMPORTS ---
import uvicorn
import tempfile
import hashlib
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# --- TTS IMPORT ---
from TTS.api import TTS

# --- LANGCHAIN AND AI LIBRARIES (MODERN IMPORTS) ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.vectorstores import VectorStore

# ===================================================================
# GLOBAL CONFIGURATION
# ===================================================================

CHROMA_DB_PATH = "./chroma_db_folder"
KNOWLEDGE_BASE_PATH = r"C:\Users\vijay\OneDrive\Desktop\RAG-Base"
AUDIO_DIR = "./static_audio" # Folder to serve generated audio

# Ensure static audio directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# ===================================================================
# LOAD MODELS ON STARTUP
# ===================================================================

print("Loading TTS model... (This may take a moment)")
try:
    tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=True)
    print("TTS model loaded successfully.")
except Exception as e:
    print(f"Error loading TTS model: {e}. TTS functionality will be disabled.")
    tts_model = None

print("Loading embedding model (all-MiniLM-L6-v2) to GPU...")
try:
    model_kwargs = {'device': 'cuda'}
    embeddings_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs=model_kwargs
    )
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to load embedding model: {e}")
    exit() # Can't run without embeddings

print("Loading Ollama LLM (phi3)...")
try:
    llm = OllamaLLM(model="phi3")
    # Do a quick test
    llm.invoke("hello")
    print("Ollama LLM (phi3) connected successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to connect to Ollama: {e}")
    print("Please ensure the Ollama server is running and 'phi3' is installed.")
    exit() # Can't run without LLM

# ===================================================================
# REFACTORED CORE FUNCTIONS (No Streamlit)
# ===================================================================

def generate_audio_file_url(text: str, base_url: str) -> Optional[str]:
    """
    Generates audio, saves it to the static folder, and returns the public URL.
    """
    if tts_model is None:
        print("Skipping TTS generation as model failed to load.")
        return None
    
    try:
        # Create a unique, predictable filename based on the content hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        filename = f"{text_hash}.wav"
        file_path = os.path.join(AUDIO_DIR, filename)
        
        # Only generate if it doesn't already exist (simple caching)
        if not os.path.exists(file_path):
            print(f"Generating new audio file: {filename}")
            tts_model.tts_to_file(text=text, file_path=file_path)
        
        # Return the full URL for the client
        return f"{base_url}audio/{filename}"
    
    except Exception as e:
        print(f"Error generating audio file: {e}")
        return None

def create_single_file_vector_store(file_path: str, file_name: str) -> VectorStore:
    """
    Processes a single file and creates an in-memory vector store.
    """
    print(f"Processing {file_name}...")
    try:
        loader = PyPDFLoader(file_path) if file_name.endswith('.pdf') else Docx2txtLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        for split in splits:
            split.metadata['source'] = file_name
        
        # Use the globally loaded embedding model
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model)
        print(f"File '{file_name}' processed and ready!")
        return vectorstore
    
    except Exception as e:
        print(f"Error processing single file {file_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

def load_or_create_folder_vector_store(folder_path: str) -> Chroma:
    """
    Loads the persistent vector store and indexes any new files.
    """
    print(f"Loading knowledge base from '{folder_path}'...")
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created knowledge base directory: {folder_path}")

    # Use the globally loaded embedding model
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings_model
    )
    
    try:
        existing_metadatas = vectorstore.get().get('metadatas', [])
        indexed_files = {metadata['source'] for metadata in existing_metadatas if 'source' in metadata}
    except Exception as e:
        print(f"Warning: Could not retrieve metadata from Chroma: {e}. May re-index files.")
        indexed_files = set()

    current_files = [f for f in os.listdir(folder_path) if f.endswith(('.pdf', '.docx'))]
    new_files = [f for f in current_files if f not in indexed_files]
    
    if new_files:
        print(f"New documents found: {', '.join(new_files)}. Processing...")
        all_new_docs = []
        for filename in new_files:
            file_path = os.path.join(folder_path, filename)
            try:
                loader = PyPDFLoader(file_path) if filename.endswith('.pdf') else Docx2txtLoader(file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                for split in splits:
                    split.metadata['source'] = filename
                all_new_docs.extend(splits)
                print(f"Processed '{filename}'.")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
        
        if all_new_docs:
            print("Creating embeddings for new documents on GPU...")
            vectorstore.add_documents(all_new_docs)
            print("Knowledge base updated successfully!")
    else:
        print("Knowledge base is up-to-date.")

    return vectorstore

# ===================================================================
# CREATE RAG CHAIN (Reusable Function)
# ===================================================================

async def get_rag_response(query: str, vectorstore: VectorStore, base_url: str) -> dict:
    """
    Performs the full RAG pipeline (retrieve, generate, TTS) and returns a dict.
    """
    print(f"Received query: {query}")
    try:
        # 1. Retrieve Documents
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})
        retrieved_docs = await retriever.ainvoke(query)
        
        if not retrieved_docs:
            print("No relevant documents found.")
            return {
                "response": "Could not find any relevant information in the provided document(s).",
                "source": None,
                "audio_url": None
            }
        
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        top_source_filename = retrieved_docs[0].metadata.get('source', 'Unknown')

        # 2. Build RAG Chain and Generate Response
        template = "Use the following context to answer the question concisely... \n\nContext: {context}\nQuestion: {question}\nAnswer:"
        prompt = PromptTemplate.from_template(template)
        
        # Use the globally loaded LLM
        rag_chain = prompt | llm | StrOutputParser()
        
        print("Generating LLM response...")
        response_text = await rag_chain.ainvoke({"context": context, "question": query})

        # 3. Generate Audio
        print("Generating audio...")
        audio_url = generate_audio_file_url(response_text, base_url)
        
        print("RAG response complete.")
        return {
            "response": response_text,
            "source": top_source_filename,
            "audio_url": audio_url
        }

    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error in RAG pipeline: {e}")

# ===================================================================
# FASTAPI APP INITIALIZATION
# ===================================================================

app = FastAPI(
    title="Advanced RAG Agent API",
    description="API for the Uni-RAG Agent (SIH)"
)

# --- Add CORS Middleware ---
# This is ESSENTIAL for your React app to connect
origins = [
    "http://localhost",
    "http://localhost:3000",  # Default React port
    "http://localhost:5173",  # Default Vite port
    "http://localhost:8081",  # <-- ADD THIS LINE for your current port
    # Add your React app's deployed URL here later
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount Static Directory ---
# This makes the /static_audio folder public at the /audio URL
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# --- Load Persistent Knowledge Base ---
# This runs ONCE on startup
try:
    folder_vector_store = load_or_create_folder_vector_store(KNOWLEDGE_BASE_PATH)
    print(f"Persistent vector store loaded. {folder_vector_store._collection.count()} documents indexed.")
except Exception as e:
    print(f"CRITICAL: Failed to load persistent vector store: {e}")
    folder_vector_store = None # Handle this in the endpoint

# ===================================================================
# API DATA MODELS (Pydantic)
# ===================================================================

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    response: str
    source: Optional[str] = None
    audio_url: Optional[str] = None

# ===================================================================
# API ENDPOINTS
# ===================================================================

@app.get("/")
def read_root():
    return {"status": "Advanced RAG Agent API is running"}

@app.post("/refresh-knowledge-base")
def refresh_knowledge_base():
    """
    Triggers a re-scan of the knowledge base folder for new files.
    """
    global folder_vector_store
    try:
        folder_vector_store = load_or_create_folder_vector_store(KNOWLEDGE_BASE_PATH)
        count = folder_vector_store._collection.count()
        return {"status": "Knowledge base refreshed successfully", "documents_indexed": count}
    except Exception as e:
        print(f"Error refreshing knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Error refreshing knowledge base: {e}")

@app.post("/chat-folder", response_model=ChatResponse)
async def chat_with_folder(request: ChatRequest):
    """
    Chat with the persistent, pre-loaded folder of documents.
    """
    if folder_vector_store is None:
        raise HTTPException(status_code=500, detail="Persistent vector store is not loaded. Check server logs.")
    
    # Base URL for audio files
    base_url = "http://127.0.0.1:8000/" # TODO: Change this if your domain changes
    
    response_data = await get_rag_response(request.query, folder_vector_store, base_url)
    
    return ChatResponse(query=request.query, **response_data)


@app.post("/chat-file", response_model=ChatResponse)
async def chat_with_single_file(
    query: str = Form(...), 
    file: UploadFile = File(...)
):
    """
    Upload a single file for a temporary chat session.
    """
    # Save the uploaded file to a temporary path
    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name
        
        # Create a temporary vector store for this file
        temp_vectorstore = create_single_file_vector_store(tmp_file_path, file.filename)
        
        # Base URL for audio files
        base_url = "http://127.0.0.1:8000/" # TODO: Change this if your domain changes
        
        # Get the RAG response
        response_data = await get_rag_response(query, temp_vectorstore, base_url)
        
        return ChatResponse(query=query, **response_data)
    
    except Exception as e:
        # This will catch errors from get_rag_response or file processing
        print(f"Error in /chat-file: {e}")
        if isinstance(e, HTTPException):
            raise e # Re-raise if it's already an HTTP exception
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        file.file.close()

# ===================================================================
# RUN THE API
# ===================================================================

if __name__ == "__main__":
    print("Starting Uvicorn server...")
    uvicorn.run(
        "app:app",  # Points to this file (app.py) and the 'app' object
        host="127.0.0.1",
        port=8000,
        reload=True   # Server will restart on code changes
    )
