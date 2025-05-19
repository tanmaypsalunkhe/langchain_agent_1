import os
import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# === Configuration ===
OPENAI_API_KEY = "secret key"
CHROMA_DB_DIR = "chroma_store"
DATA_DIR = "data"

# === Initialize session state ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === Streamlit UI ===
st.title("📄 AI Agent for Local PDF & TXT Files")
st.markdown("Upload `.pdf` or `.txt` files into the `/data` folder, then ask questions below.")

# === Load Documents ===
def load_documents_from_directory(directory):
    documents = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            continue
        loaded = loader.load()
        st.sidebar.write(f"✅ Loaded {file}: {len(loaded)} docs")
        if loaded:
            st.sidebar.text(loaded[0].page_content[:300])
        documents.extend(loaded)
    return documents

# === Show file list in sidebar ===
st.sidebar.title("📁 Files in Data Folder")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

file_list = [f for f in os.listdir(DATA_DIR) if f.endswith((".pdf", ".txt"))]
if file_list:
    for file_name in file_list:
        st.sidebar.markdown(f"- {file_name}")
else:
    st.sidebar.markdown("_No files found in `/data` folder._")



# === Split and Embed Documents ===
def process_documents():
    raw_docs = load_documents_from_directory(DATA_DIR)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(raw_docs)
    st.sidebar.write(f"🔖 Total Chunks Created: {len(docs)}")


    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    vectordb.persist()
    return vectordb

# === Load or Recreate Chroma Vector Store ===
def get_vectorstore():
    if os.path.exists(CHROMA_DB_DIR):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    else:
        return process_documents()




# === Set Up QA Chain ===
def create_qa_chain(vectordb):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # ✅ Tell memory what to expect from output
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.6, openai_api_key=OPENAI_API_KEY),
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # ✅ Tell the chain what the main output is
    )

    return qa_chain
# === Run App ===
vectordb = get_vectorstore()
qa_chain = create_qa_chain(vectordb)


sample = vectordb.similarity_search("test", k=1)
if sample:
    st.sidebar.write("🧪 Sample Document Chunk:")
    st.sidebar.text(sample[0].page_content[:300])  # Show first 300 chars
else:
    st.sidebar.warning("Vector DB is empty or no match found.")

import shutil
if st.sidebar.button("🧹 Clear & Rebuild Vector DB"):
    try:
        # Attempt to close Chroma before deleting
        vectordb = Chroma(persist_directory=CHROMA_DB_DIR)
        vectordb._client.reset()  # Properly closes DB
    except Exception as e:
        st.sidebar.warning(f"Couldn't close DB: {e}")

    try:
        shutil.rmtree(CHROMA_DB_DIR)
        st.sidebar.success("✅ Vector DB deleted. Reloading to rebuild...")
        st.experimental_rerun()
    except Exception as e:
        st.sidebar.error(f"❌ Failed to delete DB: {e}")

st.sidebar.write(f"📦 Vector DB documents: {vectordb._collection.count()}")

# === Chat Input ===
query = st.text_input("Ask a question about the documents:")
if query:
    with st.spinner("Searching..."):
        result = qa_chain({"question": query, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((query, result["answer"]))
        st.write("**Answer:**", result["answer"])

# === Chat History ===
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🧠 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
