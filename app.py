import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTORSTORE_DIR = os.path.join(working_dir, "vector_db_dir")


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )


@st.cache_resource
def build_chat_chain(_vectorstore, _memory):
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.5,
        groq_api_key=GROQ_API_KEY
    )

    retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt_template = """
You are a highly factual and helpful medical assistant.

ONLY use the content from the provided documents to answer the question.
If the answer is not clearly present in the context, respond with:
"I don’t know based on the provided documents."

Never use outside knowledge, guesswork, or fabricated information.

-------------
Context:
{context}

Question: {question}

Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=_memory,
        return_source_documents=True,
        verbose=True,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": prompt}
    )


st.set_page_config(page_title="Smart Healthcare Assistant", page_icon="🩺", layout="centered")
st.title("🩺 Smart Healthcare Assistant")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        llm=None,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = build_chat_chain(
        st.session_state.vectorstore,
        st.session_state.memory
    )


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask any healthcare-related question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response = st.session_state.conversational_chain({"question": user_input})
            assistant_response = response["answer"]

            st.markdown(assistant_response)

        except Exception as e:
            assistant_response = "⚠️ AI service is temporarily unavailable. Please try again."
            st.error(f"Error: {str(e)}")

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
