import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit app
st.title("Conversation RAG With PDF Uploads and Chat History")
st.write("Upload a PDF and chat with its content")

# Input the GROQ API Key
api_key = st.text_input("Enter your GROQ API key:", type="password")

# Check if GROQ API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")

    # Chat interface
    session_id = st.text_input("Session ID", value="default")

    # Statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Upload PDF file
    uploaded_files = st.file_uploader("Choose a PDF file:", type="pdf", accept_multiple_files=False)

    # Process the uploaded file
    if uploaded_files:
        documents = []
        # Get the current directory
        current_directory = os.getcwd()
        temppdf = os.path.join(current_directory, "temp.pdf")  # Save in the same directory as the script
        st.write(f"Temporary file will be saved to: {temppdf}")

        # Save the uploaded file to the current directory
        with open(temppdf, "wb") as file:
            file.write(uploaded_files.getvalue())
            st.write(f"File size: {os.path.getsize(temppdf)} bytes")  # Debug: Check file size

        try:
            # Load the PDF file
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error loading PDF: {e}")
            st.stop()  # Stop execution if there's an error
        finally:
            # Clean up the temporary file
            if os.path.exists(temppdf):
                os.remove(temppdf)
                st.write(f"Temporary file deleted: {temppdf}")

        # Split and create embeddings for the document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma.db")
        retriever = vectorstore.as_retriever()

        # Prompt to make LLM understand the previous context
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Answer question prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Function to get session history
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]  # Return the specific session's history

        # Wrap the RAG chain with message history
        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User input
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversation_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                }
            )
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter your GROQ API key.")