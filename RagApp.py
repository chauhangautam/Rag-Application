 
import streamlit as st
import tempfile
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.runnables import (
    RunnablePassthrough
)


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📚 RAG - PDF Question Answering")

st.write(
    "PDF → Chunks → Hugging Face Embeddings → "
    "Chroma → Retriever → Hugging Face LLM → Answer"
)


# =========================================================
# CHECK HF TOKEN
# =========================================================

if not HF_TOKEN:

    st.error(
        "HF_TOKEN not found in .env file."
    )

    st.info(
        "Create a .env file in the same folder "
        "and add HF_TOKEN=your_token"
    )

    st.stop()
else:

    st.success(
        "✅ Hugging Face Token found."
    )


# =========================================================
# SESSION STATE
# =========================================================

if "vector_store" not in st.session_state:

    st.session_state.vector_store = None


if "retriever" not in st.session_state:

    st.session_state.retriever = None


if "llm" not in st.session_state:

    st.session_state.llm = None


# =========================================================
# CHAT HISTORY
# =========================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# =========================================================
# PDF INFORMATION
# =========================================================

if "pdf_name" not in st.session_state:

    st.session_state.pdf_name = None


# =========================================================
# HUGGING FACE LLM
# =========================================================

if st.session_state.llm is None:

    try:

        st.session_state.llm = ChatOpenAI(

            # Hugging Face Inference Router
            base_url="https://router.huggingface.co/v1",

            # Hugging Face token
            api_key=HF_TOKEN,

            # Current Hugging Face supported chat model
            model="openai/gpt-oss-120b:fastest",

            temperature=0.1,

            max_tokens=512
        )

    except Exception as e:

        st.error(
            f"LLM initialization error: {str(e)}"
        )

        st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🔧How  RAG Pipeline Work")

    st.write("1️⃣ PDF Loader")

    st.write("2️⃣ Text Splitter")

    st.write("3️⃣ Hugging Face Embeddings")

    st.write("4️⃣ Chroma Vector Store")

    st.write("5️⃣ Retriever")

    st.write("6️⃣ Hugging Face LLM")

    st.write("7️⃣ Chat History")

    st.write("8️⃣ Prompt")

    st.write("9️⃣ RAG Chain")

    st.write("🔟 Answer")

    st.divider()

    # =====================================================
    # CLEAR CHAT
    # =====================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


    # =====================================================
    # PDF STATUS
    # =====================================================

    st.divider()

    st.subheader("📄 PDF Status")

    if st.session_state.pdf_name:

        st.success(
            st.session_state.pdf_name
        )

    else:

        st.info(
            "No PDF loaded"
        )


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(

    "Choose a PDF",

    type=["pdf"]
)


# =========================================================
# CREATE VECTOR STORE
# =========================================================

if st.button(
    "📚 Create Vector Store",
    use_container_width=True
):

    if uploaded_file is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        pdf_path = None

        try:

            # =================================================
            # SAVE PDF TEMPORARILY
            # =================================================

            with tempfile.NamedTemporaryFile(

                delete=False,

                suffix=".pdf"

            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                pdf_path = temp_file.name


            # =================================================
            # LOAD PDF
            # =================================================

            with st.spinner(
                "Loading PDF..."
            ):

                loader = PyPDFLoader(
                    pdf_path
                )

                documents = loader.load()


            st.success(
                f"✅ PDF loaded successfully. "
                f"Pages: {len(documents)}"
            )


            # =================================================
            # SPLIT PDF
            # =================================================

            with st.spinner(
                "Creating chunks..."
            ):

                text_splitter = (
                    RecursiveCharacterTextSplitter(

                        chunk_size=400,

                        chunk_overlap=100,

                        separators=[
                            "\n\n",
                            "\n",
                            " ",
                            ""
                        ]
                    )
                )

                chunks = (
                    text_splitter.split_documents(
                        documents
                    )
                )


            st.success(
                f"✅ Chunks created: {len(chunks)}"
            )


            # =================================================
            # HUGGING FACE EMBEDDINGS
            # =================================================

            with st.spinner(
                "Loading Hugging Face Embeddings..."
            ):

                embedding = HuggingFaceEmbeddings(

                    model_name=
                    "sentence-transformers/all-mpnet-base-v2"
                )


            st.success(
                "✅ Hugging Face Embedding Model loaded."
            )


            # =================================================
            # CHROMA VECTOR STORE
            # =================================================

            with st.spinner(
                "Creating Chroma Vector Store..."
            ):

                vector_store = Chroma.from_documents(

                    documents=chunks,

                    collection_name="pdf_collection",

                    embedding=embedding,

                    persist_directory=
                    "./chroma_langchain_db"
                )


            # =================================================
            # SAVE VECTOR STORE
            # =================================================

            st.session_state.vector_store = (
                vector_store
            )


            # =================================================
            # CREATE RETRIEVER
            # =================================================

            retriever = (
                vector_store.as_retriever(

                    search_kwargs={
                        "k": 6
                    }
                )
            )


            st.session_state.retriever = (
                retriever
            )


            # =================================================
            # SAVE PDF NAME
            # =================================================

            st.session_state.pdf_name = (
                uploaded_file.name
            )


            # =================================================
            # CLEAR OLD CHAT
            # =================================================

            st.session_state.chat_history = []


            st.success(
                "✅ Chroma Vector Store created."
            )

            st.success(
                "✅ Retriever created."
            )

            st.success(
                "✅ PDF is ready for questions."
            )


        except Exception as e:

            st.error(
                f"Vector Store Error: {str(e)}"
            )


        finally:

            # =================================================
            # DELETE TEMP FILE
            # =================================================

            if pdf_path:

                try:

                    os.remove(
                        pdf_path
                    )

                except Exception:

                    pass


# =========================================================
# FORMAT DOCUMENTS
# =========================================================

def format_docs(docs):

    return "\n\n".join(

        doc.page_content

        for doc in docs
    )


# =========================================================
# FORMAT CHAT HISTORY
# =========================================================

def format_chat_history():

    if not st.session_state.chat_history:

        return "No previous conversation."

    history = []

    for message in (
        st.session_state.chat_history
    ):

        role = message["role"]

        content = message["content"]

        if role == "user":

            history.append(
                f"User: {content}"
            )

        elif role == "assistant":

            history.append(
                f"Assistant: {content}"
            )


    return "\n".join(history)


# =========================================================
# SHOW PREVIOUS CHAT HISTORY
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# =========================================================
# CHAT SECTION
# =========================================================

if st.session_state.retriever is not None:

    st.divider()

    st.subheader(
        "💬 Ask Questions From Your PDF"
    )


    # =====================================================
    # CHAT INPUT
    # =====================================================

    user_prompt = st.chat_input(

        "Ask something about your PDF..."
    )


    if user_prompt:

        # =================================================
        # SHOW USER QUESTION
        # =================================================

        with st.chat_message(
            "user"
        ):

            st.write(
                user_prompt
            )


        # =================================================
        # SAVE USER QUESTION
        # =================================================

        st.session_state.chat_history.append(

            {
                "role": "user",

                "content": user_prompt
            }
        )


        # =================================================
        # GET PREVIOUS HISTORY
        # =================================================

        previous_history = (
            format_chat_history()
        )


        # =================================================
        # CREATE PROMPT
        # =================================================

        prompt = ChatPromptTemplate.from_messages(

            [

                # =========================================
                # SYSTEM MESSAGE
                # =========================================

                (
                    "system",

                    """
You are a helpful RAG assistant.

You answer questions using ONLY
the information contained in the PDF context.

Do not use outside knowledge.

Do not invent information.

If the requested information is not
available in the PDF context, reply exactly:

Data not found in PDF file.

You also have access to the previous
conversation.

Use the previous conversation only to
understand references such as:

- "it"
- "he"
- "she"
- "that"
- "the above"
- "previous question"

However, the actual answer must still
come from the PDF context.

=========================
PREVIOUS CONVERSATION
=========================

{chat_history}

=========================
PDF CONTEXT
=========================

{context}

=========================
END PDF CONTEXT
=========================
"""
                ),

                # =========================================
                # CURRENT USER QUESTION
                # =========================================

                (
                    "user",

                    "{input}"
                )

            ]
        )


        # =================================================
        # CREATE RAG CHAIN
        # =================================================

        rag_chain = (

            {

                # -----------------------------------------
                # RETRIEVE PDF DOCUMENTS
                # -----------------------------------------

                "context":

                st.session_state.retriever

                | format_docs,


                # -----------------------------------------
                # CURRENT QUESTION
                # -----------------------------------------

                "input":

                RunnablePassthrough(),


                # -----------------------------------------
                # PREVIOUS CHAT HISTORY
                # -----------------------------------------

                "chat_history":

                lambda x: previous_history

            }

            | prompt

            | st.session_state.llm

        )


        # =================================================
        # RUN RAG
        # =================================================

        with st.spinner(
            "🔎 Searching PDF and generating answer..."
        ):

            try:

                answer = (
                    rag_chain.invoke(
                        user_prompt
                    )
                )


                # =================================================
                # GET ANSWER TEXT
                # =================================================

                if hasattr(
                    answer,
                    "content"
                ):

                    answer = (
                        answer.content
                    )

                else:

                    answer = str(
                        answer
                    )


                # =================================================
                # SAVE ASSISTANT ANSWER
                # =================================================

                st.session_state.chat_history.append(

                    {
                        "role": "assistant",

                        "content": answer
                    }
                )


                # =================================================
                # SHOW ASSISTANT ANSWER
                # =================================================

                with st.chat_message(
                    "assistant"
                ):

                    st.write(
                        answer
                    )


                # =================================================
                # GET RETRIEVED DOCUMENTS
                # =================================================

                retrieved_docs = (

                    st.session_state.retriever.invoke(

                        user_prompt
                    )
                )


                # =================================================
                # SHOW RETRIEVED PDF CHUNKS
                # =================================================

                with st.expander(
                    "📄 View Retrieved PDF Chunks"
                ):

                    if not retrieved_docs:

                        st.write(
                            "No relevant PDF chunks found."
                        )

                    else:

                        for index, doc in enumerate(

                            retrieved_docs,

                            start=1
                        ):

                            st.markdown(

                                f"### 📄 Source {index}"
                            )


                            # -------------------------------------
                            # PAGE
                            # -------------------------------------

                            page = (
                                doc.metadata.get(
                                    "page",
                                    "Unknown"
                                )
                            )


                            if isinstance(
                                page,
                                int
                            ):

                                page = page + 1


                            st.write(
                                f"**Page:** {page}"
                            )


                            # -------------------------------------
                            # CONTENT
                            # -------------------------------------

                            st.write(
                                doc.page_content
                            )


                            st.divider()


            except Exception as e:

                st.error(
                    f"RAG Error: {str(e)}"
                )


                # Remove the user message if
                # the request failed

                if (
                    st.session_state.chat_history
                    and
                    st.session_state.chat_history[-1]["role"]
                    == "user"
                ):

                    st.session_state.chat_history.pop()


# =========================================================
# NO PDF MESSAGE
# =========================================================

else:

    st.info(
        "👆 Upload a PDF and click "
        "'Create Vector Store' to start chatting."
    )
 