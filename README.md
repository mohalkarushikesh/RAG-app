### End-to-end RAG app: PDFs to answers (step by step)

You’re building a Retrieval-Augmented Generation (RAG) pipeline. Below is a clean, working, and minimal setup that:

- Extracts text from uploaded PDFs
- Splits text into chunks and wraps them as Document objects
- Builds a FAISS vector store using HuggingFace embeddings
- Creates a retriever and a QA chain using FLAN-T5
- Serves everything with a simple Gradio UI

---

#### 1) Install required packages

```bash
pip install langchain langchain-community langchain-core langchain-huggingface transformers pypdf sentence-transformers gradio
```

---

#### 2) Full Python script

```python
# RAG: PDF -> text -> Documents -> Embeddings -> FAISS -> Retriever -> QA
# Save this as app.py and run: python app.py

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_classic.llms import HuggingFacePipeline

import gradio as gr


# -----------------------------
# PDF -> text
# -----------------------------
def extract_text_from_pdfs(uploaded_files):
    """
    uploaded_files: dict[str, IO] where values are file-like objects opened in binary mode
    Returns: list[str] page-level texts
    """
    pdf_texts = []
    for filename, file_content in uploaded_files.items():
        reader = PdfReader(file_content)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_texts.append(text)
    return pdf_texts


# -----------------------------
# text -> Documents -> chunks
# -----------------------------
def create_documents(pdf_texts, chunk_size=1000, chunk_overlap=200):
    """
    Wrap raw page texts into Document objects and split into smaller chunks.
    Returns: list[Document]
    """
    # Wrap each page as a Document
    docs = [Document(page_content=t) for t in pdf_texts]

    # Split into chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunked_docs = splitter.split_documents(docs)
    return chunked_docs


# -----------------------------
# Documents -> FAISS
# -----------------------------
def create_vector_store(documents, embedding_model):
    """
    Build FAISS index from documents and embeddings.
    """
    return FAISS.from_documents(documents, embedding_model)


# -----------------------------
# Build QA chain (Retriever + LLM + Prompt)
# -----------------------------
def build_qa_chain(vector_store):
    """
    Creates a RetrievalQA chain using FLAN-T5 and the given vector_store.
    """
    # Prompt
    prompt_template = """
Given the following information, answer the question.

Context:
{context}

Question: {question}
Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # LLM (FLAN-T5 small for speed; use 'google/flan-t5-base' for better quality)
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1  # set to 0 if you have a CUDA GPU available
    )

    llm = HuggingFacePipeline(pipeline=generator)

    retriever = vector_store.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# -----------------------------
# Gradio app
# -----------------------------
def build_index(files):
    """
    Gradio handler to build the index from uploaded PDF files.
    """
    if not files:
        return "Please upload at least one PDF.", None

    # Convert Gradio File objects to {name: file_obj}
    uploaded_files = {f.name: open(f.name, "rb") for f in files}

    try:
        # Extract text
        pdf_texts = extract_text_from_pdfs(uploaded_files)

        if not pdf_texts:
            return "No extractable text found in the uploaded PDFs.", None

        # Create chunked documents
        documents = create_documents(pdf_texts)

        # Embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # FAISS vector store
        vector_store = create_vector_store(documents, embedding_model)

        # QA chain
        qa_chain = build_qa_chain(vector_store)

        return f"Index built successfully. {len(documents)} chunks indexed.", qa_chain

    except Exception as e:
        return f"Error building index: {e}", None
    finally:
        # Close file descriptors
        for _, fh in uploaded_files.items():
            try:
                fh.close()
            except:
                pass


def answer_query(query, qa_chain_state):
    """
    Gradio handler to answer queries using the stored qa_chain.
    """
    if not query or not query.strip():
        return "Please enter a valid query."

    if qa_chain_state is None:
        return "No index found. Please upload PDFs and click 'Build Index' first."

    try:
        response = qa_chain_state.run(query)
        return response if response and response.strip() else "No answer found for your query."
    except Exception as e:
        return f"Error processing the query: {e}"


# Build Gradio UI with state
with gr.Blocks() as demo:
    gr.Markdown("# Document QA System (RAG)")
    gr.Markdown("Upload PDF files, build the index, then ask questions based on the documents.")

    qa_chain_state = gr.State(value=None)

    with gr.Row():
        file_input = gr.Files(file_types=[".pdf"], label="Upload PDF files")
    build_btn = gr.Button("Build Index")
    build_status = gr.Textbox(label="Status", interactive=False)

    gr.Markdown("### Ask a question")
    query_input = gr.Textbox(label="Your question")
    answer_output = gr.Textbox(label="Answer", interactive=False)
    ask_btn = gr.Button("Get Answer")

    build_btn.click(
        fn=build_index,
        inputs=[file_input],
        outputs=[build_status, qa_chain_state]
    )

    ask_btn.click(
        fn=answer_query,
        inputs=[query_input, qa_chain_state],
        outputs=[answer_output]
    )

if __name__ == "__main__":
    demo.launch()
```

---

#### 3) How to run

- Save the code as `app.py`.
- Run `python app.py`.
- Open the Gradio link in your browser.
- Upload one or more PDFs, click “Build Index,” then ask your question.

---

#### 4) Common pitfalls and quick fixes

- **Undefined variables:** Ensure you call the steps in order: extract → documents → embeddings → FAISS → retriever → QA chain.
- **Empty text from PDFs:** Some PDFs are scans. Use OCR (e.g., `pytesseract`) if `page.extract_text()` returns `None`.
- **GPU vs CPU:** Set `device=0` in the FLAN-T5 pipeline if you have a CUDA GPU; otherwise keep `device=-1`.
- **Large PDFs:** Increase `chunk_size` or adjust `chunk_overlap` in `create_documents` depending on your content.
