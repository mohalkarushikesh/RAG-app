# End-to-End RAG App: PDFs → Answers

A small, self-contained **Retrieval-Augmented Generation (RAG)** pipeline built step by step in a
Jupyter notebook ([RAG.ipynb](RAG.ipynb)). You give it PDFs, it finds the passages relevant to your
question and uses a local language model to answer *from those passages*.

The whole thing runs **fully offline** on locally cached models — no HuggingFace Hub, no API keys,
no network calls.

```
PDF -> text (+ OCR fallback) -> chunks -> embeddings -> FAISS -> retriever -> QA -> Gradio UI
```

---

## What it does

1. **Extract** text from PDFs (with an OCR fallback for scanned / text-as-outline PDFs).
2. **Chunk** the text and wrap it as LangChain `Document` objects.
3. **Embed** each chunk into a vector with a local embedding model.
4. **Index** the vectors in a FAISS store for fast similarity search.
5. **Answer** questions with a `RetrievalQA` chain: retrieve the closest chunks, drop them into a
   prompt, and generate a grounded answer with a local LLM.
6. **Serve** it through a simple Gradio UI.

---

## Models used (all local / offline)

| Role       | Model                              | Notes                                                        |
|------------|------------------------------------|--------------------------------------------------------------|
| Embeddings | `distilbert-base-uncased`          | Wrapped with mean-pooling → 768-dim sentence vectors         |
| LLM        | `microsoft/phi-1_5`                | Run via the `text-generation` pipeline                       |
| OCR        | RapidOCR (`rapidocr-onnxruntime`)  | Self-contained ONNX models; used only when a PDF has no text |

> **Why these?** They were the capable models already present in the local HuggingFace cache. The
> original `all-MiniLM-L6-v2` + `flan-t5-small` recipe was dropped because those weren't cached and
> `flan-t5`'s `text2text-generation` task was removed in transformers 5.x.

---

## Setup

Install the dependencies (run once, then restart the kernel):

```bash
pip install langchain langchain-community langchain-core langchain-huggingface \
            transformers pypdf sentence-transformers gradio tf-keras
```

For the OCR fallback (scanned / outlined PDFs), also install:

```bash
pip install pymupdf rapidocr-onnxruntime onnxruntime numpy
```

---

## How to run

1. Open [RAG.ipynb](RAG.ipynb) in Jupyter / VS Code.
2. In **Step 3**, set `PDF_PATHS` to your PDF file(s):
   ```python
   PDF_PATHS = ["your_document.pdf"]
   ```
3. Run the cells **top to bottom**. Each step runs on real data and prints its result, so you can
   watch the pipeline build: pages extracted → chunks → vector dimension → FAISS matches → answer.
4. **Step 10** launches the Gradio UI — open the printed local URL and ask questions.

> Run **Step 1 first**: it sets the offline environment variables *before* `transformers` / `gradio`
> are imported. If you'd already imported them, restart the kernel.

---

## Offline notes

The notebook is designed for a machine that can't reach the internet:

- `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` — load only cached models, never call the Hub.
- `GRADIO_ANALYTICS_ENABLED=False` — stop Gradio from firing telemetry over the network. On an
  offline box that background request otherwise leaves a dead HTTP client, which later surfaces as
  `RuntimeError: Cannot send a request, as the client has been closed` when you click a button.
- RapidOCR ships its models inside the wheel, so OCR works without any download.

---

## OCR fallback

Some PDFs contain **no real text** — they're scanned images, or the text was converted to
outlines/curves (every letter drawn as vector shapes). `pypdf` extracts nothing from those.

Step 4 handles this automatically:

1. **Fast path** — `pypdf.extract_text()` for normal, text-based PDFs (instant).
2. **OCR fallback** — if a file yields no text, each page is rendered to an image with PyMuPDF and
   read with RapidOCR. Slower (a few minutes for a large document on CPU), but it recovers the text.

---

## Notebook structure

| Step | What it does                                        |
|------|-----------------------------------------------------|
| 0    | Install dependencies                                |
| 1    | Force offline mode & quiet warnings                 |
| 2    | Imports                                             |
| 3    | Point at a PDF (`PDF_PATHS`)                         |
| 4    | PDF → text (with OCR fallback)                      |
| 5    | text → chunks                                       |
| 6    | Load the embedding model                            |
| 7    | Build the FAISS index                               |
| 8    | Prompt + LLM + retriever = QA chain                 |
| 9    | Ask a question                                      |
| 10   | Gradio UI (optional)                                |

---

## Tips & limitations

- **Ask specific questions.** `phi-1_5` is a small model and retrieval returns a few focused chunks,
  so precise factual questions ("Who is the founder of Vaccinology?") work far better than broad ones
  ("What is this document about?").
- **Better answers = better LLM.** Swapping `phi-1_5` for an instruction-tuned model (e.g.
  `Qwen2.5-0.5B-Instruct`) noticeably improves answer quality — provided its weights are cached.
- **OCR is one-time per build** and CPU-bound; expect a few minutes for a large scanned PDF.
- **GPU:** set `device=0` in the `text-generation` pipeline (Step 8) if you have a CUDA GPU.

---

## Roadmap

See [Todo.md](Todo.md) — next up: a `.py` script version, multi-document retrieval, and live source
retrieval.
