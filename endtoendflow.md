This repository is a fully offline document-QA app: a Flask web app loads a local PDF, extracts text (with OCR fallback), chunks it, embeds chunks into a FAISS vector store, retrieves semantically and keyword-relevant passages, and answers questions with a local extractive QA model. There is no SQL database; the “state” is in memory for the web app, and the only persistent artifacts are cached text JSON files and a FAISS index under .rag_cache.

High-level runtime flow

1. Start app
   - Entry point: app.py
   - `if __name__ == "__main__":`
     - starts a background thread: `threading.Thread(target=_build, daemon=True).start()`
     - then runs Flask: `app.run(...)`
   - This keeps the UI responsive while the expensive model/index build runs in the background.

2. Background pipeline build
   - `app.py` `_build()` calls:
     - `rag_pipeline.build_pipeline(log=_log)`
   - `build_pipeline()` does the whole RAG preparation:
     - choose PDF paths (`DEFAULT_PDF_PATHS` or `RAG_PDF_PATHS` env var)
     - load embedding model
     - fast-path load existing FAISS cache
     - otherwise extract text, split into chunks, build index, save index
     - then load the QA reader model and return `RetrieverReaderQA`

3. User submits question
   - Browser polls `/api/status` from `static/js/app.js`
   - Once status becomes `ready`, JS enables the “Get answer” button
   - Form submit sends POST to `/api/ask`
   - Flask `ask()` route calls:
     - `rag_pipeline.answer_query(qa, query)`
   - `answer_query()` calls:
     - `qa_chain.invoke({"query": query})`
   - `RetrieverReaderQA.invoke()` performs retrieval and reader scoring, returns final string
   - Flask returns JSON: `{"answer": "..."}`

4. Final output
   - Browser `static/js/app.js` renders the answer in the page and updates the question label

Now the exact file-by-file, function-by-function execution flow.

1) app.py: web server and orchestration

app.py does three main things:
- creates Flask app
- maintains a shared `_state` dict
- launches background pipeline build and exposes two HTTP endpoints

Relevant pieces:

- Imports:
  - `threading`
  - `Flask`, `jsonify`, `render_template`, `request`
  - `import rag_pipeline`

- Shared state:
  - `_state = {"qa_chain": None, "status": "loading", "message": "Starting up...", "pdf_paths": rag_pipeline.DEFAULT_PDF_PATHS}`

- `_log(msg)`
  - prints to stdout
  - updates `_state["message"]`
  - This is the status message the frontend polls

- `_build()`
  - `qa = rag_pipeline.build_pipeline(log=_log)`
  - sets:
    - `_state["qa_chain"] = qa`
    - `_state["status"] = "ready"`
    - `_state["message"] = "Pipeline ready."`
  - if build fails:
    - sets status error and message

- Routes:
  - `/`
    - returns `render_template("index.html", pdf_paths=", ".join(_state["pdf_paths"]))`
  - `/api/status`
    - returns JSON status/message
  - `/api/ask`
    - reads JSON body `{"query": "..."}`
    - if pipeline not ready: 503
    - if blank query: 400
    - else:
      - `answer = rag_pipeline.answer_query(qa, query)`
      - `return jsonify(answer=answer)`

This is the web app’s entry and orchestrator.

2) templates/index.html: browser UI shell

This is the rendered page:
- header with status chip
- hero section showing document source
- textarea for the question
- “Get answer” button
- “Clear” button
- hidden answer card for result/error

Important behavior:
- It loads CSS:
  - `static/css/styles.css`
- It loads JS:
  - `static/js/app.js`

It passes:
- `pdf_paths` from Flask into the template via Jinja:
  - `{{ pdf_paths }}`

3) static/js/app.js: browser-side request lifecycle

This file controls the user experience.

Main logic:

- DOM refs:
  - `statusChip`, `statusLabel`, `askBtn`, `clearBtn`, `form`, `queryEl`
  - `answerCard`, `answerText`, `answerQuestion`, `answerError`

- `setStatus(status, message)`
  - updates the status chip
  - if status == "ready":
    - `statusLabel.textContent = "Index ready"`
    - `ready = true`
    - enable `askBtn`
  - if error:
    - disables ask button
  - else:
    - `Preparing index…`

- `pollStatus()`
  - `fetch("/api/status")`
  - parses JSON
  - calls `setStatus(data.status, data.message)`
  - if `loading`, waits 1500ms and polls again
  - if `error`, show error modal

- `showAnswer(question, answer)`
  - fills answer card with answer and “In response to: …”

- `form.addEventListener("submit", async ...)`
  - reads textarea
  - validates non-empty
  - `fetch("/api/ask", { method: "POST", headers: ..., body: JSON.stringify({ query: question }) })`
  - parses response
  - on success: `showAnswer(question, data.answer)`
  - on failure: `showError(...)`

So the browser is simply a thin client: it polls readiness and submits question JSON to Flask.

4) rag_pipeline.py: actual RAG engine

This is the core of the app. It is a Python conversion of the notebook and contains the real pipeline logic. It sets offline environment variables before importing Hugging Face modules.

Imports and why they matter

At the top:

- `os`
  - config and environment
- `pypdf.PdfReader`
  - reads PDFs and extracts text when possible
- `pymupdf`
  - renders pages to images for OCR fallback
- `numpy as np`
  - image arrays for OCR
- `Document`
  - LangChain document wrapper
- `RecursiveCharacterTextSplitter`
  - splits pages into chunks
- `HuggingFaceEmbeddings`
  - converts text to vectors
- `FAISS`
  - vector store
- `transformers`
  - logging config
- `AutoTokenizer`, `AutoModelForQuestionAnswering`
  - local QA model

Important environment setup:
- `HF_HUB_OFFLINE = "1"`
- `TRANSFORMERS_OFFLINE = "1"`
- `GRADIO_ANALYTICS_ENABLED = "False"`
- `TRANSFORMERS_VERBOSITY = "error"`

This forces all model loading to happen from local cache only.

Configuration constants:
- `DEFAULT_PDF_PATHS = ["gk_ques_ans.pdf"]` by default
- `EMBEDDING_MODEL = "distilbert-base-uncased"`
- `QA_MODEL = os.environ.get("RAG_QA_MODEL", "deepset/roberta-base-squad2")`

This means:
- embeddings use a local DistilBERT embedding model
- answer extraction uses a local RoBERTa QA model
- nothing calls internet or external APIs

5) PDF ingestion stage

Function: `ocr_pdf(path, dpi=200, log=print)`

This is the OCR fallback for scanned PDFs or text-as-outline PDFs.

Flow:
- `from rapidocr_onnxruntime import RapidOCR`
- `engine = RapidOCR()`
- `doc = pymupdf.open(path)`
- For each page:
  - `pix = page.get_pixmap(dpi=dpi)`
  - `img = np.frombuffer(...)`
  - `result, _ = engine(img)`
  - `page_text = "\n".join(line[1] for line in (result or []))`
  - append only if non-empty
- returns a list of page texts

This is how scanned PDFs are read without external OCR services.

Function: `extract_texts(pdf_paths, log=print)`

This orchestrates extraction for every PDF file:

- For each path:
  - `reader = PdfReader(path)`
  - `extracted = [p.extract_text() for p in reader.pages]`
  - filter out empty strings
- If any page produced text:
  - `pdf_texts.extend(extracted)`
- Else:
  - `log("No embedded text ... running OCR")`
  - `pdf_texts.extend(ocr_pdf(path, log=log))`

At the end:
- if no text at all:
  - raise `RuntimeError("Still no text after OCR...")`

This is the exact “fast path + OCR fallback” pipeline.

6) Chunking stage

Function: `split_into_chunks(pdf_texts, chunk_size=1000, chunk_overlap=200, log=print)`

Flow:
- `docs = [Document(page_content=t) for t in pdf_texts]`
- `splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)`
- `documents = splitter.split_documents(docs)`
- logs chunk count

This converts raw PDF pages into LangChain `Document` objects and splits them into chunks so retrieval is narrower and more relevant.

7) Embedding and FAISS index

Function: `load_embeddings()`
- returns `HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)`

Function: `build_vector_store(documents, embedding_model, log=print)`
- `vector_store = FAISS.from_documents(documents, embedding_model)`
- logs `vector_store.index.ntotal`

This creates the vector index where each chunk becomes a vector, searchable by semantic similarity.

8) Hybrid retrieval

The repository uses a hybrid retrieval strategy:
- semantic retrieval via FAISS
- keyword overlap retrieval via custom scoring

Constants:
- `SEMANTIC_K = int(os.environ.get("RAG_SEMANTIC_K", "5"))`
- `KEYWORD_K = int(os.environ.get("RAG_KEYWORD_K", "5"))`

Stopword set:
- `_STOPWORDS = {...}`

Function: `_keywords(text)`
- lowercases
- uses regex to extract `[a-z0-9]+`
- removes 1-char tokens and stopwords
- returns a set of keywords

Class: `RetrieverReaderQA`

Constructor:
- stores `vector_store`, `reader`, `all_docs`
- precomputes `_doc_keywords = [_keywords(d.page_content) for d in all_docs]`

Methods:

- `_semantic(self, query)`
  - `self.vector_store.similarity_search(query, k=SEMANTIC_K)`
  - returns top semantic matches

- `_keyword(self, query)`
  - gets question keyword set `qk = _keywords(query)`
  - if no keywords: return []
  - compute overlap score:
    - `(len(qk & kws), i)` for each doc keyword set
  - sort descending
  - return the top `KEYWORD_K` docs with positive overlap

- `invoke(self, inputs)`
  - `query = inputs["query"]`
  - union of semantic and keyword results while deduplicating by chunk text
  - `seen, contexts = set(), []`
  - for each doc in `_semantic` + `_keyword`:
    - `text = doc.page_content`
    - if unique: add to contexts
  - returns:
    - `{"result": self.reader.best_answer(query, contexts)["answer"].strip()}`

This is the key retrieval logic:
- semantic arm catches meaning
- keyword arm catches exact factual terms like names and technical terms
- they merge and deduplicate

9) Extractive reader model

Class: `ExtractiveReader`

Constructor:
- `self.model`
- `self.tok`
- `self.torch_device`
- `self.max_answer_len = 40`

Method: `best_answer(question, contexts)`

This is the core answerer.

Pseudo-flow:
- if no contexts: return `{"answer": "", "score": 0.0}`
- tokenization:
  - `enc = self.tok([question] * len(contexts), list(contexts), return_tensors="pt", truncation="only_second", max_length=512, padding=True)`
- model forward pass:
  - `out = self.model(**{k: v.to(self.torch_device) for k, v in enc.items()})`
- get logits:
  - `starts = out.start_logits.detach().cpu()`
  - `ends = out.end_logits.detach().cpu()`
- loop through each candidate context:
  - `seq_ids = enc.sequence_ids(b)`
  - ignore tokens not in passage (`sid != 1`)
  - compute best start token among passage tokens
  - end token is chosen within a `max_answer_len` window
  - score = `start + end`
  - if this is highest, decode answer span with tokenizer
- return `{"answer": best_answer, "score": best_score}`

This is not text generation. It is extractive span selection: the model finds the best answer span directly inside retrieved passages.

10) Build the reader + QA pipeline

Function: `build_chain(vector_store, log=print)`

Flow:
- imports `torch`
- sets number of threads:
  - `torch.set_num_threads(os.cpu_count() or torch.get_num_threads())`
- chooses device:
  - `"cuda"` if available else `"cpu"`
- loads tokenizer:
  - `AutoTokenizer.from_pretrained(QA_MODEL)`
- loads model:
  - `AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)`
- moves model to device
- `model.eval()`
- creates reader:
  - `reader = ExtractiveReader(model, tokenizer, torch_device)`
- gets all docs from vector store:
  - `all_docs = list(vector_store.docstore._dict.values())`
- returns:
  - `RetrieverReaderQA(vector_store, reader, all_docs, log=log)`

This is the final QA chain object used by the web app.

11) Query execution

Function: `clean(text)`
- strips whitespace
- keeps only first paragraph before blank line:
  - `text.strip().split("\n\n")[0].strip()`

Function: `answer_query(qa_chain, query)`

Flow:
- if blank query:
  - returns `"Please enter a valid query."`
- try:
  - `response = clean(qa_chain.invoke({"query": query})["result"])`
  - returns `response if response and response.strip() else "No answer found."`
- except Exception as e:
  - returns `f"Error processing the query: {e}"`

This is the public query method consumed by Flask.

12) Caching strategy

This repo is optimized to avoid repeated expensive work.

Constants:
- `CACHE_DIR = os.environ.get("RAG_CACHE_DIR", ".rag_cache")`

Function: `_files_signature(pdf_paths)`
- hashes file bytes (not mtime)
- returns stable content hash

Function: `_source_signature(pdf_paths)`
- combines file hash + embedding/chunk config
- yields FAISS index cache key

Function: `get_texts(pdf_paths, log=print)`

Flow:
- compute `files_sig`
- cache path `CACHE_DIR/text_<hash>.json`
- if cache exists:
  - read JSON and return it
  - log “Loaded extracted text from cache”
- else:
  - `texts = extract_texts(pdf_paths, log=log)`
  - write them to JSON under `.rag_cache`
  - log cache written

This makes OCR run once per PDF content version.

Function: `build_pipeline(pdf_paths=None, log=print, use_cache=True)`

This is the main pipeline build orchestrator.

Flow:
- `pdf_paths = pdf_paths or DEFAULT_PDF_PATHS`
- `embedding_model = load_embeddings()`
- `vector_store = None`
- compute signature for index cache
- check `.rag_cache/<hash>/` for an existing FAISS index
- if cache exists:
  - `FAISS.load_local(index_dir, embedding_model, allow_dangerous_deserialization=True)`
  - else rebuild
- if no cached index:
  - `pdf_texts = get_texts(pdf_paths, log=log)`
  - `documents = split_into_chunks(pdf_texts, log=log)`
  - `vector_store = build_vector_store(documents, embedding_model, log=log)`
  - if index_dir: `vector_store.save_local(index_dir)`
- finally:
  - `return build_chain(vector_store, log=log)`

This is the key “startup speedup” optimization:
- OCR text cache avoids re-running extraction
- FAISS cache avoids re-embedding and re-indexing

13) CLI mode

`if __name__ == "__main__":`
- if first arg == `"extract"`:
  - `texts = get_texts(DEFAULT_PDF_PATHS)`
  - print count
- else:
  - `qa = build_pipeline()`
  - if extra arguments given:
    - question = " ".join(sys.argv[1:])
    - prints `Q:` and `A:`
  - else:
    - interactive REPL:
      - loop reading `input("Q: ")`
      - prints answer until blank line or Ctrl-C

This makes the same engine usable both as:
- a Python module
- a CLI tool
- a web app

End-to-end execution sequence in one chain

1. `python app.py`
2. Flask app starts
3. background thread `_build()` begins
4. `rag_pipeline.build_pipeline()`
5. `load_embeddings()`
6. check `.rag_cache` for FAISS index
7. if missing:
   - `get_texts()`
   - `extract_texts()`
   - `PdfReader` reads PDF pages
   - if no text, `ocr_pdf()` renders pages and uses RapidOCR
   - cache results to `.rag_cache/text_<hash>.json`
8. `split_into_chunks()` creates `Document` objects and chunks text
9. `build_vector_store()` creates FAISS from documents
10. save FAISS-local index to `.rag_cache/<hash>/`
11. `build_chain()` loads model and tokenizer
12. `RetrieverReaderQA` ready
13. app status becomes ready
14. browser polls `/api/status`
15. when ready, user enters question
16. browser POSTs to `/api/ask`
17. Flask route calls `rag_pipeline.answer_query()`
18. `answer_query()` calls `qa_chain.invoke({"query": ...})`
19. `RetrieverReaderQA.invoke()`:
   - semantic search in FAISS
   - keyword match across all chunks
   - deduplicate and merge contexts
20. `ExtractiveReader.best_answer()` runs batch tokenization and model inference
21. best answer span is decoded and returned
22. Flask returns JSON
23. browser renders answer on page

Important architectural interpretation

- There is no database in the usual sense.
  - all retrieval data is in FAISS files and in-memory state
  - any “persistence” is file-based cache in `.rag_cache`
- There are no API calls to external services.
  - models are local and offline
  - OCR is local via RapidOCR
- The app is essentially:
  - ingestion and retrieval engine (`rag_pipeline.py`)
  - web layer (`app.py`)
  - frontend layer (`templates/index.html` + `static/js/app.js` + `static/css/styles.css`)
- The “answer” is not generative text completion; it is extractive QA from retrieved context.

If you want, I can next turn this into a more diagrammatic call graph, or produce a “component map” showing each file and the exact function call tree with arrows.
