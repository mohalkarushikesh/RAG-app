"""Flask web application for the RAG pipeline.

Serves a U.S. Web Design System (USWDS)-styled single page that lets you ask
questions about the indexed PDF(s). The heavy work -- OCR, embedding, building
the FAISS index and loading the LLM -- happens once in a background thread at
startup; the page reports readiness and only enables the Ask button when the
pipeline is loaded.

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import threading

from flask import Flask, jsonify, render_template, request

import rag_pipeline

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Pipeline state -- built once in a background thread so the page can load
# immediately and poll /api/status while the (slow, one-time) build runs.
# ---------------------------------------------------------------------------
_state = {
    "qa_chain": None,
    "status": "loading",   # loading | ready | error
    "message": "Starting up...",
    "pdf_paths": rag_pipeline.DEFAULT_PDF_PATHS,
}
_lock = threading.Lock()


def _log(msg):
    """Route pipeline progress into the shared status message + stdout."""
    print(msg, flush=True)
    with _lock:
        _state["message"] = str(msg)


def _build():
    try:
        qa = rag_pipeline.build_pipeline(log=_log)
        with _lock:
            _state["qa_chain"] = qa
            _state["status"] = "ready"
            _state["message"] = "Pipeline ready."
    except Exception as e:  # noqa: BLE001
        with _lock:
            _state["status"] = "error"
            _state["message"] = f"Failed to build pipeline: {e}"
        print(f"[error] {e}", flush=True)


@app.route("/")
def index():
    return render_template("index.html", pdf_paths=", ".join(_state["pdf_paths"]))


@app.route("/api/status")
def status():
    with _lock:
        return jsonify(status=_state["status"], message=_state["message"])


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    with _lock:
        qa = _state["qa_chain"]
        st = _state["status"]

    if st != "ready" or qa is None:
        return jsonify(error="The document index is still being prepared. Please wait."), 503
    if not query:
        return jsonify(error="Please enter a question."), 400

    answer = rag_pipeline.answer_query(qa, query)
    return jsonify(answer=answer)


if __name__ == "__main__":
    # Kick off the one-time pipeline build in the background.
    threading.Thread(target=_build, daemon=True).start()
    # use_reloader=False so the reloader doesn't spawn a second (duplicate) build.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
