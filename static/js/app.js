// Document QA System — front-end behaviour.
// Polls /api/status until the pipeline is ready, then submits questions to
// /api/ask and renders the grounded answer.

(function () {
  "use strict";

  const statusChip = document.getElementById("status-chip");
  const statusLabel = document.getElementById("status-label");
  const askBtn = document.getElementById("ask-btn");
  const clearBtn = document.getElementById("clear-btn");
  const form = document.getElementById("ask-form");
  const queryEl = document.getElementById("query");
  const spinner = document.getElementById("spinner");

  const answerCard = document.getElementById("answer-card");
  const answerText = document.getElementById("answer-text");
  const answerQuestion = document.getElementById("answer-question");
  const answerError = document.getElementById("answer-error");
  const answerErrorText = document.getElementById("answer-error-text");

  let ready = false;

  // --- Status polling ---------------------------------------------------
  function setStatus(status, message) {
    statusChip.setAttribute("data-status", status);
    if (status === "ready") {
      statusLabel.textContent = "Index ready";
      ready = true;
      askBtn.disabled = false;
    } else if (status === "error") {
      statusLabel.textContent = "Unavailable";
      askBtn.disabled = true;
    } else {
      statusLabel.textContent = message || "Preparing index…";
      askBtn.disabled = true;
    }
  }

  async function pollStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      setStatus(data.status, data.message);
      if (data.status === "loading") {
        setTimeout(pollStatus, 1500);
      } else if (data.status === "error") {
        showError(data.message || "The document index could not be prepared.");
      }
    } catch (e) {
      setTimeout(pollStatus, 3000);
    }
  }

  // --- Answer rendering -------------------------------------------------
  function showError(message) {
    answerCard.hidden = false;
    answerError.hidden = false;
    answerErrorText.textContent = message;
    answerText.textContent = "";
    answerQuestion.textContent = "";
  }

  function showAnswer(question, answer) {
    answerCard.hidden = false;
    answerError.hidden = true;
    answerText.textContent = answer;
    answerQuestion.textContent = "In response to: " + question;
  }

  function setBusy(busy) {
    spinner.hidden = !busy;
    askBtn.disabled = busy || !ready;
  }

  // --- Submit -----------------------------------------------------------
  form.addEventListener("submit", async function (evt) {
    evt.preventDefault();
    const question = queryEl.value.trim();
    if (!question) {
      showError("Please enter a question.");
      return;
    }
    if (!ready) return;

    setBusy(true);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question }),
      });
      const data = await res.json();
      if (!res.ok) {
        showError(data.error || "Something went wrong. Please try again.");
      } else {
        showAnswer(question, data.answer);
      }
    } catch (e) {
      showError("Network error. Please try again.");
    } finally {
      setBusy(false);
    }
  });

  clearBtn.addEventListener("click", function () {
    queryEl.value = "";
    answerCard.hidden = true;
    queryEl.focus();
  });

  // Kick off status polling on load.
  pollStatus();
})();
