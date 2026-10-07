const API_URL = "http://127.0.0.1:8000/api/v1/analyze";

document.addEventListener("DOMContentLoaded", async () => {
  const claimInput = document.getElementById("claimInput");
  const verifyBtn = document.getElementById("verifyBtn");
  const loader = document.getElementById("loader");
  const resultCard = document.getElementById("resultCard");
  const verdictBadge = document.getElementById("verdictBadge");
  const confidencePill = document.getElementById("confidencePill");
  const summaryText = document.getElementById("summaryText");

  // Get selected text from current active tab if available
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) {
      const [{ result }] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => window.getSelection().toString(),
      });
      if (result && result.trim()) {
        claimInput.value = result.trim();
      }
    }
  } catch (err) {
    // Fallback if scripting not available
  }

  verifyBtn.addEventListener("click", async () => {
    const claim = claimInput.value.trim();
    if (!claim) return;

    verifyBtn.disabled = true;
    loader.classList.remove("hidden");
    resultCard.classList.add("hidden");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claim }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      verdictBadge.textContent = data.verdict_label || "Verdict";
      confidencePill.textContent = `Conf: ${(data.confidence * 100).toFixed(1)}%`;
      summaryText.textContent = data.summary || "Summary unavailable.";
      resultCard.classList.remove("hidden");

    } catch (err) {
      verdictBadge.textContent = "Error";
      confidencePill.textContent = "";
      summaryText.textContent = "Failed to connect to TrustNet backend API at 127.0.0.1:8000.";
      resultCard.classList.remove("hidden");
    } finally {
      verifyBtn.disabled = false;
      loader.classList.add("hidden");
    }
  });
});
