const promptEl = document.getElementById("prompt");
const modelSelect = document.getElementById("model-select");
const temperatureEl = document.getElementById("temperature");
const topKEl = document.getElementById("top_k");
const maxTokensEl = document.getElementById("max_new_tokens");
const generateBtn = document.getElementById("generate-btn");
const outputEl = document.getElementById("output");
const errorEl = document.getElementById("error");

const tempVal = document.getElementById("temp-val");
const topkVal = document.getElementById("topk-val");
const lenVal = document.getElementById("len-val");

temperatureEl.addEventListener("input", () => (tempVal.textContent = temperatureEl.value));
topKEl.addEventListener("input", () => (topkVal.textContent = topKEl.value));
maxTokensEl.addEventListener("input", () => (lenVal.textContent = maxTokensEl.value));

function typewrite(el, text, charsPerTick = 3, delayMs = 12) {
  el.innerHTML = "";
  const span = document.createElement("span");
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  el.appendChild(span);
  el.appendChild(cursor);

  let i = 0;
  const timer = setInterval(() => {
    span.textContent += text.slice(i, i + charsPerTick);
    i += charsPerTick;
    if (i >= text.length) {
      clearInterval(timer);
      cursor.remove();
    }
  }, delayMs);
}

async function generate() {
  const prompt = promptEl.value.trim();
  errorEl.textContent = "";

  if (!prompt) {
    errorEl.textContent = "Speak first — the page cannot fill itself.";
    return;
  }

  generateBtn.disabled = true;
  generateBtn.querySelector("span").textContent = "The Muse Composes…";
  outputEl.innerHTML = '<p class="placeholder">The Muse Composes…</p>';

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        model: modelSelect.value,
        temperature: parseFloat(temperatureEl.value),
        top_k: parseInt(topKEl.value, 10),
        max_new_tokens: parseInt(maxTokensEl.value, 10),
      }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Something went awry backstage.");
    }

    typewrite(outputEl, prompt + data.output);
  } catch (err) {
    outputEl.innerHTML = '<p class="placeholder">The page remains blank.</p>';
    errorEl.textContent = err.message;
  } finally {
    generateBtn.disabled = false;
    generateBtn.querySelector("span").textContent = "Take the Stage";
  }
}

generateBtn.addEventListener("click", generate);
