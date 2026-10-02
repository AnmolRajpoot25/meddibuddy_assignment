// Minimal Weather Advisory Chat Client
let sessionId = "session_" + Math.random().toString(36).substring(2, 9);

const messagesList = document.getElementById("messages-list");
const emptyState = document.getElementById("empty-state");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");
const simulateFailureCheckbox = document.getElementById("simulate-failure-checkbox");
const sessionIdDisplay = document.getElementById("session-id-display");
const resetSessionBtn = document.getElementById("reset-session-btn");
const viewSopsBtn = document.getElementById("view-sops-btn");
const sopModal = document.getElementById("sop-modal");
const closeModalBtn = document.getElementById("close-modal-btn");
const modalSopsList = document.getElementById("modal-sops-list");
const sopsCountBadge = document.getElementById("sops-count-badge");

sessionIdDisplay.textContent = sessionId;

// Fetch initial policies
loadPolicies();

// Form & event bindings
chatForm.addEventListener("submit", onSendMessage);
resetSessionBtn.addEventListener("click", onResetSession);
viewSopsBtn.addEventListener("click", () => sopModal.classList.remove("hidden"));
closeModalBtn.addEventListener("click", () => sopModal.classList.add("hidden"));
sopModal.addEventListener("click", (e) => {
  if (e.target === sopModal) sopModal.classList.add("hidden");
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !sopModal.classList.contains("hidden")) {
    sopModal.classList.add("hidden");
  }
});

// Suggestion click
document.querySelectorAll(".suggestion-item").forEach((btn) => {
  btn.addEventListener("click", () => {
    userInput.value = btn.dataset.query;
    userInput.focus();
  });
});

// Auto-expand input
userInput.addEventListener("input", () => {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 100) + "px";
});

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event("submit"));
  }
});

async function onSendMessage(e) {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;

  if (emptyState) {
    emptyState.remove();
  }

  const simulateFailure = simulateFailureCheckbox.checked;

  appendUserBubble(text);
  userInput.value = "";
  userInput.style.height = "auto";

  const loader = appendLoader();
  messagesList.scrollTop = messagesList.scrollHeight;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: sessionId,
        simulate_weather_failure: simulateFailure
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await res.json();
    loader.remove();
    appendAssistantResponse(data);
  } catch (err) {
    loader.remove();
    appendError(err.message);
  } finally {
    messagesList.scrollTop = messagesList.scrollHeight;
  }
}

function appendUserBubble(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `<div class="user-text">${escapeHtml(text)}</div>`;
  messagesList.appendChild(row);
}

function appendLoader() {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.innerHTML = `
    <div class="assistant-body">
      <div class="loading-dots">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  messagesList.appendChild(row);
  return row;
}

function appendError(msg) {
  const row = document.createElement("div");
  row.className = "msg-row assistant";
  row.innerHTML = `
    <div class="assistant-body">
      <div class="policy-tag tag-critical">Error</div>
      <div class="assistant-content"><p>${escapeHtml(msg)}</p></div>
    </div>
  `;
  messagesList.appendChild(row);
}

function appendAssistantResponse(data) {
  const row = document.createElement("div");
  row.className = "msg-row assistant";

  const primarySop = data.primary_sop;
  const weather = data.weather_data;
  const respType = data.response_type;

  let tagHtml = "";
  if (primarySop) {
    const sev = (primarySop.severity || "ADVISORY").toLowerCase();
    tagHtml = `<div class="policy-tag tag-${sev}">[${primarySop.id}] ${primarySop.severity}: ${escapeHtml(primarySop.title)}</div>`;
  } else if (respType === "no_guidance") {
    tagHtml = `<div class="policy-tag tag-neutral">No policy applies (Safe fallback)</div>`;
  } else if (respType === "api_error") {
    tagHtml = `<div class="policy-tag tag-critical">Weather data unavailable</div>`;
  } else if (respType === "location_unresolved") {
    tagHtml = `<div class="policy-tag tag-caution">Location required</div>`;
  }

  let telemetryHtml = "";
  if (weather) {
    const loc = escapeHtml(data.location_name || "");
    telemetryHtml = `
      <div class="telemetry-summary">
        <span class="tele-item">Location: <span>${loc}</span></span>
        <span class="tele-item">Temp: <span>${weather.temperature_2m}°C</span></span>
        <span class="tele-item">Precip: <span>${weather.precipitation}mm (${weather.precipitation_probability}%)</span></span>
        <span class="tele-item">Wind: <span>${weather.wind_speed_10m} km/h</span></span>
        <span class="tele-item">UV: <span>${weather.uv_index}</span></span>
      </div>
    `;
  }

  const contentHtml = formatMarkdown(data.response);

  row.innerHTML = `
    <div class="assistant-body">
      ${tagHtml}
      <div class="assistant-content">${contentHtml}</div>
      ${telemetryHtml}
    </div>
  `;

  messagesList.appendChild(row);
}

function onResetSession() {
  sessionId = "session_" + Math.random().toString(36).substring(2, 9);
  sessionIdDisplay.textContent = sessionId;
  messagesList.innerHTML = `
    <div class="empty-state" id="empty-state">
      <p class="empty-title">Session cleared. Ask about outdoor activity safety in any location.</p>
    </div>
  `;
}

async function loadPolicies() {
  try {
    const res = await fetch("/api/sops");
    if (!res.ok) return;
    const data = await res.json();
    sopsCountBadge.textContent = data.total;

    modalSopsList.innerHTML = data.sops.map(sop => {
      const sev = (sop.severity || "ADVISORY").toLowerCase();
      return `
        <div class="sop-row">
          <div class="sop-row-top">
            <span class="sop-row-title">${escapeHtml(sop.title)}</span>
            <span class="policy-tag tag-${sev}">[${sop.id}] ${sop.severity}</span>
          </div>
          <div class="sop-row-guidance">${escapeHtml(sop.guidance)}</div>
          <div class="sop-row-meta">Activities: ${sop.applicable_activities.join(", ")}</div>
        </div>
      `;
    }).join("");
  } catch (e) {
    console.error("Failed to load policies:", e);
  }
}

function formatMarkdown(text) {
  if (!text) return "";
  let html = escapeHtml(text);
  // Bold
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Inline code
  html = html.replace(/`(.*?)`/g, '<code style="font-family: var(--font-mono); font-size: 0.82em; background: var(--bg-muted); padding: 1px 4px; border-radius: 3px;">$1</code>');
  // Lists
  html = html.replace(/^\s*-\s+(.*)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
  // Paragraphs
  html = html.replace(/\n\n/g, '<p></p>');
  html = html.replace(/\n/g, '<br>');
  return html;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
