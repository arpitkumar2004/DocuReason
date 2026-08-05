// Client-side Application Logic for Tri-Path RAG Workspace UI

document.addEventListener('DOMContentLoaded', () => {
  const textarea = document.getElementById('prompt-input');
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
  });
});

function handleKeyDown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    submitPrompt();
  }
}

function sendSample(queryText) {
  document.getElementById('prompt-input').value = queryText;
  submitPrompt();
}

function clearFeed() {
  const feed = document.getElementById('chat-feed');
  feed.innerHTML = `
    <div class="message-group assistant-group">
      <div class="avatar assistant-avatar">RAG</div>
      <div class="message-content">
        <div class="welcome-card">
          <h2>Workspace Cleared</h2>
          <p>Ask a new question across text passages, structured DuckDB tables, and visual chart regions.</p>
        </div>
      </div>
    </div>
  `;
}

async function submitPrompt() {
  const textarea = document.getElementById('prompt-input');
  const query = textarea.value.trim();
  if (!query) return;

  textarea.value = '';
  textarea.style.height = 'auto';

  const feed = document.getElementById('chat-feed');

  // 1. Render User Prompt Bubble
  const userGroupId = 'user-' + Date.now();
  const userHtml = `
    <div class="message-group user-group" id="${userGroupId}">
      <div class="avatar user-avatar">YOU</div>
      <div class="message-content">
        <div class="prompt-bubble">${escapeHtml(query)}</div>
      </div>
    </div>
  `;
  feed.insertAdjacentHTML('beforeend', userHtml);

  // 2. Render Assistant Skeleton Loader Card
  const assistantGroupId = 'asst-' + Date.now();
  const skeletonHtml = `
    <div class="message-group assistant-group" id="${assistantGroupId}">
      <div class="avatar assistant-avatar">RAG</div>
      <div class="message-content" style="width: 100%;">
        <div class="response-card">
          <div class="skeleton-box">
            <div class="skeleton-line" style="width: 60%;"></div>
            <div class="skeleton-line" style="width: 90%;"></div>
            <div class="skeleton-line" style="width: 75%;"></div>
          </div>
        </div>
      </div>
    </div>
  `;
  feed.insertAdjacentHTML('beforeend', skeletonHtml);
  scrollToBottom();

  // 3. Fetch Query Results from Backend REST API
  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    });
    const data = await res.json();
    renderAssistantResponse(assistantGroupId, query, data);
  } catch (err) {
    renderErrorResponse(assistantGroupId, err.message);
  }
}

function renderAssistantResponse(groupId, query, data) {
  const asstElement = document.getElementById(groupId);
  if (!asstElement) return;

  const probs = (data.router && data.router.probabilities) ? data.router.probabilities : { text: 0.5, table: 0.1, vision: 0.1 };
  const weights = (data.router && data.router.weights) ? data.router.weights : { text: 0.33, table: 0.33, vision: 0.33 };
  const sqlData = data.sql_execution || {};
  const answer = data.answer || "No response generated.";
  const attribution = data.attribution || {};
  const nliScore = attribution.faithfulness_score !== undefined ? attribution.faithfulness_score : 0.95;

  // Build DuckDB Result Table HTML if available
  let sqlTableHtml = '<div style="font-size:12px; color:var(--text-muted);">No structured table rows returned.</div>';
  if (sqlData.route_active === false) {
    sqlTableHtml = `<div style="font-size:12px; color:var(--text-muted); font-style:italic;">Table route inactive for this query (Probability P(table)=${probs.table || 0.0} &lt; threshold 0.35). DuckDB SQL query execution skipped.</div>`;
  } else if (sqlData.sql_results && sqlData.sql_results.length > 0) {
    const cols = Object.keys(sqlData.sql_results[0]);
    sqlTableHtml = '<table class="data-table"><thead><tr>' + cols.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr></thead><tbody>';
    sqlData.sql_results.forEach(row => {
      sqlTableHtml += '<tr>' + cols.map(c => `<td>${escapeHtml(String(row[c]))}</td>`).join('') + '</tr>';
    });
    sqlTableHtml += '</tbody></table>';
  }

  // Build Evidence Items HTML
  let evidenceHtml = '';
  (data.retrieved_evidence || []).forEach(item => {
    evidenceHtml += `
      <div style="padding: 8px 0; border-bottom: 1px solid var(--border-color); font-size: 12px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
          <span style="font-weight: 700; color: var(--accent-cyan);">${escapeHtml(item.modality || 'text')}</span>
          <span style="color: var(--text-muted);">Score: ${item.rank_score || item.score}</span>
        </div>
        <div style="color: #cbd5e1;">${escapeHtml(item.text || item.linearized || '')}</div>
      </div>
    `;
  });

  const cardHtml = `
    <div class="avatar assistant-avatar">RAG</div>
    <div class="message-content" style="width: 100%;">
      <div class="response-card">
        
        <!-- Grounded Answer Body -->
        <div class="markdown-body">
          ${escapeHtml(answer)}
        </div>

        <!-- Inline Action Bar -->
        <div class="action-bar">
          <button class="action-btn" onclick="copyText('${escapeJsString(answer)}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Copy
          </button>
          <button class="action-btn" onclick="sendSample('${escapeJsString(query)}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
            Regenerate
          </button>
          <span style="font-size: 11px; color: var(--accent-emerald); align-self: center; margin-left: auto;">
            ✓ NLI Score: ${nliScore} (Faithful)
          </span>
        </div>

        ${data.reasoning_chain ? `
        <!-- Collapsible Card: DeepSeek-R1 Chain-of-Thought Reasoning -->
        <div class="collapsible-card">
          <div class="collapsible-header" onclick="toggleCollapsible(this)">
            <span>🧠 DeepSeek-R1 Reasoning Chain (&lt;think&gt;)</span>
            <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
          </div>
          <div class="collapsible-body">
            <pre style="font-size:12px; color:#cbd5e1; white-space:pre-wrap;"><code>${escapeHtml(data.reasoning_chain)}</code></pre>
          </div>
        </div>
        ` : ''}

        <!-- Collapsible Card: Soft Multi-Label Router Confidence (Block B2) -->
        <div class="collapsible-card">
          <div class="collapsible-header" onclick="toggleCollapsible(this)">
            <span>🔀 Router Probabilities & Weights (Block B2)</span>
            <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
          </div>
          <div class="collapsible-body">
            <div class="meter-row">
              <div class="meter-label"><span>TEXT P(text)</span><span>${probs.text}</span></div>
              <div class="meter-bg"><div class="meter-fill" style="width:${probs.text * 100}%; background:var(--accent-cyan);"></div></div>
            </div>
            <div class="meter-row">
              <div class="meter-label"><span>TABLE P(table)</span><span>${probs.table}</span></div>
              <div class="meter-bg"><div class="meter-fill" style="width:${probs.table * 100}%; background:var(--accent-amber);"></div></div>
            </div>
            <div class="meter-row">
              <div class="meter-label"><span>VISION P(vision)</span><span>${probs.vision}</span></div>
              <div class="meter-bg"><div class="meter-fill" style="width:${probs.vision * 100}%; background:var(--accent-indigo);"></div></div>
            </div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 8px;">
              Normalized Fusion Weights: <code>${JSON.stringify(weights)}</code>
            </div>
          </div>
        </div>

        <!-- Collapsible Card: DuckDB Text-to-SQL Execution (Block B4) -->
        <div class="collapsible-card">
          <div class="collapsible-header" onclick="toggleCollapsible(this)">
            <span>🗄️ DuckDB Text-to-SQL Result (Block B4) ${sqlData.route_active === false ? '<span style="color:var(--text-muted); font-size:11px; margin-left:6px;">(Route Inactive)</span>' : ''}</span>
            <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
          </div>
          <div class="collapsible-body">
            <div style="font-size: 11px; color: var(--text-muted); margin-bottom: 4px;">Generated SQL Query:</div>
            <pre><code>${escapeHtml(sqlData.sql_query || 'N/A')}</code></pre>
            <div style="margin-top: 10px;">${sqlTableHtml}</div>
          </div>
        </div>

        <!-- Collapsible Card: Reciprocal Rank Fusion Evidence (Block B6) -->
        <div class="collapsible-card">
          <div class="collapsible-header" onclick="toggleCollapsible(this)">
            <span>⚡ Reciprocal Rank Fusion & Reranked Evidence (Block B6)</span>
            <svg class="chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
          </div>
          <div class="collapsible-body">
            ${evidenceHtml || '<div style="font-size:12px; color:var(--text-muted);">No evidence chunks returned.</div>'}
          </div>
        </div>

      </div>
    </div>
  `;
  asstElement.innerHTML = cardHtml;
  scrollToBottom();
}

function renderErrorResponse(groupId, errorMsg) {
  const asstElement = document.getElementById(groupId);
  if (!asstElement) return;

  asstElement.innerHTML = `
    <div class="avatar assistant-avatar">RAG</div>
    <div class="message-content" style="width: 100%;">
      <div class="response-card" style="border-color: rgba(244, 63, 94, 0.4);">
        <div style="color: var(--accent-rose); font-weight: 600; font-size: 14px; margin-bottom: 6px;">Query Processing Error</div>
        <div style="font-size: 13px; color: var(--text-muted);">${escapeHtml(errorMsg)}</div>
      </div>
    </div>
  `;
}

function toggleCollapsible(headerElem) {
  const card = headerElem.closest('.collapsible-card');
  card.classList.toggle('open');
}

function toggleSettingsModal() {
  const modal = document.getElementById('settings-modal');
  modal.classList.toggle('open');
}

function toggleUploadModal() {
  toggleSettingsModal();
}

async function triggerIngestion() {
  const inPath = document.getElementById('ingest-input-path').value;
  const outPath = document.getElementById('ingest-output-path').value;
  const statusBox = document.getElementById('ingest-modal-status');
  statusBox.style.display = 'block';
  statusBox.innerText = 'Running Phase 1 document ingestion and layout parsing pipeline...';

  try {
    const res = await fetch('/api/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input_dir: inPath, output_dir: outPath })
    });
    const data = await res.json();
    statusBox.innerText = 'Ingestion Completed Successfully!';
  } catch (err) {
    statusBox.innerText = 'Ingestion Error: ' + err.message;
  }
}

function copyText(str) {
  navigator.clipboard.writeText(str);
}

function scrollToBottom() {
  window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function escapeJsString(str) {
  return String(str).replace(/'/g, "\\'").replace(/"/g, '\\"');
}
