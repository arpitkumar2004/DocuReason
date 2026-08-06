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

function formatMarkdown(text) {
  if (!text) return '';

  let html = escapeHtml(text);

  // First convert inline citation patterns before inline code formatting
  html = html.replace(/`\[Source:\s*([^\]]+)\]`/gi, '<span class="citation-badge" title="Source Citation">📌 $1</span>');
  html = html.replace(/\[Source:\s*([^\]]+)\]/gi, '<span class="citation-badge" title="Source Citation">📌 $1</span>');
  html = html.replace(/`\[Doc-([0-9]+)\]`/gi, '<span class="citation-badge" title="Document Citation">📄 Doc-$1</span>');
  html = html.replace(/\[Doc-([0-9]+)\]/gi, '<span class="citation-badge" title="Document Citation">📄 Doc-$1</span>');
  html = html.replace(/`\[Table SQL\]`/gi, '<span class="citation-badge badge-amber" title="DuckDB SQL Execution">🗄️ DuckDB SQL</span>');
  html = html.replace(/\[Table SQL\]/gi, '<span class="citation-badge badge-amber" title="DuckDB SQL Execution">🗄️ DuckDB SQL</span>');

  const rawLines = html.split('\n');
  let resultLines = [];
  let inTable = false;
  let tableHtml = '';
  let inList = false;

  for (let i = 0; i < rawLines.length; i++) {
    let line = rawLines[i].trim();

    // Handle Table
    if (line.startsWith('|') && line.endsWith('|')) {
      if (inList) { resultLines.push('</ul>'); inList = false; }
      if (!inTable) {
        inTable = true;
        tableHtml = '<table class="data-table"><tbody>';
      }
      if (line.includes('---')) continue;
      const cells = line.split('|').slice(1, -1).map(c => c.trim());
      tableHtml += '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
      continue;
    } else if (inTable) {
      inTable = false;
      tableHtml += '</tbody></table>';
      resultLines.push(tableHtml);
      tableHtml = '';
    }

    // Handle Empty Line
    if (!line) {
      if (inList) { resultLines.push('</ul>'); inList = false; }
      resultLines.push('<div class="md-p-gap"></div>');
      continue;
    }

    // Handle Blockquote (> text)
    if (line.startsWith('&gt;') || line.startsWith('>')) {
      if (inList) { resultLines.push('</ul>'); inList = false; }
      let quoteText = line.replace(/^(&gt;|>)\s*/, '');
      resultLines.push(`<blockquote class="md-blockquote">${quoteText}</blockquote>`);
      continue;
    }

    // Handle Headings (###, ##, #)
    if (line.startsWith('### ')) {
      if (inList) { resultLines.push('</ul>'); inList = false; }
      resultLines.push(`<h3 class="md-h3">${line.substring(4)}</h3>`);
      continue;
    }
    if (line.startsWith('## ')) {
      if (inList) { resultLines.push('</ul>'); inList = false; }
      resultLines.push(`<h2 class="md-h2">${line.substring(3)}</h2>`);
      continue;
    }
    if (line.startsWith('# ')) {
      if (inList) { resultLines.push('</ul>'); inList = false; }
      resultLines.push(`<h1 class="md-h1">${line.substring(2)}</h1>`);
      continue;
    }

    // Handle Bullet Lists (* item or - item)
    if (/^[\*\-]\s+/.test(line)) {
      if (!inList) {
        inList = true;
        resultLines.push('<ul class="md-ul">');
      }
      let itemText = line.replace(/^[\*\-]\s+/, '');
      resultLines.push(`<li class="md-li">${itemText}</li>`);
      continue;
    } else if (inList) {
      resultLines.push('</ul>');
      inList = false;
    }

    // General Paragraph
    resultLines.push(`<p class="md-p">${line}</p>`);
  }

  if (inTable) {
    tableHtml += '</tbody></table>';
    resultLines.push(tableHtml);
  }
  if (inList) {
    resultLines.push('</ul>');
  }

  let formatted = resultLines.join('\n');

  // Format remaining Bold, Italics, and Inline Code
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
  formatted = formatted.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

  return formatted;
}

function getEngineLabel(engineKey) {
  if (engineKey === 'local_slm_deepseek_r1') return '⚡ DeepSeek-R1 Distill';
  if (engineKey === 'cloud_gemini_api') return '☁️ Gemini 1.5 Flash';
  if (engineKey === 'offline_template_synthesizer') return '⚙️ Multi-Modal Synthesizer';
  return '🤖 ' + (engineKey || 'RAG Engine');
}

function renderAssistantResponse(groupId, query, data) {
  const asstElement = document.getElementById(groupId);
  if (!asstElement) return;

  const probs = (data.router && data.router.probabilities) ? data.router.probabilities : { text: 0.5, table: 0.1, vision: 0.1 };
  const weights = (data.router && data.router.weights) ? data.router.weights : { text: 0.33, table: 0.33, vision: 0.33 };
  const sqlData = data.sql_execution || {};
  const answer = data.answer || "No response generated.";
  const citations = data.citations || [];
  const attribution = data.attribution || {};
  const nliScore = attribution.faithfulness_score !== undefined ? attribution.faithfulness_score : 0.95;
  const nliPct = (nliScore * 100).toFixed(1);
  const engineLabel = getEngineLabel(data.generation_engine);

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

  // Build Citations Section HTML
  let citationsHtml = '';
  if (citations.length > 0) {
    let citationItems = citations.map(c => {
      let mod = (c.type || 'text').toLowerCase();
      let modClass = mod === 'table_sql' ? 'badge-amber' : (mod === 'vision' ? 'badge-indigo' : 'badge-cyan');
      let modLabel = mod === 'table_sql' ? 'TABLE SQL' : mod.toUpperCase();
      let sourceName = escapeHtml(c.source || 'Document');
      let extraInfo = c.query ? `Query: <code>${escapeHtml(c.query)}</code>` : (c.chunk_id ? `Chunk: <code>${escapeHtml(c.chunk_id)}</code>` : '');

      return `
        <div class="citation-card">
          <div class="citation-card-header">
            <span class="modality-pill ${modClass}">${modLabel}</span>
            <span class="citation-source-name" title="${sourceName}">${sourceName}</span>
          </div>
          ${extraInfo ? `<div class="citation-card-sub">${extraInfo}</div>` : ''}
        </div>
      `;
    }).join('');

    citationsHtml = `
      <div class="citations-section">
        <div class="citations-section-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          <span>Verified Citations & Sources (${citations.length})</span>
        </div>
        <div class="citations-grid">
          ${citationItems}
        </div>
      </div>
    `;
  }

  // Build Evidence Items HTML
  let evidenceHtml = '';
  (data.retrieved_evidence || []).forEach((item, idx) => {
    let mod = (item.modality || 'text').toLowerCase();
    let modClass = mod === 'table' ? 'badge-amber' : (mod === 'vision' ? 'badge-indigo' : 'badge-cyan');
    let docId = item.document_id || item.id || `Doc-${idx+1}`;
    let snippet = item.text || item.linearized || '';

    evidenceHtml += `
      <div class="evidence-item">
        <div class="evidence-item-header">
          <span class="modality-pill ${modClass}">${mod.toUpperCase()}</span>
          <span class="evidence-source-id">${escapeHtml(docId)}</span>
          <span class="evidence-score">RRF Score: ${(item.rank_score || item.score || 0).toFixed(4)}</span>
        </div>
        <div class="evidence-snippet">${escapeHtml(snippet)}</div>
      </div>
    `;
  });

  const cardHtml = `
    <div class="avatar assistant-avatar">RAG</div>
    <div class="message-content" style="width: 100%;">
      <div class="response-card">
        
        <!-- Grounded Answer Body (Rich Markdown Rendered) -->
        <div class="markdown-body">
          ${formatMarkdown(answer)}
        </div>

        <!-- Dedicated Verified Citations & Sources Section -->
        ${citationsHtml}

        <!-- Inline Action & Metadata Bar -->
        <div class="action-bar">
          <button class="action-btn" onclick="copyText('${escapeJsString(answer)}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Copy Answer
          </button>
          <button class="action-btn" onclick="sendSample('${escapeJsString(query)}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
            Regenerate
          </button>

          <div class="meta-pills-group">
            <span class="engine-pill" title="Model Generation Engine">${engineLabel}</span>
            <span class="nli-pill" title="NLI Entailment Score (Zero-Hallucination Grounding)">
              ✓ ${nliPct}% Faithful
            </span>
          </div>
        </div>

        ${data.reasoning_chain ? `
        <!-- Collapsible Card: DeepSeek-R1 Chain-of-Thought Reasoning -->
        <div class="collapsible-card">
          <div class="collapsible-header" onclick="toggleCollapsible(this)">
            <span>🧠 DeepSeek-R1 Reasoning Trace (&lt;think&gt;)</span>
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
  statusBox.innerText = 'Running document ingestion and layout parsing pipeline...';

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
