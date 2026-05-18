/**
 * MojoLlama Studio — Console Log Renderer
 *
 * Displays real-time server output with filtering, search, and
 * auto-scrolling support.
 */

const api = window.mojollama;
let entries = [];            // all log entries
let filteredEntries = [];    // entries after filter + search
let activeLevels = new Set(['all', 'info', 'warn', 'error', 'debug']);
let statusUnsubscribe = null;
let logUnsubscribe = null;
let clearUnsubscribe = null;
let autoScroll = true;

// ─── Init ────────────────────────────────────────────────────────────────────

async function init() {
  // Load existing log buffer
  try {
    const buffer = await api.logGet();
    entries = buffer || [];
    applyFilter();
  } catch (err) {
    console.error('Failed to load log buffer:', err);
  }

  // Listen for new entries
  if (api.onLogEntry) {
    logUnsubscribe = api.onLogEntry((entry) => {
      entries.push(entry);
      if (shouldShowEntry(entry)) {
        filteredEntries.push(entry);
        appendEntryDOM(entry);
        updateCounts();
        if (autoScroll) scrollToBottom();
      }
    });
  }

  // Listen for log cleared
  if (api.onLogCleared) {
    clearUnsubscribe = api.onLogCleared(() => {
      entries = [];
      filteredEntries = [];
      document.getElementById('logContainer').innerHTML =
        '<div class="empty-state">Log cleared.</div>';
      updateCounts();
    });
  }

  // Listen for server status
  if (api.onServerStatus) {
    statusUnsubscribe = api.onServerStatus(updateStatus);
  }

  // Fetch current status
  try {
    const status = await api.serverStatus();
    updateStatus(status);
  } catch (err) {
    console.error('Failed to get server status:', err);
  }
}

// ─── Filtering ───────────────────────────────────────────────────────────────

function shouldShowEntry(entry) {
  // Level filter
  if (!activeLevels.has('all') && !activeLevels.has(entry.level)) {
    return false;
  }

  // Search text filter
  const searchText = document.getElementById('searchInput').value.toLowerCase().trim();
  if (searchText) {
    const text = (entry.text || '').toLowerCase();
    const level = (entry.level || '').toLowerCase();
    const ts = (entry.timestamp || '').toLowerCase();
    if (!text.includes(searchText) && !level.includes(searchText) && !ts.includes(searchText)) {
      return false;
    }
  }

  return true;
}

function applyFilter() {
  filteredEntries = entries.filter(e => shouldShowEntry(e));

  const container = document.getElementById('logContainer');
  container.innerHTML = '';

  if (filteredEntries.length === 0) {
    container.innerHTML = entries.length === 0
      ? '<div class="empty-state">Waiting for server output...</div>'
      : '<div class="empty-state">No entries match the current filter.</div>';
    updateCounts();
    return;
  }

  // Batch DOM append for performance
  const fragment = document.createDocumentFragment();
  for (const entry of filteredEntries) {
    fragment.appendChild(createEntryElement(entry));
  }
  container.appendChild(fragment);
  updateCounts();

  if (autoScroll) scrollToBottom();
}

function toggleFilter(btn, level) {
  btn.classList.toggle('active');
  if (activeLevels.has(level)) {
    activeLevels.delete(level);
  } else {
    activeLevels.add(level);
  }
  applyFilter();
}

// ─── DOM rendering ───────────────────────────────────────────────────────────

function createEntryElement(entry) {
  const div = document.createElement('div');
  div.className = 'log-entry level-' + (entry.level || 'info');

  // Timestamp
  const ts = document.createElement('span');
  ts.className = 'timestamp';
  try {
    const d = new Date(entry.timestamp);
    ts.textContent = d.toLocaleTimeString('en-US', { hour12: false }) + '.' +
      String(d.getMilliseconds()).padStart(3, '0');
  } catch (e) {
    ts.textContent = '--:--:--';
  }
  div.appendChild(ts);

  // Level tag
  const tag = document.createElement('span');
  tag.className = 'level-tag';
  tag.textContent = (entry.level || 'info').toUpperCase();
  div.appendChild(tag);

  // Message
  const msg = document.createElement('span');
  msg.className = 'message';
  msg.textContent = entry.text || '';
  div.appendChild(msg);

  // Click to copy
  div.addEventListener('dblclick', () => {
    const text = `${entry.timestamp} [${entry.level}] ${entry.text}`;
    navigator.clipboard.writeText(text).catch(() => {});
  });

  return div;
}

function appendEntryDOM(entry) {
  const container = document.getElementById('logContainer');

  // Remove empty state if present
  const emptyState = container.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  container.appendChild(createEntryElement(entry));
}

function updateCounts() {
  document.getElementById('entryCount').textContent =
    `${filteredEntries.length} / ${entries.length} entries`;
}

function scrollToBottom() {
  const container = document.getElementById('logContainer');
  requestAnimationFrame(() => {
    container.scrollTop = container.scrollHeight;
  });
}

// ─── Auto-scroll ─────────────────────────────────────────────────────────────

// Detect if user scrolled up
document.getElementById('logContainer').addEventListener('scroll', () => {
  const container = document.getElementById('logContainer');
  const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 50;
  autoScroll = atBottom;
  document.getElementById('autoScrollLabel').style.opacity = autoScroll ? '1' : '0.5';
});

// ─── Status ──────────────────────────────────────────────────────────────────

function updateStatus(status) {
  if (!status) return;
  const dot = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');
  dot.className = 'badge-dot ' + status.state;
  label.textContent = `Server: ${status.state}`;
  if (status.state === 'running') {
    label.textContent = `Server: running on port ${status.port}`;
  }
}

// ─── Actions ─────────────────────────────────────────────────────────────────

async function clearLog() {
  try {
    await api.logClear();
    // The 'log-cleared' event will handle the UI
  } catch (err) {
    console.error('Failed to clear log:', err);
  }
}

// ─── Keyboard shortcuts ──────────────────────────────────────────────────────

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') window.close();
  if (e.key === 'l' && (e.ctrlKey || e.metaKey)) clearLog();
  if (e.key === 'f' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    document.getElementById('searchInput').focus();
  }
});

// ─── Init ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
