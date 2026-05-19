/**
 * MojoLlama Studio — Settings Renderer
 *
 * Manages the settings window UI, loads/saves settings via IPC,
 * and provides inline server control.
 */

const api = window.mojollama;
let currentSettings = {};
let statusUnsubscribe = null;

// ─── Initialization ──────────────────────────────────────────────────────────

async function init() {
  // Load settings
  try {
    currentSettings = await api.settingsLoad();
    populateForm(currentSettings);
  } catch (err) {
    console.error('Failed to load settings:', err);
  }

  // Listen for server status updates
  if (api.onServerStatus) {
    statusUnsubscribe = api.onServerStatus(updateStatusUI);
  }

  // Fetch current status
  try {
    const status = await api.serverStatus();
    updateStatusUI(status);
  } catch (err) {
    console.error('Failed to get server status:', err);
  }
}

function populateForm(settings) {
  document.getElementById('serverPort').value = settings.serverPort || 8080;
  document.getElementById('llamaPort').value = settings.llamaPort || 8081;
  document.getElementById('modelPath').value = settings.modelPath || '';
  document.getElementById('autoStart').checked = settings.autoStart !== false;
  document.getElementById('minimizeToTray').checked = settings.minimizeToTray !== false;
  document.getElementById('notifications').checked = settings.notifications !== false;
  document.getElementById('debug').checked = settings.debug === true;
}

function readForm() {
  return {
    serverPort: parseInt(document.getElementById('serverPort').value, 10) || 8080,
    llamaPort: parseInt(document.getElementById('llamaPort').value, 10) || 8081,
    modelPath: document.getElementById('modelPath').value.trim(),
    autoStart: document.getElementById('autoStart').checked,
    minimizeToTray: document.getElementById('minimizeToTray').checked,
    notifications: document.getElementById('notifications').checked,
    debug: document.getElementById('debug').checked,
  };
}

// ─── Server Control ──────────────────────────────────────────────────────────

async function startServer() {
  const form = readForm();
  document.getElementById('btnStart').disabled = true;
  document.getElementById('btnStart').textContent = 'Starting...';
  try {
    await api.serverStart({
      port: form.serverPort,
      modelPath: form.modelPath || undefined,
    });
  } catch (err) {
    console.error('Failed to start server:', err);
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnStart').textContent = 'Start';
  }
}

async function stopServer() {
  document.getElementById('btnStop').disabled = true;
  document.getElementById('btnStop').textContent = 'Stopping...';
  try {
    await api.serverStop();
  } catch (err) {
    console.error('Failed to stop server:', err);
    document.getElementById('btnStop').disabled = false;
    document.getElementById('btnStop').textContent = 'Stop';
  }
}

async function restartServer() {
  try {
    await api.serverRestart();
  } catch (err) {
    console.error('Failed to restart server:', err);
  }
}

// ─── Status UI ───────────────────────────────────────────────────────────────

function updateStatusUI(status) {
  if (!status) return;

  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  const btnStart = document.getElementById('btnStart');
  const btnStop = document.getElementById('btnStop');
  const btnRestart = document.getElementById('btnRestart');

  dot.className = 'status-dot ' + status.state;

  switch (status.state) {
    case 'running':
      text.textContent = `Running on port ${status.port} (PID: ${status.pid || 'N/A'})`;
      btnStart.disabled = true;
      btnStart.textContent = 'Start';
      btnStop.disabled = false;
      btnStop.textContent = 'Stop';
      btnRestart.disabled = false;
      break;
    case 'starting':
      text.textContent = 'Server is starting...';
      btnStart.disabled = true;
      btnStart.textContent = 'Starting...';
      btnStop.disabled = true;
      btnRestart.disabled = true;
      break;
    case 'stopping':
      text.textContent = 'Server is stopping...';
      btnStart.disabled = true;
      btnStop.disabled = true;
      btnStop.textContent = 'Stopping...';
      btnRestart.disabled = true;
      break;
    case 'error':
      text.textContent = 'Server encountered an error. Check the log.';
      btnStart.disabled = false;
      btnStart.textContent = 'Start';
      btnStop.disabled = true;
      btnStop.textContent = 'Stop';
      btnRestart.disabled = false;
      break;
    default: // stopped
      text.textContent = 'Server is stopped.';
      btnStart.disabled = false;
      btnStart.textContent = 'Start';
      btnStop.disabled = true;
      btnStop.textContent = 'Stop';
      btnRestart.disabled = true;
      break;
  }
}

// ─── Actions ─────────────────────────────────────────────────────────────────

async function saveSettings() {
  const newSettings = readForm();
  try {
    const result = await api.settingsSave(newSettings);
    if (result.success) {
      currentSettings = newSettings;
      showToast('Settings saved successfully.', 'success');
    } else {
      showToast('Failed to save settings.', 'error');
    }
  } catch (err) {
    console.error('Failed to save settings:', err);
    showToast('Error saving settings: ' + err.message, 'error');
  }
}

function cancel() {
  // Electron modal will close automatically or we can close via IPC
  // Since we're modal, the window closes on ESC or Cmd+W normally
  window.close();
}

function openLog() {
  if (api.openLogWindow) {
    api.openLogWindow();
  }
}

// ─── Toast ───────────────────────────────────────────────────────────────────

function showToast(msg, type) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const el = document.createElement('div');
  el.className = 'toast ' + (type || 'info');
  el.textContent = msg;
  el.style.cssText = `
    position: fixed; bottom: 16px; right: 16px; padding: 10px 16px;
    border-radius: 6px; font-size: 12px; font-weight: 500; z-index: 1000;
    background: ${type === 'success' ? '#1f3f2a' : type === 'error' ? '#3f1f1f' : '#1f2a3f'};
    border: 1px solid ${type === 'success' ? 'var(--green)' : type === 'error' ? 'var(--red)' : 'var(--accent)'};
    color: ${type === 'success' ? 'var(--green)' : type === 'error' ? 'var(--red)' : 'var(--accent)'};
    animation: slideUp .3s ease;
  `;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

// ─── Keyboard shortcuts ──────────────────────────────────────────────────────

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    window.close();
  }
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    saveSettings();
  }
});

// ─── Init ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
