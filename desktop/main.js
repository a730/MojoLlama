/**
 * MojoLlama Studio — Electron Main Process
 *
 * Manages the desktop window, server lifecycle, menu, tray,
 * settings persistence, and IPC communication.
 */

const { app, BrowserWindow, Menu, Tray, nativeImage, dialog, Notification, ipcMain, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, execSync } = require('child_process');
const http = require('http');

// ─── Constants ───────────────────────────────────────────────────────────────

const APP_NAME = 'MojoLlama Studio';
const STUDIO_URL = '/studio.html';
const CHAT_URL = '/chat.html';
const SETTINGS_FILE = 'mojollama-settings.json';
const DEFAULT_PORT = 8080;
const DEFAULT_LLAMA_PORT = 8081;
const DEFAULT_THREADS = 32;
const POLL_INTERVAL = 2000;       // ms between server health checks
const MAX_LOG_LINES = 5000;       // ring buffer for server output

// ─── State ──────────────────────────────────────────────────────────────────

let mainWindow = null;
let tray = null;
let serverProcess = null;
let serverState = 'stopped';      // 'stopped' | 'starting' | 'running' | 'stopping' | 'error'
let settings = {};
let logBuffer = [];               // ring buffer of { timestamp, level, text }
let logWindows = new Set();       // set of BrowserWindow IDs for log windows
let settingsWindow = null;
let healthInterval = null;
let serverPort = DEFAULT_PORT;
let projectRoot = null;           // auto-detected from app location

// ─── Path resolution ────────────────────────────────────────────────────────

/**
 * Determine the project root directory.
 * When running in dev: use the parent of the desktop/ directory.
 * When running in production: use process.resourcesPath.
 */
function getProjectRoot() {
  if (projectRoot) return projectRoot;

  if (process.env.ELECTRON_DEV || process.env.NODE_ENV === 'development') {
    // In dev, the project root is the parent of desktop/
    projectRoot = path.resolve(__dirname, '..');
  } else {
    // In production (packaged app), resources are at process.resourcesPath
    projectRoot = process.resourcesPath;
  }

  // Verify the project root has server.py
  const serverPy = path.join(projectRoot, 'server.py');
  if (!fs.existsSync(serverPy)) {
    // Try alternative: some packagings put it in extraResources
    const altRoot = path.resolve(__dirname, '..');
    if (fs.existsSync(path.join(altRoot, 'server.py'))) {
      projectRoot = altRoot;
    }
  }

  return projectRoot;
}

/**
 * Find the server script to execute.
 * Returns the command and args array.
 */
function getServerCommand(overrides = {}) {
  const root = getProjectRoot();
  const serverPy = path.join(root, 'src', 'mojollama', 'server_moe.py');
  const port = overrides.port || settings.serverPort || DEFAULT_PORT;
  const modelPath = overrides.modelPath || settings.modelPath || '';

  // Use python3 to run server_moe.py (positional args: model_path port)
  const pythonCmd = findPython();
  const resolvedModel = modelPath && fs.existsSync(modelPath)
    ? modelPath
    : path.join(root, 'models', 'Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf');
  const args = [serverPy, resolvedModel, String(port)];

  // Add threads override if set in settings
  if (settings.threads && settings.threads !== DEFAULT_THREADS) {
    // server_moe reads OMP_NUM_THREADS from env
  }

  // Check for debug mode in settings
  if (settings.debug) {
    args.push('--debug');
  }

  return { cmd: pythonCmd, args, port, serverPy };
}

/**
 * Find a working Python 3 executable.
 */
function findPython() {
  const candidates = ['python3', 'python'];
  for (const cmd of candidates) {
    try {
      const out = execSync(`${cmd} --version`, { encoding: 'utf8' });
      if (out.toLowerCase().includes('python 3')) {
        return cmd;
      }
    } catch (e) {
      continue;
    }
  }
  return 'python3'; // fallback
}

// ─── Settings persistence ────────────────────────────────────────────────────

function getSettingsPath() {
  const userDataPath = app.getPath('userData');
  return path.join(userDataPath, SETTINGS_FILE);
}

function loadSettings() {
  const settingsPath = getSettingsPath();
  const defaults = {
    serverPort: DEFAULT_PORT,
    llamaPort: DEFAULT_LLAMA_PORT,
    modelPath: '',
    threads: DEFAULT_THREADS,
    engine: 'server_batch_moe',
    autoStart: true,
    debug: false,
    minimizeToTray: true,
    notifications: true,
  };

  try {
    if (fs.existsSync(settingsPath)) {
      const data = fs.readFileSync(settingsPath, 'utf8');
      const parsed = JSON.parse(data);
      settings = { ...defaults, ...parsed };
    } else {
      settings = { ...defaults };
    }
  } catch (err) {
    console.error('Failed to load settings:', err);
    settings = { ...defaults };
  }

  serverPort = settings.serverPort || DEFAULT_PORT;
  return settings;
}

function saveSettings(newSettings) {
  settings = { ...settings, ...newSettings };
  serverPort = settings.serverPort || DEFAULT_PORT;
  const settingsPath = getSettingsPath();

  try {
    fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2), 'utf8');
    return true;
  } catch (err) {
    console.error('Failed to save settings:', err);
    return false;
  }
}

// ─── Logging ─────────────────────────────────────────────────────────────────

function appendLog(level, text) {
  const entry = {
    timestamp: new Date().toISOString(),
    level,
    text,
  };
  logBuffer.push(entry);
  if (logBuffer.length > MAX_LOG_LINES) {
    logBuffer = logBuffer.slice(-MAX_LOG_LINES);
  }

  // Broadcast to all log windows
  for (const winId of logWindows) {
    try {
      const win = BrowserWindow.fromId(winId);
      if (win && !win.isDestroyed()) {
        win.webContents.send('server-log', entry);
      }
    } catch (e) {
      logWindows.delete(winId);
    }
  }

  // Also send to main window if it exists
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('server-log', entry);
  }
}

function getLogBuffer() {
  return logBuffer;
}

function clearLogBuffer() {
  logBuffer = [];
  for (const winId of logWindows) {
    try {
      const win = BrowserWindow.fromId(winId);
      if (win && !win.isDestroyed()) {
        win.webContents.send('log-cleared');
      }
    } catch (e) {
      logWindows.delete(winId);
    }
  }
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('log-cleared');
  }
}

// ─── Server management ───────────────────────────────────────────────────────

function setServerState(newState) {
  serverState = newState;
  updateMenu();
  updateTray();
  broadcastServerStatus();
}

function broadcastServerStatus() {
  const status = {
    state: serverState,
    port: serverPort,
    pid: serverProcess ? serverProcess.pid : null,
  };
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('server-status', status);
  }
  for (const winId of logWindows) {
    try {
      const win = BrowserWindow.fromId(winId);
      if (win && !win.isDestroyed()) {
        win.webContents.send('server-status', status);
      }
    } catch (e) {
      logWindows.delete(winId);
    }
  }
}

function startServer(overrides = {}) {
  if (serverProcess) {
    appendLog('warn', 'Server is already running or starting.');
    return;
  }

  setServerState('starting');
  appendLog('info', 'Starting MojoLlama server...');

  const { cmd, args, port } = getServerCommand(overrides);
  appendLog('debug', `Command: ${cmd} ${args.join(' ')}`);
  serverPort = port;

  const root = getProjectRoot();
  const env = {
    ...process.env,
    PORT: String(port),
    OMP_NUM_THREADS: String(settings.threads || DEFAULT_THREADS),
  };

  // If model path is set in settings, pass as environment variable
  if (settings.modelPath) {
    env.MODEL_PATH = settings.modelPath;
  }

  try {
    serverProcess = spawn(cmd, args, {
      cwd: root,
      env,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    serverProcess.stdout.on('data', (data) => {
      const lines = data.toString().split('\n').filter(l => l.trim());
      for (const line of lines) {
        appendLog('info', line);
      }
    });

    serverProcess.stderr.on('data', (data) => {
      const lines = data.toString().split('\n').filter(l => l.trim());
      for (const line of lines) {
        // stderr might have warnings or errors
        const isError = line.toLowerCase().includes('error') || line.toLowerCase().includes('traceback');
        appendLog(isError ? 'error' : 'warn', line);
      }
    });

    serverProcess.on('error', (err) => {
      appendLog('error', `Failed to start server: ${err.message}`);
      setServerState('error');
      serverProcess = null;
      showNotification('Server Error', `Failed to start: ${err.message}`);
    });

    serverProcess.on('exit', (code, signal) => {
      const reason = signal ? `signal ${signal}` : `exit code ${code}`;
      appendLog('info', `Server process exited (${reason})`);

      if (serverState === 'stopping') {
        appendLog('info', 'Server stopped gracefully.');
        setServerState('stopped');
      } else if (code !== 0) {
        appendLog('error', `Server crashed with exit code ${code}`);
        setServerState('error');
        showNotification('Server Crashed', `Server exited with code ${code}. Check the log for details.`);
      } else {
        setServerState('stopped');
      }

      serverProcess = null;
      stopHealthCheck();
    });

    // Start health check polling
    startHealthCheck(port);

  } catch (err) {
    appendLog('error', `Failed to spawn server: ${err.message}`);
    setServerState('error');
    showNotification('Server Error', `Failed to start: ${err.message}`);
  }
}

function stopServer() {
  if (!serverProcess) {
    appendLog('warn', 'Server is not running.');
    return;
  }

  setServerState('stopping');
  appendLog('info', 'Stopping MojoLlama server...');

  // Send SIGTERM first, then SIGKILL after timeout
  const killTimeout = setTimeout(() => {
    if (serverProcess) {
      appendLog('warn', 'Server did not stop gracefully, sending SIGKILL...');
      serverProcess.kill('SIGKILL');
    }
  }, 8000);

  serverProcess.once('exit', () => {
    clearTimeout(killTimeout);
  });

  serverProcess.kill('SIGTERM');
}

function restartServer() {
  appendLog('info', 'Restarting server...');
  if (serverProcess) {
    stopServer();
    // Give it a moment to fully exit before restarting
    const checkInterval = setInterval(() => {
      if (!serverProcess) {
        clearInterval(checkInterval);
        // Small delay to let the port release
        setTimeout(() => startServer(), 500);
      }
    }, 200);
  } else {
    startServer();
  }
}

// ─── Health check ────────────────────────────────────────────────────────────

function startHealthCheck(port) {
  stopHealthCheck();
  healthInterval = setInterval(() => {
    checkServerHealth(port);
  }, POLL_INTERVAL);
}

function stopHealthCheck() {
  if (healthInterval) {
    clearInterval(healthInterval);
    healthInterval = null;
  }
}

function checkServerHealth(port) {
  const req = http.get(`http://127.0.0.1:${port}/v1/models`, (res) => {
    if (res.statusCode === 200 && serverState === 'starting') {
      appendLog('info', `Server is ready on port ${port}`);
      setServerState('running');
      showNotification('Server Ready', `MojoLlama server is running on port ${port}`);
      // Auto-reload main window to show the studio page
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.loadURL(`http://localhost:${port}${STUDIO_URL}`);
      }
    } else if (res.statusCode !== 200 && serverState === 'running') {
      appendLog('warn', `Server health check returned ${res.statusCode}`);
    }
  });

  req.on('error', () => {
    // Don't log connection errors during startup — they're expected
    if (serverState === 'running') {
      appendLog('warn', 'Lost connection to server. Is it still running?');
    }
  });

  req.setTimeout(3000, () => {
    req.destroy();
  });
}

// ─── Notifications ──────────────────────────────────────────────────────────

function showNotification(title, body) {
  if (!settings.notifications) return;
  try {
    const notif = new Notification({ title, body, icon: getIconPath() });
    notif.show();
    return notif;
  } catch (err) {
    console.error('Notification failed:', err);
  }
}

function getIconPath() {
  const iconPath = path.join(__dirname, 'icons', 'icon.png');
  if (fs.existsSync(iconPath)) return iconPath;
  return undefined;
}

// ─── Window creation ─────────────────────────────────────────────────────────

function createMainWindow(url) {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: APP_NAME,
    icon: getIconPath(),
    backgroundColor: '#0d1117',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      webviewTag: false,
    },
  });

  // Center window
  win.center();

  // Load www/studio.html directly from disk (no server needed for static UI)
  const studioPath = path.join(getProjectRoot(), 'www', 'studio.html');
  if (fs.existsSync(studioPath)) {
    win.loadFile(studioPath);
  } else {
    // Fallback: try loading from server if it's running
    const targetUrl = url || getTargetUrl();
    win.loadURL(targetUrl);
  }

  win.once('ready-to-show', () => {
    win.show();
    if (settings.autoStart && serverState === 'stopped') {
      // Small delay to let the window render first
      setTimeout(() => startServer(), 1000);
    }
  });

  // Handle load failures gracefully (server not ready yet)
  win.webContents.on('did-fail-load', (event, errorCode, errorDescription, validatedURL) => {
    // Only show connecting page if server isn't running yet
    if (serverState !== 'running' && validatedURL && validatedURL.includes('localhost')) {
      win.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(`
        <!DOCTYPE html>
        <html lang="en">
        <head><meta charset="UTF-8"><title>${APP_NAME}</title>
        <style>
          body {
            background: #0d1117; color: #e6edf3; font-family: -apple-system, sans-serif;
            display: flex; align-items: center; justify-content: center;
            height: 100vh; margin: 0; flex-direction: column; gap: 20px;
          }
          .spinner {
            width: 40px; height: 40px; border: 3px solid #30363d;
            border-top-color: #58a6ff; border-radius: 50%;
            animation: spin 0.8s linear infinite;
          }
          @keyframes spin { to { transform: rotate(360deg); } }
          h2 { font-size: 18px; font-weight: 600; color: #e6edf3; }
          p { font-size: 13px; color: #8b949e; text-align: center; max-width: 400px; line-height: 1.5; }
          .btn { padding: 8px 20px; border: none; border-radius: 6px; cursor: pointer;
                 font-size: 13px; font-weight: 600; background: #58a6ff; color: #fff; }
          .btn:hover { filter: brightness(1.15); }
          .btn-start { background: #3fb950; }
          .btn-start:hover { filter: brightness(1.15); }
        </style>
        </head>
        <body>
          <div class="spinner"></div>
          <h2>MojoLlama server is not running</h2>
          <p>Start the server to access the full Studio experience, or browse the UI in offline mode.</p>
          <div style="display:flex; gap: 10px;">
            <button class="btn btn-start" onclick="window.mojollama.serverStart()">Start Server</button>
            <button class="btn" onclick="location.reload()">Retry</button>
          </div>
        </body>
        </html>
      `)}`).catch(() => {});
    }
  });

  win.on('close', (event) => {
    if (settings.minimizeToTray && tray) {
      event.preventDefault();
      win.hide();
    }
  });

  win.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow = win;
  return win;
}

function getTargetUrl() {
  // Allow overriding via command-line argument
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--url' && args[i + 1]) {
      return args[i + 1];
    }
    // Accept a standalone URL as the first argument
    if (args[i].startsWith('http://') || args[i].startsWith('https://')) {
      return args[i];
    }
  }
  return `http://localhost:${serverPort}/studio.html`;
}

function createSettingsWindow() {
  if (settingsWindow && !settingsWindow.isDestroyed()) {
    settingsWindow.focus();
    return;
  }

  const win = new BrowserWindow({
    width: 600,
    height: 520,
    resizable: false,
    title: `${APP_NAME} — Settings`,
    icon: getIconPath(),
    parent: mainWindow,
    modal: true,
    backgroundColor: '#0d1117',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  win.loadFile(path.join(__dirname, 'renderer', 'settings.html'));

  win.on('closed', () => {
    settingsWindow = null;
  });

  settingsWindow = win;
}

function createLogWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 500,
    title: `${APP_NAME} — Console Log`,
    icon: getIconPath(),
    backgroundColor: '#0d1117',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  win.loadFile(path.join(__dirname, 'renderer', 'log.html'));

  win.on('closed', () => {
    logWindows.delete(win.id);
  });

  logWindows.add(win.id);
  return win;
}

// ─── Menu bar ────────────────────────────────────────────────────────────────

function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Studio',
          accelerator: 'CmdOrCtrl+1',
          click: () => {
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.loadURL(`http://localhost:${serverPort}${STUDIO_URL}`);
              mainWindow.show();
            }
          },
        },
        {
          label: 'Open Chat',
          accelerator: 'CmdOrCtrl+2',
          click: () => {
            if (mainWindow && !mainWindow.isDestroyed()) {
              mainWindow.loadURL(`http://localhost:${serverPort}${CHAT_URL}`);
              mainWindow.show();
            }
          },
        },
        { type: 'separator' },
        {
          label: 'Settings',
          accelerator: 'CmdOrCtrl+,',
          click: () => createSettingsWindow(),
        },
        { type: 'separator' },
        {
          label: 'Quit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Alt+F4',
          click: () => {
            if (settings.minimizeToTray && tray) {
              // Quit for real
              if (serverProcess) {
                stopServer();
              }
              app.quit();
            } else {
              app.quit();
            }
          },
        },
      ],
    },
    {
      label: 'Server',
      id: 'server-menu',
      submenu: [
        {
          label: 'Start Server',
          id: 'start-server',
          accelerator: 'CmdOrCtrl+S',
          click: () => startServer(),
        },
        {
          label: 'Stop Server',
          id: 'stop-server',
          accelerator: 'CmdOrCtrl+Shift+S',
          click: () => stopServer(),
        },
        {
          label: 'Restart Server',
          id: 'restart-server',
          accelerator: 'CmdOrCtrl+R',
          click: () => restartServer(),
        },
        { type: 'separator' },
        {
          label: 'Settings...',
          click: () => createSettingsWindow(),
        },
        { type: 'separator' },
        {
          label: 'Console Log',
          accelerator: 'CmdOrCtrl+L',
          click: () => createLogWindow(),
        },
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload', label: 'Reload Window' },
        { role: 'forceReload', label: 'Force Reload' },
        { role: 'toggleDevTools', label: 'Toggle Developer Tools' },
        { type: 'separator' },
        { role: 'resetZoom', label: 'Actual Size' },
        { role: 'zoomIn', label: 'Zoom In' },
        { role: 'zoomOut', label: 'Zoom Out' },
        { type: 'separator' },
        { role: 'togglefullscreen', label: 'Toggle Fullscreen' },
      ],
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize', label: 'Minimize' },
        { role: 'zoom', label: 'Zoom' },
        { role: 'close', label: 'Close' },
      ],
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About MojoLlama Studio',
          click: () => {
            dialog.showMessageBox({
              type: 'info',
              title: `About ${APP_NAME}`,
              message: `${APP_NAME} v0.2.0`,
              detail:
                'MojoLlama Studio is a desktop wrapper for the MojoLlama LLM inference engine.\n\n' +
                'Built with Electron + Flask + Mojo 🔥\n\n' +
                'Features:\n' +
                '  • Multi-backend inference (CPU, GPU, MAX)\n' +
                '  • LoRA fine-tuning and merging\n' +
                '  • Dataset creation and auto-labeling\n' +
                '  • Benchmark suite\n' +
                '  • HuggingFace to GGUF export\n\n' +
                'https://git.bamse.cloud/a730/MojoLlama',
              icon: getIconPath(),
            });
          },
        },
        { type: 'separator' },
        {
          label: 'Documentation',
          click: () => shell.openExternal('https://git.bamse.cloud/a730/MojoLlama'),
        },
        {
          label: 'Report Issue',
          click: () => shell.openExternal('https://git.bamse.cloud/a730/MojoLlama/issues'),
        },
        { type: 'separator' },
        {
          label: 'Check for Updates...',
          click: () => {
            const { autoUpdater } = require('electron-updater');
            autoUpdater.checkForUpdates();
            dialog.showMessageBox({
              type: 'info',
              title: 'Checking for Updates',
              message: 'Checking for updates...',
              detail: 'If an update is available, it will be downloaded and installed automatically.',
            });
          },
        },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
  updateMenu();
}

function updateMenu() {
  const menu = Menu.getApplicationMenu();
  if (!menu) return;

  const startItem = menu.getMenuItemById('start-server');
  const stopItem = menu.getMenuItemById('stop-server');
  const restartItem = menu.getMenuItemById('restart-server');

  if (startItem) startItem.enabled = serverState === 'stopped' || serverState === 'error';
  if (stopItem) stopItem.enabled = serverState === 'running' || serverState === 'starting';
  if (restartItem) restartItem.enabled = serverState === 'running' || serverState === 'stopped' || serverState === 'error';
}

// ─── System tray ─────────────────────────────────────────────────────────────

function createTray() {
  const iconPath = getIconPath();
  if (!iconPath) return;

  // Create a proper tray icon (16x16 or 22x22)
  const trayIcon = nativeImage.createFromPath(iconPath);
  // Electron will resize appropriately on each platform

  try {
    tray = new Tray(trayIcon);
    tray.setToolTip(APP_NAME);

    updateTrayContextMenu();
  } catch (err) {
    console.error('Failed to create tray:', err.message);
  }
}

function updateTray() {
  if (!tray) return;
  updateTrayContextMenu();
  const statusText = serverState === 'running'
    ? `Running on port ${serverPort}`
    : serverState === 'starting' ? 'Starting...'
    : serverState === 'stopping' ? 'Stopping...'
    : serverState === 'error' ? 'Error'
    : 'Stopped';
  tray.setToolTip(`${APP_NAME} — ${statusText}`);
}

function updateTrayContextMenu() {
  if (!tray) return;

  const contextMenu = Menu.buildFromTemplate([
    {
      label: `${APP_NAME} (${serverState})`,
      enabled: false,
    },
    { type: 'separator' },
    {
      label: 'Show Window',
      click: () => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.show();
          mainWindow.focus();
        } else {
          createMainWindow();
        }
      },
    },
    {
      label: 'Open Studio',
      click: () => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.loadURL(`http://localhost:${serverPort}${STUDIO_URL}`);
          mainWindow.show();
        }
      },
    },
    {
      label: 'Open Chat',
      click: () => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.loadURL(`http://localhost:${serverPort}${CHAT_URL}`);
          mainWindow.show();
        }
      },
    },
    { type: 'separator' },
    {
      label: serverState === 'running' ? 'Stop Server' : 'Start Server',
      click: () => {
        if (serverState === 'running') {
          stopServer();
        } else {
          startServer();
        }
      },
    },
    {
      label: 'Restart Server',
      enabled: serverState === 'running' || serverState === 'stopped' || serverState === 'error',
      click: () => restartServer(),
    },
    { type: 'separator' },
    {
      label: 'Console Log',
      click: () => createLogWindow(),
    },
    {
      label: 'Settings...',
      click: () => createSettingsWindow(),
    },
    { type: 'separator' },
    {
      label: 'Quit',
      click: () => {
        // Save any unsaved state
        if (serverProcess) {
          stopServer();
        }
        app.quit();
      },
    },
  ]);

  tray.setContextMenu(contextMenu);
  tray.setToolTip(`${APP_NAME} — ${serverState}`);
}

// ─── IPC handlers ────────────────────────────────────────────────────────────

function setupIPC() {
  // Server control
  ipcMain.handle('server:start', async (event, overrides) => {
    startServer(overrides || {});
    return { success: true };
  });

  ipcMain.handle('server:stop', async () => {
    stopServer();
    return { success: true };
  });

  ipcMain.handle('server:restart', async () => {
    restartServer();
    return { success: true };
  });

  ipcMain.handle('server:status', async () => {
    return {
      state: serverState,
      port: serverPort,
      pid: serverProcess ? serverProcess.pid : null,
    };
  });

  // Settings
  ipcMain.handle('settings:load', async () => {
    return loadSettings();
  });

  ipcMain.handle('settings:save', async (event, newSettings) => {
    const success = saveSettings(newSettings);
    // If auto-start setting changed, handle accordingly
    if (newSettings.autoStart !== undefined && newSettings.autoStart && serverState === 'stopped') {
      startServer();
    }
    return { success };
  });

  // Logs
  ipcMain.handle('log:get', async () => {
    return getLogBuffer();
  });

  ipcMain.handle('log:clear', async () => {
    clearLogBuffer();
    return { success: true };
  });

  // Window management
  ipcMain.handle('window:openLog', async () => {
    createLogWindow();
    return { success: true };
  });

  ipcMain.handle('window:openSettings', async () => {
    createSettingsWindow();
    return { success: true };
  });

  // App info
  ipcMain.handle('app:info', async () => {
    return {
      name: APP_NAME,
      version: app.getVersion(),
      electronVersion: process.versions.electron,
      nodeVersion: process.versions.node,
      platform: process.platform,
      arch: process.arch,
      projectRoot: getProjectRoot(),
      pythonPath: findPython(),
      serverState,
      serverPort,
    };
  });

  // Open external link
  ipcMain.handle('shell:openExternal', async (event, url) => {
    if (typeof url === 'string' && (url.startsWith('http://') || url.startsWith('https://'))) {
      shell.openExternal(url);
    }
  });
}

// ─── App lifecycle ───────────────────────────────────────────────────────────

app.whenReady().then(() => {
  loadSettings();
  createMenu();
  setupIPC();

  createMainWindow();

  // Create tray after a small delay to let the app icon load
  setTimeout(() => {
    createTray();
  }, 500);

  app.on('activate', () => {
    if (mainWindow === null) {
      createMainWindow();
    } else {
      mainWindow.show();
    }
  });

  // Setup auto-updater
  if (!process.env.ELECTRON_DEV) {
    try {
      const { autoUpdater } = require('electron-updater');
      autoUpdater.logger = console;
      autoUpdater.autoDownload = true;

      autoUpdater.on('update-available', (info) => {
        appendLog('info', `Update available: v${info.version}`);
        showNotification('Update Available', `Version ${info.version} is downloading...`);
      });

      autoUpdater.on('update-downloaded', (info) => {
        appendLog('info', `Update v${info.version} downloaded`);
        dialog.showMessageBox({
          type: 'info',
          title: 'Update Ready',
          message: `Version ${info.version} has been downloaded.`,
          detail: 'Restart the application to install the update.',
          buttons: ['Restart Now', 'Later'],
        }).then(({ response }) => {
          if (response === 0) {
            autoUpdater.quitAndInstall();
          }
        });
      });

      autoUpdater.on('error', (err) => {
        appendLog('error', `Auto-update error: ${err.message}`);
      });

      // Check for updates on startup (with a delay)
      setTimeout(() => {
        autoUpdater.checkForUpdates().catch(err => {
          console.error('Update check failed:', err.message);
        });
      }, 10000);
    } catch (err) {
      console.error('Auto-updater setup failed:', err.message);
    }
  }
});

app.on('window-all-closed', () => {
  // On macOS, keep the app running in the background (tray)
  if (process.platform !== 'darwin' && !tray) {
    if (serverProcess) {
      stopServer();
    }
    app.quit();
  }
});

app.on('before-quit', () => {
  // Clean up before quitting
  if (serverProcess) {
    try {
      serverProcess.kill('SIGTERM');
    } catch (e) {
      // ignore
    }
  }
  stopHealthCheck();

  // Close log windows
  for (const winId of logWindows) {
    try {
      const win = BrowserWindow.fromId(winId);
      if (win && !win.isDestroyed()) {
        win.close();
      }
    } catch (e) {}
  }

  // Close settings window
  if (settingsWindow && !settingsWindow.isDestroyed()) {
    settingsWindow.close();
  }
});
