/**
 * MojoLlama Studio — Secure Preload Bridge
 *
 * Exposes a safe API to renderer processes via contextBridge.
 * All communication with the main process goes through ipcRenderer.
 */

const { contextBridge, ipcRenderer } = require('electron');

/**
 * Main API exposed to renderer windows.
 * Every method uses ipcRenderer.invoke for request/response patterns
 * or ipcRenderer.on for event streams.
 */
const api = {
  // ─── Server Control ──────────────────────────────────────────────────────

  /** Start the MojoLlama server with optional overrides. */
  serverStart: (overrides) => ipcRenderer.invoke('server:start', overrides),

  /** Stop the running server. */
  serverStop: () => ipcRenderer.invoke('server:stop'),

  /** Restart the server. */
  serverRestart: () => ipcRenderer.invoke('server:restart'),

  /** Get current server status. */
  serverStatus: () => ipcRenderer.invoke('server:status'),

  /** Listen for server status updates. */
  onServerStatus: (callback) => {
    const handler = (_event, status) => callback(status);
    ipcRenderer.on('server-status', handler);
    // Return unsubscribe function
    return () => ipcRenderer.removeListener('server-status', handler);
  },

  // ─── Settings ────────────────────────────────────────────────────────────

  /** Load saved settings. */
  settingsLoad: () => ipcRenderer.invoke('settings:load'),

  /** Save settings. */
  settingsSave: (newSettings) => ipcRenderer.invoke('settings:save', newSettings),

  // ─── Console Log ─────────────────────────────────────────────────────────

  /** Get the full log buffer. */
  logGet: () => ipcRenderer.invoke('log:get'),

  /** Clear the log buffer. */
  logClear: () => ipcRenderer.invoke('log:clear'),

  /** Listen for new log entries. */
  onLogEntry: (callback) => {
    const handler = (_event, entry) => callback(entry);
    ipcRenderer.on('server-log', handler);
    return () => ipcRenderer.removeListener('server-log', handler);
  },

  /** Listen for log cleared event. */
  onLogCleared: (callback) => {
    const handler = () => callback();
    ipcRenderer.on('log-cleared', handler);
    return () => ipcRenderer.removeListener('log-cleared', handler);
  },

  // ─── Window Management ───────────────────────────────────────────────────

  /** Open the console log window. */
  openLogWindow: () => ipcRenderer.invoke('window:openLog'),

  /** Open the settings window. */
  openSettingsWindow: () => ipcRenderer.invoke('window:openSettings'),

  // ─── App Info ────────────────────────────────────────────────────────────

  /** Get application and environment info. */
  appInfo: () => ipcRenderer.invoke('app:info'),

  // ─── Shell ───────────────────────────────────────────────────────────────

  /** Open a URL in the default browser. */
  openExternal: (url) => ipcRenderer.invoke('shell:openExternal', url),
};

/**
 * Expose the API via contextBridge.
 * In settings and log windows, we expose the same API so they're
 * interchangeable.
 */
contextBridge.exposeInMainWorld('mojollama', api);
