const API_BASE = import.meta.env.VITE_API_BASE || "/api";

async function parseResponse(response) {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "请求失败");
  }
  return response.json();
}

export async function createTask(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/tasks`, {
    method: "POST",
    body: formData
  });
  return parseResponse(response);
}

export async function listTasks(limit = 20) {
  const response = await fetch(`${API_BASE}/tasks?limit=${limit}`);
  return parseResponse(response);
}

export async function getTaskStatus(taskId) {
  const response = await fetch(`${API_BASE}/tasks/${taskId}`);
  return parseResponse(response);
}

export async function getTaskResult(taskId) {
  const response = await fetch(`${API_BASE}/tasks/${taskId}/result`);
  return parseResponse(response);
}

export async function getSystemStatus() {
  const response = await fetch(`${API_BASE}/status`);
  return parseResponse(response);
}

export function toArtifactUrl(url) {
  return url;
}

