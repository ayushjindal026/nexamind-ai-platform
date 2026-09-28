// Phase 1: a single function proving the browser can reach the backend.
// Grows into a real API client (auth, documents, assistant, ask) from Phase 2 onward.

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function getBackendHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Backend health check failed: ${response.status}`);
  }
  return response.json();
}
