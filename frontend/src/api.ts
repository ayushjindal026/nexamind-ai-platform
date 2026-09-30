const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type Organization = { id: string; name: string };
export type SessionUser = { id: string; email: string; organization: Organization; role: string };
export type Document = {
  id: string;
  filename: string;
  content_type: string;
  file_size_bytes: number;
  page_count: number;
  status: "processing" | "ready" | "failed" | string;
  failure_reason: string | null;
  created_at: string;
  chunk_count: number;
  embedded_chunk_count: number;
};
export type AssistantStatus = { configured: boolean; assistant_id: string | null; created_at: string | null };
export type AssistantLink = { assistant_id: string; assistant_token: string; assistant_url: string; embed_code: string };
export type Usage = { questions_asked: number; questions_answered: number; questions_unavailable: number };
export type Citation = { document_id: string; document_name: string; page_number: number; citation: string };
export type Decision = {
  question_id: string;
  question: string;
  decision_type: "choice" | "boolean" | "score" | "classification";
  value: string | boolean | number;
  probability: number | null;
};
export type AskResult = { answer: string; citations: Citation[]; decisions: Decision[]; decision_outcome: string | null };

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, options: RequestInit = {}, token?: string): Promise<T> {
  const headers = new Headers(options.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  } catch {
    throw new ApiError("We could not reach the Nexa Mind service. Check your connection and try again.", 0);
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({})) as { detail?: string };
    const detail = payload.detail ?? "The request could not be completed.";
    const messages: Record<string, string> = {
      "Invalid email or password": "That email and password combination was not recognized.",
      "Invalid assistant token": "This assistant link is no longer valid. Ask the organization for a new link.",
      "Assistant is temporarily unavailable": "The assistant is temporarily unavailable. Please try again shortly.",
      "Structured decision service is temporarily unavailable": "The decision service is temporarily unavailable. Please try again shortly.",
      "Failed to process document": "The PDF could not be processed. Please try again with a valid PDF.",
      "Failed to delete document": "The document could not be deleted. Please try again.",
      "Assistant is already configured": "An assistant is already configured for this organization.",
      "Assistant is not configured": "Create an assistant before managing its visitor link.",
    };
    let message = messages[detail] ?? detail;
    if (response.status === 401 && token) message = "Your session expired. Sign in again to continue.";
    if (response.status === 422 && detail.includes("maximum allowed size")) message = "This PDF is larger than the 10 MB upload limit.";
    if (response.status === 422 && detail.includes("maximum allowed page count")) message = "This PDF has more than the 20 page limit.";
    if (response.status === 422 && detail.includes("upload limit")) message = "This organization has reached its 10 document limit. Delete a PDF before uploading another.";
    if (response.status === 422 && detail.includes("valid PDF")) message = "We could not read that file as a valid PDF.";
    if (response.status >= 500 && !messages[detail]) message = "Something went wrong on the service. Please try again.";
    throw new ApiError(message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  register: (body: { email: string; password: string; organization_name: string }) =>
    request<{ access_token: string }>("/api/auth/register", { method: "POST", body: JSON.stringify(body) }),
  login: (body: { email: string; password: string }) =>
    request<{ access_token: string }>("/api/auth/login", { method: "POST", body: JSON.stringify(body) }),
  me: (token: string) => request<SessionUser>("/api/auth/me", {}, token),
  documents: (token: string) => request<Document[]>("/api/documents", {}, token),
  uploadDocument: (token: string, file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<Document>("/api/documents", { method: "POST", body }, token);
  },
  deleteDocument: (token: string, id: string) =>
    request<void>(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" }, token),
  assistant: (token: string) => request<AssistantStatus>("/api/assistant", {}, token),
  createAssistant: (token: string) =>
    request<AssistantLink>("/api/assistant", { method: "POST" }, token),
  regenerateAssistant: (token: string) =>
    request<AssistantLink>("/api/assistant/regenerate", { method: "POST" }, token),
  usage: (token: string) => request<Usage>("/api/assistant/usage", {}, token),
  ask: (assistantToken: string, question: string) =>
    request<AskResult>("/api/ask", { method: "POST", body: JSON.stringify({ assistant_token: assistantToken, question }) }),
};

export function friendlyError(error: unknown): string {
  return error instanceof Error ? error.message : "Something unexpected happened. Please try again.";
}
