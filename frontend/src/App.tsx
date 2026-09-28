import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  ApiError,
  AssistantLink,
  AssistantStatus,
  Document,
  friendlyError,
  SessionUser,
  Usage,
} from "./api";
import "./styles.css";

const SESSION_KEY = "panscience.dashboard.session";
type Route = "login" | "register" | "dashboard";

function navigate(path: string) {
  window.history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

function routeFromPath(path: string): Route {
  if (path === "/register") return "register";
  if (path === "/login") return "login";
  return "dashboard";
}

export default function App() {
  const [path, setPath] = useState(window.location.pathname);
  const [token, setToken] = useState(() => sessionStorage.getItem(SESSION_KEY));
  const route = routeFromPath(path);

  useEffect(() => {
    const onPopState = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const signOut = useCallback(() => {
    sessionStorage.removeItem(SESSION_KEY);
    setToken(null);
    navigate("/login");
  }, []);

  const signedIn = useCallback((accessToken: string) => {
    sessionStorage.setItem(SESSION_KEY, accessToken);
    setToken(accessToken);
    navigate("/");
  }, []);

  if (!token || route !== "dashboard") {
    if (token && route !== "dashboard") return <Dashboard token={token} onSignOut={signOut} />;
    return <AuthScreen initialMode={route === "register" ? "register" : "login"} onSignedIn={signedIn} />;
  }
  return <Dashboard token={token} onSignOut={signOut} />;
}

function AuthScreen({ initialMode, onSignedIn }: { initialMode: "login" | "register"; onSignedIn: (token: string) => void }) {
  const [mode, setMode] = useState(initialMode);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [organization, setOrganization] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => setMode(initialMode), [initialMode]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const result = mode === "login"
        ? await api.login({ email, password })
        : await api.register({ email, password, organization_name: organization });
      onSignedIn(result.access_token);
    } catch (cause) {
      setError(friendlyError(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-shell">
      <div className="auth-orbit orbit-one" /><div className="auth-orbit orbit-two" />
      <section className="auth-card">
        <Brand />
        <p className="eyebrow">KNOWLEDGE, READY WHEN YOU ARE</p>
        <h1>{mode === "login" ? "Welcome back" : "Create your workspace"}</h1>
        <p className="muted">{mode === "login" ? "Sign in to manage your organization’s knowledge assistant." : "Bring your organization’s documents and answers together."}</p>
        <form className="stack-form" onSubmit={submit}>
          {mode === "register" && <label>Organization name<input autoComplete="organization" required maxLength={255} value={organization} onChange={(e) => setOrganization(e.target.value)} placeholder="e.g. Acme University" /></label>}
          <label>Work email<input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@organization.com" /></label>
          <label>Password<input type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} required minLength={mode === "register" ? 8 : undefined} maxLength={128} value={password} onChange={(e) => setPassword(e.target.value)} placeholder={mode === "register" ? "At least 8 characters" : "Your password"} /></label>
          {error && <p className="notice error" role="alert">{error}</p>}
          <button className="button primary full" type="submit" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}<span aria-hidden="true">↗</span></button>
        </form>
        <p className="auth-switch">{mode === "login" ? "New to PanScience?" : "Already have a workspace?"} <button className="text-button" onClick={() => { setError(""); setMode(mode === "login" ? "register" : "login"); navigate(mode === "login" ? "/register" : "/login"); }}>{mode === "login" ? "Create an account" : "Sign in"}</button></p>
      </section>
      <footer className="auth-footer">A private knowledge assistant for every organization.</footer>
    </main>
  );
}

function Dashboard({ token, onSignOut }: { token: string; onSignOut: () => void }) {
  const [user, setUser] = useState<SessionUser | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [assistant, setAssistant] = useState<AssistantStatus | null>(null);
  const [link, setLink] = useState<AssistantLink | null>(null);
  const [usage, setUsage] = useState<Usage>({ questions_asked: 0, questions_answered: 0, questions_unavailable: 0 });
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const uploadRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    setError("");
    try {
      const [me, docs, assistantStatus, stats] = await Promise.all([
        api.me(token), api.documents(token), api.assistant(token), api.usage(token),
      ]);
      setUser(me);
      setDocuments(docs);
      setAssistant(assistantStatus);
      setUsage(stats);
      if (assistantStatus.assistant_id) {
        const saved = sessionStorage.getItem(`panscience.assistant.link.${assistantStatus.assistant_id}`);
        setLink(saved ? JSON.parse(saved) as AssistantLink : null);
      } else setLink(null);
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 401) onSignOut();
      else setError(friendlyError(cause));
    } finally {
      setLoading(false);
    }
  }, [onSignOut, token]);

  useEffect(() => { void load(); }, [load]);

  const readyCount = useMemo(() => documents.filter((doc) => doc.status === "ready").length, [documents]);

  async function uploadFile(file?: File) {
    if (!file) return;
    setError(""); setNotice(""); setBusy(true);
    try {
      await api.uploadDocument(token, file);
      setNotice(`${file.name} was added to your knowledge base.`);
      await load();
    } catch (cause) { setError(friendlyError(cause)); }
    finally { setBusy(false); if (uploadRef.current) uploadRef.current.value = ""; }
  }

  async function deleteFile(doc: Document) {
    if (!window.confirm(`Delete “${doc.filename}”? This also removes its indexed content.`)) return;
    setError(""); setNotice(""); setBusy(true);
    try { await api.deleteDocument(token, doc.id); setNotice(`${doc.filename} was deleted.`); await load(); }
    catch (cause) { setError(friendlyError(cause)); }
    finally { setBusy(false); }
  }

  async function configureAssistant(regenerate = false) {
    if (regenerate && !window.confirm("Regenerate the visitor link? The current link and embed will stop working immediately.")) return;
    setError(""); setNotice(""); setBusy(true);
    try {
      const result = regenerate ? await api.regenerateAssistant(token) : await api.createAssistant(token);
      sessionStorage.setItem(`panscience.assistant.link.${result.assistant_id}`, JSON.stringify(result));
      setLink(result);
      setAssistant({ configured: true, assistant_id: result.assistant_id, created_at: assistant?.created_at ?? new Date().toISOString() });
      setNotice(regenerate ? "A new visitor link is ready. The previous link has been revoked." : "Your assistant is ready to share.");
    } catch (cause) { setError(friendlyError(cause)); }
    finally { setBusy(false); }
  }

  async function copyText(value: string, label: string) {
    try { await navigator.clipboard.writeText(value); setNotice(`${label} copied to clipboard.`); }
    catch { setError("Clipboard access is unavailable. Select and copy the text manually."); }
  }

  if (loading) return <div className="loading-screen"><span className="spinner" /> Loading your workspace…</div>;

  return (
    <div className="dashboard-shell">
      <aside className="sidebar">
        <Brand />
        <div className="sidebar-label">WORKSPACE</div>
        <a className="side-link active" href="#overview"><span>▦</span> Overview</a>
        <a className="side-link" href="#knowledge"><span>▤</span> Knowledge base</a>
        <a className="side-link" href="#assistant"><span>✳</span> Assistant</a>
        <a className="side-link" href="#usage"><span>◷</span> Usage</a>
        <div className="sidebar-bottom"><div className="avatar">{user?.organization.name.slice(0, 1).toUpperCase() ?? "P"}</div><div className="side-org"><strong>{user?.organization.name ?? "Organization"}</strong><span>{user?.email ?? ""}</span></div><button className="icon-button logout-icon" onClick={onSignOut} title="Sign out" aria-label="Sign out">↪</button></div>
      </aside>
      <main className="dashboard-main" id="overview">
        <header className="topbar"><div className="crumb">Workspace <span>/</span> Overview</div><div className="top-user"><span className="online-dot" />{user?.organization.name}<button className="button subtle small" onClick={onSignOut}>Sign out</button></div></header>
        <div className="page-content">
          <div className="welcome-row"><div><p className="eyebrow">ORGANIZATION WORKSPACE</p><h1>{user?.organization.name ?? "Your workspace"}</h1><p className="muted">Manage what your assistant knows and how people access it.</p></div><button className="button primary" onClick={() => uploadRef.current?.click()} disabled={busy || documents.length >= 10}><span>＋</span> Add PDF</button><input ref={uploadRef} className="sr-only" type="file" accept="application/pdf,.pdf" onChange={(e) => void uploadFile(e.target.files?.[0])} /></div>
          {error && <div className="notice error" role="alert">{error}<button onClick={() => setError("")} aria-label="Dismiss">×</button></div>}
          {notice && <div className="notice success" role="status">{notice}<button onClick={() => setNotice("")} aria-label="Dismiss">×</button></div>}
          <section className="metric-grid" aria-label="Workspace summary">
            <Metric label="Documents" value={`${documents.length} / 10`} detail="PDFs in your library" icon="▤" tone="blue" />
            <Metric label="Ready to answer" value={readyCount} detail="Documents processed" icon="✓" tone="green" />
            <Metric label="Questions asked" value={usage.questions_asked} detail="All accepted visitor questions" icon="◷" tone="violet" />
          </section>

          <section className="panel knowledge-panel" id="knowledge">
            <div className="panel-heading"><div><p className="eyebrow">YOUR CONTENT</p><h2>Knowledge base</h2><p className="muted">PDFs are private to your organization and power assistant answers.</p></div><span className="count-pill">{documents.length} of 10 PDFs</span></div>
            <div className="limits-row"><span>⌁ Upload limits</span><span>10 MB per file</span><i /><span>20 pages per PDF</span><i /><span>10 PDFs total</span></div>
            {documents.length === 0 ? <div className="empty-state"><div className="empty-icon">▤</div><h3>Your knowledge starts here</h3><p>Add a PDF policy or guide. Your assistant will use its content to answer visitor questions.</p><button className="button secondary" onClick={() => uploadRef.current?.click()} disabled={busy}>Choose a PDF</button></div> : <div className="table-wrap"><table><thead><tr><th>DOCUMENT</th><th>PAGES</th><th>SIZE</th><th>STATUS</th><th>ADDED</th><th><span className="sr-only">Actions</span></th></tr></thead><tbody>{documents.map((doc) => <tr key={doc.id}><td><div className="file-cell"><span className="pdf-icon">PDF</span><span className="file-name">{doc.filename}</span></div></td><td>{doc.page_count}</td><td>{formatBytes(doc.file_size_bytes)}</td><td><StatusPill status={doc.status} /></td><td>{new Date(doc.created_at).toLocaleDateString()}</td><td><button className="icon-button delete-button" title={`Delete ${doc.filename}`} aria-label={`Delete ${doc.filename}`} disabled={busy} onClick={() => void deleteFile(doc)}>⌫</button></td></tr>)}</tbody></table></div>}
            {documents.length > 0 && <div className="panel-footer"><span>Showing {documents.length} document{documents.length === 1 ? "" : "s"}</span><button className="button secondary small" onClick={() => uploadRef.current?.click()} disabled={busy || documents.length >= 10}>＋ Upload PDF</button></div>}
          </section>

          <div className="lower-grid">
            <section className="panel assistant-panel" id="assistant"><div className="panel-heading compact"><div><p className="eyebrow">SHARE YOUR KNOWLEDGE</p><h2>Assistant access</h2></div><span className={`status-dot ${assistant?.configured ? "is-on" : ""}`}>{assistant?.configured ? "Active" : "Not set up"}</span></div>
              {!assistant?.configured ? <div className="assistant-empty"><p>Create a secure public assistant link and an embed snippet for your website.</p><button className="button primary" disabled={busy} onClick={() => void configureAssistant()}>Create assistant</button></div> : <>
                {link ? <><label className="field-label" htmlFor="assistant-link">Public assistant URL</label><div className="copy-field"><input id="assistant-link" readOnly value={link.assistant_url} /><button className="button secondary small" onClick={() => void copyText(link.assistant_url, "Assistant URL")}>Copy</button><a className="button icon-link" href={link.assistant_url} target="_blank" rel="noreferrer">↗</a></div><label className="field-label embed-label" htmlFor="embed-code">Website embed code</label><div className="embed-wrap"><textarea id="embed-code" readOnly rows={3} value={link.embed_code} /><button className="button secondary small" onClick={() => void copyText(link.embed_code, "Embed code")}>Copy code</button></div></> : <div className="link-notice"><span>◉</span><p>This assistant is configured, but its one-time visitor link is not available in this browser session. Regenerate the link to reveal a fresh URL; the current URL will then stop working.</p></div>}
                <div className="assistant-actions"><button className="button secondary small" disabled={!link} onClick={() => link && window.open(link.assistant_url, "_blank", "noopener,noreferrer")}>Open assistant ↗</button><button className="text-button danger-text" disabled={busy} onClick={() => void configureAssistant(true)}>Regenerate link</button></div>
              </>}
            </section>
            <section className="panel usage-panel" id="usage"><div className="panel-heading compact"><div><p className="eyebrow">ACTIVITY</p><h2>Assistant usage</h2></div><span className="usage-range">All time</span></div><p className="muted usage-caption">Counts for visitor questions to this organization’s assistant.</p><div className="usage-rows"><UsageRow label="Questions asked" value={usage.questions_asked} tone="violet" /><UsageRow label="Answered with sources" value={usage.questions_answered} tone="green" /><UsageRow label="Unavailable / no sources" value={usage.questions_unavailable} tone="amber" /></div><p className="usage-footnote">Provider failures are included as unavailable when the assistant cannot complete a request.</p></section>
          </div>
          <footer className="page-footer">Private by design <span>·</span> Every answer stays within your organization’s knowledge base.</footer>
        </div>
      </main>
    </div>
  );
}

function Brand() { return <div className="brand"><span className="brand-mark">P</span><span>PanScience<span className="brand-light"> Assistant</span></span></div>; }
function Metric({ label, value, detail, icon, tone }: { label: string; value: string | number; detail: string; icon: string; tone: string }) { return <div className="metric-card"><div className={`metric-icon ${tone}`}>{icon}</div><div className="metric-text"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div><span className={`metric-accent ${tone}`} /></div>; }
function StatusPill({ status }: { status: string }) { const className = status === "ready" ? "ready" : status === "failed" ? "failed" : "processing"; const label = status === "ready" ? "Ready" : status === "failed" ? "Failed" : "Processing"; return <span className={`status-pill ${className}`}><i />{label}</span>; }
function UsageRow({ label, value, tone }: { label: string; value: number; tone: string }) { return <div className="usage-row"><span className={`usage-mark ${tone}`} /><span>{label}</span><strong>{value}</strong></div>; }
function formatBytes(value: number) { return value < 1024 * 1024 ? `${Math.max(1, Math.round(value / 1024))} KB` : `${(value / (1024 * 1024)).toFixed(1)} MB`; }
