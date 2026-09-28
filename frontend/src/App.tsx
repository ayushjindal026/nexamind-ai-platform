import { useEffect, useState } from "react";
import { getBackendHealth } from "./api";

// Phase 1: this component's only job is to prove the three-container
// architecture works end to end (frontend container -> browser -> backend
// container -> Postgres, via /health/db separately). Real pages (login,
// documents, assistant, usage) replace this starting Phase 2/14.
export default function App() {
  const [status, setStatus] = useState<"loading" | "ok" | "error">("loading");
  const [detail, setDetail] = useState<string>("");

  useEffect(() => {
    getBackendHealth()
      .then((res) => {
        setStatus("ok");
        setDetail(res.status);
      })
      .catch((err) => {
        setStatus("error");
        setDetail(String(err));
      });
  }, []);

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", padding: "2rem" }}>
      <h1>AI Knowledge & Decision Assistant</h1>
      <p>Phase 1 — repository and Docker Compose scaffolding.</p>
      <p>
        Backend status:{" "}
        {status === "loading" && "checking..."}
        {status === "ok" && <strong style={{ color: "green" }}>reachable ({detail})</strong>}
        {status === "error" && <strong style={{ color: "red" }}>unreachable — {detail}</strong>}
      </p>
    </main>
  );
}
