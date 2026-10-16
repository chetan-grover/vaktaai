import { useEffect, useState } from "react";
import { Link, Navigate, NavLink, Route, Routes } from "react-router-dom";
import { api, loadAuth, saveAuth, UNAUTHORIZED_EVENT, type Auth } from "./api";
import History from "./pages/History";
import Login from "./pages/Login";
import NewSession from "./pages/NewSession";
import SessionPage from "./pages/SessionPage";

export default function App() {
  const [auth, setAuth] = useState<Auth | null>(loadAuth);

  const signIn = (a: Auth) => {
    saveAuth(a);
    setAuth(a);
  };
  const signOut = () => {
    saveAuth(null);
    setAuth(null);
  };
  const logout = async () => {
    try {
      await api.logout();
    } finally {
      signOut();
    }
  };

  useEffect(() => {
    window.addEventListener(UNAUTHORIZED_EVENT, signOut);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, signOut);
  }, []);

  useEffect(() => {
    // Revalidate a remembered session (token may have expired or been revoked).
    if (auth) api.me().catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!auth) return <Login onSignedIn={signIn} />;

  const nav = ({ isActive }: { isActive: boolean }) =>
    `rounded-lg px-3 py-1.5 text-sm font-medium ${isActive ? "bg-grid text-ink" : "text-ink2 hover:text-ink"}`;
  const initials = auth.user.name.split(/\s+/).map((p) => p[0]).join("").slice(0, 2).toUpperCase();

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-line bg-surface">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
          <Link to="/" className="text-lg font-bold tracking-tight">
            Vakta<span className="text-accent">AI</span>
          </Link>
          <nav className="flex gap-1">
            <NavLink to="/" end className={nav}>New session</NavLink>
            <NavLink to="/history" className={nav}>History</NavLink>
          </nav>
          <div className="ml-auto flex items-center gap-3 text-sm">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-xs font-semibold text-white" aria-hidden>
              {initials}
            </span>
            <span className="hidden flex-col leading-tight sm:flex">
              <span className="font-medium text-ink">{auth.user.name}</span>
              {auth.user.email && <span className="text-xs text-muted">{auth.user.email}</span>}
            </span>
            <button className="btn" onClick={logout}>Sign out</button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">
        <Routes>
          <Route path="/" element={<NewSession />} />
          <Route path="/sessions/:id" element={<SessionPage />} />
          <Route path="/history" element={<History />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </main>
    </div>
  );
}
