import { useState } from "react";
import { api, type Auth } from "../api";

type Mode = "signin" | "signup";

const FEATURES: { title: string; body: string; icon: JSX.Element }[] = [
  {
    title: "Voice & words",
    body: "Pace, pauses, pitch, filler words (Hinglish too) and rhetoric.",
    icon: <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Zm-7 9a7 7 0 0 0 14 0M12 19v3" />,
  },
  {
    title: "Audience gaze",
    body: "See which parts of the room you looked at and which you ignored.",
    icon: <><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" /><circle cx="12" cy="12" r="3" /></>,
  },
  {
    title: "Heart rate from your webcam",
    body: "A stress curve on the same timeline as your confidence.",
    icon: <path d="M3 12h4l2-5 4 10 2-5h6" />,
  },
  {
    title: "Drills, not just scores",
    body: "Every weakness links to the exact moment and 1–3 practice drills.",
    icon: <path d="m5 12 4 4L19 6" />,
  },
];

function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm font-medium text-ink">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-muted">{hint}</span>}
    </label>
  );
}

const inputCls =
  "w-full rounded-lg border border-line bg-surface px-3 py-2.5 text-ink outline-none transition focus:border-accent focus:ring-2 focus:ring-[#2a78d64d]";

export default function Login({ onSignedIn }: { onSignedIn: (a: Auth) => void }) {
  const [mode, setMode] = useState<Mode>("signin");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const switchMode = (m: Mode) => {
    setMode(m);
    setError(null);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (mode === "signup" && password.length < 8) return setError("Password must be at least 8 characters");
    setBusy(true);
    try {
      onSignedIn(mode === "signup" ? await api.register(name, email, password) : await api.login(email, password));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const tab = (m: Mode, label: string) => (
    <button
      type="button"
      role="tab"
      aria-selected={mode === m}
      onClick={() => switchMode(m)}
      className={`flex-1 rounded-md py-2 text-sm font-semibold transition ${mode === m ? "bg-surface text-ink shadow-sm" : "text-ink2 hover:text-ink"}`}
    >
      {label}
    </button>
  );

  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      <aside className="flex flex-col justify-between bg-[#1f4e8c] px-6 py-8 text-white sm:px-12 lg:py-12">
        <div className="text-2xl font-bold tracking-tight">
          Vakta<span className="text-[#9ec5f4]">AI</span>
        </div>
        <div className="my-6 max-w-md lg:my-0">
          <h1 className="text-2xl font-bold leading-tight sm:text-4xl">Practise like you have an audience.</h1>
          <p className="mt-3 text-base text-white/80">
            Record a talk and get feedback on how you sound, look and feel — moment by moment.
          </p>
          <ul className="mt-8 hidden space-y-5 sm:block">
            {FEATURES.map((f) => (
              <li key={f.title} className="flex gap-4">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-white/10">
                  <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
                    {f.icon}
                  </svg>
                </span>
                <span>
                  <span className="block font-semibold">{f.title}</span>
                  <span className="block text-sm text-white/75">{f.body}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
        <p className="hidden text-xs text-white/60 lg:block">
          Videos are analysed on this computer and deleted after 7 days unless you keep them. Heart-rate and emotion
          readings are indicative, not medical.
        </p>
      </aside>

      <main className="flex items-center justify-center bg-page px-4 py-10">
        <div className="w-full max-w-sm">
          <h2 className="text-2xl font-bold text-ink">{mode === "signin" ? "Welcome back" : "Create your account"}</h2>
          <p className="mt-1 text-sm text-ink2">
            {mode === "signin" ? "Sign in to see your sessions and progress." : "Your sessions and progress stay private to you."}
          </p>

          <div role="tablist" className="mt-6 flex gap-1 rounded-lg bg-grid p-1">
            {tab("signin", "Sign in")}
            {tab("signup", "Create account")}
          </div>

          <form onSubmit={submit} className="mt-6 space-y-4" noValidate>
            {mode === "signup" && (
              <Field label="Full name">
                <input className={inputCls} value={name} onChange={(e) => setName(e.target.value)} autoComplete="name"
                  placeholder="e.g. Chetan Grover" required />
              </Field>
            )}
            <Field label="Email">
              <input className={inputCls} type="email" value={email} onChange={(e) => setEmail(e.target.value)}
                autoComplete="email" placeholder="you@example.com" required />
            </Field>
            <Field label="Password" hint={mode === "signup" ? "At least 8 characters." : undefined}>
              <div className="relative">
                <input className={`${inputCls} pr-16`} type={show ? "text" : "password"} value={password}
                  onChange={(e) => setPassword(e.target.value)} autoComplete={mode === "signin" ? "current-password" : "new-password"} required />
                <button type="button" onClick={() => setShow((s) => !s)}
                  className="absolute inset-y-0 right-2 my-auto h-7 rounded px-2 text-xs font-medium text-ink2 hover:bg-grid"
                  aria-label={show ? "Hide password" : "Show password"}>
                  {show ? "Hide" : "Show"}
                </button>
              </div>
            </Field>

            {error && (
              <p role="alert" className="rounded-lg border border-[#d03b3b66] bg-[#d03b3b14] px-3 py-2 text-sm text-critical">
                {error}
              </p>
            )}

            <button className="btn-primary w-full justify-center py-2.5"
              disabled={busy || !email.trim() || !password || (mode === "signup" && !name.trim())}>
              {busy ? "Please wait…" : mode === "signin" ? "Sign in" : "Create account"}
            </button>
          </form>

          <p className="mt-6 text-center text-sm text-ink2">
            {mode === "signin" ? "New to VaktaAI? " : "Already have an account? "}
            <button type="button" className="font-semibold text-accent hover:underline"
              onClick={() => switchMode(mode === "signin" ? "signup" : "signin")}>
              {mode === "signin" ? "Create an account" : "Sign in"}
            </button>
          </p>
          {mode === "signup" && (
            <p className="mt-3 text-center text-xs text-muted">
              Used VaktaAI before with just a name? Sign up with the same name to keep your old sessions.
            </p>
          )}
        </div>
      </main>
    </div>
  );
}
