import type { PersonaMap, ProgressPoint, Report, SessionRow, User } from "./types";

const AUTH_KEY = "vakta.auth";
export const UNAUTHORIZED_EVENT = "vakta:unauthorized";

export interface Auth {
  token: string;
  user: User;
}

export function loadAuth(): Auth | null {
  try {
    const raw = localStorage.getItem(AUTH_KEY);
    return raw ? (JSON.parse(raw) as Auth) : null;
  } catch {
    return null;
  }
}

export function saveAuth(auth: Auth | null): void {
  try {
    if (auth) localStorage.setItem(AUTH_KEY, JSON.stringify(auth));
    else localStorage.removeItem(AUTH_KEY);
  } catch {
    /* private mode: the session lasts for this tab only */
  }
  current = auth;
}

let current: Auth | null = loadAuth();

function humanise(detail: unknown): string {
  if (typeof detail === "string") return detail.charAt(0).toUpperCase() + detail.slice(1);
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc: [..., "password"], msg: "..."}]
    return detail
      .map((d: { loc?: string[]; msg?: string }) => {
        const field = d.loc?.[d.loc.length - 1];
        const msg = (d.msg ?? "").replace(/^Value error, /, "");
        if (field === "password" && /at least/.test(msg)) return "Password must be at least 8 characters";
        return field && !/email|name/.test(msg) ? `${field}: ${msg}` : msg.charAt(0).toUpperCase() + msg.slice(1);
      })
      .join(". ");
  }
  return "Something went wrong";
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (current) headers.set("Authorization", `Bearer ${current.token}`);
  const res = await fetch(`/api${path}`, { ...init, headers });
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    if (res.status === 401 && current && !path.startsWith("/auth/")) window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
    throw new Error(humanise(detail));
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  register: (name: string, email: string, password: string) => request<Auth>("/auth/register", json("POST", { name, email, password })),
  login: (email: string, password: string) => request<Auth>("/auth/login", json("POST", { email, password })),
  logout: () => request<void>("/auth/logout", { method: "POST" }),
  me: () => request<User>("/auth/me"),
  sessions: () => request<SessionRow[]>("/sessions"),
  session: (id: string) => request<SessionRow>(`/sessions/${id}`),
  report: (id: string) => request<Report>(`/sessions/${id}/report`),
  videoUrl: (id: string) => `/api/sessions/${id}/video?token=${encodeURIComponent(current?.token ?? "")}`,
  upload: (form: FormData) => request<SessionRow>("/sessions", { method: "POST", body: form }),
  markDrill: (id: string, drillId: string, done: boolean) =>
    request(`/sessions/${id}/drills/${drillId}`, json("PATCH", { done })),
  patchSession: (id: string, body: { title?: string; keep_video?: boolean }) =>
    request<SessionRow>(`/sessions/${id}`, json("PATCH", body)),
  deleteSession: (id: string) => request<void>(`/sessions/${id}`, { method: "DELETE" }),
  reanalyse: (id: string) => request<SessionRow>(`/sessions/${id}/reanalyse`, { method: "POST" }),
  deleteVideo: (id: string) => request<SessionRow>(`/sessions/${id}/video`, { method: "DELETE" }),
  progress: () => request<{ points: ProgressPoint[] }>("/me/progress"),
  persona: () => request<PersonaMap>("/me/persona"),
};
