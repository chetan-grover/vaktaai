import { useEffect, useState } from "react";

export const fmtTime = (s: number | null | undefined): string => {
  if (s == null || Number.isNaN(s)) return "–";
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};

export const fmt = (v: unknown, digits = 0, suffix = ""): string =>
  typeof v === "number" && Number.isFinite(v) ? `${v.toFixed(digits)}${suffix}` : "–";

export const pct = (v: unknown): string => (typeof v === "number" ? `${Math.round(v * 100)}%` : "–");

const TOKENS = [
  "--surface-1", "--text-primary", "--text-secondary", "--text-muted", "--grid", "--axis",
  "--series-1", "--series-2", "--series-3", "--status-critical", "--status-serious", "--status-warning",
  "--seq-100", "--seq-250", "--seq-450", "--seq-650",
] as const;
export type Tokens = Record<(typeof TOKENS)[number], string>;

const readTokens = (): Tokens => {
  const css = getComputedStyle(document.documentElement);
  return Object.fromEntries(TOKENS.map((t) => [t, css.getPropertyValue(t).trim()])) as Tokens;
};

/** Resolved colour tokens for SVG charts; re-read when the OS theme flips. */
export function useTokens(): Tokens {
  const [tokens, setTokens] = useState<Tokens>(readTokens);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setTokens(readTokens());
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return tokens;
}

export const scoreTone = (score: number | null | undefined): string =>
  score == null ? "text-muted" : score >= 70 ? "text-[color:var(--success-text)]" : score >= 50 ? "text-ink" : "text-critical";
