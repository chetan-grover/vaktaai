import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  CartesianGrid, LabelList, Legend, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { api } from "../api";
import type { PersonaMap, ProgressPoint, SessionRow } from "../types";
import { fmt, fmtTime, pct, useTokens } from "../util";

const COMPARE: [keyof ProgressPoint, string, (v: number | null) => string, boolean][] = [
  ["confidence", "Confidence", (v) => fmt(v), true],
  ["authoritativeness", "Authoritativeness", (v) => fmt(v), true],
  ["engagement", "Engagement", (v) => fmt(v), true],
  ["congruence", "Congruence", (v) => fmt(v), true],
  ["wpm", "Pace (wpm)", (v) => fmt(v), false],
  ["filler_rate_per_min", "Fillers / min", (v) => fmt(v, 1), false],
  ["eye_contact_ratio", "Camera / centre gaze", pct, true],
  ["gaze_coverage", "Room coverage", pct, true],
  ["bpm_rise", "Heart-rate rise (BPM)", (v) => fmt(v), false],
];

type TipPayload = { payload: { name?: string; description?: string; title?: string; created_at?: string; persona?: { name: string } } };

function PersonaTip({ active, payload, style }: { active?: boolean; payload?: TipPayload[]; style: React.CSSProperties }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{ ...style, padding: "6px 10px", maxWidth: 240 }}>
      {d.persona ? (
        <>
          <div className="font-semibold">{d.title}</div>
          <div>{d.created_at ? new Date(d.created_at).toLocaleDateString() : ""} · {d.persona.name}</div>
        </>
      ) : (
        <>
          <div className="font-semibold">{d.name}</div>
          <div>{d.description}</div>
        </>
      )}
    </div>
  );
}

export default function History() {
  const tk = useTokens();
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [points, setPoints] = useState<ProgressPoint[]>([]);
  const [persona, setPersona] = useState<PersonaMap | null>(null);
  const [a, setA] = useState("");
  const [b, setB] = useState("");

  const load = async () => {
    const [s, p, m] = await Promise.all([api.sessions(), api.progress(), api.persona()]);
    setSessions(s);
    setPoints(p.points);
    setPersona(m);
    if (p.points.length >= 2) {
      setA(p.points[0].session_id);
      setB(p.points[p.points.length - 1].session_id);
    }
  };
  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const trend = useMemo(() => points.map((p, i) => ({ ...p, n: i + 1 })), [points]);
  const pa = points.find((p) => p.session_id === a);
  const pb = points.find((p) => p.session_id === b);
  const axis = { stroke: tk["--axis"], tick: { fill: tk["--text-muted"], fontSize: 11 } };
  const tooltipStyle = { background: tk["--surface-1"], border: `1px solid ${tk["--grid"]}`, borderRadius: 8, fontSize: 12, color: tk["--text-primary"] };

  const remove = async (id: string) => {
    if (!window.confirm("Delete this session, its report and its video?")) return;
    await api.deleteSession(id);
    await load();
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Progress</h1>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="card">
          <h2 className="mb-2 font-semibold">Trait scores per session</h2>
          {trend.length < 2 ? (
            <p className="text-sm text-muted">Finish at least two sessions to see a trend.</p>
          ) : (
            <div className="h-64">
              <ResponsiveContainer>
                <LineChart data={trend} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
                  <CartesianGrid stroke={tk["--grid"]} vertical={false} />
                  <XAxis dataKey="n" tickFormatter={(n) => `#${n}`} {...axis} />
                  <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} {...axis} />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => fmt(v)}
                    labelFormatter={(n) => trend[Number(n) - 1]?.title ?? `#${n}`} />
                  <Legend wrapperStyle={{ fontSize: 12 }} iconType="plainline" />
                  <Line dataKey="confidence" name="Confidence" stroke={tk["--series-1"]} strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
                  <Line dataKey="authoritativeness" name="Authoritativeness" stroke={tk["--series-2"]} strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
                  <Line dataKey="engagement" name="Engagement" stroke={tk["--series-3"]} strokeWidth={2} dot={{ r: 4 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

        <section className="card">
          <div className="mb-2 flex items-baseline justify-between">
            <h2 className="font-semibold">Speaker persona map</h2>
            <span className="text-xs text-muted">{persona?.kind === "vae+gmm" ? "VAE latent space" : "rule map (train the VAE for a learned map)"}</span>
          </div>
          {persona?.current && (
            <p className="mb-2 text-sm"><span className="font-semibold">{persona.current.name}</span> — <span className="text-ink2">{persona.current.description}</span></p>
          )}
          <div className="h-64">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 12, right: 24, bottom: 16, left: -8 }}>
                <CartesianGrid stroke={tk["--grid"]} />
                <XAxis type="number" dataKey="x" name={persona?.axes[0]} {...axis} domain={["dataMin - 0.5", "dataMax + 0.5"]} tickFormatter={(v) => v.toFixed(1)}
                  label={{ value: persona?.axes[0], position: "insideBottom", offset: -8, fill: tk["--text-muted"], fontSize: 11 }} />
                <YAxis type="number" dataKey="y" name={persona?.axes[1]} {...axis} domain={["dataMin - 0.5", "dataMax + 0.5"]} tickFormatter={(v) => v.toFixed(1)}
                  label={{ value: persona?.axes[1], angle: -90, position: "insideLeft", offset: 16, fill: tk["--text-muted"], fontSize: 11 }} />
                <ZAxis range={[70, 70]} />
                <Tooltip content={<PersonaTip style={tooltipStyle} />} cursor={false} />
                <Scatter name="Personas" data={persona?.regions ?? []} fill={tk["--axis"]} isAnimationActive={false}>
                  <LabelList dataKey="name" position="top" style={{ fontSize: 10, fill: tk["--text-secondary"], fontWeight: 600 }}
                    formatter={(n: string) => (n === persona?.current?.name ? n : "")} />
                </Scatter>
                <Scatter name="Your sessions" data={persona?.points ?? []} fill={tk["--series-1"]} line={{ stroke: tk["--series-1"], strokeWidth: 2 }}
                  stroke={tk["--surface-1"]} strokeWidth={2} isAnimationActive={false}>
                  <LabelList dataKey="title" position="right" style={{ fontSize: 10, fill: tk["--text-secondary"] }}
                    formatter={(t: string) => (persona?.points.length && t === persona.points[persona.points.length - 1].title ? "latest" : "")} />
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-xs text-muted">Grey dots are persona prototypes (hover to see each one); the blue path is your sessions in order.</p>
        </section>
      </div>

      {points.length >= 2 && (
        <section className="card">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <h2 className="font-semibold">Compare sessions</h2>
            {[["Before", a, setA], ["After", b, setB]].map(([label, value, set]) => (
              <label key={label as string} className="flex items-center gap-1 text-sm">
                <span className="text-muted">{label as string}</span>
                <select className="rounded-md border border-line bg-surface px-2 py-1" value={value as string}
                  onChange={(e) => (set as (v: string) => void)(e.target.value)}>
                  {points.map((p, i) => <option key={p.session_id} value={p.session_id}>#{i + 1} {p.title}</option>)}
                </select>
              </label>
            ))}
          </div>
          {pa && pb && (
            <table className="tabular w-full max-w-2xl text-sm">
              <thead><tr className="text-left text-muted"><th className="py-1 font-medium">Metric</th><th className="text-right font-medium">Before</th><th className="text-right font-medium">After</th><th className="text-right font-medium">Change</th></tr></thead>
              <tbody>
                {COMPARE.map(([key, label, f, higherBetter]) => {
                  const va = pa[key] as number | null;
                  const vb = pb[key] as number | null;
                  const d = va != null && vb != null ? vb - va : null;
                  const better = d != null && d !== 0 && (d > 0) === higherBetter;
                  return (
                    <tr key={key} className="border-t border-line">
                      <td className="py-1">{label}</td>
                      <td className="text-right">{f(va)}</td>
                      <td className="text-right">{f(vb)}</td>
                      <td className={`text-right ${d == null || d === 0 ? "text-muted" : better ? "text-[color:var(--success-text)]" : "text-critical"}`}>
                        {d == null ? "–" : `${d > 0 ? "▲" : d < 0 ? "▼" : ""} ${f(Math.abs(d))}`}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </section>
      )}

      <section className="card overflow-x-auto">
        <h2 className="mb-2 font-semibold">All sessions</h2>
        {sessions.length === 0 && <p className="text-sm text-muted">No sessions yet. <Link className="text-accent" to="/">Record one.</Link></p>}
        {sessions.length > 0 && (
          <table className="tabular w-full text-sm">
            <thead><tr className="text-left text-muted">
              {["Title", "Date", "Length", "Status", "Conf.", "Auth.", "Eng.", "Persona", ""].map((h) => <th key={h} className="py-1 pr-3 font-medium">{h}</th>)}
            </tr></thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.id} className="border-t border-line">
                  <td className="py-1.5 pr-3"><Link className="font-medium text-accent hover:underline" to={`/sessions/${s.id}`}>{s.title}</Link></td>
                  <td className="pr-3 text-ink2">{new Date(s.created_at).toLocaleDateString()}</td>
                  <td className="pr-3">{fmtTime(s.duration_s)}</td>
                  <td className={`pr-3 ${s.status === "failed" ? "text-critical" : "text-ink2"}`}>{s.status}</td>
                  <td className="pr-3">{fmt(s.confidence)}</td>
                  <td className="pr-3">{fmt(s.authoritativeness)}</td>
                  <td className="pr-3">{fmt(s.engagement)}</td>
                  <td className="pr-3 text-ink2">{s.persona ?? "–"}</td>
                  <td><button className="text-xs text-critical hover:underline" onClick={() => remove(s.id)}>delete</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
