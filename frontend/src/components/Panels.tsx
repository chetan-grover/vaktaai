import { useState } from "react";
import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { api } from "../api";
import type { Report } from "../types";
import { fmt, fmtTime, pct, scoreTone, useTokens } from "../util";

type Seek = (t: number) => void;

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-line p-3">
      <div className="label">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      {hint && <div className="text-xs text-muted">{hint}</div>}
    </div>
  );
}

function Section({ title, children, aside }: { title: string; children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <section className="card space-y-3">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

const ProxyNote = ({ show }: { show: boolean }) =>
  show ? <span className="text-xs text-muted">fallback model (proxy) — train/download the learned one for best accuracy</span> : null;

// ------------------------------------------------------------------ traits (F11, NFR6)
export function TraitCards({ report }: { report: Report }) {
  const traits = report.traits;
  const cong = report.congruence;
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {traits && (["confidence", "authoritativeness", "engagement"] as const).map((k) => (
        <div key={k} className="card">
          <div className="label">{traits[k].label}</div>
          <div className={`mt-1 text-4xl font-bold ${scoreTone(traits[k].score)}`}>{fmt(traits[k].score)}</div>
          <p className="mt-2 text-sm text-ink2">{traits[k].why}</p>
        </div>
      ))}
      <div className="card">
        <div className="label">Emotional congruence</div>
        <div className={`mt-1 text-4xl font-bold ${scoreTone(cong?.score)}`}>{fmt(cong?.score)}</div>
        <p className="mt-2 text-sm text-ink2">
          {cong?.mismatch_count ? `${cong.mismatch_count} segment(s) where voice and face disagree.` : "Voice and face tell the same story."}
          <span className="block text-xs text-muted">method: {cong?.method ?? "–"}{cong?.proxy_inputs ? " · proxy inputs" : ""}</span>
        </p>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ voice (F4, F5, S2)
export function VoicePanel({ report, seek }: { report: Report; seek: Seek }) {
  const v = report.voice;
  if (!v) return <Section title="Voice"><p className="text-sm text-muted">No audio analysis.</p></Section>;
  const s = v.summary as Record<string, number | null>;
  return (
    <Section title="Voice" aside={<span className="text-xs text-muted">ideal pace 130–160 wpm</span>}>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Pace" value={fmt(s.wpm, 0, " wpm")} hint={`± ${fmt(s.wpm_std, 0)} across windows`} />
        <Stat label="Fillers" value={fmt(s.filler_rate_per_min, 1, "/min")} hint={`${s.filler_count ?? 0} total, ${s.hinglish_filler_count ?? 0} Hinglish`} />
        <Stat label="Pitch variation" value={fmt(s.pitch_std_st, 1, " st")} hint="below 2 st sounds monotone" />
        <Stat label="Pauses" value={fmt(s.pauses_per_min, 1, "/min")} hint={`${s.long_pause_count ?? 0} longer than 2 s`} />
      </div>
      {v.fillers.length > 0 && (
        <div>
          <div className="label mb-1">Fillers (click to hear)</div>
          <div className="flex flex-wrap gap-1.5">
            {v.fillers.map((f, i) => (
              <button key={i} onClick={() => seek(f.start)} className="rounded-full border border-line px-2 py-0.5 text-xs hover:bg-grid">
                {f.text} <span className="tabular text-muted">{fmtTime(f.start)}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </Section>
  );
}

// ------------------------------------------------------------------ gaze heatmap (E2, F18)
function mix(a: string, b: string, t: number): string {
  const p = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * t)).join(",")})`;
}

export function GazePanel({ report }: { report: Report }) {
  const tk = useTokens();
  const g = report.gaze;
  if (!g) return <Section title="Audience gaze"><p className="text-sm text-muted">No face found, so gaze was not analysed.</p></Section>;
  const s = g.summary;
  const max = Math.max(...s.dwell, 0.01);
  return (
    <Section title="Audience gaze" aside={<span className="text-xs text-muted">calibration: {g.calibration.method}</span>}>
      <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div>
          <div className="mb-1 text-center text-xs text-muted">back of the room</div>
          <div className="grid grid-cols-3 gap-0.5 overflow-hidden rounded-lg">
            {s.dwell.map((d, i) => {
              const t = d / max;
              const neglected = s.neglected_cells.includes(i);
              return (
                <div key={i} title={`${g.cell_names[i]}: ${pct(d)} of the time`}
                  className="flex aspect-[4/3] flex-col items-center justify-center text-center"
                  style={{ background: mix(tk["--seq-100"] || "#cde2fb", tk["--seq-650"] || "#104281", t), color: t > 0.55 ? "#fff" : tk["--text-primary"] }}>
                  <span className="text-lg font-semibold">{pct(d)}</span>
                  <span className="text-[11px] opacity-80">{g.cell_names[i]}</span>
                  {neglected && <span className="mt-0.5 text-[11px] font-semibold">⚠ neglected</span>}
                </div>
              );
            })}
          </div>
          <div className="mt-1 text-center text-xs text-muted">front · left/right are <em>your</em> left/right</div>
        </div>
        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Room coverage" value={pct(s.coverage)} hint="zones looked at ≥ 4% of the time" />
            <Stat label="Camera / centre" value={pct(s.eye_contact_ratio)} hint={`off-audience ${pct(s.off_grid_ratio)}`} />
          </div>
          {g.alerts.map((a) => (
            <p key={a.cell} className="text-sm"><span className="font-semibold text-warn">⚠</span> {a.message}</p>
          ))}
        </div>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ physiology (E1, F17)
export function PhysioPanel({ report }: { report: Report }) {
  const p = report.physio;
  if (!p) return <Section title="Heart rate (rPPG)"><p className="text-sm text-muted">Not enough stable face video for rPPG.</p></Section>;
  const s = p.summary as Record<string, number | null>;
  return (
    <Section title="Heart rate (rPPG)" aside={<span className="text-xs text-muted">{report.reproducibility.models.rppg}</span>}>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Start (baseline)" value={fmt(s.bpm_baseline, 0, " BPM")} />
        <Stat label="Peak" value={fmt(s.bpm_max, 0, " BPM")} hint={s.bpm_rise != null ? `+${fmt(s.bpm_rise, 0)} over baseline` : undefined} />
        <Stat label="HRV (RMSSD)" value={fmt(s.rmssd_mean_ms, 0, " ms")} />
        <Stat label="Reliable windows" value={pct(s.reliable_ratio)} hint="low-quality windows are hidden" />
      </div>
      <p className="text-xs text-muted">{p.disclaimer}</p>
    </Section>
  );
}

// ------------------------------------------------------------------ affect (E3, F19)
export function AffectPanel({ report }: { report: Report }) {
  const tk = useTokens();
  const segs = report.segments.filter((s) => s.features?.face_valence != null || s.features?.audio_valence != null);
  if (!segs.length) return null;
  const face = segs.filter((s) => s.features.face_valence != null).map((s) => ({ x: s.features.face_valence, y: s.features.face_arousal, t: s.start }));
  const voice = segs.filter((s) => s.features.audio_valence != null).map((s) => ({ x: s.features.audio_valence, y: s.features.audio_arousal, t: s.start }));
  const proxy = Boolean(report.affect.face?.proxy || report.affect.audio?.proxy);
  return (
    <Section title="Valence–arousal per segment" aside={<ProxyNote show={proxy} />}>
      <div className="h-64">
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: -12 }}>
            <CartesianGrid stroke={tk["--grid"]} />
            <XAxis type="number" dataKey="x" name="valence" domain={[-1, 1]} ticks={[-1, 0, 1]} stroke={tk["--axis"]} tick={{ fill: tk["--text-muted"], fontSize: 11 }}
              label={{ value: "valence →", position: "insideBottomRight", fill: tk["--text-muted"], fontSize: 11 }} />
            <YAxis type="number" dataKey="y" name="arousal" domain={[-1, 1]} ticks={[-1, 0, 1]} stroke={tk["--axis"]} tick={{ fill: tk["--text-muted"], fontSize: 11 }}
              label={{ value: "arousal", angle: -90, position: "insideLeft", fill: tk["--text-muted"], fontSize: 11 }} />
            <ZAxis range={[60, 60]} />
            <Tooltip formatter={(v: number) => fmt(v, 2)} contentStyle={{ background: tk["--surface-1"], border: `1px solid ${tk["--grid"]}`, borderRadius: 8, fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Scatter name="Face" data={face} fill={tk["--series-1"]} stroke={tk["--surface-1"]} strokeWidth={2} />
            <Scatter name="Voice" data={voice} fill={tk["--series-2"]} stroke={tk["--surface-1"]} strokeWidth={2} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-muted">Each dot is a ~5 s segment. Far-apart face and voice dots for the same moment mean the channels disagree. Indicative only; not lie detection.</p>
    </Section>
  );
}

// ------------------------------------------------------------------ rhetoric (E4, F9, F20)
export function RhetoricPanel({ report, seek }: { report: Report; seek: Seek }) {
  const c = report.content;
  if (!c) return <Section title="Content & rhetoric"><p className="text-sm text-muted">No transcript, so content was not analysed.</p></Section>;
  const r = c.rhetoric;
  const appeals = [["Ethos (credibility)", r.appeals.ethos], ["Pathos (emotion)", r.appeals.pathos], ["Logos (logic)", r.appeals.logos]] as const;
  return (
    <Section title="Content & rhetoric" aside={<span className="text-xs text-muted">{report.reproducibility.models.rhetoric}</span>}>
      <p className="text-sm text-ink2">{r.summary}</p>
      {c.note && <p className="text-xs text-muted">{c.note}</p>}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-3">
          <div>
            <div className="label mb-1">Appeal balance</div>
            {appeals.map(([label, v]) => (
              <div key={label} className="mb-1.5" title={`${label}: ${pct(v)}`}>
                <div className="flex justify-between text-sm"><span>{label}</span><span className="tabular">{pct(v)}</span></div>
                <div className="h-2 rounded-full bg-grid"><div className="h-2 rounded-full bg-accent" style={{ width: `${v * 100}%` }} /></div>
              </div>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Opening" value={c.structure.opener.type} hint={`“${c.structure.opener.text.slice(0, 60)}”`} />
            <Stat label="Closing" value={c.structure.closer.type} hint={`“${c.structure.closer.text.slice(0, 60)}”`} />
          </div>
          <Stat label="Transitions" value={pct(r.transition_quality)} hint={r.transitions.map((t) => t.text).slice(0, 8).join(", ") || "none signposted"} />
        </div>
        <div className="space-y-3">
          <div>
            <div className="label mb-1">Devices found ({r.devices.length})</div>
            <ul className="max-h-48 space-y-1 overflow-y-auto text-sm">
              {r.devices.map((d, i) => (
                <li key={i}>
                  <button className="text-left hover:underline" onClick={() => d.start != null && seek(d.start)}>
                    <span className="font-medium">{d.type.replace(/_/g, " ")}</span>
                    {d.start != null && <span className="tabular text-xs text-muted"> {fmtTime(d.start)}</span>}
                    <span className="block text-xs text-ink2">“{d.text.slice(0, 120)}”</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
          {r.suggestions.length > 0 && (
            <div>
              <div className="label mb-1">Rewrite suggestions</div>
              {r.suggestions.map((s, i) => (
                <div key={i} className="mb-2 rounded-lg border border-line p-2 text-sm">
                  <div className="line-clamp-2 text-ink2 line-through decoration-muted" title={s.original}>{s.original}</div>
                  <div className="line-clamp-4 font-medium" title={s.rewrite}>{s.rewrite}</div>
                  <div className="text-xs text-muted">{s.reason}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Section>
  );
}

// ------------------------------------------------------------------ drills (F14, UC3)
export function DrillsPanel({ report, seek }: { report: Report; seek: Seek }) {
  const [done, setDone] = useState<Record<string, boolean>>(report.drill_progress ?? {});
  const weaknesses = report.drills?.weaknesses ?? [];
  const toggle = async (id: string) => {
    const next = !done[id];
    setDone((d) => ({ ...d, [id]: next }));
    try {
      await api.markDrill(report.session_id, id, next);
    } catch {
      setDone((d) => ({ ...d, [id]: !next }));
    }
  };
  const total = new Set(weaknesses.flatMap((w) => w.drills.map((d) => d.id))).size;
  const completed = Object.values(done).filter(Boolean).length;
  return (
    <Section title="Your drills" aside={<span className="text-xs text-muted">{completed}/{total} done</span>}>
      {weaknesses.length === 0 && <p className="text-sm text-ink2">No clear weaknesses this time. Keep practising.</p>}
      {weaknesses.map((w) => (
        <div key={w.id} className="rounded-lg border border-line p-3">
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="font-semibold">{w.label}</span>
            <span className="text-xs text-muted">{w.metric.split(".").pop()?.replace(/_/g, " ")} = {fmt(w.value, 2)} (threshold {w.threshold})</span>
            {w.evidence && (
              <button className="ml-auto text-xs font-medium text-accent hover:underline" onClick={() => seek(w.evidence!.start)}>
                ▶ evidence at {fmtTime(w.evidence.start)}
              </button>
            )}
          </div>
          <div className="mt-2 grid gap-2 md:grid-cols-3">
            {w.drills.map((d) => (
              <label key={d.id} className={`flex cursor-pointer gap-2 rounded-lg border border-line p-2 text-sm ${done[d.id] ? "opacity-60" : ""}`}>
                <input type="checkbox" className="mt-1" checked={Boolean(done[d.id])} onChange={() => toggle(d.id)} />
                <span>
                  <span className="font-medium">{d.title}</span> <span className="text-xs text-muted">{d.minutes} min</span>
                  <ol className="ml-4 mt-1 list-decimal text-xs text-ink2">{d.steps.map((s, i) => <li key={i}>{s}</li>)}</ol>
                  <span className="mt-1 block text-xs text-muted">{d.why}</span>
                </span>
              </label>
            ))}
          </div>
        </div>
      ))}
    </Section>
  );
}

// ------------------------------------------------------------------ transcript
export function TranscriptPanel({ report, seek }: { report: Report; seek: Seek }) {
  const words = report.transcript?.words ?? [];
  const fillerStarts = new Set((report.voice?.fillers ?? []).map((f) => f.start));
  if (!words.length) return null;
  return (
    <Section title="Transcript" aside={<span className="text-xs text-muted">language: {report.transcript?.language} · fillers underlined</span>}>
      <p className="max-h-56 overflow-y-auto text-sm leading-7">
        {words.map((w, i) => (
          <span key={i} onClick={() => seek(w.start)}
            className={`cursor-pointer rounded px-0.5 hover:bg-grid ${fillerStarts.has(w.start) ? "font-semibold text-serious underline" : ""}`}>
            {w.w}{" "}
          </span>
        ))}
      </p>
    </Section>
  );
}

// ------------------------------------------------------------------ modules + privacy (NFR3-NFR5, FR10)
export function ModulesPanel({ report, onDeleteVideo, onDeleteSession, onKeep, onReanalyse }: {
  report: Report; onDeleteVideo: () => void; onDeleteSession: () => void; onKeep: (keep: boolean) => void; onReanalyse: () => void;
}) {
  const tone: Record<string, string> = { ok: "text-[color:var(--success-text)]", skipped: "text-muted", error: "text-critical", missing: "text-muted" };
  return (
    <Section title="Under the hood">
      <div className="grid gap-4 md:grid-cols-2">
        <table className="tabular w-full text-sm">
          <tbody>
            {Object.entries(report.modules).map(([name, m]) => (
              <tr key={name} className="border-b border-line last:border-0" title={m.error ?? ""}>
                <td className="py-1">{name}</td>
                <td className={`py-1 ${tone[m.status] ?? ""}`}>{m.status}</td>
                <td className="py-1 text-right text-muted">{m.elapsed_s != null ? `${m.elapsed_s.toFixed(1)} s` : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="space-y-2 text-sm">
          <div className="label">Models used</div>
          <ul className="text-xs text-ink2">
            {Object.entries(report.reproducibility.models).map(([k, v]) => <li key={k}><span className="text-muted">{k}:</span> {v}</li>)}
          </ul>
          <p className="text-xs text-muted">{report.disclaimer}</p>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={report.session.keep_video} onChange={(e) => onKeep(e.target.checked)} />
            Keep raw video beyond 7 days
          </label>
          <div className="flex flex-wrap gap-2">
            <button className="btn" onClick={onReanalyse} disabled={report.session.video_deleted}>Re-analyse</button>
            <button className="btn" onClick={onDeleteVideo} disabled={report.session.video_deleted}>Delete video only</button>
            <button className="btn text-critical" onClick={onDeleteSession}>Delete session</button>
          </div>
        </div>
      </div>
    </Section>
  );
}
