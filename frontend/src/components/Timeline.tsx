import { forwardRef, useImperativeHandle, useMemo, useRef, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceArea, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Report, TimelineEvent } from "../types";
import { fmt, fmtTime, useTokens } from "../util";

export interface TimelineHandle {
  seek: (t: number) => void;
}

const EVENT_META: Record<TimelineEvent["type"], { icon: string; label: string; tone: string }> = {
  confidence_dip: { icon: "▼", label: "Confidence dip", tone: "text-critical" },
  mismatch: { icon: "≠", label: "Voice/face mismatch", tone: "text-serious" },
  bpm_spike: { icon: "♥", label: "Heart-rate rise", tone: "text-serious" },
  filler_cluster: { icon: "…", label: "Filler cluster", tone: "text-warn" },
  long_pause: { icon: "‖", label: "Long pause", tone: "text-muted" },
  neglected_zone: { icon: "◌", label: "Neglected zone", tone: "text-warn" },
};

/** UC2: video synced with stacked timeline charts (one y-axis each, shared time axis). */
const Timeline = forwardRef<TimelineHandle, { report: Report; videoUrl: string | null }>(({ report, videoUrl }, ref) => {
  const tk = useTokens();
  const video = useRef<HTMLVideoElement>(null);
  const [now, setNow] = useState(0);
  const [filter, setFilter] = useState<string>("all");

  const seek = (t: number) => {
    if (video.current) {
      video.current.currentTime = Math.max(0, t - 0.5);
      void video.current.play().catch(() => undefined);
    }
    setNow(t);
  };
  useImperativeHandle(ref, () => ({ seek }));

  const ch = report.timeline?.channels ?? {};
  const rows = useMemo(
    () => (ch.t ?? []).map((t, i) => ({
      t: t as number,
      confidence: ch.confidence?.[i] ?? null,
      bpm: ch.bpm?.[i] ?? null,
      face: ch.face_valence?.[i] ?? null,
      voice: ch.audio_valence?.[i] ?? null,
    })),
    [ch],
  );
  const events = report.timeline?.events ?? [];
  const has = (k: "bpm" | "face" | "voice") => rows.some((r) => r[k] != null);
  const duration = report.duration_s;
  const onClick = (s: { activeLabel?: string | number } | null) => s?.activeLabel != null && seek(Number(s.activeLabel));

  const axis = { stroke: tk["--axis"], tick: { fill: tk["--text-muted"], fontSize: 11 } };
  const xAxis = (show: boolean) => (
    <XAxis dataKey="t" type="number" domain={[0, Math.ceil(duration)]} tickFormatter={fmtTime} {...axis} hide={!show} />
  );
  const tooltip = (
    <Tooltip
      labelFormatter={(t) => fmtTime(Number(t))}
      contentStyle={{ background: tk["--surface-1"], border: `1px solid ${tk["--grid"]}`, borderRadius: 8, fontSize: 12, color: tk["--text-primary"] }}
    />
  );
  const cursor = <ReferenceLine x={now} stroke={tk["--text-secondary"]} strokeDasharray="3 3" />;
  const mismatchAreas = events.filter((e) => e.type === "mismatch").map((e, i) => (
    <ReferenceArea key={`m${i}`} x1={e.start} x2={e.end} fill={tk["--status-serious"]} fillOpacity={0.15} stroke="none" />
  ));
  const shown = events.filter((e) => filter === "all" || e.type === filter);

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <div className="space-y-3">
        {videoUrl ? (
          <video ref={video} src={videoUrl} controls className="aspect-video w-full rounded-xl border border-line bg-black"
            onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)} />
        ) : (
          <div className="flex aspect-video items-center justify-center rounded-xl border border-line text-sm text-muted">Video deleted; the report is kept.</div>
        )}
        <div className="card max-h-80 overflow-y-auto p-0">
          <div className="sticky top-0 flex flex-wrap items-center gap-2 border-b border-line bg-surface px-3 py-2">
            <span className="label">Moments</span>
            <select className="ml-auto rounded-md border border-line bg-surface px-2 py-1 text-xs" value={filter} onChange={(e) => setFilter(e.target.value)}>
              <option value="all">All ({events.length})</option>
              {Object.entries(EVENT_META).map(([k, m]) => (
                <option key={k} value={k}>{m.label} ({report.timeline?.counts[k] ?? 0})</option>
              ))}
            </select>
          </div>
          {shown.length === 0 && <p className="p-3 text-sm text-muted">Nothing flagged.</p>}
          {shown.map((e, i) => (
            <button key={i} onClick={() => seek(e.start)} className="flex w-full gap-3 border-b border-line px-3 py-2 text-left last:border-0 hover:bg-grid">
              <span className={`w-4 text-center font-bold ${EVENT_META[e.type].tone}`}>{EVENT_META[e.type].icon}</span>
              <span className="tabular w-10 text-xs text-muted">{fmtTime(e.start)}</span>
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-medium">{e.label}</span>
                <span className="block text-xs text-ink2">{e.why}</span>
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="card space-y-1">
        <div className="flex items-baseline justify-between">
          <span className="label">Confidence per segment (0-100)</span>
          <span className="text-xs text-muted">click a chart to jump the video</span>
        </div>
        <div className="h-44">
          <ResponsiveContainer>
            <LineChart data={rows} syncId="tl" onClick={onClick} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
              <CartesianGrid stroke={tk["--grid"]} vertical={false} />
              {xAxis(false)}
              <YAxis domain={[0, 100]} ticks={[0, 50, 100]} {...axis} />
              {mismatchAreas}
              <Line dataKey="confidence" name="Confidence" type="stepAfter" stroke={tk["--series-1"]} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
              {events.filter((e) => e.type === "confidence_dip").map((e, i) => {
                const y = rows[Math.floor((e.start + e.end) / 2)]?.confidence;
                return y != null ? <ReferenceDot key={i} x={(e.start + e.end) / 2} y={y} r={5} fill={tk["--status-critical"]} stroke={tk["--surface-1"]} strokeWidth={2} /> : null;
              })}
              {cursor}
              {tooltip}
            </LineChart>
          </ResponsiveContainer>
        </div>
        {has("bpm") && (
          <>
            <span className="label">Heart rate, BPM (indicative)</span>
            <div className="h-32">
              <ResponsiveContainer>
                <LineChart data={rows} syncId="tl" onClick={onClick} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
                  <CartesianGrid stroke={tk["--grid"]} vertical={false} />
                  {xAxis(false)}
                  <YAxis domain={[(lo: number) => Math.floor((lo - 5) / 10) * 10, (hi: number) => Math.ceil((hi + 5) / 10) * 10]}
                    allowDecimals={false} tickCount={4} {...axis} />
                  {events.filter((e) => e.type === "bpm_spike").map((e, i) => (
                    <ReferenceArea key={i} x1={e.start} x2={e.end} fill={tk["--status-serious"]} fillOpacity={0.12} stroke="none" />
                  ))}
                  <Line dataKey="bpm" name="BPM" stroke={tk["--series-1"]} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                  {cursor}
                  {tooltip}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
        {(has("face") || has("voice")) && (
          <>
            <span className="label">Valence: face vs voice (−1 negative … +1 positive)</span>
            <div className="h-40">
              <ResponsiveContainer>
                <LineChart data={rows} syncId="tl" onClick={onClick} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
                  <CartesianGrid stroke={tk["--grid"]} vertical={false} />
                  {xAxis(true)}
                  <YAxis domain={[-1, 1]} ticks={[-1, 0, 1]} {...axis} />
                  <ReferenceLine y={0} stroke={tk["--axis"]} />
                  {mismatchAreas}
                  <Line dataKey="face" name="Face" stroke={tk["--series-1"]} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                  <Line dataKey="voice" name="Voice" stroke={tk["--series-2"]} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                  <Legend verticalAlign="top" height={20} iconType="plainline" wrapperStyle={{ fontSize: 12, color: tk["--text-secondary"] }} />
                  {cursor}
                  <Tooltip formatter={(v: number) => fmt(v, 2)} labelFormatter={(t) => fmtTime(Number(t))}
                    contentStyle={{ background: tk["--surface-1"], border: `1px solid ${tk["--grid"]}`, borderRadius: 8, fontSize: 12, color: tk["--text-primary"] }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
        <p className="pt-1 text-xs text-muted">Shaded bands: voice/face mismatch or heart-rate rise. Red dots: confidence dips.</p>
      </div>
    </div>
  );
});

export default Timeline;
