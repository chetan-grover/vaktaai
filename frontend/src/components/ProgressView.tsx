import type { SessionRow, StageStatus } from "../types";

const LABELS: Record<string, string> = {
  ingest: "Ingest (ffmpeg)", asr: "Transcription", voice: "Voice + fillers", audio_va: "Voice emotion (E3)",
  content: "Rhetoric (E4)", face: "Face mesh + blinks", gaze: "Audience gaze (E2)", face_va: "Facial emotion (E3)",
  physio: "Heart rate (E1)", fusion: "Fusion + traits", timeline: "Timeline", drills: "Drills", report: "Report",
};
const STREAMS: [string, string[]][] = [
  ["Preparation", ["ingest"]],
  ["Audio stream", ["asr", "voice", "audio_va"]],
  ["Transcript stream", ["content"]],
  ["Video stream", ["face", "gaze", "face_va", "physio"]],
  ["Fusion", ["fusion", "timeline", "drills", "report"]],
];

function Badge({ s }: { s: StageStatus | undefined }) {
  const status = s?.status ?? "pending";
  const map: Record<string, [string, string]> = {
    pending: ["○", "text-muted"], running: ["◔", "text-accent animate-pulse"], ok: ["✓", "text-[color:var(--success-text)]"],
    skipped: ["–", "text-muted"], error: ["✕", "text-critical"],
  };
  const [icon, cls] = map[status];
  return <span className={`inline-block w-5 text-center font-bold ${cls}`} aria-label={status}>{icon}</span>;
}

export default function ProgressView({ session }: { session: SessionRow }) {
  const stages = session.progress?.stages ?? {};
  const done = session.progress?.done ?? 0;
  const total = session.progress?.total ?? 13;
  return (
    <div className="card mx-auto max-w-2xl space-y-4">
      <div>
        <h2 className="text-lg font-semibold">
          {session.status === "failed" ? "Analysis failed" : session.status === "queued" ? "Waiting in the queue…" : "Analysing your talk…"}
        </h2>
        <p className="text-sm text-ink2">{session.title}</p>
      </div>
      {session.status === "queued" && (
        <p className="text-sm text-ink2">Another analysis is running; this one starts as soon as it finishes.</p>
      )}
      {session.status === "failed" ? (
        <p className="text-sm text-critical">{session.error}</p>
      ) : (
        <div className="h-2 overflow-hidden rounded-full bg-grid">
          <div className="h-full bg-accent transition-all" style={{ width: `${(100 * done) / total}%` }} />
        </div>
      )}
      <div className="grid gap-4 sm:grid-cols-2">
        {STREAMS.map(([name, names]) => (
          <div key={name}>
            <div className="label mb-1">{name}</div>
            {names.map((n) => (
              <div key={n} className="flex items-center gap-2 text-sm">
                <Badge s={stages[n]} />
                <span className={stages[n]?.status === "running" ? "font-medium" : "text-ink2"}>{LABELS[n]}</span>
                {stages[n]?.elapsed_s != null && <span className="tabular ml-auto text-xs text-muted">{stages[n]!.elapsed_s!.toFixed(1)} s</span>}
              </div>
            ))}
          </div>
        ))}
      </div>
      <p className="text-xs text-muted">Skipped or failed modules don't stop the report; it shows what ran.</p>
    </div>
  );
}
