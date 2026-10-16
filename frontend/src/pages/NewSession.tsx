import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import Recorder, { CALIBRATION_S } from "../components/Recorder";

const MAX_MB = 200;

export default function NewSession() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<"record" | "upload">("record");
  const [title, setTitle] = useState("");
  const [file, setFile] = useState<File | Blob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!file) return;
    if (file.size > MAX_MB * 1024 * 1024) return setError(`File is larger than ${MAX_MB} MB.`);
    setBusy(true);
    setError(null);
    const form = new FormData();
    const name = file instanceof File ? file.name : "recording.webm";
    form.append("file", file, name);
    form.append("title", title.trim() || (mode === "record" ? "Webcam practice" : name));
    form.append("source", mode === "record" ? "webcam" : "upload");
    if (mode === "record") form.append("calibration_s", String(CALIBRATION_S));
    try {
      const session = await api.upload(form);
      navigate(`/sessions/${session.id}`);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  };

  const tab = (m: typeof mode, label: string) => (
    <button
      className={`rounded-lg px-3 py-1.5 text-sm font-medium ${mode === m ? "bg-grid text-ink" : "text-ink2"}`}
      onClick={() => {
        setMode(m);
        setFile(null);
      }}
    >
      {label}
    </button>
  );

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <div>
        <h1 className="text-2xl font-bold">New practice session</h1>
        <p className="text-sm text-ink2">Record a 30 s to 5 min talk, interview answer or debate speech, or upload a video.</p>
      </div>
      <div className="card space-y-4">
        <div className="flex gap-1">{tab("record", "Record with webcam")}{tab("upload", "Upload a video")}</div>
        {mode === "record" ? (
          <Recorder onDone={setFile} />
        ) : (
          <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-line p-10 text-center hover:bg-grid">
            <span className="font-medium">{file instanceof File ? file.name : "Choose an MP4, WebM or MOV file"}</span>
            <span className="mt-1 text-xs text-muted">Max {MAX_MB} MB, 30 s to 5 min, one frontal speaker.</span>
            <input type="file" accept="video/mp4,video/webm,video/quicktime,.mov" className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
        )}
        <label className="block">
          <span className="label">Title</span>
          <input className="mt-1 w-full rounded-lg border border-line bg-surface px-3 py-2"
            value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Seminar opening, take 2" />
        </label>
        {error && <p className="text-sm text-critical">{error}</p>}
        <button className="btn-primary" disabled={!file || busy} onClick={submit}>
          {busy ? "Uploading…" : "Analyse"}
        </button>
      </div>
    </div>
  );
}
