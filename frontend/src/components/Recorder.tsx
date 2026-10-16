import { useEffect, useRef, useState } from "react";
import { fmtTime } from "../util";

const MIN_S = 30;
const MAX_S = 300;
export const CALIBRATION_S = 3;

/** Webcam recorder (F1). The first 3 s ask the speaker to look into the camera: gaze.py uses
 * that window to calibrate the centre of the virtual audience grid. */
export default function Recorder({ onDone }: { onDone: (blob: Blob) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recRef = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const [state, setState] = useState<"idle" | "preview" | "recording" | "done">("idle");
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => () => streamRef.current?.getTracks().forEach((t) => t.stop()), []);

  useEffect(() => {
    if (state !== "recording") return;
    const started = Date.now();
    const id = window.setInterval(() => {
      const s = (Date.now() - started) / 1000;
      setElapsed(s);
      if (s >= MAX_S) stop();
    }, 250);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  const openCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, frameRate: 30 },
        audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setState("preview");
    } catch (err) {
      setError(`Camera/microphone unavailable: ${(err as Error).message}`);
    }
  };

  const start = () => {
    if (!streamRef.current) return;
    chunks.current = [];
    const mime = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"].find((m) => MediaRecorder.isTypeSupported(m));
    const rec = new MediaRecorder(streamRef.current, mime ? { mimeType: mime } : undefined);
    rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data);
    rec.onstop = () => {
      onDone(new Blob(chunks.current, { type: "video/webm" }));
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
    rec.start(1000);
    recRef.current = rec;
    setElapsed(0);
    setState("recording");
  };

  const stop = () => {
    if (recRef.current?.state === "recording") recRef.current.stop();
    setState("done");
  };

  const calibrating = state === "recording" && elapsed < CALIBRATION_S;

  return (
    <div className="space-y-3">
      <div className="relative aspect-video overflow-hidden rounded-xl border border-line bg-black">
        <video ref={videoRef} muted playsInline className="h-full w-full -scale-x-100 object-cover" />
        {state === "idle" && (
          <div className="absolute inset-0 flex items-center justify-center">
            <button className="btn-primary" onClick={openCamera}>Turn on camera</button>
          </div>
        )}
        {calibrating && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/50 text-white">
            <div className="text-5xl font-bold">{Math.ceil(CALIBRATION_S - elapsed)}</div>
            <div className="mt-2 text-lg">Look straight into the camera</div>
          </div>
        )}
        {state === "recording" && (
          <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-sm text-white">
            <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-[#d03b3b]" /> REC {fmtTime(elapsed)} / {fmtTime(MAX_S)}
          </div>
        )}
      </div>
      <div className="flex items-center gap-3">
        {state === "preview" && <button className="btn-primary" onClick={start}>Start recording</button>}
        {state === "recording" && (
          <button className="btn" onClick={stop} disabled={elapsed < MIN_S}>
            {elapsed < MIN_S ? `Stop (available in ${Math.ceil(MIN_S - elapsed)} s)` : "Stop recording"}
          </button>
        )}
        {state === "done" && <span className="text-sm text-ink2">Recorded {fmtTime(elapsed)}.</span>}
        <span className="text-xs text-muted">30 s to 5 min. Speak to an imagined audience spread across the room.</span>
      </div>
      {error && <p className="text-sm text-critical">{error}</p>}
    </div>
  );
}
