import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import { AffectPanel, DrillsPanel, GazePanel, ModulesPanel, PhysioPanel, RhetoricPanel, TraitCards, TranscriptPanel, VoicePanel } from "../components/Panels";
import ProgressView from "../components/ProgressView";
import Timeline, { type TimelineHandle } from "../components/Timeline";
import type { Report, SessionRow } from "../types";
import { fmtTime } from "../util";

export default function SessionPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const [session, setSession] = useState<SessionRow | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timeline = useRef<TimelineHandle>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const s = await api.session(id);
        if (cancelled) return;
        setSession(s);
        if (s.status === "done") setReport(await api.report(id));
        else if (s.status !== "failed") timer = window.setTimeout(poll, 1500);
      } catch (err) {
        if (!cancelled) setError((err as Error).message);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [id]);

  if (error) return <p className="text-critical">{error}</p>;
  if (!session) return <p className="text-muted">Loading…</p>;
  if (!report) return <ProgressView session={session} />;

  const seek = (t: number) => {
    timeline.current?.seek(t);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const refresh = async () => setReport(await api.report(id));
  const deleteSession = async () => {
    if (!window.confirm("Delete this session, its report and its video? This cannot be undone.")) return;
    await api.deleteSession(id);
    navigate("/history");
  };
  const deleteVideo = async () => {
    if (!window.confirm("Delete the raw video? The report stays.")) return;
    await api.deleteVideo(id);
    await refresh();
  };
  const reanalyse = async () => {
    await api.reanalyse(id);
    window.location.reload();
  };
  const keep = async (keepVideo: boolean) => {
    await api.patchSession(id, { keep_video: keepVideo });
    await refresh();
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-bold">{session.title}</h1>
        <span className="text-sm text-muted">
          {new Date(session.created_at).toLocaleString()} · {fmtTime(report.duration_s)}
          {session.persona && <> · persona: <span className="font-medium text-ink2">{session.persona}</span></>}
        </span>
      </div>
      <TraitCards report={report} />
      <Timeline ref={timeline} report={report} videoUrl={session.video_deleted ? null : api.videoUrl(id)} />
      <DrillsPanel report={report} seek={seek} />
      <div className="grid gap-4 lg:grid-cols-2">
        <VoicePanel report={report} seek={seek} />
        <GazePanel report={report} />
        <PhysioPanel report={report} />
        <AffectPanel report={report} />
      </div>
      <RhetoricPanel report={report} seek={seek} />
      <TranscriptPanel report={report} seek={seek} />
      <ModulesPanel report={report} onDeleteVideo={deleteVideo} onDeleteSession={deleteSession} onKeep={keep} onReanalyse={reanalyse} />
    </div>
  );
}
