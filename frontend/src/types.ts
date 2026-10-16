export interface User {
  id: number;
  name: string;
  email: string | null;
}

export interface StageStatus {
  status: "pending" | "running" | "ok" | "skipped" | "error";
  elapsed_s?: number;
  error?: string | null;
}

export interface SessionRow {
  id: string;
  user_id: number;
  title: string;
  source: string;
  created_at: string;
  status: "queued" | "processing" | "done" | "failed";
  error: string | null;
  duration_s: number | null;
  confidence: number | null;
  authoritativeness: number | null;
  engagement: number | null;
  congruence: number | null;
  persona: string | null;
  keep_video: boolean;
  video_deleted: boolean;
  progress?: { stages: Record<string, StageStatus>; done: number; total: number } | null;
}

export interface Trait {
  label: string;
  score: number | null;
  why: string;
}

export interface TimelineEvent {
  type: "confidence_dip" | "mismatch" | "bpm_spike" | "filler_cluster" | "long_pause" | "neglected_zone";
  start: number;
  end: number;
  severity: number;
  label: string;
  why: string;
}

export interface Word {
  w: string;
  start: number;
  end: number;
}

export interface Device {
  type: string;
  text: string;
  explanation: string;
  start: number | null;
}

export interface Drill {
  id: string;
  title: string;
  minutes: number;
  focus: string;
  steps: string[];
  why: string;
}

export interface Weakness {
  id: string;
  label: string;
  metric: string;
  value: number;
  threshold: number;
  severity: number;
  evidence: { start: number; end: number; text: string; value: number | null } | null;
  drills: Drill[];
}

type Num = number | null;
type Summary = Record<string, Num | number[] | undefined>;

export interface Report {
  session_id: string;
  duration_s: number;
  disclaimer: string;
  modules: Record<string, { status: string; error: string | null; elapsed_s: number | null }>;
  reproducibility: { models: Record<string, string>; packages: Record<string, string>; seed: number };
  traits: Record<"confidence" | "authoritativeness" | "engagement", Trait> | null;
  congruence: { score: Num; mean_distance: Num; method: string; mismatch_count: number; proxy_inputs: boolean } | null;
  segments: { id: number; start: number; end: number; text: string; traits: Record<string, Num>; mismatch: boolean; congruence_distance: Num; features: Record<string, Num> }[];
  mismatches: { start: number; end: number; reason: string; text: string }[];
  timeline: {
    channels: Record<string, (number | null)[]>;
    events: TimelineEvent[];
    counts: Record<string, number>;
  } | null;
  transcript: { text: string; words: Word[]; language: string } | null;
  voice: {
    summary: Summary;
    fillers: { text: string; start: number; end: number; lang: string }[];
    pauses: { start: number; end: number; duration: number }[];
  } | null;
  face: { summary: Summary; blinks: number[] } | null;
  gaze: {
    summary: Summary & { dwell: number[]; neglected_cells: number[] };
    alerts: { cell: number; name: string; message: string }[];
    cell_names: string[];
    calibration: { method: string };
  } | null;
  physio: { summary: Summary; windows: { start: number; end: number; bpm: Num; rmssd_ms: Num; reliable: boolean }[]; disclaimer: string } | null;
  affect: { face: (Summary & { proxy: boolean }) | null; audio: (Summary & { proxy: boolean }) | null };
  content: {
    structure: { opener: { type: string; text: string; score: number }; closer: { type: string; text: string; score: number }; mean_sentence_words: Num; repetition: { phrase: string; count: number }[] };
    rhetoric: {
      devices: Device[];
      appeals: { ethos: number; pathos: number; logos: number };
      transitions: { text: string; category: string; start: number | null }[];
      transition_quality: number;
      suggestions: { original: string; rewrite: string; reason: string }[];
      summary: string;
    };
    note: string | null;
  } | null;
  drills: { weaknesses: Weakness[]; plan: (Drill & { weakness: string })[] } | null;
  drill_progress: Record<string, boolean>;
  session: SessionRow;
}

export interface ProgressPoint {
  session_id: string;
  title: string;
  created_at: string;
  confidence: Num;
  authoritativeness: Num;
  engagement: Num;
  congruence: Num;
  persona: string | null;
  wpm: Num;
  filler_rate_per_min: Num;
  eye_contact_ratio: Num;
  gaze_coverage: Num;
  bpm_rise: Num;
}

export interface PersonaMap {
  kind: string;
  axes: [string, string];
  regions: { name: string; description: string; x: number; y: number }[];
  points: { session_id: string; title: string; created_at: string; x: number; y: number; persona: { name: string; description: string } }[];
  current: { name: string; description: string } | null;
}
