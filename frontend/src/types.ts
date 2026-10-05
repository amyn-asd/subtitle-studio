export type Track = {stream_index: number; audio_ordinal: number; channels: number; codec: string; title: string; detection: string; languages: {code: string; score?: number}[]};
export type EmbeddedTrack = {stream_index:number;subtitle_ordinal:number;cue_track:number;codec:string;kind:'text'|'image'|'unknown';title:string;language:string;metadata_language:string;extract_extension:string;default:boolean;forced:boolean};
export type Job = {id: string; kind: string; status: string; stage: string; progress: number; message: string; elapsed: number; error?: string; warnings: string[]; tracks?:number[]; target_language?:string; settings?:{preset?:'fast'|'accurate';enhance_audio?:boolean}; result?:{path?:string;subtitle_count?:number;subtitles?:{track:number;path:string;cue_count?:number}[]}};
export type Project = {id: string; media: {name: string; path: string; duration: number; size: number; warnings: string[]; audio_tracks: Track[]; subtitle_tracks?:EmbeddedTrack[];auxiliary_tracks?:{type:string}[]}; jobs: Job[]; cue_counts: {track: number; count: number}[]};
export type ReviewSource = {cue_track:number;label:string;source_kind:'audio'|'embedded';codec:string};
export type Cue = {id: string; track: number; start: number; end: number; text: string; raw_text: string; language: string; flags: string[]; reviewed: boolean; verification?: string; translated_text?: string;};
export type System = {version: string; gpu?: {name: string; total_mb: number; free_mb: number}; tools: Record<string, string | null>; models: {id: string; name: string; ready: boolean; size_gb: number}[]; models_root: string; ollama_models: string; setup_jobs: Job[]; preferences: Record<string, unknown>};
export type Clip = {url: string; start: number; duration: number; peaks: number[]};
export function reviewSources(project:Project):ReviewSource[] {
  return [...project.media.audio_tracks.map(t=>({cue_track:t.stream_index,label:t.title||`Audio ${t.audio_ordinal+1}`,source_kind:'audio' as const,codec:t.codec})),
    ...(project.media.subtitle_tracks||[]).filter(t=>project.cue_counts.some(c=>c.track===t.cue_track)).map(t=>({cue_track:t.cue_track,label:`Embedded ${t.subtitle_ordinal+1} · ${t.title||t.metadata_language} · ${t.codec.toUpperCase()}`,source_kind:'embedded' as const,codec:t.codec}))];
}
