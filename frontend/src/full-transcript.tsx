import {useEffect, useState} from 'react';
import {Copy, Download, Globe2, LoaderCircle, Pause, Play, TriangleAlert} from 'lucide-react';
import {api,busy,lang} from './api';
import type {Cue,Job,Project,ReviewSource} from './types';

type Block = {id:string;language:string;start:number;text:string;translation:string|null;parts:{text:string;translation:string|null}[]};
type Transcript = {blocks:Block[];cue_count:number;translated_count:number;review_count:number;target_language:string};
type View = 'original'|'translated'|'parallel';

export function FullTranscript({project,track,cues,languages,target,onTarget,onTranslated,job,onJob,fail,notify}:{project:Project;track:ReviewSource;cues:Cue[];languages:string[];target:string;onTarget:(language:string)=>void;onTranslated:(translated:boolean)=>void;job?:Job;onJob:(job:Job)=>void;fail:(error:unknown)=>void;notify:(message:string)=>void}){
  const [view,setView]=useState<View>('parallel'),[data,setData]=useState<Transcript>(),[error,setError]=useState(''),[loading,setLoading]=useState(true);
  useEffect(()=>{
    let cancelled=false;setLoading(true);setError('');setData(undefined);
    api<Transcript>(`/projects/${project.id}/transcript?track=${track.cue_track}&language=${target}`).then(value=>{if(!cancelled)setData(value);}).catch(e=>{if(!cancelled)setError(e.message);}).finally(()=>{if(!cancelled)setLoading(false);});
    return()=>{cancelled=true;};
  },[project.id,track.cue_track,target,cues]);
  const matchesTarget=data?.target_language===target;
  const complete=matchesTarget&&!!data?.cue_count&&data.translated_count===data.cue_count;
  const canExport=matchesTarget&&!!data?.cue_count&&(view==='original'||complete)&&!loading;
  async function copy(){
    if(!canExport||!data)return;
    const parts=data.blocks.map(block=>view==='original'?block.text:view==='translated'?block.translation:`Original (${lang(block.language)})\n${block.text}\n\n${lang(target)}\n${block.translation}`);
    await navigator.clipboard.writeText(parts.join('\n\n')+'\n');notify('Full transcript copied.');
  }
  function download(){
    if(!canExport)return;
    const link=document.createElement('a');link.href=`/api/projects/${project.id}/transcript/download?track=${track.cue_track}&language=${target}&view=${view}&k=${encodeURIComponent(window.STUDIO_TOKEN)}`;link.download='';link.click();
  }
  return <section className="full-transcript-panel" role="tabpanel" id="full-transcript-panel" aria-labelledby="full-transcript-tab">
    <div className="section-heading"><div><h2>Full transcript</h2><p>All the words in this track, in reading order.</p></div><span className="pill">{data?.cue_count??cues.length} cues</span></div>
    <div className="full-transcript-controls"><label>View<select aria-label="Full transcript view" value={view} onChange={e=>{const value=e.target.value as View;setView(value);onTranslated(value==='translated');}}><option value="original">Original text</option><option value="translated">Translation</option><option value="parallel">Original + translation</option></select></label><label>Translation language<select aria-label="Full transcript translation language" value={target} onChange={e=>onTarget(e.target.value)}>{languages.map(code=><option value={code} key={code}>{lang(code)}</option>)}</select></label><button className="button secondary" disabled={!cues.length||busy(job)} onClick={()=>api<Job>(`/projects/${project.id}/translate`,'POST',{tracks:[track.cue_track],target_language:target}).then(onJob).catch(fail)}><Globe2 size={16}/>{complete?'Translate again':'Translate track'}</button><div className="full-transcript-export"><button className="button quiet small" disabled={!canExport} onClick={()=>copy().catch(fail)}><Copy size={15}/>Copy text</button><button className="button secondary small" disabled={!canExport} onClick={download}><Download size={15}/>Download TXT</button></div></div>
    {error&&<div className="alert error" role="alert">{error}</div>}
    {data&&data.cue_count>0&&view!=='original'&&!complete&&<div className="transcript-note"><Globe2 size={16}/><span>{data.translated_count} of {data.cue_count} cues translated into {lang(target)}. Missing or stale translations are marked below. Translate the track to copy or download this view.</span></div>}
    {!!data?.review_count&&<div className="transcript-note"><TriangleAlert size={16}/><span>{data.review_count} passages still need review. You can listen and correct them in the subtitle editor.</span></div>}
    {job?.kind==='translate'&&<div className="full-translation-status" role="status"><p>{job.status==='done'?'Translation ready.':`${job.message} · ${Math.round(job.progress*100)}%`}</p>{busy(job)?<button className="button quiet small" onClick={()=>api<Job>(`/jobs/${job.id}/pause`,'POST',{}).then(onJob).catch(fail)}><Pause size={14}/>Pause translation</button>:['paused','failed','cancelled'].includes(job.status)?<button className="button secondary small" onClick={()=>api<Job>(`/jobs/${job.id}/resume`,'POST',{}).then(onJob).catch(fail)}><Play size={14}/>Resume translation</button>:null}{job.error&&<div className="alert error" role="alert">{job.error}</div>}{job.warnings.map(w=><div className="alert warning" key={w}>{w}</div>)}</div>}
    {loading?<div className="empty"><LoaderCircle className="spin" size={25}/><p>Loading the complete transcript…</p></div>:!data?.blocks.length?<div className="empty"><p>Transcribe the audio or import a subtitle track to read it here.</p></div>:<div className={`transcript-reading ${view}`}>
      <div className="reading-headings">{view!=='translated'&&<strong>Original text</strong>}{view!=='original'&&<strong>{lang(target)} translation</strong>}</div>
      {data.blocks.map(block=><div className="reading-row" key={block.id}>{view!=='translated'&&<article className="reading-original"><small>{lang(block.language)}</small><p dir="auto">{block.text}</p></article>}{view!=='original'&&<article className="reading-translation"><small>{lang(target)}</small><p dir="auto">{block.parts.map((part,i)=><span key={i}>{i>0?' ':''}{part.translation??<em className="missing-translation">[Translation pending]</em>}</span>)}</p></article>}</div>)}
    </div>}
  </section>;
}
