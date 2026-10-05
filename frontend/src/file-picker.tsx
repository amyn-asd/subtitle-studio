import {useEffect, useRef, useState} from 'react';
import {ArrowUp, Check, FileVideo, Folder, FolderOpen, LoaderCircle, Search, X} from 'lucide-react';
import {api} from './api';

type Location = {name:string;path:string};
type Entry = {name:string;path:string;kind:'file'|'folder';size:number|null};
type Listing = {path:string;parent:string|null;locations:Location[];entries:Entry[];total:number;skipped:number};

export function FilePicker({kind='file',initialDirectory='',close,onChoose}:{kind?:'file'|'folder';initialDirectory?:string;close:()=>void;onChoose:(path:string)=>void|Promise<void>}) {
  const [directory,setDirectory]=useState(initialDirectory),[draft,setDraft]=useState(initialDirectory),[query,setQuery]=useState(''),[offset,setOffset]=useState(0);
  const [data,setData]=useState<Listing>(),[shortcuts,setShortcuts]=useState<Location[]>([]),[selected,setSelected]=useState(''),[error,setError]=useState(''),[loading,setLoading]=useState(true),[choosing,setChoosing]=useState(false);
  const [refresh,setRefresh]=useState(0);
  const modal=useRef<HTMLElement>(null),pathInput=useRef<HTMLInputElement>(null);
  useEffect(()=>{
    const previous=document.activeElement as HTMLElement|null;
    pathInput.current?.focus();
    const key=(e:KeyboardEvent)=>{
      if(e.key==='Escape'){e.preventDefault();e.stopPropagation();close();}
      if(e.key==='Tab'){
        const elements=Array.from(modal.current?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),select:not(:disabled),a[href]')||[]);
        const first=elements[0],last=elements.at(-1);
        if(e.shiftKey&&document.activeElement===first){e.preventDefault();last?.focus();}
        else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first?.focus();}
      }
    };
    document.addEventListener('keydown',key,true);
    return()=>{document.removeEventListener('keydown',key,true);previous?.focus();};
  },[]);
  useEffect(()=>{
    let cancelled=false;
    setLoading(true);setError('');setData(undefined);
    const timer=window.setTimeout(()=>{
      const params=new URLSearchParams({kind,query,offset:String(offset)});
      if(directory)params.set('path',directory);
      api<Listing>('/files?'+params).then(value=>{
        if(!cancelled){setData(value);setDraft(value.path);setShortcuts(value.locations);}
      }).catch(e=>{if(!cancelled)setError(e.message);}).finally(()=>{if(!cancelled)setLoading(false);});
    },query?180:0);
    return()=>{cancelled=true;clearTimeout(timer);};
  },[directory,kind,query,offset,refresh]);
  function navigate(path:string){setDirectory(path);setDraft(path);setQuery('');setOffset(0);setSelected('');setRefresh(value=>value+1);}
  async function choose(path:string){setChoosing(true);try{await onChoose(path);}catch(e){setError(e instanceof Error?e.message:String(e));setChoosing(false);}}
  const title=kind==='file'?'Choose a video':'Choose a folder';
  return <div className="modal-backdrop file-picker-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget&&!choosing)close();}}>
    <section ref={modal} className="settings-modal file-picker" role="dialog" aria-modal="true" aria-labelledby="file-picker-title">
      <div className="modal-heading"><div><div className="eyebrow">ON YOUR COMPUTER</div><h2 id="file-picker-title">{title}</h2></div><button className="icon-button" aria-label="Close file chooser" disabled={choosing} onClick={close}><X/></button></div>
      <p>{kind==='file'?'Browse your folders. Select a video or audio file, then open it.':'Open a folder, then choose it as the destination.'}</p>
      <div className="picker-locations">{shortcuts.map(location=><button className="button quiet small" key={location.path} disabled={choosing} onClick={()=>navigate(location.path)}><FolderOpen size={14}/>{location.name}</button>)}</div>
      <form className="picker-path" onSubmit={e=>{e.preventDefault();navigate(draft);}}><button type="button" className="icon-button" aria-label="Parent folder" disabled={!data?.parent||choosing} onClick={()=>{if(data?.parent)navigate(data.parent);}}><ArrowUp size={17}/></button><input ref={pathInput} aria-label="Folder path" value={draft} disabled={choosing} onChange={e=>setDraft(e.target.value)}/><button className="button secondary small" type="submit" disabled={choosing}>Go</button></form>
      <label className="picker-search"><Search size={16}/><input aria-label="Search files in this folder" placeholder="Search this folder…" value={query} disabled={choosing} onChange={e=>{setQuery(e.target.value);setOffset(0);setSelected('');}}/></label>
      {error&&<div className="alert error" role="alert">{error}</div>}
      <div className="picker-entries" aria-busy={loading}>
        {loading?<div className="empty"><LoaderCircle className="spin" size={23}/><p>Reading folder…</p></div>:data?.entries.length?data.entries.map(entry=><button type="button" key={entry.path} className={`picker-entry ${selected===entry.path?'selected':''}`} disabled={choosing} onClick={()=>entry.kind==='folder'?navigate(entry.path):setSelected(entry.path)} onDoubleClick={()=>{if(entry.kind==='file')choose(entry.path);}} onKeyDown={e=>{if(e.key==='Enter'&&entry.kind==='file'){e.preventDefault();choose(entry.path);}}}>
          {entry.kind==='folder'?<Folder size={20}/>:<FileVideo size={20}/>}<span>{entry.name}<small>{entry.kind==='folder'?'Folder':entry.size!==null?`${(entry.size/1048576).toFixed(entry.size<1048576?2:1)} MB`:''}</small></span>{selected===entry.path&&<Check size={17}/>}
        </button>):!error&&<div className="empty"><FolderOpen size={24}/><p>{query?'No matching files or folders.':kind==='folder'?'There are no subfolders here. You can select this folder.':'There are no supported media files or subfolders here.'}</p></div>}
      </div>
      {data&&data.total>200&&<div className="picker-pagination"><button className="button quiet small" disabled={!offset||loading} onClick={()=>{setOffset(Math.max(0,offset-200));setSelected('');}}>Previous</button><span>{offset+1}–{Math.min(offset+200,data.total)} of {data.total}</span><button className="button quiet small" disabled={offset+200>=data.total||loading} onClick={()=>{setOffset(offset+200);setSelected('');}}>Next</button></div>}
      {data?.skipped? <p className="muted small">Some entries could not be read.</p>:null}
      <div className="picker-footer"><span>{kind==='file'?'Video and audio stay local.':data?.path||'Choose a destination.'}</span><button className="button secondary" disabled={choosing} onClick={close}>Cancel</button><button className="button primary" disabled={loading||choosing||!!error||(kind==='file'?!selected:!data)} onClick={()=>choose(kind==='folder'?data!.path:selected)}>{choosing?<LoaderCircle className="spin" size={16}/>:<FolderOpen size={16}/>} {kind==='file'?'Open video':'Select folder'}</button></div>
    </section>
  </div>;
}
