import hashlib
import json
import shutil
import struct
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from subtitle_studio.app import create_app
from subtitle_studio.embedded import extraction, remux_plan, subtitle_choices
from subtitle_studio.jobs import Jobs
from subtitle_studio.media import probe, run, fingerprint
from subtitle_studio.storage import Store
from subtitle_studio.subtitle_formats import subtitle_description, preview_audio
from subtitle_studio.subtitles import parse_srt


@pytest.fixture(scope="module")
def captioned(tmp_path_factory):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg not installed")
    folder = tmp_path_factory.mktemp("embedded")
    (folder / "words.srt").write_text("1\n00:00:01,500 --> 00:00:03,200\n<i>Hello, world.</i>\n\n2\n00:00:03,500 --> 00:00:04,700\nExact repeated repeated words.\n", encoding="utf-8")
    (folder / "styled.ass").write_text("""[Script Info]
ScriptType: v4.00+
PlayResX: 320
PlayResY: 180
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,24,&H0000FF00,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,8,10,10,10,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.75,0:00:02.00,Default,,0,0,0,,{\\i1}سلام دنیا.{\\i0}
Dialogue: 1,0:00:01.50,0:00:02.50,Default,,0,0,0,,دوباره\\Nدوباره.
""", encoding="utf-8")
    (folder / "words.vtt").write_text("WEBVTT\n\n00:02.800 --> 00:04.500\n日本語の字幕。\n", encoding="utf-8")
    (folder / "chapter.txt").write_text(";FFMETADATA1\ntitle=Caption test\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=2500\ntitle=Opening\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=2500\nEND=5000\ntitle=Second\n", encoding="utf-8")
    (folder / "note.txt").write_text("Attachment must survive unchanged.", encoding="utf-8")
    path = folder / "زبان-日本語.mkv"
    run([ffmpeg,"-nostdin","-v","error","-y","-f","lavfi","-i","color=c=green:s=320x180:r=10:d=5",
         "-f","lavfi","-i","sine=frequency=440:duration=5","-itsoffset","1.2","-f","lavfi","-i","sine=frequency=880:duration=3.8",
         "-i",str(folder/"words.srt"),"-i",str(folder/"styled.ass"),"-i",str(folder/"words.vtt"),"-i",str(folder/"chapter.txt"),
         "-map","0:v","-map","1:a","-map","2:a","-map","3:s","-map","4:s","-map","5:s","-map_metadata","6","-map_chapters","6",
         "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-c:s","copy","-metadata:s:a:0","language=eng","-metadata:s:a:1","language=fas",
         "-metadata:s:s:0","language=eng","-metadata:s:s:0","title=English words","-metadata:s:s:1","language=per","-metadata:s:s:1","title=Persian styled",
         "-metadata:s:s:2","language=jpn","-disposition:s:0","default","-disposition:s:1","forced","-disposition:s:2","0",
         "-attach",str(folder/"note.txt"),"-metadata:s:t:0","mimetype=text/plain","-metadata:s:t:0","filename=note.txt",str(path)])
    return path


@pytest.fixture
def workspace(tmp_path, monkeypatch, captioned):
    import subtitle_studio.jobs as jobs_module
    import subtitle_studio.app as app_module
    root = tmp_path / "data"
    (root / "logs").mkdir(parents=True)
    monkeypatch.setattr(jobs_module, "DATA", root)
    monkeypatch.setattr(app_module, "DATA", root)
    monkeypatch.setattr(app_module, "model_ready", lambda key: False)
    store = Store(tmp_path / "studio.sqlite")
    project = store.create_project(probe(str(captioned)))
    manager = Jobs(store)
    yield store, project, manager, root
    manager.pool.shutdown(wait=True)


def wait_job(store, job):
    until = time.monotonic() + 30
    while time.monotonic() < until:
        saved = store.job(job["id"])
        if saved["status"] in ("done", "failed", "paused", "cancelled"):
            return saved
        time.sleep(.03)
    raise AssertionError("Media job timed out")


def import_tracks(store, project, manager):
    tracks = [t["cue_track"] for t in project["media"]["subtitle_tracks"]]
    job = wait_job(store, manager.create(project["id"], "import_subtitles", tracks=tracks, languages={}))
    assert job["status"] == "done", job.get("error")
    return tracks


def packets(path, selector):
    info = json.loads(run([shutil.which("ffprobe"),"-v","error","-select_streams",selector,"-show_packets","-show_data_hash","sha256","-of","json",str(path)]).stdout)
    return [(p["data_hash"],p.get("pts_time")) for p in info["packets"]]


def test_import_formats_timestamps_styles_unicode_and_audio_isolation(workspace, captioned):
    store, project, manager, root = workspace
    tracks = project["media"]["subtitle_tracks"]
    assert [t["codec"] for t in tracks] == ["subrip","ass","webvtt"]
    assert [t["language"] for t in tracks] == ["en","fa","ja"]
    before = hashlib.sha256(captioned.read_bytes()).hexdigest()
    selected = import_tracks(store, project, manager)
    english = store.cues(project["id"], selected[0])
    assert english[0]["text"] == "Hello, world."
    assert english[1]["text"] == "Exact repeated repeated words."
    assert english[0]["start"] == pytest.approx(1.5, abs=.025)
    assert english[0]["end"] == pytest.approx(3.2, abs=.025)
    persian = store.cues(project["id"], selected[1])
    assert persian[0]["text"] == "سلام دنیا."
    assert persian[1]["text"] == "دوباره\nدوباره."
    assert persian[1]["start"] < persian[0]["end"]  # Real overlaps are not flattened.
    assert all(c["track"] < 0 and not c["words"] for c in store.cues(project["id"]))
    assert not store.cues(project["id"], 1)
    native = (root / "projects" / project["id"] / "embedded" / "subtitle-2.ass").read_text(encoding="utf-8")
    assert "Arial,24" in native and r"{\i1}" in native
    assert hashlib.sha256(captioned.read_bytes()).hexdigest() == before
    assert preview_audio(project["media"], selected[1])["audio_ordinal"] == 1
    assert preview_audio(project["media"], selected[0], 2)["audio_ordinal"] == 1


def test_mkv_translation_mux_keeps_av_packets_chapters_attachment_and_styles(workspace, captioned, tmp_path):
    store, project, manager, root = workspace
    tracks = import_tracks(store, project, manager)
    cues = store.cues(project["id"], tracks[0])
    for cue, text in zip(cues, ["سلام دنیا.", "واژه‌های دقیق تکرار تکرار."]):
        store.save_translation(project["id"], "fa", cue, text)
    output = tmp_path / "new فارسی.mkv"
    choices = [{"track":tracks[1],"language":"original","default":False},{"track":tracks[0],"language":"fa","default":True}]
    job = wait_job(store,manager.create(project["id"],"remux",selections=choices,output_path=str(output),keep_embedded=True))
    assert job["status"] == "done", job.get("error")
    saved = probe(str(output))
    assert len(saved["audio_tracks"]) == 2 and len(saved["subtitle_tracks"]) == 4
    assert len(saved["chapters"]) == 2 and len(saved["auxiliary_tracks"]) == 1
    assert [t["default"] for t in saved["subtitle_tracks"]] == [False,False,False,True]
    assert saved["subtitle_tracks"][2]["forced"]
    for selector in ("v:0","a:0","a:1"):
        old, new = packets(captioned, selector), packets(output, selector)
        assert [p[0] for p in old] == [p[0] for p in new]  # Bit-identical compressed video/audio.
        assert float(old[0][1])-float(new[0][1]) == pytest.approx(0,abs=.025)
    folder = tmp_path / "extracted"
    folder.mkdir()
    args, translated = extraction(saved, saved["subtitle_tracks"][3]["cue_track"], folder, normalized=True)
    run(args)
    text = translated.read_text(encoding="utf-8")
    assert "واژه‌های دقیق تکرار تکرار." in text and "00:00:01,500" in text
    args, styled = extraction(saved, saved["subtitle_tracks"][2]["cue_track"], folder)
    run(args)
    assert r"{\i1}" in styled.read_text(encoding="utf-8")
    assert fingerprint(captioned) == project["media"]["fingerprint"]


def test_mp4_timed_text_and_mkv_conversion_retain_timing(workspace, captioned, tmp_path):
    store, project, manager, root = workspace
    without_attachment = tmp_path / "text-only.mkv"
    run([shutil.which("ffmpeg"),"-nostdin","-v","error","-y","-i",str(captioned),"-map","0","-map","-0:t?","-c","copy",str(without_attachment)])
    project = store.create_project(probe(str(without_attachment)))
    track = project["media"]["subtitle_tracks"][1]["cue_track"]
    output = tmp_path / "timed.mp4"
    job = wait_job(store, manager.create(project["id"],"remux",selections=[{"track":track,"language":"original","default":True}],output_path=str(output),keep_embedded=False))
    assert job["status"] == "done", job.get("error")
    media = probe(str(output))
    assert len(media["audio_tracks"]) == 2 and len(media["subtitle_tracks"]) == 1
    assert media["subtitle_tracks"][0]["codec"] == "mov_text"
    mp4_project = store.create_project(media)
    import_tracks(store, mp4_project, manager)
    cues = store.cues(mp4_project["id"])
    assert cues[0]["text"] == "سلام دنیا." and cues[0]["start"] == pytest.approx(.75,abs=.025)
    # MOV timed text cannot be copied as-is into MKV; convert that track to SRT.
    mkv = tmp_path / "converted.mkv"
    job = wait_job(store, manager.create(mp4_project["id"],"remux",selections=[{"track":cues[0]["track"],"language":"original","default":True}],output_path=str(mkv),keep_embedded=True))
    assert job["status"] == "done", job.get("error")
    assert probe(str(mkv))["subtitle_tracks"][0]["codec"] == "subrip"


def test_protection_stale_translation_and_destination_race(workspace, captioned, tmp_path):
    store, project, manager, root = workspace
    track = import_tracks(store, project, manager)[0]
    selection = [{"track":track,"language":"original","default":True}]
    with pytest.raises(ValueError,match="original video"):
        remux_plan(store,project["id"],selection,str(captioned))
    existing = tmp_path / "existing.mkv"
    existing.write_bytes(b"keep")
    with pytest.raises(ValueError,match="already exists"):
        remux_plan(store,project["id"],selection,str(existing))
    with pytest.raises(ValueError,match="attachments"):
        remux_plan(store,project["id"],selection,str(tmp_path/"no-fonts.mp4"))
    with pytest.raises(ValueError,match="incomplete or stale"):
        remux_plan(store,project["id"],[{"track":track,"language":"de"}],str(tmp_path/"stale.mkv"))
    output = tmp_path / "race.mkv"
    command = manager.media_command
    def create_competing_file(*args, **kwargs):
        command(*args, **kwargs)
        output.write_bytes(b"User file appeared while copying")
    manager.media_command = create_competing_file
    job = wait_job(store,manager.create(project["id"],"remux",selections=selection,output_path=str(output),keep_embedded=True))
    assert job["status"] == "failed"
    assert output.read_bytes() == b"User file appeared while copying"
    assert not list(tmp_path.glob("*.partial.mkv"))


def test_bitmap_classification_raw_extraction_and_container_limits(workspace, tmp_path):
    store, project, manager, root = workspace
    bitmap = subtitle_description({"index":9,"codec_name":"hdmv_pgs_subtitle","tags":{"language":"eng"},"disposition":{"forced":1}},0)
    media = {**project["media"],"subtitle_tracks":[bitmap],"auxiliary_tracks":[]}
    store.update_media(project["id"],media)
    args, path = extraction(media,bitmap["cue_track"],tmp_path)
    assert path.suffix == ".sup" and args[args.index("-c:s")+1] == "copy"
    with pytest.raises(ValueError,match="OCR"):
        extraction(media,bitmap["cue_track"],tmp_path,normalized=True)
    selections = [{"track":bitmap["cue_track"],"language":"original"}]
    assert remux_plan(store,project["id"],selections,str(tmp_path/"bitmap.mkv"))["entries"][0]["native"]
    with pytest.raises(ValueError,match="image subtitle"):
        remux_plan(store,project["id"],selections,str(tmp_path/"bitmap.mp4"))


def test_api_import_without_models_extract_download_preview_and_remux(workspace, tmp_path):
    store, project, manager, root = workspace
    client = TestClient(create_app(store,manager,"test"))
    headers = {"X-Studio-Token":"test"}
    pid = project["id"]
    assert client.post(f"/api/projects/{pid}/subtitles/import",headers=headers,json={"tracks":[1]}).status_code == 400
    track = project["media"]["subtitle_tracks"][0]["cue_track"]
    result = client.post(f"/api/projects/{pid}/subtitles/import",headers=headers,json={"tracks":[track],"languages":{str(track):"en"}})
    assert result.status_code == 200
    assert wait_job(store,result.json())["status"] == "done"
    downloaded = client.get(f"/api/projects/{pid}/subtitles/file?track={track}",headers=headers)
    assert downloaded.status_code == 200 and "Hello, world." in downloaded.text
    options = client.get(f"/api/projects/{pid}/subtitles/options",headers=headers).json()
    assert len(options["subtitles"]) == 3
    output = tmp_path/"API.mkv"
    result = client.post(f"/api/projects/{pid}/subtitles/embed",headers=headers,json={"subtitles":[{"track":track,"default":True}],"output_path":str(output),"keep_embedded":False})
    assert result.status_code == 200
    assert wait_job(store,result.json())["status"] == "done"
    assert len(probe(str(output))["subtitle_tracks"]) == 1
    assert client.post(f"/api/projects/{pid}/export",headers=headers,json={"track":track,"directory":str(tmp_path)}).status_code == 200
    # Explicit secondary audio remains selectable for imported text tracks.
    result = client.post(f"/api/projects/{pid}/preview",headers=headers,json={"track":track,"audio_track":2,"start":1.5,"duration":1})
    assert result.status_code == 200 and result.json()["audio_track"] == 2


def test_srt_parser_preserves_literal_text_entities_and_repeated_words():
    cues = parse_srt("1\n01:02:03,004 --> 01:02:04,500\n<i>repeat repeat &amp; &lt;name&gt;</i>\n", "p", -3, "en", "ass")
    assert cues[0]["text"] == "repeat repeat & <name>" and cues[0]["start"] == 3723.004


def test_cancelled_import_and_remux_clean_partial_files_and_resume(workspace, tmp_path):
    store, project, manager, root = workspace
    track = project["media"]["subtitle_tracks"][0]["cue_track"]
    original_command = manager.media_command
    def interrupted(job, args, *extra):
        Path(args[-1]).write_bytes(b"unfinished")
        manager.controls[job["id"]].set()
        raise InterruptedError("Paused")
    manager.media_command = interrupted
    job = wait_job(store, manager.create(project["id"], "import_subtitles", tracks=[track], languages={}))
    assert job["status"] == "paused" and not store.cues(project["id"])
    assert not list((root/"projects"/project["id"]).rglob("*.partial"))
    manager.media_command = original_command
    job = wait_job(store, manager.resume(job["id"]))
    assert job["status"] == "done"
    cue = store.cues(project["id"],track)[0]
    store.update_cue(project["id"],cue["id"],{"text":"Human correction.","edited":True})
    import_tracks(store, project, manager)
    assert store.cues(project["id"],track)[0]["text"] == "Human correction."
    output = tmp_path/"resumed.mkv"
    manager.media_command = interrupted
    job = wait_job(store, manager.create(project["id"],"remux",selections=[{"track":track,"language":"original"}],output_path=str(output),keep_embedded=True))
    assert job["status"] == "paused" and not output.exists()
    assert not list(tmp_path.glob("*.partial.mkv"))
    manager.media_command = original_command
    job = wait_job(store, manager.resume(job["id"]))
    assert job["status"] == "done" and output.exists()


def test_unknown_subtitle_language_requires_choice_without_loading_ai(workspace):
    store, project, manager, root = workspace
    media = project["media"]
    media["subtitle_tracks"][0]["language"] = "und"
    store.update_media(project["id"],media)
    client = TestClient(create_app(store,manager,"test"))
    headers = {"X-Studio-Token":"test"}
    track = media["subtitle_tracks"][0]["cue_track"]
    path = f"/api/projects/{project['id']}/subtitles/import"
    result = client.post(path,headers=headers,json={"tracks":[track]})
    assert result.status_code == 400 and "language" in result.json()["detail"]
    result = client.post(path,headers=headers,json={"tracks":[track],"languages":{str(track):"en"}})
    assert result.status_code == 200 and wait_job(store,result.json())["status"] == "done"
    assert all(c["language"] == "en" for c in store.cues(project["id"]))


def pgs_bitmap() -> bytes:
    """An original 2x2 white bitmap displayed at 1s, then cleared at 3s. No downloaded fixtures."""
    def segment(kind, payload, second):
        ticks = second*90000
        return b"PG"+struct.pack(">IIBH",ticks,ticks,kind,len(payload))+payload
    pcs = struct.pack(">HHBHBBBB",320,180,0x10,0,0x80,0,0,1)+struct.pack(">HBBHH",0,0,0,10,10)
    wds = b"\x01"+struct.pack(">BHHHH",0,0,0,320,180)
    palette = b"\x00\x00"+bytes([0,16,128,128,0,1,235,128,128,255])
    rle = b"\x01\x01\x00\x00"*2
    image = struct.pack(">HBB",0,0,0xc0)+(4+len(rle)).to_bytes(3,"big")+struct.pack(">HH",2,2)+rle
    clear = struct.pack(">HHBHBBBB",320,180,0x10,1,0,0,0,0)
    return b"".join(segment(k,v,1) for k,v in [(0x16,pcs),(0x17,wds),(0x14,palette),(0x15,image),(0x80,b"")])+segment(0x16,clear,3)+segment(0x80,b"",3)


def test_real_pgs_and_dvd_bitmap_extraction_and_copy_mux(workspace, captioned, tmp_path):
    store, project, manager, root = workspace
    ffmpeg = shutil.which("ffmpeg")
    sup = tmp_path/"image.sup"
    sup.write_bytes(pgs_bitmap())
    dvd = tmp_path/"image.mks"
    # This decode/encode proves that the synthetic PGS contains a valid bitmap, not just a codec label.
    run([ffmpeg,"-nostdin","-v","error","-y","-i",str(sup),"-map","0:s","-c:s","dvdsub","-f","matroska",str(dvd)])
    bitmap_video = tmp_path/"with-images.mkv"
    run([ffmpeg,"-nostdin","-v","error","-y","-i",str(captioned),"-i",str(sup),"-i",str(dvd),
         "-map","0:v","-map","0:a","-map","1:s","-map","2:s","-c","copy",
         "-metadata:s:s:0","language=eng","-disposition:s:0","default+hearing_impaired",str(bitmap_video)])
    project = store.create_project(probe(str(bitmap_video)))
    tracks = project["media"]["subtitle_tracks"]
    assert [t["codec"] for t in tracks] == ["hdmv_pgs_subtitle","dvd_subtitle"]
    assert all(t["kind"] == "image" for t in tracks)
    job = wait_job(store,manager.create(project["id"],"extract_subtitles",tracks=[t["cue_track"] for t in tracks]))
    assert job["status"] == "done", job.get("error")
    results = job["result"]["subtitles"]
    assert Path(results[0]["path"]).suffix == ".sup" and Path(results[0]["path"]).stat().st_size > 100
    assert Path(results[1]["path"]).suffix == ".mks"
    assert not store.cues(project["id"])
    output = tmp_path/"preserved.mkv"
    job = wait_job(store,manager.create(project["id"],"remux",selections=[{"track":tracks[0]["cue_track"],"language":"original","default":True}],output_path=str(output),keep_embedded=True))
    assert job["status"] == "done", job.get("error")
    saved = probe(str(output))
    assert [t["codec"] for t in saved["subtitle_tracks"]] == ["dvd_subtitle","hdmv_pgs_subtitle"]
    assert "hearing_impaired" in saved["subtitle_tracks"][1]["dispositions"]
    assert [p[0] for p in packets(bitmap_video,"s:0")] == [p[0] for p in packets(output,"s:1")]
    assert [p[0] for p in packets(bitmap_video,"s:1")] == [p[0] for p in packets(output,"s:0")]


def test_video_without_audio_can_import_preview_and_embed(workspace, captioned, tmp_path):
    store, project, manager, root = workspace
    source = tmp_path/"silent-captioned.mkv"
    run([shutil.which("ffmpeg"),"-nostdin","-v","error","-y","-i",str(captioned),"-map","0:v","-map","0:s:0","-c","copy",str(source)])
    project = store.create_project(probe(str(source)))
    tracks = import_tracks(store,project,manager)
    assert not project["media"]["audio_tracks"]
    client = TestClient(create_app(store,manager,"test"))
    headers = {"X-Studio-Token":"test"}
    result = client.post(f"/api/projects/{project['id']}/preview",headers=headers,json={"track":tracks[0],"start":1.5,"duration":1})
    assert result.status_code == 200 and result.json()["peaks"] == [] and result.json()["audio_track"] is None
    output = tmp_path/"with-subtitles.mkv"
    job = wait_job(store,manager.create(project["id"],"remux",selections=[{"track":tracks[0],"language":"original","default":True}],output_path=str(output),keep_embedded=False))
    assert job["status"] == "done",job.get("error")
    assert not probe(str(output))["audio_tracks"]
