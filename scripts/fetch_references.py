"""Extract a few public FLEURS test recordings, keeping datasets out of the repository."""
import json
from pathlib import Path
import httpx
import pyarrow.parquet as pq

from subtitle_studio.config import DATA


def main():
    root = DATA / "references"
    root.mkdir(parents=True, exist_ok=True)
    references = []
    for config, language in (("en_us","en"),("fa_ir","fa"),("ja_jp","ja")):
        file = root / f"{config}.parquet"
        if not file.exists():
            print(f"Fetching public {language} test recordings",flush=True)
            url = f"https://huggingface.co/datasets/google/fleurs/resolve/refs%2Fconvert%2Fparquet/{config}/test/0000.parquet"
            partial = file.with_suffix(".partial")
            with httpx.stream("GET",url,follow_redirects=True,timeout=300) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_bytes(1048576):
                        output.write(chunk)
            partial.replace(file)
        reader = pq.ParquetFile(file)
        rows = next(reader.iter_batches(batch_size=3)).to_pylist()
        for index, row in enumerate(rows):
            audio = row["audio"]
            destination = root / f"{language}-{index}.wav"
            destination.write_bytes(audio["bytes"])
            references.append({"path":str(destination),"language":language,"reference":row.get("raw_transcription",row["transcription"]),
                               "source":"google/fleurs test","config":config,"row":index})
        print(f"Prepared {len(rows)} {language} clips",flush=True)
    (root / "references.json").write_text(json.dumps(references,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
