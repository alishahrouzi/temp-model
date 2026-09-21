"""Inspect the Temp Model dataset without modifying source files."""
from __future__ import annotations
import argparse, csv, io, json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any
from PIL import Image, UnidentifiedImageError
DEFAULT_DATASET_ID = "sidd707/jewelry-design-dataset"
DEFAULT_OUTPUT_DIR = Path("results/dataset-inspection")
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
IMAGE_EXTENSIONS = {".jpg",".jpeg",".png",".gif",".bmp",".webp",".tif",".tiff"}
IGNORED_DIRECTORY_NAMES = {".cache",".git","__pycache__"}

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Inspect local dataset without modifying it.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    return p.parse_args()

def normalise_relative_path(value: str) -> Path:
    return Path(*PureWindowsPath(value).parts)

def resolve_dataset_root(input_dir: Path) -> Path:
    if not input_dir.exists(): raise FileNotFoundError(f"Dataset path does not exist: {input_dir}")
    if not input_dir.is_dir(): raise NotADirectoryError(f"Dataset path is not a directory: {input_dir}")
    if (input_dir/"dataset_labels.csv").is_file(): return input_dir
    nested=input_dir/"dataset"
    if (nested/"dataset_labels.csv").is_file(): return nested
    raise RuntimeError("Unrecognized dataset structure.")

def read_labels_csv(path: Path) -> tuple[list[dict[str,str]],str]:
    raw=path.read_bytes()
    for enc in CSV_ENCODINGS:
        try: return list(csv.DictReader(io.StringIO(raw.decode(enc)))), enc
        except UnicodeDecodeError: continue
    raise RuntimeError(f"Could not decode {path} with {CSV_ENCODINGS}")

def resolve_csv_paths(root: Path, rows: list[dict[str,str]]):
    referenced={}; missing=[]; duplicate=[]
    for row in rows:
        raw=row["image_path"]; rel=normalise_relative_path(raw)
        if rel in referenced: duplicate.append(raw)
        referenced[rel]=raw
        if not (root/rel).is_file(): missing.append(raw)
    return referenced, missing, duplicate

def discover_image_files(root: Path) -> list[Path]:
    result=[]
    for p in root.rglob("*"):
        if not p.is_file() or any(x in IGNORED_DIRECTORY_NAMES for x in p.parts): continue
        if p.suffix.lower() in IMAGE_EXTENSIONS: result.append(p.relative_to(root))
    return sorted(result,key=lambda p:str(p).lower())

def inspect_image(root: Path, rel: Path) -> dict[str,Any]:
    path=root/rel
    try:
        with Image.open(path) as im:
            info={"path":str(rel),"extension":path.suffix.lower().lstrip("."),"format":im.format or "UNKNOWN","width":im.width,"height":im.height,"mode":im.mode,"frame_count":getattr(im,"n_frames",1),"readable":True,"error":None}
            im.verify()
        return info
    except (UnidentifiedImageError,OSError,ValueError) as exc:
        return {"path":str(rel),"extension":path.suffix.lower().lstrip("."),"format":"UNREADABLE","width":None,"height":None,"mode":None,"frame_count":None,"readable":False,"error":f"{type(exc).__name__}: {exc}"}

def inspect_dataset(dataset_id: str, input_dir: Path, out: Path) -> dict[str,Any]:
    root=resolve_dataset_root(input_dir); labels=root/"dataset_labels.csv"; rows,enc=read_labels_csv(labels)
    if not rows: raise RuntimeError("No data rows found in dataset_labels.csv")
    required={"image_path","description"}
    if required-set(rows[0]): raise RuntimeError(f"Missing required columns: {sorted(required-set(rows[0]))}")
    referenced,missing,duplicate=resolve_csv_paths(root,rows)
    discovered=discover_image_files(root); dset=set(discovered)
    unlisted=sorted(dset-set(referenced),key=lambda p:str(p).lower())
    infos=[inspect_image(root,p) for p in discovered]; readable=[x for x in infos if x["readable"]]; bad=[x for x in infos if not x["readable"]]
    resolutions=Counter((x["width"],x["height"]) for x in readable); widths=[x["width"] for x in readable]; heights=[x["height"] for x in readable]
    report={"dataset_id":dataset_id,"inspected_at_utc":datetime.now(timezone.utc).isoformat(),"dataset_root":str(root.resolve()),"labels_encoding":enc,
      "summary":{"csv_records":len(rows),"discovered_image_files":len(discovered),"resolvable_csv_images":len(rows)-len(missing),"missing_csv_images":len(missing),"duplicate_csv_paths":len(duplicate),"unlisted_images":len(unlisted),"readable_images":len(readable),"unreadable_images":len(bad)},
      "class_distribution":dict(sorted(Counter((p.parts[0] if len(p.parts)>1 else "__root__") for p in discovered).items())),
      "format_distribution":dict(sorted(Counter(x["format"] for x in infos).items())),
      "extension_distribution":dict(sorted(Counter(x["extension"] for x in infos).items())),
      "resolution_summary":{"unique_resolutions":len(resolutions),"min_width":min(widths) if widths else None,"max_width":max(widths) if widths else None,"min_height":min(heights) if heights else None,"max_height":max(heights) if heights else None,"top_resolutions":[{"width":w,"height":h,"count":c} for (w,h),c in resolutions.most_common(20)]},
      "missing_csv_paths":missing,"duplicate_csv_paths":duplicate,"unlisted_image_paths":[str(p) for p in unlisted],"unreadable_images":bad,"image_records":infos}
    out.mkdir(parents=True,exist_ok=True)
    (out/"dataset_inspection.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    md=["# S0.2 — Dataset Inspection Report","","## Summary"]
    for k,v in report["summary"].items(): md.append(f"- {k}: **{v}**")
    md += ["","## Class Distribution"]+[f"- {k}: **{v}**" for k,v in report["class_distribution"].items()]
    md += ["","## Format Distribution"]+[f"- {k}: **{v}**" for k,v in report["format_distribution"].items()]
    md += ["","## Resolution Summary",f"- Unique resolutions: **{report["resolution_summary"]["unique_resolutions"]}**",f"- Width range: **{report["resolution_summary"]["min_width"]}–{report["resolution_summary"]["max_width"]}**",f"- Height range: **{report["resolution_summary"]["min_height"]}–{report["resolution_summary"]["max_height"]}**","", "## Missing CSV References"]+[f"- {x}" for x in missing] or ["- None"]
    md += ["","## Unlisted Image Files"]+([f"- {x}" for x in unlisted] if unlisted else ["- None"])
    md += ["","## Unreadable Images"]+([f"- {x["path"]}: {x["error"]}" for x in bad] if bad else ["- None"])
    md += ["","## Scope","This is a read-only inspection. Source files are not modified."]
    (out/"dataset_inspection.md").write_text("\n".join(md),encoding="utf-8")
    return report

if __name__=="__main__":
    a=parse_args(); r=inspect_dataset(a.dataset_id,a.input_dir,a.output_dir); s=r["summary"]
    print("Dataset inspection completed.")
    for k in ("csv_records","discovered_image_files","resolvable_csv_images","missing_csv_images","unlisted_images","readable_images","unreadable_images"): print(f"  - {k}: {s[k]}")
    print(f"  - Report: {a.output_dir}")