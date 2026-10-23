"""Re-make a finished voice run's looks (keyframe + pose-matched clip) from its final lead takes,
e.g. after a Face Match fix, and point the plan at the new takes. Writes the job graphs to
<out_dir> (keyframes first, then the clips, then a final take-QC pass that assembles the cut);
queue them in that order. Plain Python (json only), so it runs anywhere that sees the run folder.

    python redo_looks.py <run_dir> <out_dir> [tag]
"""
from __future__ import annotations

import glob
import json
import os
import sys


def main(run_dir: str, out_dir: str, tag: str = "fm") -> list[str]:
    pj = os.path.join(run_dir, "plan.json")
    p = json.load(open(pj, encoding="utf-8"))
    by_id = {r["id"]: r for r in p["renders"]}
    rel = p.get("run_rel", "")
    srv_dir = p["run_dir"]  # the server's own path to the run
    sep = "\\" if "\\" in srv_dir else "/"

    def job(name: str) -> dict | None:
        hits = sorted(glob.glob(os.path.join(run_dir, "jobs", f"*_{name}.json")))
        hits = [h for h in hits if not os.path.basename(h).startswith("qc")] or hits
        return json.load(open(hits[0], encoding="utf-8")) if hits else None

    def swap(g: dict, old: str, new: str) -> dict:
        return json.loads(json.dumps(g).replace(json.dumps(old)[1:-1], json.dumps(new)[1:-1]))

    def retag(g: dict, cls: str, name: str, new: str) -> dict:
        for node in g.values():
            ins = node.get("inputs", {})
            if node.get("class_type") == cls and str(ins.get("filename_prefix", "")).endswith(name):
                ins["filename_prefix"] = f"{rel}/{new}"
        return g

    def stem(path: str) -> str:  # .../k012_t1_00001_.png -> k012_t1
        b = os.path.basename(path.replace("\\", "/"))
        return b.rsplit("_00001_", 1)[0]

    def first_image(g: dict) -> str | None:
        for node in g.values():
            if node.get("class_type") == "FVLoadImagePath":
                return node["inputs"].get("path")
        return None

    os.makedirs(out_dir, exist_ok=True)
    keys, clips, done = [], [], []
    for r in p["renders"]:
        if r.get("role") != "B" or r.get("twin_of") not in by_id:
            continue
        lead = by_id[r["twin_of"]]
        kname, rname = f"k{r['id']:03d}", f"r{r['id']:03d}B"
        # the jobs that made the current keyframe and take (a QC re-roll's, if any: its seeds)
        kg = job(stem(r.get("key") or kname)) or job(kname)
        rg = job(stem(r["path"])) or job(rname)
        if kg is None or rg is None:
            continue
        # follow the lead's final take, whichever take the saved jobs were made from
        for t in lead.get("takes", []):
            if t.get("path") and t["path"] != lead["path"]:
                kg, rg = swap(kg, t["path"], lead["path"]), swap(rg, t["path"], lead["path"])
        new_key = f"{srv_dir}{sep}{kname}_{tag}_00001_.png"
        new_clip = f"{srv_dir}{sep}{rname}_{tag}_00001_.mp4"
        kg = retag(kg, "SaveImage", kname, f"{kname}_{tag}")
        for node in kg.values():  # the save prefix may carry a take tag (k012_t1)
            ins = node.get("inputs", {})
            if node.get("class_type") == "SaveImage":
                ins["filename_prefix"] = f"{rel}/{kname}_{tag}"
        old_key = first_image(rg)
        if old_key:
            rg = swap(rg, old_key, new_key)
        for node in rg.values():
            ins = node.get("inputs", {})
            if node.get("class_type") == "SaveVideo":
                ins["filename_prefix"] = f"{rel}/{rname}_{tag}"
        keys.append((kname, kg))
        clips.append((rname, rg))
        r.update(key=new_key, path=new_clip, takes=[])
        r.pop("unusable", None)
        done.append(r["id"])
    files = []
    for i, (name, g) in enumerate(keys + clips):
        f = os.path.join(out_dir, f"{i:03d}_{name}_{tag}.json")
        json.dump(g, open(f, "w", encoding="utf-8"), indent=1)
        files.append(f)
    passes = int(p.get("qc_retries", 2) or 2)
    qc = {"1": {"class_type": "FVQC", "inputs": {"run_dir": srv_dir, "pass_index": passes, "max_passes": passes,
                                                 "then_assemble": True}}}
    f = os.path.join(out_dir, f"{len(files):03d}_zz_qc_final.json")
    json.dump(qc, open(f, "w", encoding="utf-8"), indent=1)
    files.append(f)
    p["looks_redone"] = {"tag": tag, "ids": done}
    json.dump(p, open(pj, "w", encoding="utf-8"), indent=1)
    print(f"{len(done)} looks re-made: {len(keys)} keyframes, {len(clips)} clips -> {out_dir}")
    return files


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2] if len(a) > 2 else "fm")
