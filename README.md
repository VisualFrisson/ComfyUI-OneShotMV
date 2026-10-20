# Frisson Vision

**One performance, many performers.** Drop in a song and Frisson Vision makes a music video
in which a different person sings every line, cut on the beat, with pose-matched "twins"
that flash in on the snare, audio-reactive light and colour, and a filmic finish. Drop in a
dance clip instead and every beat brings a new dancer doing the same move.

Everything runs locally in ComfyUI on open-weights models: **Krea 2** casts the faces,
**LTX** performs them (lip sync from the isolated vocal), and the cut, effects and finish are
built from the song itself.

> Status: work in progress for the Comfy "Open Source Higgsfield" challenge.

## What it does

| Mode | Input | What you get |
|---|---|---|
| **Voice** | a song | A crowd of strangers (or people from every era) each sing a line straight to camera. Loud sections cut on the bar with twin flash inserts every two beats; quiet sections breathe on clean bar cuts. |
| **Motion** | a song + a dance clip | One choreography, a new character on every beat. Every cut is pose-aligned, so it reads as a transformation, not a jump. |

The Director does the whole run from one queue press:

1. **Listens.** Beats, bars, sections, energy and the drop (`fv/beat.py`); vocal and drum
   stems with Hybrid Demucs (`fv/stems.py`).
2. **Plans the cut** in the chosen style pack (`fv/plan.py`): shot lengths per section,
   which shots get twins, where inserts land, which beats ping-pong between performers.
3. **Casts** every performer from the pack's casting grammar (`fv/casting.py`) and writes
   the Krea 2 portrait prompt and the LTX performance prompt for each.
4. **Queues the renders, grouped by model** so each model loads once:
   Krea 2 portraits → LTX lip-sync renders → twin keyframes → pose-guided LTX twins →
   assembly. Every job is also saved to the run folder, so a single shot can be re-rolled.
5. **Assembles** (`fv/assemble.py`): cuts on the plan, flashes the twins in, drives
   exposure, zoom punches, shake, RGB split and echo trails from the kick / snare / hat
   envelopes and vocal energy (`fv/audiofx.py`, `fv/fx.py`), applies the finish
   (`fv/finish.py`), scales to the delivery size and muxes the song. It also writes an EDL
   so the cut opens in Resolve or Premiere.

## Style packs

A style pack is one JSON file (`fv/packs/`): the casting grammar, the edit grammar, the
effects map and the finish. Copy one, change it, and the whole tool takes on a new style.

| Pack | Mode | Look |
|---|---|---|
| `crowd` | voice | Extreme close-ups, wide lens, flash and practical light against saturated walls; warm grade, halation, grain, rounded gate. |
| `eras` | voice | Invented people from every era and corner of the world sing in their own setting; bloom and an oval gate. |
| `wardrobe` | motion | Bold outfits, seamless studio colour, hard light; a new dancer every two beats. |

The finish of `crowd` was learned from a reference edit (Lab mean/std of its frames), and
`FV Finish` applies any pack's finish to any frames.

## Formats and presets

| Format | Quality render | Fast render | Delivered |
|---|---|---|---|
| 9:16 (default) | 640×1152 | 448×768 | 1080×1920 |
| 1:1 | 1152×1152 | 768×768 | 1080×1080 |
| 16:9 | 1152×640 | 768×448 | 1920×1080 |

* **quality** (default): LTX-2.3 22B dev + distilled LoRA, three stages (¼ → ½ → full),
  50 fps, audio-to-video attention boost.
* **fast**: LTX-2.5 22B distilled, two stages, 25 fps.

## Nodes

| Node | Purpose |
|---|---|
| **Frisson Vision: Director** | Song (+ dance clip) in; plans, casts and queues the whole run. |
| **Frisson Vision: Assemble** | Cuts a run together (queued automatically; queue again to re-cut). |
| **Frisson Vision: Finish** | The pack's finish on any frames. |
| Load Image / Video / Audio (path) | Path loaders that don't check the file at queue time. |

## Install

1. Clone into `ComfyUI/custom_nodes/` and restart ComfyUI.
2. Required node packs: ComfyUI-LTXVideo, comfyui_controlnet_aux (DWPose), RES4LYF
   (ClownsharKSampler for the Krea 2 redetail), ComfyUI-KJNodes (optional patches).
3. Models (edit `fv/models.json` to match your folders): Krea 2 Turbo + RAW + Turbo LoRA +
   Qwen3-VL 4B encoder; LTX-2.3 22B dev + distilled LoRA + spatial upscaler (quality);
   LTX-2.5 22B distilled + VAEs + Gemma 4 encoder + upscaler (fast); LTX union-control
   IC-LoRA; FLUX.2 Klein 9B (motion keyframes); Qwen-Image 2.1 (twin keyframes).
4. `pip install -r requirements.txt` (imageio-ffmpeg for encoding).

## Use it

1. Add **Load Audio** → **Frisson Vision: Director**. Pick mode, pack, format, preset and
   length (0 = whole song), then queue.
2. The Director queues everything. The finished video lands in
   `output/frissonvision/runs/<run>/final_<format>.mp4`, next to the plan, the EDL, every
   portrait, render and job.
3. Don't like a shot? Re-queue its job file from `<run>/jobs/`, then re-queue the
   `zz_assemble` job.

## Licence

MIT (see `LICENSE`). Models are under their own licences. Only publish edits made with
music and footage you have the rights to.
