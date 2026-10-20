# Frisson Vision

**One performance, many performers.** Drop in a song and Frisson Vision makes a music video
where a different person sings every line. It cuts on the beat, flashes pose-matched "twins"
in on the hits, drives light and colour from the drums, and finishes the picture like film.
Drop in a dance clip instead, and a new dancer takes over the same choreography on every
beat.

Everything runs locally in ComfyUI on open-weights models:
- **Krea 2** casts the faces.
- **LTX** performs them, lip-syncing to the isolated vocal and following a dance's pose.
- **FLUX.2 Klein** dresses the twins.

The cut, the effects and the finish come from the song itself.

> Status: work in progress for the Comfy "Open Source Higgsfield" challenge.

## What it does

| Mode | Input | What you get |
|---|---|---|
| **Voice** | a song (optionally one photo of the artist) | A crowd of strangers, people from every era, or one artist in a new look every shot. Each sings their line straight to camera. Loud sections flicker between faces on the beat; quiet sections breathe on clean bar cuts. |
| **Motion** | a song and a dance clip | One choreography, a new character on the beat, faster as the section builds. Every cut is pose-aligned, so it reads as a transformation, not a jump. |

The Director does the whole run from one press of Queue:

1. **Listens.**
   - Beats, bars, sections, loudness and the drop (`fv/beat.py`).
   - Vocal and drum stems with Hybrid Demucs (`fv/stems.py`).
2. **Plans the cut** in the chosen style pack (`fv/plan.py`):
   - shots on bar lines;
   - which shots get pose-matched twins;
   - where the flash inserts land;
   - optional medium and wide shots in quiet stretches.
3. **Casts** every performer from the pack's casting grammar (`fv/casting.py`). It writes the portrait prompt and the performance prompt for each.
4. **Queues the renders, grouped by model** so each model loads once:
   1. Krea 2 portraits (or Qwen-Image identity edits of the artist photo);
   2. LTX lip-sync clips;
   3. twin keyframes;
   4. pose-guided LTX twins;
   5. take QC;
   6. the final cut.
5. **Checks every take** (`fv/qc.py`). A take that is missing, frozen, blown out, or has a dead mouth is re-rendered with a new seed before the cut. A re-rolled lead clip also re-renders its twins.
6. **Cuts it together** (`fv/assemble.py`):
   - **The edit.** Every shot plays its lead performer. On most beats (and many off-beats) a 3–5 frame insert of another face flashes in: the shot's pose-matched twin, or another performer. Then it ghosts back.
   - **Match cuts.** Fast cutting only works when the eye has somewhere to rest. The assembler tracks every face (`fv/faces.py`) and only flashes in a face that sits where the lead's face is, at about the same size. It then nudges and scales the insert so the two faces line up exactly. An insert with no good match is dropped, so the loud parts read as one face changing, not as chaos. Motion runs get the same effect for free: each segment is rendered twice from the same pose guide, so every flicker lands on the same move.
   - **Lip sync.** LTX's mouths run slightly ahead of its audio. Measured against the vocal stem, the lead is about 0.24 s on LTX-2.3 and 0.12 s on LTX-2.5. Each preset shows its clips that much later, minus the lead real singers' lips naturally have (`sync_delay` in `fv/formats.py`).
   - **Effects.** Exposure flashes, bloom, zoom punches, shake, RGB split and motion trails, driven by the kick, snare and hat envelopes and the drop (`fv/audiofx.py`, `fv/fx.py`).
   - **The finish** (`fv/finish.py`): a grade learned from a reference edit, halation, glow, contrast, a gate vignette and grain.
   - **Output.** It scales to the delivery size, muxes the song, and writes an EDL so the cut opens in Resolve or Premiere.

## Style packs

A style pack is one JSON file in `fv/packs/` holding the casting grammar, the edit grammar, the effects map and the finish. Copy one, change it, and the tool takes on a new style.

| Pack | Mode | Look |
|---|---|---|
| `crowd` | voice | Extreme close-ups on a wide lens, flash and practical light against saturated walls; warm grade, halation, grain. A new face every shot. |
| `eras` | voice | Invented people from every era and corner of the world, each in their own setting. |
| `artist` | voice | **One artist from one photo.** Every shot puts the same person in a new outfit, light and place; the flash inserts are quick costume changes. |
| `spotlight` | motion | One hazy stage under a single spotlight. The dancer's look flickers on the beat between two characters doing the same move, faster as the section builds, with motion trails. |
| `wardrobe` | motion | A new character and a new world on every switch, with same-move flickers in the loud parts. |

The `crowd` and `spotlight` finishes were measured from reference edits (the Lab mean and standard deviation of their frames). `Frisson Vision: Finish` applies any pack's finish to any frames.

**Framing** (voice):
- `close`: close-ups only, the house style.
- `mixed`: medium and wide shots in the quiet stretches, punching back in close for the loud ones.

## Formats and presets

| Format | Quality render | Fast render | Delivered |
|---|---|---|---|
| 9:16 (default) | 768×1280 | 512×896 | 1080×1920 |
| 1:1 | 1024×1024 | 768×768 | 1080×1080 |
| 16:9 | 1280×768 | 896×512 | 1920×1080 |

Render sizes are a touch off the exact aspect: the stage-1 size of the multi-stage pipeline must be a multiple of 64 for the pose guide. The cut is centre-cropped to the delivery aspect.

**Presets**
- **quality** (default): LTX-2.3 22B dev with the distilled, detailer, crisp and soft LoRAs. Three stages (¼ → ½ → full), 50 fps, an audio-to-video attention boost.
- **fast**: LTX-2.5 22B distilled, two stages, 25 fps. About 15 s per clip on an RTX PRO 6000.

## Nodes

| Node | Purpose |
|---|---|
| **Frisson Vision: Director** | Takes a song, plus an optional artist photo or dance clip. Plans, casts and queues the whole run. |
| **Frisson Vision: Take QC** | Checks every take and re-rolls bad ones, then queues the cut (queued by the Director). |
| **Frisson Vision: Assemble** | Cuts a run together. Queue it again to re-cut with different FX intensity; nothing re-renders. |
| **Frisson Vision: Finish** | A pack's finish on any frames. |
| **Frisson Vision: Weight Backups to RAM** | Keeps LoRA weight backups in system RAM (see Performance). |
| Load Image / Video / Audio (path), Free Memory | Helpers the Director's jobs use. |

## Install

1. Clone into `ComfyUI/custom_nodes/` and restart ComfyUI.
2. Required node packs:
   - ComfyUI-LTXVideo;
   - comfyui_controlnet_aux (DWPose);
   - RES4LYF (samplers for the Krea 2 redetail and the quality preset);
   - ComfyUI-KJNodes (Sage attention and the LTX attention tuner).
3. Models. Edit `fv/models.json` to match your folders.
   - **Krea 2:** Turbo, RAW, the Turbo LoRA, and the Qwen3-VL 4B encoder.
   - **Quality (LTX-2.3):** 22B dev, its LoRAs, the spatial upscaler and the Gemma 3 encoder.
   - **Fast (LTX-2.5):** 22B distilled, its VAEs, the Gemma 4 encoder and the upscaler.
   - **LTX union-control IC-LoRA.**
   - **FLUX.2 Klein 9B** for twin and character keyframes.
   - **Qwen-Image 2.1** for the artist pack.
4. `pip install -r requirements.txt`. This installs imageio-ffmpeg for encoding and OpenCV for face tracking. Match cuts use OpenCV 4's Haar cascades, which OpenCV 5 no longer ships. With OpenCV 5 the cut still works, but the flashes aren't face-aligned.

## Use it

**In ComfyUI.** Open one of the example workflows (Workflow → Browse Templates → Frisson Vision, or `example_workflows/`):
- `FrissonVision_Voice.json`: a song in, the crowd or artist cut out. For the artist pack, un-bypass the Load Image node.
- `FrissonVision_Motion.json`: a song and a dance clip in.
- `FrissonVision_Recut.json`: re-cut a finished run with different FX intensity.

Press Queue once. The Director queues every job, and the finished video lands in `output/frissonvision/runs/<run>/final_<format>.mp4`. Next to it you'll find:
- the plan;
- the QC report;
- the EDL;
- every portrait, clip and job.

To re-roll one shot, re-queue its file from `<run>/jobs/`, then re-cut.

**With the Comfy SDK** (`sdk_run.py`), the script drives the whole run through the Comfy API v2:
1. The Director plans.
2. The SDK submits every render.
3. Take QC returns re-rolls, which the SDK submits.
4. Assemble runs and the SDK downloads the MP4.

Point it at your own ComfyUI through `comfy-api-proxy`, or at a Comfy API deployment:

```
pip install comfy-sdk comfy-api-proxy
comfy-api-proxy --max-upload-mb 2000                  # beside your ComfyUI
export COMFY_BASE_URL=http://127.0.0.1:8189
python sdk_run.py --song track.mp3 --pack crowd --format 9:16 --seconds 30
python sdk_run.py --song track.mp3 --pack artist --artist me.png --framing mixed
python sdk_run.py --song track.mp3 --mode motion --dance dance.mp4 --pack spotlight --format 1:1
```

## Performance notes

- A 30 s voice run is about 20–40 jobs. On an RTX PRO 6000:
  - quality: about 1 minute per clip, about 45 minutes for a 30 s run;
  - fast: about 15 s per clip.
  
  Model loads dominate a cold start: about 4 minutes per model from a hard disk.
- **Re-cuts are free.** Face tracks are cached in `<run>/faces.json`, and Assemble re-derives the edit from the pack each time. You can change a pack's edit settings (`match_threshold`, `insert_on_beat`, and so on) and re-queue Recut without re-rendering.
- **`--highvram` and LoRAs.** In high-VRAM mode ComfyUI keeps the original weights behind every LoRA patch on the GPU. With the quality preset's four LoRAs on a 22B model, that doubles its footprint. On Windows the overflow spills into shared memory, and sampling gets many times slower. The Director inserts **Weight Backups to RAM** after the LTX loaders to prevent this.

## Licence

MIT (see `LICENSE`). Models are under their own licences. Only publish edits made with music and footage you have the rights to, and only cast invented people or people who have agreed to it.
