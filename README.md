# OneShot MV

**One song, one prompt, one shot: a full music video.** Drop in a song and a one-line World
("the gladiator games in ancient Rome"), and OneShot MV casts it, plans the cut from the song's
own energy, and renders a music video where a different invented person sings every line: held
shots in the intro, a cut every bar in the verses, a new face on the beat in the chorus. Drop in a
dance clip instead, and a new dancer takes over the same choreography.

Everything runs locally in ComfyUI on open-weights models:
- **Krea 2** casts the faces.
- **LTX** performs them, lip-syncing to the isolated vocal and following a dance's pose.
- **FLUX.2 Klein** dresses the pose-matched looks.

The cut, the effects and the finish come from the song itself.

![Twelve of the 84 invented people in one full-song crowd run](media/crowd_grid.jpg)

> Status: work in progress for the Comfy "Open Source Higgsfield" challenge. Every face shown here is invented.

## What it does

| Mode | Input | What you get |
|---|---|---|
| **Voice** | a song (optionally one photo of the artist) | A crowd of strangers, people from every era, or one artist in a new look every shot. Each sings their line to camera. Verses hold each shot; the drop cuts on the beat, every cut a match cut. |
| **Motion** | a song and a dance clip | One choreography, a new character every two bars in the quiet parts and every two beats on the drop. Every cut is pose-aligned, so it reads as a transformation, not a jump. A crew dancing behind the lead stays out of it: only the lead dancer's pose guides the render. |

**Match cuts on the drop.** Each cut lands the next face where the last one was:

![Ten frames from a loud passage: each cut keeps the face in place](media/matchcut_strip.jpg)

**One artist, a new look on every line** (from one photo):

![The same invented singer in ten looks across one song](media/artist_strip.jpg)

**Every era, in 16:9:**

![Ten invented people from different eras singing the same song](media/eras_grid.jpg)

**Motion: one dance, many dancers**, flickering on the drop to a second character doing the same move:

![Ten frames from a spotlight motion run](media/motion_strip.jpg)

**A theme sets the world.** Type a line into the Director's `theme` box ("a retro-futurist 1970s space station, orange and teal light") and it decides who the characters are, what they wear, where every shot happens, the light and the colour grade; the pack still decides the edit. The line is expanded once into a small world bible by Gemma 3, the LTX-2.3 text encoder you already have, through ComfyUI's own Generate Text, and cached. **OneShot MV: Theme Preview** shows that bible before you spend a run on it.

The Director does the whole run from one click of Run:

1. **Listens.**
   - Beats, bars, sections, loudness and the drop (`fv/beat.py`).
   - Vocal and drum stems with Hybrid Demucs (`fv/stems.py`).
2. **Plans the cut** in the chosen style pack (`fv/plan.py`):
   - splits the song into quiet stretches, builds and drops (the drop and the choruses);
   - shots on bar lines: two bars in the quiet stretches, one in the build;
   - on the drop, one-bar groups, each a chain of pose-matched looks of the same moment;
   - wide, medium and close shots in the quiet stretches (`mixed` framing).
3. **Casts** every performer from the pack's casting grammar, or from the theme (`fv/casting.py`, `fv/theme.py`). It writes the portrait prompt and the performance prompt for each. Wide shots, and half the medium ones, are in-scene shots: the performer walks toward the camera, leans in a doorway or works at something in their world while singing, instead of standing at the lens.
   - **Lyrics.** Whisper large-v3-turbo transcribes the vocal stem with word timings (`fv/lyrics.py`), for the b-roll below.
4. **Queues the renders in four model stages** (`staged`, the default when the Director queues the run itself), so each model loads as few times as possible:
   1. **Krea 2:** every portrait (or Qwen-Image identity edits of the artist photo), plus stills for a few b-roll candidates;
   2. **LTX:** every lip-sync clip and the b-roll clips. Take QC checks the singer shots and re-renders the bad ones while LTX stays loaded (**Keep Models Loaded** wires LTX's loaders into the QC and Klein jobs, so ComfyUI doesn't drop the model between video jobs);
   3. **Klein:** the keyframes for the pose-matched looks, made from the final singer takes, so no look is ever made twice;
   4. **LTX:** the looks. With `look_groups` (default), each drop group's looks are **one** clip: the lead's pose guide, with each look's keyframe pinned on its beat, so LTX cuts from person to person inside the clip. The assembler finds where each switch actually lands and lines it up with its beat;
   5. take QC for the looks, then the final cut.

   A test 30 s run went from 12 model loads to 4. A three-look drop group renders in about 65 s instead of about 156 s as three clips.
5. **Checks every take** (`fv/qc.py`). A take is re-rendered with a new seed before the cut if it:
   - is missing, frozen or blown out;
   - has a dead mouth;
   - cuts away (or slowly drifts) from the planned shot to another person, place or framing mid-take, which LTX sometimes does;
   - **is out of lip sync**, by SyncNet (`fv/lipsync.py`, see Install). SyncNet also measures how far each take's mouth runs ahead of the voice, and the cut shows that take that much later;
   - **is badly framed**: no face in most of the shot (the head walked out of frame), or a medium or wide shot whose face fills the frame (`fv/identity.py`, SCRFD; needs the buffalo_l models, see Install). A fault that is already in the portrait gets a new portrait;
   - **is no longer the artist** (artist pack): a look whose keyframe face doesn't match the artist photo (ArcFace) gets a new keyframe. The looks are edited with the artist photo as a second reference, so a new outfit, hairstyle and place keep the artist's face.

   The worst takes are re-rendered first, and re-renders stop once they would double the run's render time. In staged runs the looks are made only after their lead is final; otherwise a re-rolled lead clip also re-renders its pose-matched looks, and each look keeps a take made from the lead take that is kept. The best take of each shot goes into the cut.

   **Lip-sync repair (experimental, off by default).** With `lip_repair` on, a sung lead that SyncNet still scores low after the re-rolls goes through LatentSync 1.6 (`fv/lipfix.py`, vendored in `fv/latentsync/`), which re-draws the mouth to the vocal; the repaired take replaces the original only when SyncNet scores it clearly higher. In our tests it smeared the mouth on fast head and camera moves, which LTX shots have a lot of, so it is off unless you turn it on.

   **B-roll.** About one shot in eight (`broll_share`) becomes a cutaway where nobody sings and the picture shows what the words are about (`fv/broll.py`): first where lip sync is still weak, then over instrumental gaps; never on the drop, never twice in a row. The language model turns the line sung there into an image prompt in the theme's (or pack's) world; Krea 2 makes it as a real photograph and LTX moves it. In staged runs, twice the slots are rendered up front (with the portraits and singer clips, so no extra model load), and the best ones are kept once singer QC knows where lip sync is weak. Lyric cutaways have nobody in them; any letterbox bars are cropped in the cut.
6. **Cuts it together** (`fv/assemble.py`):
   - **The edit.** Every shot plays its lead performer, and the cut never goes back to a shot it has left. Quiet stretches hold; the build tightens; the drop steps through each group's chain of looks on the beat.
   - **Match cuts.** Fast cutting only works when the eye has somewhere to rest. Each look in a drop chain is rendered from the lead's own pose and words (DWPose plus the LTX union-control IC-LoRA, keyframed from the lead's frame at that beat), so the next look lands on the same pose and syllable. The assembler tracks every face (`fv/faces.py`), nudges each look so the faces line up exactly, and skips a look whose face doesn't land. Motion runs get the same effect on the drop: each segment is rendered twice from the same pose guide, so every flicker lands on the same move.
   - **Lip sync.** LTX's mouths run slightly ahead of its audio. SyncNet measures each take's offset, and the cut shows that take that much later. Without SyncNet, each preset shows its clips by an average measured offset (`sync_delay` in `fv/formats.py`).
   - **Effects.** Exposure flashes, bloom, zoom punches, shake, RGB split and motion trails, driven by the kick, snare and hat envelopes and the drop (`fv/audiofx.py`, `fv/fx.py`). They stay low in the quiet stretches and rise through the build, so the drop lands at full strength.
   - **The finish** (`fv/finish.py`): a grade learned from a reference edit, halation, glow, contrast, a gate vignette and grain.
   - **Output.** It scales to the delivery size, muxes the song, and writes an EDL so the cut opens in Resolve or Premiere. The cut runs at 25 fps, exactly every other frame of the 50 fps quality clips (at 24 fps the uneven frame steps read as stutter on the mouths).

## Style packs

A style pack is one JSON file in `fv/packs/` holding the casting grammar, the edit grammar, the effects map and the finish. Copy one, change it, and the tool takes on a new style.

| Pack | Mode | Look |
|---|---|---|
| `crowd` | voice | Extreme close-ups on a wide lens, flash and practical light against saturated walls; warm grade, halation, grain. A new face every shot. |
| `eras` | voice | Invented people from every era and corner of the world, each in their own setting. |
| `artist` | voice | **One artist from one photo.** Every shot puts the same person in a new outfit, light and place; on the drop, each beat is a quick costume change on the same pose. |
| `spotlight` | motion | One hazy stage under a single spotlight. The dancer's look flickers on the beat between two characters doing the same move, faster as the section builds, with motion trails. |
| `wardrobe` | motion | A new character and a new world on every switch, with same-move flickers in the loud parts. |

The `crowd` and `spotlight` finishes were measured from reference edits (the Lab mean and standard deviation of their frames). `OneShot MV: Finish` applies any pack's finish to any frames.

**Framing** (voice):
- `mixed` (default): wide, medium and close shots in the quiet stretches, medium and close in the build, close-ups on the drop. In the `artist` pack, the wide and medium shots also move the artist somewhere new, doing something: dancing, walking, turning to the camera.
- `close`: close-ups only.

## Formats and presets

| Format | Quality render | Fast motion render | Delivered |
|---|---|---|---|
| 16:9 (default) | 1280×768 | 896×512 | 1920×1080 |
| 9:16 | 768×1280 | 512×896 | 1080×1920 |
| 1:1 | 1024×1024 | 768×768 | 1080×1080 |

Render sizes are a touch off the exact aspect: the stage-1 size of the multi-stage pipeline must be a multiple of 64 for the pose guide. The cut is centre-cropped to the delivery aspect.

**Presets**
- **quality** (default): LTX-2.3 22B dev with the distilled, detailer, crisp and soft LoRAs. Three stages (¼ → ½ → full), 50 fps, an audio-to-video attention boost.
- **fast**:
  - voice: the quality model at its full size, 25 fps, with fewer first-stage steps. Singing always uses LTX-2.3: its lip sync is the stronger of the two;
  - motion: LTX-2.5 22B distilled, two stages, 25 fps. About 15 s per clip on an RTX PRO 6000.

## Nodes

| Node | Purpose |
|---|---|
| **OneShot MV: Director** | Takes a song, plus an optional artist photo or dance clip. Plans, casts and queues the whole run. |
| **OneShot MV: Take QC** | Checks every take, lip sync included, and re-rolls bad ones, then queues the cut (queued by the Director). |
| **OneShot MV: Lip Sync Score** | Scores a finished run's lip sync with SyncNet, take by take, into `<run>/lipsync.json`. |
| **OneShot MV: Theme Preview** | Shows what a theme turns into: characters, outfits, places, light, in-scene actions and the grade. |
| **OneShot MV: Lip Sync Repair** | Re-draws the mouth of any clip to an audio track with LatentSync 1.6. |
| **OneShot MV: Face Match** | Moves and scales an edited keyframe so its face sits where the reference frame's face is (used for the pose-matched looks). |
| **OneShot MV: Assemble** | Cuts a run together. Run it again to re-cut with different FX intensity; nothing re-renders. |
| **OneShot MV: Run Result** | Says where a run is and, once it's finished, returns the final video as a job output (used by `sdk_run.py` and other API clients). |
| **OneShot MV: Finish** | A pack's finish on any frames. |
| **OneShot MV: Weight Backups to RAM** | Keeps LoRA weight backups in system RAM (see Performance). |
| **OneShot MV: Lead Dancer Pose** | Keeps only the lead dancer in a DWPose guide (queued by the Director for motion runs). |
| **OneShot MV: Lead Dancer Only** | Greys out everyone but the lead dancer in a frame before the keyframe edit. |
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
4. `pip install -r requirements.txt`. This installs imageio-ffmpeg for encoding and OpenCV for face tracking. Match cuts use OpenCV 4's Haar cascades, which OpenCV 5 no longer ships. With OpenCV 5 the cut still works, but the looks aren't face-aligned.
5. **Lip-sync scoring (recommended).** Put the two SyncNet weight files in `ComfyUI/models/syncnet/`:
   - `syncnet_v2.model`;
   - `sfd_face.pth`.

   They come from the [Oxford VGG SyncNet page](https://www.robots.ox.ac.uk/~vgg/software/lipsync/), and are also mirrored in [syncnet-python](https://github.com/nawta/SyncNet_py309_313/tree/main/weights). The weights are for research use (Creative Commons Attribution), so they are not shipped here. The model code is vendored in `fv/syncnet/` (MIT and Apache-2.0, see its NOTICE). Without the weights, take QC skips the lip-sync check.
6. **Lyrics for b-roll (optional).** [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (about 1.6 GB, MIT) in `ComfyUI/models/whisper/whisper-large-v3-turbo/` (the folder comfy-mtb's Whisper loader also uses). Without it, b-roll illustrates the theme or pack world instead of the words.
7. **Lip-sync repair (optional).** In `custom_nodes/ComfyUI-OneShotMV/weights/latentsync/`:
   - `latentsync_unet.pt` and `whisper/tiny.pt` from [ByteDance/LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6);
   - `sd-vae-ft-mse/` (`config.json` and `diffusion_pytorch_model.safetensors`) from [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse).

   It also needs insightface's buffalo_l face models (`det_10g.onnx`, `2d106det.onnx`) in `ComfyUI/models/insightface/models/buffalo_l/`; many face-swap packs install them. Take QC's framing and identity checks use the same folder (`det_10g.onnx`, `w600k_r50.onnx`) and are skipped without it. They run with onnxruntime, so the insightface package itself isn't needed. The buffalo_l models are for non-commercial research use. The LatentSync code is vendored under Apache-2.0 (see `fv/latentsync/NOTICE`); the weights are not shipped.

## Use it

**In ComfyUI.** Open one of the example workflows: **Templates** in the left sidebar, then **Extensions → ComfyUI-OneShotMV** (or drag a file from `example_workflows/` onto the page):
- `OneShotMV_App.json`: the simplest way in. It opens as an **app** (ComfyUI's App mode): a song, a World line and a few settings on the right, **Run**, and the results in the middle. **App / Graph** at the top left shows the node graph behind it.
- `OneShotMV_Voice.json`: the same run as a node graph with every setting. For the artist pack, un-bypass the Load Image node.
- `OneShotMV_Motion.json`: a song and a dance clip in.
- `OneShotMV_Recut.json`: re-cut a finished run with different FX intensity.

Run it once with **Make the video** (`queue_renders`) off to see the plan, then again with it on. The Director queues every job, and the finished video lands in `output/oneshot-mv/runs/<run>/final_<format>.mp4`. Next to it you'll find:
- the plan;
- the QC report;
- the EDL;
- every portrait, clip and job.

To re-roll one shot, queue its file from `<run>/jobs/` again, then re-cut.

**With the Comfy SDK** (`sdk_run.py`): one command takes a song and a World line to a finished MP4 through the Comfy API v2.
1. The SDK uploads the song and runs the Director, then prints the shot plan (the same preview the Run summary shows).
2. The run's stages render in ComfyUI in the staged order: stills, vision check, lip-sync clips, take QC, b-roll, the cut.
3. A **Run Result** job waits behind those stages and hands back the final video, which the SDK downloads.
4. `--recut` re-cuts a finished run at another FX level without re-rendering.

Comfy Cloud doesn't install custom node packs yet, so the SDK talks to your own ComfyUI through Comfy's open-source [`comfy-api-proxy`](https://github.com/Comfy-Org/comfy-api-proxy). The proxy gives a local ComfyUI the same API as Comfy Cloud, and `sdk_run.py` starts it for you if it isn't already running. Start ComfyUI first.

On Windows (ComfyUI portable), double-click **`run_sdk.bat`** in the pack folder. The first time, it installs `comfy-sdk` and `comfy-api-proxy` into `.sdk\` next to it and leaves ComfyUI's own Python packages alone. It then asks for the song, the World line and the length. You can also pass the options yourself:

```
run_sdk.bat --song "D:\music\track.wav" --theme "the gladiator games in ancient Rome" --seconds 40
```

Anywhere else:

```
pip install comfy-sdk comfy-api-proxy
python sdk_run.py --song track.wav --theme "a 1965 airline: pilots, stewardesses and passengers" --preview
python sdk_run.py --song track.wav --theme "a 1965 airline: pilots, stewardesses and passengers" --seconds 30
python sdk_run.py --song track.wav --format 9:16 --fx intense --cuts-high "to the music (intense)"
python sdk_run.py --recut "<run folder>" --fx none
```

`python sdk_run.py --help` lists every option; they match the Director's inputs. To use another deployment that has the OneShot MV nodes installed, set `COMFY_BASE_URL` (and `COMFY_API_KEY` if it needs one).

## Performance notes

- A 30 s voice run is about 40–100 jobs: portraits, lead clips, look keyframes and the short look clips for each drop. On an RTX PRO 6000:
  - quality: about 1 minute per clip, about 45 minutes for a 30 s run;
  - fast: about 15 s per clip.
  
  Model loads dominate a cold start: LTX-2.3 and its Gemma 3 text encoder (about 65 GB) take 5–9 minutes from a SATA drive, about a minute from NVMe. The staged order loads LTX twice per run.
- **Re-cuts are free.** Face tracks are cached in `<run>/faces.json`, and Assemble re-derives the edit from the pack each time. You can change a pack's edit settings (`match_threshold`, `insert_on_beat`, and so on) and re-queue Recut without re-rendering.
- **`--highvram` and LoRAs.** In high-VRAM mode ComfyUI keeps the original weights behind every LoRA patch on the GPU. With the quality preset's four LoRAs on a 22B model, that doubles its footprint. On Windows the overflow spills into shared memory, and sampling gets many times slower. The Director inserts **Weight Backups to RAM** after the LTX loaders to prevent this.

## Licence

MIT (see `LICENSE`). Models are under their own licences. Only publish edits made with music and footage you have the rights to, and only cast invented people or people who have agreed to it.
