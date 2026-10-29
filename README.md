# OneShot MV

**A song and one line of text in, a finished music video out.** Give OneShot MV a song and a
World ("the gladiator games in ancient Rome"), and it listens to the song, invents a cast for
that world, plans every shot from the song's own energy, and renders a lip-synced music video in
which a different invented person sings each line. It all runs locally in ComfyUI on
open-weights models.

![Twelve of the invented people from one full-song run](media/crowd_grid.jpg)

> Built for Comfy's **Open Source Higgsfield** challenge. Every face in this repo and in the
> videos it makes is invented. Use music you have the rights to.

## What it does

1. **Listens.** Beats, bars, sections and loudness, the drop, and the vocal and drum stems.
2. **Writes the World.** Your line becomes a small world bible: who the people are, what they
   wear, where each scene happens, the light and the camera.
3. **Plans the cut by energy.** Every part of the song is low, medium or high energy, and each
   level is cut its own way: held shots in the quiet parts, a cut every bar in the verses, a new
   face on the beat in the chorus or drop.
4. **Casts it.** Krea 2 makes a portrait of every performer, each one a different person.
5. **Checks every still.** A vision check looks at each portrait against what it should be (one
   person, the right gender, glasses, facial hair, the whole head in frame, no writing, nothing
   from after the World's period, the right outfit) and re-makes the ones that are wrong.
6. **Shoots it.** LTX-2.3 turns each portrait into a clip that sings the isolated vocal.
7. **Checks every take.** Missing, frozen or blown-out takes, dead mouths, a take that drifts to
   another person or place, bad framing, and weak lip sync (SyncNet) are re-shot.
8. **Adds b-roll.** A few shots become cutaways: a World character in a candid moment, not
   singing.
9. **Cuts and finishes it.** The edit on the beat, audio-reactive FX driven by the drums, a color
   grade, grain at 1080p and a vignette. Out come an MP4, an EDL and a Final Cut Pro XML.

## Three ways to run it

| | For | How |
|---|---|---|
| **The Studio** (recommended) | Everyone | A full app inside ComfyUI: brief, plan, live rendering, result, post processing and re-shoots. |
| **The ComfyUI workflow** | People who build in ComfyUI | The Director node with every setting, or the same run in ComfyUI's App mode. |
| **The Comfy SDK** | Scripts and automation | `run_sdk.bat` / `sdk_run.py`: one command takes a song and a World line to an MP4. |

All three drive the same nodes, and every run shows up in the Studio's **Your videos**.

### The Studio

With ComfyUI running, open **http://127.0.0.1:8188/oneshot-mv/**, or click **OneShot MV** in
ComfyUI's top bar. ComfyUI serves the page itself, so there's nothing else to install.

1. **The Brief.** Drop in a song. Its waveform shows the low, medium and high energy sections and
   the stretch the video will use (click the waveform to start somewhere else). Pick a length, a World (four examples, or
   write your own) and a format. The buttons show how long the plan and the video will take.
2. **The plan.** The World the tool wrote and every shot (when, what energy, who, wearing what,
   where) before anything long renders. Change the World and plan again, or make the video.
3. **Rendering.** The stages (plan, stills, vision check, clips, take check, b-roll, edit), a
   storyboard that fills in as stills and clips arrive, the checks' notes and the time left. You
   can close the tab and come back.
4. **Result.**
   - The video, **Download MP4** and **Export XML** (opens in Premiere Pro and DaVinci Resolve,
     every clip with its in and out points).
   - **Shots:** click any shot to re-shoot it: a **new take** of the same person, or a **new
     person**, with a note on what to change ("a woman in her seventies, in bright sun"). Pick
     several, shoot them together, and the video is cut again. The cut before is kept as a
     version you can switch back to.
   - **Post Processing**, without re-rendering anything:
     - **Audio-reactive FX** for low, medium and high energy sections: None, Subtle or Intense.
     - **Color grade:** None, or one of nine grades (Neon Saturated, Technicolor, Teal & Orange,
       Golden Hour, Warm Romantic, Soft Eterna, Low Key, Industrial Green, Silver Mono), each
       shown on a still from your video.
     - **Grain** (intensity and size, with a pixel-for-pixel 1080p preview) and **Vignette**.
     - **Apply changes** makes the new cut in about two minutes.
5. **Your videos** lists every run, from the Studio, the workflow or the SDK.

Every video starts at FX Subtle, color grade None (a natural grade is always applied underneath),
grain 25% at the finest size, and no vignette.

### The ComfyUI workflow

Open an example: **Templates** in the left sidebar, then **Extensions → ComfyUI-OneShotMV** (or
drag a file from `example_workflows/` onto the canvas).

- `OneShotMV_App.json` opens in ComfyUI's **App mode**: a song, a World line and a few settings,
  then **Run**. **App / Graph** at the top left shows the graph behind it.
- `OneShotMV_Voice.json`: the Director with every setting.
- `OneShotMV_Recut.json`: cut a finished run again at another FX level.

Turn **queue_renders** off and run once to see the plan in the summary (it renders nothing),
then turn it on and run again. The song analysis and the World are cached, so the second run
starts faster. The Director queues every job itself; the video lands in
`output/oneshot-mv/runs/<run>/final_<format>.mp4`.

### The Comfy SDK

`sdk_run.py` drives a whole run through the [Comfy SDK](https://pypi.org/project/comfy-sdk/):

1. it uploads the song and runs the Director, then prints the shot plan;
2. the run's stages render in ComfyUI;
3. a **Run Result** job waits for the finished video and hands it back, and the SDK downloads it
   into `oneshot_mv_out/`.

Comfy Cloud doesn't install custom node packs yet, so the SDK talks to your own ComfyUI through
Comfy's open-source [`comfy-api-proxy`](https://github.com/Comfy-Org/comfy-api-proxy), which gives
a local ComfyUI the same API as Comfy Cloud. `sdk_run.py` starts the proxy for you. Start ComfyUI
first.

On Windows (ComfyUI portable), double-click **`run_sdk.bat`** in the pack folder. The first time,
it installs `comfy-sdk` and `comfy-api-proxy` into `.sdk\` next to it, leaving ComfyUI's own
packages alone. It asks for the song (drag the file in), the World line and the length.

With options:

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

`python sdk_run.py --help` lists every option (length and start, format, preset, seed, FX, the cut
for each energy level, cast order, crowd, b-roll share, take-check rounds). `--preview` prints the
plan without rendering. To use another deployment with the OneShot MV nodes installed, set
`COMFY_BASE_URL` (and `COMFY_API_KEY` if it needs one).

## Writing a World

**One line is enough.** "The gladiator games in ancient Rome" gets you gladiators, an emperor,
senators and noblewomen in the arena, the stands and the imperial box, in period clothes. The
tool writes the rest.

**For more control, write it the way the Studio's examples are written:** the tone, then the
people in the order you want them, then the places, then the light.

> A 1980s Florida retirement community, played completely straight and over the top: seniors in
> their seventies and eighties with big attitude, in velour tracksuits, sun visors, pearls, gold
> chains and oversized sunglasses, at aqua aerobics in the condo pool, the bingo hall, the
> shuffleboard court, the early-bird buffet and rows of golf carts; harsh Florida sun and
> sun-faded pastel colour

Tips:
- **People come first, in your order** ("as written"). The people you name are cast before any
  the tool adds. Set **cast order** to "mixed" to shuffle them.
- **Name the period** ("1965", "ancient Rome", "1813"). It keeps modern clothes and objects out,
  and the vision check rejects stills that show them.
- **Say the tone:** "played straight like an epic film", "over the top", "cinematic".
- **Ages and adults:** casts are adults only. Give ages when they matter ("in their seventies").
- **Night or day:** say it ("at 3 a.m.", "hard midday sun"); the light follows.
- **Crowds:** for events (a stadium, a ball, a club), people appear behind the singers, dressed
  for the World. **Crowd** turns this on or off.
- **No real people or brands.** Brand names are replaced and signs are kept free of writing.

## Settings

| Setting | Choices | Default |
|---|---|---|
| Length / start | seconds (0 = whole song); start time or auto | 30 s from the strongest stretch |
| Format | 16:9 (1920×1080), 9:16 (1080×1920), 1:1 (1080×1080) | 16:9 |
| Quality | quality (50 fps renders), fast (25 fps) | quality |
| Cuts, per energy level | every 2 bars, every bar, to the music (subtle), to the music (intense) | low: every 2 bars; medium: every bar; high: to the music (subtle) |
| Audio-reactive FX | none, subtle, intense (per energy level in the Studio) | subtle |
| Cast order | as written, mixed | as written |
| Crowd behind singers | auto, on, off | auto |
| B-roll share | 0 to 0.5 of the shots | 0.125 |
| Take-check rounds | 0 to 3 | 2 |
| Seed | any number | 1 |

"To the music" cuts on the beat to a new face in the same spot: subtle about every half bar,
intense about every beat.

## Install

1. **The pack.** Clone into `ComfyUI/custom_nodes/` and restart ComfyUI:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/VisualFrisson/ComfyUI-OneShotMV
   ```
   Then `pip install -r requirements.txt` with ComfyUI's Python (numpy, OpenCV, imageio-ffmpeg).
2. **Node packs:** [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo),
   [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) (Sage attention and the LTX
   attention tuner) and [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) (samplers).
3. **Models.** See [Models and licences](#models-and-licences). Put them in ComfyUI's model
   folders, then edit `fv/models.json` so the file names match yours.
4. **Optional checks** (recommended):
   - **Lip-sync check (SyncNet):** `syncnet_v2.model` and `sfd_face.pth` in
     `ComfyUI/models/syncnet/`, from the
     [Oxford VGG SyncNet page](https://www.robots.ox.ac.uk/~vgg/software/lipsync/) (also mirrored
     in [syncnet-python](https://github.com/nawta/SyncNet_py309_313/tree/main/weights)). Without
     them the take check skips lip sync.
   - **Framing checks:** insightface's buffalo_l face models (`det_10g.onnx`, `w600k_r50.onnx`,
     non-commercial research use only) in `ComfyUI/models/insightface/models/buffalo_l/`. They run
     with onnxruntime. Without them the framing checks are skipped.
   - The vocal and drum stems use torchaudio's Hybrid Demucs, which downloads its weights once on
     first use.

### Hardware and time

Built and tested on an NVIDIA RTX PRO 6000 (96 GB). On it:

| Video | Time |
|---|---|
| 32 s, quality | about 70 min |
| 1:51, quality | about 6 h 45 min |
| Re-cut (Post Processing) | about 2 min |
| Re-shoot one shot | about 10–15 min, mostly model loads |

Model loads matter on a cold start: LTX-2.3 and its Gemma 3 text encoder (about 65 GB) take 5–9
minutes from a SATA SSD and about a minute from NVMe. Smaller GPUs are untested.

## What a run leaves behind

Everything for a run is in `output/oneshot-mv/runs/<run>/`:
- `final_<format>.mp4`, the EDL and (after Export XML) the XML;
- `plan.json` (the World, the cast and every shot) and the check reports;
- every portrait, clip and job (`jobs/`), so any shot can be queued again.

## Troubleshooting

- **No OneShot MV button or page:** restart ComfyUI once after installing, and check the console
  for import errors from the pack.
- **"Model not found" errors:** the names in `fv/models.json` must match your files as ComfyUI
  lists them, sub-folders included.
- **The lip-sync check or the framing checks were skipped:** their weights aren't installed (see
  Install, step 4). The video is still made.
- **A run stopped partway:** the Studio's Rendering screen says so. The ComfyUI console shows the
  error. Runs queued together render one after another.
- **Off-theme people or modern things in a period World:** name the period and the people in the
  World line, then re-shoot the wrong shots from the Result screen with a note.
- **Slow sampling on Windows with `--highvram`:** ComfyUI keeps LoRA weight backups on the GPU and
  overflows into shared memory. The Director adds **Weight Backups to RAM** after the LTX loaders
  to prevent this.

## Nodes

| Node | Purpose |
|---|---|
| **OneShot MV: Director** | Takes a song and a World line. Plans, casts and queues the whole run. |
| **OneShot MV: Vision Check** | Checks every portrait and b-roll still and re-makes the wrong ones (queued by the Director). |
| **OneShot MV: Take QC** | Checks every take, lip sync included, re-shoots bad ones, then queues the cut. |
| **OneShot MV: Assemble** | Cuts a run together with FX and the finish. Run it again to re-cut; nothing re-renders. |
| **OneShot MV: Run Result** | Says where a run is and returns the finished video as a job output (used by the SDK). |
| **OneShot MV: Theme Preview** | Shows what a World line turns into before you spend a run on it. |
| **OneShot MV: Lip Sync Score** | Scores a finished run's lip sync with SyncNet, take by take. |
| **OneShot MV: Finish** | The finish (grade, grain, vignette) on any frames. |
| **OneShot MV: Keep Models Loaded**, **Weight Backups to RAM**, **Free Memory** | Memory helpers the Director's jobs use. |
| **OneShot MV: Load Image / Video / Audio (path)** | Loaders the Director's jobs use. |
| **OneShot MV: Face Match**, **Lead Dancer Pose / Only** | Older experiments: one-artist looks, motion mode. |

**Style packs** (`fv/packs/`): one JSON file each with the casting grammar, the edit and the
effects. `crowd` (the default) is a different stranger on every line, close and to camera. `eras`
casts people from every era, and `artist` puts one singer from one photo in a new look every
shot. The dance-driven motion packs (`spotlight`, `wardrobe`) are experimental.

## Models and licences

The code is MIT; the models are open weights under their own terms, and none are included.
In short: below US$1M company revenue, the default models (Krea 2, LTX-2.3, Gemma 3) allow
commercial use, outputs included; the optional insightface face models are for non-commercial
research only; LTX's licence asks you to label what you publish as AI-generated. Read each
licence before you rely on it.

**A default run** (voice mode, `crowd` pack):

| Model | Used for | Licence | Download |
|---|---|---|---|
| Krea 2 Turbo and Raw, the Turbo LoRA (rank 64), the Qwen3-VL 4B text encoder and the Qwen-Image VAE | Every portrait and b-roll still | Krea 2 Community License: commercial use, of the model and its images, only for companies under US$1M annual revenue (Qwen3-VL and the VAE are Apache-2.0 upstream) | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) |
| LTX-2.3 22B dev, the distilled LoRA 384 v1.1 and the spatial upscaler x2 v1.1 | Every clip: lip sync, b-roll | LTX-2 Community License: free under US$10M annual revenue; label output as machine-generated; no impersonation without consent | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) |
| LTX-2 19B IC-LoRA Detailer | Fine detail in every clip | LTX-2 Community License | [Lightricks/LTX-2-19b-IC-LoRA-Detailer](https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer) |
| LTX2.3 Crisp Enhance and Soft Enhance (community LoRAs by vrgamedevgirl84) | Sharpness and detail in every clip | No licence stated by the author. To run without them, remove them from `ltx23_loras` in `fv/models.json` | [CivitAI: LTX 2.3 Enhancers](https://civitai.com/models/2535622/ltx-23-enhancers) |
| Gemma 3 12B IT | LTX-2.3's text encoder; also writes the World and the b-roll prompts, and runs the vision check | Gemma Terms of Use and Prohibited Use Policy | [Comfy-Org/ltx-2](https://huggingface.co/Comfy-Org/ltx-2/tree/main/split_files/text_encoders) |
| Hybrid Demucs (`HDEMUCS_HIGH_MUSDB_PLUS`) | Vocal and drum stems | Code MIT; no separate weights licence is published | Downloaded by torchaudio on first use |

**Optional checks** (Install, step 4):

| Model | Used for | Licence | Download |
|---|---|---|---|
| SyncNet v2 and the S3FD face detector | Lip-sync scores and each take's sync offset | Research use under CC-BY (Oxford VGG); no licence stated for S3FD | [VGG SyncNet page](https://www.robots.ox.ac.uk/~vgg/software/lipsync/) |
| insightface buffalo_l (`det_10g`, `w600k_r50`) | Framing and identity checks | Non-commercial research use only | [insightface model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) |

**Only for other packs and modes** (not loaded by a default run):

| Model | Used for | Licence | Download |
|---|---|---|---|
| FLUX.2 [klein] 9B fp8, its Qwen3 8B text encoder and the FLUX.2 small-decoder VAE | Look keyframes in the `eras` and `artist` packs; motion mode | FLUX Non-Commercial License (outputs may be used commercially); the VAE is Apache-2.0 | [black-forest-labs/FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8), [Comfy-Org/flux2-klein-9B](https://huggingface.co/Comfy-Org/flux2-klein-9B), [FLUX.2-small-decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) |
| LTX-2.3 IC-LoRA Union Control (ref0.5) | Pose-matched looks; motion mode | LTX-2 Community License | [Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control) |
| DWPose and YOLOX-L (TorchScript) | Pose guides for looks and motion mode | Apache-2.0 | Downloaded by comfyui_controlnet_aux |
| Qwen-Image 2.1, its VAE and the Qwen3-VL 8B text encoder | Portraits in the `artist` pack | Check the licence on the model page before commercial use | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) |
| LTX-2.5 22B distilled, its VAEs, upscaler and Gemma 4 text encoder | Motion mode at the `fast` preset | LTX-2.x Community License | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) |

File names and folders are set in `fv/models.json`.

## Licence

The code is MIT (see `LICENSE`). Vendored code keeps its own licence (`fv/syncnet/`, see
its NOTICE file). Models are under their own licences, listed above, and
are not included. Only publish videos made with music you have the rights to, label them as
AI-generated, and only cast invented people or people who have agreed to it.
