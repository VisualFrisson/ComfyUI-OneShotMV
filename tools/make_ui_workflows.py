"""Writes the ComfyUI (UI-format) example workflows in workflows/. Run after changing node inputs."""
import json
import os
import uuid

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "example_workflows")

VOICE_NOTE = """## OneShot MV: voice

**One performance, many performers.** Drop in a song and press **Queue**. The Director:
1. finds the beats, bars, sections and the drop;
2. pulls the vocal and drum stems;
3. plans the cut;
4. casts a performer per shot (Krea 2 portraits);
5. queues the jobs and then the final cut.

Each performer lip-syncs the vocal (LTX audio-to-video).

**The edit**
- Verses hold each shot for about 2 bars.
- The build tightens to a bar a shot.
- The drop cuts on the beat through pose-matched looks of the same moment, so every fast cut is a match cut.
- The cut never goes back to a shot it has left.

**theme** (optional): one line that sets the world, e.g. *a retro-futurist 1970s space station, warm orange and teal light*. It decides who the characters are, what they wear, the places, the light and the grade; the pack keeps the edit. Try it first in **Theme Preview**.

**Take QC** checks every take and re-renders bad ones, at most doubling the render time. With the SyncNet weights installed (see the README), it also scores lip sync.

**broll_share**: about 1 shot in 8 becomes a cutaway where nobody sings, never on the drop. With **broll_style** `mix`, weak lip sync gets a World character in a candid moment and instrumental gaps get a performer dancing to the track; `world` or `dance` uses one kind. 0 turns it off.

The finished MP4 lands in `output/oneshot-mv/runs/<run>/final_<format>.mp4`. Watch the queue: a 30 s run is roughly 40 to 100 jobs.

**Packs**
- **crowd**: a new face every shot.
- **eras**: invented archetypes through the decades.
- **artist**: one singer from a photo. Un-bypass **Load Image**, pick the photo and set the pack to `artist`.

**framing**
- **mixed** (default): wide, medium and close shots in the verses, tighter in the build, close-ups on the drop. Wide shots show the singer doing something in the scene (walking, leaning, working).
- **close**: close-ups only.

**preset**
- **quality**: LTX-2.3, 3 stages, 50 fps.
- **fast**: the same model at 25 fps with fewer steps.

**seconds**: 0 renders the whole song. **start**: -1 picks the strongest stretch.

Use music you have the rights to."""

MOTION_NOTE = """## OneShot MV: motion

**One dance, many dancers.** Load a song and a dance clip (full body, steady camera), then press **Queue**. A crew behind the lead dancer is fine: only the lead's pose guides the render. A new character takes over the same choreography. Every cut is a match cut, because each segment is rendered from the clip's pose (DWPose plus the LTX union-control IC-LoRA). The cuts are calm in the quiet parts, a bar a character in the build, and fast on the drop, where a second character flickers in doing the same move.

**Packs**
- **spotlight**: one dark stage, costume changes, motion trails.
- **wardrobe**: a new world every beat.

The finished MP4 lands in `output/oneshot-mv/runs/<run>/final_<format>.mp4`.

Use music and footage you have the rights to."""

RECUT_NOTE = """## OneShot MV: re-cut

Re-cuts a finished run without re-rendering anything. Paste the run folder (it is printed by the Director) and set **fx_intensity**:
- 0 = a clean cut;
- 1 = the house style;
- 2 to 3 = heavier.

The cut is re-derived from the pack's edit settings each time."""


class W:
    def __init__(self):
        self.nodes, self.links, self.nid, self.lid = [], [], 0, 0

    def node(self, type_, pos, size, widgets=None, inputs=(), outputs=(), mode=0, title=None, props=None):
        self.nid += 1
        n = {"id": self.nid, "type": type_, "pos": pos, "size": size, "flags": {}, "order": self.nid - 1,
             "mode": mode, "inputs": [dict(i, link=None) for i in inputs],
             "outputs": [dict(o, links=[]) for o in outputs],
             "properties": dict({"Node name for S&R": type_}, **(props or {})),
             "widgets_values": widgets if widgets is not None else []}
        if title:
            n["title"] = title
        self.nodes.append(n)
        return n

    def link(self, a, a_slot, b, b_name):
        self.lid += 1
        t = a["outputs"][a_slot]["type"]
        b_slot = next(i for i, x in enumerate(b["inputs"]) if x["name"] == b_name)
        a["outputs"][a_slot]["links"].append(self.lid)
        b["inputs"][b_slot]["link"] = self.lid
        self.links.append([self.lid, a["id"], a_slot, b["id"], b_slot, t])

    def dump(self, name):
        wf = {"id": str(uuid.uuid4()), "revision": 0, "last_node_id": self.nid, "last_link_id": self.lid,
              "nodes": self.nodes, "links": self.links, "groups": [], "config": {},
              "extra": {"ds": {"scale": 0.9, "offset": [40, 40]}}, "version": 0.4}
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            json.dump(wf, f, indent=1)


DIRECTOR_IN = [{"name": "audio", "type": "AUDIO"},
               {"name": "artist_image", "type": "IMAGE", "shape": 7},
               {"name": "dance_video", "type": "VIDEO", "shape": 7}]
DIRECTOR_OUT = [{"name": "summary", "type": "STRING"}, {"name": "run_dir", "type": "STRING"}]


def director_widgets(mode, pack, preset="quality", fmt="16:9", seconds=30.0, framing="mixed", seed=1, theme=""):
    # required: mode, pack, format, preset, seconds, start, seed (+ control), twins, queue_renders
    # optional widgets: theme, dance_video_path, run_name, twin_editor, guide_strength, queue_assemble,
    #                   framing, artist_image_path, broll_share, broll_style, qc_retries
    return [mode, pack, fmt, preset, seconds, -1.0, seed, "fixed", "auto", True,
            theme, "", "", "klein", 1.0, True, framing, "", 0.125 if mode == "voice" else 0.0, "mix", 2]


def voice():
    w = W()
    note = w.node("MarkdownNote", [0, 0], [420, 560], [VOICE_NOTE], title="How it works")
    song = w.node("LoadAudio", [450, 0], [320, 180], ["song.mp3", None, None],
                  inputs=[], outputs=[{"name": "AUDIO", "type": "AUDIO"}], title="Song")
    art = w.node("LoadImage", [450, 220], [320, 340], ["artist.png", "image"],
                 outputs=[{"name": "IMAGE", "type": "IMAGE"}, {"name": "MASK", "type": "MASK"}],
                 mode=4, title="Artist photo (artist pack only)")
    d = w.node("FVDirector", [800, 0], [380, 560], director_widgets("voice", "crowd"),
               inputs=DIRECTOR_IN, outputs=DIRECTOR_OUT, title="OneShot MV: Director")
    pv = w.node("PreviewAny", [1210, 0], [420, 260], [],
                inputs=[{"name": "source", "type": "*"}], outputs=[{"name": "STRING", "type": "STRING"}],
                title="Run summary")
    w.link(song, 0, d, "audio")
    w.link(art, 0, d, "artist_image")
    w.link(d, 0, pv, "source")
    w.dump("OneShotMV_Voice.json")


def motion():
    w = W()
    w.node("MarkdownNote", [0, 0], [420, 480], [MOTION_NOTE], title="How it works")
    song = w.node("LoadAudio", [450, 0], [320, 180], ["song.mp3", None, None],
                  outputs=[{"name": "AUDIO", "type": "AUDIO"}], title="Song")
    vid = w.node("LoadVideo", [450, 220], [320, 400], ["dance.mp4", "image"],
                 outputs=[{"name": "VIDEO", "type": "VIDEO"}], title="Dance clip")
    d = w.node("FVDirector", [800, 0], [380, 560],
               director_widgets("motion", "spotlight", seconds=16.0),
               inputs=DIRECTOR_IN, outputs=DIRECTOR_OUT, title="OneShot MV: Director")
    pv = w.node("PreviewAny", [1210, 0], [420, 260], [],
                inputs=[{"name": "source", "type": "*"}], outputs=[{"name": "STRING", "type": "STRING"}],
                title="Run summary")
    w.link(song, 0, d, "audio")
    w.link(vid, 0, d, "dance_video")
    w.link(d, 0, pv, "source")
    w.dump("OneShotMV_Motion.json")


def recut():
    w = W()
    w.node("MarkdownNote", [0, 0], [380, 260], [RECUT_NOTE], title="How it works")
    a = w.node("FVAssemble", [410, 0], [380, 200], ["oneshot-mv/runs/<run>", "oneshot-mv/recut", 1.0, True],
               outputs=[{"name": "preview", "type": "IMAGE"}, {"name": "video_path", "type": "STRING"}],
               title="OneShot MV: Assemble")
    p = w.node("PreviewImage", [820, 0], [420, 420], [], inputs=[{"name": "images", "type": "IMAGE"}])
    w.link(a, 0, p, "images")
    w.dump("OneShotMV_Recut.json")


if __name__ == "__main__":
    voice()
    motion()
    recut()
    print("wrote", sorted(f for f in os.listdir(OUT) if f.startswith("OneShotMV_")))
