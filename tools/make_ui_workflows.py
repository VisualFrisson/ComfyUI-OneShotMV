"""Writes the ComfyUI (UI-format) example workflows in workflows/. Run after changing node inputs."""
import json
import os
import uuid

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "example_workflows")

VOICE_NOTE = """## Frisson Vision: voice

**One performance, many performers.** Drop in a song and press **Queue**. The Director:
1. finds the beats, bars, sections and the drop;
2. pulls the vocal and drum stems;
3. plans a cut on the bar lines;
4. casts a performer per shot (Krea 2 portraits);
5. queues the jobs and then the final cut.

Each performer lip-syncs the vocal (LTX audio-to-video). Loud shots get pose-matched twins for flash inserts, and every take is checked and re-rolled if it is bad.

The finished MP4 lands in `output/frissonvision/runs/<run>/final_<format>.mp4`. Watch the queue: a 30 s run is roughly 20 to 40 jobs.

**Packs**
- **crowd**: a new face every shot (the edamame look).
- **eras**: invented archetypes through the decades.
- **artist**: one singer from a photo. Un-bypass **Load Image**, pick the photo and set the pack to `artist`.

**framing**
- **close**: close-ups only.
- **mixed**: medium and wide shots in the quiet stretches.

**preset**
- **quality**: LTX-2.3, 3 stages, 50 fps.
- **fast**: LTX-2.5 distilled, 2 stages, 25 fps.

**seconds**: 0 renders the whole song. **start**: -1 picks the strongest stretch.

Use music you have the rights to."""

MOTION_NOTE = """## Frisson Vision: motion

**One dance, many dancers.** Load a song and a dance clip (one person, full body, steady camera), then press **Queue**. A new character takes over the same choreography on the beat. Every cut is a match cut, because each segment is rendered from the clip's pose (DWPose plus the LTX union-control IC-LoRA). The cuts get faster as the section builds.

**Packs**
- **spotlight**: one dark stage, costume changes, motion trails.
- **wardrobe**: a new world every beat.

The finished MP4 lands in `output/frissonvision/runs/<run>/final_<format>.mp4`.

Use music and footage you have the rights to."""

RECUT_NOTE = """## Frisson Vision: re-cut

Re-cuts a finished run without re-rendering anything. Paste the run folder (it is printed by the Director) and set **fx_intensity**:
- 0 = a clean cut;
- 1 = the house style;
- 2 = maximum.

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


def director_widgets(mode, pack, preset="quality", fmt="9:16", seconds=30.0, framing="close", seed=1):
    # required: mode, pack, format, preset, seconds, start, seed (+ control), twins, queue_renders
    # optional widgets: dance_video_path, run_name, twin_editor, guide_strength, queue_assemble,
    #                   framing, artist_image_path, qc_retries
    return [mode, pack, fmt, preset, seconds, -1.0, seed, "fixed", "auto", True,
            "", "", "klein", 1.0, True, framing, "", 1]


def voice():
    w = W()
    note = w.node("MarkdownNote", [0, 0], [420, 560], [VOICE_NOTE], title="How it works")
    song = w.node("LoadAudio", [450, 0], [320, 180], ["song.mp3", None, None],
                  inputs=[], outputs=[{"name": "AUDIO", "type": "AUDIO"}], title="Song")
    art = w.node("LoadImage", [450, 220], [320, 340], ["artist.png", "image"],
                 outputs=[{"name": "IMAGE", "type": "IMAGE"}, {"name": "MASK", "type": "MASK"}],
                 mode=4, title="Artist photo (artist pack only)")
    d = w.node("FVDirector", [800, 0], [380, 560], director_widgets("voice", "crowd"),
               inputs=DIRECTOR_IN, outputs=DIRECTOR_OUT, title="Frisson Vision: Director")
    pv = w.node("PreviewAny", [1210, 0], [420, 260], [],
                inputs=[{"name": "source", "type": "*"}], outputs=[{"name": "STRING", "type": "STRING"}],
                title="Run summary")
    w.link(song, 0, d, "audio")
    w.link(art, 0, d, "artist_image")
    w.link(d, 0, pv, "source")
    w.dump("FrissonVision_Voice.json")


def motion():
    w = W()
    w.node("MarkdownNote", [0, 0], [420, 480], [MOTION_NOTE], title="How it works")
    song = w.node("LoadAudio", [450, 0], [320, 180], ["song.mp3", None, None],
                  outputs=[{"name": "AUDIO", "type": "AUDIO"}], title="Song")
    vid = w.node("LoadVideo", [450, 220], [320, 400], ["dance.mp4", "image"],
                 outputs=[{"name": "VIDEO", "type": "VIDEO"}], title="Dance clip")
    d = w.node("FVDirector", [800, 0], [380, 560],
               director_widgets("motion", "spotlight", fmt="1:1", seconds=16.0),
               inputs=DIRECTOR_IN, outputs=DIRECTOR_OUT, title="Frisson Vision: Director")
    pv = w.node("PreviewAny", [1210, 0], [420, 260], [],
                inputs=[{"name": "source", "type": "*"}], outputs=[{"name": "STRING", "type": "STRING"}],
                title="Run summary")
    w.link(song, 0, d, "audio")
    w.link(vid, 0, d, "dance_video")
    w.link(d, 0, pv, "source")
    w.dump("FrissonVision_Motion.json")


def recut():
    w = W()
    w.node("MarkdownNote", [0, 0], [380, 260], [RECUT_NOTE], title="How it works")
    a = w.node("FVAssemble", [410, 0], [380, 200], ["frissonvision/runs/<run>", "frissonvision/recut", 1.0, True],
               outputs=[{"name": "preview", "type": "IMAGE"}, {"name": "video_path", "type": "STRING"}],
               title="Frisson Vision: Assemble")
    p = w.node("PreviewImage", [820, 0], [420, 420], [], inputs=[{"name": "images", "type": "IMAGE"}])
    w.link(a, 0, p, "images")
    w.dump("FrissonVision_Recut.json")


if __name__ == "__main__":
    voice()
    motion()
    recut()
    print("wrote", sorted(f for f in os.listdir(OUT) if f.startswith("FrissonVision_")))
