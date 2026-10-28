// OneShot MV: a "Studio" button in ComfyUI that opens the Studio page in a new tab.
import { app } from "../../scripts/app.js";

const open = () => window.open("/oneshot-mv/", "_blank");

app.registerExtension({
  name: "OneShotMV.Studio",
  commands: [{ id: "OneShotMV.OpenStudio", label: "Open OneShot MV Studio", icon: "pi pi-video", function: open }],
  menuCommands: [{ path: ["OneShot MV"], commands: ["OneShotMV.OpenStudio"] }],
  actionBarButtons: [{ icon: "icon-[lucide--clapperboard]", label: "OneShot MV", tooltip: "Open OneShot MV Studio", onClick: open }],
});
