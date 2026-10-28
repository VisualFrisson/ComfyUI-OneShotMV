"""OneShot MV: one song and one World line in, a full lip-sync music video out.
Krea 2 + LTX on ComfyUI. The Studio page: http://127.0.0.1:8188/oneshot-mv/"""
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:  # the Studio page and its API (studio_routes.py, fv/studio.py, studio/)
    from . import studio_routes  # noqa: F401
except Exception as e:  # never let the page stop the nodes from loading
    print(f"[OneShotMV] Studio page not available: {e}")

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
