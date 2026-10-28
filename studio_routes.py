"""The OneShot MV Studio page, served by ComfyUI itself at http://127.0.0.1:8188/oneshot-mv/
(no extra install). Every request goes to fv/studio.py, reloaded on each call like the nodes'
modules, so the page and its API can change without restarting ComfyUI."""
from __future__ import annotations

import importlib
import os
import traceback

from aiohttp import web
from server import PromptServer

HERE = os.path.dirname(os.path.abspath(__file__))
routes = PromptServer.instance.routes


@routes.get("/oneshot-mv")
async def _studio_root(request):
    raise web.HTTPFound("/oneshot-mv/")


@routes.route("*", "/oneshot-mv/{tail:.*}")
async def _studio_dispatch(request):
    try:
        from .fv import studio
        if not os.environ.get("FV_NO_RELOAD"):
            importlib.reload(studio)
        return await studio.handle(request, HERE)
    except web.HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
