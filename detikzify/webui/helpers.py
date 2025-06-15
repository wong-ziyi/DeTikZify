from functools import cache, lru_cache
from inspect import signature
from operator import itemgetter
from os import fdopen
from tempfile import mkstemp

import gradio as gr

from ..infer import TikzDocument
from ..model import load

def to_svg(
    tikzdoc: TikzDocument,
    build_dir: str
):
    if not tikzdoc.is_rasterizable:
        if tikzdoc.compiled_with_errors:
            raise gr.Error("TikZ code did not compile!")
        else:
            gr.Warning("TikZ code compiled to an empty image!")
    elif tikzdoc.compiled_with_errors:
        gr.Warning("TikZ code compiled with errors!")

    fd, path = mkstemp(dir=build_dir, suffix=".svg")
    with fdopen(fd, "w") as f:
        if pdf:=tikzdoc.pdf:
            f.write(pdf[0].get_svg_image())
    return path if pdf else None

# https://stackoverflow.com/a/50992575
def make_ordinal(n):
    n = int(n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix

class MctsOutputs(set):
    def __init__(self, build_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_dir, self.svgmap, self.fails = build_dir, dict(), 0

    def add(self, score, tikzdoc): # type: ignore
        if (score, tikzdoc) not in self:
            try:
                 svg = to_svg(tikzdoc, build_dir=self.build_dir)
                 super().add((score, tikzdoc))
                 self.svgmap[tikzdoc] = svg
            except gr.Error:
                gr.Warning("TikZ code did not compile, discarding output!")
                if len(self): self.fails += 1
        elif len(self): self.fails += 1

    @property
    def programs(self):
        return [tikzdoc.code for _, tikzdoc in sorted(self, key=itemgetter(0), reverse=True)]

    @property
    def images(self):
        return [
            (self.svgmap[tikzdoc], make_ordinal(idx))
            for idx, (_, tikzdoc) in enumerate(sorted(self, key=itemgetter(0), reverse=True), 1)
        ]

    @property
    def first_success(self):
        return len(self) == 1 and not self.fails

def make_light(stylable):
    """
    Patch gradio to only contain light mode colors.
    """
    if isinstance(stylable, gr.themes.Base): # remove dark variants from the entire theme
        params = signature(stylable.set).parameters
        colors = {color: getattr(stylable, color.removesuffix("_dark")) for color in dir(stylable) if color in params}
        return stylable.set(**colors)
    elif isinstance(stylable, gr.Blocks): # also handle components which do not use the theme (e.g. modals)
        stylable.load(
            fn=None,
            js="() => document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'))"
        )
        return stylable
    else:
        raise ValueError

_cached = None

@lru_cache(maxsize=1)
def cached_load(*args, **kwargs):
    """Load the model once and cache it."""
    global _cached
    gr.Info("Instantiating model. This could take a while...")
    _cached = load(*args, **kwargs)
    return _cached

@cache
def info_once(message):
    gr.Info(message)

def clear_cached_model():
    """Release cached model and free GPU memory."""
    global _cached
    cached_load.cache_clear()
    try:
        import gc
        import torch
        if _cached:
            model, _ = _cached
            try:
                model.cpu()
            except Exception:
                pass
            _cached = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

class GeneratorLock:
    """
    Ensure that only one instance of a given generator is active.
    Useful when a previous invocation was canceled. See
    https://github.com/gradio-app/gradio/issues/8503 for more information.
    """
    def __init__(self, gen_func):
        self.gen_func = gen_func
        self.generator = None

    def generate(self, *args, **kwargs):
        if self.generator:
            if self.generator.gi_running:
                return # somehow we can end up here
            self.generator.close()
        self.generator = self.gen_func(*args, **kwargs)
        yield from self.generator

    def __call__(self, *args, **kwargs):
        yield from self.generate(*args, **kwargs)

# ---------------------------------------------------------------------------
# Connection helpers for integration with ShinyAppManage system
# ---------------------------------------------------------------------------

_user_info: dict = {}
_enable_hooks: bool = False

def configure_hooks(root_path: str | None):
    """Enable connection hooks when running behind a proxy."""
    global _enable_hooks
    _enable_hooks = bool(root_path)

def hooks_enabled() -> bool:
    return _enable_hooks

def url_execute(curl_type, cur_url, cur_data=None, cur_header=None):
    """Utility to send HTTP requests used by the management API."""
    if not _enable_hooks:
        return {"code": -1}
    import requests
    try:
        if curl_type == 1:
            res = requests.get(cur_url, headers=cur_header)
        else:
            res = requests.post(cur_url, headers=cur_header, json=cur_data)
        if res.status_code == 200:
            return res.json()
        else:
            return {"code": -1}
    except Exception:
        return {"code": -1}

def connect_user(request: gr.Request):
    """Notify ShinyAppManage that a session was created."""
    if not _enable_hooks:
        return _user_info
    query = dict(request.query_params or {})
    _user_info.clear()
    _user_info.update({
        "id": query.get("id"),
        "appName": query.get("appName"),
        "token": query.get("token"),
    })
    token = _user_info.get("token")
    if token:
        headers = {"Token": token, "Content-Type": "application/json"}
        connect_req = {
            "appName": _user_info.get("appName"),
            "action": "connect",
            "id": _user_info.get("id"),
        }
        data = url_execute(
            2,
            "http://10.2.26.152/sqx_fast/app/workstation/shiny-connect-action",
            connect_req,
            headers,
        )
        if data.get("code", -1) != 0:
            gr.Warning("Failed to notify connect action")
    return _user_info

def disconnect_user():
    """Notify ShinyAppManage that the session ended."""
    if not (_enable_hooks and _user_info):
        return
    headers = {
        "Token": _user_info.get("token"),
        "Content-Type": "application/json",
    }
    disconnect_req = {
        "appName": _user_info.get("appName"),
        "action": "disconnect",
        "id": _user_info.get("id"),
    }
    url_execute(
        2,
        "http://10.2.26.152/sqx_fast/app/workstation/shiny-connect-action",
        disconnect_req,
        headers,
    )
