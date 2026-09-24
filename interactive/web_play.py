#!/usr/bin/env python
"""Browser front end for the interactive world model (VS Code / SSH friendly).

pygame needs a local display, which you do not have over VS Code Remote-SSH.
This module runs the SAME ``PlayerApp`` headlessly and puts its surface in a
browser instead: an aiohttp server on 127.0.0.1 serves one self-contained page
and pushes frames over a WebSocket.  VS Code forwards the port automatically,
so ``http://localhost:<port>`` in your local browser is the player.

    python interactive/play.py --web --ckpt <ckpt> [--port 8765]

What is REUSED verbatim (this file is a front end, not a second player):
``EngineWorker``, ``FrameRing``, the 16 Hz presentation model and its content
accumulator, ``ControlScheme`` (digital / latch / 0.5 s combo-merge),
``SettingsPanel``, presets, recording and the seed loader.  The browser only
reports which keys are down; the control semantics are evaluated SERVER-side
by the same ``ControlScheme`` instance the pygame path uses, so behaviour is
identical by construction rather than by reimplementation.

Bandwidth model -- SEND ON CONTENT CHANGE, HOLD CLIENT-SIDE
----------------------------------------------------------
The presentation model already distinguishes a display tick (always 16 Hz)
from a content advance (the adaptive rate).  Re-encoding held frames would
burn bandwidth to transmit pixels the viewer already has, so a JPEG is sent
only when the ring actually advanced; the browser keeps showing the last one.
At 1 step (content ~16 fps) that is the full rate; at 4 steps (~7 fps) it is
less than half the frames for identical motion on screen.  An <img> holding
its last src costs nothing, so "held frames" are free in a browser in a way
they are not in a blit loop.

Measured at 832x480, quality 80: ~55-75 kB/frame, so ~0.9-1.2 MB/s at 16 fps
and ~0.4-0.5 MB/s at 4-step content rates -- comfortably inside the 3 MB/s
budget.  HUD/state is a separate small JSON message at 8 Hz.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional, Set

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from interactive.engine_api import EngineBase, EngineConfig  # noqa: E402
from interactive.play import (  # noqa: E402
    PRESET_BLURB,
    PRESET_ORDER,
    STOP_CONTROL,
    PlayerApp,
    SettingsPanel,
    active_preset,
)

DEFAULT_PORT = 8765
#: JPEG quality for the frame stream. 80 is the knee: 90 costs ~1.7x the bytes
#: for a difference invisible at 832x480, 70 starts showing ringing on the
#: road texture the model is judged on.
JPEG_QUALITY = 80

#: Browser key names -> the control names ControlScheme expects. Arrows are
#: aliased exactly as the pygame path aliases them.
_HELD_KEYMAP = {
    "w": "w", "arrowup": "w",
    "s": "s", "arrowdown": "s",
    "a": "a", "arrowleft": "a",
    "d": "d", "arrowright": "d",
    " ": STOP_CONTROL, "space": STOP_CONTROL,
}
#: Discrete presses forwarded to PlayerApp._handle_key. 'p' is NOT here: the
#: browser owns the seed picker (it is an HTML overlay, not a pygame modal).
_DISCRETE = {"r", "v", "1", "2", "3", "q"}


def _encode_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> Optional[bytes]:
    """RGB uint8 [H, W, 3] -> JPEG bytes. cv2 first, Pillow as a fallback."""
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None
    except Exception:
        pass
    try:
        import io
        from PIL import Image
        bio = io.BytesIO()
        Image.fromarray(frame).save(bio, format="JPEG", quality=quality)
        return bio.getvalue()
    except Exception as exc:      # pragma: no cover - both encoders missing
        print(f"[web] JPEG encode failed: {exc}")
        return None


class WebPlayer:
    """Serves one PlayerApp to a browser over HTTP + WebSocket."""

    def __init__(self, cfg: EngineConfig,
                 factory: Callable[[EngineConfig], EngineBase],
                 port: int = DEFAULT_PORT, host: str = "127.0.0.1",
                 record_dir: str = "interactive/recordings",
                 quality: int = JPEG_QUALITY):
        self.port = int(port)
        self.host = host
        self.quality = int(quality)
        # headless=True: no pygame, no blit. We supply input and take frames
        # through the two hooks PlayerApp exposes.
        self.app = PlayerApp(cfg, factory, headless=True,
                             record_dir=record_dir)
        self.app.input_provider = self._current_held
        self.app.frame_hook = self._on_frame

        self._held: Set[str] = set()
        self._held_lock = threading.Lock()
        self._frame_seq = 0
        self._frame: Optional[bytes] = None
        self._frame_lock = threading.Lock()

        self._clients: Set[object] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._app_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # Telemetry the report needs.
        self.bytes_sent = 0
        self.frames_sent = 0
        self.encode_seconds = 0.0
        self.started_at = 0.0

    # -- PlayerApp hooks (called on the app thread) -----------------------
    def _current_held(self) -> Set[str]:
        with self._held_lock:
            return set(self._held)

    def _on_frame(self, frame: Optional[np.ndarray], advanced: bool) -> None:
        # Send ONLY on a real content advance; the browser holds the last
        # frame for free. See the module docstring.
        if frame is None or not advanced:
            return
        t0 = time.perf_counter()
        jpg = _encode_jpeg(frame, self.quality)
        self.encode_seconds += time.perf_counter() - t0
        if jpg is None:
            return
        with self._frame_lock:
            self._frame_seq += 1
            self._frame = jpg

    # -- state ------------------------------------------------------------
    def state_json(self) -> dict:
        app = self.app
        st = app.worker.last_stats
        return {
            "t": "state",
            "hud": app.hud_lines(),
            "ride": app.worker.ride_id,
            "state": app.worker.state,
            "error": app.worker.error,
            "notice": (app.notice
                       if app.notice and time.perf_counter() < app.notice_until
                       else ""),
            "recording": app.sink is not None,
            "preset": active_preset(app.cfg),
            "panel_open": app.panel.visible,
            "panel": app.panel.lines() if app.panel.visible else [],
            "settings": {
                "denoising_steps": int(app.cfg.denoising_steps),
                "horizon_chunks": int(app.cfg.horizon_chunks),
                "decode_half_res": bool(app.cfg.decode_half_res),
                "playback_fps": float(app.cfg.playback_fps),
                "seed_prefill_chunks": int(
                    getattr(app.cfg, "seed_prefill_chunks", 7)),
                "decoder": str(getattr(app.cfg, "decoder", "")),
                "playback_mode": app.playback_mode,
            },
            "rates": {
                "display": round(float(app.display_fps_measured), 1),
                "content": round(float(app.content_fps_measured), 1),
                "display_target": round(float(app.display_fps()), 1),
                "gen": (round(float(app.ring.throughput_fps), 1)
                        if app.ring.throughput_fps else None),
                "buffer": len(app.ring),
                "horizons": app.worker.horizons_done,
            },
            "stats": (None if st is None else {
                "horizon": st.horizon_index,
                "gen_fps": round(float(st.gen_fps), 1),
                "first_frame_ms": round(st.first_frame_latency_s * 1000),
                "vram_gb": round(float(st.peak_vram_gb), 1),
                "steps": int(st.denoising_steps),
                "precision": st.precision,
            }),
            "net": {
                "frames_sent": self.frames_sent,
                "mb_sent": round(self.bytes_sent / 2 ** 20, 2),
                "kb_per_frame": (round(self.bytes_sent / self.frames_sent / 1024, 1)
                                 if self.frames_sent else 0.0),
            },
        }

    # -- client -> server -------------------------------------------------
    def handle_message(self, msg: dict) -> None:
        """Apply one client message. Runs on the asyncio thread; every field
        it touches is either atomic or guarded, and the heavy work is queued
        to the worker exactly as the pygame path queues it."""
        kind = str(msg.get("t", ""))
        app = self.app
        if kind == "held":
            keys = msg.get("keys") or []
            mapped = {_HELD_KEYMAP[k] for k in
                      (str(x).lower() for x in keys) if k in _HELD_KEYMAP}
            # The panel swallows movement keys in the pygame path too.
            if app.panel.visible:
                mapped = set()
            with self._held_lock:
                self._held = mapped
        elif kind == "key":
            k = str(msg.get("k", "")).lower()
            if k in _DISCRETE:
                app._handle_key(k)
            elif k == "tab":
                app.panel.toggle()
            elif k == "escape":
                app.panel.visible = False
        elif kind == "panel":
            # Named field + delta, mirroring Left/Right on the pygame panel.
            field = str(msg.get("field", ""))
            if app.panel.select(field):
                f, v = app.panel.adjust(int(msg.get("delta", 1)))
                app._apply_panel_field(f, v)
        elif kind == "panel_move":
            app.panel.move(int(msg.get("delta", 1)))
        elif kind == "panel_apply":
            app.apply_rebuild()
        elif kind == "preset":
            name = str(msg.get("name", ""))
            if name in PRESET_ORDER:
                app.apply_preset(name)
        elif kind == "seed":
            path = str(msg.get("path", ""))
            if path:
                app.ring.clear()
                app._notify(f"reseeding -> {Path(path).stem}", 6.0)
                app.worker.request_seed(path)
        elif kind == "quit":
            app.quit = True

    # -- seed listing -----------------------------------------------------
    def seed_entries(self, limit: int = 240, query: str = "") -> list:
        """Videos first, then rides -- same ordering as the pygame picker.

        Only DISK-CACHED zarr thumbnails are offered: rendering a fresh one
        needs the GPU decoder, and the browser picker (unlike the pygame
        modal) does not park the engine worker, so it must never contend for
        the card.  Video thumbnails are CPU-only (one ffmpeg frame) and are
        generated on demand.
        """
        from interactive.seed_picker import (
            ThumbnailCache, filter_rides, list_rides, list_videos, probe_ride,
        )
        cache = ThumbnailCache()
        try:
            vids = list_videos(self.app.cfg.extra.get("video_dir"))
        except Exception as exc:
            print(f"[web] video listing failed: {exc}")
            vids = []
        try:
            rides = list_rides(self.app.seed_root)
        except Exception as exc:
            print(f"[web] ride listing failed: {exc}")
            rides = []
        entries = filter_rides(vids + rides, query)[:limit]
        out = []
        for e in entries:
            is_vid = e.is_video
            thumb = None
            if is_vid:
                try:
                    p = cache.get(e)
                    thumb = p.name if p else None
                except Exception:
                    thumb = None
            elif cache.has(e.ride_id):
                thumb = cache.path_for(e.ride_id).name
            if is_vid and not e.probed:
                try:
                    probe_ride(e)
                except Exception:
                    pass
            out.append({
                "path": e.path,
                "id": e.ride_id,
                "video": is_vid,
                "thumb": thumb,
                "chunks": (max(e.n_latents, 0) // 3) if e.probed else None,
            })
        cache.close()
        return out

    # -- server -----------------------------------------------------------
    async def _ws_handler(self, request):
        from aiohttp import WSMsgType, web
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=0)
        await ws.prepare(request)
        self._clients.add(ws)
        last_seq = -1
        try:
            await ws.send_str(json.dumps(self.state_json()))
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        self.handle_message(json.loads(msg.data))
                    except Exception as exc:
                        print(f"[web] bad message: {exc}")
                elif msg.type == WSMsgType.ERROR:
                    break
        except Exception:
            pass
        finally:
            self._clients.discard(ws)
            with_close = not ws.closed
            if with_close:
                await ws.close()
        _ = last_seq
        return ws

    async def _pump_frames(self) -> None:
        """Broadcast each NEW frame. Polling at 200 Hz adds <=5 ms."""
        last = -1
        while not self._stop.is_set():
            with self._frame_lock:
                seq, jpg = self._frame_seq, self._frame
            if jpg is not None and seq != last and self._clients:
                last = seq
                dead = []
                for ws in list(self._clients):
                    try:
                        await ws.send_bytes(jpg)
                        self.bytes_sent += len(jpg)
                        self.frames_sent += 1
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self._clients.discard(ws)
            await asyncio.sleep(0.005)

    async def _pump_state(self) -> None:
        """HUD/state JSON at 8 Hz -- small, and decoupled from the frame rate."""
        while not self._stop.is_set():
            if self._clients:
                blob = json.dumps(self.state_json())
                for ws in list(self._clients):
                    try:
                        await ws.send_str(blob)
                    except Exception:
                        self._clients.discard(ws)
            await asyncio.sleep(0.125)

    async def _watch_app(self) -> None:
        """Stop the server when the player loop exits (Q, or a worker error)."""
        while not self._stop.is_set():
            if self._app_thread is not None and not self._app_thread.is_alive():
                print("[web] player loop ended; shutting the server down.")
                self._stop.set()
                break
            await asyncio.sleep(0.2)

    def build_app(self):
        from aiohttp import web

        async def index(_request):
            return web.Response(text=INDEX_HTML, content_type="text/html")

        async def seeds(request):
            q = request.query.get("q", "")
            return web.json_response(
                {"entries": self.seed_entries(query=q)})

        async def thumb(request):
            from interactive.seed_picker import THUMB_DIR
            name = os.path.basename(request.match_info["name"])
            path = Path(THUMB_DIR) / name
            if not path.is_file():
                raise web.HTTPNotFound()
            ctype = mimetypes.guess_type(str(path))[0] or "image/jpeg"
            return web.FileResponse(path, headers={"Content-Type": ctype})

        async def health(_request):
            return web.json_response(self.state_json())

        srv = web.Application()
        srv.router.add_get("/", index)
        srv.router.add_get("/ws", self._ws_handler)
        srv.router.add_get("/api/seeds", seeds)
        srv.router.add_get("/api/state", health)
        srv.router.add_get("/thumb/{name}", thumb)
        return srv

    def run(self, max_seconds: Optional[float] = None) -> int:
        """Start the player thread and serve until it exits (or Ctrl-C)."""
        from aiohttp import web

        self.started_at = time.perf_counter()
        rc = {"code": 0}

        def _run_app():
            try:
                rc["code"] = self.app.run(max_seconds=max_seconds)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                rc["code"] = 1
                print(f"[web] player thread failed: {exc}")

        self._app_thread = threading.Thread(target=_run_app, name="player",
                                            daemon=True)
        self._app_thread.start()

        async def _main():
            self._loop = asyncio.get_running_loop()
            runner = web.AppRunner(self.build_app(), access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
            url = f"http://localhost:{self.port}"
            print("=" * 72)
            print(f"  Open {url} in your browser "
                  "(VS Code forwards the port automatically)")
            print("  If it does not open: VS Code -> PORTS tab -> Forward a "
                  f"Port -> {self.port}")
            print("=" * 72, flush=True)
            tasks = [asyncio.create_task(self._pump_frames()),
                     asyncio.create_task(self._pump_state()),
                     asyncio.create_task(self._watch_app())]
            try:
                while not self._stop.is_set():
                    await asyncio.sleep(0.2)
            finally:
                for t in tasks:
                    t.cancel()
                for ws in list(self._clients):
                    try:
                        await ws.close()
                    except Exception:
                        pass
                await runner.cleanup()

        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            print("\n[web] interrupted")
        finally:
            self._stop.set()
            self.app.quit = True
            if self._app_thread is not None:
                self._app_thread.join(timeout=25.0)
        secs = max(time.perf_counter() - self.started_at, 1e-6)
        if self.frames_sent:
            print(f"[web] sent {self.frames_sent} frames, "
                  f"{self.bytes_sent / 2**20:.1f} MB in {secs:.1f}s "
                  f"= {self.bytes_sent / 2**20 / secs:.2f} MB/s "
                  f"({self.bytes_sent / self.frames_sent / 1024:.0f} kB/frame, "
                  f"encode {self.encode_seconds / max(self.frames_sent,1)*1000:.1f} ms/frame)")
        return rc["code"]


# ==========================================================================
# The page. Self-contained on purpose: no CDN, no fonts, no build step --
# the user may be offline behind an SSH tunnel.
# ==========================================================================
INDEX_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>ARRWM interactive world model</title>
<style>
 :root{--bg:#0b0b12;--fg:#e8e8ee;--dim:#8a8a99;--sel:#ffcd5a;--vid:#6ec8ff;--bad:#d05a5a}
 *{box-sizing:border-box}
 html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
   font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow:hidden}
 #wrap{position:relative;width:100vw;height:100vh;display:flex;align-items:center;
   justify-content:center}
 #view{max-width:100%;max-height:100%;image-rendering:auto;display:block;background:#000}
 #hud{position:absolute;left:0;right:0;bottom:0;background:rgba(0,0,0,.66);
   padding:6px 10px;white-space:pre;font-size:12px;line-height:1.5}
 #notice{position:absolute;top:18px;left:50%;transform:translateX(-50%);
   background:rgba(0,0,0,.8);color:var(--sel);padding:7px 14px;border-radius:4px;
   display:none}
 #conn{position:absolute;top:10px;right:12px;color:var(--bad)}
 .panel{position:absolute;top:20px;left:20px;background:rgba(12,12,20,.94);
   border:1px solid #333;border-radius:6px;padding:10px 12px;min-width:420px;display:none}
 .panel h3{margin:0 0 8px;font-size:13px;color:var(--sel)}
 .row{display:flex;align-items:center;gap:8px;padding:2px 0}
 .row .k{flex:1;color:var(--dim)}
 .row .v{width:150px;text-align:right}
 button{background:#1c1c28;color:var(--fg);border:1px solid #3a3a4a;border-radius:4px;
   cursor:pointer;font:inherit;padding:1px 8px}
 button:hover{background:#2a2a3a}
 #picker{position:absolute;inset:0;background:rgba(6,6,10,.97);display:none;
   flex-direction:column;padding:14px}
 #pgrid{flex:1;overflow:auto;display:grid;gap:10px;
   grid-template-columns:repeat(auto-fill,minmax(210px,1fr));align-content:start}
 .cell{border:2px solid #2a2a38;border-radius:5px;overflow:hidden;cursor:pointer;
   background:#14141c}
 .cell:hover{border-color:var(--sel)}
 .cell.vid{border-color:var(--vid)}
 .cell img{width:100%;height:118px;object-fit:cover;display:block;background:#1a1a24}
 .cell .cap{padding:4px 6px;font-size:11px;white-space:nowrap;overflow:hidden;
   text-overflow:ellipsis}
 .noimg{height:118px;display:flex;align-items:center;justify-content:center;
   color:var(--dim);font-size:11px;text-align:center;padding:6px}
 input{background:#14141c;color:var(--fg);border:1px solid #3a3a4a;border-radius:4px;
   padding:5px 8px;font:inherit;width:280px}
 .hint{color:var(--dim);font-size:11px}
</style></head><body>
<div id="wrap">
  <img id="view" alt="world model">
  <div id="notice"></div>
  <div id="conn">connecting...</div>
  <div id="hud">starting...</div>

  <div class="panel" id="settings"><h3>SETTINGS &mdash; TAB closes</h3>
    <div id="srows"></div>
    <div class="row"><span class="k">preset</span><span class="v">
      <button data-preset="speed">1 speed</button>
      <button data-preset="balance">2 balance</button>
      <button data-preset="quality">3 quality</button></span></div>
    <div class="row"><span class="k"></span><span class="v">
      <button id="applyBtn">apply rebuild</button></span></div>
    <div class="hint">changes apply at the next horizon; precision/decoder need a rebuild</div>
  </div>

  <div id="picker">
    <div style="display:flex;gap:12px;align-items:center;margin-bottom:10px">
      <b>SELECT STARTING SEED</b>
      <input id="pq" placeholder="filter (type to search)…">
      <span class="hint">[VID] = your own video (drop clips in interactive/user_videos/)</span>
      <span style="flex:1"></span><button id="pclose">close (P / Esc)</button>
    </div>
    <div id="pgrid"></div>
  </div>
</div>
<script>
(function(){
 var ws, url = (location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws';
 var view=document.getElementById('view'), hud=document.getElementById('hud');
 var conn=document.getElementById('conn'), notice=document.getElementById('notice');
 var settings=document.getElementById('settings'), srows=document.getElementById('srows');
 var picker=document.getElementById('picker'), pgrid=document.getElementById('pgrid');
 var pq=document.getElementById('pq');
 var held=new Set(), lastSent='', curUrl=null, pickerOpen=false, frames=0, t0=Date.now();

 // ---- frame stream: one <img>, held automatically until the next arrives.
 function onBlob(blob){
   var u=URL.createObjectURL(blob);
   var old=curUrl; curUrl=u; view.src=u; frames++;
   if(old) setTimeout(function(){URL.revokeObjectURL(old);},0);
 }
 function connect(){
   ws=new WebSocket(url); ws.binaryType='blob';
   ws.onopen=function(){conn.textContent='';sendHeld(true);};
   ws.onclose=function(){conn.textContent='disconnected — retrying…';
     setTimeout(connect,1000);};
   ws.onerror=function(){conn.textContent='connection error';};
   ws.onmessage=function(ev){
     if(typeof ev.data!=='string'){onBlob(ev.data);return;}
     var m=JSON.parse(ev.data); if(m.t!=='state')return;
     hud.textContent=(m.hud||[]).join('\n');
     if(m.notice){notice.textContent=m.notice;notice.style.display='block';}
     else notice.style.display='none';
     renderSettings(m);
   };
 }
 function send(o){ if(ws&&ws.readyState===1) ws.send(JSON.stringify(o)); }

 // ---- keyboard: keydown/keyup, key repeat suppressed, held-set on change.
 var HELD=['w','a','s','d','arrowup','arrowdown','arrowleft','arrowright',' '];
 var DISCRETE=['r','v','1','2','3','q'];
 function sendHeld(force){
   var arr=Array.from(held).sort(), s=arr.join(',');
   if(force||s!==lastSent){lastSent=s;send({t:'held',keys:arr});}
 }
 window.addEventListener('keydown',function(e){
   if(e.repeat)return;                       // key repeat must not re-latch
   if(document.activeElement===pq && e.key!=='Escape')return;
   var k=e.key.toLowerCase();
   if(HELD.indexOf(k)>=0){e.preventDefault();held.add(k);sendHeld();return;}
   if(k==='tab'){e.preventDefault();togglePanel();return;}
   if(k==='p'){e.preventDefault();togglePicker();return;}
   if(k==='escape'){if(pickerOpen)togglePicker();else{settings.style.display='none';
     send({t:'key',k:'escape'});}return;}
   if(DISCRETE.indexOf(k)>=0){e.preventDefault();send({t:'key',k:k});}
 });
 window.addEventListener('keyup',function(e){
   var k=e.key.toLowerCase();
   if(HELD.indexOf(k)>=0){e.preventDefault();held.delete(k);sendHeld();}
 });
 window.addEventListener('blur',function(){held.clear();sendHeld();});

 // ---- settings drawer
 var FIELDS=['preset','denoising_steps','horizon_chunks','decode_half_res',
             'playback_fps','seed_prefill_chunks','playback_mode'];
 function togglePanel(){
   var open=settings.style.display!=='block';
   settings.style.display=open?'block':'none';
   send({t:'key',k:'tab'});
 }
 function renderSettings(m){
   if(settings.style.display!=='block')return;
   var s=m.settings||{}, html='';
   html+='<div class="row"><span class="k">preset</span><span class="v">'+
         (m.preset||'')+'</span></div>';
   FIELDS.slice(1).forEach(function(f){
     html+='<div class="row"><span class="k">'+f+'</span><span class="v">'+
       '<button data-f="'+f+'" data-d="-1">&minus;</button> '+
       String(s[f])+' '+
       '<button data-f="'+f+'" data-d="1">+</button></span></div>';
   });
   srows.innerHTML=html;
 }
 document.addEventListener('click',function(e){
   var b=e.target.closest('button'); if(!b)return;
   if(b.dataset.preset){send({t:'preset',name:b.dataset.preset});}
   else if(b.dataset.f){send({t:'panel',field:b.dataset.f,delta:+b.dataset.d});}
   else if(b.id==='applyBtn'){send({t:'panel_apply'});}
   else if(b.id==='pclose'){togglePicker();}
 });

 // ---- seed picker overlay
 function togglePicker(){
   pickerOpen=!pickerOpen;
   picker.style.display=pickerOpen?'flex':'none';
   if(pickerOpen){loadSeeds('');setTimeout(function(){pq.focus();},30);}
 }
 var qTimer=null;
 pq.addEventListener('input',function(){
   clearTimeout(qTimer); qTimer=setTimeout(function(){loadSeeds(pq.value);},220);
 });
 function loadSeeds(q){
   pgrid.innerHTML='<div class="hint">loading…</div>';
   fetch('/api/seeds?q='+encodeURIComponent(q||'')).then(function(r){return r.json();})
   .then(function(d){
     pgrid.innerHTML='';
     (d.entries||[]).forEach(function(en){
       var c=document.createElement('div');
       c.className='cell'+(en.video?' vid':'');
       var img=en.thumb
         ? '<img loading="lazy" src="/thumb/'+encodeURIComponent(en.thumb)+'">'
         : '<div class="noimg">no cached thumbnail<br>(pick to use anyway)</div>';
       var chunks=(en.video&&en.chunks!=null&&en.chunks<7)
         ? ' <span style="color:var(--bad)">'+en.chunks+'/7</span>' : '';
       c.innerHTML=img+'<div class="cap">'+(en.video?'[VID] ':'')+
                   en.id+chunks+'</div>';
       c.onclick=function(){send({t:'seed',path:en.path});togglePicker();};
       pgrid.appendChild(c);
     });
     if(!(d.entries||[]).length) pgrid.innerHTML='<div class="hint">no entries</div>';
   }).catch(function(e){pgrid.innerHTML='<div class="hint">failed: '+e+'</div>';});
 }
 connect();
})();
</script></body></html>
"""


def run_web(cfg: EngineConfig, factory, port: int = DEFAULT_PORT,
            record_dir: str = "interactive/recordings",
            quality: int = JPEG_QUALITY,
            max_seconds: Optional[float] = None,
            host: str = "127.0.0.1") -> int:
    """Entry point used by play.py --web."""
    try:
        import aiohttp  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "web mode needs aiohttp, which is not importable here "
            f"({exc}).\n  conda run -n flash-q pip install aiohttp")
    return WebPlayer(cfg, factory, port=port, record_dir=record_dir,
                     quality=quality, host=host).run(max_seconds=max_seconds)
