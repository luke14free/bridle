"""The simulator window: a local viewer for the rig, its skills, and what the robot is doing.

    from bridle.ui import Viewer
    v = Viewer(store, rig).start()        # http://127.0.0.1:8799
    v.push_frame(jpeg_bytes)              # whatever your sim just rendered
    v.set_job("descend_to_target", "training", "epoch 286 / 20.3M steps")

WHY BRIDLE SHIPS A UI AT ALL. A terminal agent cannot show you a robot. You can read that a skill
returned `ok`, and still not notice it dragged the cube 20cm across the table on the way — which is
exactly the class of failure that cost days in the codebase bridle came from, and was only ever
caught by watching. The agent conversation and the robot view are different senses and want
different windows.

WHY IT IS NOT A CODING AGENT. Pi (https://pi.dev, MIT) already is one, with an extension API and an
RPC mode. Forking it would mean maintaining a coding agent forever as a tax on the robotics work.
bridle ships the half nobody else has — the robot view, and the skill list annotated with whether
each skill actually runs on *your* rig — and integrates with Pi rather than replacing it. See
`docs/pi-extension.md`.

Stdlib only (http.server, no framework, no build step), because bridle core takes no dependencies and
a UI is not a good enough reason to break that. It is a viewer, not an application: it shows state
and streams frames. Anything that MUTATES the robot belongs behind the agent, where the contract
checks are.
"""
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bridle.orchestrator import build_tools
from bridle.resolve import ADAPT, RETRAIN, RUN
from bridle.store import BLOCKED

VERDICT_STYLE = {RUN: ("ready", "#3fb950"), ADAPT: ("needs re-distil", "#d29922"),
                 RETRAIN: ("needs rebuild", "#f85149"), BLOCKED: ("rig can't run it", "#8b949e")}

PAGE = """<!doctype html><meta charset=utf-8><title>bridle — %(rig)s</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#0d1117;color:#e6edf3;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
 header{padding:10px 16px;border-bottom:1px solid #30363d;display:flex;gap:16px;align-items:baseline}
 header b{font-size:15px}header span{color:#8b949e}
 main{display:grid;grid-template-columns:minmax(280px,1fr) 2fr;gap:0;height:calc(100vh - 44px)}
 #left{border-right:1px solid #30363d;overflow:auto;padding:12px 0}
 h2{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#8b949e;margin:14px 16px 6px}
 .app{padding:6px 16px;display:flex;gap:8px;align-items:baseline;border-left:2px solid transparent}
 .app:hover{background:#161b22}
 .dot{width:7px;height:7px;border-radius:50%%;flex:0 0 auto;margin-top:5px}
 .nm{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .vd{color:#8b949e;font-size:11px}
 #right{display:flex;flex-direction:column;min-width:0}
 #view{flex:1;display:flex;align-items:center;justify-content:center;background:#010409;overflow:hidden}
 #view img{max-width:100%%;max-height:100%%;image-rendering:pixelated}
 #view .idle{color:#484f58;text-align:center;line-height:1.8}
 #jobs{border-top:1px solid #30363d;padding:8px 16px;max-height:30vh;overflow:auto}
 .job{display:flex;gap:10px;padding:3px 0}.job b{min-width:180px}
 .st{color:#58a6ff}.age{color:#484f58;margin-left:auto}
</style>
<header><b>bridle</b><span id=rig></span><span id=cnt></span></header>
<main>
 <div id=left></div>
 <div id=right>
   <div id=view><div class=idle>no frames yet<br><small>push_frame() to stream the simulator here</small></div></div>
   <div id=jobs></div>
 </div>
</main>
<script>
let seen=-1;
async function tick(){
 try{
  const s=await (await fetch('api/state')).json();
  rig.textContent=s.rig.name+' · '+s.rig.embodiment+' · '+s.rig.cameras.join('+')+' cam';
  cnt.textContent=s.counts.run+' ready / '+s.apps.length+' skills';
  left.innerHTML=Object.entries(s.grouped).map(([v,list])=>!list.length?'':
    '<h2>'+s.labels[v]+' ('+list.length+')</h2>'+list.map(a=>
      '<div class=app title="'+(a.why||'').replace(/"/g,'')+'"><div class=dot style="background:'+s.colors[v]+'"></div>'
      +'<div class=nm>'+a.name+'</div><div class=vd>'+(a.why||'').slice(0,44)+'</div></div>').join('')).join('');
  jobs.innerHTML=s.jobs.length?s.jobs.map(j=>'<div class=job><b>'+j.name+'</b><span class=st>'+j.state
      +'</span><span>'+j.detail+'</span><span class=age>'+j.age+'s ago</span></div>').join(''):'';
  if(s.frame_seq!==seen&&s.frame_seq>=0){seen=s.frame_seq;
    view.innerHTML='<img src="frame.jpg?'+seen+'">';}
 }catch(e){}
 setTimeout(tick,%(poll)d);
}
tick();
</script>
"""


@dataclass
class Viewer:
    """A read-only window onto a rig, its skills, and the running robot."""

    store: object
    rig: object
    host: str = "127.0.0.1"
    port: int = 8799
    poll_ms: int = 500
    _frame: bytes = field(default=b"", repr=False)
    _frame_seq: int = -1
    _jobs: dict = field(default_factory=dict)
    _lock: object = field(default_factory=threading.Lock, repr=False)
    _srv: object = None
    _cache: tuple = field(default=(0.0, None), repr=False)

    # ── host-facing API ───────────────────────────────────────────────────────────────────────
    def push_frame(self, jpeg: bytes) -> None:
        """Publish the latest simulator frame. Encoding is the host's business; bridle core takes
        no image dependency, so this wants bytes a browser can render (JPEG or PNG)."""
        with self._lock:
            self._frame = jpeg
            self._frame_seq += 1

    def set_job(self, name, state, detail="") -> None:
        """Report a long-running job (a Foundry stage, a training run, an eval)."""
        with self._lock:
            self._jobs[name] = {"name": name, "state": state, "detail": detail, "t": time.time()}

    def clear_job(self, name) -> None:
        with self._lock:
            self._jobs.pop(name, None)

    # ── state ─────────────────────────────────────────────────────────────────────────────────
    def state(self) -> dict:
        """Rig, skills annotated with their verdicts, and jobs.

        Plans are cached for a second: `plan()` walks every app in the store and the page polls
        twice a second, and a viewer that pegs a core to render a static list is a viewer people
        turn off.
        """
        now = time.time()
        if self._cache[1] is not None and now - self._cache[0] < 1.0:
            apps = self._cache[1]
        else:
            apps = []
            for app in self.store.apps():
                try:
                    p = self.store.plan(app, self.rig)
                    why = p.reason if p.action != BLOCKED else ", ".join(p.blockers)
                    apps.append({"name": app.name, "verdict": p.action, "why": why})
                except Exception as e:
                    apps.append({"name": app.name, "verdict": RETRAIN,
                                 "why": f"{type(e).__name__}: {e}"})
            apps.sort(key=lambda a: a["name"])
            self._cache = (now, apps)

        grouped = {v: [a for a in apps if a["verdict"] == v] for v in (RUN, ADAPT, RETRAIN, BLOCKED)}
        with self._lock:
            jobs = [{**j, "age": int(now - j["t"])} for j in self._jobs.values()]
            seq = self._frame_seq
        return {
            "rig": {"name": self.rig.name, "embodiment": self.rig.embodiment,
                    "cameras": [c.name for c in self.rig.cameras],
                    "fingerprint": self.rig.fingerprint()},
            "apps": apps, "grouped": grouped,
            "labels": {v: VERDICT_STYLE[v][0] for v in VERDICT_STYLE},
            "colors": {v: VERDICT_STYLE[v][1] for v in VERDICT_STYLE},
            "counts": {v: len(grouped[v]) for v in grouped},
            "jobs": sorted(jobs, key=lambda j: -j["t"]), "frame_seq": seq,
        }

    # ── server ────────────────────────────────────────────────────────────────────────────────
    def _handler(self):
        viewer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass                                  # a viewer must not spam the agent's terminal

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0].lstrip("/")
                if path in ("", "index.html"):
                    html = PAGE % {"rig": viewer.rig.name, "poll": viewer.poll_ms}
                    return self._send(200, html.encode(), "text/html; charset=utf-8")
                if path == "api/state":
                    return self._send(200, json.dumps(viewer.state()).encode(), "application/json")
                if path == "frame.jpg":
                    with viewer._lock:
                        f = viewer._frame
                    if not f:
                        return self._send(404, b"no frame", "text/plain")
                    return self._send(200, f, "image/jpeg")
                return self._send(404, b"not found", "text/plain")

        return H

    def start(self) -> "Viewer":
        """Serve in a daemon thread. Returns self so it can be chained."""
        self._srv = ThreadingHTTPServer((self.host, self.port), self._handler())
        self.port = self._srv.server_address[1]        # resolve port=0
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
