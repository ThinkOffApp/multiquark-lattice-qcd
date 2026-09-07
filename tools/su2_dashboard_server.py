#!/usr/bin/env python3
import argparse
import hmac
import ipaddress
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class DashboardHandler(SimpleHTTPRequestHandler):
    root = Path(".").resolve()
    poll_interval = 0.5
    heartbeat_interval = 10.0
    chat_model = "gpt-4o-mini"
    allowed_tailscale_login = ""
    auth_token = ""
    # Peers whose Tailscale identity headers are believed (#37). Only a
    # `tailscale serve` proxy on this host, i.e. loopback, sets them honestly;
    # anyone else on the network can type them. Extra proxies via --trusted-proxy.
    trusted_proxies = ()
    cors_origin = ""
    protect_results = True
    json_cache = {}
    json_cache_lock = threading.Lock()
    json_cache_max_entries = max(
        32, int(os.environ.get("SU2_DASHBOARD_JSON_CACHE_MAX_ENTRIES", "256"))
    )
    telemetry_cache = {}
    telemetry_cache_lock = threading.Lock()
    telemetry_cache_ttl_sec = max(
        0.2, float(os.environ.get("SU2_DASHBOARD_TELEMETRY_CACHE_TTL_SEC", "1.0"))
    )
    control_lock = threading.Lock()
    terminal_phases = {
        "interrupted",
        "complete",
        "done",
        "failed",
        "error",
        "aborted",
        "stopped",
    }
    dashboard_alias_path = "/tools/su2_dashboard.html"
    dashboard_alias_seed = "petrus-su2-signal-e"
    dashboard_alias_combine = True
    dashboard_alias_token = ""
    worker_launcher_auto = "tools/start_su2_worker.sh"
    worker_launcher_cpu = "tools/start_su2_worker_ml84.sh"
    worker_launcher_gpu = "tools/start_su2_worker_gpu.sh"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.root), **kwargs)

    def end_headers(self):
        # Hardening headers applied to every response. Kept conservative so the
        # vanilla-JS dashboard (no inline eval, no remote scripts) is unaffected
        # but the server exposes less surface if it ever leaves 127.0.0.1.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        if self.cors_origin:
            self.send_header("Access-Control-Allow-Origin", self.cors_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Auth-Token")
            self.send_header("Vary", "Origin")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def list_directory(self, path):
        # Block the default Python directory listing. Browsing the repo root
        # over HTTP would expose .git/, .venv/, build/, etc. Anything that
        # legitimately needs file access goes through the JSON / SSE
        # endpoints (which are gated by protect_results auth where relevant).
        self.send_error(404, "Directory listing disabled")
        return None

    def send_head(self):
        # send_head is shared between do_GET and do_HEAD in
        # SimpleHTTPRequestHandler. Apply the denylist here so HEAD probes
        # for .git/, build/, etc. are also rejected.
        parsed = urlparse(self.path)
        if self._path_is_denied(parsed.path):
            self.send_error(404, "Not found")
            return None
        return super().send_head()

    # Path components matching any of these are refused outright. This blocks
    # serving developer / VCS artefacts that happen to live under --root
    # (e.g. .git/, .venv/, build/) and that the dashboard never legitimately
    # requests. Keep the list narrow; data lives under results/ and the page
    # itself under tools/.
    _denylist_components = ("..",)
    _denylist_prefixes = (".git/", ".venv/", "build/", "__pycache__/")

    def _path_is_denied(self, path):
        if path.startswith("/"):
            path = path[1:]
        if not path:
            return False
        if any(part in self._denylist_components for part in path.split("/")):
            return True
        if any(part.startswith(".") for part in path.split("/")):
            return True
        for prefix in self._denylist_prefixes:
            if path == prefix.rstrip("/") or path.startswith(prefix):
                return True
        return False

    def do_GET(self):
        parsed = urlparse(self.path)
        # Treat root as an alias for the dashboard so the user can paste a
        # bare host:port URL and land on the right page instead of getting
        # a 404 from the disabled directory listing.
        if parsed.path in {"", "/"}:
            self.send_response(302)
            self.send_header("Location", "/dashboard")
            self.end_headers()
            return
        if self._path_is_denied(parsed.path):
            self.send_error(404, "Not found")
            return
        if parsed.path in {"/dashboard", "/d"}:
            qs = parse_qs(parsed.query)
            seed = (qs.get("seed", [self.dashboard_alias_seed])[0] or "").strip()
            token = (qs.get("token", [self.dashboard_alias_token])[0] or "").strip()
            combine_raw = (qs.get("combine", ["1" if self.dashboard_alias_combine else "0"])[0] or "").strip().lower()
            combine_flag = combine_raw in {"1", "true", "yes", "on"}
            params = {}
            if seed:
                params["seed"] = seed
            if token:
                params["token"] = token
            if combine_flag:
                params["combine"] = "1"
            target = self.dashboard_alias_path
            if params:
                target = f"{target}?{urlencode(params)}"
            self.send_response(302)
            self.send_header("Location", target)
            self.end_headers()
            return
        if self.protect_results and parsed.path.startswith("/results/"):
            if not self.is_authorized(parsed=parsed):
                self.send_unauthorized()
                return
        if parsed.path == "/thread_telemetry":
            if not self.is_authorized(parsed=parsed):
                self.send_unauthorized()
                return
            self.handle_thread_telemetry(parsed)
            return
        if parsed.path == "/events":
            if not self.is_authorized(parsed=parsed, allow_query_token=True):
                self.send_unauthorized()
                return
            self.handle_events(parsed)
            return
        if parsed.path == "/api/runs":
            if self.protect_results and not self.is_authorized(parsed=parsed):
                self.send_unauthorized()
                return
            self.handle_list_runs()
            return
        # Serve live_*.json with JSONL measurements merged in
        if parsed.path.startswith("/results/") and "/live_" in parsed.path and parsed.path.endswith(".json"):
            try:
                local = self.resolve_local_path(parsed.path)
            except ValueError:
                pass
            else:
                qs = parse_qs(parsed.query)
                combine = qs.get("combine", ["0"])[0].strip() in ("1", "true", "yes")
                if combine:
                    data, err = self.load_live_combined(local)
                else:
                    data, err = self.load_live_with_jsonl(local)
                if data is not None:
                    body = json.dumps(data, separators=(",", ":")).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/chat":
            self.handle_chat(parsed)
            return
        if parsed.path == "/thread_control":
            self.handle_thread_control(parsed)
            return
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error":"not found"}')

    def handle_list_runs(self):
        """Enumerate progress_*.json under results/ so the dashboard can offer
        a run picker instead of hand-typed paths."""
        runs = []
        results_root = self.root / "results"
        if results_root.is_dir():
            for prog in sorted(results_root.glob("*/progress_*.json")):
                rel = prog.relative_to(self.root)
                name = prog.name[len("progress_"):-len(".json")]
                # Per-seed remeasure files clutter the list; keep the total.
                if name.startswith("remeasure_") and name != "remeasure_total":
                    continue
                live = prog.parent / f"live_{name}.json"
                if name == "remeasure_total":
                    candidates = sorted(prog.parent.glob("live_*.json"))
                    live = candidates[0] if candidates else live
                entry = {
                    "label": f"{prog.parent.name} / {name}",
                    "progress": f"/{rel.as_posix()}",
                    "live": f"/{(live.relative_to(self.root)).as_posix()}" if live.exists() else "",
                }
                try:
                    entry["mtime"] = int(prog.stat().st_mtime)
                except OSError:
                    entry["mtime"] = 0
                runs.append(entry)
        runs.sort(key=lambda r: -r["mtime"])
        body = json.dumps({"runs": runs}, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    _TOKEN_IN_URL = re.compile(r"(token=)[^&\s]*")

    def log_message(self, fmt, *args):
        # Query-string tokens (the SSE stream has no header channel) must not
        # end up in the access log (#37).
        try:
            args = tuple(self._TOKEN_IN_URL.sub(r"\1<redacted>", a) if isinstance(a, str) else a for a in args)
        except Exception:
            pass
        super().log_message(fmt, *args)

    def peer_is_trusted_proxy(self) -> bool:
        """True when the TCP peer is loopback or an explicitly trusted proxy."""
        try:
            host = self.client_address[0]
        except (AttributeError, IndexError, TypeError):
            return False
        try:
            ip = ipaddress.ip_address(str(host).split("%", 1)[0])
        except ValueError:
            return False
        if ip.is_loopback:
            return True
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped and ip.ipv4_mapped.is_loopback:
            return True
        for entry in self.trusted_proxies or ():
            try:
                net = ipaddress.ip_network(str(entry).strip(), strict=False)
            except ValueError:
                continue
            probe = ip.ipv4_mapped if (isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped and net.version == 4) else ip
            if probe.version == net.version and probe in net:
                return True
        return False

    def get_tailscale_login(self) -> str:
        # Identity headers are plain request headers; they prove nothing unless
        # the peer is the proxy that sets them (#37).
        if not self.peer_is_trusted_proxy():
            return ""
        for key in (
            "Tailscale-User-Login",
            "X-Tailscale-User-Login",
            "Tailscale-User-Name",
            "X-Tailscale-User-Name",
        ):
            v = self.headers.get(key)
            if v:
                return v.strip().lower()
        return ""

    def get_token(self, parsed=None, payload=None, allow_query=False) -> str:
        auth = (self.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        header_token = (self.headers.get("X-Auth-Token") or "").strip()
        if header_token:
            return header_token
        # ?token= is accepted only where the browser cannot send a header
        # (EventSource); everywhere else it would just leak into logs (#37).
        if allow_query and parsed:
            qs_token = parse_qs(parsed.query).get("token", [""])[0].strip()
            if qs_token:
                return qs_token
        if isinstance(payload, dict):
            body_token = payload.get("token")
            if isinstance(body_token, str):
                return body_token.strip()
        return ""

    def is_authorized(self, parsed=None, payload=None, allow_query_token=False) -> bool:
        expected_login = (self.allowed_tailscale_login or "").strip().lower()
        expected_token = (self.auth_token or "").strip()
        if not expected_login and not expected_token:
            return True

        if expected_login:
            login = self.get_tailscale_login()
            if login and login == expected_login:
                return True

        if expected_token:
            token = self.get_token(parsed=parsed, payload=payload, allow_query=allow_query_token)
            if token and hmac.compare_digest(token, expected_token):
                return True

        return False

    def send_unauthorized(self):
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("WWW-Authenticate", 'Bearer realm="su2-dashboard"')
        self.end_headers()
        self.wfile.write(
            b'{"error":"unauthorized","hint":"use Tailscale login allowlist or valid auth token"}'
        )

    def resolve_local_path(self, url_path: str) -> Path:
        p = (url_path or "").strip()
        if p.startswith("http://") or p.startswith("https://"):
            raise ValueError("absolute URLs are not supported for streaming endpoint")
        rel = p[1:] if p.startswith("/") else p
        full = (self.root / rel).resolve()
        if not (full == self.root or self.root in full.parents):
            raise ValueError("path escapes workspace root")
        return full

    @staticmethod
    def file_sig(path: Path):
        if not path.exists():
            return (-1, -1)
        try:
            st = path.stat()
        except Exception:
            return (-1, -1)
        return (int(st.st_mtime_ns), int(st.st_size))

    @staticmethod
    def cache_get(key: str, sig):
        now_ns = time.monotonic_ns()
        with DashboardHandler.json_cache_lock:
            cached = DashboardHandler.json_cache.get(key)
            if cached and cached.get("sig") == sig:
                cached["atime_ns"] = now_ns
                return cached.get("data")
        return None

    @staticmethod
    def cache_put(key: str, sig, data):
        now_ns = time.monotonic_ns()
        with DashboardHandler.json_cache_lock:
            DashboardHandler.json_cache[key] = {
                "sig": sig,
                "data": data,
                "atime_ns": now_ns,
            }
            max_entries = int(DashboardHandler.json_cache_max_entries)
            if max_entries > 0 and len(DashboardHandler.json_cache) > max_entries:
                overflow = len(DashboardHandler.json_cache) - max_entries
                oldest = sorted(
                    DashboardHandler.json_cache.items(),
                    key=lambda kv: int(kv[1].get("atime_ns", 0)),
                )[:overflow]
                for old_key, _ in oldest:
                    DashboardHandler.json_cache.pop(old_key, None)

    @staticmethod
    def read_json(path: Path):
        if not path.exists():
            return None, f"{path} not found"
        try:
            sig = DashboardHandler.file_sig(path)
        except Exception as e:
            return None, str(e)

        key = f"json:{path}"
        cached = DashboardHandler.cache_get(key, sig)
        if cached is not None:
            return cached, None

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            DashboardHandler.cache_put(key, sig, data)
            return data, None
        except Exception as e:
            return None, str(e)

    @staticmethod
    def read_jsonl(path: Path):
        if not path.exists():
            return [], None
        sig = DashboardHandler.file_sig(path)
        key = f"jsonl:{path}"
        cached = DashboardHandler.cache_get(key, sig)
        if cached is not None:
            return cached, None
        try:
            meas = []
            with path.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        # Ignore partial/truncated lines while writer appends.
                        continue
                    if isinstance(row, dict):
                        meas.append(row)
            DashboardHandler.cache_put(key, sig, meas)
            return meas, None
        except Exception as e:
            return None, str(e)

    @staticmethod
    def infer_seed_from_live_path(path: Path):
        name = path.name
        if name.startswith("live_") and name.endswith(".json"):
            return name[len("live_") : -len(".json")]
        return ""

    @staticmethod
    def infer_seed_from_progress_path(path: Path):
        name = path.name
        if name.startswith("progress_") and name.endswith(".json"):
            return name[len("progress_") : -len(".json")]
        return ""

    @staticmethod
    def base_seed(seed: str):
        txt = str(seed or "").strip()
        return re.sub(r"-(b|c|d|e|f|g|h)$", "", txt)

    @staticmethod
    def seed_group(base_seed: str):
        root = DashboardHandler.base_seed(base_seed)
        if not root:
            return []
        # Must match `threadSlots` in tools/su2_dashboard.html (root, -b, -c,
        # -d). Until Sep 2026 this returned -e..-h, so telemetry never lined up
        # with the page: thread A showed "cpu -" / "mem -" and sibling rows
        # could never be recognised (issue #52). The root seed is included so
        # thread A itself is covered.
        return [root, f"{root}-b", f"{root}-c", f"{root}-d"]

    @staticmethod
    def parse_ps_number(value, default=None):
        try:
            return float(str(value).strip())
        except Exception:
            return default

    @staticmethod
    def infer_pipeline_from_command(command: str):
        cmd = str(command or "").lower()
        if not cmd:
            return None
        if "start_su2_worker_gpu.sh" in cmd:
            return "gpu"
        if "start_su2_worker_ml84.sh" in cmd:
            return "cpu"
        if "--pipeline-label gpu" in cmd:
            return "gpu"
        if "--pipeline-label cpu" in cmd:
            return "cpu"
        if "start_su2_worker.sh" in cmd:
            if re.search(r"\bstart_su2_worker\.sh\b.*\bgpu\b", cmd):
                return "gpu"
            if re.search(r"\bstart_su2_worker\.sh\b.*\bcpu\b", cmd):
                return "cpu"
            return "auto"
        return None

    @staticmethod
    def collect_thread_telemetry(base_seed: str):
        root = DashboardHandler.base_seed(base_seed)
        if not root:
            return {"base_seed": "", "threads": {}}

        cache_key = f"telemetry:{root}"
        now = time.time()
        with DashboardHandler.telemetry_cache_lock:
            cached = DashboardHandler.telemetry_cache.get(cache_key)
            if cached:
                age = now - float(cached.get("ts", 0.0))
                if age <= DashboardHandler.telemetry_cache_ttl_sec:
                    return dict(cached.get("payload") or {})

        seeds = DashboardHandler.seed_group(root)
        patterns = {
            s: re.compile(rf"(?:^|\s)--seed(?:=|\s+){re.escape(s)}(?:\s|$)")
            for s in seeds
        }
        best = {}
        try:
            # macOS: rss/vsz are in KiB, %cpu is floating-point.
            out = subprocess.check_output(
                ["ps", "-Ao", "pid,state,%cpu,rss,vsz,etime,command"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            out = ""

        for raw in out.splitlines()[1:]:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(None, 6)
            if len(parts) < 7:
                continue
            pid_s, state_s, cpu_s, rss_s, vsz_s, etime_s, command = parts
            if "su2_2q_signal_scan.py" not in command:
                continue
            for seed, pat in patterns.items():
                if not pat.search(command):
                    continue
                pid = int(DashboardHandler.parse_ps_number(pid_s, -1))
                cpu = DashboardHandler.parse_ps_number(cpu_s, None)
                rss_kb = DashboardHandler.parse_ps_number(rss_s, None)
                vsz_kb = DashboardHandler.parse_ps_number(vsz_s, None)
                rec = {
                    "found": True,
                    "pid": pid if pid > 0 else None,
                    "state": state_s.strip(),
                    "paused": str(state_s or "").strip().upper().startswith("T"),
                    "cpu_pct": cpu,
                    "rss_kb": rss_kb,
                    "rss_mb": (rss_kb / 1024.0) if isinstance(rss_kb, (int, float)) else None,
                    "vsz_kb": vsz_kb,
                    "vsz_mb": (vsz_kb / 1024.0) if isinstance(vsz_kb, (int, float)) else None,
                    "etime": etime_s,
                    "command": command,
                    "pipeline": DashboardHandler.infer_pipeline_from_command(command),
                }
                prev = best.get(seed)
                if not prev:
                    best[seed] = rec
                    continue
                prev_cpu = DashboardHandler.parse_ps_number(prev.get("cpu_pct"), -1.0)
                cur_cpu = DashboardHandler.parse_ps_number(rec.get("cpu_pct"), -1.0)
                prev_pid = int(DashboardHandler.parse_ps_number(prev.get("pid"), -1))
                cur_pid = int(DashboardHandler.parse_ps_number(rec.get("pid"), -1))
                # Prefer higher CPU usage; tie-break by newer PID.
                if cur_cpu > prev_cpu + 1e-9 or (abs(cur_cpu - prev_cpu) <= 1e-9 and cur_pid > prev_pid):
                    best[seed] = rec

        threads = {}
        for seed in seeds:
            if seed in best:
                threads[seed] = best[seed]
            else:
                threads[seed] = {
                    "found": False,
                    "pid": None,
                    "state": None,
                    "paused": False,
                    "cpu_pct": None,
                    "rss_kb": None,
                    "rss_mb": None,
                    "vsz_kb": None,
                    "vsz_mb": None,
                    "etime": None,
                    "command": None,
                    "pipeline": None,
                }

        payload = {
            "base_seed": root,
            "threads": threads,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        with DashboardHandler.telemetry_cache_lock:
            DashboardHandler.telemetry_cache[cache_key] = {"ts": now, "payload": payload}
        return dict(payload)

    @staticmethod
    def invalidate_telemetry_cache(base_seed: str):
        root = DashboardHandler.base_seed(base_seed)
        if not root:
            return
        key = f"telemetry:{root}"
        with DashboardHandler.telemetry_cache_lock:
            DashboardHandler.telemetry_cache.pop(key, None)

    @staticmethod
    def process_alive(pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False

    @staticmethod
    def find_worker_record(seed: str):
        s = str(seed or "").strip()
        if not s:
            return None
        payload = DashboardHandler.collect_thread_telemetry(DashboardHandler.base_seed(s))
        threads = payload.get("threads") if isinstance(payload, dict) else {}
        rec = threads.get(s) if isinstance(threads, dict) else None
        if isinstance(rec, dict) and rec.get("found") and rec.get("pid"):
            out = dict(rec)
            out["seed"] = s
            return out
        return None

    @staticmethod
    def launch_worker(seed: str, command_hint: str = "", pipeline: str = "auto"):
        s = str(seed or "").strip()
        if not s:
            raise RuntimeError("seed is required")
        requested_pipeline = str(pipeline or "auto").strip().lower()
        if requested_pipeline not in {"auto", "cpu", "gpu"}:
            requested_pipeline = "auto"

        root = DashboardHandler.root
        launcher_auto = (root / DashboardHandler.worker_launcher_auto).resolve()
        launcher_cpu = (root / DashboardHandler.worker_launcher_cpu).resolve()
        launcher_gpu = (root / DashboardHandler.worker_launcher_gpu).resolve()

        argv = None
        launcher = None
        effective_pipeline = requested_pipeline
        if requested_pipeline == "gpu":
            if launcher_gpu.exists():
                launcher = launcher_gpu
                argv = ["/bin/bash", str(launcher), s]
            elif launcher_auto.exists():
                launcher = launcher_auto
                argv = ["/bin/bash", str(launcher), s, "gpu"]
            else:
                raise RuntimeError("gpu pipeline requested but gpu launcher not found")
        elif requested_pipeline == "cpu":
            if launcher_auto.exists():
                launcher = launcher_auto
                argv = ["/bin/bash", str(launcher), s, "cpu"]
            elif launcher_cpu.exists():
                launcher = launcher_cpu
                argv = ["/bin/bash", str(launcher), s]
            else:
                raise RuntimeError("cpu pipeline requested but cpu launcher not found")
        else:
            # auto: prefer profile-aware launcher, then legacy CPU launcher.
            if launcher_auto.exists():
                launcher = launcher_auto
                argv = ["/bin/bash", str(launcher), s, "auto"]
            elif launcher_cpu.exists():
                launcher = launcher_cpu
                argv = ["/bin/bash", str(launcher), s]
                effective_pipeline = "cpu"
            else:
                effective_pipeline = "auto"

        if launcher and argv:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return {
                "method": "launcher",
                "pid": int(proc.pid),
                "command": " ".join(shlex.quote(x) for x in argv),
                "launcher": str(launcher),
                "pipeline_requested": requested_pipeline,
                "pipeline_effective": effective_pipeline,
            }

        cmd = str(command_hint or "").strip()
        if not cmd:
            raise RuntimeError("no launch method available (missing launcher and command hint)")
        argv = shlex.split(cmd)
        if len(argv) == 0:
            raise RuntimeError("empty command")
        cwd_candidates = [
            (root / "grid-gpt" / "gpt").resolve(),
            (root / "gpt").resolve(),
            root.resolve(),
        ]
        cwd = None
        for c in cwd_candidates:
            if c.exists():
                cwd = c
                break
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {
            "method": "command",
            "pid": int(proc.pid),
            "command": cmd,
            "pipeline_requested": requested_pipeline,
            "pipeline_effective": DashboardHandler.infer_pipeline_from_command(cmd) or requested_pipeline,
        }

    @staticmethod
    def measurement_key(row):
        if not isinstance(row, dict):
            return None
        seed = row.get("_seed")
        idx = row.get("idx")
        if isinstance(idx, (int, float)):
            return ("idx", int(idx), seed)
        cfg_idx = row.get("cfg_idx")
        if isinstance(cfg_idx, (int, float)):
            return ("cfg_idx", int(cfg_idx), seed)
        return None

    @staticmethod
    def merge_measurements(base_rows, tail_rows):
        out = []
        seen = set()
        if isinstance(base_rows, list):
            for row in base_rows:
                if not isinstance(row, dict):
                    continue
                out.append(row)
                key = DashboardHandler.measurement_key(row)
                if key is not None:
                    seen.add(key)
        if isinstance(tail_rows, list):
            for row in tail_rows:
                if not isinstance(row, dict):
                    continue
                key = DashboardHandler.measurement_key(row)
                if key is not None and key in seen:
                    continue
                out.append(row)
                if key is not None:
                    seen.add(key)
        return out

    @staticmethod
    def load_live_with_jsonl(path: Path):
        """Read the live JSON and merge measurements from checkpoint + JSONL tail."""
        data, err = DashboardHandler.read_json(path)
        if data is None:
            return data, err
        if not isinstance(data, dict):
            return data, err
        # Never mutate the cached JSON object in-place.
        out = dict(data)

        jsonl_path = None
        jsonl_name = (out.get("meta") or {}).get("jsonl_path")
        if isinstance(jsonl_name, str) and jsonl_name.strip():
            jsonl_path = path.parent / jsonl_name.strip()
        else:
            fallback = path.with_suffix(".jsonl")
            if fallback.exists():
                jsonl_path = fallback

        seed = ((out.get("meta") or {}).get("seed") or "").strip()
        if not seed:
            seed = DashboardHandler.infer_seed_from_live_path(path)

        combined_rows = []
        inline_rows = out.get("measurements")
        if isinstance(inline_rows, list):
            combined_rows = DashboardHandler.merge_measurements(combined_rows, inline_rows)

        checkpoint_path = None
        if seed:
            checkpoint_path = path.parent / f"checkpoint_{seed}.json"
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_data, checkpoint_err = DashboardHandler.read_json(checkpoint_path)
            if isinstance(checkpoint_data, dict):
                cp_rows = checkpoint_data.get("measurements")
                if isinstance(cp_rows, list):
                    combined_rows = DashboardHandler.merge_measurements(combined_rows, cp_rows)
            if checkpoint_err and not err:
                err = checkpoint_err

        if jsonl_path and jsonl_path.exists():
            meas, meas_err = DashboardHandler.read_jsonl(jsonl_path)
            if isinstance(meas, list):
                combined_rows = DashboardHandler.merge_measurements(combined_rows, meas)
            if meas_err and not err:
                err = meas_err

        if isinstance(combined_rows, list) and len(combined_rows) > 0:
            out["measurements"] = combined_rows
        return out, err

    @staticmethod
    def load_live_combined(path: Path):
        """Load live data from all seeds in the group, combining measurements."""
        data, err = DashboardHandler.load_live_with_jsonl(path)
        if data is None:
            return data, err
        seed = ((data.get("meta") or {}).get("seed") or "").strip()
        if not seed:
            seed = DashboardHandler.infer_seed_from_live_path(path)
        if not seed:
            return data, err

        group = DashboardHandler.seed_group(seed)
        sibling_seeds = [s for s in group if s != seed]
        if not sibling_seeds:
            return data, err

        all_meas = list(data.get("measurements") or [])
        seeds_used = [seed]
        for sib in sibling_seeds:
            sib_path = path.parent / f"live_{sib}.json"
            if not sib_path.exists():
                continue
            sib_data, _ = DashboardHandler.load_live_with_jsonl(sib_path)
            if not isinstance(sib_data, dict):
                continue
            sib_meas = sib_data.get("measurements") or []
            if sib_meas:
                # Tag each measurement with its seed so dedup doesn't
                # drop measurements with the same idx from different seeds.
                for i, m in enumerate(sib_meas):
                    if isinstance(m, dict):
                        sib_meas[i] = {**m, "_seed": sib}
                all_meas.extend(sib_meas)
                seeds_used.append(sib)

        out = dict(data)
        out["measurements"] = all_meas
        meta = dict(out.get("meta") or {})
        meta["combined_seeds"] = seeds_used
        meta["combined_n_configs"] = len(all_meas)
        out["meta"] = meta
        return out, err

    @staticmethod
    def with_live_elapsed(progress, progress_mtime):
        if not isinstance(progress, dict):
            return progress
        out = dict(progress)
        phase = str(out.get("phase") or "").strip().lower()
        if out.get("done") or phase in DashboardHandler.terminal_phases:
            return out
        if not isinstance(progress_mtime, (int, float)) or progress_mtime < 0:
            return out
        base_elapsed = out.get("elapsed_sec")
        if not isinstance(base_elapsed, (int, float)):
            return out
        live_elapsed = float(base_elapsed) + max(0.0, time.time() - float(progress_mtime))
        out["elapsed_sec"] = live_elapsed
        out["elapsed_sec_live"] = live_elapsed

        sweeps_done = out.get("sweeps_done")
        total_sweeps = out.get("total_sweeps")
        if isinstance(sweeps_done, (int, float)) and isinstance(total_sweeps, (int, float)):
            if sweeps_done > 0 and total_sweeps > sweeps_done and live_elapsed > 0:
                sec_per_sweep = live_elapsed / float(sweeps_done)
                if sec_per_sweep > 0:
                    out["eta_sec"] = max(0.0, (float(total_sweeps) - float(sweeps_done)) * sec_per_sweep)
        return out

    def handle_events(self, parsed):
        # Dead-client guard: without a socket timeout, a vanished peer (sleeping
        # laptop, dropped VPN) leaves this thread blocked in wfile.write holding
        # a multi-MB payload while the browser's EventSource reconnect spawns
        # another thread — threads + pinned buffers accumulate for the whole
        # run. A timeout makes the write raise and the thread exit.
        try:
            self.connection.settimeout(30)
        except Exception:
            pass
        qs = parse_qs(parsed.query)
        progress_arg = qs.get("progress", ["/results/su2_signal_scan/progress_petrus-su2-signal.json"])[0]
        live_arg = qs.get("live", ["/results/su2_signal_scan/live_petrus-su2-signal.json"])[0]
        combine = qs.get("combine", ["0"])[0].strip() in ("1", "true", "yes")
        try:
            progress_path = self.resolve_local_path(progress_arg)
            live_path = self.resolve_local_path(live_arg)
        except ValueError as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        # Resolve sibling JSONL paths for combined-mode file watching.
        sibling_jsonl_paths = []
        if combine:
            seed = self.infer_seed_from_live_path(live_path)
            if seed:
                for sib in self.seed_group(seed):
                    if sib == seed:
                        continue
                    sib_jsonl = live_path.parent / f"live_{sib}.jsonl"
                    sibling_jsonl_paths.append(sib_jsonl)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        last_sig = None
        last_ping = 0.0
        last_emit = 0.0
        try:
            while True:
                now = time.time()
                pm_sig = DashboardHandler.file_sig(progress_path)
                lm_sig = DashboardHandler.file_sig(live_path)
                jsonl_path = live_path.with_suffix(".jsonl")
                jm_sig = DashboardHandler.file_sig(jsonl_path)
                sib_sigs = tuple(DashboardHandler.file_sig(p) for p in sibling_jsonl_paths)
                sig = (pm_sig, lm_sig, jm_sig, sib_sigs)

                should_emit = sig != last_sig
                if not should_emit and self.heartbeat_interval > 0:
                    should_emit = (now - last_emit) >= self.heartbeat_interval

                if should_emit:
                    progress, progress_err = self.read_json(progress_path)
                    if combine:
                        live, live_err = self.load_live_combined(live_path)
                    else:
                        live, live_err = self.load_live_with_jsonl(live_path)
                    pm = (pm_sig[0] / 1e9) if pm_sig[0] > 0 else -1
                    progress = self.with_live_elapsed(progress, pm)
                    seed = ""
                    if isinstance(progress, dict):
                        seed = str(progress.get("seed") or "").strip()
                    if not seed:
                        seed = DashboardHandler.infer_seed_from_progress_path(progress_path)
                    root_seed = DashboardHandler.base_seed(seed)
                    telemetry_payload = (
                        DashboardHandler.collect_thread_telemetry(root_seed)
                        if root_seed
                        else {"threads": {}}
                    )
                    payload = {
                        "ts": now,
                        "progress": progress,
                        "live": live,
                        "thread_telemetry": telemetry_payload.get("threads", {}),
                        "errors": {},
                    }
                    if sig == last_sig:
                        payload["heartbeat"] = True
                    if progress_err:
                        payload["errors"]["progress"] = progress_err
                    if live_err:
                        payload["errors"]["live"] = live_err
                    msg = f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                    last_sig = sig
                    last_ping = now
                    last_emit = now
                elif now - last_ping >= 20:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    last_ping = now

                time.sleep(self.poll_interval)
        except (BrokenPipeError, ConnectionResetError):
            return

    def handle_thread_telemetry(self, parsed):
        qs = parse_qs(parsed.query)
        base_seed = (qs.get("base_seed", [""])[0] or "").strip()
        if not base_seed:
            progress_arg = (qs.get("progress", [""])[0] or "").strip()
            if progress_arg:
                try:
                    progress_path = self.resolve_local_path(progress_arg)
                except Exception:
                    progress_path = None
                if progress_path is not None:
                    seed = DashboardHandler.infer_seed_from_progress_path(progress_path)
                    base_seed = DashboardHandler.base_seed(seed)
        if not base_seed:
            base_seed = DashboardHandler.base_seed("petrus-su2-signal")

        payload = DashboardHandler.collect_thread_telemetry(base_seed)
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_thread_control(self, parsed):
        clen = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(clen) if clen > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"invalid json"}')
            return

        if not self.is_authorized(parsed=parsed, payload=payload):
            self.send_unauthorized()
            return

        action = str(payload.get("action") or "").strip().lower()
        seed = str(payload.get("seed") or "").strip()
        base_seed = str(payload.get("base_seed") or "").strip()
        pipeline = str(payload.get("pipeline") or "auto").strip().lower()
        if pipeline not in {"auto", "cpu", "gpu"}:
            pipeline = "auto"
        all_actions = {"pause_all", "resume_all", "restart_all"}
        single_actions = {"pause", "resume", "restart"}
        if action not in (all_actions | single_actions):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"action must be pause|resume|restart|pause_all|resume_all|restart_all"}')
            return
        if action in single_actions and not seed:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"seed is required"}')
            return
        root = DashboardHandler.base_seed(base_seed or seed)
        if not root:
            root = DashboardHandler.base_seed("petrus-su2-signal")

        def run_single(single_action: str, one_seed: str, pipeline_choice: str = "auto"):
            rec = DashboardHandler.find_worker_record(one_seed)
            pid = int(rec.get("pid")) if isinstance(rec, dict) and rec.get("pid") else None
            cmd_hint = str(rec.get("command") or "") if isinstance(rec, dict) else ""
            out = {
                "seed": one_seed,
                "pid_before": pid,
                "ok": True,
            }
            if single_action == "pause":
                if not pid:
                    raise RuntimeError(f"{one_seed}: worker not running")
                os.kill(pid, signal.SIGSTOP)
                out["message"] = f"{one_seed}: paused pid {pid}"
                return out
            if single_action == "resume":
                if not pid:
                    raise RuntimeError(f"{one_seed}: worker not running")
                os.kill(pid, signal.SIGCONT)
                out["message"] = f"{one_seed}: resumed pid {pid}"
                return out
            if single_action == "restart":
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    deadline = time.time() + 3.0
                    while time.time() < deadline and DashboardHandler.process_alive(pid):
                        time.sleep(0.1)
                    if DashboardHandler.process_alive(pid):
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                launched = DashboardHandler.launch_worker(one_seed, cmd_hint, pipeline=pipeline_choice)
                out["pid_after"] = launched.get("pid")
                out["launch_method"] = launched.get("method")
                out["launch_command"] = launched.get("command")
                out["pipeline_requested"] = launched.get("pipeline_requested")
                out["pipeline_effective"] = launched.get("pipeline_effective")
                out["message"] = f"{one_seed}: restarted"
                return out
            raise RuntimeError(f"unsupported action: {single_action}")

        try:
            with DashboardHandler.control_lock:
                if action in single_actions:
                    result = run_single(action, seed, pipeline)
                    response = {
                        "ok": True,
                        "action": action,
                        "seed": seed,
                        "pipeline": pipeline,
                        **result,
                    }
                else:
                    mapped = {
                        "pause_all": "pause",
                        "resume_all": "resume",
                        "restart_all": "restart",
                    }
                    single = mapped[action]
                    seeds = DashboardHandler.seed_group(root)
                    details = []
                    for s in seeds:
                        try:
                            details.append(run_single(single, s, pipeline))
                        except Exception as inner:
                            details.append({
                                "seed": s,
                                "ok": False,
                                "error": str(inner),
                            })
                    ok_count = sum(1 for d in details if d.get("ok"))
                    response = {
                        "ok": ok_count > 0,
                        "action": action,
                        "base_seed": root,
                        "pipeline": pipeline,
                        "details": details,
                        "message": f"{action} done: {ok_count}/{len(details)} threads",
                    }

            DashboardHandler.invalidate_telemetry_cache(root)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))
        except Exception as e:
            DashboardHandler.invalidate_telemetry_cache(root)
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "ok": False,
                        "action": action,
                        "seed": seed,
                        "error": str(e),
                    }
                ).encode("utf-8")
            )

    def handle_chat(self, parsed):
        clen = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(clen) if clen > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"invalid json"}')
            return

        if not self.is_authorized(parsed=parsed, payload=payload):
            self.send_unauthorized()
            return

        messages = payload.get("messages", [])
        if not isinstance(messages, list) or len(messages) == 0:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"messages must be a non-empty list"}')
            return

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "error": "OPENAI_API_KEY is not set for dashboard server; chat endpoint unavailable."
                    }
                ).encode("utf-8")
            )
            return

        system = {
            "role": "system",
            "content": (
                "You are an admin helper for SU(2) lattice runs. "
                "Answer concisely and prioritize run status, interpretation, and next actions."
            ),
        }
        chat_messages = [system]
        for m in messages[-20:]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                chat_messages.append({"role": role, "content": content})

        req_payload = {
            "model": self.chat_model,
            "messages": chat_messages,
            "temperature": 0.2,
        }
        req = Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(req_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=60) as r:
                out = json.loads(r.read().decode("utf-8"))
            reply = (
                out.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"reply": reply}).encode("utf-8"))
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": f"upstream http error: {e.code}", "detail": detail}).encode("utf-8")
            )
        except URLError as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"network error: {e.reason}"}).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


def main():
    default_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="SU2 dashboard static+SSE server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--root", default=str(default_root))
    p.add_argument("--poll-ms", type=int, default=500)
    p.add_argument(
        "--heartbeat-sec",
        type=float,
        default=float(os.environ.get("SU2_DASHBOARD_HEARTBEAT_SEC", "10")),
        help="Emit SSE snapshots periodically even if files are unchanged.",
    )
    p.add_argument("--chat-model", default=os.environ.get("SU2_DASHBOARD_CHAT_MODEL", "gpt-4o-mini"))
    p.add_argument(
        "--allowed-tailscale-login",
        default=os.environ.get("SU2_DASHBOARD_ALLOWED_TS_LOGIN", os.environ.get("SU2_DASHBOARD_ALLOWED_USER", "")),
        help="Allow requests with matching Tailscale user login header (email).",
    )
    p.add_argument(
        "--auth-token",
        default=os.environ.get("SU2_DASHBOARD_AUTH_TOKEN", ""),
        help="Shared token (Authorization: Bearer <token> or X-Auth-Token; ?token=... only on /events).",
    )
    p.add_argument(
        "--trusted-proxy",
        action="append",
        default=[x.strip() for x in os.environ.get("SU2_DASHBOARD_TRUSTED_PROXIES", "").split(",") if x.strip()],
        metavar="IP_OR_CIDR",
        help="Peer allowed to assert Tailscale identity headers, in addition to loopback (repeatable).",
    )
    p.add_argument(
        "--cors-origin",
        default=os.environ.get("SU2_DASHBOARD_CORS_ORIGIN", ""),
        help="Optional CORS allow origin (for external dashboard UI).",
    )
    p.add_argument(
        "--no-protect-results",
        action="store_true",
        help="Do not require auth for /results/* files.",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")
    # Catch the dangling-symlink case (e.g. results/ -> external drive that is
    # not currently mounted): exists() / is_dir() can both pass on the symlink
    # itself while the contents are unreachable. Try a real directory read so
    # the operator sees the failure at startup rather than as silent empty
    # data later.
    try:
        next(iter(os.scandir(root)), None)
    except OSError as exc:
        raise SystemExit(f"Root is not readable: {root}: {exc}")

    DashboardHandler.root = root
    DashboardHandler.poll_interval = max(0.05, args.poll_ms / 1000.0)
    DashboardHandler.heartbeat_interval = max(0.0, float(args.heartbeat_sec))
    DashboardHandler.chat_model = args.chat_model
    DashboardHandler.allowed_tailscale_login = args.allowed_tailscale_login
    DashboardHandler.auth_token = args.auth_token
    DashboardHandler.trusted_proxies = tuple(args.trusted_proxy or ())
    DashboardHandler.cors_origin = args.cors_origin
    DashboardHandler.protect_results = not args.no_protect_results

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving {root} at http://{args.host}:{args.port}")
    print("SSE endpoint: /events")
    print(f"Chat endpoint: /chat (model={DashboardHandler.chat_model})")
    if DashboardHandler.allowed_tailscale_login or DashboardHandler.auth_token:
        print("Auth enabled:")
        if DashboardHandler.allowed_tailscale_login:
            print(f"  - Tailscale login allowlist: {DashboardHandler.allowed_tailscale_login}")
            proxies = ", ".join(("loopback",) + tuple(DashboardHandler.trusted_proxies))
            print(f"  - Identity headers trusted only from: {proxies}")
        if DashboardHandler.auth_token:
            print("  - Shared auth token: enabled")
        print(f"  - Protect /results/*: {DashboardHandler.protect_results}")
    else:
        print("\n" + "="*50)
        print("WARNING: Auth is DISABLED (no allowlist or token configured).")
        print("Control endpoints (/thread_control) are open to anyone who can reach this host!")
        print("Recommend binding only to 127.0.0.1 or setting SU2_DASHBOARD_AUTH_TOKEN.")
        print("="*50 + "\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
