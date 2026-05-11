#!/usr/bin/env python3
"""
Qwen3.5-122B-A10B-BF16 mac-port training monitor TUI.

Tails the mlx_lm.lora train log + watchdog memcsv, renders a live dashboard
with rich: progress bar, iter/ETA, loss sparkline, val trajectory, memory
(RSS/swap/peak Metal), rate (it/s, tok/s, LR), health checks, recent log.

Run on Studio:
    .venv/bin/python tools/train_monitor_tui.py

Or from GrosMac with a TTY:
    ssh -t studio "cd ailiance-mac-tuner && .venv/bin/python tools/train_monitor_tui.py"

Options:
    --log PATH       explicit train log (default: latest train-*.log under --root)
    --memcsv PATH    explicit memcsv (default: latest memcsv-*.csv under --root)
    --root DIR       logs/122b-macport base dir
    --total-iters N  total iters for progress bar (default 3000)
    --refresh SEC    UI refresh interval (default 2.0)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import queue
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

try:
    from rich.align import Align
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
except ImportError:
    sys.stderr.write("rich missing. Install in the venv: pip install rich\n")
    sys.exit(1)


ITER_RE = re.compile(
    r"Iter\s+(\d+):\s+Train loss\s+([\d.]+)"
    r"(?:.*?Learning Rate\s+([\d.eE+-]+))?"
    r"(?:.*?It/sec\s+([\d.]+))?"
    r"(?:.*?Tokens/sec\s+([\d.]+))?"
    r"(?:.*?Trained Tokens\s+(\d+))?"
    r"(?:.*?Peak mem\s+([\d.]+)\s*GB)?"
)
VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")
SAVE_RE = re.compile(r"Sav(?:ed|ing).*adapter", re.IGNORECASE)
WARM_DONE = 100   # phase switch warmup→training


class TrainState:
    def __init__(self, total_iters: int):
        self.total_iters = total_iters
        self.iter = 0
        self.train_loss: float | None = None
        self.val_loss: float | None = None
        self.best_val: float | None = None
        self.lr: float | None = None
        self.it_per_sec: float | None = None
        self.tok_per_sec: float | None = None
        self.peak_mem_gb: float | None = None
        self.train_hist: deque = deque(maxlen=200)
        self.val_hist: deque = deque(maxlen=50)
        self.log_lines: deque = deque(maxlen=12)
        self.start_time: float | None = None
        self.last_update: float | None = None
        self.last_save_iter = 0
        self.phase = "waiting"

    def ingest(self, line: str) -> None:
        line = line.rstrip("\n")
        if not line.strip():
            return
        self.log_lines.append((time.strftime("%H:%M:%S"), line))
        m = ITER_RE.search(line)
        if m:
            self.iter = int(m.group(1))
            self.train_loss = float(m.group(2))
            if m.group(3):
                self.lr = float(m.group(3))
            if m.group(4):
                self.it_per_sec = float(m.group(4))
            if m.group(5):
                self.tok_per_sec = float(m.group(5))
            if m.group(7):
                self.peak_mem_gb = float(m.group(7))
            self.train_hist.append((self.iter, self.train_loss))
            if self.start_time is None:
                self.start_time = time.time()
            self.last_update = time.time()
            self.phase = "warmup" if self.iter < WARM_DONE else "training"
            return
        m = VAL_RE.search(line)
        if m:
            it = int(m.group(1))
            vl = float(m.group(2))
            self.val_loss = vl
            self.val_hist.append((it, vl))
            if self.best_val is None or vl < self.best_val:
                self.best_val = vl
            self.phase = "validating"
            return
        if SAVE_RE.search(line):
            self.last_save_iter = self.iter
            self.phase = "saving"
            return


def sparkline(values: list[float], width: int = 50) -> str:
    if not values:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1e-9
    chars = "▁▂▃▄▅▆▇█"
    return "".join(chars[int((v - lo) / span * (len(chars) - 1))] for v in vals)


def bar(pct: float, width: int = 30) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled)


def tail_worker(path: Path, q: "queue.Queue[str]", stop: threading.Event) -> None:
    """Background tail -F. Handles rotation by reopening on stat change."""
    while not stop.is_set():
        try:
            with path.open("r", errors="replace") as f:
                f.seek(0, os.SEEK_END)
                inode = os.fstat(f.fileno()).st_ino
                while not stop.is_set():
                    line = f.readline()
                    if line:
                        q.put(line)
                        continue
                    # idle: check rotation
                    try:
                        if os.stat(path).st_ino != inode:
                            break
                    except FileNotFoundError:
                        break
                    time.sleep(0.2)
        except FileNotFoundError:
            time.sleep(1.0)


def read_memcsv_tail(path: Path | None, n: int = 3) -> list[dict]:
    if path is None or not path.exists():
        return []
    try:
        with path.open() as f:
            rows = list(csv.DictReader(f))
        return rows[-n:]
    except Exception:
        return []


def find_latest(pattern: str) -> Path | None:
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None


def phase_style(phase: str) -> str:
    return {
        "waiting": "grey58",
        "warmup": "yellow",
        "training": "green",
        "validating": "cyan",
        "saving": "magenta",
        "done": "bold green",
    }.get(phase, "white")


def build_layout(state: TrainState, mem_tail: list[dict]) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=10),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="progress", size=7),
        Layout(name="memory", size=8),
        Layout(name="rate", size=6),
    )
    layout["right"].split_column(
        Layout(name="loss"),
        Layout(name="quality", size=9),
    )

    # header
    now = datetime.now().strftime("%H:%M:%S")
    header_txt = Text()
    header_txt.append("Qwen3.5-122B-A10B-BF16  ·  mac-port  ·  Studio", style="bold")
    header_txt.append("   phase=", style="dim")
    header_txt.append(state.phase, style=phase_style(state.phase))
    header_txt.append(f"   {now}", style="dim")
    layout["header"].update(Panel(Align.center(header_txt), box=box.HEAVY))

    # progress
    pct = (state.iter / state.total_iters * 100) if state.total_iters else 0
    eta_str = "?"
    if state.it_per_sec and state.iter > 0 and state.iter < state.total_iters:
        secs = (state.total_iters - state.iter) / max(state.it_per_sec, 1e-6)
        eta_str = str(timedelta(seconds=int(secs)))
    elapsed = "?"
    if state.start_time:
        elapsed = str(timedelta(seconds=int(time.time() - state.start_time)))
    g = Table.grid(padding=(0, 1))
    g.add_row(Text(bar(pct), style="bold green"))
    g.add_row(Text(f"Iter {state.iter}/{state.total_iters}   {pct:5.1f}%", style="bold"))
    g.add_row(Text(f"elapsed {elapsed}   ETA {eta_str}", style="dim"))
    layout["progress"].update(Panel(g, title="Progression", box=box.ROUNDED))

    # memory
    rss = swap = mp = None
    if mem_tail:
        last = mem_tail[-1]
        try:
            rss = float(last.get("rss_gb") or "nan")
        except Exception:
            pass
        try:
            swap = float(last.get("swap_used_gb") or "nan")
        except Exception:
            pass
        mp = last.get("mem_pressure_free_pct") or ""
    g = Table.grid(padding=(0, 1))
    g.add_row(Text(f"RSS:   {rss:6.1f} GB" if rss is not None else "RSS:      —"))
    g.add_row(Text(f"Swap:  {swap:6.1f} GB" if swap is not None else "Swap:     —",
                   style="yellow" if (swap and swap > 50) else "white"))
    g.add_row(Text(f"Peak:  {state.peak_mem_gb:6.1f} GB" if state.peak_mem_gb else "Peak:     —",
                   style="yellow" if (state.peak_mem_gb and state.peak_mem_gb > 400) else "white"))
    g.add_row(Text(f"Free:  {mp}%" if mp else "Free:     —"))
    layout["memory"].update(Panel(g, title="Mémoire (Studio 512 GB)", box=box.ROUNDED))

    # rate
    g = Table.grid(padding=(0, 1))
    g.add_row(Text(f"{state.it_per_sec:.2f} it/s" if state.it_per_sec else "— it/s"))
    g.add_row(Text(f"{state.tok_per_sec:.0f} tok/s" if state.tok_per_sec else "— tok/s"))
    g.add_row(Text(f"LR {state.lr:.2e}" if state.lr else "LR —"))
    layout["rate"].update(Panel(g, title="Débit", box=box.ROUNDED))

    # loss
    losses = [v for _, v in state.train_hist]
    spk = sparkline(losses, width=56)
    body = Text()
    if state.train_loss is not None:
        trend = ""
        if len(losses) >= 20:
            recent = losses[-10:]
            prev = losses[-20:-10]
            d = sum(recent) / len(recent) - sum(prev) / len(prev)
            trend = f"  Δ10={d:+.4f}"
        body.append(f"Train loss: {state.train_loss:.4f}{trend}\n", style="green")
    body.append(spk + "\n\n", style="green")
    if state.val_hist:
        vl_seq = "  ".join(f"i{it}→{v:.3f}" for it, v in list(state.val_hist)[-6:])
        body.append("Val trajectory:\n", style="cyan")
        body.append(vl_seq + "\n", style="cyan")
        if state.best_val is not None:
            body.append(f"Best val: {state.best_val:.4f}\n", style="bold cyan")
    else:
        body.append("Val loss: (première éval à iter 250)\n", style="dim")
    layout["loss"].update(Panel(body, title="Loss · raisonnement", box=box.ROUNDED))

    # quality / health
    msgs: list[tuple[str, str]] = []
    if state.train_loss is None:
        msgs.append(("• En attente du premier log train…", "grey58"))
    else:
        if state.train_loss != state.train_loss:  # NaN
            msgs.append(("✗ loss = NaN — STOP, réduire LR", "bold red"))
        elif state.train_loss > 10:
            msgs.append(("⚠ loss diverge (>10) — vérifier LR", "red"))
        elif state.train_loss < 1.5:
            msgs.append(("✓ loss dans la zone cible", "green"))
        else:
            msgs.append(("· loss en convergence", "white"))
    if state.peak_mem_gb:
        if state.peak_mem_gb > 420:
            msgs.append((f"⚠ peak {state.peak_mem_gb:.0f} GB proche limite wired", "yellow"))
        else:
            msgs.append((f"✓ mémoire sous contrôle ({state.peak_mem_gb:.0f} GB)", "green"))
    if swap is not None:
        if swap > 80:
            msgs.append((f"⚠ swap thrash {swap:.0f} GB (watchdog kill à 80 sustained)", "red"))
        elif swap > 20:
            msgs.append((f"· paging actif {swap:.0f} GB", "yellow"))
        else:
            msgs.append(("✓ swap calme", "green"))
    if state.last_save_iter:
        msgs.append((f"✓ dernier checkpoint : iter {state.last_save_iter}", "cyan"))
    if state.iter >= state.total_iters > 0:
        msgs.append(("🎉 training terminé", "bold green"))

    qtxt = Text()
    for m_, style in msgs:
        qtxt.append(m_ + "\n", style=style)
    layout["quality"].update(Panel(qtxt, title="Qualité & santé", box=box.ROUNDED))

    # footer: recent log
    if state.log_lines:
        foot = Text()
        for ts, ln in list(state.log_lines)[-8:]:
            foot.append(f"[{ts}] ", style="dim")
            foot.append(ln[:220] + "\n")
    else:
        foot = Text("En attente du premier log…", style="dim")
    layout["footer"].update(Panel(foot, title="Log stream", box=box.ROUNDED))
    return layout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=None)
    ap.add_argument("--memcsv", default=None)
    ap.add_argument("--root", default="/Users/clems/ailiance-mac-tuner/logs/122b-macport")
    ap.add_argument("--total-iters", type=int, default=3000)
    ap.add_argument("--refresh", type=float, default=2.0)
    args = ap.parse_args()

    root = Path(args.root)
    console = Console()

    # Wait for log file to appear if not present yet
    log_path = Path(args.log) if args.log else find_latest(str(root / "train-*.log"))
    while log_path is None or not log_path.exists():
        console.print(f"[yellow]En attente d'un train log sous[/] [cyan]{root}[/] (scan toutes les 5s). Ctrl-C pour quitter.")
        time.sleep(5)
        log_path = Path(args.log) if args.log else find_latest(str(root / "train-*.log"))

    memcsv_path = Path(args.memcsv) if args.memcsv else find_latest(str(root / "memcsv-*.csv"))

    console.print(f"[bold]Monitoring[/] log=[cyan]{log_path}[/]  memcsv=[cyan]{memcsv_path}[/]")

    state = TrainState(total_iters=args.total_iters)
    q: "queue.Queue[str]" = queue.Queue()
    stop = threading.Event()
    t = threading.Thread(target=tail_worker, args=(log_path, q, stop), daemon=True)
    t.start()

    last_mem_refresh = 0.0
    mem_tail: list[dict] = []

    try:
        with Live(build_layout(state, mem_tail), refresh_per_second=max(1.0, 1.0 / args.refresh), screen=True) as live:
            while True:
                drained = 0
                while drained < 200:
                    try:
                        line = q.get_nowait()
                    except queue.Empty:
                        break
                    state.ingest(line)
                    drained += 1

                now = time.time()
                if now - last_mem_refresh > args.refresh:
                    mem_tail = read_memcsv_tail(memcsv_path)
                    last_mem_refresh = now

                live.update(build_layout(state, mem_tail))
                time.sleep(args.refresh)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        console.print("\n[bold]Moniteur arrêté.[/]")


if __name__ == "__main__":
    main()
