#!/usr/bin/env python3
"""
resource_runner_pavia.py
Merged runner:
 - in-process exec like %run (preserves inline figures) or subprocess fallback
 - GPU metrics: nvidia-smi -> pynvml -> torch.cuda
 - robust summarizer and CSV logging (Duration, Avg GPU Power, Avg GPU Memory, Avg RAM, Energy)
"""

import os
import time
import csv
import psutil
import subprocess
import threading
from datetime import datetime
import shlex
import sys

# plotting / IPython helpers (optional)
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except Exception:
    plt = None
    Figure = None

try:
    from IPython.display import display
except Exception:
    display = None

# ---------------- GPU helpers ----------------

def _get_gpu_metrics():
    """
    Return (power_W_or_None, gpu_mem_mib_or_None, torch_proc_gpu_mem_mib_or_None)
    Attempts: nvidia-smi -> pynvml -> torch.cuda
    """
    # 1) nvidia-smi cli
    try:
        out = subprocess.check_output(
            ['nvidia-smi','--query-gpu=power.draw,memory.used','--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if out:
            total_p, total_m, parsed = 0.0, 0.0, False
            for line in out.splitlines():
                parts = [x.strip() for x in line.split(',')]
                if len(parts) < 2:
                    continue
                p_str, m_str = parts[0], parts[1]
                p = float(p_str) if (p_str not in ('N/A','')) else 0.0
                m = float(m_str) if (m_str not in ('N/A','')) else 0.0
                total_p += p; total_m += m; parsed = True
            if parsed:
                return total_p, total_m, None
    except Exception:
        pass

    # 2) pynvml
    try:
        from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetMemoryInfo
        nvmlInit()
        cnt = nvmlDeviceGetCount()
        if cnt > 0:
            total_p, total_m = 0.0, 0.0
            for i in range(cnt):
                h = nvmlDeviceGetHandleByIndex(i)
                try:
                    p_mw = nvmlDeviceGetPowerUsage(h)
                    p_w = float(p_mw) / 1000.0
                except Exception:
                    p_w = 0.0
                try:
                    mem = nvmlDeviceGetMemoryInfo(h)
                    mem_used_mib = float(mem.used) / (1024*1024)
                except Exception:
                    mem_used_mib = 0.0
                total_p += p_w
                total_m += mem_used_mib
            return total_p, total_m, None
    except Exception:
        pass

    # 3) torch.cuda process memory (no power)
    try:
        import torch
        if torch.cuda.is_available():
            # memory_allocated returns bytes for current device (we report MiB)
            torch_mem = float(torch.cuda.memory_allocated(0)) / (1024*1024)
            return None, None, torch_mem
    except Exception:
        pass

    return None, None, None

def _proc_ram_mb():
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
    except Exception:
        return None

# ---------------- Resource Monitor ----------------

class ResourceMonitor:
    """
    Samples GPU power/mem and process RAM at interval seconds.
    Samples saved to self.samples and optional timeseries CSV.
    sample tuple may be:
      (wall, t_rel, power_W, gpu_mem_MiB, torch_proc_gpu_mem_MiB, ram_MiB)
    or (wall, t_rel, power_W, gpu_mem_MiB, ram_MiB)
    """
    def __init__(self, interval=1.0, ts_csv=None):
        self.interval = float(interval)
        self.ts_csv = ts_csv
        self.samples = []
        self._running = False

    def _ensure_ts_header(self):
        if not self.ts_csv:
            return
        if not os.path.exists(self.ts_csv):
            # write units comment then header
            with open(self.ts_csv, "w", newline="") as f:
                f.write("# Units: t_rel_s=seconds, gpu_power_W=Watts (nvidia-smi/pynvml), gpu_mem_MiB=MiB (nvidia-smi/pynvml), torch_proc_gpu_mem_MiB=MiB (torch.cuda.memory_allocated), ram_MiB=MiB\n")
                csv.writer(f).writerow(["timestamp","t_rel_s","gpu_power_W","gpu_mem_MiB","torch_proc_gpu_mem_MiB","ram_MiB"])

    def _loop(self, t0):
        self._ensure_ts_header()
        while self._running:
            wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            t_rel = time.time() - t0

            power_W, gpu_mem_mib, torch_proc_gpu_mem_mib = _get_gpu_metrics()
            ram_MB = _proc_ram_mb()

            # append a normalized 6-tuple (use None for missing slots)
            self.samples.append((wall, t_rel, power_W, gpu_mem_mib, torch_proc_gpu_mem_mib, ram_MB))

            if self.ts_csv:
                with open(self.ts_csv, "a", newline="") as f:
                    csv.writer(f).writerow([
                        wall,
                        round(t_rel, 3),
                        None if power_W is None else round(power_W, 3),
                        None if gpu_mem_mib is None else round(gpu_mem_mib, 3),
                        None if torch_proc_gpu_mem_mib is None else round(torch_proc_gpu_mem_mib, 3),
                        None if ram_MB is None else round(ram_MB, 3)
                    ])

            time.sleep(self.interval)

    def start(self):
        self._running = True
        self._t0 = time.time()
        self._thr = threading.Thread(target=self._loop, args=(self._t0,), daemon=True)
        self._thr.start()

    def stop(self):
        self._running = False
        if hasattr(self, "_thr"):
            try:
                self._thr.join(timeout=5)
            except Exception:
                pass

    def summarize(self):
        """
        Robust summarizer supporting different tuple shapes.
        Returns (avg_power_W, avg_gpu_mem_MiB, avg_ram_MiB, energy_J)
        """
        if not self.samples:
            return None, None, None, None

        # Build numeric columns
        numeric_cols = {}
        max_len = max(len(s) for s in self.samples)
        for idx in range(max_len):
            vals = [s[idx] for s in self.samples if len(s) > idx and isinstance(s[idx], (int, float))]
            if vals:
                numeric_cols[idx] = vals

        # avg power: prefer idx 2
        avg_power = None; power_idx = None
        if 2 in numeric_cols:
            avg_power = sum(numeric_cols[2]) / len(numeric_cols[2]); power_idx = 2
        else:
            for k in sorted(numeric_cols.keys()):
                if k > 1:
                    avg_power = sum(numeric_cols[k]) / len(numeric_cols[k]); power_idx = k; break

        # avg gpu mem: prefer idx 3 else next numeric after power
        avg_gpu_mem = None
        if 3 in numeric_cols:
            avg_gpu_mem = sum(numeric_cols[3]) / len(numeric_cols[3])
        else:
            if power_idx is not None and (power_idx + 1) in numeric_cols:
                avg_gpu_mem = sum(numeric_cols[power_idx+1]) / len(numeric_cols[power_idx+1])

        # avg ram: choose last numeric column
        avg_ram = None
        if numeric_cols:
            last_idx = max(numeric_cols.keys())
            avg_ram = sum(numeric_cols[last_idx]) / len(numeric_cols[last_idx])

        # energy: integrate using the detected power column and times (idx 1)
        energy_J = None
        times = [s[1] for s in self.samples if len(s) > 1 and isinstance(s[1], (int,float))]
        if power_idx is not None and len(times) >= 2:
            powers = []
            for s in self.samples:
                if len(s) > power_idx and isinstance(s[power_idx], (int,float)) and isinstance(s[1], (int,float)):
                    powers.append((s[1], float(s[power_idx])))
                else:
                    if len(s) > 1 and isinstance(s[1], (int,float)):
                        powers.append((s[1], 0.0))
            powers.sort(key=lambda x: x[0])
            e = 0.0
            for i in range(1, len(powers)):
                t0, p0 = powers[i-1]
                t1, p1 = powers[i]
                dt = t1 - t0
                if dt > 0:
                    e += 0.5 * (p0 + p1) * dt
            energy_J = e

        return avg_power, avg_gpu_mem, avg_ram, energy_J

# ---------------- Exec helper ----------------

def _exec_python_script_in_process(script_path, script_args=None, exec_globals_extra=None):
    if script_args is None: script_args = []
    if exec_globals_extra is None: exec_globals_extra = {}

    script_path = os.path.abspath(script_path)
    script_dir = os.path.dirname(script_path)

    with open(script_path, "r", encoding="utf-8") as f:
        code = compile(f.read(), script_path, "exec")

    exec_globals = {"__name__": "__main__", "__file__": script_path, "__package__": None}
    exec_globals.update({k: v for k, v in globals().items() if k not in exec_globals})
    exec_globals.update(exec_globals_extra)

    old_cwd = os.getcwd()
    old_argv = sys.argv.copy()

    try:
        if script_dir:
            os.chdir(script_dir)
        sys.argv = [script_path] + list(script_args)
        exec(code, exec_globals, exec_globals)
    finally:
        try:
            os.chdir(old_cwd)
        except Exception:
            pass
        sys.argv = old_argv

    # Try to show figures inline (best-effort)
    try:
        if plt is not None and plt.get_fignums():
            try:
                plt.show()
            except Exception:
                if display is not None:
                    for num in plt.get_fignums():
                        fig = plt.figure(num)
                        try:
                            display(fig)
                        except Exception:
                            pass
    except Exception:
        pass

# ---------------- Top-level runner ----------------

def _ensure_summary_header(path):
    if not path:
        return
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            # Units comment
            f.write("# Units: Duration = seconds, Avg GPU Power = Watts, Avg GPU Memory = MiB, Avg RAM = MiB, Energy = Joules\n")
            csv.writer(f).writerow(["timestamp","command",
                                    "Duration",
                                    "Avg GPU Power",
                                    "Avg GPU Memory",
                                    "Avg RAM",
                                    "Energy (est., J)",
                                    "samples"])

def run_with_monitor(command_str,
                     ts_csv=None,
                     summary_csv=None,
                     sample_interval=1.0,
                     run_in_process=False,
                     python_exec_path=None,
                     script_args=None):
    """
    Execute `command_str` while monitoring resources.
    If run_in_process=True and command points to a .py file, it will be exec'd in-process
    (preserving inline figures). Otherwise it runs as a subprocess and streams stdout.
    """
    _ensure_summary_header(summary_csv)

    mon = ResourceMonitor(interval=sample_interval, ts_csv=ts_csv)
    t0 = time.time()
    mon.start()

    try:
        script_path_for_exec = None
        inferred_args = []

        if run_in_process:
            # try to parse the command_str for a .py script and args
            toks = shlex.split(command_str) if isinstance(command_str, str) else []
            if isinstance(command_str, str) and os.path.exists(command_str) and command_str.endswith(".py"):
                script_path_for_exec = os.path.abspath(command_str)
            elif toks and toks[0].endswith(".py"):
                candidate = toks[0]
                if not os.path.isabs(candidate):
                    candidate = os.path.join(os.getcwd(), candidate)
                if os.path.exists(candidate):
                    script_path_for_exec = os.path.abspath(candidate)
                    inferred_args = toks[1:]
        # prefer explicit script_args param
        if script_args is not None and isinstance(command_str, str) and os.path.exists(command_str):
            script_path_for_exec = os.path.abspath(command_str)
            inferred_args = list(script_args)

        if script_path_for_exec:
            # inject helpful libs (h5py, einops) if available
            extras = {}
            try:
                import h5py
                extras['h5py'] = h5py
            except Exception:
                pass
            try:
                import einops
                extras['einops'] = einops
            except Exception:
                pass

            final_args = list(script_args) if (script_args is not None) else inferred_args
            print(f"[INFO] Executing {script_path_for_exec} in-process (inline plots enabled).")
            _exec_python_script_in_process(script_path_for_exec, script_args=final_args, exec_globals_extra=extras)

        else:
            # subprocess path
            cmd_list = shlex.split(command_str) if isinstance(command_str, str) else command_str
            if python_exec_path and isinstance(cmd_list, list) and cmd_list[0].endswith("python"):
                cmd_list[0] = python_exec_path

            print(f"[INFO] Running command with monitoring: {command_str}")
            proc = subprocess.Popen(
                cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            try:
                for line in proc.stdout:
                    print(line.rstrip())
                proc.wait()
            except KeyboardInterrupt:
                print("[INFO] Interrupted by user, terminating subprocess...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

    except SystemExit:
        pass
    except Exception as e:
        print(f"[ERROR] Script raised: {e.__class__.__name__}: {e}")
    finally:
        mon.stop()

    duration = time.time() - t0
    avg_power, avg_gpu_mem, avg_ram, energy = mon.summarize()

    if summary_csv:
        write_header = not os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp","command",
                            "Duration",
                            "Avg GPU Power",
                            "Avg GPU Memory",
                            "Avg RAM",
                            "Energy (est., J)",
                            "samples"])
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                command_str,
                round(duration, 3),
                None if avg_power is None else round(avg_power, 3),
                None if avg_gpu_mem is None else round(avg_gpu_mem, 3),
                None if avg_ram is None else round(avg_ram, 3),
                None if energy is None else round(energy, 3),
                len(mon.samples)
            ])

    # friendly output
    print("\n=== RUN COMPLETE ===")
    print(f"Duration             : {duration:.2f} s")
    print(f"Avg GPU Power        : {avg_power if avg_power is not None else 'N/A'} W")
    print(f"Avg GPU Memory      : {avg_gpu_mem if avg_gpu_mem is not None else 'N/A'} MiB")
    print(f"Avg RAM             : {avg_ram if avg_ram is not None else 'N/A'} MiB")
    print(f"Energy (est., J)    : {energy if energy is not None else 'N/A'} J")
    print(f"Timeseries CSV       : {ts_csv}")
    print(f"Summary CSV         : {summary_csv}")
