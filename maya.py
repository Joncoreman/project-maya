#!/usr/bin/env python3
"""Project Maya (working name) - set up and start GLM-5.3-Flash on your own NVIDIA GPU(s).  Linux.

    ./maya.sh                 the first run sets everything up and starts the dashboard; later runs just start it
    ./maya.sh --setup         set up again (other GPUs, another context length, another model folder)
    ./maya.sh --check         only check this PC

./maya.sh makes the private Python environment (.venv, the way Strata's setup.sh does) and runs this file.  It
reuses Strata's installer (setup.py, imported unchanged) for the PC checks, pip, llama.cpp's source and resumable
downloads.

What the first run does (each step is skipped when it is already done):

  1. checks the PC: NVIDIA GPU(s) of compute capability 7.0+, driver, CUDA toolkit (nvcc), g++, CMake, RAM, CPU
  2. asks: which GPUs (one, or two that split the layers), how much context
  3. Python packages into .venv, llama.cpp's source at the pinned commit (it lists them and asks first)
  4. compiles the engine (`build/strata`) for your GPU(s): 10-30 minutes, once
  5. the model: GGUF files you already have (--gguf-dir), or a download it shows you first - the exact commands
     and the size - and starts only after you answer y (or pass --download-model)
  6. builds the pack (the engine's index of the GGUF files) inside the model folder
  7. writes maya-<model>.json and run-maya-<model>.sh, and starts the dashboard on http://127.0.0.1:8080

Nothing is installed system-wide: a missing tool is reported with the command that installs it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
import setup as S  # noqa: E402  Strata's installer: PC checks, pip, llama.cpp, downloads (nothing runs on import)
from setup import ask, fail, ok, run, say, step, warn  # noqa: E402

ROOT = S.ROOT
BUILD = ROOT / "build"
EXE = BUILD / "strata"
STAMP = BUILD / "MAYA-BUILD.json"                  # what the engine in build/ was compiled from and for
MIN_CC = 70                                        # Volta (V100) and newer (the GLM path; see arch_setting)
PY_PACKAGES = [p for p in S.PY_PACKAGES if p != "pillow"]   # pillow is for Strata's image encoder only
CONTEXTS = [8192, 32768, 65536, 131072]
DEFAULT_CONTEXT = 32768
MODEL_NAME = "glm-5.3-flash"
SAMPLING = {"temperature": 1.0, "top_p": 0.95}     # the dashboard's and the API's defaults for requests that set none
EFFORT = "medium"                                  # thinking level for requests that name none
HF = "https://huggingface.co/{repo}/resolve/{revision}/{folder}/{file}"
# The models the installer can download.  Only UD-IQ1_S has been measured end to end (README-MAYA.md); another glm5-next
# GGUF can be used with --gguf-dir (experimental).  Before a release (RELEASE-CHECKLIST.md): "revision" = the
# repository commit the files were checked at (instead of main, which can change), "sha256" = per file name (the
# download is then verified).
MODELS = {
    "UD-IQ1_S": {"about": "Unsloth's 1.6-bit dynamic quant - every published Maya number was measured with it",
                 "repo": "unsloth/GLM-5.3-Flash-GGUF", "revision": "main", "folder": "UD-IQ1_S",
                 "file": "GLM-5.3-Flash-UD-IQ1_S-{i:05d}-of-{n:05d}.gguf", "shards": 3, "download_gb": 93.0,
                 "sha256": {}},
}
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
PACK_FILES = ("index.txt", "native_experts.txt", "dense.bin", "tokenizer/vocab.json", "tokenizer/merges.txt",
              "tokenizer/token_type.json", "tokenizer/chat_template.jinja")


# ------------------------------------------------------------------------------------------------ small helpers
def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def configs() -> list:
    """The installed models' configs, the most recently used first."""
    return sorted(ROOT.glob("maya-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def mem_gb() -> tuple:
    """(total, available) RAM in GiB - the engine sizes its RAM tier from MemAvailable."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024 / 2**30
    except (OSError, ValueError):
        pass
    return info.get("MemTotal", 0.0), info.get("MemAvailable", 0.0)


def existing(path: Path) -> Path:
    """`path`, or its nearest folder that exists (where it would be created)."""
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def rotational(path: Path) -> bool:
    """True when `path` is on a spinning disk (best effort: findmnt + lsblk)."""
    src = S.out(["findmnt", "-no", "SOURCE", "--target", str(existing(path))]).strip().split("[")[0]
    rota = S.out(["lsblk", "-ndo", "ROTA", src]).split() if src.startswith("/dev/") else []
    return bool(rota) and rota[0] == "1"


def tool_version(exe: str) -> tuple:
    m = re.search(r"(\d+)\.(\d+)", S.out([exe, "--version"]))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def venv_tool(name: str):
    p = Path(sys.executable).parent / name
    return str(p) if p.exists() else None


def pick_cmake():
    """CMake 3.24 or newer: the one pip put into .venv first, then the system's."""
    for c in (venv_tool("cmake"), shutil.which("cmake")):
        if c and tool_version(c) >= (3, 24):
            return c
    return None


def gpu_label(g) -> str:
    return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB, compute capability {S.cc(g)})"


def nvcc_range(archs) -> tuple:
    """The CUDA toolkit versions that can build for these GPUs: (lowest, first too new or None)."""
    lo = (12, 8) if max(archs) >= 100 else (12, 0)  # Blackwell needs 12.8
    hi = (13, 0) if min(archs) < 75 else None       # CUDA 13 no longer compiles for Volta (sm_70)
    return lo, hi


def find_nvcc(archs, given=None) -> tuple:
    """(nvcc, version) for these GPUs: --nvcc, else the newest toolkit that can build for them (Strata's find_nvcc
    takes the newest of all, but CUDA 13 cannot build for Volta), else the newest found (reported as unfit), else
    (None, None)."""
    def version(c):
        v = re.search(r"release (\d+)\.(\d+)", S.out([c, "--version"]))
        return (int(v.group(1)), int(v.group(2))) if v else None
    if given:
        return given, version(given)
    cands = [shutil.which("nvcc"), os.environ.get("CUDA_PATH") and str(Path(os.environ["CUDA_PATH"]) / "bin" / "nvcc")]
    cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/usr/local").glob("cuda*"))]
    cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/opt").glob("cuda*"))] + ["/usr/bin/nvcc"]
    return pick_nvcc([(c, version(c)) for c in dict.fromkeys(c for c in cands if c and Path(c).exists())], archs)


def pick_nvcc(found, archs) -> tuple:
    """Of [(nvcc, version)], the newest that can build for `archs`, else the newest, else (None, None)."""
    found = sorted([f for f in found if f[1]], key=lambda f: f[1], reverse=True)
    lo, hi = nvcc_range(archs)
    fit = [f for f in found if f[1] >= lo and (hi is None or f[1] < hi)]
    return (fit or found or [(None, None)])[0]


def toolkit_hint(archs) -> str:
    if min(archs) < 75:
        return ("Volta (V100) needs CUDA 12.x - Ubuntu 24.04: sudo apt-get install -y nvidia-cuda-toolkit (CUDA 12.0), "
                "or NVIDIA's cuda-toolkit-12-8 package: https://developer.nvidia.com/cuda-12-8-0-download-archive")
    if max(archs) >= 100:
        return "NVIDIA's cuda-toolkit-12-8 (or newer) package: https://developer.nvidia.com/cuda-downloads"
    return ("Ubuntu 24.04: sudo apt-get install -y nvidia-cuda-toolkit (CUDA 12.0), or NVIDIA's packages: "
            "https://developer.nvidia.com/cuda-downloads")


def cuda_lib_dirs(nvcc: str) -> list:
    """The toolkit's library folders for the engine's LD_LIBRARY_PATH (none for a distribution's /usr/bin/nvcc)."""
    base = Path(nvcc).resolve().parent.parent
    if base == Path("/usr"):
        return []
    return [str(p) for p in (base / "lib64", base / "targets" / "x86_64-linux" / "lib") if p.is_dir()]


# ------------------------------------------------------------------------------------------------ 1. the PC
def choose_gpus(a, found) -> list:
    """The GPUs the model runs on: --gpu / --gpus, else asked when two can share it (both recommended)."""
    byid = {g["index"]: g for g in found}
    usable = [g for g in found if int(g["arch"]) >= MIN_CC]
    if a.gpus or a.gpu is not None:
        try:
            want = [int(x) for x in str(a.gpus).split(",") if x.strip()] if a.gpus else [a.gpu]
        except ValueError:
            fail(f"--gpus takes GPU numbers as nvidia-smi numbers them, e.g. --gpus 0,1, not {a.gpus!r}")
        if not 1 <= len(want) <= 2 or len(set(want)) != len(want):
            fail("Maya runs on one GPU or splits the model across two: --gpu 0, or --gpus 0,1")
        for i in want:
            if i not in byid or int(byid[i]["arch"]) < MIN_CC:
                fail(f"GPU {i} cannot be used" + ("" if i in byid else " (not found)"),
                     "use one of: " + ", ".join(str(g["index"]) for g in usable))
        return [byid[i] for i in want]
    if not usable:
        fail("none of the GPUs can run Maya", "it needs an NVIDIA GPU of compute capability 7.0 or newer (V100 and newer)")
    best = sorted(usable, key=lambda g: (-round(g["vram_gb"]), g["index"]))
    if len(best) == 1:
        return best
    pair = best[:2]
    say()
    say("  Maya can run on one GPU, or split the model's layers across two: each then caches the experts of its own")
    say("  layers, so together they hold about twice as many (V100s: ~50 tok/s benchmark on two, ~24 on one).")
    say(f"  1) {gpu_label(pair[0])} + {gpu_label(pair[1])} together   (recommended)")
    say(f"  2) {gpu_label(best[0])} only")
    if len(best) > 2:
        say("     (the engine splits across two GPUs at most: --gpus picks another pair)")
    return pair if ask("Which GPUs?", ["1", "2"], "1", a.yes or a.check) == "1" else best[:1]


def check_pc(a) -> dict:
    step(1, "checking this PC")
    if not sys.platform.startswith("linux"):
        fail("Project Maya runs on Linux only for now",
             "the engine's GLM model loader has no Windows file mapping yet (README-MAYA.md, START-MAYA.bat)")
    if S.is_wsl():
        warn("this is WSL2, which Maya does not support: Strata measured that WSL2's driver pins only about 1 GB of "
             "RAM for the GPU, and Maya's RAM tier pins tens of GB. Use a native Linux install.")
    found = S.gpus()
    if not found:
        fail("no NVIDIA GPU found (nvidia-smi did not answer)",
             "install the NVIDIA driver (Ubuntu: sudo ubuntu-drivers install), restart, and run ./maya.sh again")
    say("  NVIDIA GPUs:")
    for g in found:
        say(f"    {gpu_label(g)} - " + ("can be used" if int(g["arch"]) >= MIN_CC else
                                       "too old (Maya needs compute capability 7.0 or newer)"))
    chosen = choose_gpus(a, found)
    archs = sorted({int(g["arch"]) for g in chosen})
    vram = sum(g["vram_gb"] for g in chosen)
    ok("using " + " + ".join(gpu_label(g) for g in chosen) +
       (": the model's layers are split across both" if len(chosen) > 1 else ""))
    if vram < 23:
        warn(f"{vram:.0f} GB of VRAM in total. Maya was measured with one or two 32 GB V100s; below 24 GB it is "
             "untested: more experts come from RAM and the SSD, so it is slower")
    problems = []                                      # (what is wrong, how to fix it): all of them at once

    nvcc, nv = find_nvcc(archs, a.nvcc)
    lo, hi = nvcc_range(archs)
    want = f"CUDA {lo[0]}.{lo[1]} or newer" + (f" but older than {hi[0]}.0 (CUDA 13 dropped Volta)" if hi else "")
    if nvcc is None:
        problems.append((f"the CUDA toolkit (nvcc) is not installed; these GPUs need {want}", toolkit_hint(archs)))
    elif nv is None:
        problems.append((f"{nvcc} does not say its version (nvcc --version)", "pass the toolkit's nvcc with --nvcc"))
    elif nv < lo or (hi is not None and nv >= hi):
        problems.append((f"nvcc {nv[0]}.{nv[1]} ({nvcc}) cannot build for these GPUs; they need {want}",
                         toolkit_hint(archs) + " (toolkits can be installed side by side: the newest that fits is "
                                                "used, or pass one with --nvcc)"))
    else:
        ok(f"CUDA toolkit {nv[0]}.{nv[1]}: {nvcc}")

    drv = min(S.driver_major(g) for g in chosen)
    need_drv = 580 if nv and nv >= (13, 0) else 525
    if drv < need_drv:
        problems.append((f"the NVIDIA driver {chosen[0]['driver']} is too old: {need_drv} or newer is needed",
                         "Ubuntu: sudo ubuntu-drivers install, then restart; or https://www.nvidia.com/drivers"))
    else:
        ok(f"NVIDIA driver {chosen[0]['driver']}")

    if a.host_compiler:
        if shutil.which(a.host_compiler) is None:
            problems.append((f"--host-compiler {a.host_compiler} is not installed",
                             f"Ubuntu/Debian: sudo apt-get install -y {Path(a.host_compiler).name}"))
        else:
            ok(f"C++ compiler for nvcc: {a.host_compiler}")
    elif shutil.which("g++") is None:
        problems.append(("the C++ compiler (g++) is not installed", "Ubuntu/Debian: sudo apt-get install -y build-essential"))
    else:
        ok("C++ compiler: " + (S.out(["g++", "--version"]).splitlines() or ["g++"])[0])

    sc = shutil.which("cmake")
    scv = tool_version(sc) if sc else (0, 0)
    if scv >= (3, 24):
        ok(f"CMake {scv[0]}.{scv[1]}")
    else:
        ok((f"CMake {scv[0]}.{scv[1]} is older than 3.24" if sc else "no CMake on PATH") +
           ": a current one goes into .venv in step 3 (pip)")
    ok(f"Python {sys.version.split()[0]} ({sys.prefix})")

    total, avail = mem_gb()
    msg = f"RAM: {total:.0f} GB, {avail:.0f} GB available now"
    if total < 60:
        warn(msg + ". 64 GB is recommended: the engine keeps as many experts in RAM as fit, and every expert that "
                   "does not fit is read from the SSD while it answers (slower)")
    else:
        ok(msg)
    cpu, avx2, avx512 = S.cpu_info()
    if not avx2:
        problems.append((f"the CPU ({cpu}) has no AVX2", "the engine's CPU expert lane and ggml need AVX2"))
    else:
        ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2'}, {os.cpu_count()} threads)")

    if problems:
        say()
        for what, how in problems:
            say(f"  [X]  {what}")
            say(f"       {how}")
        fail("something Maya needs is missing (above)",
             "install it and run ./maya.sh again - this script installs nothing system-wide")
    return {"gpus": chosen, "archs": archs, "nvcc": nvcc}


# ------------------------------------------------------------------------------------------------ 2. choices
def choose_context(a, prev_ctx) -> int:
    if a.context:
        return a.context
    default = prev_ctx if prev_ctx in CONTEXTS else DEFAULT_CONTEXT
    say()
    say("  Context length = how much text the model sees at once (the chat, files, tool output). A longer one takes")
    say("  VRAM from the expert cache, so answers get a little slower:")
    for i, c in enumerate(CONTEXTS, 1):
        say(f"  {i}) {c // 1024}K tokens" + ("   (recommended)" if c == DEFAULT_CONTEXT else ""))
    pick = ask("Context?", [str(i) for i in range(1, len(CONTEXTS) + 1)], str(CONTEXTS.index(default) + 1), a.yes)
    return CONTEXTS[int(pick) - 1]


# ------------------------------------------------------------------------------------------------ 3. tools
def tools_step(a) -> Path:
    """The Python packages (into .venv) and llama.cpp's source at the pinned commit - listed, and asked, first."""
    step(3, "Python packages and llama.cpp's source")
    stamp = Path(sys.prefix) / ".strata-pip.json"          # S.pip_install's record of what it installed
    have = read_json(stamp) if stamp.exists() else []
    need = [p for p in PY_PACKAGES if p not in (have if isinstance(have, list) else [])]
    llama = Path(a.llama_dir).expanduser().resolve() if a.llama_dir else ROOT / "third_party" / "llama.cpp"
    llama_ok = (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir()
    if a.llama_dir and not llama_ok:
        fail(f"{llama} is not a llama.cpp checkout (its ggml/ and gguf-py/ folders are missing)")
    downloads = []
    if need:
        downloads.append("Python packages from PyPI into .venv: " + ", ".join(need) + " (roughly 50-100 MB)")
    if not llama_ok:
        downloads.append(f"llama.cpp's source at the pinned commit {S.LLAMA_CPP_COMMIT[:10]} from GitHub, "
                         f"{S.LLAMA_CPP_ZIP} (tens of MB): ggml for the engine build, gguf-py for the pack builder")
    if downloads:
        say("  This step downloads:")
        for d in downloads:
            say("    - " + d)
        if ask("  Download them now?", ["y", "n"], "y", a.yes) != "y":
            fail("nothing was downloaded", "run ./maya.sh again when you are ready (or pass --llama-dir with a "
                                           f"llama.cpp checkout at commit {S.LLAMA_CPP_COMMIT[:10]})")
    S.pip_install(PY_PACKAGES, ", ".join(PY_PACKAGES))
    if not llama_ok:
        llama = S.get_llama_cpp()
    ok(f"llama.cpp {S.LLAMA_CPP_COMMIT[:10]}: {llama}")
    return llama


# ------------------------------------------------------------------------------------------------ 4. the engine
def cached_generator():
    """The CMake generator build/ was configured with before (another one cannot be used there), or None."""
    m = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", (BUILD / "CMakeCache.txt").read_text(errors="replace"), re.M) \
        if (BUILD / "CMakeCache.txt").exists() else None
    return m.group(1).strip() if m else None


def arch_setting(archs) -> str:
    """CMAKE_CUDA_ARCHITECTURES for these GPUs.  CMakeLists.txt refuses an explicit arch below 75 (Strata's own
    qwen4exp floor), but the GLM path is developed and measured on Volta V100s (sm_70) - so for those the build
    asks CMake for this machine's own GPUs ("native", which the guard leaves to CMake), with CUDA_VISIBLE_DEVICES
    limited to the chosen cards.  RELEASE-CHECKLIST.md has the one-line CMake fix that makes this unnecessary."""
    return ";".join(str(x) for x in archs) if min(archs) >= 75 else "native"


def compile_engine(archs, gpu_ids, nvcc, host_compiler, llama: Path, src: str, soft=False) -> dict | None:
    """cmake configure + build of the `strata` target in build/; the exact commands are printed as they run.
    soft: a failure warns and returns None (the engine already there keeps working) instead of stopping."""
    def stop(what):
        msg = f"the engine build stopped while {what} (the reason is above)"
        hint = ("common causes: 'unsupported GNU version' - run again with --host-compiler g++-12 (an older g++ your "
                "CUDA accepts);\n       'Unsupported gpu architecture compute_70' - CUDA 13 cannot build for Volta, "
                "install CUDA 12.x;\n       the compiler killed (out of memory) - close programs and run it again "
                "(it continues)")
        if soft:
            warn(msg + "; starting the engine compiled before")
            return None
        fail(msg, hint)

    cmake = pick_cmake()
    if cmake is None:
        return stop("looking for CMake 3.24+ (pip installs one into .venv: run ./maya.sh --setup)")
    gen = cached_generator()
    ninja = venv_tool("ninja") or shutil.which("ninja")
    conf = [cmake]
    if ninja and gen in (None, "Ninja"):
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    cuda_archs = arch_setting(archs)
    conf += ["-S", str(ROOT), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
             "-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
             "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF",
             f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DSTRATA_GGML_DIR={llama}"]
    if host_compiler:
        conf.append(f"-DCMAKE_CUDA_HOST_COMPILER={shutil.which(host_compiler) or host_compiler}")
    # "native" is what CMake sees: only the chosen cards, numbered as nvidia-smi numbers them
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(str(i) for i in gpu_ids))
    # nvcc takes 2-4 GB per job on the big kernels: half the threads, and no more jobs than RAM / 4 GB
    jobs = max(2, min((os.cpu_count() or 4) // 2, int(mem_gb()[0] // 4) or 2))
    build = [cmake, "--build", str(BUILD), "--target", "strata", "-j", str(jobs)]
    say("  Compiling the engine for " + ", ".join(f"sm_{x}" for x in archs) +
        " (10-30 minutes the first time, a few minutes after an update) ...")
    if cuda_archs == "native":
        say("  (CMakeLists.txt refuses an explicit sm_70 - Strata's own floor is sm_75 - so CMake is asked for this")
        say(f"  machine's GPUs instead: CMAKE_CUDA_ARCHITECTURES=native with CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
    if run(conf, env=env, check=False).returncode != 0:
        return stop("configuring")
    if run(build, env=env, check=False).returncode != 0:
        say("  (the build stopped - trying it once more: it continues where it stopped)")
        if run(build, env=env, check=False).returncode != 0:
            return stop("compiling")
    meta = {"src": src, "archs": archs, "gpu_ids": list(gpu_ids), "nvcc": nvcc, "host_compiler": host_compiler,
            "llama": str(llama), "lib_dirs": cuda_lib_dirs(nvcc), "date": time.strftime("%Y-%m-%d %H:%M")}
    STAMP.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ok(f"engine compiled: {EXE}")
    return meta


def build_step(a, pc, llama: Path) -> dict:
    step(4, "the engine")
    src = S.source_hash(S.ENGINE_SOURCES)              # src/, include/, third_party/ggml, CMakeLists.txt
    meta = read_json(STAMP)
    if (EXE.exists() and not a.rebuild and meta.get("src") == src and set(pc["archs"]) <= set(meta.get("archs", []))
            and (a.host_compiler or None) == meta.get("host_compiler")):
        ok(f"engine already compiled for " + ", ".join(f"sm_{x}" for x in meta["archs"]) + f": {EXE}")
        return meta
    return compile_engine(pc["archs"], [g["index"] for g in pc["gpus"]], pc["nvcc"], a.host_compiler, llama, src)


def refresh_engine(cfg: dict) -> None:
    """At a start: the engine's source changed since it was compiled (a git pull) - compile what changed."""
    meta = read_json(STAMP)
    if not meta or Path(cfg["exe"]).resolve() != EXE.resolve():
        return
    src = S.source_hash(S.ENGINE_SOURCES)
    if meta.get("src") == src:
        return
    say("  The engine's source changed since it was compiled (an update): compiling what changed ...")
    nvcc, llama = meta.get("nvcc"), Path(meta.get("llama") or ROOT / "third_party" / "llama.cpp")
    if not nvcc or not Path(nvcc).exists() or not (llama / "ggml" / "CMakeLists.txt").exists():
        warn("the compiler or llama.cpp's source it was built with is gone: starting the engine compiled before "
             "(./maya.sh --setup --rebuild compiles it again)")
        return
    gpu_ids = meta.get("gpu_ids") or (cfg.get("gpu") if isinstance(cfg.get("gpu"), list) else [0])
    compile_engine(meta["archs"], gpu_ids, nvcc, meta.get("host_compiler"), llama, src, soft=True)


# ------------------------------------------------------------------------------------------------ 5. the model
def shard_set(first: Path) -> list:
    """All files of a split GGUF from its first one (<name>-00001-of-0000N.gguf), or just the file."""
    m = SHARD_RE.search(first.name)
    if not m:
        return [first]
    n, stem = int(m.group(2)), first.name[:m.start()]
    return [first.with_name(f"{stem}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def find_first_shard(d: Path):
    c = sorted(d.glob("*-00001-of-*.gguf")) or sorted(p for p in d.glob("*.gguf") if not SHARD_RE.search(p.name))
    return c[0] if c else None


def incomplete(path: Path):
    """Why this GGUF file is not whole, or None when it is: as long as its own tensor directory says (the same test
    as Strata's check_shards, without stopping).  Reads the header only."""
    if not path.exists():
        return "missing"
    from gguf_reader import GGUFFile
    try:
        g = GGUFFile(path)
    except Exception as e:                             # noqa: BLE001 - a partial header raises anything
        return f"not a whole GGUF file ({e})"
    need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
    have = path.stat().st_size
    return None if have >= need else f"short: {have:,} of {need:,} bytes"


def quant_of(first: Path) -> str:
    m = re.match(r"GLM-5\.3-Flash-(.+?)(-\d{5}-of-\d{5})?\.gguf$", first.name, re.I)
    return m.group(1) if m else first.parent.name


def sha256_ok(path: Path, want: str) -> bool:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest() == want.lower()


def offer_download(a, quant: str, d: Path, shards: list) -> bool:
    """Shows what would be downloaded - the source, the size, the exact commands - and downloads only after a yes
    (or --download-model).  False: nothing was downloaded."""
    m = MODELS[quant]
    missing = [s for s in shards if incomplete(s)]
    urls = {s: HF.format(repo=m["repo"], revision=m["revision"], folder=m["folder"], file=s.name) for s in shards}
    on_disk = sum(s.stat().st_size for s in shards if s.exists()) / 1e9
    remaining = max(0.0, m["download_gb"] - on_disk)
    free = shutil.disk_usage(existing(d)).free / 1e9
    curl = shutil.which("curl")
    cmds = [["mkdir", "-p", str(d)]] + [["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(s), urls[s]]
                                        for s in missing]
    say(f"  {quant}: {m['about']}.")
    say(f"  Source: https://huggingface.co/{m['repo']} (folder {m['folder']}/); the files' own license applies.")
    say(f"  The model is {m['download_gb']:.1f} GB in {len(shards)} files; {len(missing)} still to download, about "
        f"{remaining:.0f} GB, into")
    say(f"    {d}   ({free:.0f} GB free there)")
    if free < remaining + 3:
        fail(f"not enough free space in {d}: about {remaining + 3:.0f} GB are needed (the model + its pack)",
             "free some space, or put the model on another drive: ./maya.sh --setup --data-dir /path/on/nvme")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: the engine reads experts from these files while it answers - use an "
             "NVMe SSD (--data-dir)")
    say("  The exact commands (resumable: running them again continues an interrupted download):")
    for c in cmds:
        say("    " + shlex.join(c))
    say("  You can also run them yourself (or download the files any other way into that folder, or pass")
    say("  --gguf-dir <folder with the files>), then run ./maya.sh again.")
    if not curl:
        warn("curl is not installed (sudo apt-get install -y curl): if you say yes, the same URLs are downloaded "
             "with Python instead (also resumable)")
    if not a.download_model and ask(f"  Download about {remaining:.0f} GB now with these commands? (y/n)",
                                    ["y", "n"], "n", False) != "y":
        say()
        say("Nothing was downloaded. When the files are in place, run ./maya.sh again "
            "(./maya.sh --download-model downloads them without asking).")
        return False
    d.mkdir(parents=True, exist_ok=True)
    for s in missing:
        if curl:
            cmd = ["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(s), urls[s]]
            if run(cmd, check=False).returncode != 0:
                fail(f"the download of {s.name} stopped (the reason is above)",
                     "run ./maya.sh --download-model again: it continues where it stopped")
        else:
            S.download(urls[s], s)
        why = incomplete(s)
        if why:
            fail(f"{s.name} after the download: {why}", "delete it and run ./maya.sh --download-model again")
        want = m["sha256"].get(s.name)
        if want:
            say(f"  checking {s.name}'s sha256 ...")
            if not sha256_ok(s, want):
                fail(f"{s.name}: the sha256 does not match the published one", "delete it and download it again")
        ok(f"{s.name} downloaded")
    return True


def model_step(a, data: Path):
    """(model folder, shards, quant name), or None when the files are not there and were not downloaded."""
    step(5, "the model (GLM-5.3-Flash, GGUF)")
    if a.gguf_dir:
        d = Path(a.gguf_dir).expanduser().resolve()
        first = find_first_shard(d) if d.is_dir() else None
        if first is None:
            fail(f"no GGUF file in {d}", "--gguf-dir takes the folder that holds the GLM-5.3-Flash .gguf files")
        shards, quant = shard_set(first), quant_of(first)
        for s in shards:
            why = incomplete(s)
            if why:
                fail(f"{s.name}: {why}", "finish copying or downloading it, then run ./maya.sh again")
        if quant not in MODELS:
            warn(f"{quant}: experimental - only {', '.join(MODELS)} has been measured with Maya (README-MAYA.md)")
    else:
        quant = a.model or next(iter(MODELS))
        m = MODELS[quant]
        d = data / "models" / f"glm-5.3-flash-{quant}".lower()
        shards = [d / m["file"].format(i=i, n=m["shards"]) for i in range(1, m["shards"] + 1)]
        if any(incomplete(s) for s in shards) and not offer_download(a, quant, d, shards):
            return None
    from gguf_reader import GGUFFile
    arch = str(GGUFFile(shards[0]).metadata.get("general.architecture", ""))
    if arch not in ("glm5-next", "glm5next"):
        fail(f"{shards[0].name} is a {arch!r} model, not GLM-5.3-Flash (glm5-next)")
    gb = sum(s.stat().st_size for s in shards) / 1e9
    ok(f"{quant}: {len(shards)} file(s), {gb:.1f} GB in {d}")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: expect slow answers (the engine reads experts from it) - an NVMe SSD "
             "is strongly recommended")
    return d, shards, quant


# ------------------------------------------------------------------------------------------------ 6. the pack
def pack_step(a, d: Path, shards: list, llama: Path) -> Path:
    step(6, "the pack (the engine's index of the model files)")
    # the engine finds the GGUF files at <pack>/.. (src/core/glm_model.cu, load_pack): the pack lives in their folder
    pack = d / "pack"
    if not a.repack and all((pack / f).exists() for f in PACK_FILES):
        ok(f"pack already built: {pack}")
        return pack
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the pack is written next to the GGUF files (the engine finds them at <pack>/..)",
             "link the .gguf files into a writable folder (mkdir -p DIR && ln -s /path/to/*.gguf DIR/) and pass "
             "--gguf-dir DIR")
    say("  Writing the index of every tensor, the small float weights (about 1 GB) and the tokenizer; the experts")
    say("  stay in the GGUF files (a few minutes) ...")
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    # --compat-bf16: the GLM pack stores the projections its kernels read as BF16 (router, indexer, absorbed MLA,
    # KDA gates) in BF16 even where the GGUF quantized them or keeps them F32 (tools/iq_pack.py GLM5NEXT_BF16)
    run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
         "--compat-bf16"], env=env)
    missing = [f for f in PACK_FILES if not (pack / f).exists()]
    if missing:
        fail(f"the pack builder did not write {pack / missing[0]}", "the reason is in the messages above")
    ok(f"pack: {pack}")
    return pack


# ------------------------------------------------------------------------------------------------ 7. config + start
def parse_env(items) -> dict:
    env = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            fail(f"--env takes KEY=VALUE, e.g. --env STRATA_GLM_RAM_HEADROOM_GB=4, not {it!r}")
        env[k] = v
    return env


def write_run_script(cfg_path: Path, port: int) -> Path:
    script = ROOT / f"run-{cfg_path.stem}.sh"
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port)]
    script.write_text("#!/bin/sh\n# starts the Project Maya dashboard and API (written by maya.py; ./maya.sh does the "
                      "same)\ncd " + shlex.quote(str(ROOT)) + " || exit 1\nexec " + shlex.join(cmd) + ' "$@"\n',
                      encoding="utf-8")
    script.chmod(0o755)
    return script


def write_config(a, pc, meta, pack: Path, quant: str, ctx: int, data: Path) -> Path:
    step(7, "the configuration and the start script")
    port = a.port or 8080
    cfg = {"exe": str(EXE), "args": ["--glm-pack", str(pack), "--max-context", str(ctx)], "cwd": str(ROOT),
           "tokenizer": str(pack / "tokenizer"), "model_name": MODEL_NAME, "gpu": [g["index"] for g in pc["gpus"]],
           "sampling": dict(SAMPLING), "reasoning_effort": EFFORT, "lib_dirs": meta.get("lib_dirs") or [],
           "port": port}
    env = parse_env(a.env)
    if env:
        cfg["env"] = env
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    cfg_path = ROOT / f"maya-{quant.lower()}.json"
    cfg["log"] = str(cfg_path.with_suffix(".log"))
    cfg["installer"] = {"data_dir": str(data), "quant": quant,
                        "gguf_dir": str(Path(a.gguf_dir).expanduser().resolve()) if a.gguf_dir else None,
                        "written": time.strftime("%Y-%m-%d %H:%M")}
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    ok(f"config: {cfg_path}")
    ok(f"start script: {write_run_script(cfg_path, port)}")
    return cfg_path


def start(cfg_path: Path, a) -> int:
    cfg = read_json(cfg_path)
    args = cfg.get("args") or []
    pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
    for p, what in ((Path(cfg.get("exe", "")), "the engine"), (pack, "the pack"),
                    (Path(cfg.get("tokenizer", "")) / "vocab.json", "the tokenizer")):
        if p is None or not p.exists():
            fail(f"{cfg_path.name}: {what} is missing ({p})", "run ./maya.sh --setup to repair it")
    cfg_path.touch()                                   # the most recently used model
    refresh_engine(cfg)
    port = a.port or cfg.get("port") or 8080
    host = a.host or cfg.get("host") or "127.0.0.1"
    key = a.api_key or cfg.get("api_key")
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port)]
    if a.host:
        cmd += ["--host", a.host]
    if a.api_key:
        cmd += ["--api-key", a.api_key]
    if a.gpus or a.gpu is not None:                    # this start only, on these cards
        cmd += ["--gpu", str(a.gpus or a.gpu)]
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        cmd.append("--open")                           # a desktop: the browser opens when the model is ready
    here = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    say()
    say("  " + "-" * 100)
    say(f"  Starting {cfg.get('model_name', MODEL_NAME)} ({cfg_path.name}).")
    say(f"  Dashboard:        http://{here}:{port}/")
    say(f"  API (OpenAI):     http://{here}:{port}/v1        (any model name; the API key: "
        + ("the one you set)" if key else "any)"))
    say(f"  API (Anthropic):  http://{here}:{port}/v1/messages")
    if host not in ("127.0.0.1", "localhost", "::1"):
        say("  Other devices:    the server prints this machine's addresses when it is ready" +
            ("" if key else " - WARNING: no API key, anyone on your network can use it (--api-key KEY)"))
    say("  Loading takes a few minutes: the engine pins up to all but about 6 GB of the free RAM for its expert tier")
    say("  (STRATA_GLM_RAM_HEADROOM_GB changes the 6) and warms its caches - the first answers are the slowest.")
    say("  Ctrl+C (or closing this terminal) stops it. Engine log: " + str(cfg.get("log", "")))
    say("  " + "-" * 100)
    return subprocess.call(cmd)


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", action="store_true", help="set up again instead of starting the installed model")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--no-start", action="store_true", help="set up, but do not start the dashboard")
    ap.add_argument("--yes", action="store_true",
                    help="take the recommended answers (the model download still needs --download-model)")
    ap.add_argument("--download-model", action="store_true",
                    help="download the model without asking (the commands and the size are still printed)")
    ap.add_argument("--model", choices=list(MODELS), help="which download (default: the first, UD-IQ1_S)")
    ap.add_argument("--gguf-dir", help="use GLM-5.3-Flash GGUF files you already have: the folder with all of them "
                                       "(it must be writable - the pack is written inside it)")
    ap.add_argument("--data-dir", help="where a downloaded model goes (default: Maya-data next to this folder); use a "
                                       "fast NVMe SSD with ~100 GB free")
    ap.add_argument("--gpu", type=int, help="run on this one GPU (as nvidia-smi numbers them)")
    ap.add_argument("--gpus", help="split the model across these two GPUs, e.g. 0,1")
    ap.add_argument("--context", type=int, help=f"context length in tokens (default {DEFAULT_CONTEXT})")
    ap.add_argument("--port", type=int, help="the dashboard's and the API's port (default 8080)")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this machine only (default), 0.0.0.0 = also "
                                   "other devices on your network (set --api-key too)")
    ap.add_argument("--api-key", help="require this key from API clients and the dashboard")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="an engine setting kept in the config, e.g. STRATA_GLM_RAM_HEADROOM_GB=4 (README-MAYA.md)")
    ap.add_argument("--nvcc", help="the CUDA toolkit's nvcc to build with (default: the newest that fits your GPUs)")
    ap.add_argument("--host-compiler", help="the C++ compiler nvcc uses, e.g. g++-12 (when the default g++ is newer "
                                            "than your CUDA accepts)")
    ap.add_argument("--llama-dir", help="a llama.cpp checkout at the pinned commit, instead of downloading its source")
    ap.add_argument("--rebuild", action="store_true", help="compile the engine again")
    ap.add_argument("--repack", action="store_true", help="build the pack again")
    a = ap.parse_args()
    say("Project Maya - GLM-5.3-Flash on your own NVIDIA GPU(s). Built on Strata (MIT) and ggml/llama.cpp (MIT).")

    have = configs()
    setting_up = a.setup or a.check or a.no_start or a.gguf_dir or a.model or a.rebuild or a.repack or a.download_model
    if have and not setting_up:
        pick = have[0]
        if len(have) > 1:
            say()
            for i, c in enumerate(have, 1):
                say(f"  {i}) {c.name}")
            pick = have[int(ask("Which one?", [str(i) for i in range(1, len(have) + 1)], "1", a.yes)) - 1]
        return start(pick, a)
    prev = read_json(have[0]) if have else {}
    inst = prev.get("installer") or {}
    a.gguf_dir = a.gguf_dir or inst.get("gguf_dir")
    prev_args = prev.get("args") or []
    prev_ctx = int(prev_args[prev_args.index("--max-context") + 1]) if "--max-context" in prev_args[:-1] else None

    pc = check_pc(a)                                   # 1
    if a.check:
        say()
        say("This PC can run Maya. Run ./maya.sh without --check to set it up.")
        return 0
    step(2, "your choices")                            # 2
    ctx = choose_context(a, prev_ctx)
    ok(f"context: {ctx} tokens")
    data = (Path(a.data_dir).expanduser().resolve() if a.data_dir else
            Path(inst["data_dir"]) if inst.get("data_dir") else ROOT.parent / "Maya-data")
    if not a.gguf_dir:
        ok(f"model folder: {data / 'models'}")
    llama = tools_step(a)                              # 3
    meta = build_step(a, pc, llama)                    # 4
    got = model_step(a, data)                          # 5
    if got is None:
        return 0
    d, shards, quant = got
    pack = pack_step(a, d, shards, llama)              # 6
    cfg_path = write_config(a, pc, meta, pack, quant, ctx, data)   # 7
    if a.no_start:
        say()
        say(f"All set. Start it with ./maya.sh (or ./run-{cfg_path.stem}.sh).")
        return 0
    return start(cfg_path, a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
