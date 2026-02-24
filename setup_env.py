"""
Setup Environment — P-DARTS Palm Vein NAS
==========================================
Otomatis install semua dependensi yang diperlukan untuk menjalankan
P-DARTS NAS, termasuk PyTorch dengan dukungan CUDA untuk NVIDIA RTX 5090.

Cara pakai:
    python setup_env.py

Syarat sistem (untuk training GPU):
    - OS        : Linux (disarankan), Windows (WSL2), macOS (CPU/MPS only)
    - Python    : 3.10 / 3.11 / 3.12
    - CUDA      : 12.6+ (untuk NVIDIA RTX 5090 / Blackwell sm_100)
    - Driver    : >= 560.x (NVIDIA)
"""

import subprocess
import sys
import platform
import os
from pathlib import Path


# ─── Warna Terminal ───────────────────────────────────────────────────────────
class C:
    OK    = "\033[92m"   # hijau
    WARN  = "\033[93m"   # kuning
    ERR   = "\033[91m"   # merah
    INFO  = "\033[96m"   # cyan
    BOLD  = "\033[1m"
    RESET = "\033[0m"

def ok(msg):   print(f"{C.OK}[✓] {msg}{C.RESET}")
def warn(msg): print(f"{C.WARN}[!] {msg}{C.RESET}")
def err(msg):  print(f"{C.ERR}[✗] {msg}{C.RESET}")
def info(msg): print(f"{C.INFO}[→] {msg}{C.RESET}")
def bold(msg): print(f"{C.BOLD}{msg}{C.RESET}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    """Jalankan perintah dan tampilkan output secara real-time."""
    info(f"Menjalankan: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result


def pip(*packages: str, extra_args: list[str] | None = None):
    """Install satu atau lebih paket via pip."""
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + list(packages)
    if extra_args:
        cmd += extra_args
    run(cmd)


def check_python_version():
    """Pastikan Python 3.10+."""
    bold("\n=== Cek Versi Python ===")
    v = sys.version_info
    info(f"Python {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        err("Python 3.10+ diperlukan. Install Python terbaru terlebih dahulu.")
        sys.exit(1)
    ok(f"Python {v.major}.{v.minor} — OK")


def detect_platform() -> dict:
    """Deteksi OS, ketersediaan CUDA, dan arsitektur GPU."""
    bold("\n=== Deteksi Platform ===")

    os_name   = platform.system()          # 'Linux', 'Darwin', 'Windows'
    arch      = platform.machine()          # 'x86_64', 'arm64', etc.
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH", "")

    info(f"OS      : {os_name} ({arch})")
    info(f"CUDA_HOME: {cuda_home or '(tidak ditemukan di env)'}")

    # Cek nvidia-smi
    nvidia_available = False
    cuda_version     = None
    gpu_names        = []
    has_5090         = False

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        for line in smi.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            gpu_names.append(parts[0] if parts else "Unknown")
            if "5090" in parts[0]:
                has_5090 = True
        nvidia_available = True

        # Cek versi CUDA dari nvidia-smi
        smi2 = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        info(f"Driver  : {smi2.stdout.strip()}")

        # Cek nvcc untuk versi CUDA toolkit
        nvcc = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=False
        )
        if nvcc.returncode == 0:
            for line in nvcc.stdout.splitlines():
                if "release" in line.lower():
                    cuda_version = line.strip()
                    info(f"CUDA    : {cuda_version}")
                    break

        for g in gpu_names:
            ok(f"GPU     : {g}")

        if has_5090:
            ok("NVIDIA RTX 5090 (Blackwell sm_100) terdeteksi!")
        elif gpu_names:
            warn("RTX 5090 tidak terdeteksi, tapi GPU NVIDIA ditemukan.")

    except FileNotFoundError:
        warn("nvidia-smi tidak ditemukan — GPU CUDA tidak tersedia.")
        if os_name == "Darwin":
            info("macOS: akan menggunakan MPS (Apple Silicon) atau CPU.")

    return {
        "os":        os_name,
        "arch":      arch,
        "nvidia":    nvidia_available,
        "cuda_ver":  cuda_version,
        "has_5090":  has_5090,
        "gpus":      gpu_names,
    }


def get_torch_install_cmd(plat: dict) -> list[str]:
    """
    Tentukan perintah install PyTorch yang tepat sesuai platform.

    RTX 5090 (Blackwell / sm_100) butuh:
      PyTorch >= 2.6.0  +  CUDA 12.6
    """
    os_name      = plat["os"]
    nvidia_avail = plat["nvidia"]
    has_5090     = plat["has_5090"]

    if os_name == "Darwin":
        # macOS: Apple Silicon pakai MPS, Intel pakai CPU
        arch = plat["arch"]
        if arch == "arm64":
            info("macOS Apple Silicon → install PyTorch dengan MPS support")
        else:
            info("macOS Intel → install PyTorch CPU")
        # PyTorch 2.6 terbaru mendukung MPS di Apple Silicon
        return [
            sys.executable, "-m", "pip", "install", "--upgrade",
            "torch>=2.6.0", "torchvision>=0.21.0",
        ]

    elif os_name == "Linux" or os_name == "Windows":
        if nvidia_avail:
            if has_5090:
                # RTX 5090 butuh PyTorch 2.6+ dengan CUDA 12.6
                info("RTX 5090 terdeteksi → install PyTorch 2.6+ CUDA 12.6")
                return [
                    sys.executable, "-m", "pip", "install", "--upgrade",
                    "torch>=2.6.0", "torchvision>=0.21.0",
                    "--index-url", "https://download.pytorch.org/whl/cu126",
                ]
            else:
                # GPU NVIDIA lain: pakai CUDA 12.4 (lebih stabil)
                info("GPU NVIDIA terdeteksi → install PyTorch CUDA 12.4")
                return [
                    sys.executable, "-m", "pip", "install", "--upgrade",
                    "torch>=2.5.0", "torchvision>=0.20.0",
                    "--index-url", "https://download.pytorch.org/whl/cu124",
                ]
        else:
            warn("Tidak ada GPU NVIDIA — install PyTorch CPU only")
            return [
                sys.executable, "-m", "pip", "install", "--upgrade",
                "torch>=2.5.0", "torchvision>=0.20.0",
                "--index-url", "https://download.pytorch.org/whl/cpu",
            ]

    else:
        warn(f"OS tidak dikenal ({os_name}) — install PyTorch default")
        return [
            sys.executable, "-m", "pip", "install", "--upgrade",
            "torch>=2.5.0", "torchvision>=0.20.0",
        ]


def install_pytorch(plat: dict):
    """Install PyTorch + torchvision sesuai platform."""
    bold("\n=== Install PyTorch ===")
    cmd = get_torch_install_cmd(plat)
    run(cmd)


def install_dependencies():
    """Install semua library Python yang dibutuhkan oleh P-DARTS."""
    bold("\n=== Install Dependensi P-DARTS ===")

    packages = [
        # Data & komputasi
        "numpy>=1.24",
        "Pillow>=10.0",

        # Visualisasi
        "matplotlib>=3.7",
        "seaborn>=0.13",

        # Progress bar
        "tqdm>=4.65",

        # Profiling (opsional, untuk measure_latency & count_flops)
        "thop>=0.1.1",            # FLOPs counter (digunakan di utils.py)

        # Optional tapi berguna
        "scipy>=1.11",
        "scikit-learn>=1.3",
    ]

    pip(*packages)


def upgrade_pip():
    """Update pip ke versi terbaru."""
    bold("\n=== Update pip ===")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])


def verify_installation(plat: dict):
    """Verifikasi semua library terinstall dengan benar dan CUDA aktif."""
    bold("\n=== Verifikasi Instalasi ===")

    checks = {
        "torch":       "import torch; print(f'PyTorch {torch.__version__}')",
        "torchvision": "import torchvision; print(f'torchvision {torchvision.__version__}')",
        "numpy":       "import numpy as np; print(f'NumPy {np.__version__}')",
        "PIL":         "from PIL import Image; import PIL; print(f'Pillow {PIL.__version__}')",
        "matplotlib":  "import matplotlib; print(f'matplotlib {matplotlib.__version__}')",
        "seaborn":     "import seaborn as sns; print(f'seaborn {sns.__version__}')",
        "tqdm":        "import tqdm; print(f'tqdm {tqdm.__version__}')",
    }

    all_ok = True
    for name, code in checks.items():
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            ok(f"{name:15s} {result.stdout.strip()}")
        else:
            err(f"{name:15s} GAGAL: {result.stderr.strip()[:80]}")
            all_ok = False

    return all_ok


def verify_cuda(plat: dict):
    """Verifikasi CUDA tersedia di PyTorch dan GPU terdeteksi."""
    bold("\n=== Verifikasi CUDA / GPU ===")

    cuda_check = """
import torch
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU count       : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} | {p.total_memory // 1024**3} GB | sm_{p.major}{p.minor}")
else:
    import platform
    if platform.system() == "Darwin":
        import torch.backends.mps as mps
        print(f"MPS available   : {mps.is_available()}")
"""

    result = subprocess.run(
        [sys.executable, "-c", cuda_check],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        warn(f"Error: {result.stderr.strip()[:120]}")

    # Khusus RTX 5090: cek sm_100
    if plat["nvidia"] and plat["has_5090"]:
        sm_check = """
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    if p.major >= 10:
        print("RTX 5090 Blackwell sm_100 — SIAP untuk training!")
    else:
        print(f"GPU compute capability: sm_{p.major}{p.minor}")
"""
        result2 = subprocess.run([sys.executable, "-c", sm_check],
                                  capture_output=True, text=True)
        if result2.returncode == 0:
            ok(result2.stdout.strip())


def verify_project_imports():
    """Cek apakah semua modul lokal project bisa di-import."""
    bold("\n=== Verifikasi Modul Project P-DARTS ===")

    student_dir = Path(__file__).resolve().parent
    os.chdir(student_dir)

    modules = [
        "nas_config",
        "genotypes",
        "operations",
        "model_search",
        "model_eval",
        "palm_vein_dataset",
        "architect",
        "utils",
    ]

    all_ok = True
    for mod in modules:
        result = subprocess.run(
            [sys.executable, "-c", f"import {mod}; print('OK')"],
            capture_output=True, text=True,
            cwd=str(student_dir)
        )
        if result.returncode == 0:
            ok(f"{mod:25s} — OK")
        else:
            err(f"{mod:25s} — GAGAL")
            # Tampilkan baris error yang relevan
            for line in result.stderr.strip().splitlines()[-3:]:
                print(f"    {line}")
            all_ok = False

    return all_ok


def print_summary(plat: dict, install_ok: bool, import_ok: bool):
    """Tampilkan ringkasan dan instruksi selanjutnya."""
    bold("\n" + "="*60)
    bold("  RINGKASAN SETUP")
    bold("="*60)

    info(f"OS           : {plat['os']} ({plat['arch']})")
    info(f"GPU NVIDIA   : {'✓ ' + ', '.join(plat['gpus']) if plat['nvidia'] else '✗ tidak ada'}")
    info(f"RTX 5090     : {'✓ YA' if plat['has_5090'] else '✗ tidak terdeteksi'}")
    info(f"Library      : {'✓ semua terinstall' if install_ok else '✗ ada yang gagal'}")
    info(f"Modul Project: {'✓ semua bisa diimport' if import_ok else '✗ ada yang gagal'}")

    if install_ok and import_ok:
        ok("\nSetup SELESAI — semua siap!")
        bold("\nLangkah selanjutnya:")
        print("  1. Jalankan P-DARTS search:")
        print("       cd Student")
        print("       python search.py")
        print()
        print("  2. Quick test (lebih cepat, 15 epoch per stage):")
        print("       python search.py --epochs_per_stage 15 --batch_size 32")
        print()
        print("  3. Setelah search selesai, retrain:")
        print("       python retrain.py --genotype nas_results/search/genotype_final.json")
        print()
        if plat["has_5090"]:
            print("  [RTX 5090] Tips untuk performa maksimal:")
            print("    - Gunakan batch_size 64-128 (VRAM 32GB)")
            print("    - Export: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
            print("    - Export: CUDA_VISIBLE_DEVICES=0")
    else:
        err("\nAda masalah dalam setup. Cek error di atas.")
        print("  - Pastikan CUDA Toolkit 12.6+ terinstall di sistem")
        print("  - Pastikan driver NVIDIA >= 560.x")
        print("  - Coba jalankan ulang: python setup_env.py")

    bold("="*60 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    bold("╔══════════════════════════════════════════════════════╗")
    bold("║  Setup Env — P-DARTS Palm Vein NAS                  ║")
    bold("║  Target: NVIDIA RTX 5090 (CUDA 12.6 / Blackwell)   ║")
    bold("╚══════════════════════════════════════════════════════╝")

    check_python_version()
    plat = detect_platform()
    upgrade_pip()
    install_pytorch(plat)
    install_dependencies()
    install_ok = verify_installation(plat)
    verify_cuda(plat)
    import_ok = verify_project_imports()
    print_summary(plat, install_ok, import_ok)


if __name__ == "__main__":
    main()
