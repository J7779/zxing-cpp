#!/usr/bin/env python3
"""
update_model.py - Update the embedded NanoDet ONNX model and rebuild WASM

Converts an ONNX model file into a C++ byte array, then optionally rebuilds
the WASM so the new model is bundled into the binary.

Usage:
    python update_model.py                        # Use default nanodet_barcode.onnx
    python update_model.py path/to/model.onnx     # Use a custom model file
    python update_model.py --no-build             # Generate C++ only, skip WASM build
    python update_model.py --no-copy              # Build but don't copy to wrappers/wasm/
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Paths (relative to this script's location) ──────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "nanodet_barcode.onnx"
CPP_OUTPUT    = SCRIPT_DIR / "core" / "src" / "onnx" / "NanoDetModelData.cpp"
HEADER_FILE   = SCRIPT_DIR / "core" / "src" / "onnx" / "NanoDetModelData.h"
BUILD_DIR     = SCRIPT_DIR / "build-wasm"
WASM_SRC_DIR  = SCRIPT_DIR / "wrappers" / "wasm"
WASM_CMAKE    = WASM_SRC_DIR / "CMakeLists.txt"
EMSDK_DIR     = SCRIPT_DIR / "emsdk"


def generate_cpp(model_path: Path, output_path: Path) -> int:
    """Convert ONNX binary to C++ byte array. Returns model size in bytes."""
    print(f"\n{'─'*60}")
    print(f"  Model:  {model_path}")
    print(f"  Output: {output_path}")
    print(f"{'─'*60}")

    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    data = model_path.read_bytes()
    model_size = len(data)
    print(f"  Model size: {model_size:,} bytes ({model_size / 1024 / 1024:.2f} MB)")

    # Validate ONNX magic bytes (protobuf field tag 0x08 for ir_version)
    if len(data) < 4 or data[0] != 0x08:
        print("WARNING: File doesn't look like a valid ONNX model (unexpected first byte).")
        resp = input("  Continue anyway? [y/N] ").strip().lower()
        if resp != "y":
            sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with open(output_path, "w") as f:
        f.write(f"// Auto-generated from {model_path.name} - DO NOT EDIT\n")
        f.write("// Re-generate with:  python update_model.py\n")
        f.write("// This file embeds the NanoDet barcode detection ONNX model as a C++ byte array.\n\n")
        f.write('#include "NanoDetModelData.h"\n\n')
        f.write(f"const unsigned int NANODET_MODEL_SIZE = {model_size};\n\n")
        f.write(f"const unsigned char NANODET_MODEL_DATA[{model_size}] = {{\n")

        # Write 20 bytes per line
        for i in range(0, model_size, 20):
            chunk = data[i : i + 20]
            hex_vals = ",".join(f"0x{b:02x}" for b in chunk)
            trailing = "," if i + 20 < model_size else ""
            f.write(hex_vals + trailing + "\n")

        f.write("};\n")

    elapsed = time.perf_counter() - t0
    cpp_size = output_path.stat().st_size
    print(f"  Generated {cpp_size:,} bytes of C++ ({elapsed:.1f}s)")
    return model_size


def find_emscripten() -> dict:
    """Find Emscripten tools and return env dict, or None."""
    emcc = shutil.which("emcc")
    if emcc:
        return {}  # already on PATH

    # Try local emsdk
    if not EMSDK_DIR.exists():
        return None

    em_dir = EMSDK_DIR / "upstream" / "emscripten"
    python_dirs = list((EMSDK_DIR / "python").glob("*"))
    node_dirs = list((EMSDK_DIR / "node").glob("*"))

    extra_path = [str(em_dir), str(EMSDK_DIR)]
    if python_dirs:
        extra_path.append(str(python_dirs[0]))
    if node_dirs:
        extra_path.append(str(node_dirs[0] / "bin"))

    env = os.environ.copy()
    env["PATH"] = ";".join(extra_path) + ";" + env.get("PATH", "")
    env["EMSDK"] = str(EMSDK_DIR)
    if node_dirs:
        env["EMSDK_NODE"] = str(node_dirs[0] / "bin" / "node.exe")

    # Verify emcc is now reachable
    try:
        subprocess.run(
            ["emcc", "--version"], capture_output=True, env=env, check=True
        )
        return env
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_wasm(env_override: dict | None) -> bool:
    """Run cmake configure + build. Returns True on success."""
    print(f"\n{'─'*60}")
    print("  Building WASM...")
    print(f"{'─'*60}")

    env = os.environ.copy()
    if env_override:
        env.update(env_override)

    BUILD_DIR.mkdir(exist_ok=True)

    # Configure
    print("  [1/2] Configuring (emcmake cmake)...")
    cfg = subprocess.run(
        [
            "emcmake", "cmake",
            str(WASM_CMAKE.parent),
            "-DZXING_READERS=ON",
            "-DZXING_WRITERS=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=str(BUILD_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    if cfg.returncode != 0:
        print("  ERROR: cmake configure failed:\n" + cfg.stderr, file=sys.stderr)
        return False
    print("  Configure OK")

    # Build
    print("  [2/2] Building (cmake --build)...")
    bld = subprocess.run(
        ["cmake", "--build", "."],
        cwd=str(BUILD_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    if bld.returncode != 0:
        print("  ERROR: build failed:\n" + bld.stderr, file=sys.stderr)
        return False

    # Count build steps from ninja output
    lines = [l for l in bld.stdout.splitlines() if l.startswith("[")]
    if lines:
        print(f"  Build OK ({len(lines)} steps)")
    else:
        print("  Build OK")

    return True


def copy_artifacts():
    """Copy build output to wrappers/wasm/ for Live Server."""
    print(f"\n{'─'*60}")
    print("  Copying build artifacts to wrappers/wasm/")
    print(f"{'─'*60}")

    artifacts = [
        "zxing.js", "zxing.wasm",
        "zxing_reader.js", "zxing_reader.wasm",
        "zxing_writer.js", "zxing_writer.wasm",
        "demo_cam_reader.html", "demo_reader.html", "demo_writer.html",
    ]

    for name in artifacts:
        src = BUILD_DIR / name
        dst = WASM_SRC_DIR / name
        if src.exists():
            shutil.copy2(src, dst)
            size_kb = src.stat().st_size / 1024
            print(f"  {name:30s} {size_kb:8.1f} KB")
        else:
            print(f"  {name:30s}  (not found, skipped)")


def main():
    parser = argparse.ArgumentParser(
        description="Update embedded NanoDet ONNX model and rebuild WASM"
    )
    parser.add_argument(
        "model", nargs="?", default=str(DEFAULT_MODEL),
        help=f"Path to ONNX model file (default: {DEFAULT_MODEL.name})"
    )
    parser.add_argument(
        "--no-build", action="store_true",
        help="Only generate the C++ byte array, skip WASM build"
    )
    parser.add_argument(
        "--no-copy", action="store_true",
        help="Build WASM but don't copy artifacts to wrappers/wasm/"
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()

    print("=" * 60)
    print("  NanoDet Model Updater")
    print("=" * 60)

    # Step 1: Generate C++
    model_size = generate_cpp(model_path, CPP_OUTPUT)

    if args.no_build:
        print(f"\n  Done (--no-build). Remember to rebuild WASM manually.")
        return

    # Step 2: Find Emscripten
    print("\n  Looking for Emscripten...")
    em_env = find_emscripten()
    if em_env is None:
        print("  WARNING: Emscripten not found. C++ file was generated but WASM not built.")
        print("  To build manually:")
        print(f"    cd {BUILD_DIR}")
        print(f"    emcmake cmake {WASM_CMAKE.parent} -DZXING_READERS=ON -DZXING_WRITERS=ON -DCMAKE_BUILD_TYPE=Release")
        print(f"    cmake --build .")
        return
    print("  Emscripten found ✓")

    # Step 3: Build WASM
    if not build_wasm(em_env):
        sys.exit(1)

    # Step 4: Verify model is embedded
    wasm_file = BUILD_DIR / "zxing_reader.wasm"
    if wasm_file.exists():
        wasm_size = wasm_file.stat().st_size
        print(f"\n  zxing_reader.wasm: {wasm_size:,} bytes ({wasm_size / 1024 / 1024:.2f} MB)")
        if wasm_size > model_size:
            print(f"  Model ({model_size:,} bytes) is embedded ✓")
        else:
            print("  WARNING: WASM is smaller than the model — something may be wrong.")

    # Step 5: Copy to wrappers/wasm/
    if not args.no_copy:
        copy_artifacts()

    print(f"\n{'='*60}")
    print("  All done! Reload your browser to use the new model.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
