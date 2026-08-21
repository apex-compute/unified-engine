#!/usr/bin/env bash
# One-command setup for pi0.5 + LIBERO on the UnifiedEngine.
#
#   bash models/pi05/setup_env.sh                # everything
#   bash models/pi05/setup_env.sh --osmesa-only  # just the OSMesa step
#   bash models/pi05/setup_env.sh --verify-only  # just re-run the checks
#
# INSTALLS INTO THE PYTHON ENV YOU ARE ALREADY IN -- it does NOT create one.
# Activate the env you want first (conda, venv, whatever), then run this. It pip-
# installs libero_requirements.txt, drops in the one library pip cannot supply
# (libOSMesa), seeds LIBERO's interactive first-run prompt, and verifies the result.
# Idempotent: safe to re-run.
#
# Needs NO sudo. `apt-get download` is a plain file download, not an install.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ="$REPO_ROOT/models/pi05/libero_requirements.txt"

MODE=all
[[ "${1:-}" == "--osmesa-only" ]] && MODE=osmesa
[[ "${1:-}" == "--verify-only" ]] && MODE=verify

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------- 1. use the active env
# PREFIX is where libOSMesa gets installed and what LD_LIBRARY_PATH must point at
# later, so we need a real, writable env root -- not the system python.
PREFIX="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"
[[ -n "$PREFIX" ]] || die "no active Python env detected (\$CONDA_PREFIX / \$VIRTUAL_ENV are unset).
   Activate the env you want pi05 installed into, then re-run:
       conda activate <your-env>     # or:  source <venv>/bin/activate
   This script deliberately does NOT create an env for you."
[[ -w "$PREFIX" ]] || die "active env is not writable: $PREFIX"

say "using active env: $PREFIX  ($(python -V 2>&1))"

if [[ $MODE == all ]]; then
  # cd to repo root: the LIBERO editable install clones to ./src/libero
  say "pip install -r $REQ"
  ( cd "$REPO_ROOT" && pip install -r "$REQ" )
fi

# ------------------------------------------------------------------ 2. OSMesa
# The ONE dependency no Python package manager supplies. conda-forge's mesalib
# ships GL/GLX/EGL and llvmpipe but NOT libOSMesa (verified absent in 25.2.8 and
# 26.1.6), there is no standalone `osmesa` package, and pip has nothing at all.
# So it is fetched from the Debian archive and unpacked into the active env's
# lib/ -- no sudo, nothing touched outside $PREFIX. EGL is not a substitute: robosuite's
# EGLGLContext needs the PLATFORM_DEVICE extension and therefore /dev/dri, which
# is denied unless you are in the render/video groups.
if [[ $MODE == all || $MODE == osmesa ]]; then
  if [[ -f "$PREFIX/lib/libOSMesa.so.8" ]]; then
    say "libOSMesa already present -- skipping"
  else
    say "installing libOSMesa into $PREFIX/lib (no sudo)"
    TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
    ( cd "$TMP" && apt-get download libosmesa6 ) \
      || die "apt-get download libosmesa6 failed. If you are not on Debian/Ubuntu,
   install libOSMesa by hand into $PREFIX/lib, or use MUJOCO_GL=egl if you have a
   usable /dev/dri render node and are in the 'render' group.
   NOTE: this script no longer creates a conda env, so it no longer pulls conda-forge's
   mesalib/libglu either. If verification fails on a missing libGL/libGLU rather than
   libOSMesa, install those into the active env yourself (conda-forge mesalib libglu)
   or via your distro."
    dpkg -x "$TMP"/libosmesa6_*.deb "$TMP/root"
    cp "$TMP/root/usr/lib/x86_64-linux-gnu/libOSMesa.so.8.0.0" "$PREFIX/lib/"
    ln -sf libOSMesa.so.8.0.0 "$PREFIX/lib/libOSMesa.so.8"
    ln -sf libOSMesa.so.8.0.0 "$PREFIX/lib/libOSMesa.so"
    say "libOSMesa installed"
  fi
fi

# ------------------------------------------------- 3. LIBERO first-run config
# First `import libero.libero` asks an interactive question; answer it with the
# defaults so no later run blocks on stdin.
if [[ $MODE == all ]]; then
  if [[ ! -f "$HOME/.libero/config.yaml" ]]; then
    say "seeding LIBERO's first-run config prompt"
    printf 'N\n' | python -c "import libero.libero" >/dev/null 2>&1 || true
  fi
fi

# ------------------------------------------------------------------ 4. verify
say "verifying"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH="$PREFIX/lib"

python - <<'PY' || die "verification failed -- see models/pi05/README.md section 5 (Troubleshooting)"
import functools, os, sys

# torch >= 2.6 flipped torch.load's weights_only default to True, which refuses
# LIBERO's pickled-numpy init states. libero_eval.py installs this shim itself.
import torch
torch.load = functools.partial(torch.load, weights_only=False)

# NEVER trust pip's success message for LIBERO -- it happily "installs" an empty
# metadata-only package. Import is the only real check.
import libero.libero
print(f"  libero      OK  {libero.libero.__file__}")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bm = benchmark.get_benchmark_dict()["libero_object"]()
t = bm.get_task(0)
print(f"  benchmark   OK  libero_object has {bm.n_tasks} tasks")

f = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=f, camera_heights=256, camera_widths=256)
env.seed(7); env.reset()
o = env.set_init_state(bm.get_task_init_states(0)[0])
img = o["agentview_image"]
assert img.shape == (256, 256, 3) and img.any(), "renderer produced an empty frame"
env.step([0, 0, 0, 0, 0, 0, -1])
env.close()
print(f"  osmesa      OK  rendered {img.shape}, sim steps")
print(f"  task 0      \"{t.language}\"")
PY

cat <<EOF

$(printf '\033[32m%s\033[0m' "SETUP COMPLETE")

With your env active, every LIBERO run needs these two exports:

    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=$PREFIX/lib

Next:
    # single inference on the FPGA (~50s, downloads+exports weights on first run)
    python models/pi05/pi05_test.py --engines max

    # closed-loop episode with video
    python models/pi05/utility/libero_eval.py --backend fpga \\
        --task-suite libero_object --tasks 1 --trials 1 --engines max

See models/pi05/README.md for the full command reference.
EOF
