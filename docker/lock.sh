#!/usr/bin/env bash
# Regenerate the hash-locked requirements the Docker images install with `pip install --require-hashes`.
# Run from the repository root after changing dependencies, or to pick up new releases:  bash docker/lock.sh
#
#   requirements-cpu.txt   python:3.14-slim, torch from PyTorch's CPU index; linux amd64 and arm64 (both wheels' hashes)
#   requirements-cuda.txt  pytorch/pytorch (CUDA 12.8, Python 3.12): what OpenDecider adds, locked against the base
#                          image's own packages (docker/cuda-base-packages.txt); its torch and the packages only torch
#                          needs are left out, so the image keeps its CUDA build of torch
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH_CUDA="2.11.0"   # the torch in docker/Dockerfile.cuda's base image
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
printf 'setuptools>=68\nwheel\n' > "$tmp/build.in"   # the build backend, so `pip install --no-build-isolation .` works
lock=(uv pip compile pyproject.toml "$tmp/build.in" --extra serve --extra small --generate-hashes --no-header -q)
cpu=(--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
     --index-strategy unsafe-best-match --python-version 3.14)

"${lock[@]}" "${cpu[@]}" --python-platform x86_64-manylinux_2_28 -o "$tmp/amd64.txt"
"${lock[@]}" "${cpu[@]}" --python-platform aarch64-manylinux_2_28 -o "$tmp/arm64.txt"
# the base image's packages as constraints, apart from its CUDA stack (its torch is a +cu128 build that is not on PyPI,
# so it is constrained by version, and PyPI's torch of that version declares other CUDA libraries; both are left to the
# base image below anyway)
grep -vE '^(#|nvidia-|cuda-|triton)' docker/cuda-base-packages.txt | sed -E 's/^torch==([0-9.]+)\+.*/torch==\1/' \
    > "$tmp/torch.txt"
grep -q "^torch==$TORCH_CUDA$" "$tmp/torch.txt" || { echo "docker/cuda-base-packages.txt is not torch $TORCH_CUDA" >&2; exit 1; }
"${lock[@]}" --python-version 3.12 --python-platform x86_64-manylinux_2_28 --constraint "$tmp/torch.txt" \
    -o "$tmp/cuda.txt"

python3 - "$tmp" <<'PY'
import re, sys
tmp = sys.argv[1]

# CPU: one file for both architectures; the two locks differ only in platform-specific wheel hashes
a, b = (open(f"{tmp}/{x}.txt").read().split("\n") for x in ("amd64", "arm64"))
if len(a) != len(b):
    sys.exit("the amd64 and arm64 locks resolved different packages; lock them separately")
out = []
for x, y in zip(a, b):
    if x != y:
        if "--hash=sha256:" not in x or "--hash=sha256:" not in y:
            sys.exit(f"the amd64 and arm64 locks differ beyond hashes: {x!r} vs {y!r}")
        out += [x if x.rstrip().endswith("\\") else x + " \\", y]
    else:
        out.append(x)
open("docker/requirements-cpu.txt", "w").write("\n".join(out).rstrip("\n") + "\n")

# CUDA: drop torch and every package that only torch (transitively) needs, from uv's "# via" annotations
blocks, cur = [], None
for line in open(f"{tmp}/cuda.txt").read().split("\n"):
    m = re.match(r"^([A-Za-z0-9_.\-]+)==", line)
    if m:
        cur = {"name": m.group(1).lower(), "lines": [line], "via": set()}
        blocks.append(cur)
        continue
    if cur is None:
        continue
    cur["lines"].append(line)
    s = line.strip()
    if not s.startswith("#"):
        continue
    body = s.lstrip("#").strip()
    if body.startswith("via"):
        body = body[3:].strip()
    if "pyproject.toml" in body or body.startswith(("-r", "-c")) or "build.in" in body or "torch.txt" in body:
        cur["via"].add("<root>")
    elif body:
        cur["via"].add(body.split()[0].lower())
drop, changed = {"torch"}, True
while changed:
    changed = False
    for blk in blocks:
        if blk["name"] not in drop and blk["via"] and blk["via"] <= drop:
            drop.add(blk["name"])
            changed = True
keep = [blk for blk in blocks if blk["name"] not in drop]
open("docker/requirements-cuda.txt", "w").write("\n".join(l for blk in keep for l in blk["lines"]).rstrip("\n") + "\n")
print("cpu:", sum(1 for l in out if re.match(r"^[A-Za-z0-9]", l)), "packages; cuda:", len(keep),
      "packages (left to the base image:", ", ".join(sorted(drop)) + ")")
PY
