# Fork deviations from NVIDIA/swe-serve

This fork is based on `NVIDIA/swe-serve` tag `v1.0.1` (`84d8fd9`). It adds one judging profile so that
six tasks can be graded on NVIDIA H200 hosts. Everything else is upstream content.

## Added profile: `h200_1`

H200 is `sm90` like H100, so `compute_capability` checks and every CUDA/Triton kernel path are
unchanged; only the device *name* differs. The six tasks below assert the device name, so upstream
grades them on H100 hosts only and has no profile that fits H200.

| task | change |
|---|---|
| `ss-sgl-gemma4-moe-core-serving` | `tests/model_assets.json` adds `h200_1` (clone of `h100_1`, `gpu_name_contains: "H200"`); `task.toml` adds it to `hardware_profiles` |
| `ss-sgl-gemma4-pcg-execution` | same, plus `validation_profiles` |
| `ss-sgl-gemma4-speculative-execution` | same (`online_quantization: "fp8"` kept) |
| `ss-sgl-qwen35-dense-moe-core-serving` | same (keeps `moe_server_args: ["--quantization", "fp8"]`) |
| `ss-sgl-cluster-hisparse-decode-lifecycle` | `tests/prep.sh` accepts `h200_1` with `expected_gpu=H200`; `task.toml` declares it |
| `ss-sgl-moe-lora-kernel-correctness` | same, plus `TORCH_CUDA_ARCH_LIST=9.0`; `task.toml` declares it |

Every other profile (`h100_1`, `h100_4`, `gb200_*`, `gb300_1`) and every assertion, F2P node list and
`score.py` threshold is untouched. All six tasks are correctness-only F2P (`f2p_mode = "correctness"`,
`performance_f2p_count = 0`), so no performance calibration is affected.

## Runner support

`run_task.py` and `scripts/preflight.py` select the profile instead of hard-coding `h100_1`:

```bash
python3 scripts/preflight.py --cache-root <cache> --profile h200_1 --task <task_id>
python3 run_task.py --profile h200_1 <task_id>
```

`--profile h200_1` prepares and mounts the `views/h200_1/<view>` asset lane (hardlinked from the same
pinned snapshots) and exports `SWE_SERVE_HARDWARE_PROFILE=h200_1` to the task containers. The default
stays `h100_1`.

## Comparability

Scores produced with `h200_1` are **not** interchangeable with the NVIDIA leaderboard, which is
measured on H100. Report them as a separate column.

## manifest.json

`manifest.json` records SHA-256 hashes of the execution files (`tasks/`, `scripts/`, `closed_book/`,
`run_task.py`, `task_list.csv`) and is regenerated after the change above, so
`python3 scripts/verify_manifest.py` passes on this fork. Because the manifest now describes the fork
rather than the release tag, use git to establish upstream identity:

```bash
git fetch origin --tags
git diff --stat v1.0.1 -- tasks scripts run_task.py   # the deviation set
```

Regenerate the manifest after further execution-file edits:

```bash
python3 - <<'PY'
import hashlib, importlib.util, json, sys
from pathlib import Path
root = Path.cwd()
spec = importlib.util.spec_from_file_location("vm", root / "scripts/verify_manifest.py")
vm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vm)
manifest = json.loads((root / "manifest.json").read_text())
changed = {}
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    rel = "/".join(path.relative_to(root).parts)
    if not vm.covered(rel):
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get(rel) != digest:
        changed[rel] = digest
    manifest[rel] = digest
diff = json.loads(json.dumps(manifest, sort_keys=True, indent=2))
(root / "manifest.json").write_text(diff + "\n")
print(json.dumps(changed, indent=2), file=sys.stderr)
PY
python3 scripts/verify_manifest.py
```
