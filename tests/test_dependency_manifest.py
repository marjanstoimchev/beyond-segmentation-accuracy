"""The dependency-manifest test: every import in src/ must be stdlib, local, or declared in pyproject —
and the core must import without the deep-learning stack."""
import ast
import subprocess
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MODULE_TO_DIST = {"cv2": "opencv-python-headless", "yaml": "PyYAML", "PIL": "Pillow", "skimage": "scikit-image", "segmentation_models_pytorch": "segmentation-models-pytorch",
                  "sklearn": "scikit-learn", "instanseg": "instanseg-torch", "pytorch_lightning": "lightning", "sam2": "SAM-2"}
TORCH_ONLY_IN = ("mxdata/data", "mxseg", "kodiak", "dinov3")      # where torch may be imported at module scope


def _declared():
    py = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = py["project"]["dependencies"] + [d for v in py["project"]["optional-dependencies"].values() for d in v]
    names = set()
    for d in deps:
        name = d.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
        names.add(name.lower())
    return names


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0], node.lineno


def test_every_import_is_declared():
    declared = _declared()
    stdlib = set(sys.stdlib_module_names)
    bad = []
    for f in SRC.rglob("*.py"):
        rel = f.relative_to(SRC).as_posix()
        for mod, line in _imports(f):
            if mod in stdlib or mod in ("mxdata", "mxseg", "kodiak", "dinov3", "tests"):
                continue
            dist = MODULE_TO_DIST.get(mod, mod).lower()
            if dist not in declared:
                bad.append(f"{f.relative_to(ROOT)}:{line} imports {mod!r} ({dist}) which pyproject does not declare")
            if mod == "torch" and not any(rel.startswith(t) for t in TORCH_ONLY_IN):
                bad.append(f"{f.relative_to(ROOT)}:{line} imports torch outside {TORCH_ONLY_IN}")
    assert not bad, "\n".join(bad)


def test_core_imports_without_torch():
    code = "import sys, mxdata, mxdata.pipeline, mxdata.cli; assert 'torch' not in sys.modules, 'torch was imported'; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]
