from __future__ import annotations

from hashlib import sha256
from pathlib import Path


COMPILER_FINGERPRINT_SCHEMA = "llmoop.package_compiler_sha256.v1"


def package_compiler_fingerprint(shader_source_dir: Path) -> str:
    compiler_dir = Path(__file__).resolve().parent
    inputs = [
        *((f"llmoop/{path.name}", path) for path in compiler_dir.glob("*.py")),
        *(
            (f"runtime-rs/shaders/{path.name}", path)
            for path in shader_source_dir.iterdir()
            if path.is_file()
        ),
    ]
    digest = sha256()
    for relative_path, source_path in sorted(inputs):
        path_bytes = relative_path.encode("utf-8")
        source_bytes = source_path.read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "little"))
        digest.update(path_bytes)
        digest.update(len(source_bytes).to_bytes(8, "little"))
        digest.update(source_bytes)
    return f"{COMPILER_FINGERPRINT_SCHEMA}:{digest.hexdigest()}"
