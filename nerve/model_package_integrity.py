from nerve.model_package_common import *
from nerve.model_package_assets import *

def build_package_artifact_integrity(package_dir: Path) -> Json:
    files = {}
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_dir)
        if (
            relative.parts[0] == WEIGHTS_PACKAGE_DIR
            or relative.name == "vulkan_resident_package.json"
        ):
            continue
        payload = path.read_bytes()
        files[relative.as_posix()] = {
            "byte_count": len(payload),
            "sha256": sha256(payload).hexdigest(),
        }
    return {
        "schema": PACKAGE_ARTIFACT_INTEGRITY_SCHEMA,
        "algorithm": "sha256",
        "files": files,
    }


