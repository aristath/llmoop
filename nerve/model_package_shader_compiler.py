from nerve.model_package_common import *
from nerve.model_package_shader_templates import *

def compile_shader_artifacts(
    shader_dir: Path,
    *,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    compiler = shutil.which("glslangValidator")
    if compiler is None:
        raise ModelCompileError(
            "compiling a Vulkan model package requires glslangValidator"
        )

    sources = sorted(shader_dir.glob("*.comp"))
    if not sources:
        raise ModelCompileError(
            f"no Vulkan shader sources were rendered in {shader_dir}"
        )
    total = len(sources)
    for index, source in enumerate(sources, start=1):
        check_compile_cancelled(cancel_requested)
        if progress is not None:
            progress(index, total, source.name)
        destination = source.with_suffix(".spv")
        completed = subprocess.run(
            [
                compiler,
                "-V",
                "--target-env",
                "vulkan1.4",
                str(source),
                "-o",
                str(destination),
            ],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise ModelCompileError(
                f"failed to compile Vulkan shader {source}: {diagnostic}"
            )
        compiled = destination.read_bytes()
        if len(compiled) < 4 or compiled[:4] != b"\x03\x02#\x07":
            raise ModelCompileError(
                f"shader compiler produced invalid SPIR-V artifact {destination}"
            )
        source.unlink()


def compiled_shader_path(source_path: str) -> str:
    if not source_path.endswith(".comp"):
        raise ModelCompileError(
            f"compiled Vulkan shader source path must end in .comp: {source_path!r}"
        )
    return f"{source_path[:-5]}.spv"


