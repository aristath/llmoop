# Repository rules

## Test execution

- Never run tests in parallel in this repository. Reliability is more important than speed.
- Every test command must explicitly select sequential execution, even when the test runner is sequential by default.
- Rust tests must use `-- --test-threads=1` (plus `--exact` when targeting a specific test).
- Do not run a broad test filter that can initialize Vulkan. Vulkan tests must be selected individually and run sequentially.
- If a test runner cannot guarantee sequential execution, do not run it until a safe sequential invocation is established.

## GPU residency

- Do not use the NVIDIA GPU for any llmoop workload. This includes model execution, tests, benchmarks, compilation probes, device enumeration, and diagnostic probes.
- Use only AMD GPUs that have been verified idle immediately before the workload, and verify that they returned to their idle baseline immediately afterward.
- Never load a model or start a GPU-executing test on a GPU that already has a resident workload.
- Before loading anything onto a GPU, inspect that device, unload existing workloads from it, and verify that the unload completed.
- Do not co-locate an llmoop test or model with another model server merely because free VRAM appears sufficient.
- If an existing GPU workload cannot be safely unloaded and verified, use a different idle GPU or do not start the new workload.
