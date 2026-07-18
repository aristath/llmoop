# Repository rules

## Test execution

- Never run tests in parallel in this repository. Reliability is more important than speed.
- Every test command must explicitly select sequential execution, even when the test runner is sequential by default.
- Rust tests must use `-- --test-threads=1` (plus `--exact` when targeting a specific test).
- Do not run a broad test filter that can initialize Vulkan. Vulkan tests must be selected individually and run sequentially.
- If a test runner cannot guarantee sequential execution, do not run it until a safe sequential invocation is established.
