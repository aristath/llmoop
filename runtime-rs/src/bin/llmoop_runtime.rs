use std::collections::BTreeMap;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use llmoop_runtime::{
    CircuitPort, PedalCablePlacement, PedalPlacement, RUNTIME_TOPOLOGY_SCHEMA,
    RuntimeAvailableDevice, RuntimeAvailableMemoryHeap, RuntimeBoundDevice,
    RuntimeCableRouteTarget, RuntimeCableRoutes, RuntimeCapacityProfileSummary,
    RuntimeCompiledPedalboardSummary, RuntimeDeviceBindings, RuntimeDeviceSliceReport,
    RuntimeDeviceTickPlanReport, RuntimeEffectivePedalboardTopology, RuntimeLocalCableBufferReport,
    RuntimePackageInspectionReport, RuntimePatchControls, RuntimePatchDuplicateAfterControl,
    RuntimePatchInspectionReport, RuntimePatchPlacementReport, RuntimePatchSourceChainEntry,
    RuntimePedalPortSummary, RuntimePlacedPedalDispatchTimingReport,
    RuntimePlacedPedalTimingReport, RuntimePlacedPedalTimingSummaryReport,
    RuntimePlacedPromptRunReport, RuntimePlacedTransportReport, RuntimePlacedTransportStatsReport,
    RuntimePlacementReport, RuntimePromptBenchmarkReport, RuntimePromptBenchmarkRunReport,
    RuntimePromptBenchmarkTransportTotalsReport, RuntimePromptBenchmarkU64MetricReport,
    RuntimePromptBenchmarkUsizeMetricReport, RuntimePromptTimingReport,
    RuntimeRemoteCableBufferReport, RuntimeSingleDevicePromptRunReport, RuntimeSourcePedal,
    RuntimeTokenizerOptionsReport, RuntimeTopologyReport, VulkanComputeDevice,
    VulkanPlacedCableTransportStats, VulkanResidentGreedyInProcessPlacedPromptEngine,
    VulkanResidentGreedyInProcessPlacedPromptEngineInputRequest,
    VulkanResidentGreedyInProcessPlacedPromptEventRun,
    VulkanResidentGreedyInProcessPlacedPromptStream, VulkanResidentGreedyModelPackage,
    VulkanResidentGreedyModelPackageDeviceSlice, VulkanResidentGreedyModelPackageManifest,
    VulkanResidentHfTokenizerTextCodec, VulkanResidentTokenEngine,
    VulkanResidentTokenEngineRunBudget, VulkanResidentTokenEngineRunStopCondition,
    VulkanResidentTokenInputEvent, VulkanResidentTokenTextCodec,
    VulkanReusableKernelArtifactManifest,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Args {
    package_manifest: Option<PathBuf>,
    prompt: Option<String>,
    inspect_runtime: bool,
    inspect_package: bool,
    inspect_patch: bool,
    inspect_placement: bool,
    inspect_device_slice: Option<String>,
    default_device_id: Option<String>,
    pedal_devices: BTreeMap<String, String>,
    device_bindings: BTreeMap<String, String>,
    duplicate_after: Vec<(String, String)>,
    source_chain: Option<Vec<(String, String)>>,
    max_new_tokens: usize,
    capacity: Option<usize>,
    vulkan_device_index: Option<usize>,
    cycle_ticks: usize,
    max_scheduler_turns: usize,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    generated_only: bool,
    profile: bool,
    profile_runs: usize,
    json: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            package_manifest: None,
            prompt: None,
            inspect_runtime: false,
            inspect_package: false,
            inspect_patch: false,
            inspect_placement: false,
            inspect_device_slice: None,
            default_device_id: None,
            pedal_devices: BTreeMap::new(),
            device_bindings: BTreeMap::new(),
            duplicate_after: Vec::new(),
            source_chain: None,
            max_new_tokens: 4,
            capacity: None,
            vulkan_device_index: None,
            cycle_ticks: 4,
            max_scheduler_turns: 1_024,
            add_special_tokens: true,
            skip_special_tokens: true,
            generated_only: false,
            profile: false,
            profile_runs: 1,
            json: false,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("llmoop-runtime error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    if std::env::args()
        .skip(1)
        .any(|arg| arg == "--help" || arg == "-h")
    {
        print_usage();
        return Ok(());
    }

    let args = parse_args().map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let package_manifest = args.package_manifest.as_ref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "--package is required; run `python -m llmoop --compile-model <MODEL_DIR>` first",
        )
    })?;
    let manifest_dir = package_manifest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    if args.inspect_runtime {
        let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(package_manifest)?;
        return inspect_runtime_topology(&args, package_manifest, &manifest_dir, manifest);
    }
    if args.inspect_package {
        let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(package_manifest)?;
        return inspect_package(&args, package_manifest, &manifest_dir, manifest);
    }
    if args.inspect_patch {
        let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(package_manifest)?;
        return inspect_patch(&args, package_manifest, &manifest_dir, manifest);
    }
    let manifest = runtime_manifest(&args, package_manifest)?;
    if args.inspect_placement {
        return inspect_placement(&args, package_manifest, &manifest_dir, manifest);
    }
    if let Some(device_id) = args.inspect_device_slice.as_deref() {
        return inspect_device_slice(&args, package_manifest, &manifest_dir, manifest, device_id);
    }
    let prompt = args
        .prompt
        .as_ref()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--prompt is required"))?;
    let tokenizer_dir = tokenizer_dir_from_package(package_manifest)?;
    let codec = VulkanResidentHfTokenizerTextCodec::from_model_dir(&tokenizer_dir)?
        .with_add_special_tokens(args.add_special_tokens)
        .with_skip_special_tokens(args.skip_special_tokens);
    let prompt_ids = codec.encode_text(prompt)?;
    let needed_capacity = prompt_ids
        .len()
        .checked_add(args.max_new_tokens)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "prompt token count plus --max-new-tokens overflowed usize",
            )
        })?;
    let capacity = choose_runtime_capacity(package_manifest, args.capacity, needed_capacity)?;

    if manifest.placement_device_ids().len() > 1 {
        if args.profile_runs > 1 {
            return run_placed_prompt_benchmark(
                &args,
                package_manifest,
                &manifest_dir,
                &tokenizer_dir,
                prompt,
                &prompt_ids,
                capacity,
                manifest,
                &codec,
            );
        }
        return run_placed_prompt(
            &args,
            package_manifest,
            &manifest_dir,
            &tokenizer_dir,
            prompt,
            &prompt_ids,
            capacity,
            manifest,
            &codec,
        );
    }

    if args.profile_runs > 1 {
        return run_single_device_prompt_benchmark(
            &args,
            package_manifest,
            &manifest_dir,
            &tokenizer_dir,
            prompt,
            &prompt_ids,
            needed_capacity,
            capacity,
            manifest,
            &codec,
        );
    }

    let report = execute_single_device_prompt_run(
        &args,
        package_manifest,
        &manifest_dir,
        &tokenizer_dir,
        prompt,
        needed_capacity,
        capacity,
        manifest,
        &codec,
    )?;
    print_single_device_prompt_report(&args, &report)?;

    Ok(())
}

fn execute_single_device_prompt_run(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    prompt: &str,
    needed_capacity: usize,
    capacity: usize,
    manifest: VulkanResidentGreedyModelPackageManifest,
    codec: &VulkanResidentHfTokenizerTextCodec,
) -> Result<RuntimeSingleDevicePromptRunReport, Box<dyn Error>> {
    let setup_start = Instant::now();
    let device = runtime_vulkan_device(args)?;
    let model = VulkanResidentGreedyModelPackage::from_manifest(
        &device,
        manifest_dir,
        manifest,
        Some(capacity),
    )?;
    let mut engine = VulkanResidentTokenEngine::new(device);
    engine.add_model_package("compiled_model", model)?;
    engine.create_stream_from_model("compiled_model", "main")?;
    let setup_time_ns = elapsed_nanos_u64(setup_start);

    let run_start = Instant::now();
    let turn = engine.submit_live_text_turn_until_idle(
        "main",
        "prompt",
        prompt,
        args.max_new_tokens,
        "cli",
        VulkanResidentTokenEngineRunBudget::new(args.max_scheduler_turns, 1, args.cycle_ticks),
        codec,
    )?;
    let run_time_ns = elapsed_nanos_u64(run_start);
    let stream = engine
        .stream("main")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "runtime stream disappeared"))?;
    let snapshot = engine.snapshot();
    let scheduler_turns = turn.scheduler_turn_count();
    let runtime_cycles = turn.runtime_cycle_count;
    let generated_token_count = turn.generated_token_ids.len();
    let timing = runtime_prompt_timing_report(
        setup_time_ns,
        run_time_ns,
        generated_token_count,
        runtime_cycles,
        scheduler_turns,
    );

    Ok(RuntimeSingleDevicePromptRunReport {
        ok: true,
        execution_mode: "single_device_resident".to_string(),
        package_manifest: package_manifest.to_path_buf(),
        tokenizer_dir: tokenizer_dir.to_path_buf(),
        device_name: snapshot.device_name,
        device_id: stream.device_id.clone(),
        runtime_patch: runtime_patch_report(args),
        device_bindings: runtime_device_bindings_report(args, &[stream.device_id.clone()]),
        pedal_count: stream.pedal_count,
        dispatches_per_tick: stream.per_tick_dispatch_count,
        descriptors_per_tick: stream.per_tick_descriptor_count,
        push_constant_bytes_per_tick: stream.per_tick_push_constant_byte_count,
        resident_capacity_activations: stream.dynamic_state_capacity_activations,
        needed_capacity_activations: needed_capacity,
        tokenizer: tokenizer_options_report(args),
        prompt_text: prompt.to_string(),
        prompt_ids: turn.queued_input_event.encoded_token_ids.clone(),
        generated_ids: turn.generated_token_ids.clone(),
        generated_text: turn.generated_text.clone(),
        output_text: turn.output_text.clone(),
        stop_reason: engine_stop_label(turn.stop_condition).to_string(),
        scheduler_turns,
        runtime_cycles,
        timing,
    })
}

fn print_single_device_prompt_report(
    args: &Args,
    report: &RuntimeSingleDevicePromptRunReport,
) -> Result<(), Box<dyn Error>> {
    if args.json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else if args.generated_only {
        print_text(&report.generated_text);
    } else {
        print_text(&report.output_text);
        if args.profile {
            print_prompt_timing_profile(&report.timing);
        }
    }
    Ok(())
}

fn run_placed_prompt(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    prompt: &str,
    prompt_ids: &[u32],
    capacity: usize,
    manifest: VulkanResidentGreedyModelPackageManifest,
    codec: &VulkanResidentHfTokenizerTextCodec,
) -> Result<(), Box<dyn Error>> {
    let report = execute_placed_prompt_run(
        args,
        package_manifest,
        manifest_dir,
        tokenizer_dir,
        prompt,
        prompt_ids,
        capacity,
        manifest,
        codec,
    )?;
    print_placed_prompt_report(args, &report)
}

fn execute_placed_prompt_run(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    prompt: &str,
    prompt_ids: &[u32],
    capacity: usize,
    manifest: VulkanResidentGreedyModelPackageManifest,
    codec: &VulkanResidentHfTokenizerTextCodec,
) -> Result<RuntimePlacedPromptRunReport, Box<dyn Error>> {
    let setup_start = Instant::now();
    let mut logical_device_ids = manifest.placement_device_ids();
    if !logical_device_ids.contains(&manifest.device_id) {
        logical_device_ids.push(manifest.device_id.clone());
    }
    let placement = runtime_manifest_placement(manifest_dir, &manifest)?;
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let stream = VulkanResidentGreedyInProcessPlacedPromptStream::from_manifest_for_bound_devices(
        bound_devices.devices.clone(),
        manifest_dir,
        manifest,
        Some(capacity),
    )?;
    let mut engine = VulkanResidentGreedyInProcessPlacedPromptEngine::new();
    let stream_snapshot = engine.add_stream("main", stream)?;
    let setup_time_ns = elapsed_nanos_u64(setup_start);
    let run_start = Instant::now();
    let input_event =
        VulkanResidentTokenInputEvent::new("prompt", prompt_ids.to_vec(), args.max_new_tokens);
    let input_event_id = input_event.id.clone();
    let batch_run = engine.submit_input_events_until_idle_bounded(
        vec![VulkanResidentGreedyInProcessPlacedPromptEngineInputRequest::new("main", input_event)],
        1,
        args.max_scheduler_turns,
    )?;
    let run_time_ns = elapsed_nanos_u64(run_start);
    let run = batch_run
        .engine_run
        .input_runs
        .into_iter()
        .find(|run| run.stream_id == "main" && run.submitted_run.input_event.id == input_event_id)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "placed prompt engine run loop did not return the submitted prompt event run",
            )
        })?
        .submitted_run
        .session_run
        .run;
    let generated_text = codec.decode_tokens(&run.generated_token_ids)?;
    let output_text = codec.decode_tokens(&run.output_token_ids)?;
    let total_scheduler_turns = run
        .tick_runs
        .iter()
        .map(|tick| tick.tick_run.placed_run.scheduler_turn_count)
        .sum::<usize>();
    let completed_stage_deltas = run
        .tick_runs
        .iter()
        .map(|tick| tick.tick_run.placed_run.completed_stage_delta)
        .collect::<Vec<_>>();
    let tick_count = run.tick_runs.len();
    let generated_token_count = run.generated_token_ids.len();
    let timing = runtime_prompt_timing_report(
        setup_time_ns,
        run_time_ns,
        generated_token_count,
        tick_count,
        total_scheduler_turns,
    );
    let pedal_timings = runtime_placed_pedal_timings_report(&run);
    let pedal_timing_summaries = runtime_placed_pedal_timing_summaries_report(&pedal_timings);
    let transport_stats_by_tick = run
        .tick_runs
        .iter()
        .map(|tick| runtime_transport_stats_report(&tick.tick_run.placed_run.transport_stats))
        .collect::<Vec<_>>();
    let transport_published_packet_count = run
        .tick_runs
        .iter()
        .map(|tick| {
            tick.tick_run
                .placed_run
                .transport_stats
                .published_packet_count
        })
        .sum::<usize>();
    let transport_published_byte_count = run
        .tick_runs
        .iter()
        .map(|tick| {
            tick.tick_run
                .placed_run
                .transport_stats
                .published_byte_count
        })
        .sum::<usize>();
    let transport_received_packet_count = run
        .tick_runs
        .iter()
        .map(|tick| {
            tick.tick_run
                .placed_run
                .transport_stats
                .received_packet_count
        })
        .sum::<usize>();
    let transport_received_byte_count = run
        .tick_runs
        .iter()
        .map(|tick| tick.tick_run.placed_run.transport_stats.received_byte_count)
        .sum::<usize>();
    let transport_direct_copy_count = run
        .tick_runs
        .iter()
        .map(|tick| tick.tick_run.placed_run.transport_stats.direct_copy_count)
        .sum::<usize>();
    let transport_direct_copy_byte_count = run
        .tick_runs
        .iter()
        .map(|tick| {
            tick.tick_run
                .placed_run
                .transport_stats
                .direct_copy_byte_count
        })
        .sum::<usize>();
    let transport_direct_receive_count = run
        .tick_runs
        .iter()
        .map(|tick| {
            tick.tick_run
                .placed_run
                .transport_stats
                .direct_receive_count
        })
        .sum::<usize>();
    let transport_direct_receive_byte_count = run
        .tick_runs
        .iter()
        .map(|tick| {
            tick.tick_run
                .placed_run
                .transport_stats
                .direct_receive_byte_count
        })
        .sum::<usize>();

    Ok(RuntimePlacedPromptRunReport {
        ok: true,
        execution_mode: "placed_in_process".to_string(),
        package_manifest: package_manifest.to_path_buf(),
        tokenizer_dir: tokenizer_dir.to_path_buf(),
        boundary_device_id: stream_snapshot.boundary_device_id.clone(),
        device_count: stream_snapshot.device_ids.len(),
        device_ids: stream_snapshot.device_ids.clone(),
        bound_devices: bound_devices_report(&bound_devices),
        cable_routes: bound_cable_routes_report(&bound_devices, &placement.cables),
        runtime_patch: runtime_patch_report(args),
        device_bindings: runtime_device_bindings_report(args, &stream_snapshot.device_ids),
        hosted_pedal_count: stream_snapshot.hosted_pedal_count,
        resident_capacity_activations: stream_snapshot.resident_capacity_activations,
        needed_capacity_activations: prompt_ids.len() + args.max_new_tokens,
        tokenizer: tokenizer_options_report(args),
        prompt_text: prompt.to_string(),
        prompt_ids: run.prompt_token_ids.clone(),
        generated_ids: run.generated_token_ids.clone(),
        generated_text: generated_text.clone(),
        output_text: output_text.clone(),
        stop_reason: run.stop_reason.clone(),
        tick_count,
        scheduler_turns: total_scheduler_turns,
        max_scheduler_turns_per_tick: args.max_scheduler_turns,
        completed_stage_deltas,
        transport: RuntimePlacedTransportReport {
            published_packet_count: transport_published_packet_count,
            published_byte_count: transport_published_byte_count,
            received_packet_count: transport_received_packet_count,
            received_byte_count: transport_received_byte_count,
            direct_copy_count: transport_direct_copy_count,
            direct_copy_byte_count: transport_direct_copy_byte_count,
            direct_receive_count: transport_direct_receive_count,
            direct_receive_byte_count: transport_direct_receive_byte_count,
            by_tick: transport_stats_by_tick,
        },
        timing,
        pedal_timings,
        pedal_timing_summaries,
    })
}

fn print_placed_prompt_report(
    args: &Args,
    report: &RuntimePlacedPromptRunReport,
) -> Result<(), Box<dyn Error>> {
    if args.json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else if args.generated_only {
        print_text(&report.generated_text);
    } else {
        print_text(&report.output_text);
        if args.profile {
            print_prompt_timing_profile(&report.timing);
            print_placed_pedal_timing_profile(&report.pedal_timing_summaries, 5);
        }
    }
    Ok(())
}

fn run_single_device_prompt_benchmark(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    prompt: &str,
    prompt_ids: &[u32],
    needed_capacity: usize,
    capacity: usize,
    manifest: VulkanResidentGreedyModelPackageManifest,
    codec: &VulkanResidentHfTokenizerTextCodec,
) -> Result<(), Box<dyn Error>> {
    let mut runs = Vec::with_capacity(args.profile_runs);
    for _ in 0..args.profile_runs {
        runs.push(execute_single_device_prompt_run(
            args,
            package_manifest,
            manifest_dir,
            tokenizer_dir,
            prompt,
            needed_capacity,
            capacity,
            manifest.clone(),
            codec,
        )?);
    }
    let first = runs
        .first()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--profile-runs is empty"))?;
    let benchmark_runs = runs
        .iter()
        .enumerate()
        .map(|(run_index, run)| single_device_benchmark_run_report(run_index, run))
        .collect::<Vec<_>>();
    let benchmark = runtime_prompt_benchmark_report(
        args,
        package_manifest,
        tokenizer_dir,
        prompt,
        prompt_ids,
        &first.execution_mode,
        vec![first.device_id.clone()],
        first.device_bindings.clone(),
        benchmark_runs,
    );
    print_prompt_benchmark_report(args, &benchmark)
}

fn run_placed_prompt_benchmark(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    prompt: &str,
    prompt_ids: &[u32],
    capacity: usize,
    manifest: VulkanResidentGreedyModelPackageManifest,
    codec: &VulkanResidentHfTokenizerTextCodec,
) -> Result<(), Box<dyn Error>> {
    let mut runs = Vec::with_capacity(args.profile_runs);
    for _ in 0..args.profile_runs {
        runs.push(execute_placed_prompt_run(
            args,
            package_manifest,
            manifest_dir,
            tokenizer_dir,
            prompt,
            prompt_ids,
            capacity,
            manifest.clone(),
            codec,
        )?);
    }
    let first = runs
        .first()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "--profile-runs is empty"))?;
    let benchmark_runs = runs
        .iter()
        .enumerate()
        .map(|(run_index, run)| placed_benchmark_run_report(run_index, run))
        .collect::<Vec<_>>();
    let benchmark = runtime_prompt_benchmark_report(
        args,
        package_manifest,
        tokenizer_dir,
        prompt,
        prompt_ids,
        &first.execution_mode,
        first.device_ids.clone(),
        first.device_bindings.clone(),
        benchmark_runs,
    );
    print_prompt_benchmark_report(args, &benchmark)
}

fn single_device_benchmark_run_report(
    run_index: usize,
    run: &RuntimeSingleDevicePromptRunReport,
) -> RuntimePromptBenchmarkRunReport {
    RuntimePromptBenchmarkRunReport {
        run_index,
        execution_mode: run.execution_mode.clone(),
        stop_reason: run.stop_reason.clone(),
        generated_token_count: run.timing.generated_token_count,
        tick_count: run.timing.tick_count,
        scheduler_turn_count: run.timing.scheduler_turn_count,
        setup_time_ns: run.timing.setup_time_ns,
        run_time_ns: run.timing.run_time_ns,
        total_time_ns: run.timing.total_time_ns,
        generated_tokens_per_second: generated_tokens_per_second(
            run.timing.generated_token_count,
            run.timing.run_time_ns,
        ),
        transport: None,
        pedal_timing_summaries: Vec::new(),
    }
}

fn placed_benchmark_run_report(
    run_index: usize,
    run: &RuntimePlacedPromptRunReport,
) -> RuntimePromptBenchmarkRunReport {
    RuntimePromptBenchmarkRunReport {
        run_index,
        execution_mode: run.execution_mode.clone(),
        stop_reason: run.stop_reason.clone(),
        generated_token_count: run.timing.generated_token_count,
        tick_count: run.timing.tick_count,
        scheduler_turn_count: run.timing.scheduler_turn_count,
        setup_time_ns: run.timing.setup_time_ns,
        run_time_ns: run.timing.run_time_ns,
        total_time_ns: run.timing.total_time_ns,
        generated_tokens_per_second: generated_tokens_per_second(
            run.timing.generated_token_count,
            run.timing.run_time_ns,
        ),
        transport: Some(benchmark_transport_totals_from_report(&run.transport)),
        pedal_timing_summaries: run.pedal_timing_summaries.clone(),
    }
}

fn runtime_prompt_benchmark_report(
    args: &Args,
    package_manifest: &Path,
    tokenizer_dir: &Path,
    prompt: &str,
    prompt_ids: &[u32],
    execution_mode: &str,
    device_ids: Vec<String>,
    device_bindings: RuntimeDeviceBindings,
    runs: Vec<RuntimePromptBenchmarkRunReport>,
) -> RuntimePromptBenchmarkReport {
    let setup_values = runs.iter().map(|run| run.setup_time_ns).collect::<Vec<_>>();
    let run_values = runs.iter().map(|run| run.run_time_ns).collect::<Vec<_>>();
    let total_values = runs.iter().map(|run| run.total_time_ns).collect::<Vec<_>>();
    let generated_token_values = runs
        .iter()
        .map(|run| run.generated_token_count)
        .collect::<Vec<_>>();
    let tick_values = runs.iter().map(|run| run.tick_count).collect::<Vec<_>>();
    let scheduler_turn_values = runs
        .iter()
        .map(|run| run.scheduler_turn_count)
        .collect::<Vec<_>>();
    let mut stop_reasons = BTreeMap::new();
    for run in &runs {
        *stop_reasons.entry(run.stop_reason.clone()).or_insert(0) += 1;
    }
    let total_generated_tokens = generated_token_values.iter().sum::<usize>();
    let total_run_time_ns = run_values.iter().sum::<u64>();

    RuntimePromptBenchmarkReport {
        ok: true,
        execution_mode: execution_mode.to_string(),
        package_manifest: package_manifest.to_path_buf(),
        tokenizer_dir: tokenizer_dir.to_path_buf(),
        runtime_patch: runtime_patch_report(args),
        device_bindings,
        device_count: device_ids.len(),
        device_ids,
        profile_runs: runs.len(),
        prompt_text: prompt.to_string(),
        prompt_ids: prompt_ids.to_vec(),
        max_new_tokens: args.max_new_tokens,
        setup_time_ns: benchmark_u64_metric(&setup_values),
        run_time_ns: benchmark_u64_metric(&run_values),
        total_time_ns: benchmark_u64_metric(&total_values),
        generated_token_count: benchmark_usize_metric(&generated_token_values),
        tick_count: benchmark_usize_metric(&tick_values),
        scheduler_turn_count: benchmark_usize_metric(&scheduler_turn_values),
        generated_tokens_per_second: generated_tokens_per_second(
            total_generated_tokens,
            total_run_time_ns,
        ),
        stop_reasons,
        transport_totals: aggregate_benchmark_transport_totals(&runs),
        pedal_timing_summaries: aggregate_benchmark_pedal_timing_summaries(&runs),
        runs,
    }
}

fn benchmark_u64_metric(values: &[u64]) -> RuntimePromptBenchmarkU64MetricReport {
    let total = values.iter().sum::<u64>();
    RuntimePromptBenchmarkU64MetricReport {
        total,
        min: values.iter().copied().min().unwrap_or(0),
        max: values.iter().copied().max().unwrap_or(0),
        average: if values.is_empty() {
            0.0
        } else {
            total as f64 / values.len() as f64
        },
    }
}

fn benchmark_usize_metric(values: &[usize]) -> RuntimePromptBenchmarkUsizeMetricReport {
    let total = values.iter().sum::<usize>();
    RuntimePromptBenchmarkUsizeMetricReport {
        total,
        min: values.iter().copied().min().unwrap_or(0),
        max: values.iter().copied().max().unwrap_or(0),
        average: if values.is_empty() {
            0.0
        } else {
            total as f64 / values.len() as f64
        },
    }
}

fn generated_tokens_per_second(generated_token_count: usize, run_time_ns: u64) -> Option<f64> {
    if run_time_ns == 0 {
        None
    } else {
        Some(generated_token_count as f64 / (run_time_ns as f64 / 1_000_000_000.0))
    }
}

fn benchmark_transport_totals_from_report(
    transport: &RuntimePlacedTransportReport,
) -> RuntimePromptBenchmarkTransportTotalsReport {
    RuntimePromptBenchmarkTransportTotalsReport {
        published_packet_count: transport.published_packet_count,
        published_byte_count: transport.published_byte_count,
        received_packet_count: transport.received_packet_count,
        received_byte_count: transport.received_byte_count,
        direct_copy_count: transport.direct_copy_count,
        direct_copy_byte_count: transport.direct_copy_byte_count,
        direct_receive_count: transport.direct_receive_count,
        direct_receive_byte_count: transport.direct_receive_byte_count,
    }
}

fn aggregate_benchmark_transport_totals(
    runs: &[RuntimePromptBenchmarkRunReport],
) -> Option<RuntimePromptBenchmarkTransportTotalsReport> {
    let mut total = RuntimePromptBenchmarkTransportTotalsReport {
        published_packet_count: 0,
        published_byte_count: 0,
        received_packet_count: 0,
        received_byte_count: 0,
        direct_copy_count: 0,
        direct_copy_byte_count: 0,
        direct_receive_count: 0,
        direct_receive_byte_count: 0,
    };
    let mut seen = false;
    for transport in runs.iter().filter_map(|run| run.transport.as_ref()) {
        seen = true;
        total.published_packet_count = total
            .published_packet_count
            .saturating_add(transport.published_packet_count);
        total.published_byte_count = total
            .published_byte_count
            .saturating_add(transport.published_byte_count);
        total.received_packet_count = total
            .received_packet_count
            .saturating_add(transport.received_packet_count);
        total.received_byte_count = total
            .received_byte_count
            .saturating_add(transport.received_byte_count);
        total.direct_copy_count = total
            .direct_copy_count
            .saturating_add(transport.direct_copy_count);
        total.direct_copy_byte_count = total
            .direct_copy_byte_count
            .saturating_add(transport.direct_copy_byte_count);
        total.direct_receive_count = total
            .direct_receive_count
            .saturating_add(transport.direct_receive_count);
        total.direct_receive_byte_count = total
            .direct_receive_byte_count
            .saturating_add(transport.direct_receive_byte_count);
    }
    seen.then_some(total)
}

fn aggregate_benchmark_pedal_timing_summaries(
    runs: &[RuntimePromptBenchmarkRunReport],
) -> Vec<RuntimePlacedPedalTimingSummaryReport> {
    let mut summaries = BTreeMap::<(String, String), RuntimePlacedPedalTimingSummaryReport>::new();
    for run in runs {
        for timing in &run.pedal_timing_summaries {
            let entry = summaries
                .entry((timing.device_id.clone(), timing.pedal_id.clone()))
                .or_insert_with(|| RuntimePlacedPedalTimingSummaryReport {
                    device_id: timing.device_id.clone(),
                    pedal_id: timing.pedal_id.clone(),
                    tick_count: 0,
                    dispatch_count: 0,
                    total_run_time_ns: 0,
                    average_tick_time_ns: None,
                    average_dispatch_time_ns: None,
                });
            entry.tick_count += timing.tick_count;
            entry.dispatch_count += timing.dispatch_count;
            entry.total_run_time_ns = entry
                .total_run_time_ns
                .saturating_add(timing.total_run_time_ns);
        }
    }
    let mut summaries = summaries.into_values().collect::<Vec<_>>();
    for summary in &mut summaries {
        summary.average_tick_time_ns = average_nanos(summary.total_run_time_ns, summary.tick_count);
        summary.average_dispatch_time_ns =
            average_nanos(summary.total_run_time_ns, summary.dispatch_count);
    }
    summaries.sort_by(|left, right| {
        right
            .total_run_time_ns
            .cmp(&left.total_run_time_ns)
            .then_with(|| left.device_id.cmp(&right.device_id))
            .then_with(|| left.pedal_id.cmp(&right.pedal_id))
    });
    summaries
}

fn print_prompt_benchmark_report(
    args: &Args,
    report: &RuntimePromptBenchmarkReport,
) -> Result<(), Box<dyn Error>> {
    if args.json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }

    println!("benchmark:");
    println!("  execution_mode={}", report.execution_mode);
    println!("  runs={}", report.profile_runs);
    println!("  devices={}", report.device_ids.join(","));
    println!(
        "  setup_ms avg={:.3} min={:.3} max={:.3}",
        nanos_to_millis_f64(report.setup_time_ns.average),
        nanos_to_millis(report.setup_time_ns.min),
        nanos_to_millis(report.setup_time_ns.max)
    );
    println!(
        "  run_ms avg={:.3} min={:.3} max={:.3}",
        nanos_to_millis_f64(report.run_time_ns.average),
        nanos_to_millis(report.run_time_ns.min),
        nanos_to_millis(report.run_time_ns.max)
    );
    println!(
        "  total_ms avg={:.3} min={:.3} max={:.3}",
        nanos_to_millis_f64(report.total_time_ns.average),
        nanos_to_millis(report.total_time_ns.min),
        nanos_to_millis(report.total_time_ns.max)
    );
    println!(
        "  generated_tokens total={} avg={:.3}",
        report.generated_token_count.total, report.generated_token_count.average
    );
    if let Some(tokens_per_second) = report.generated_tokens_per_second {
        println!("  generated_tokens_per_second={tokens_per_second:.3}");
    }
    if !report.stop_reasons.is_empty() {
        println!("stop_reasons:");
        for (reason, count) in &report.stop_reasons {
            println!("  {reason}={count}");
        }
    }
    if let Some(transport) = &report.transport_totals {
        println!("transport_totals:");
        println!("  published_packets={}", transport.published_packet_count);
        println!("  received_packets={}", transport.received_packet_count);
        println!("  direct_copies={}", transport.direct_copy_count);
    }
    print_placed_pedal_timing_profile(&report.pedal_timing_summaries, 5);
    Ok(())
}

fn inspect_runtime_topology(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentGreedyModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let default_device_id = args
        .default_device_id
        .as_deref()
        .unwrap_or(&manifest.placement.default_device_id);
    let available_devices = inspect_available_devices(
        default_device_id,
        runtime_report_default_vulkan_physical_device_index(args),
    );
    let source_graph = manifest.resolved_source_graph(manifest_dir.to_path_buf())?;
    let patch = manifest.runtime_patch_from_controls(
        args.default_device_id.as_deref(),
        &args.pedal_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?;
    let effective_graph = source_graph.instantiate_runtime_patch(&patch)?;
    let placement = effective_graph.placement_plan(&patch.placement_spec())?;
    let placement_device_ids = placement_device_ids(&placement.pedals);
    let runtime_routes = runtime_cable_routes_report(args, &placement.cables);
    let device_bindings = runtime_device_bindings_report(args, &placement_device_ids);
    let source_pedals = source_pedals_report(&manifest);
    let capacity_profiles = capacity_profiles_report(&manifest);

    let payload = RuntimeTopologyReport {
        ok: true,
        schema: RUNTIME_TOPOLOGY_SCHEMA.to_string(),
        package_manifest: package_manifest.to_path_buf(),
        package_root: manifest_dir.to_path_buf(),
        package_id: manifest.package_id.clone(),
        compiled_schema: manifest.schema.clone(),
        config_path: manifest.config_path.clone(),
        tokenizer: serde_json::to_value(&manifest.tokenizer)?,
        available_devices,
        compiled: RuntimeCompiledPedalboardSummary {
            wiring: manifest.circuit_graph.wiring.clone(),
            default_device_id: manifest.placement.default_device_id.clone(),
            pedal_devices: manifest.placement.pedal_devices.clone(),
            source_pedal_count: source_pedals.len(),
            source_pedals,
            dynamic_state_capacity_activations: manifest.dynamic_state_capacity_activations,
            capacity_profiles,
        },
        runtime_patch_controls: runtime_patch_report(args),
        runtime_patch: patch,
        effective: RuntimeEffectivePedalboardTopology {
            wiring: placement.wiring,
            pedal_count: placement.pedals.len(),
            cable_count: placement.cables.len(),
            local_cable_count: placement.local_cable_count,
            cross_device_cable_count: placement.cross_device_cable_count,
            device_count: placement_device_ids.len(),
            device_ids: placement_device_ids,
            device_bindings,
            cable_routes: runtime_routes,
            pedals: placement.pedals,
            cables: placement.cables,
        },
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("source_pedal_count={}", payload.compiled.source_pedal_count);
        println!("effective_pedal_count={}", payload.effective.pedal_count);
        println!("device_count={}", payload.effective.device_count);
        println!(
            "cross_device_cable_count={}",
            payload.effective.cross_device_cable_count
        );
        println!(
            "same_physical_target_cable_count={}",
            payload
                .effective
                .cable_routes
                .same_physical_target_cable_count
        );
        println!(
            "cross_physical_target_cable_count={}",
            payload
                .effective
                .cable_routes
                .cross_physical_target_cable_count
        );
    }

    Ok(())
}

fn inspect_package(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentGreedyModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let default_device_id = args
        .default_device_id
        .as_deref()
        .unwrap_or(&manifest.placement.default_device_id);
    let available_devices = inspect_available_devices(
        default_device_id,
        runtime_report_default_vulkan_physical_device_index(args),
    );
    let source_pedals = source_pedals_report(&manifest);
    let source_pedal_count = source_pedals.len();
    let capacity_profiles = capacity_profiles_report(&manifest);
    let payload = RuntimePackageInspectionReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        package_root: manifest_dir.to_path_buf(),
        schema: manifest.schema.clone(),
        package_id: manifest.package_id.clone(),
        config_path: manifest.config_path.clone(),
        tokenizer: serde_json::to_value(&manifest.tokenizer)?,
        compiled_wiring: manifest.circuit_graph.wiring.clone(),
        compiled_default_device_id: manifest.placement.default_device_id.clone(),
        compiled_pedal_devices: manifest.placement.pedal_devices.clone(),
        runtime_patch: runtime_patch_report(args),
        device_bindings: runtime_device_bindings_report(args, &[]),
        dynamic_state_capacity_activations: manifest.dynamic_state_capacity_activations,
        capacity_profiles,
        source_pedal_count,
        source_pedals,
        available_devices,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("source_pedal_count={}", payload.source_pedal_count);
        println!("compiled_wiring={}", payload.compiled_wiring);
        println!("default_device_id={}", payload.compiled_default_device_id);
        for pedal in &payload.source_pedals {
            println!(
                "{} {} kernels={} state_ports={}",
                pedal.pedal_id, pedal.operator_type, pedal.kernel_count, pedal.state_port_count
            );
        }
    }

    Ok(())
}

fn source_pedals_report(
    manifest: &VulkanResidentGreedyModelPackageManifest,
) -> Vec<RuntimeSourcePedal> {
    let execution_by_pedal = manifest
        .pedal_executions
        .iter()
        .map(|execution| (execution.pedal_id.as_str(), execution))
        .collect::<BTreeMap<_, _>>();

    manifest
        .circuit_graph
        .pedals
        .iter()
        .enumerate()
        .map(|(pedal_index, pedal)| {
            let execution = execution_by_pedal.get(pedal.pedal_id.as_str());
            RuntimeSourcePedal {
                pedal_index,
                pedal_id: pedal.pedal_id.clone(),
                operator_type: pedal.operator_type.clone(),
                implementation: pedal.implementation.clone(),
                behavioral_role: pedal.behavioral_role.clone(),
                source_layer_index: pedal.circuit.source.source_layer_index,
                circuit_id: pedal.circuit.id.clone(),
                input_ports: pedal
                    .circuit
                    .boundary
                    .inputs
                    .iter()
                    .map(package_port_report)
                    .collect::<Vec<_>>(),
                output_ports: pedal
                    .circuit
                    .boundary
                    .outputs
                    .iter()
                    .map(package_port_report)
                    .collect::<Vec<_>>(),
                state_port_count: pedal.circuit.state_ports.len(),
                parameter_ref_count: pedal.params.refs.len(),
                node_count: pedal.circuit.nodes.len(),
                kernel_count: execution
                    .map(|execution| execution.kernels.len())
                    .unwrap_or(0),
            }
        })
        .collect::<Vec<_>>()
}

fn inspect_available_devices(
    default_device_id: &str,
    selected_vulkan_device_index: Option<usize>,
) -> Vec<RuntimeAvailableDevice> {
    match VulkanComputeDevice::available_compute_devices() {
        Ok(devices) if devices.is_empty() => vec![RuntimeAvailableDevice {
            device_id: default_device_id.to_string(),
            backend: "vulkan_compute".to_string(),
            available: false,
            runtime_device_id: None,
            physical_device_id: None,
            physical_device_index: None,
            device_name: None,
            device_type: None,
            vendor_id: None,
            raw_device_id: None,
            api_version: None,
            driver_version: None,
            compute_queue_family_indices: None,
            memory_heaps: None,
            selected_by_default: None,
            selected_by_runtime: None,
            runtime_binding: None,
            can_host_runtime_pedals_on_physical_device: None,
            notes: vec!["no compute-capable Vulkan physical devices were found".to_string()],
            error: None,
        }],
        Ok(devices) => {
            let mut cpu_device_ordinal = 0usize;
            devices
            .iter()
            .map(|device| {
                let selected_by_runtime = selected_vulkan_device_index
                    .map(|index| index == device.physical_device_index)
                    .unwrap_or(device.selected_by_default);
                let cpu_runtime_device_id = if device.device_type == "cpu" {
                    let runtime_device_id = format!("cpu{cpu_device_ordinal}");
                    cpu_device_ordinal += 1;
                    Some(runtime_device_id)
                } else {
                    None
                };
                let runtime_device_id = selected_by_runtime
                    .then(|| default_device_id.to_string())
                    .or(cpu_runtime_device_id.clone());
                let device_id = runtime_device_id
                    .clone()
                    .unwrap_or_else(|| device.physical_device_id.clone());
                RuntimeAvailableDevice {
                    device_id,
                    backend: "vulkan_compute".to_string(),
                    available: true,
                    runtime_device_id,
                    physical_device_id: Some(device.physical_device_id.clone()),
                    physical_device_index: Some(device.physical_device_index),
                    device_name: Some(device.device_name.clone()),
                    device_type: Some(device.device_type.clone()),
                    vendor_id: Some(device.vendor_id),
                    raw_device_id: Some(device.device_id),
                    api_version: Some(device.api_version),
                    driver_version: Some(device.driver_version),
                    compute_queue_family_indices: Some(device.compute_queue_family_indices.clone()),
                    memory_heaps: Some(
                        device
                            .memory_heaps
                            .iter()
                            .map(|heap| RuntimeAvailableMemoryHeap {
                                heap_index: heap.heap_index,
                                size_bytes: heap.size_bytes,
                                device_local: heap.device_local,
                            })
                            .collect::<Vec<_>>(),
                    ),
                    selected_by_default: Some(device.selected_by_default),
                    selected_by_runtime: Some(selected_by_runtime),
                    runtime_binding: Some(if selected_by_runtime {
                        "default_local_vulkan_target".to_string()
                    } else {
                        "inventory_only".to_string()
                    }),
                    can_host_runtime_pedals_on_physical_device: Some(true),
                    notes: if selected_by_runtime {
                        if selected_vulkan_device_index.is_some() {
                            vec![
                                "selected by --vulkan-device-index as the default target for unbound logical devices"
                                    .to_string(),
                            ]
                        } else {
                            vec![
                                "auto-detected as the default target for unbound logical devices"
                                    .to_string(),
                            ]
                        }
                    } else if let Some(cpu_runtime_device_id) = cpu_runtime_device_id {
                        vec![format!(
                            "auto-detected CPU runtime target {cpu_runtime_device_id}; backed by Vulkan CPU compute device {}",
                            device.physical_device_id
                        )]
                    } else {
                        vec![
                            "auto-detected by Vulkan inventory; can be selected with --bind-device LOGICAL=vulkan:N"
                                .to_string(),
                        ]
                    },
                    error: None,
                }
            })
            .collect()
        }
        Err(error) => vec![RuntimeAvailableDevice {
            device_id: default_device_id.to_string(),
            backend: "vulkan_compute".to_string(),
            available: false,
            runtime_device_id: None,
            physical_device_id: None,
            physical_device_index: None,
            device_name: None,
            device_type: None,
            vendor_id: None,
            raw_device_id: None,
            api_version: None,
            driver_version: None,
            compute_queue_family_indices: None,
            memory_heaps: None,
            selected_by_default: None,
            selected_by_runtime: None,
            runtime_binding: None,
            can_host_runtime_pedals_on_physical_device: None,
            notes: Vec::new(),
            error: Some(error.to_string()),
        }],
    }
}

fn inspect_patch(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentGreedyModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let source_graph = manifest.resolved_source_graph(manifest_dir.to_path_buf())?;
    let patch = manifest.runtime_patch_from_controls(
        args.default_device_id.as_deref(),
        &args.pedal_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?;
    let effective_graph = source_graph.instantiate_runtime_patch(&patch)?;
    let placement = effective_graph.placement_plan(&patch.placement_spec())?;
    let placement_device_ids = placement_device_ids(&placement.pedals);
    let instance_count = patch.instances.len();
    let cable_count = placement.cables.len();
    let payload = RuntimePatchInspectionReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        package_root: manifest_dir.to_path_buf(),
        package_id: manifest.package_id.clone(),
        compiled_source_pedal_count: source_graph.circuits.len(),
        runtime_patch_controls: runtime_patch_report(args),
        runtime_patch: patch,
        device_bindings: runtime_device_bindings_report(args, &placement_device_ids),
        effective_pedal_count: instance_count,
        effective_cable_count: cable_count,
        placement: RuntimePatchPlacementReport {
            schema: placement.schema,
            wiring: placement.wiring,
            local_cable_count: placement.local_cable_count,
            cross_device_cable_count: placement.cross_device_cable_count,
            runtime_routes: runtime_cable_routes_report(args, &placement.cables),
            pedals: placement.pedals,
            cables: placement.cables,
        },
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("package_id={}", payload.package_id);
        println!("effective_pedal_count={}", payload.effective_pedal_count);
        println!("effective_cable_count={}", payload.effective_cable_count);
        println!(
            "cross_device_cable_count={}",
            payload.placement.cross_device_cable_count
        );
        for pedal in &payload.placement.pedals {
            println!(
                "{} circuit={} device={}",
                pedal.pedal_id, pedal.circuit_id, pedal.device_id
            );
        }
    }

    Ok(())
}

fn package_port_report(port: &CircuitPort) -> RuntimePedalPortSummary {
    RuntimePedalPortSummary {
        id: port.id.clone(),
        signal: port.signal.clone(),
        shape: port.shape.clone(),
        source: port.source.clone(),
        pedal_port: port.pedal_port.clone(),
    }
}

fn capacity_profiles_report(
    manifest: &VulkanResidentGreedyModelPackageManifest,
) -> Vec<RuntimeCapacityProfileSummary> {
    manifest
        .capacity_profiles
        .iter()
        .map(|profile| RuntimeCapacityProfileSummary {
            min_dynamic_state_capacity_activations: profile.min_dynamic_state_capacity_activations,
            max_dynamic_state_capacity_activations: profile.max_dynamic_state_capacity_activations,
            shader_override_count: profile.pedal_execution_shader_overrides.len(),
        })
        .collect::<Vec<_>>()
}

fn inspect_device_slice(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentGreedyModelPackageManifest,
    device_id: &str,
) -> Result<(), Box<dyn Error>> {
    let capacity = choose_runtime_capacity(package_manifest, args.capacity, 1)?;
    let logical_device_ids = vec![device_id.to_string()];
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let device = bound_devices.devices.get(device_id).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("logical device {device_id:?} was not mounted"),
        )
    })?;
    let payload = inspect_device_slice_payload(
        device,
        package_manifest,
        manifest_dir,
        manifest,
        device_id,
        capacity,
    )?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("device_id={}", payload.device_id);
        println!("hosted_pedal_count={}", payload.hosted_pedal_count);
        println!("incoming_cable_count={}", payload.incoming_cable_count);
        println!("outgoing_cable_count={}", payload.outgoing_cable_count);
        println!("dispatch_count={}", payload.dispatch_count);
        println!("descriptor_count={}", payload.descriptor_count);
        println!("tick_stage_count={}", payload.tick_plan.stage_count);
    }

    Ok(())
}

fn inspect_placement(
    args: &Args,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentGreedyModelPackageManifest,
) -> Result<(), Box<dyn Error>> {
    let capacity = choose_runtime_capacity(package_manifest, args.capacity, 1)?;
    let device_ids = manifest.placement_device_ids();
    let placement = runtime_manifest_placement(manifest_dir, &manifest)?;
    let bound_devices = runtime_bound_vulkan_devices(args, &device_ids)?;
    let slices = device_ids
        .iter()
        .map(|device_id| {
            let device = bound_devices.devices.get(device_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("logical device {device_id:?} was not mounted"),
                )
            })?;
            inspect_device_slice_payload(
                device,
                package_manifest,
                manifest_dir,
                manifest.clone(),
                device_id,
                capacity,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let payload = RuntimePlacementReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        resident_capacity_activations: capacity,
        runtime_patch: runtime_patch_report(args),
        device_bindings: runtime_device_bindings_report(args, &device_ids),
        bound_devices: bound_devices_report(&bound_devices),
        cable_routes: bound_cable_routes_report(&bound_devices, &placement.cables),
        device_count: device_ids.len(),
        device_ids,
        devices: slices,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("device_count={}", payload.device_count);
        for device in &payload.devices {
            println!(
                "{} pedals={} incoming={} outgoing={} dispatches={}",
                device.device_id,
                device.hosted_pedal_count,
                device.incoming_cable_count,
                device.outgoing_cable_count,
                device.dispatch_count
            );
        }
    }

    Ok(())
}

fn inspect_device_slice_payload(
    device: &VulkanComputeDevice,
    package_manifest: &Path,
    manifest_dir: &Path,
    manifest: VulkanResidentGreedyModelPackageManifest,
    device_id: &str,
    capacity: usize,
) -> Result<RuntimeDeviceSliceReport, Box<dyn Error>> {
    let slice = VulkanResidentGreedyModelPackageDeviceSlice::from_manifest_for_device(
        device,
        manifest_dir,
        manifest,
        device_id,
        Some(capacity),
    )?;
    let mounted = slice.create_mounted_stream_circuit(device)?;
    let reusable_manifest = VulkanReusableKernelArtifactManifest::new(
        slice
            .loaded_manifest()
            .artifacts
            .iter()
            .map(|artifact| artifact.artifact.clone())
            .collect(),
    );
    let mounted_bound = mounted.mounted_placed_bound_dispatch_plan(&reusable_manifest)?;
    let tick_plan = mounted.stream_tick_plan(&reusable_manifest)?;
    let resident_plan = &mounted.placed_plan.placed_resident_plan;
    let loaded_kernel_artifact_count = slice.loaded_manifest().artifacts.len();

    Ok(RuntimeDeviceSliceReport {
        ok: true,
        package_manifest: package_manifest.to_path_buf(),
        device_name: device.device_name().to_string(),
        device_id: slice.device_id,
        resident_capacity_activations: capacity,
        hosted_pedals: resident_plan.hosted_pedal_ids.clone(),
        local_cables: resident_plan
            .local_cables
            .iter()
            .map(|cable| RuntimeLocalCableBufferReport {
                cable_index: cable.cable_index,
                signal: cable.signal.clone(),
                source_pedal_id: cable.source_pedal_id.clone(),
                destination_pedal_id: cable.destination_pedal_id.clone(),
                device_id: cable.source_device_id.clone(),
                byte_capacity: mounted
                    .cable_io
                    .local_cable_buffer(cable.cable_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        incoming_cables: resident_plan
            .incoming_cables
            .iter()
            .map(|cable| RuntimeRemoteCableBufferReport {
                cable_index: cable.cable_index,
                signal: cable.signal.clone(),
                source_device_id: cable.source_device_id.clone(),
                source_pedal_id: cable.source_pedal_id.clone(),
                destination_device_id: cable.destination_device_id.clone(),
                destination_pedal_id: cable.destination_pedal_id.clone(),
                byte_capacity: mounted
                    .cable_io
                    .incoming_buffer(cable.cable_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        outgoing_cables: resident_plan
            .outgoing_cables
            .iter()
            .map(|cable| RuntimeRemoteCableBufferReport {
                cable_index: cable.cable_index,
                signal: cable.signal.clone(),
                source_device_id: cable.source_device_id.clone(),
                source_pedal_id: cable.source_pedal_id.clone(),
                destination_device_id: cable.destination_device_id.clone(),
                destination_pedal_id: cable.destination_pedal_id.clone(),
                byte_capacity: mounted
                    .cable_io
                    .outgoing_buffer(cable.cable_index)
                    .map(|buffer| buffer.byte_capacity),
            })
            .collect::<Vec<_>>(),
        hosted_pedal_count: slice.hosted_pedal_count,
        incoming_cable_count: slice.incoming_cable_count,
        outgoing_cable_count: slice.outgoing_cable_count,
        permanent_parameter_count: slice.permanent_parameter_count,
        permanent_parameter_bytes: slice.permanent_parameter_bytes,
        reusable_kernel_word_count: slice.reusable_kernel_word_count,
        loaded_kernel_artifact_count,
        dispatch_count: mounted_bound.dispatches.len(),
        descriptor_count: mounted_bound.total_descriptor_count,
        model_boundary_descriptor_count: mounted_bound.model_boundary_descriptor_count,
        incoming_cable_descriptor_count: mounted_bound.incoming_cable_descriptor_count,
        outgoing_cable_descriptor_count: mounted_bound.outgoing_cable_descriptor_count,
        tick_plan: RuntimeDeviceTickPlanReport {
            stage_count: tick_plan.stage_count,
            receive_stage_count: tick_plan.receive_stage_count,
            dispatch_stage_count: tick_plan.dispatch_stage_count,
            publish_stage_count: tick_plan.publish_stage_count,
            local_cable_read_count: tick_plan.local_cable_read_count,
            local_cable_write_count: tick_plan.local_cable_write_count,
            incoming_cable_read_count: tick_plan.incoming_cable_read_count,
            outgoing_cable_write_count: tick_plan.outgoing_cable_write_count,
            model_input_read_count: tick_plan.model_input_read_count,
            model_output_write_count: tick_plan.model_output_write_count,
            can_execute: tick_plan.can_execute,
        },
    })
}

fn placement_device_ids(pedals: &[PedalPlacement]) -> Vec<String> {
    let mut device_ids = pedals
        .iter()
        .map(|pedal| pedal.device_id.clone())
        .collect::<Vec<_>>();
    device_ids.sort();
    device_ids.dedup();
    device_ids
}

fn runtime_manifest_placement(
    manifest_dir: &Path,
    manifest: &VulkanResidentGreedyModelPackageManifest,
) -> Result<llmoop_runtime::StreamCircuitPlacementPlan, Box<dyn Error>> {
    let graph = manifest.resolved_source_graph(manifest_dir.to_path_buf())?;
    graph
        .placement_plan(&manifest.placement)
        .map_err(|error| Box::new(error) as Box<dyn Error>)
}

fn tokenizer_dir_from_package(package_manifest: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(package_manifest)?;
    let manifest_dir = package_manifest
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let tokenizer_dir = resolve_package_path(&manifest_dir, &manifest.tokenizer.path);
    if !tokenizer_dir.join("tokenizer.json").is_file() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "compiled package declares tokenizer at {}, but tokenizer.json is missing",
                tokenizer_dir.display()
            ),
        )));
    }
    Ok(tokenizer_dir)
}

fn resolve_package_path(manifest_dir: &Path, raw_path: &str) -> PathBuf {
    let path = Path::new(raw_path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        manifest_dir.join(path)
    }
}

fn runtime_manifest(
    args: &Args,
    package_manifest: &Path,
) -> Result<VulkanResidentGreedyModelPackageManifest, Box<dyn Error>> {
    let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(package_manifest)?;
    Ok(manifest.with_runtime_patch_controls(
        args.default_device_id.as_deref(),
        &args.pedal_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?)
}

fn runtime_vulkan_device(args: &Args) -> Result<VulkanComputeDevice, Box<dyn Error>> {
    if let Some(physical_device_index) = runtime_physical_device_index(args)? {
        Ok(VulkanComputeDevice::new_for_physical_device_index(
            physical_device_index,
        )?)
    } else {
        Ok(VulkanComputeDevice::new()?)
    }
}

struct RuntimeBoundVulkanDevices {
    devices: BTreeMap<String, Arc<VulkanComputeDevice>>,
    physical_device_indices: BTreeMap<String, usize>,
}

fn runtime_bound_vulkan_devices(
    args: &Args,
    logical_device_ids: &[String],
) -> Result<RuntimeBoundVulkanDevices, Box<dyn Error>> {
    let default_physical_device_index = if let Some(index) = args.vulkan_device_index {
        index
    } else {
        runtime_default_vulkan_physical_device_index()?
    };
    let mut logical_device_ids = logical_device_ids.to_vec();
    logical_device_ids.sort();
    logical_device_ids.dedup();
    let mut devices = BTreeMap::new();
    let mut physical_devices: BTreeMap<usize, Arc<VulkanComputeDevice>> = BTreeMap::new();
    let mut physical_device_indices = BTreeMap::new();

    for logical_device_id in &logical_device_ids {
        let physical_device_index = runtime_mount_physical_device_index(
            args,
            logical_device_id,
            default_physical_device_index,
        )?;
        let device = if let Some(device) = physical_devices.get(&physical_device_index) {
            Arc::clone(device)
        } else {
            let device = Arc::new(VulkanComputeDevice::new_for_physical_device_index(
                physical_device_index,
            )?);
            physical_devices.insert(physical_device_index, Arc::clone(&device));
            device
        };
        devices.insert(logical_device_id.clone(), device);
        physical_device_indices.insert(logical_device_id.clone(), physical_device_index);
    }

    Ok(RuntimeBoundVulkanDevices {
        devices,
        physical_device_indices,
    })
}

fn runtime_default_vulkan_physical_device_index() -> Result<usize, Box<dyn Error>> {
    let devices = VulkanComputeDevice::available_compute_devices()?;
    devices
        .iter()
        .find(|device| device.selected_by_default)
        .or_else(|| devices.first())
        .map(|device| device.physical_device_index)
        .ok_or_else(|| {
            Box::new(io::Error::new(
                io::ErrorKind::NotFound,
                "no Vulkan compute-capable physical devices are available",
            )) as Box<dyn Error>
        })
}

fn bound_devices_report(bound_devices: &RuntimeBoundVulkanDevices) -> Vec<RuntimeBoundDevice> {
    bound_devices
        .devices
        .iter()
        .map(|(logical_device_id, device)| {
            let physical_device_index = bound_devices
                .physical_device_indices
                .get(logical_device_id)
                .copied();
            RuntimeBoundDevice {
                device_id: logical_device_id.clone(),
                target: physical_device_index.map(|index| format!("vulkan:{index}")),
                physical_device_index,
                device_name: device.device_name().to_string(),
            }
        })
        .collect::<Vec<_>>()
}

fn runtime_cable_routes_report(args: &Args, cables: &[PedalCablePlacement]) -> RuntimeCableRoutes {
    RuntimeCableRoutes::from_cables(cables, |device_id| {
        runtime_target_for_logical_device(args, device_id)
    })
}

fn bound_cable_routes_report(
    bound_devices: &RuntimeBoundVulkanDevices,
    cables: &[PedalCablePlacement],
) -> RuntimeCableRoutes {
    RuntimeCableRoutes::from_cables(cables, |device_id| {
        let physical_device_index = bound_devices
            .physical_device_indices
            .get(device_id)
            .copied();
        RuntimeCableRouteTarget {
            target: physical_device_index.map(|index| format!("vulkan:{index}")),
            physical_device_index,
            binding_source: "mounted".to_string(),
        }
    })
}

fn runtime_physical_device_index(args: &Args) -> Result<Option<usize>, Box<dyn Error>> {
    let mut selected = args.vulkan_device_index;
    let mut unsupported_bindings = Vec::new();
    if let Some(default_device_id) = args.default_device_id.as_deref() {
        match resolve_runtime_vulkan_physical_device_ref(default_device_id) {
            Ok(Some(index)) => {
                if let Some(existing) = selected {
                    if existing != index {
                        return Err(Box::new(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!(
                                "runtime default device requests Vulkan physical device {index}, but --vulkan-device-index selected {existing}"
                            ),
                        )));
                    }
                } else {
                    selected = Some(index);
                }
            }
            Ok(None) if default_device_id.contains(':') => {
                unsupported_bindings.push(default_device_id.to_string())
            }
            Ok(None) => {}
            Err(error) => {
                return Err(Box::new(io::Error::new(io::ErrorKind::InvalidInput, error)));
            }
        }
    }
    for (logical_device_id, target) in &args.device_bindings {
        match resolve_runtime_vulkan_physical_device_ref(target) {
            Ok(Some(index)) => {
                if let Some(existing) = selected {
                    if existing != index {
                        return Err(Box::new(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!(
                                "logical device bindings request multiple Vulkan physical devices ({existing} and {index}); mounted execution still supports one VulkanComputeDevice per process, so use --inspect-patch to preview or bind all logical devices to the same physical device"
                            ),
                        )));
                    }
                } else {
                    selected = Some(index);
                }
            }
            Ok(None) => unsupported_bindings.push(format!("{logical_device_id}={target}")),
            Err(error) => {
                return Err(Box::new(io::Error::new(io::ErrorKind::InvalidInput, error)));
            }
        }
    }
    if !unsupported_bindings.is_empty() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "logical device bindings are not mountable by the local Vulkan runtime yet: {}",
                unsupported_bindings.join(", ")
            ),
        )));
    }
    Ok(selected)
}

fn runtime_mount_physical_device_index(
    args: &Args,
    logical_device_id: &str,
    default_physical_device_index: usize,
) -> Result<usize, io::Error> {
    if let Some(target) = args.device_bindings.get(logical_device_id) {
        return resolve_runtime_vulkan_physical_device_ref(target)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "logical device {logical_device_id:?} is bound to unsupported target {target:?}; local mounted execution supports vulkan:N or cpuN targets"
                    ),
                )
            });
    }
    match resolve_runtime_vulkan_physical_device_ref(logical_device_id)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
    {
        Some(index) => Ok(index),
        None if logical_device_id.contains(':') => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "logical device id {logical_device_id:?} looks like an unsupported direct runtime target; local mounted execution supports vulkan:N or cpuN targets"
            ),
        )),
        None => Ok(default_physical_device_index),
    }
}

fn runtime_target_for_logical_device(
    args: &Args,
    logical_device_id: &str,
) -> RuntimeCableRouteTarget {
    if let Some(target) = args.device_bindings.get(logical_device_id) {
        let physical_device_index = resolve_runtime_vulkan_physical_device_ref(target)
            .ok()
            .flatten();
        return RuntimeCableRouteTarget {
            target: physical_device_index
                .map(|index| format!("vulkan:{index}"))
                .or_else(|| Some(target.clone())),
            physical_device_index,
            binding_source: "explicit".to_string(),
        };
    }
    match resolve_runtime_vulkan_physical_device_ref(logical_device_id) {
        Ok(Some(index)) => RuntimeCableRouteTarget {
            target: Some(format!("vulkan:{index}")),
            physical_device_index: Some(index),
            binding_source: "device_id".to_string(),
        },
        Ok(None) | Err(_) if logical_device_id.contains(':') => RuntimeCableRouteTarget {
            target: Some(logical_device_id.to_string()),
            physical_device_index: None,
            binding_source: "device_id".to_string(),
        },
        Ok(None) | Err(_) => {
            let default_physical_device_index =
                runtime_report_default_vulkan_physical_device_index(args);
            let target = default_physical_device_index.map(|index| format!("vulkan:{index}"));
            RuntimeCableRouteTarget {
                physical_device_index: default_physical_device_index,
                target,
                binding_source: if args.vulkan_device_index.is_some() {
                    "process_default".to_string()
                } else {
                    "runtime_default".to_string()
                },
            }
        }
    }
}

fn runtime_report_default_vulkan_physical_device_index(args: &Args) -> Option<usize> {
    args.vulkan_device_index
        .or_else(|| {
            args.default_device_id.as_deref().and_then(|device_id| {
                resolve_runtime_vulkan_physical_device_ref(device_id)
                    .ok()
                    .flatten()
            })
        })
        .or_else(|| runtime_default_vulkan_physical_device_index().ok())
}

fn elapsed_nanos_u64(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn average_nanos(total_ns: u64, count: usize) -> Option<u64> {
    if count == 0 {
        None
    } else {
        Some(total_ns / u64::try_from(count).unwrap_or(u64::MAX))
    }
}

fn runtime_prompt_timing_report(
    setup_time_ns: u64,
    run_time_ns: u64,
    generated_token_count: usize,
    tick_count: usize,
    scheduler_turn_count: usize,
) -> RuntimePromptTimingReport {
    RuntimePromptTimingReport {
        setup_time_ns,
        run_time_ns,
        total_time_ns: setup_time_ns.saturating_add(run_time_ns),
        generated_token_count,
        tick_count,
        scheduler_turn_count,
        average_generated_token_time_ns: average_nanos(run_time_ns, generated_token_count),
        average_tick_time_ns: average_nanos(run_time_ns, tick_count),
        average_scheduler_turn_time_ns: average_nanos(run_time_ns, scheduler_turn_count),
    }
}

fn runtime_placed_pedal_timings_report(
    run: &VulkanResidentGreedyInProcessPlacedPromptEventRun,
) -> Vec<RuntimePlacedPedalTimingReport> {
    let mut timings = Vec::new();
    for tick in &run.tick_runs {
        for device_run in &tick.tick_run.placed_run.device_runs {
            let Some(pedalboard_run) = &device_run.pedalboard_run else {
                continue;
            };
            for pedal_run in &pedalboard_run.pedal_runs {
                let run_time_ns = pedal_run.run_time_ns();
                let dispatch_count = pedal_run.dispatch_count();
                let dispatches = pedal_run
                    .dispatch_runs
                    .iter()
                    .map(|dispatch| RuntimePlacedPedalDispatchTimingReport {
                        dispatch_index: dispatch.dispatch_index,
                        kernel_id: dispatch.kernel_id.clone(),
                        node_id: dispatch.node_id.clone(),
                        op: dispatch.op.clone(),
                        reusable_family_id: dispatch.reusable_family_id.clone(),
                        run_time_ns: dispatch.run_time_ns,
                    })
                    .collect::<Vec<_>>();
                timings.push(RuntimePlacedPedalTimingReport {
                    stream_tick: tick.stream_tick,
                    device_id: pedalboard_run.device_id.clone(),
                    pedal_id: pedal_run.pedal_id.clone(),
                    dispatch_count,
                    run_time_ns,
                    average_dispatch_time_ns: average_nanos(run_time_ns, dispatch_count),
                    dispatches,
                });
            }
        }
    }
    timings
}

fn runtime_placed_pedal_timing_summaries_report(
    pedal_timings: &[RuntimePlacedPedalTimingReport],
) -> Vec<RuntimePlacedPedalTimingSummaryReport> {
    let mut summaries = BTreeMap::<(String, String), RuntimePlacedPedalTimingSummaryReport>::new();
    for timing in pedal_timings {
        let entry = summaries
            .entry((timing.device_id.clone(), timing.pedal_id.clone()))
            .or_insert_with(|| RuntimePlacedPedalTimingSummaryReport {
                device_id: timing.device_id.clone(),
                pedal_id: timing.pedal_id.clone(),
                tick_count: 0,
                dispatch_count: 0,
                total_run_time_ns: 0,
                average_tick_time_ns: None,
                average_dispatch_time_ns: None,
            });
        entry.tick_count += 1;
        entry.dispatch_count += timing.dispatch_count;
        entry.total_run_time_ns = entry.total_run_time_ns.saturating_add(timing.run_time_ns);
    }

    let mut summaries = summaries.into_values().collect::<Vec<_>>();
    for summary in &mut summaries {
        summary.average_tick_time_ns = average_nanos(summary.total_run_time_ns, summary.tick_count);
        summary.average_dispatch_time_ns =
            average_nanos(summary.total_run_time_ns, summary.dispatch_count);
    }
    summaries.sort_by(|left, right| {
        right
            .total_run_time_ns
            .cmp(&left.total_run_time_ns)
            .then_with(|| left.device_id.cmp(&right.device_id))
            .then_with(|| left.pedal_id.cmp(&right.pedal_id))
    });
    summaries
}

fn tokenizer_options_report(args: &Args) -> RuntimeTokenizerOptionsReport {
    RuntimeTokenizerOptionsReport {
        add_special_tokens: args.add_special_tokens,
        skip_special_tokens: args.skip_special_tokens,
    }
}

fn runtime_transport_stats_report(
    stats: &VulkanPlacedCableTransportStats,
) -> RuntimePlacedTransportStatsReport {
    RuntimePlacedTransportStatsReport {
        pending_packet_count: stats.pending_packet_count,
        pending_byte_count: stats.pending_byte_count,
        pending_direct_cable_count: stats.pending_direct_cable_count,
        pending_direct_byte_count: stats.pending_direct_byte_count,
        published_packet_count: stats.published_packet_count,
        published_byte_count: stats.published_byte_count,
        received_packet_count: stats.received_packet_count,
        received_byte_count: stats.received_byte_count,
        direct_copy_count: stats.direct_copy_count,
        direct_copy_byte_count: stats.direct_copy_byte_count,
        direct_receive_count: stats.direct_receive_count,
        direct_receive_byte_count: stats.direct_receive_byte_count,
    }
}

fn runtime_patch_report(args: &Args) -> RuntimePatchControls {
    RuntimePatchControls {
        default_device_id: args.default_device_id.clone(),
        pedal_devices: args.pedal_devices.clone(),
        source_chain: args.source_chain.as_ref().map(|source_chain| {
            source_chain
                .iter()
                .map(
                    |(instance_id, source_pedal_id)| RuntimePatchSourceChainEntry {
                        instance_id: instance_id.clone(),
                        source_pedal_id: source_pedal_id.clone(),
                    },
                )
                .collect::<Vec<_>>()
        }),
        duplicate_after: args
            .duplicate_after
            .iter()
            .map(
                |(after_instance_id, new_instance_id)| RuntimePatchDuplicateAfterControl {
                    after_instance_id: after_instance_id.clone(),
                    new_instance_id: new_instance_id.clone(),
                },
            )
            .collect::<Vec<_>>(),
    }
}

fn runtime_device_bindings_report(
    args: &Args,
    logical_device_ids: &[String],
) -> RuntimeDeviceBindings {
    RuntimeDeviceBindings::from_vulkan_targets(
        logical_device_ids,
        &args.device_bindings,
        runtime_report_default_vulkan_physical_device_index(args),
        resolve_runtime_vulkan_physical_device_ref,
    )
}

fn choose_runtime_capacity(
    package_manifest: &Path,
    requested_capacity: Option<usize>,
    needed_capacity: usize,
) -> Result<usize, Box<dyn Error>> {
    let manifest = VulkanResidentGreedyModelPackageManifest::from_json_file(package_manifest)?;
    let default_capacity = manifest.dynamic_state_capacity_activations;
    let max_supported_capacity = manifest
        .capacity_profiles
        .iter()
        .map(|profile| profile.max_dynamic_state_capacity_activations)
        .chain(std::iter::once(default_capacity))
        .max()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "compiled package does not declare any supported dynamic-state capacity",
            )
        })?;

    if let Some(capacity) = requested_capacity {
        if capacity < needed_capacity {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "requested capacity {capacity} is too small: prompt plus generation needs {needed_capacity} activations"
                ),
            )));
        }
        if capacity > max_supported_capacity {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "requested capacity {capacity} exceeds compiled package support ({max_supported_capacity}); recompile with a larger capacity"
                ),
            )));
        }
        let supported = capacity == default_capacity
            || manifest.capacity_profiles.iter().any(|profile| {
                profile.min_dynamic_state_capacity_activations <= capacity
                    && capacity <= profile.max_dynamic_state_capacity_activations
            });
        if !supported {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "requested capacity {capacity} is not supported by this compiled package; recompile with a matching capacity profile"
                ),
            )));
        }
        return Ok(capacity);
    }

    if default_capacity >= needed_capacity {
        return Ok(default_capacity);
    }

    let mut profiles = manifest.capacity_profiles;
    profiles.sort_by_key(|profile| {
        (
            profile.max_dynamic_state_capacity_activations,
            profile.min_dynamic_state_capacity_activations,
        )
    });
    if let Some(profile) = profiles
        .into_iter()
        .find(|profile| needed_capacity <= profile.max_dynamic_state_capacity_activations)
    {
        return Ok(needed_capacity.max(profile.min_dynamic_state_capacity_activations));
    }

    Err(Box::new(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "prompt plus generation needs {needed_capacity} activations, but compiled package supports up to {max_supported_capacity}; recompile with a larger capacity"
        ),
    )))
}

fn parse_args() -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut raw = std::env::args().skip(1);

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--package" | "--package-manifest" => {
                parsed.package_manifest = Some(PathBuf::from(next_value(&mut raw, &arg)?));
            }
            "--prompt" => {
                parsed.prompt = Some(next_value(&mut raw, "--prompt")?);
            }
            "--inspect-runtime" | "--inspect-topology" => {
                parsed.inspect_runtime = true;
            }
            "--inspect-package" | "--inspect-pedals" => {
                parsed.inspect_package = true;
            }
            "--inspect-patch" => {
                parsed.inspect_patch = true;
            }
            "--inspect-placement" => {
                parsed.inspect_placement = true;
            }
            "--inspect-device-slice" => {
                parsed.inspect_device_slice = Some(next_value(&mut raw, "--inspect-device-slice")?);
            }
            "--device" | "--default-device-id" => {
                parsed.default_device_id = Some(next_value(&mut raw, &arg)?);
            }
            "--place-pedal" | "--place" => {
                let assignment = next_value(&mut raw, &arg)?;
                let (pedal_id, device_id) = parse_pedal_device_assignment(&assignment)?;
                if parsed
                    .pedal_devices
                    .insert(pedal_id.clone(), device_id)
                    .is_some()
                {
                    return Err(format!(
                        "duplicate runtime placement for pedal {pedal_id:?}"
                    ));
                }
            }
            "--bind-device" | "--device-binding" => {
                let assignment = next_value(&mut raw, &arg)?;
                let (device_id, target) = parse_device_binding_assignment(&assignment)?;
                if parsed
                    .device_bindings
                    .insert(device_id.clone(), target)
                    .is_some()
                {
                    return Err(format!(
                        "duplicate runtime device binding for logical device {device_id:?}"
                    ));
                }
            }
            "--duplicate-after" => {
                let assignment = next_value(&mut raw, "--duplicate-after")?;
                parsed
                    .duplicate_after
                    .push(parse_duplicate_after_assignment(&assignment)?);
            }
            "--chain" | "--source-chain" => {
                let chain = parse_source_chain(&next_value(&mut raw, &arg)?)?;
                if parsed.source_chain.replace(chain).is_some() {
                    return Err("--chain may only be supplied once".to_string());
                }
            }
            "--max-new-tokens" => {
                parsed.max_new_tokens = parse_next(&mut raw, "--max-new-tokens")?;
            }
            "--capacity" => {
                parsed.capacity = Some(parse_next(&mut raw, "--capacity")?);
            }
            "--vulkan-device-index" => {
                parsed.vulkan_device_index = Some(parse_next(&mut raw, "--vulkan-device-index")?);
            }
            "--cycle-ticks" => {
                parsed.cycle_ticks = parse_next(&mut raw, "--cycle-ticks")?;
            }
            "--max-scheduler-turns" => {
                parsed.max_scheduler_turns = parse_next(&mut raw, "--max-scheduler-turns")?;
            }
            "--no-special-tokens" => {
                parsed.add_special_tokens = false;
            }
            "--keep-special-tokens" => {
                parsed.skip_special_tokens = false;
            }
            "--generated-only" => {
                parsed.generated_only = true;
            }
            "--profile" => {
                parsed.profile = true;
            }
            "--profile-runs" => {
                parsed.profile_runs = parse_next(&mut raw, "--profile-runs")?;
            }
            "--json" => {
                parsed.json = true;
            }
            _ => {
                return Err(format!("unknown argument {arg:?}\n\n{}", usage()));
            }
        }
    }

    if matches!(parsed.prompt.as_deref(), Some("")) {
        return Err("--prompt must not be empty".to_string());
    }
    let inspect_mode_count = usize::from(parsed.inspect_runtime)
        + usize::from(parsed.inspect_package)
        + usize::from(parsed.inspect_patch)
        + usize::from(parsed.inspect_placement)
        + usize::from(parsed.inspect_device_slice.is_some());
    if inspect_mode_count > 1 {
        return Err(
            "--inspect-runtime, --inspect-package, --inspect-patch, --inspect-placement, and --inspect-device-slice are mutually exclusive"
                .to_string(),
        );
    }
    if matches!(parsed.inspect_device_slice.as_deref(), Some("")) {
        return Err("--inspect-device-slice must not be empty".to_string());
    }
    if matches!(parsed.default_device_id.as_deref(), Some("")) {
        return Err("--device must not be empty".to_string());
    }
    if parsed.max_new_tokens == 0 {
        return Err("--max-new-tokens must be at least 1".to_string());
    }
    if matches!(parsed.capacity, Some(0)) {
        return Err("--capacity must be at least 1".to_string());
    }
    if parsed.cycle_ticks == 0 {
        return Err("--cycle-ticks must be at least 1".to_string());
    }
    if parsed.max_scheduler_turns == 0 {
        return Err("--max-scheduler-turns must be at least 1".to_string());
    }
    if parsed.profile_runs == 0 {
        return Err("--profile-runs must be at least 1".to_string());
    }

    Ok(parsed)
}

fn parse_pedal_device_assignment(raw: &str) -> Result<(String, String), String> {
    let (pedal_id, device_id) = raw
        .split_once('=')
        .ok_or_else(|| format!("invalid runtime placement {raw:?}; expected PEDAL_ID=DEVICE_ID"))?;
    let pedal_id = pedal_id.trim();
    let device_id = device_id.trim();
    if pedal_id.is_empty() {
        return Err(format!(
            "invalid runtime placement {raw:?}; pedal id must not be empty"
        ));
    }
    if device_id.is_empty() {
        return Err(format!(
            "invalid runtime placement {raw:?}; device id must not be empty"
        ));
    }
    Ok((pedal_id.to_string(), device_id.to_string()))
}

fn parse_device_binding_assignment(raw: &str) -> Result<(String, String), String> {
    let (device_id, target) = raw.split_once('=').ok_or_else(|| {
        format!("invalid runtime device binding {raw:?}; expected LOGICAL_DEVICE_ID=TARGET")
    })?;
    let device_id = device_id.trim();
    let target = target.trim();
    if device_id.is_empty() {
        return Err(format!(
            "invalid runtime device binding {raw:?}; logical device id must not be empty"
        ));
    }
    if target.is_empty() {
        return Err(format!(
            "invalid runtime device binding {raw:?}; target must not be empty"
        ));
    }
    if let Err(error) = resolve_runtime_vulkan_physical_device_ref(target) {
        return Err(error);
    }
    Ok((device_id.to_string(), target.to_string()))
}

fn resolve_runtime_vulkan_physical_device_ref(raw: &str) -> Result<Option<usize>, String> {
    if let Some(index) = parse_vulkan_physical_device_ref(raw)? {
        return Ok(Some(index));
    }
    if let Some(cpu_ordinal) = parse_cpu_runtime_device_ref(raw)? {
        return runtime_cpu_vulkan_physical_device_index(cpu_ordinal).map(Some);
    }
    Ok(None)
}

fn parse_vulkan_physical_device_ref(raw: &str) -> Result<Option<usize>, String> {
    if let Some(index) = raw.strip_prefix("vulkan:") {
        if index.is_empty() {
            return Err(format!(
                "invalid Vulkan physical device reference {raw:?}; expected vulkan:N"
            ));
        }
        return index
            .parse::<usize>()
            .map(Some)
            .map_err(|error| format!("invalid Vulkan physical device reference {raw:?}: {error}"));
    }
    Ok(None)
}

fn parse_cpu_runtime_device_ref(raw: &str) -> Result<Option<usize>, String> {
    if raw == "cpu" {
        return Ok(Some(0));
    }
    if let Some(index) = raw.strip_prefix("cpu:") {
        if index.is_empty() {
            return Err(format!(
                "invalid CPU runtime device reference {raw:?}; expected cpuN or cpu:N"
            ));
        }
        return index
            .parse::<usize>()
            .map(Some)
            .map_err(|error| format!("invalid CPU runtime device reference {raw:?}: {error}"));
    }
    if let Some(index) = raw.strip_prefix("cpu") {
        if index.is_empty() {
            return Err(format!(
                "invalid CPU runtime device reference {raw:?}; expected cpuN or cpu:N"
            ));
        }
        if index.chars().all(|ch| ch.is_ascii_digit()) {
            return index
                .parse::<usize>()
                .map(Some)
                .map_err(|error| format!("invalid CPU runtime device reference {raw:?}: {error}"));
        }
        return Err(format!(
            "invalid CPU runtime device reference {raw:?}; expected cpuN or cpu:N"
        ));
    }
    Ok(None)
}

fn runtime_cpu_vulkan_physical_device_index(cpu_ordinal: usize) -> Result<usize, String> {
    let devices = VulkanComputeDevice::available_compute_devices()
        .map_err(|error| format!("failed to inspect Vulkan devices for CPU target: {error}"))?;
    devices
        .iter()
        .filter(|device| device.device_type == "cpu")
        .nth(cpu_ordinal)
        .map(|device| device.physical_device_index)
        .ok_or_else(|| {
            format!(
                "CPU runtime target cpu{cpu_ordinal} requested, but no matching Vulkan CPU compute device is available"
            )
        })
}

fn parse_duplicate_after_assignment(raw: &str) -> Result<(String, String), String> {
    let (after_instance_id, new_instance_id) = raw.split_once('=').ok_or_else(|| {
        format!("invalid runtime duplicate {raw:?}; expected AFTER_INSTANCE_ID=NEW_INSTANCE_ID")
    })?;
    let after_instance_id = after_instance_id.trim();
    let new_instance_id = new_instance_id.trim();
    if after_instance_id.is_empty() {
        return Err(format!(
            "invalid runtime duplicate {raw:?}; source instance id must not be empty"
        ));
    }
    if new_instance_id.is_empty() {
        return Err(format!(
            "invalid runtime duplicate {raw:?}; new instance id must not be empty"
        ));
    }
    Ok((after_instance_id.to_string(), new_instance_id.to_string()))
}

fn parse_source_chain(raw: &str) -> Result<Vec<(String, String)>, String> {
    let separator = if raw.contains("->") { "->" } else { "," };
    let mut chain = Vec::new();
    let mut instance_ids = std::collections::BTreeSet::new();

    for raw_item in raw.split(separator) {
        let raw_item = raw_item.trim();
        if raw_item.is_empty() {
            return Err(format!(
                "invalid runtime chain {raw:?}; chain items must not be empty"
            ));
        }
        let (instance_id, source_pedal_id) =
            if let Some((instance_id, source_pedal_id)) = raw_item.split_once('=') {
                (instance_id.trim(), source_pedal_id.trim())
            } else {
                (raw_item, raw_item)
            };
        if instance_id.is_empty() {
            return Err(format!(
                "invalid runtime chain item {raw_item:?}; instance id must not be empty"
            ));
        }
        if source_pedal_id.is_empty() {
            return Err(format!(
                "invalid runtime chain item {raw_item:?}; source pedal id must not be empty"
            ));
        }
        if !instance_ids.insert(instance_id.to_string()) {
            return Err(format!(
                "invalid runtime chain {raw:?}; duplicate instance id {instance_id:?}"
            ));
        }
        chain.push((instance_id.to_string(), source_pedal_id.to_string()));
    }

    if chain.is_empty() {
        return Err("runtime chain must contain at least one pedal".to_string());
    }

    Ok(chain)
}

fn parse_next<T: std::str::FromStr>(
    raw: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    let value = next_value(raw, flag)?;
    value
        .parse::<T>()
        .map_err(|error| format!("invalid value {value:?} for {flag}: {error}"))
}

fn next_value(raw: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    raw.next()
        .ok_or_else(|| format!("{flag} requires a value\n\n{}", usage()))
}

fn print_text(text: &str) {
    print!("{text}");
    if !text.ends_with('\n') {
        println!();
    }
}

fn print_prompt_timing_profile(timing: &RuntimePromptTimingReport) {
    println!("profile:");
    println!("  setup_ms={:.3}", nanos_to_millis(timing.setup_time_ns));
    println!("  run_ms={:.3}", nanos_to_millis(timing.run_time_ns));
    println!("  total_ms={:.3}", nanos_to_millis(timing.total_time_ns));
    println!("  generated_tokens={}", timing.generated_token_count);
    println!("  ticks={}", timing.tick_count);
    println!("  scheduler_turns={}", timing.scheduler_turn_count);
    if let Some(average) = timing.average_generated_token_time_ns {
        println!("  avg_generated_token_ms={:.3}", nanos_to_millis(average));
    }
    if let Some(average) = timing.average_tick_time_ns {
        println!("  avg_tick_ms={:.3}", nanos_to_millis(average));
    }
    if let Some(average) = timing.average_scheduler_turn_time_ns {
        println!("  avg_scheduler_turn_ms={:.3}", nanos_to_millis(average));
    }
}

fn print_placed_pedal_timing_profile(
    summaries: &[RuntimePlacedPedalTimingSummaryReport],
    max_rows: usize,
) {
    if summaries.is_empty() || max_rows == 0 {
        return;
    }
    println!("top_pedals:");
    for summary in summaries.iter().take(max_rows) {
        println!(
            "  {}:{} total_ms={:.3} ticks={} dispatches={} avg_tick_ms={} avg_dispatch_ms={}",
            summary.device_id,
            summary.pedal_id,
            nanos_to_millis(summary.total_run_time_ns),
            summary.tick_count,
            summary.dispatch_count,
            optional_nanos_to_millis(summary.average_tick_time_ns),
            optional_nanos_to_millis(summary.average_dispatch_time_ns)
        );
    }
}

fn optional_nanos_to_millis(value: Option<u64>) -> String {
    value
        .map(|nanos| format!("{:.3}", nanos_to_millis(nanos)))
        .unwrap_or_else(|| "n/a".to_string())
}

fn nanos_to_millis(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000.0
}

fn nanos_to_millis_f64(nanos: f64) -> f64 {
    nanos / 1_000_000.0
}

fn engine_stop_label(stop: VulkanResidentTokenEngineRunStopCondition) -> &'static str {
    match stop {
        VulkanResidentTokenEngineRunStopCondition::Idle => "idle",
        VulkanResidentTokenEngineRunStopCondition::SchedulerTurnBudget => "scheduler_turn_budget",
    }
}

fn print_usage() {
    println!("{}", usage());
}

fn usage() -> &'static str {
    "Usage: llmoop-runtime --package <COMPILED_PACKAGE.json> --prompt <TEXT> [OPTIONS]

Options:
  --package <PATH>           Compiled resident model package manifest. Required.
  --package-manifest <PATH>  Alias for --package.
  --prompt <TEXT>            External text event to inject into the resident stream. Required.
  --device <DEVICE_ID>       Default logical device for this runtime patch.
  --default-device-id <ID>   Alias for --device.
  --place-pedal <PEDAL=DEV>  Assign one runtime pedal instance to a logical device.
  --place <PEDAL=DEV>        Alias for --place-pedal.
  --bind-device <DEV=TARGET> Bind a logical device to a target, e.g. gpu1=vulkan:5.
  --device-binding <DEV=TARGET>
                             Alias for --bind-device.
  --chain <ITEM[,ITEM...]>    Runtime source chain. ITEM is SOURCE or INSTANCE=SOURCE.
  --duplicate-after <AFTER=NEW>
                             Duplicate runtime pedal instance AFTER with id NEW.
  --inspect-runtime          Preview UI-ready package, patch, placement, device, and route facts.
  --inspect-topology         Alias for --inspect-runtime.
  --inspect-package          Summarize the compiled source pedal kit and available devices.
  --inspect-pedals           Alias for --inspect-package.
  --inspect-patch            Preview the effective runtime patch without mounting devices.
  --inspect-placement        Mount and summarize every logical device slice in the runtime patch.
  --inspect-device-slice <DEVICE_ID>
                             Mount and summarize only the runtime patch pedals assigned to DEVICE_ID.
  --max-new-tokens <N>       Public output tokens to emit after the prompt. Default: 4
  --capacity <N>             Override resident activation capacity selected from the package.
  --vulkan-device-index <N>  Use Vulkan physical device index N as the default local target.
  --cycle-ticks <N>          Max runtime ticks per always-on cycle. Default: 4
  --max-scheduler-turns <N>  Max engine scheduler turns before stopping. Default: 1024
  --no-special-tokens        Do not add tokenizer special tokens to input text.
  --keep-special-tokens      Keep tokenizer special tokens in decoded output text.
  --generated-only           Print only newly generated text instead of prompt + generated text.
  --profile                  Print human-readable timing and top-pedal summaries.
  --profile-runs <N>         Run N fresh prompt trials and report aggregate benchmark stats.
  --json                     Print a machine-readable run report.
  -h, --help                 Show this help.

Example:
  python -m llmoop --compile-model <MODEL_DIR>
  cargo run --manifest-path runtime-rs/Cargo.toml --features 'vulkan tokenizers' --bin llmoop-runtime -- --package packages/model_xxx/vulkan_resident_greedy_package.json --prompt Hello --max-new-tokens 4"
}
