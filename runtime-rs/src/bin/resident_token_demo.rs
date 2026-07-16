use std::error::Error;
use std::io;

use llmoop_runtime::{
    VulkanResidentTokenEngine, VulkanResidentTokenInputEvent, VulkanResidentTokenRuntimeCycleRun,
    VulkanResidentTokenRuntimeCycleStopCondition,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Args {
    capacity: usize,
    prompt: Vec<u32>,
    max_new_tokens: usize,
    then_prompt: Vec<u32>,
    then_max_new_tokens: usize,
    cycle_ticks: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            capacity: 8,
            prompt: vec![1],
            max_new_tokens: 3,
            then_prompt: vec![36_309],
            then_max_new_tokens: 1,
            cycle_ticks: 2,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("resident-token-demo error: {error}");
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
    let stream_id = "demo_stream";
    let mut engine = VulkanResidentTokenEngine::default_lfm2_5_230m(stream_id, args.capacity)?;
    let stream = engine
        .stream(stream_id)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "demo stream was not registered"))?;
    let engine_snapshot = engine.snapshot();

    println!("resident-token-demo");
    println!("device_name={}", engine_snapshot.device_name);
    println!("device_id={}", stream.device_id);
    println!(
        "pedals={} dispatches_per_tick={} descriptors_per_tick={} push_constant_bytes_per_tick={}",
        stream.pedal_count,
        stream.per_tick_dispatch_count,
        stream.per_tick_descriptor_count,
        stream.per_tick_push_constant_byte_count
    );
    println!(
        "resident_capacity_activations={}",
        stream.dynamic_state_capacity_activations
    );

    engine.enqueue_input_event(
        stream_id,
        VulkanResidentTokenInputEvent::new("first", args.prompt, args.max_new_tokens)
            .with_origin("cli"),
    )?;
    engine.enqueue_input_event(
        stream_id,
        VulkanResidentTokenInputEvent::new("second", args.then_prompt, args.then_max_new_tokens)
            .with_origin("cli"),
    )?;

    println!("cycle_ticks={}", args.cycle_ticks);
    println!("scheduler.max_runtime_cycles_per_turn=1");

    let mut generated = Vec::new();
    let mut cycle_index = 0usize;
    while engine.snapshot().scheduler.running {
        let run = engine.run_cycle(1, args.cycle_ticks)?;
        if run.runtime_cycles.is_empty() && engine.snapshot().scheduler.running {
            return Err(io::Error::other(
                "resident token engine is running but produced no runtime cycles",
            )
            .into());
        }
        generated.extend(
            run.output_events
                .iter()
                .map(|event| event.output_event.token_id),
        );
        for cycle in &run.runtime_cycles {
            print_cycle(cycle_index, cycle);
            cycle_index += 1;
        }
    }

    let engine_snapshot = engine.snapshot();
    let snapshot = engine
        .runtime_snapshot(stream_id)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "demo stream disappeared"))?;
    println!("runtime.generated={generated:?}");
    println!("runtime.cycles={cycle_index}");
    println!(
        "runtime.next_stream_tick={}",
        snapshot.stream.next_stream_tick
    );
    println!(
        "runtime.public_outputs={}",
        snapshot.stream.total_public_outputs
    );
    println!(
        "runtime.pending_inputs={}",
        snapshot.pending_input_event_count
    );
    println!("runtime.idle={}", snapshot.idle);
    println!(
        "scheduler.registered_runtimes={}",
        engine_snapshot.scheduler.registered_runtime_count
    );
    println!(
        "scheduler.active_runtimes={}",
        engine_snapshot.scheduler.active_runtime_count
    );
    println!("scheduler.idle={}", engine_snapshot.scheduler.idle);

    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut raw = std::env::args().skip(1);

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--capacity" => {
                parsed.capacity = parse_next(&mut raw, "--capacity")?;
            }
            "--prompt" => {
                parsed.prompt = parse_token_list(&next_value(&mut raw, "--prompt")?)?;
            }
            "--max-new-tokens" => {
                parsed.max_new_tokens = parse_next(&mut raw, "--max-new-tokens")?;
            }
            "--then-prompt" => {
                parsed.then_prompt = parse_token_list(&next_value(&mut raw, "--then-prompt")?)?;
            }
            "--then-max-new-tokens" => {
                parsed.then_max_new_tokens = parse_next(&mut raw, "--then-max-new-tokens")?;
            }
            "--cycle-ticks" => {
                parsed.cycle_ticks = parse_next(&mut raw, "--cycle-ticks")?;
            }
            _ => {
                return Err(format!("unknown argument {arg:?}\n\n{}", usage()));
            }
        }
    }

    if parsed.prompt.is_empty() {
        return Err("--prompt must contain at least one token id".to_string());
    }
    if parsed.then_prompt.is_empty() {
        return Err("--then-prompt must contain at least one token id".to_string());
    }
    if parsed.cycle_ticks == 0 {
        return Err("--cycle-ticks must be at least 1".to_string());
    }

    Ok(parsed)
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

fn parse_token_list(value: &str) -> Result<Vec<u32>, String> {
    value
        .split(',')
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u32>()
                .map_err(|error| format!("invalid token id {part:?}: {error}"))
        })
        .collect()
}

fn print_cycle(index: usize, cycle: &VulkanResidentTokenRuntimeCycleRun) {
    println!(
        "cycle_{index}.start_stream_tick={} cycle_{index}.next_stream_tick={} cycle_{index}.ticks_used={} cycle_{index}.stop={}",
        cycle.start_stream_tick,
        cycle.next_stream_tick,
        cycle.ticks_used,
        cycle_stop_label(cycle.stop_condition)
    );
    println!(
        "cycle_{index}.queued_inputs={:?} cycle_{index}.pending_inputs={} cycle_{index}.stream_idle={}",
        cycle
            .queued_input_events
            .iter()
            .map(|event| event.input_event.id.as_str())
            .collect::<Vec<_>>(),
        cycle.pending_input_event_count,
        cycle.stream_idle
    );
    println!(
        "cycle_{index}.outputs={:?} cycle_{index}.processed_ticks={} cycle_{index}.idle_ticks={}",
        cycle
            .output_events
            .iter()
            .map(|event| (
                event.input_event_id.as_str(),
                event.output_index,
                event.token_id
            ))
            .collect::<Vec<_>>(),
        cycle.processed_tick_count,
        cycle.idle_tick_count
    );
}

fn cycle_stop_label(stop: VulkanResidentTokenRuntimeCycleStopCondition) -> &'static str {
    match stop {
        VulkanResidentTokenRuntimeCycleStopCondition::Idle => "idle",
        VulkanResidentTokenRuntimeCycleStopCondition::TickBudget => "tick_budget",
    }
}

fn print_usage() {
    println!("{}", usage());
}

fn usage() -> &'static str {
    "Usage: resident-token-demo [OPTIONS]

Options:
  --capacity <N>              Resident activation capacity. Default: 8
  --prompt <TOKENS>           Comma-separated first external token event. Default: 1
  --max-new-tokens <N>        Public outputs to emit after the first prompt. Default: 3
  --then-prompt <TOKENS>      Comma-separated later external token event. Default: 36309
  --then-max-new-tokens <N>   Public outputs to emit after later input. Default: 1
  --cycle-ticks <N>           Max runtime ticks per always-on cycle. Default: 2
  -h, --help                  Show this help

Example:
  cargo run --manifest-path runtime-rs/Cargo.toml --features vulkan --bin resident-token-demo -- --capacity 8 --prompt 1 --max-new-tokens 3 --then-prompt 36309 --then-max-new-tokens 1 --cycle-ticks 2"
}
