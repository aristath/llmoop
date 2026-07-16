use std::error::Error;
use std::io;
use std::path::PathBuf;

use llmoop_runtime::{
    VulkanComputeDevice, VulkanLfm2ResidentGreedyStreamProcessorModel,
    VulkanResidentHfTokenizerTextCodec, VulkanResidentTokenEngine,
    VulkanResidentTokenEngineLiveTextTurnRun, VulkanResidentTokenEngineRunBudget,
    VulkanResidentTokenEngineRunStopCondition, VulkanResidentTokenRuntimeCycleRun,
    VulkanResidentTokenRuntimeCycleStopCondition, VulkanResidentTokenTextCodec,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Args {
    model_dir: PathBuf,
    capacity: usize,
    prompt: String,
    max_new_tokens: usize,
    then_prompt: Option<String>,
    then_max_new_tokens: usize,
    cycle_ticks: usize,
    max_scheduler_turns: usize,
    add_special_tokens: bool,
    skip_special_tokens: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("/home/aristath/models/lfm2.5/230m"),
            capacity: 8,
            prompt: "Hello".to_string(),
            max_new_tokens: 3,
            then_prompt: None,
            then_max_new_tokens: 1,
            cycle_ticks: 2,
            max_scheduler_turns: 1_024,
            add_special_tokens: true,
            skip_special_tokens: true,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("resident-text-demo error: {error}");
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
    let codec = VulkanResidentHfTokenizerTextCodec::from_model_dir(&args.model_dir)?
        .with_add_special_tokens(args.add_special_tokens)
        .with_skip_special_tokens(args.skip_special_tokens);
    let stream_id = "demo_text_stream";
    let device = VulkanComputeDevice::new()?;
    let model =
        VulkanLfm2ResidentGreedyStreamProcessorModel::default_for_capacity(&device, args.capacity)?;
    let mut engine = VulkanResidentTokenEngine::new(device);
    engine.add_model_package("demo_model", model)?;
    engine.create_stream_from_model("demo_model", stream_id)?;
    let stream = engine
        .stream(stream_id)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "demo stream was not registered"))?;
    let engine_snapshot = engine.snapshot();

    println!("resident-text-demo");
    println!("model_dir={}", args.model_dir.display());
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
    println!(
        "tokenizer.add_special_tokens={}",
        codec.add_special_tokens()
    );
    println!(
        "tokenizer.skip_special_tokens={}",
        codec.skip_special_tokens()
    );
    println!("cycle_ticks={}", args.cycle_ticks);
    println!("scheduler.max_runtime_cycles_per_turn=1");
    println!("engine.max_scheduler_turns={}", args.max_scheduler_turns);

    let mut submitted_turns = Vec::new();
    submitted_turns.push(run_live_text_turn(
        &mut engine,
        stream_id,
        "first",
        args.prompt.clone(),
        args.max_new_tokens,
        &codec,
        &args,
    )?);
    if let Some(then_prompt) = args.then_prompt.clone() {
        submitted_turns.push(run_live_text_turn(
            &mut engine,
            stream_id,
            "second",
            then_prompt,
            args.then_max_new_tokens,
            &codec,
            &args,
        )?);
    }

    let mut cycle_index = 0usize;
    for (turn_index, turn) in submitted_turns.iter().enumerate() {
        print_live_text_turn(turn_index, turn)?;
        print_live_turn_cycles(turn, &mut cycle_index);
    }

    let generated = submitted_turns
        .iter()
        .flat_map(|turn| turn.generated_token_ids.iter().copied())
        .collect::<Vec<_>>();
    let generated_text = codec.decode_tokens(&generated)?;
    let engine_snapshot = engine.snapshot();
    let snapshot = engine
        .runtime_snapshot(stream_id)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "demo stream disappeared"))?;

    println!("runtime.generated={generated:?}");
    println!("runtime.generated_text={generated_text:?}");
    println!("runtime.cycles={cycle_index}");
    println!(
        "engine.scheduler_turns={}",
        submitted_turns
            .iter()
            .map(VulkanResidentTokenEngineLiveTextTurnRun::scheduler_turn_count)
            .sum::<usize>()
    );
    println!(
        "engine.runtime_cycles={}",
        submitted_turns
            .iter()
            .map(|turn| turn.runtime_cycle_count)
            .sum::<usize>()
    );
    println!(
        "engine.stop={}",
        engine_stop_label(aggregate_engine_stop(&submitted_turns))
    );
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
            "--model-dir" => {
                parsed.model_dir = PathBuf::from(next_value(&mut raw, "--model-dir")?);
            }
            "--capacity" => {
                parsed.capacity = parse_next(&mut raw, "--capacity")?;
            }
            "--prompt" => {
                parsed.prompt = next_value(&mut raw, "--prompt")?;
            }
            "--max-new-tokens" => {
                parsed.max_new_tokens = parse_next(&mut raw, "--max-new-tokens")?;
            }
            "--then-prompt" => {
                parsed.then_prompt = Some(next_value(&mut raw, "--then-prompt")?);
            }
            "--then-max-new-tokens" => {
                parsed.then_max_new_tokens = parse_next(&mut raw, "--then-max-new-tokens")?;
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
            _ => {
                return Err(format!("unknown argument {arg:?}\n\n{}", usage()));
            }
        }
    }

    if parsed.prompt.is_empty() {
        return Err("--prompt must not be empty".to_string());
    }
    if matches!(parsed.then_prompt.as_deref(), Some("")) {
        return Err("--then-prompt must not be empty".to_string());
    }
    if parsed.cycle_ticks == 0 {
        return Err("--cycle-ticks must be at least 1".to_string());
    }
    if parsed.max_scheduler_turns == 0 {
        return Err("--max-scheduler-turns must be at least 1".to_string());
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

fn run_live_text_turn(
    engine: &mut VulkanResidentTokenEngine,
    stream_id: &str,
    input_event_id: &str,
    input_text: String,
    max_new_tokens: usize,
    codec: &impl VulkanResidentTokenTextCodec,
    args: &Args,
) -> Result<VulkanResidentTokenEngineLiveTextTurnRun, Box<dyn Error>> {
    Ok(engine.submit_live_text_turn_until_idle(
        stream_id,
        input_event_id,
        input_text,
        max_new_tokens,
        "cli",
        VulkanResidentTokenEngineRunBudget::new(args.max_scheduler_turns, 1, args.cycle_ticks),
        codec,
    )?)
}

fn print_live_text_turn(
    index: usize,
    turn: &VulkanResidentTokenEngineLiveTextTurnRun,
) -> Result<(), Box<dyn Error>> {
    println!(
        "turn_{index}.input_text={:?}",
        turn.queued_input_event.input_text
    );
    println!(
        "turn_{index}.encoded={:?}",
        turn.queued_input_event.encoded_token_ids
    );
    println!("turn_{index}.generated={:?}", turn.generated_token_ids);
    println!("turn_{index}.generated_text={:?}", turn.generated_text);
    println!("turn_{index}.output_text={:?}", turn.output_text);
    println!(
        "turn_{index}.engine_stop={}",
        engine_stop_label(turn.stop_condition)
    );
    Ok(())
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

fn print_live_turn_cycles(
    turn: &VulkanResidentTokenEngineLiveTextTurnRun,
    cycle_index: &mut usize,
) {
    for text_cycle in &turn.cycles {
        for cycle in &text_cycle.scheduler_run.runtime_cycles {
            let index = *cycle_index;
            print_cycle(index, cycle);
            println!(
                "cycle_{index}.text_outputs={:?} cycle_{index}.text={:?}",
                text_cycle
                    .output_events
                    .iter()
                    .map(|event| (
                        event.input_event_id.as_str(),
                        event.output_index,
                        event.token_id,
                        event.text.as_str()
                    ))
                    .collect::<Vec<_>>(),
                text_cycle.generated_text
            );
            *cycle_index += 1;
        }
    }
}

fn cycle_stop_label(stop: VulkanResidentTokenRuntimeCycleStopCondition) -> &'static str {
    match stop {
        VulkanResidentTokenRuntimeCycleStopCondition::Idle => "idle",
        VulkanResidentTokenRuntimeCycleStopCondition::TickBudget => "tick_budget",
    }
}

fn engine_stop_label(stop: VulkanResidentTokenEngineRunStopCondition) -> &'static str {
    match stop {
        VulkanResidentTokenEngineRunStopCondition::Idle => "idle",
        VulkanResidentTokenEngineRunStopCondition::SchedulerTurnBudget => "scheduler_turn_budget",
    }
}

fn aggregate_engine_stop(
    submitted_turns: &[VulkanResidentTokenEngineLiveTextTurnRun],
) -> VulkanResidentTokenEngineRunStopCondition {
    if submitted_turns.iter().any(|turn| {
        turn.stop_condition == VulkanResidentTokenEngineRunStopCondition::SchedulerTurnBudget
    }) {
        VulkanResidentTokenEngineRunStopCondition::SchedulerTurnBudget
    } else {
        VulkanResidentTokenEngineRunStopCondition::Idle
    }
}

fn print_usage() {
    println!("{}", usage());
}

fn usage() -> &'static str {
    "Usage: resident-text-demo [OPTIONS]

Options:
  --model-dir <PATH>          Directory containing tokenizer.json. Default: /home/aristath/models/lfm2.5/230m
  --capacity <N>              Resident activation capacity. Default: 8
  --prompt <TEXT>             First external text event. Default: Hello
  --max-new-tokens <N>        Public outputs to emit after the first prompt. Default: 3
  --then-prompt <TEXT>        Optional later external text event.
  --then-max-new-tokens <N>   Public outputs to emit after later input. Default: 1
  --cycle-ticks <N>           Max runtime ticks per always-on cycle. Default: 2
  --max-scheduler-turns <N>   Max engine scheduler turns before stopping. Default: 1024
  --no-special-tokens         Do not add tokenizer special tokens to input text.
  --keep-special-tokens       Keep tokenizer special tokens in decoded output text.
  -h, --help                  Show this help

Example:
  cargo run --manifest-path runtime-rs/Cargo.toml --features 'vulkan tokenizers' --bin resident-text-demo -- --prompt Hello --max-new-tokens 3 --then-prompt ' again' --then-max-new-tokens 1"
}
