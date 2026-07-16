use std::error::Error;
use std::io;

use llmoop_runtime::{
    VulkanComputeDevice, VulkanResidentGreedyRunningStreamRun,
    create_default_lfm2_5_230m_resident_greedy_stream_processor,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Args {
    capacity: usize,
    prompt: Vec<u32>,
    max_new_tokens: usize,
    then_prompt: Vec<u32>,
    then_max_new_tokens: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            capacity: 8,
            prompt: vec![1],
            max_new_tokens: 3,
            then_prompt: vec![36_309],
            then_max_new_tokens: 1,
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
    let device = VulkanComputeDevice::new()?;
    let processor =
        create_default_lfm2_5_230m_resident_greedy_stream_processor(&device, args.capacity)?;

    println!("resident-token-demo");
    println!("device_id={}", processor.device_id);
    println!(
        "pedals={} dispatches_per_tick={} descriptors_per_tick={} push_constant_bytes_per_tick={}",
        processor.pedal_count,
        processor.per_tick_dispatch_count,
        processor.per_tick_descriptor_count,
        processor.per_tick_push_constant_byte_count
    );
    println!(
        "resident_capacity_activations={}",
        processor.dynamic_state_capacity_activations
    );

    let mut stream = processor.into_running_stream("demo_stream");
    let first = stream.run_prompt(&device, &args.prompt, args.max_new_tokens, None)?;
    print_run("first", &first);

    let second = stream.run_prompt(&device, &args.then_prompt, args.then_max_new_tokens, None)?;
    print_run("second", &second);

    println!("stream.next_stream_tick={}", stream.next_stream_tick);
    println!("stream.public_outputs={}", stream.public_outputs().len());
    println!(
        "stream.private_feedback_history={}",
        stream.private_feedback_history().len()
    );

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

fn print_run(label: &str, run: &VulkanResidentGreedyRunningStreamRun) {
    let processed_ticks = run
        .ticks
        .iter()
        .filter(|tick| tick.stream_tick.is_some())
        .count();
    let idle_ticks = run.ticks.len() - processed_ticks;

    println!(
        "{label}.prompt={:?} {label}.generated={:?} {label}.output={:?}",
        run.prompt_token_ids, run.generated_token_ids, run.output_token_ids
    );
    println!(
        "{label}.start_stream_tick={} {label}.next_stream_tick={} {label}.stop_reason={}",
        run.start_stream_tick, run.next_stream_tick, run.stop_reason
    );
    println!("{label}.processed_ticks={processed_ticks} {label}.idle_ticks={idle_ticks}");
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
  -h, --help                  Show this help

Example:
  cargo run --manifest-path runtime-rs/Cargo.toml --features vulkan --bin resident-token-demo -- --capacity 8 --prompt 1 --max-new-tokens 3 --then-prompt 36309 --then-max-new-tokens 1"
}
