use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};

use llmoop_runtime::{
    VulkanComputeDevice, VulkanResidentGreedyModelPackage,
    VulkanResidentGreedyModelPackageManifest, VulkanResidentHfTokenizerTextCodec,
    VulkanResidentTokenEngine, VulkanResidentTokenEngineRunBudget,
    VulkanResidentTokenEngineRunStopCondition, VulkanResidentTokenTextCodec,
};
use serde_json::json;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Args {
    package_manifest: Option<PathBuf>,
    prompt: Option<String>,
    max_new_tokens: usize,
    capacity: Option<usize>,
    cycle_ticks: usize,
    max_scheduler_turns: usize,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    generated_only: bool,
    json: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            package_manifest: None,
            prompt: None,
            max_new_tokens: 4,
            capacity: None,
            cycle_ticks: 4,
            max_scheduler_turns: 1_024,
            add_special_tokens: true,
            skip_special_tokens: true,
            generated_only: false,
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

    let device = VulkanComputeDevice::new()?;
    let model = VulkanResidentGreedyModelPackage::from_manifest_file_with_capacity(
        &device,
        package_manifest,
        Some(capacity),
    )?;
    let mut engine = VulkanResidentTokenEngine::new(device);
    engine.add_model_package("compiled_model", model)?;
    engine.create_stream_from_model("compiled_model", "main")?;

    let turn = engine.submit_live_text_turn_until_idle(
        "main",
        "prompt",
        prompt.clone(),
        args.max_new_tokens,
        "cli",
        VulkanResidentTokenEngineRunBudget::new(args.max_scheduler_turns, 1, args.cycle_ticks),
        &codec,
    )?;
    let stream = engine
        .stream("main")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "runtime stream disappeared"))?;
    let snapshot = engine.snapshot();

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "ok": true,
                "package_manifest": package_manifest,
                "tokenizer_dir": tokenizer_dir,
                "device_name": snapshot.device_name,
                "device_id": stream.device_id,
                "pedal_count": stream.pedal_count,
                "dispatches_per_tick": stream.per_tick_dispatch_count,
                "descriptors_per_tick": stream.per_tick_descriptor_count,
                "push_constant_bytes_per_tick": stream.per_tick_push_constant_byte_count,
                "resident_capacity_activations": stream.dynamic_state_capacity_activations,
                "needed_capacity_activations": needed_capacity,
                "tokenizer": {
                    "add_special_tokens": args.add_special_tokens,
                    "skip_special_tokens": args.skip_special_tokens,
                },
                "prompt_text": prompt,
                "prompt_ids": turn.queued_input_event.encoded_token_ids,
                "generated_ids": turn.generated_token_ids,
                "generated_text": turn.generated_text,
                "output_text": turn.output_text,
                "stop_reason": engine_stop_label(turn.stop_condition),
                "scheduler_turns": turn.scheduler_turn_count(),
                "runtime_cycles": turn.runtime_cycle_count,
            }))?
        );
    } else if args.generated_only {
        print_text(&turn.generated_text);
    } else {
        print_text(&turn.output_text);
    }

    Ok(())
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
            "--max-new-tokens" => {
                parsed.max_new_tokens = parse_next(&mut raw, "--max-new-tokens")?;
            }
            "--capacity" => {
                parsed.capacity = Some(parse_next(&mut raw, "--capacity")?);
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

fn print_text(text: &str) {
    print!("{text}");
    if !text.ends_with('\n') {
        println!();
    }
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
  --max-new-tokens <N>       Public output tokens to emit after the prompt. Default: 4
  --capacity <N>             Override resident activation capacity selected from the package.
  --cycle-ticks <N>          Max runtime ticks per always-on cycle. Default: 4
  --max-scheduler-turns <N>  Max engine scheduler turns before stopping. Default: 1024
  --no-special-tokens        Do not add tokenizer special tokens to input text.
  --keep-special-tokens      Keep tokenizer special tokens in decoded output text.
  --generated-only           Print only newly generated text instead of prompt + generated text.
  --json                     Print a machine-readable run report.
  -h, --help                 Show this help.

Example:
  python -m llmoop --compile-model <MODEL_DIR>
  cargo run --manifest-path runtime-rs/Cargo.toml --features 'vulkan tokenizers' --bin llmoop-runtime -- --package lowered/model_xxx/vulkan_resident_greedy_package.json --prompt Hello --max-new-tokens 4"
}
