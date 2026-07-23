fn print_text(text: &str) {
    print!("{text}");
    if !text.ends_with('\n') {
        println!();
    }
}

fn print_runtime_timing_stats(label: &str, timing: &RuntimePromptTimingReport) {
    println!("{label}:");
    println!("  setup_ms={:.3}", nanos_to_millis(timing.setup_time_ns));
    println!("  run_ms={:.3}", nanos_to_millis(timing.run_time_ns));
    println!("  total_ms={:.3}", nanos_to_millis(timing.total_time_ns));
    println!("  generated_tokens={}", timing.generated_token_count);
    if let Some(tokens_per_second) =
        generated_tokens_per_second(timing.generated_token_count, timing.run_time_ns)
    {
        println!("  generated_tokens_per_second={tokens_per_second:.3}");
    }
    println!("  prefill_tokens={}", timing.prefill_token_count);
    if let Some(tokens_per_second) =
        generated_tokens_per_second(timing.prefill_token_count, timing.prefill_time_ns)
    {
        println!("  prefill_tokens_per_second={tokens_per_second:.3}");
    }
    println!("  decode_tokens={}", timing.decode_token_count);
    if let Some(tokens_per_second) =
        generated_tokens_per_second(timing.decode_token_count, timing.decode_time_ns)
    {
        println!("  decode_tokens_per_second={tokens_per_second:.3}");
    }
    println!("  prefill_activations={}", timing.prefill_activation_count);
    println!("  decode_activations={}", timing.decode_activation_count);
    println!("  scheduler_steps={}", timing.scheduler_step_count);
    println!("  activation_batches={}", timing.activation_batch_count);
    println!(
        "  prefill_activation_batches={}",
        timing.prefill_activation_batch_count
    );
    println!(
        "  decode_activation_batches={}",
        timing.decode_activation_batch_count
    );
    println!(
        "  max_activation_batch_width={}",
        timing.max_activation_batch_width
    );
    println!(
        "  prefill_ms={:.3}",
        nanos_to_millis(timing.prefill_time_ns)
    );
    println!("  decode_ms={:.3}", nanos_to_millis(timing.decode_time_ns));
    println!("  ticks={}", timing.tick_count);
    println!("  scheduler_turns={}", timing.scheduler_turn_count);
    if let Some(average) = timing.average_generated_token_time_ns {
        println!("  avg_generated_token_ms={:.3}", nanos_to_millis(average));
    }
    if let Some(average) = timing.average_prefill_activation_time_ns {
        println!(
            "  avg_prefill_activation_ms={:.3}",
            nanos_to_millis(average)
        );
    }
    if let Some(average) = timing.average_decode_activation_time_ns {
        println!(
            "  avg_decode_activation_ms={:.3}",
            nanos_to_millis(average)
        );
    }
    if let Some(average) = timing.average_tick_time_ns {
        println!("  avg_tick_ms={:.3}", nanos_to_millis(average));
    }
    if let Some(average) = timing.average_scheduler_turn_time_ns {
        println!("  avg_scheduler_turn_ms={:.3}", nanos_to_millis(average));
    }
}

fn print_runtime_execution_counters(counters: &VulkanResidentExecutionCounters) {
    println!("execution:");
    println!(
        "  resident_sequence_prepare_calls={}",
        counters.resident_sequence_prepare_calls
    );
    println!(
        "  resident_sequence_recorded_command_buffers={}",
        counters.resident_sequence_recorded_command_buffers
    );
    println!(
        "  resident_sequence_reused_command_buffers={}",
        counters.resident_sequence_reused_command_buffers
    );
    println!(
        "  resident_sequence_queue_submits={}",
        counters.resident_sequence_queue_submits
    );
    println!(
        "  resident_sequence_fence_waits={}",
        counters.resident_sequence_fence_waits
    );
    println!(
        "  resident_queue_batch_submits={}",
        counters.resident_queue_batch_submits
    );
    println!(
        "  resident_queue_batch_commands={}",
        counters.resident_queue_batch_commands
    );
    println!(
        "  resident_copy_queue_submits={}",
        counters.resident_copy_queue_submits
    );
    println!("  resident_copy_waits={}", counters.resident_copy_waits);
}

fn print_runtime_speculative_stats(
    cycle_count: usize,
    proposed_draft_token_count: usize,
    accepted_draft_token_count: usize,
    emitted_token_count: usize,
    draft_time_ns: u64,
    target_verification_time_ns: u64,
    draft_catch_up_time_ns: u64,
) {
    if cycle_count == 0 {
        return;
    }
    let acceptance = if proposed_draft_token_count == 0 {
        0.0
    } else {
        100.0 * accepted_draft_token_count as f64 / proposed_draft_token_count as f64
    };
    println!("speculative:");
    println!("  cycles={cycle_count}");
    println!(
        "  drafts proposed={} accepted={} acceptance={acceptance:.2}%",
        proposed_draft_token_count, accepted_draft_token_count
    );
    println!("  emitted_tokens={emitted_token_count}");
    println!("  draft_ms={:.3}", nanos_to_millis(draft_time_ns));
    println!(
        "  target_verification_ms={:.3}",
        nanos_to_millis(target_verification_time_ns)
    );
    println!(
        "  draft_catch_up_ms={:.3}",
        nanos_to_millis(draft_catch_up_time_ns)
    );
}

fn print_placed_component_timing_profile(
    summaries: &[RuntimePlacedComponentTimingSummaryReport],
    max_rows: usize,
) {
    if summaries.is_empty() || max_rows == 0 {
        return;
    }
    println!("top_nodes:");
    for summary in summaries.iter().take(max_rows) {
        println!(
            "  {}:{} total_ms={:.3} ticks={} dispatches={} avg_tick_ms={} avg_dispatch_ms={}",
            summary.device_id,
            summary.component_id,
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

fn print_usage() {
    println!("{}", usage());
}

fn usage() -> &'static str {
    "Usage: nerve-runtime --package <COMPILED_PACKAGE.json> (--prompt <TEXT> | --chat) [OPTIONS]

Options:
  --package <PATH>           Compiled resident model package manifest. Required.
  --prompt <TEXT>            External text event to inject into the resident stream.
                             With --chat, this is the optional first message.
  --chat                     Start an interactive resident text session.
  --chat-template-var <NAME=JSON>
                             Set a model-owned chat template variable; may be repeated.
  --device <DEVICE_ID>       Default logical device for unplaced nodes. May be supplied once.
  --place-node <NODE=DEV>    Assign one runtime node instance to a logical device.
  --bind-device <DEV=TARGET> Bind a logical device to a discovered Vulkan device ID.
  --chain <ITEM[,ITEM...]>    Runtime source chain. ITEM is SOURCE or INSTANCE=SOURCE.
  --duplicate-after <AFTER=NEW>
                             Duplicate runtime node instance AFTER with id NEW.
  --inspect-runtime          Preview UI-ready package, runtime graph, placement, device, and route facts.
  --inspect-package          Summarize the compiled component catalog and available devices.
  --inspect-graph            Preview the effective runtime graph without mounting devices.
  --inspect-placement        Mount and summarize every logical device slice in the runtime graph.
  --inspect-device-slice <DEVICE_ID>
                             Mount and summarize only the runtime graph nodes assigned to DEVICE_ID.
  --max-new-tokens <N>       Generation stop condition, independent of context size. Default: 65536
  --speculative-draft-tokens <N>
                             MTP draft tokens proposed per verification cycle. Default: 0 (disabled).
  --context-size <N>         Runtime transient-state window. Default: auto, up to the model maximum.
  --vulkan-device-index <N>  Use Vulkan physical device index N as the default local target.
  --seed <U32>               Explicit sampler randomness seed. Default: 0
  --temperature <F32>        Override the compiled sampler temperature for this stream.
  --top-k <N>                Override top-k, up to the package's compiled runtime capacity.
  --top-p <F32>              Override nucleus probability in (0, 1].
  --min-p <F32>              Override minimum relative token probability in [0, 1].
  --presence-penalty <F32>   Subtract this value once from logits of previously seen tokens.
  --repetition-penalty <F32> Override the positive multiplicative repetition penalty.
  --no-special-tokens        Do not add tokenizer special tokens to raw --prompt input.
                             Chat templates always own their complete special-token framing.
  --keep-special-tokens      Keep tokenizer special tokens in decoded output text.
  --generated-only           Print only newly generated text instead of prompt + generated text.
  --json                     Print a machine-readable run report.
  -h, --help                 Show this help.

Example:
  python -m nerve --compile-model <MODEL_DIR>
  cargo run --manifest-path runtime-rs/Cargo.toml --features 'vulkan tokenizers' --bin nerve-runtime -- --package compiled_models/model_xxx/vulkan_resident_package.json --chat"
}
