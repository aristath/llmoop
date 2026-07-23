fn run_placed_chat(
    args: &Args,
    manifest_dir: &Path,
    tokenizer_dir: &Path,
    runtime_model: VulkanResidentRuntimeModel,
    capacity: usize,
    codec: &VulkanResidentHfTokenizerTextCodec,
    initial_prompt: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let setup_start = Instant::now();
    let chat_session =
        RuntimeChatSession::from_tokenizer_dir(tokenizer_dir, &args.chat_template_variables)?;
    let stop_token_ids = chat_stop_token_ids_from_manifest(
        manifest_dir,
        tokenizer_dir,
        &runtime_model.package,
        &chat_session.formatter,
    )?;
    let transcript_codec = chat_transcript_codec(tokenizer_dir)?;
    let logical_device_ids = runtime_model.placement_device_ids();
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices_with_sampler_config(
        bound_devices.devices.clone(),
        manifest_dir,
        runtime_model,
        Some(capacity),
        args.random_seed,
        args.speculative_draft_tokens,
        sampler_runtime_config(args),
    )?;
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    let stream_snapshot = engine.add_stream("main", stream)?;
    println!(
        "nerve chat ready: placed_in_process, devices={:?}, context_size={}, setup_ms={:.3}",
        stream_snapshot.device_ids,
        stream_snapshot.context_window_activations,
        nanos_to_millis(elapsed_nanos_u64(setup_start))
    );

    run_chat_repl(
        initial_prompt,
        chat_session,
        codec,
        &transcript_codec,
        &stop_token_ids,
        |turn_index, token_ids| {
            print!("llm> ");
            io::stdout().flush()?;
            let mut decoder = codec.decode_stream();
            let mut output_error = None;
            let mut event = VulkanResidentTokenInputEvent::new(
                format!("chat_{turn_index}"),
                token_ids.to_vec(),
                args.max_new_tokens,
            )
            .with_origin("cli_chat");
            let input_event_id = event.id.clone();
            if !stop_token_ids.is_empty() {
                event = event.with_stop_tokens(stop_token_ids.clone());
            }
            reset_vulkan_resident_execution_counters();
            let run_start = Instant::now();
            let run = engine.submit_input_event_until_idle_with_output(
                "main",
                event,
                |output_event| {
                    if output_error.is_some() {
                        return;
                    }
                    match decoder.step(output_event.output_event.token_id) {
                        Ok(Some(text)) => {
                            print!("{text}");
                            if let Err(error) = io::stdout().flush() {
                                output_error = Some(error.to_string());
                            }
                        }
                        Ok(None) => {}
                        Err(error) => output_error = Some(error.to_string()),
                    }
                },
            )?;
            let run_time_ns = elapsed_nanos_u64(run_start);
            let execution_counters = vulkan_resident_execution_counters();
            let submitted_run = run
                .engine_run
                .input_runs
                .iter()
                .find(|input_run| {
                    input_run.stream_id == "main"
                        && input_run.submitted_run.input_event.id == input_event_id
                })
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        "placed chat engine run loop did not return the submitted chat event run",
                    )
                })?;
            let timing = runtime_prompt_timing_report(
                0,
                run_time_ns,
                token_ids.len(),
                run.generated_token_ids.len(),
                run.engine_run.scheduler_step_count,
                run.engine_run.activation_batch_count,
                run.engine_run.prefill_activation_batch_count,
                run.engine_run.decode_activation_batch_count,
                run.engine_run.max_activation_batch_width,
                run.engine_run.prefill_activation_count,
                run.engine_run.decode_activation_count,
                run.engine_run.prefill_time_ns,
                run.engine_run.decode_time_ns,
                submitted_run.submitted_run.session_run.tick_count,
                submitted_run
                    .submitted_run
                    .session_run
                    .run
                    .scheduler_turn_count,
            );
            if let Some(error) = output_error {
                return Err(Box::new(io::Error::new(io::ErrorKind::InvalidData, error)));
            }
            Ok(RuntimeChatTurn {
                generated_token_ids: run.generated_token_ids,
                streamed: true,
                timing,
                execution_counters,
                speculative_cycle_count: submitted_run
                    .submitted_run
                    .session_run
                    .run
                    .speculative_decode
                    .cycle_count,
                proposed_draft_token_count: submitted_run
                    .submitted_run
                    .session_run
                    .run
                    .speculative_decode
                    .proposed_draft_token_count,
                accepted_draft_token_count: submitted_run
                    .submitted_run
                    .session_run
                    .run
                    .speculative_decode
                    .accepted_draft_token_count,
                speculative_emitted_token_count: submitted_run
                    .submitted_run
                    .session_run
                    .run
                    .speculative_decode
                    .emitted_token_count,
                speculative_draft_time_ns: submitted_run
                    .submitted_run
                    .session_run
                    .run
                    .speculative_decode
                    .draft_time_ns,
                speculative_target_verification_time_ns: submitted_run
                    .submitted_run
                    .session_run
                    .run
                    .speculative_decode
                    .target_verification_time_ns,
                speculative_draft_catch_up_time_ns: submitted_run
                    .submitted_run
                    .session_run
                    .run
                    .speculative_decode
                    .draft_catch_up_time_ns,
            })
        },
    )
}

fn run_placed_prompt(
    context: &PromptRunContext<'_>,
    runtime_model: VulkanResidentRuntimeModel,
) -> Result<(), Box<dyn Error>> {
    let report = execute_placed_prompt_run(context, runtime_model)?;
    print_placed_prompt_report(context.args, &report)
}

fn execute_placed_prompt_run(
    context: &PromptRunContext<'_>,
    runtime_model: VulkanResidentRuntimeModel,
) -> Result<RuntimePlacedPromptRunReport, Box<dyn Error>> {
    let PromptRunContext {
        args,
        package_manifest,
        manifest_dir,
        tokenizer_dir,
        prompt,
        prompt_ids,
        scheduled_token_activations,
        capacity,
        codec,
        ..
    } = context;
    let setup_start = Instant::now();
    let logical_device_ids = runtime_model.placement_device_ids();
    let placement = runtime_model_placement(manifest_dir, &runtime_model)?;
    let bound_devices = runtime_bound_vulkan_devices(args, &logical_device_ids)?;
    let stream = VulkanResidentInProcessPlacedPromptStream::from_runtime_model_for_bound_devices_with_sampler_config(
        bound_devices.devices.clone(),
        manifest_dir,
        runtime_model,
        Some(*capacity),
        args.random_seed,
        args.speculative_draft_tokens,
        sampler_runtime_config(args),
    )?;
    let mut engine = VulkanResidentInProcessPlacedPromptEngine::new();
    let stream_snapshot = engine.add_stream("main", stream)?;
    let setup_time_ns = elapsed_nanos_u64(setup_start);
    let run_start = Instant::now();
    let input_event =
        VulkanResidentTokenInputEvent::new("prompt", prompt_ids.to_vec(), args.max_new_tokens);
    let input_event_id = input_event.id.clone();
    reset_vulkan_resident_execution_counters();
    let submitted_run = engine.submit_input_event_until_idle("main", input_event)?;
    let run_time_ns = elapsed_nanos_u64(run_start);
    let engine_run = submitted_run.engine_run;
    let prefill_activation_count = engine_run.prefill_activation_count;
    let decode_activation_count = engine_run.decode_activation_count;
    let prefill_time_ns = engine_run.prefill_time_ns;
    let decode_time_ns = engine_run.decode_time_ns;
    let scheduler_step_count = engine_run.scheduler_step_count;
    let activation_batch_count = engine_run.activation_batch_count;
    let prefill_activation_batch_count = engine_run.prefill_activation_batch_count;
    let decode_activation_batch_count = engine_run.decode_activation_batch_count;
    let max_activation_batch_width = engine_run.max_activation_batch_width;
    let run = engine_run
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
    let total_scheduler_turns = run.scheduler_turn_count;
    let completed_stage_deltas = vec![run.completed_stage_count];
    let tick_count = run.tick_count;
    let generated_token_count = run.generated_token_ids.len();
    let timing = runtime_prompt_timing_report(
        setup_time_ns,
        run_time_ns,
        prompt_ids.len(),
        generated_token_count,
        scheduler_step_count,
        activation_batch_count,
        prefill_activation_batch_count,
        decode_activation_batch_count,
        max_activation_batch_width,
        prefill_activation_count,
        decode_activation_count,
        prefill_time_ns,
        decode_time_ns,
        tick_count,
        total_scheduler_turns,
    );
    let component_timings = Vec::new();
    let component_timing_summaries = Vec::new();
    let transport_stats_by_tick = Vec::new();
    let transport_published_packet_count = run.transport_stats.published_packet_count;
    let transport_published_byte_count = run.transport_stats.published_byte_count;
    let transport_received_packet_count = run.transport_stats.received_packet_count;
    let transport_received_byte_count = run.transport_stats.received_byte_count;
    let transport_direct_copy_count = run.transport_stats.direct_copy_count;
    let transport_direct_copy_byte_count = run.transport_stats.direct_copy_byte_count;
    let transport_direct_receive_count = run.transport_stats.direct_receive_count;
    let transport_direct_receive_byte_count = run.transport_stats.direct_receive_byte_count;

    Ok(RuntimePlacedPromptRunReport {
        ok: true,
        execution_mode: "placed_in_process".to_string(),
        package_manifest: package_manifest.to_path_buf(),
        tokenizer_dir: tokenizer_dir.to_path_buf(),
        input_device_id: stream_snapshot.input_device_id.clone(),
        output_device_id: stream_snapshot.output_device_id.clone(),
        device_count: stream_snapshot.device_ids.len(),
        device_ids: stream_snapshot.device_ids.clone(),
        bound_devices: bound_devices_report(&bound_devices),
        edge_routes: bound_edge_routes_report(&bound_devices, &placement.edges),
        runtime_graph: runtime_graph_report(args),
        device_bindings: runtime_device_bindings_report(args, &stream_snapshot.device_ids),
        hosted_component_count: stream_snapshot.hosted_component_count,
        context_window_activations: stream_snapshot.context_window_activations,
        scheduled_token_activations: *scheduled_token_activations,
        tokenizer: tokenizer_options_report(args),
        prompt_text: prompt.to_string(),
        prompt_ids: run.prompt_token_ids.clone(),
        generated_ids: run.generated_token_ids.clone(),
        generated_text: generated_text.clone(),
        output_text: output_text.clone(),
        stop_reason: run.stop_reason.clone(),
        tick_count,
        scheduler_turns: total_scheduler_turns,
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
        component_timings,
        component_timing_summaries,
        speculative_cycle_count: run.speculative_decode.cycle_count,
        proposed_draft_token_count: run.speculative_decode.proposed_draft_token_count,
        accepted_draft_token_count: run.speculative_decode.accepted_draft_token_count,
        speculative_emitted_token_count: run.speculative_decode.emitted_token_count,
        speculative_draft_time_ns: run.speculative_decode.draft_time_ns,
        speculative_target_verification_time_ns: run.speculative_decode.target_verification_time_ns,
        speculative_draft_catch_up_time_ns: run.speculative_decode.draft_catch_up_time_ns,
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
        print_runtime_timing_stats("stats", &report.timing);
        print_runtime_execution_counters(&vulkan_resident_execution_counters());
        print_speculative_profile(report);
        print_placed_component_timing_profile(&report.component_timing_summaries, 5);
    }
    Ok(())
}

fn print_speculative_profile(report: &RuntimePlacedPromptRunReport) {
    print_runtime_speculative_stats(
        report.speculative_cycle_count,
        report.proposed_draft_token_count,
        report.accepted_draft_token_count,
        report.speculative_emitted_token_count,
        report.speculative_draft_time_ns,
        report.speculative_target_verification_time_ns,
        report.speculative_draft_catch_up_time_ns,
    );
}

fn generated_tokens_per_second(generated_token_count: usize, run_time_ns: u64) -> Option<f64> {
    if run_time_ns == 0 {
        None
    } else {
        Some(generated_token_count as f64 / (run_time_ns as f64 / 1_000_000_000.0))
    }
}
