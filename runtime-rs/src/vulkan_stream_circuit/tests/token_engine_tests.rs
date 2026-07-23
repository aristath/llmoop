#[test]
fn resident_token_runtime_queues_events_and_runs_bounded_cycles() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident token runtime cycle: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "resident token runtime cycle",
        8,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    ) else {
        return;
    };
    let mut runtime = VulkanResidentTokenRuntime::from_processor("runtime_stream_0", processor);
    let initial = runtime.snapshot();
    assert!(initial.idle);
    assert!(!initial.running);
    assert!(initial.stream.idle);
    assert_eq!(initial.pending_input_event_count, 0);

    let queued_first = runtime
        .enqueue_input_event(
            VulkanResidentTokenInputEvent::new("event_0", vec![1], 3).with_origin("test_host"),
        )
        .unwrap();
    assert_eq!(queued_first.pending_input_event_count, 1);
    let queued_second = runtime
        .enqueue_input_event(
            VulkanResidentTokenInputEvent::new("event_1", vec![36_309], 1).with_origin("test_host"),
        )
        .unwrap();
    assert_eq!(queued_second.pending_input_event_count, 2);
    let queued_snapshot = runtime.snapshot();
    assert!(!queued_snapshot.idle);
    assert!(queued_snapshot.running);
    assert!(queued_snapshot.stream.idle);
    assert_eq!(queued_snapshot.pending_input_event_count, 2);

    let no_budget = runtime.run_cycle(&device, 0).unwrap();
    assert_eq!(
        no_budget.stop_condition,
        VulkanResidentTokenRuntimeCycleStopCondition::TickBudget
    );
    assert_eq!(no_budget.ticks_used, 0);
    assert_eq!(no_budget.pending_input_event_count, 2);
    assert!(no_budget.stream_idle);

    let first_cycle = runtime.run_cycle(&device, 2).unwrap();
    assert_eq!(first_cycle.stream_id, "runtime_stream_0");
    assert_eq!(first_cycle.start_stream_tick, 0);
    assert_eq!(first_cycle.next_stream_tick, 2);
    assert_eq!(first_cycle.max_ticks, 2);
    assert_eq!(first_cycle.ticks_used, 2);
    assert_eq!(
        first_cycle.stop_condition,
        VulkanResidentTokenRuntimeCycleStopCondition::TickBudget
    );
    assert_eq!(first_cycle.queued_input_events.len(), 1);
    assert_eq!(first_cycle.queued_input_events[0].input_event.id, "event_0");
    assert_eq!(first_cycle.pending_input_event_count, 1);
    assert!(!first_cycle.stream_idle);
    assert_eq!(first_cycle.processed_tick_count, 2);
    assert_eq!(first_cycle.idle_tick_count, 0);
    assert_eq!(first_cycle.output_events.len(), 2);
    assert_eq!(
        first_cycle
            .output_events
            .iter()
            .map(|event| (event.input_event_id.as_str(), event.output_index))
            .collect::<Vec<_>>(),
        vec![("event_0", 0), ("event_0", 1)]
    );

    let second_cycle = runtime.run_cycle(&device, 4).unwrap();
    assert_eq!(second_cycle.start_stream_tick, 2);
    assert_eq!(second_cycle.next_stream_tick, 5);
    assert_eq!(second_cycle.ticks_used, 4);
    assert_eq!(
        second_cycle.stop_condition,
        VulkanResidentTokenRuntimeCycleStopCondition::TickBudget
    );
    assert_eq!(second_cycle.queued_input_events.len(), 1);
    assert_eq!(
        second_cycle.queued_input_events[0].input_event.id,
        "event_1"
    );
    assert_eq!(second_cycle.pending_input_event_count, 0);
    assert!(!second_cycle.stream_idle);
    assert_eq!(second_cycle.processed_tick_count, 3);
    assert_eq!(second_cycle.idle_tick_count, 1);
    assert_eq!(second_cycle.output_events.len(), 2);
    assert_eq!(
        second_cycle
            .output_events
            .iter()
            .map(|event| (event.input_event_id.as_str(), event.output_index))
            .collect::<Vec<_>>(),
        vec![("event_0", 2), ("event_1", 0)]
    );
    assert_eq!(second_cycle.output_events[0].source_stream_tick, 2);
    assert_eq!(second_cycle.output_events[1].source_stream_tick, 4);

    let final_cycle = runtime.run_cycle(&device, 3).unwrap();
    assert_eq!(final_cycle.start_stream_tick, 5);
    assert_eq!(final_cycle.next_stream_tick, 6);
    assert_eq!(
        final_cycle.stop_condition,
        VulkanResidentTokenRuntimeCycleStopCondition::Idle
    );
    assert_eq!(final_cycle.ticks_used, 2);
    assert_eq!(final_cycle.processed_tick_count, 1);
    assert_eq!(final_cycle.idle_tick_count, 1);
    assert!(final_cycle.output_events.is_empty());
    assert_eq!(final_cycle.pending_input_event_count, 0);
    assert!(final_cycle.stream_idle);

    let idle_cycle = runtime.run_cycle(&device, 3).unwrap();
    assert_eq!(
        idle_cycle.stop_condition,
        VulkanResidentTokenRuntimeCycleStopCondition::Idle
    );
    assert_eq!(idle_cycle.ticks_used, 0);
    assert_eq!(idle_cycle.pending_input_event_count, 0);
    assert!(idle_cycle.stream_idle);

    let final_snapshot = runtime.snapshot();
    assert!(final_snapshot.idle);
    assert!(!final_snapshot.running);
    assert_eq!(final_snapshot.stream.next_stream_tick, 6);
    assert_eq!(final_snapshot.stream.total_public_outputs, 4);
    assert_eq!(final_snapshot.pending_input_event_count, 0);
}

#[test]
fn resident_token_runtime_scheduler_round_robins_registered_runtime() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident token runtime scheduler: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "resident token runtime scheduler",
        8,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    ) else {
        return;
    };
    let runtime = VulkanResidentTokenRuntime::from_processor("scheduler_stream_0", processor);
    let mut scheduler = VulkanResidentTokenRuntimeScheduler::new();

    let initial = scheduler.snapshot();
    assert_eq!(initial.registered_runtime_count, 0);
    assert_eq!(initial.active_runtime_count, 0);
    assert!(initial.idle);
    assert!(!initial.running);
    assert!(initial.runtimes.is_empty());

    scheduler.add_runtime(runtime).unwrap();
    assert!(scheduler.has_runtime("scheduler_stream_0"));
    let registered = scheduler.snapshot();
    assert_eq!(registered.registered_runtime_count, 1);
    assert_eq!(registered.active_runtime_count, 0);
    assert!(registered.idle);
    assert!(!registered.running);
    assert_eq!(registered.runtimes.len(), 1);
    assert_eq!(
        registered.runtimes[0].stream.stream_id,
        "scheduler_stream_0"
    );

    let queued_first = scheduler
        .enqueue_input_event(
            "scheduler_stream_0",
            VulkanResidentTokenInputEvent::new("event_0", vec![1], 3).with_origin("test_host"),
        )
        .unwrap();
    assert_eq!(queued_first.pending_input_event_count, 1);
    let queued_second = scheduler
        .enqueue_input_event(
            "scheduler_stream_0",
            VulkanResidentTokenInputEvent::new("event_1", vec![36_309], 1).with_origin("test_host"),
        )
        .unwrap();
    assert_eq!(queued_second.pending_input_event_count, 2);
    let queued = scheduler.snapshot();
    assert_eq!(queued.registered_runtime_count, 1);
    assert_eq!(queued.active_runtime_count, 1);
    assert!(!queued.idle);
    assert!(queued.running);
    assert_eq!(queued.runtimes[0].pending_input_event_count, 2);
    assert!(queued.runtimes[0].stream.idle);

    let no_budget = scheduler.run_cycle(&device, 0, 2).unwrap();
    assert_eq!(
        no_budget.stop_condition,
        VulkanResidentTokenRuntimeSchedulerStopCondition::RuntimeCycleBudget
    );
    assert!(no_budget.runtime_cycles.is_empty());
    assert!(no_budget.output_events.is_empty());
    assert_eq!(no_budget.active_runtime_count, 1);
    assert_eq!(no_budget.registered_runtime_count, 1);

    let first = scheduler.run_cycle(&device, 1, 2).unwrap();
    assert_eq!(
        first.stop_condition,
        VulkanResidentTokenRuntimeSchedulerStopCondition::RuntimeCycleBudget
    );
    assert_eq!(first.runtime_cycles.len(), 1);
    assert_eq!(first.runtime_cycles[0].stream_id, "scheduler_stream_0");
    assert_eq!(first.runtime_cycles[0].start_stream_tick, 0);
    assert_eq!(first.runtime_cycles[0].next_stream_tick, 2);
    assert_eq!(first.runtime_cycles[0].pending_input_event_count, 1);
    assert_eq!(first.output_events.len(), 2);
    assert!(
        first
            .output_events
            .iter()
            .all(|event| event.stream_id == "scheduler_stream_0")
    );
    assert_eq!(
        first
            .output_events
            .iter()
            .map(|event| {
                (
                    event.output_event.input_event_id.as_str(),
                    event.output_event.output_index,
                )
            })
            .collect::<Vec<_>>(),
        vec![("event_0", 0), ("event_0", 1)]
    );
    assert_eq!(first.active_runtime_count, 1);
    assert_eq!(first.registered_runtime_count, 1);

    let second = scheduler.run_cycle(&device, 1, 4).unwrap();
    assert_eq!(
        second.stop_condition,
        VulkanResidentTokenRuntimeSchedulerStopCondition::RuntimeCycleBudget
    );
    assert_eq!(second.runtime_cycles.len(), 1);
    assert_eq!(second.runtime_cycles[0].stream_id, "scheduler_stream_0");
    assert_eq!(second.runtime_cycles[0].start_stream_tick, 2);
    assert_eq!(second.runtime_cycles[0].next_stream_tick, 5);
    assert_eq!(second.runtime_cycles[0].pending_input_event_count, 0);
    assert_eq!(second.output_events.len(), 2);
    assert!(
        second
            .output_events
            .iter()
            .all(|event| event.stream_id == "scheduler_stream_0")
    );
    assert_eq!(
        second
            .output_events
            .iter()
            .map(|event| {
                (
                    event.output_event.input_event_id.as_str(),
                    event.output_event.output_index,
                )
            })
            .collect::<Vec<_>>(),
        vec![("event_0", 2), ("event_1", 0)]
    );
    assert_eq!(second.active_runtime_count, 1);
    assert_eq!(second.registered_runtime_count, 1);

    let final_run = scheduler.run_cycle(&device, 1, 3).unwrap();
    assert_eq!(
        final_run.stop_condition,
        VulkanResidentTokenRuntimeSchedulerStopCondition::Idle
    );
    assert_eq!(final_run.runtime_cycles.len(), 1);
    assert_eq!(final_run.runtime_cycles[0].stream_id, "scheduler_stream_0");
    assert_eq!(final_run.runtime_cycles[0].start_stream_tick, 5);
    assert_eq!(final_run.runtime_cycles[0].next_stream_tick, 6);
    assert_eq!(
        final_run.runtime_cycles[0].stop_condition,
        VulkanResidentTokenRuntimeCycleStopCondition::Idle
    );
    assert!(final_run.output_events.is_empty());
    assert_eq!(final_run.active_runtime_count, 0);
    assert_eq!(final_run.registered_runtime_count, 1);

    let idle_run = scheduler.run_cycle(&device, 1, 3).unwrap();
    assert_eq!(
        idle_run.stop_condition,
        VulkanResidentTokenRuntimeSchedulerStopCondition::Idle
    );
    assert!(idle_run.runtime_cycles.is_empty());
    assert!(idle_run.output_events.is_empty());
    assert_eq!(idle_run.active_runtime_count, 0);
    assert_eq!(idle_run.registered_runtime_count, 1);

    let final_snapshot = scheduler.snapshot();
    assert_eq!(final_snapshot.registered_runtime_count, 1);
    assert_eq!(final_snapshot.active_runtime_count, 0);
    assert!(final_snapshot.idle);
    assert!(!final_snapshot.running);
    assert_eq!(final_snapshot.runtimes[0].stream.next_stream_tick, 6);
    assert_eq!(final_snapshot.runtimes[0].stream.total_public_outputs, 4);
    assert_eq!(final_snapshot.runtimes[0].pending_input_event_count, 0);
}

#[test]
fn resident_token_id_text_codec_encodes_and_decodes_numeric_token_text() {
    let codec = VulkanResidentTokenIdTextCodec;

    assert_eq!(codec.encode_text("1, 2\n3\t4").unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(codec.decode_tokens(&[1, 2, 3, 4]).unwrap(), "1 2 3 4");
    assert!(codec.encode_text("").is_err());
    assert!(codec.encode_text("not-a-token").is_err());
}

#[cfg(feature = "tokenizers")]
#[test]
fn resident_hf_tokenizer_text_codec_loads_fixture_model_tokenizer_json() {
    let Some(codec) = fixture_model_tokenizer_codec_or_skip("resident hf tokenizer text codec")
    else {
        return;
    };

    assert!(codec.add_special_tokens());
    assert!(codec.skip_special_tokens());
    assert_eq!(codec.encode_text("Hello").unwrap(), vec![1, 36_309]);
    assert_eq!(codec.decode_tokens(&[1, 36_309]).unwrap(), "Hello");

    let codec_with_specials = codec.with_skip_special_tokens(false);
    assert_eq!(
        codec_with_specials.decode_tokens(&[1, 36_309]).unwrap(),
        "<|startoftext|>Hello"
    );
}

#[test]
fn resident_token_engine_owns_device_scheduler_and_registered_stream() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident token engine: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "resident token engine",
        8,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    ) else {
        return;
    };
    let mut engine =
        VulkanResidentTokenEngine::from_processor(device, "engine_stream_0", processor).unwrap();

    let initial = engine.snapshot();
    assert!(!initial.device_name.is_empty());
    assert_eq!(initial.streams.len(), 1);
    assert_eq!(initial.streams[0].stream_id, "engine_stream_0");
    assert_eq!(initial.streams[0].device_id, "gpu0");
    assert_eq!(initial.streams[0].pedal_count, 14);
    assert_eq!(initial.streams[0].dynamic_state_capacity_activations, 8);
    assert_eq!(initial.scheduler.registered_runtime_count, 1);
    assert_eq!(initial.scheduler.active_runtime_count, 0);
    assert!(initial.scheduler.idle);
    assert!(!initial.scheduler.running);

    let submitted_text = engine
        .submit_text_until_idle(
            VulkanResidentTokenEngineTextInputRequest::new("engine_stream_0", "event_0", "1", 2)
                .with_origin("test_host"),
            VulkanResidentTokenEngineRunBudget::new(8, 1, 2),
            &VulkanResidentTokenIdTextCodec,
        )
        .unwrap();
    assert_eq!(submitted_text.stream_id, "engine_stream_0");
    assert_eq!(submitted_text.input_event_id, "event_0");
    assert_eq!(submitted_text.input_text, "1");
    assert_eq!(submitted_text.encoded_token_ids, vec![1]);
    assert_eq!(submitted_text.generated_text, "1 1");
    let submitted = submitted_text.submitted_tokens;
    assert_eq!(submitted.stream_id, "engine_stream_0");
    assert_eq!(submitted.input_event_id, "event_0");
    assert_eq!(submitted.queued_input_event.pending_input_event_count, 1);
    assert_eq!(submitted.generated_token_ids, vec![1, 1]);
    assert_eq!(submitted.output_events.len(), 2);
    assert!(
        submitted
            .output_events
            .iter()
            .all(|event| event.stream_id == "engine_stream_0"
                && event.output_event.input_event_id == "event_0")
    );
    let run = submitted.run;

    assert_eq!(
        run.stop_condition,
        VulkanResidentTokenEngineRunStopCondition::Idle
    );
    assert!(run.runtime_cycle_count >= 1);
    assert_eq!(run.end_snapshot.scheduler.active_runtime_count, 0);
    let final_snapshot = run.end_snapshot;
    assert_eq!(final_snapshot.scheduler.registered_runtime_count, 1);
    assert_eq!(final_snapshot.scheduler.active_runtime_count, 0);
    assert!(final_snapshot.scheduler.idle);
    assert!(!final_snapshot.scheduler.running);
    let runtime = engine.runtime_snapshot("engine_stream_0").unwrap();
    assert!(runtime.idle);
    assert!(!runtime.running);
    assert_eq!(runtime.stream.total_public_outputs, 2);
    assert_eq!(runtime.pending_input_event_count, 0);
}

#[test]
fn resident_token_engine_drains_text_output_cycle_by_cycle() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident token text cycle engine: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "resident token text cycle engine",
        8,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    ) else {
        return;
    };
    let mut engine =
        VulkanResidentTokenEngine::from_processor(device, "engine_text_cycle", processor).unwrap();
    let codec = VulkanResidentTokenIdTextCodec;

    let queued = engine
        .enqueue_text_input_event("engine_text_cycle", "event_0", "1", 2, "test_host", &codec)
        .unwrap();
    assert_eq!(queued.stream_id, "engine_text_cycle");
    assert_eq!(queued.input_event_id, "event_0");
    assert_eq!(queued.input_text, "1");
    assert_eq!(queued.encoded_token_ids, vec![1]);
    assert_eq!(queued.queued_input_event.pending_input_event_count, 1);

    let first = engine.run_text_cycle(1, 2, &codec).unwrap();
    assert_eq!(
        first.scheduler_run.stop_condition,
        VulkanResidentTokenRuntimeSchedulerStopCondition::RuntimeCycleBudget
    );
    assert_eq!(first.generated_token_ids, vec![1, 1]);
    assert_eq!(first.generated_text, "1 1");
    assert_eq!(
        first
            .output_events
            .iter()
            .map(|event| {
                (
                    event.stream_id.as_str(),
                    event.input_event_id.as_str(),
                    event.output_index,
                    event.token_id,
                    event.text.as_str(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            ("engine_text_cycle", "event_0", 0, 1, "1"),
            ("engine_text_cycle", "event_0", 1, 1, "1")
        ]
    );

    let final_cycle = engine.run_text_cycle(1, 2, &codec).unwrap();
    assert_eq!(
        final_cycle.scheduler_run.stop_condition,
        VulkanResidentTokenRuntimeSchedulerStopCondition::Idle
    );
    assert!(final_cycle.output_events.is_empty());
    assert!(final_cycle.generated_token_ids.is_empty());
    assert_eq!(final_cycle.generated_text, "");
    assert!(engine.runtime_snapshot("engine_text_cycle").unwrap().idle);
}

#[test]
fn resident_token_engine_live_text_turn_accumulates_filtered_outputs() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident token live text turn engine: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "resident token live text turn engine",
        8,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    ) else {
        return;
    };
    let mut engine =
        VulkanResidentTokenEngine::from_processor(device, "engine_live_text_turn", processor)
            .unwrap();
    let codec = VulkanResidentTokenIdTextCodec;

    let turn = engine
        .submit_live_text_turn_until_idle(
            VulkanResidentTokenEngineTextInputRequest::new(
                "engine_live_text_turn",
                "event_0",
                "1",
                2,
            )
            .with_origin("test_host"),
            VulkanResidentTokenEngineRunBudget::new(8, 1, 2),
            &codec,
        )
        .unwrap();

    assert_eq!(turn.stream_id, "engine_live_text_turn");
    assert_eq!(turn.input_event_id, "event_0");
    assert_eq!(turn.queued_input_event.input_text, "1");
    assert_eq!(turn.queued_input_event.encoded_token_ids, vec![1]);
    assert_eq!(turn.scheduler_turn_count(), 2);
    assert_eq!(turn.runtime_cycle_count, 2);
    assert_eq!(turn.generated_token_ids, vec![1, 1]);
    assert_eq!(turn.generated_text, "1 1");
    assert_eq!(turn.output_text, "1 1 1");
    assert_eq!(
        turn.stop_condition,
        VulkanResidentTokenEngineRunStopCondition::Idle
    );
    assert_eq!(turn.output_events.len(), 2);
    assert!(turn.output_events.iter().all(
        |event| event.stream_id == "engine_live_text_turn" && event.input_event_id == "event_0"
    ));
    assert!(
        engine
            .runtime_snapshot("engine_live_text_turn")
            .unwrap()
            .idle
    );
}

#[test]
fn resident_token_engine_live_text_batch_round_robins_shared_model_streams() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident token live text batch streams: {error}");
            return;
        }
    };
    let model = fixture_model_resident_greedy_model(&device, 8).unwrap();
    let mut engine = VulkanResidentTokenEngine::new(device);
    engine
        .add_model_package("shared_compiled_model", model)
        .unwrap();
    engine
        .create_stream_from_model("shared_compiled_model", "text_batch_stream_a", 0)
        .unwrap();
    engine
        .create_stream_from_model("shared_compiled_model", "text_batch_stream_b", 1)
        .unwrap();
    let codec = VulkanResidentTokenIdTextCodec;

    let batch = engine
        .submit_live_text_batch_until_idle(
            vec![
                VulkanResidentTokenEngineTextInputRequest::new(
                    "text_batch_stream_a",
                    "event_a",
                    "1",
                    2,
                )
                .with_origin("test_host"),
                VulkanResidentTokenEngineTextInputRequest::new(
                    "text_batch_stream_b",
                    "event_b",
                    "1",
                    2,
                )
                .with_origin("test_host"),
            ],
            VulkanResidentTokenEngineRunBudget::new(8, 2, 2),
            &codec,
        )
        .unwrap();

    assert_eq!(batch.queued_input_events.len(), 2);
    assert_eq!(batch.scheduler_turn_count(), 2);
    assert_eq!(batch.runtime_cycle_count, 4);
    assert_eq!(batch.output_events.len(), 4);
    assert_eq!(batch.generated_token_ids, vec![1, 1, 1, 1]);
    assert_eq!(batch.generated_text, "1 1 1 1");
    assert_eq!(
        batch.stop_condition,
        VulkanResidentTokenEngineRunStopCondition::Idle
    );
    assert_eq!(
        batch.generated_token_ids_for("text_batch_stream_a", "event_a"),
        vec![1, 1]
    );
    assert_eq!(
        batch.generated_token_ids_for("text_batch_stream_b", "event_b"),
        vec![1, 1]
    );
    assert_eq!(
        batch
            .cycles
            .iter()
            .flat_map(|cycle| cycle.scheduler_run.runtime_cycles.iter())
            .map(|cycle| cycle.stream_id.as_str())
            .collect::<Vec<_>>(),
        vec![
            "text_batch_stream_a",
            "text_batch_stream_b",
            "text_batch_stream_a",
            "text_batch_stream_b"
        ]
    );
    assert!(engine.runtime_snapshot("text_batch_stream_a").unwrap().idle);
    assert!(engine.runtime_snapshot("text_batch_stream_b").unwrap().idle);
}

#[cfg(feature = "tokenizers")]
#[test]
fn resident_token_engine_accepts_hf_tokenizer_text_input() {
    let Some(codec) =
        fixture_model_tokenizer_codec_or_skip("resident token engine hf tokenizer input")
    else {
        return;
    };
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident token engine hf tokenizer input: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor_with_capacity(
        &device,
        "resident token engine hf tokenizer input",
        8,
        "gqa_attention_bf16_q16_kv8_d64.comp",
    ) else {
        return;
    };
    let mut engine =
        VulkanResidentTokenEngine::from_processor(device, "engine_text_stream", processor).unwrap();

    let submitted = engine
        .submit_text_until_idle(
            VulkanResidentTokenEngineTextInputRequest::new(
                "engine_text_stream",
                "hello_event",
                "Hello",
                1,
            )
            .with_origin("test_host"),
            VulkanResidentTokenEngineRunBudget::new(8, 1, 3),
            &codec,
        )
        .unwrap();

    assert_eq!(submitted.stream_id, "engine_text_stream");
    assert_eq!(submitted.input_event_id, "hello_event");
    assert_eq!(submitted.input_text, "Hello");
    assert_eq!(submitted.encoded_token_ids, vec![1, 36_309]);
    assert_eq!(submitted.submitted_tokens.generated_token_ids.len(), 1);
    assert_eq!(
        submitted.generated_text,
        codec
            .decode_tokens(&submitted.submitted_tokens.generated_token_ids)
            .unwrap()
    );
    assert_eq!(
        submitted.submitted_tokens.run.stop_condition,
        VulkanResidentTokenEngineRunStopCondition::Idle
    );
    assert!(engine.runtime_snapshot("engine_text_stream").unwrap().idle);
}

#[test]
fn resident_token_engine_creates_two_streams_from_one_shared_model() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident token shared model streams: {error}");
            return;
        }
    };
    let model = fixture_model_resident_greedy_model(&device, 8).unwrap();
    let mut engine = VulkanResidentTokenEngine::new(device);
    let loaded_model = engine
        .add_model_package("shared_compiled_model", model)
        .unwrap();
    assert_eq!(loaded_model.model_id, "shared_compiled_model");
    assert_eq!(loaded_model.device_id, "runtime_default");
    assert_eq!(loaded_model.registered_stream_count, 0);
    assert_eq!(loaded_model.dynamic_state_capacity_activations, 8);
    assert!(loaded_model.permanent_parameter_count > 0);
    assert!(loaded_model.permanent_parameter_bytes > 0);
    assert!(loaded_model.transducer_parameter_count > 0);
    assert!(loaded_model.transducer_parameter_bytes > 0);
    assert!(loaded_model.reusable_kernel_word_count > 0);

    let stream_a = engine
        .create_stream_from_model("shared_compiled_model", "shared_stream_a", 0)
        .unwrap();
    let stream_b = engine
        .create_stream_from_model("shared_compiled_model", "shared_stream_b", 1)
        .unwrap();
    assert_eq!(stream_a.model_id.as_deref(), Some("shared_compiled_model"));
    assert_eq!(stream_b.model_id.as_deref(), Some("shared_compiled_model"));
    assert_eq!(
        stream_a.residency,
        VulkanResidentTokenEngineStreamResidency::SharedModel
    );
    assert_eq!(
        stream_b.residency,
        VulkanResidentTokenEngineStreamResidency::SharedModel
    );
    assert_eq!(stream_a.pedal_count, stream_b.pedal_count);
    assert_eq!(
        stream_a.dynamic_state_capacity_activations,
        stream_b.dynamic_state_capacity_activations
    );

    let registered = engine.snapshot();
    assert_eq!(registered.models.len(), 1);
    assert_eq!(registered.models[0].model_id, "shared_compiled_model");
    assert_eq!(registered.models[0].registered_stream_count, 2);
    assert_eq!(registered.streams.len(), 2);
    assert_eq!(registered.scheduler.registered_runtime_count, 2);
    assert_eq!(registered.scheduler.active_runtime_count, 0);

    engine
        .enqueue_input_event(
            "shared_stream_a",
            VulkanResidentTokenInputEvent::new("event_a", vec![1], 2).with_origin("test_host"),
        )
        .unwrap();
    engine
        .enqueue_input_event(
            "shared_stream_b",
            VulkanResidentTokenInputEvent::new("event_b", vec![1], 2).with_origin("test_host"),
        )
        .unwrap();
    assert_eq!(engine.snapshot().scheduler.active_runtime_count, 2);

    let run = engine.run_until_idle(8, 2, 2).unwrap();
    assert_eq!(
        run.stop_condition,
        VulkanResidentTokenEngineRunStopCondition::Idle
    );
    assert_eq!(run.start_snapshot.scheduler.active_runtime_count, 2);
    assert_eq!(run.end_snapshot.scheduler.active_runtime_count, 0);
    assert_eq!(run.runtime_cycle_count, 4);
    assert_eq!(run.scheduler_runs.len(), 2);

    let mut generated_by_stream = BTreeMap::<String, Vec<u32>>::new();
    for event in &run.output_events {
        generated_by_stream
            .entry(event.stream_id.clone())
            .or_default()
            .push(event.output_event.token_id);
    }

    assert_eq!(
        generated_by_stream.get("shared_stream_a"),
        Some(&vec![1, 1])
    );
    assert_eq!(
        generated_by_stream.get("shared_stream_b"),
        Some(&vec![1, 1])
    );
    let final_snapshot = run.end_snapshot;
    assert_eq!(final_snapshot.models[0].registered_stream_count, 2);
    assert_eq!(final_snapshot.scheduler.registered_runtime_count, 2);
    assert_eq!(final_snapshot.scheduler.active_runtime_count, 0);
    assert!(final_snapshot.scheduler.idle);
    let runtime_a = engine.runtime_snapshot("shared_stream_a").unwrap();
    let runtime_b = engine.runtime_snapshot("shared_stream_b").unwrap();
    assert!(runtime_a.idle);
    assert!(runtime_b.idle);
    assert_eq!(runtime_a.stream.total_public_outputs, 2);
    assert_eq!(runtime_b.stream.total_public_outputs, 2);
    assert_eq!(runtime_a.stream.next_stream_tick, 3);
    assert_eq!(runtime_b.stream.next_stream_tick, 3);
    assert_eq!(runtime_a.pending_input_event_count, 0);
    assert_eq!(runtime_b.pending_input_event_count, 0);
}

