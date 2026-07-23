#[test]
fn resident_greedy_running_stream_interrupt_clears_feedback_without_resetting_state() {
    let device = match VulkanComputeDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping resident running stream interrupt: {error}");
            return;
        }
    };
    let Some(processor) = create_fixture_model_resident_greedy_stream_processor(
        &device,
        "resident running stream interrupt",
    ) else {
        return;
    };
    let mut stream = processor.into_running_stream("stream_0");

    stream.inject_prompt(&[1], 3, None).unwrap();
    let first_tick = stream.tick(&device).unwrap();
    assert_eq!(first_tick.stream_tick, Some(0));
    assert_eq!(
        first_tick.input_signal.as_ref().unwrap().route(),
        VulkanResidentPromptEventInputRoute::ExternalInput
    );
    assert!(first_tick.public_output.is_some());
    assert!(first_tick.private_feedback.is_some());
    assert_eq!(stream.remaining_public_outputs, 2);
    assert_eq!(stream.pending_private_feedback_count(), 1);
    assert_eq!(stream.next_stream_tick, 1);

    let event = stream.interrupt("user_interrupt");
    assert_eq!(
        event.event_type,
        VulkanResidentStreamControlEventType::Interrupt
    );
    assert_eq!(event.reason, "user_interrupt");
    assert_eq!(event.cleared_private_feedback_ids, vec!["feedback_0"]);
    assert_eq!(event.closing_private_feedback_id, None);
    assert!(event.state_preserved);
    assert_eq!(stream.pending_private_feedback_count(), 0);
    assert_eq!(stream.remaining_public_outputs, 0);
    assert!(!stream.loop_open);
    assert_eq!(stream.last_stop_reason.as_deref(), Some("user_interrupt"));

    let idle = stream.tick(&device).unwrap();
    assert_eq!(idle.status, VulkanResidentRunningStreamTickStatus::Idle);
    assert_eq!(idle.stream_tick, None);
    assert_eq!(stream.next_stream_tick, 1);

    let resumed = stream.run_prompt(&device, &[36_309], 1, None).unwrap();
    assert_eq!(resumed.start_stream_tick, 1);
    assert_eq!(resumed.next_stream_tick, 3);
    assert_eq!(resumed.prompt_token_ids, vec![36_309]);
    assert_eq!(resumed.generated_token_ids.len(), 1);
    assert_eq!(stream.next_stream_tick, 3);
    assert_eq!(stream.public_outputs().len(), 2);
    assert_eq!(stream.private_feedback_history().len(), 2);
}

