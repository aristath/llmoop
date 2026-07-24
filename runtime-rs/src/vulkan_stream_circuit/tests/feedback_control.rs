#[test]
fn feedback_control_words_encode_stop_tokens_and_dispatches_without_token_caps() {
    let vocabulary_size = 65usize;
    let stop_mask_word_count = vocabulary_size.div_ceil(u32::BITS as usize);
    let dispatch_word_offset =
        VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT + stop_mask_word_count;
    let dimensions = [[11, 2, 1], [7, 1, 1]];

    let words = resident_feedback_control_words(
        vocabulary_size,
        stop_mask_word_count,
        dispatch_word_offset,
        &dimensions,
        1,
        1,
        64,
        &[0, 31, 32, 64],
    )
    .unwrap();

    assert_eq!(
        &words[..VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT],
        &[
            VULKAN_FEEDBACK_CONTROL_ENABLED,
            0,
            VULKAN_FEEDBACK_STOP_REASON_NONE,
            64,
            dispatch_word_offset as u32,
            2,
            1,
            1,
            0,
            0,
            0,
            0,
        ]
    );
    assert_eq!(
        &words[VULKAN_FEEDBACK_CONTROL_HEADER_WORD_COUNT..dispatch_word_offset],
        &[0x8000_0001, 1, 1]
    );
    assert_eq!(&words[dispatch_word_offset..], &[11, 2, 1, 7, 1, 1]);

    let error = resident_feedback_control_words(
        vocabulary_size,
        stop_mask_word_count,
        dispatch_word_offset,
        &dimensions,
        1,
        1,
        64,
        &[65],
    )
    .unwrap_err();
    assert_eq!(
        error,
        VulkanError("stop token id 65 exceeds vocabulary size 65".to_string())
    );
}

#[test]
fn feedback_execution_stats_distinguish_committed_work_from_predicated_tail() {
    let mut stats = VulkanResidentFeedbackExecutionStats::default();
    stats.record_window(64, 7, 6, false);
    stats.record_window(64, 64, 64, true);

    assert_eq!(
        stats,
        VulkanResidentFeedbackExecutionStats {
            window_count: 2,
            planned_tick_count: 128,
            submitted_tick_count: 128,
            executed_tick_count: 71,
            retained_tick_count: 71,
            sampled_tick_count: 70,
            discarded_tick_count: 57,
            template_record_count: 1,
            template_replay_count: 1,
            asynchronous_submission_count: 0,
            completion_poll_count: 0,
            bounded_wait_count: 0,
            bounded_wait_timeout_count: 0,
        }
    );
}

#[test]
fn feedback_cancellation_handle_can_cross_the_runtime_worker_boundary() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VulkanResidentFeedbackCancellationHandle>();
}

#[test]
fn feedback_window_policy_learns_a_responsive_execution_width() {
    let policy = VulkanResidentFeedbackWindowPolicy::new(64);
    assert_eq!(policy.next_tick_count(), 2);

    policy.observe_completed_window(2, 2, 100_000_000, false);
    assert_eq!(policy.next_tick_count(), 5);

    policy.observe_completed_window(5, 5, 500_000_000, false);
    assert_eq!(policy.next_tick_count(), 4);

    policy.observe_completed_window(4, 2, 1, true);
    assert_eq!(policy.next_tick_count(), 4);
}
