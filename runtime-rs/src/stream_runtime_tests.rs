use super::*;
use crate::stream_prefix_cache::RuntimePrefixStateCacheKey;

fn budget(max_activations: usize) -> RuntimeStreamSchedulerBudget {
    RuntimeStreamSchedulerBudget::new(max_activations, 2, 16)
}

fn state_key() -> TransientStateKey {
    TransientStateKey::new("layer_00", "kv_memory")
}

fn state_shape() -> TransientStateBlockShape {
    TransientStateBlockShape::new(16, 2).unwrap()
}

#[test]
fn scheduler_adds_stream_with_declared_transient_state_atomically() {
    let mut scheduler = RuntimeStreamScheduler::new();

    let snapshot = scheduler
        .add_stream_with_state_declarations("stream_a", [(state_key(), state_shape())])
        .unwrap();

    assert_eq!(snapshot.transient_state_entry_count, 1);
    assert_eq!(
        scheduler
            .stream_transient_state_snapshot("stream_a")
            .unwrap()
            .entries[0]
            .key,
        state_key()
    );
}

#[test]
fn scheduler_chunks_prefill_before_decode_feedback() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler
        .enqueue_input_event(
            "stream_a",
            RuntimeStreamInputEvent::new("event_0", [1, 2, 3], 2),
        )
        .unwrap();

    let first = scheduler.schedule_step(budget(1)).unwrap();
    assert_eq!(
        first.activations[0].kind,
        RuntimeStreamActivationKind::PrefillChunk {
            token_offset: 0,
            token_ids: vec![1, 2],
            remaining_prompt_token_count: 1,
        }
    );
    scheduler
        .complete_activation(
            first.activations[0].id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();

    let second = scheduler.schedule_step(budget(1)).unwrap();
    assert_eq!(
        second.activations[0].kind,
        RuntimeStreamActivationKind::PrefillChunk {
            token_offset: 2,
            token_ids: vec![3],
            remaining_prompt_token_count: 0,
        }
    );
    scheduler
        .complete_activation(
            second.activations[0].id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();

    let third = scheduler.schedule_step(budget(1)).unwrap();
    assert_eq!(
        third.activations[0].kind,
        RuntimeStreamActivationKind::DecodeFeedback {
            feedback_depth: 0,
            max_tokens: 1,
        }
    );
}

#[test]
fn scheduler_prioritizes_running_streams_before_new_waiting_events() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler.add_stream("stream_b").unwrap();
    scheduler
        .enqueue_input_event("stream_a", RuntimeStreamInputEvent::new("event_a", [10], 1))
        .unwrap();
    let prefill = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();

    scheduler
        .enqueue_input_event("stream_b", RuntimeStreamInputEvent::new("event_b", [20], 1))
        .unwrap();
    let next = scheduler.schedule_step(budget(1)).unwrap();
    assert_eq!(next.activations[0].stream_id, "stream_a");
    assert!(matches!(
        next.activations[0].kind,
        RuntimeStreamActivationKind::DecodeFeedback { .. }
    ));
}

#[test]
fn scheduler_prefill_chunk_lands_on_the_next_state_page_boundary() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler
        .add_stream_with_state_declarations(
            "stream_a",
            [(state_key(), TransientStateBlockShape::new(16, 4).unwrap())],
        )
        .unwrap();
    scheduler
        .enqueue_input_event(
            "stream_a",
            RuntimeStreamInputEvent::new("first", [1, 2, 3], 0),
        )
        .unwrap();
    let first = scheduler
        .schedule_step(RuntimeStreamSchedulerBudget::new(1, 64, 64))
        .unwrap()
        .activations[0]
        .clone();
    scheduler
        .complete_activation(first.id, RuntimeStreamActivationOutcome::prefill_complete())
        .unwrap();
    scheduler
        .enqueue_input_event(
            "stream_a",
            RuntimeStreamInputEvent::new("second", [4, 5], 0),
        )
        .unwrap();

    let boundary = scheduler
        .schedule_step(RuntimeStreamSchedulerBudget::new(1, 64, 64))
        .unwrap()
        .activations[0]
        .clone();

    assert_eq!(
        boundary.kind,
        RuntimeStreamActivationKind::PrefillChunk {
            token_offset: 0,
            token_ids: vec![4],
            remaining_prompt_token_count: 1,
        }
    );
    assert_eq!(boundary.state_reservations[0].slots.len(), 1);
    assert_eq!(
        boundary.state_reservations[0].slots[0].block_activation_offset,
        3
    );
}

#[test]
fn scheduler_completes_stream_without_destroying_it() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler
        .enqueue_input_event("stream_a", RuntimeStreamInputEvent::new("event_0", [1], 1))
        .unwrap();
    let prefill = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();
    let decode = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    let done = scheduler
        .complete_activation(
            decode.id,
            RuntimeStreamActivationOutcome::generated(42, true),
        )
        .unwrap();

    assert_eq!(done.status, RuntimeStreamStatus::Idle);
    assert_eq!(done.completed_input_event_count, 1);
    assert_eq!(scheduler.snapshot().stream_count, 1);
    assert_eq!(scheduler.snapshot().active_stream_count, 0);
}

#[test]
fn scheduler_interrupt_clears_work_and_keeps_stream_registered() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler
        .enqueue_input_event(
            "stream_a",
            RuntimeStreamInputEvent::new("event_0", [1, 2], 8),
        )
        .unwrap();
    let scheduled = scheduler.schedule_step(budget(1)).unwrap();
    assert_eq!(scheduled.activations.len(), 1);

    let interrupted = scheduler
        .interrupt_stream("stream_a", "user requested stop")
        .unwrap();
    assert_eq!(interrupted.status, RuntimeStreamStatus::Interrupted);
    assert_eq!(interrupted.in_flight_activation_count, 0);
    assert_eq!(scheduler.snapshot().stream_count, 1);
    assert_eq!(scheduler.snapshot().in_flight_activation_count, 0);
}

#[test]
fn scheduler_reserves_transient_state_slots_for_prefill_and_decode() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler
        .declare_stream_state("stream_a", state_key(), state_shape())
        .unwrap();
    scheduler
        .enqueue_input_event(
            "stream_a",
            RuntimeStreamInputEvent::new("event_0", [1, 2, 3], 1),
        )
        .unwrap();

    let first = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    assert_eq!(first.kind.work_units(), 2);
    assert_eq!(first.state_reservations.len(), 1);
    assert_eq!(first.state_reservations[0].slots.len(), 2);
    assert_eq!(
        first.state_reservations[0].slots[0].block_activation_offset,
        0
    );
    assert_eq!(
        first.state_reservations[0].slots[1].block_activation_offset,
        1
    );
    assert_eq!(
        scheduler.snapshot().transient_state_arena.live_block_count,
        1
    );
    scheduler
        .complete_activation(first.id, RuntimeStreamActivationOutcome::prefill_complete())
        .unwrap();

    let second = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    assert_eq!(second.kind.work_units(), 1);
    assert_eq!(second.state_reservations[0].slots.len(), 2);
    assert_eq!(
        second.state_reservations[0].slots[0].block_activation_offset,
        0
    );
    assert_eq!(
        scheduler.snapshot().transient_state_arena.live_block_count,
        2
    );
    scheduler
        .complete_activation(
            second.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();

    let decode = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    assert!(matches!(
        decode.kind,
        RuntimeStreamActivationKind::DecodeFeedback { .. }
    ));
    assert_eq!(decode.state_reservations[0].slots.len(), 2);
    assert_eq!(
        decode.state_reservations[0].slots[0].block_activation_offset,
        1
    );
}

#[test]
fn scheduler_interrupt_rolls_back_unexecuted_reservations_and_preserves_committed_state() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler
        .declare_stream_state("stream_a", state_key(), state_shape())
        .unwrap();
    scheduler
        .enqueue_input_event(
            "stream_a",
            RuntimeStreamInputEvent::new("event_0", [1, 2], 8),
        )
        .unwrap();
    let prefill = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();
    let scheduled = scheduler
        .schedule_step(
            RuntimeStreamSchedulerBudget::new(1, 2, 16).with_max_decode_tokens_per_activation(4),
        )
        .unwrap();
    assert_eq!(
        scheduled.activations[0].state_reservations[0].slots.len(),
        5
    );
    assert_eq!(
        scheduler.snapshot().transient_state_arena.live_block_count,
        4
    );

    let interrupted = scheduler.interrupt_stream("stream_a", "cancel").unwrap();

    assert_eq!(interrupted.transient_state_block_count, 1);
    assert_eq!(interrupted.transient_state_logical_activation_count, 2);
    assert_eq!(
        scheduler.snapshot().transient_state_arena.live_block_count,
        1
    );
    scheduler.reset_stream_transient_state("stream_a").unwrap();
    assert_eq!(
        scheduler.snapshot().transient_state_arena.live_block_count,
        0
    );
    assert_eq!(
        scheduler.snapshot().transient_state_arena.free_block_count,
        4
    );
}

#[test]
fn scheduler_stream_removal_reclaims_all_transient_blocks() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler
        .add_stream_with_state_declarations("stream_a", [(state_key(), state_shape())])
        .unwrap();
    scheduler
        .enqueue_input_event("stream_a", RuntimeStreamInputEvent::new("event_0", [1], 0))
        .unwrap();
    let activation = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    scheduler
        .complete_activation(
            activation.id,
            RuntimeStreamActivationOutcome::generated_tokens([], false),
        )
        .unwrap();
    assert_eq!(
        scheduler.snapshot().transient_state_arena.live_block_count,
        1
    );

    scheduler.remove_stream("stream_a").unwrap();

    assert_eq!(scheduler.snapshot().stream_count, 0);
    assert_eq!(
        scheduler.snapshot().transient_state_arena.live_block_count,
        0
    );
    assert_eq!(
        scheduler.snapshot().transient_state_arena.free_block_count,
        1
    );
}

#[test]
fn scheduler_forks_stream_transient_state_without_copying_blocks() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler
        .add_stream_with_state_declarations("source", [(state_key(), state_shape())])
        .unwrap();
    scheduler
        .enqueue_input_event("source", RuntimeStreamInputEvent::new("event_0", [1, 2], 8))
        .unwrap();
    let prefill = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();

    let forked = scheduler
        .fork_stream_transient_state("source", "child", "same_package")
        .unwrap();

    assert_eq!(forked.stream_id, "child");
    assert_eq!(forked.execution_class_id, "same_package");
    assert_eq!(forked.status, RuntimeStreamStatus::Idle);
    assert_eq!(forked.queued_input_event_count, 0);
    assert_eq!(forked.current_input_event_id, None);
    assert_eq!(forked.transient_state_entry_count, 1);
    assert_eq!(forked.transient_state_block_count, 1);
    assert_eq!(forked.transient_state_logical_activation_count, 2);
    let arena = scheduler.transient_state_arena_snapshot().unwrap();
    assert_eq!(arena.live_block_count, 1);
    assert_eq!(arena.blocks[0].ref_count, 2);

    scheduler
        .interrupt_stream("source", "discard source")
        .unwrap();

    let child_state = scheduler.stream_transient_state_snapshot("child").unwrap();
    assert_eq!(child_state.block_count, 1);
    assert_eq!(child_state.logical_activation_count, 2);
    let arena_after_source_interrupt = scheduler.transient_state_arena_snapshot().unwrap();
    assert_eq!(arena_after_source_interrupt.live_block_count, 1);
    assert_eq!(arena_after_source_interrupt.free_block_count, 1);
    assert_eq!(arena_after_source_interrupt.blocks[0].ref_count, 2);
}

#[test]
fn scheduler_rejects_invalid_fork_before_retaining_blocks() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler
        .add_stream_with_state_declarations("source", [(state_key(), state_shape())])
        .unwrap();
    scheduler
        .enqueue_input_event("source", RuntimeStreamInputEvent::new("event_0", [1, 2], 8))
        .unwrap();
    let prefill = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();

    let before = scheduler.transient_state_arena_snapshot().unwrap();
    let error = scheduler
        .fork_stream_transient_state("source", "child", "")
        .unwrap_err();
    let after = scheduler.transient_state_arena_snapshot().unwrap();

    assert!(
        error
            .0
            .contains("forked stream execution class id must not be empty")
    );
    assert_eq!(after, before);
    assert_eq!(scheduler.snapshot().stream_count, 1);
}

#[test]
fn scheduler_shares_single_component_state_between_streams() {
    let kv = state_key();
    let conv = TransientStateKey::new("layer_00", "conv_state");
    let shape = state_shape();
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler
        .add_stream_with_state_declarations(
            "source",
            [(kv.clone(), shape.clone()), (conv.clone(), shape.clone())],
        )
        .unwrap();
    scheduler
        .add_stream_with_state_declarations("target", [(kv.clone(), shape.clone())])
        .unwrap();
    scheduler
        .enqueue_input_event("source", RuntimeStreamInputEvent::new("event_0", [1, 2], 8))
        .unwrap();
    let prefill = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();

    let shared = scheduler
        .share_stream_state("target", "source", &kv)
        .unwrap();

    assert_eq!(shared.stream_id, "target");
    assert_eq!(shared.status, RuntimeStreamStatus::Idle);
    assert_eq!(shared.transient_state_entry_count, 1);
    assert_eq!(shared.transient_state_block_count, 1);
    assert_eq!(shared.transient_state_logical_activation_count, 2);
    let source_state = scheduler.stream_transient_state_snapshot("source").unwrap();
    let target_state = scheduler.stream_transient_state_snapshot("target").unwrap();
    let source_kv_blocks = source_state
        .entries
        .iter()
        .find(|entry| entry.key == kv)
        .unwrap()
        .block_ids
        .clone();
    let source_conv_blocks = source_state
        .entries
        .iter()
        .find(|entry| entry.key == conv)
        .unwrap()
        .block_ids
        .clone();
    let target_kv_blocks = target_state.entries[0].block_ids.clone();
    assert_eq!(target_kv_blocks, source_kv_blocks);
    assert_ne!(target_kv_blocks, source_conv_blocks);

    let arena = scheduler.transient_state_arena_snapshot().unwrap();
    assert_eq!(arena.live_block_count, 2);
    let kv_ref_count = arena
        .blocks
        .iter()
        .find(|block| block.block_id == target_kv_blocks[0])
        .unwrap()
        .ref_count;
    let conv_ref_count = arena
        .blocks
        .iter()
        .find(|block| block.block_id == source_conv_blocks[0])
        .unwrap()
        .ref_count;
    assert_eq!(kv_ref_count, 2);
    assert_eq!(conv_ref_count, 1);
}

#[test]
fn scheduler_prefix_cache_restores_state_after_source_reset() {
    let prefix_key = RuntimePrefixStateCacheKey::from_token_prefix(
        "package_a",
        "graph_a",
        &[1, 2],
        b"reasoning=true",
        [state_key()],
    )
    .unwrap();
    let mut scheduler = RuntimeStreamScheduler::with_prefix_state_cache_capacity(2);
    scheduler
        .add_stream_with_state_declarations_and_execution_class(
            "source",
            "package_a",
            [(state_key(), state_shape())],
        )
        .unwrap();
    scheduler
        .enqueue_input_event("source", RuntimeStreamInputEvent::new("event_0", [1, 2], 8))
        .unwrap();
    let prefill = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();

    scheduler
        .cache_stream_prefix_state("source", prefix_key.clone())
        .unwrap();
    let cached_block = scheduler
        .stream_transient_state_snapshot("source")
        .unwrap()
        .entries[0]
        .block_ids[0];
    assert_eq!(
        scheduler
            .transient_state_arena_snapshot()
            .unwrap()
            .blocks
            .iter()
            .find(|block| block.block_id == cached_block)
            .unwrap()
            .ref_count,
        2
    );

    scheduler.interrupt_stream("source", "drop source").unwrap();
    scheduler.reset_stream_transient_state("source").unwrap();
    scheduler
        .add_stream_with_state_declarations_and_execution_class(
            "target",
            "package_a",
            [(state_key(), state_shape())],
        )
        .unwrap();
    let restored = scheduler
        .restore_stream_prefix_state("target", &prefix_key)
        .unwrap();

    assert!(restored);
    assert_eq!(
        scheduler
            .stream_transient_state_snapshot("target")
            .unwrap()
            .entries[0]
            .block_ids,
        vec![cached_block]
    );
    let cache_snapshot = scheduler.prefix_state_cache_snapshot();
    assert_eq!(cache_snapshot.entry_count, 1);
    assert_eq!(cache_snapshot.entries[0].use_count, 1);
    assert_eq!(
        scheduler
            .transient_state_arena_snapshot()
            .unwrap()
            .blocks
            .iter()
            .find(|block| block.block_id == cached_block)
            .unwrap()
            .ref_count,
        2
    );
}

#[test]
fn scheduler_prefix_cache_rejects_execution_class_mismatch_without_mutation() {
    let prefix_key = RuntimePrefixStateCacheKey::from_token_prefix(
        "package_a",
        "graph_a",
        &[1, 2],
        b"reasoning=true",
        [state_key()],
    )
    .unwrap();
    let mut scheduler = RuntimeStreamScheduler::with_prefix_state_cache_capacity(2);
    scheduler
        .add_stream_with_state_declarations_and_execution_class(
            "source",
            "package_a",
            [(state_key(), state_shape())],
        )
        .unwrap();
    scheduler
        .add_stream_with_state_declarations_and_execution_class(
            "target",
            "package_b",
            [(state_key(), state_shape())],
        )
        .unwrap();
    scheduler
        .enqueue_input_event("source", RuntimeStreamInputEvent::new("event_0", [1, 2], 8))
        .unwrap();
    let prefill = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();
    scheduler
        .cache_stream_prefix_state("source", prefix_key.clone())
        .unwrap();
    let before_target = scheduler.stream_transient_state_snapshot("target").unwrap();

    let error = scheduler
        .restore_stream_prefix_state("target", &prefix_key)
        .unwrap_err();

    assert!(
        error
            .0
            .contains("cannot restore prefix key execution class")
    );
    assert_eq!(
        scheduler.stream_transient_state_snapshot("target").unwrap(),
        before_target
    );
    assert_eq!(
        scheduler.prefix_state_cache_snapshot().entries[0].use_count,
        0
    );
}

#[test]
fn scheduler_restores_longest_matching_prefix_state() {
    let mut scheduler = RuntimeStreamScheduler::with_prefix_state_cache_capacity(4);
    scheduler
        .add_stream_with_state_declarations_and_execution_class(
            "short_source",
            "package_a",
            [(state_key(), state_shape())],
        )
        .unwrap();
    scheduler
        .add_stream_with_state_declarations_and_execution_class(
            "long_source",
            "package_a",
            [(state_key(), state_shape())],
        )
        .unwrap();

    scheduler
        .enqueue_input_event(
            "short_source",
            RuntimeStreamInputEvent::new("short_event", [1, 2], 0),
        )
        .unwrap();
    let short_prefill = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    scheduler
        .complete_activation(
            short_prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();

    scheduler
        .enqueue_input_event(
            "long_source",
            RuntimeStreamInputEvent::new("long_event", [1, 2, 3, 4], 0),
        )
        .unwrap();
    for _ in 0..2 {
        let prefill = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
        scheduler
            .complete_activation(
                prefill.id,
                RuntimeStreamActivationOutcome::prefill_complete(),
            )
            .unwrap();
    }

    let short_key = scheduler
        .cache_stream_prefix_state_for_tokens(
            "short_source",
            "graph_a",
            &[1, 2],
            b"reasoning=true",
            [state_key()],
        )
        .unwrap();
    let long_key = scheduler
        .cache_stream_prefix_state_for_tokens(
            "long_source",
            "graph_a",
            &[1, 2, 3, 4],
            b"reasoning=true",
            [state_key()],
        )
        .unwrap();
    scheduler
        .add_stream_with_state_declarations_and_execution_class(
            "target",
            "package_a",
            [(state_key(), state_shape())],
        )
        .unwrap();

    let matched = scheduler
        .restore_longest_stream_prefix_state(
            "target",
            "graph_a",
            &[1, 2, 3, 4, 5],
            b"reasoning=true",
            [state_key()],
        )
        .unwrap()
        .unwrap();

    assert_eq!(matched, long_key);
    assert_ne!(matched, short_key);
    assert_eq!(
        scheduler
            .stream_transient_state_snapshot("target")
            .unwrap()
            .logical_activation_count,
        4
    );
    assert_eq!(
        scheduler
            .prefix_state_cache_snapshot()
            .entries
            .iter()
            .find(|entry| entry.key == long_key)
            .unwrap()
            .use_count,
        1
    );
}

#[test]
fn scheduler_run_drives_executor_until_stream_is_idle() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler
        .enqueue_input_event(
            "stream_a",
            RuntimeStreamInputEvent::new("event_0", [1, 2, 3], 2),
        )
        .unwrap();

    let mut decode_tokens = vec![200, 201].into_iter();
    let run = scheduler
        .run_until_idle_with(budget(1), 16, |activation| match activation.kind {
            RuntimeStreamActivationKind::PrefillChunk { .. } => {
                Ok(RuntimeStreamActivationOutcome::prefill_complete())
            }
            RuntimeStreamActivationKind::DecodeFeedback { .. } => Ok(
                RuntimeStreamActivationOutcome::generated(decode_tokens.next().unwrap(), true),
            ),
        })
        .unwrap();

    assert_eq!(
        run.stop_condition,
        RuntimeStreamSchedulerRunStopCondition::Idle
    );
    assert_eq!(run.completed_activations.len(), 4);
    assert_eq!(run.final_snapshot.active_stream_count, 0);
    assert_eq!(run.final_snapshot.streams[0].completed_input_event_count, 1);
    assert_eq!(run.final_snapshot.streams[0].generated_token_count, 2);
    assert_eq!(
        run.completed_activations
            .iter()
            .filter(|completed| matches!(
                completed.activation.kind,
                RuntimeStreamActivationKind::PrefillChunk { .. }
            ))
            .count(),
        2
    );
}

#[test]
fn scheduler_batch_step_groups_compatible_decode_windows_across_streams() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler.add_stream("stream_b").unwrap();
    scheduler
        .declare_stream_state("stream_a", state_key(), state_shape())
        .unwrap();
    scheduler
        .declare_stream_state("stream_b", state_key(), state_shape())
        .unwrap();
    scheduler
        .enqueue_input_event("stream_a", RuntimeStreamInputEvent::new("event_a", [10], 4))
        .unwrap();
    scheduler
        .enqueue_input_event("stream_b", RuntimeStreamInputEvent::new("event_b", [20], 4))
        .unwrap();

    let prefill = scheduler.schedule_batch_step(budget(2)).unwrap();
    assert_eq!(prefill.batches.len(), 1);
    assert_eq!(
        prefill.batches[0].kind,
        RuntimeStreamActivationBatchKind::PrefillChunk {
            execution_class_id: "default".to_string(),
            token_count: 1
        }
    );
    for activation in &prefill.batches[0].activations {
        scheduler
            .complete_activation(
                activation.id,
                RuntimeStreamActivationOutcome::prefill_complete(),
            )
            .unwrap();
    }

    let decode_budget =
        RuntimeStreamSchedulerBudget::new(2, 2, 16).with_max_decode_tokens_per_activation(4);
    let decode = scheduler.schedule_batch_step(decode_budget).unwrap();

    assert_eq!(decode.batches.len(), 1);
    assert_eq!(
        decode.batches[0].kind,
        RuntimeStreamActivationBatchKind::DecodeFeedback {
            execution_class_id: "default".to_string(),
            max_tokens: 4
        }
    );
    let stream_ids = decode.batches[0]
        .activations
        .iter()
        .map(|activation| activation.stream_id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(stream_ids, vec!["stream_a", "stream_b"]);
    assert_ne!(
        decode.batches[0].activations[0].state_reservations[0].slots[0].block_id,
        decode.batches[0].activations[1].state_reservations[0].slots[0].block_id
    );
}

#[test]
fn scheduler_batch_step_keeps_incompatible_prefill_shapes_separate() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler.add_stream("stream_b").unwrap();
    scheduler
        .enqueue_input_event(
            "stream_a",
            RuntimeStreamInputEvent::new("event_a", [1, 2], 1),
        )
        .unwrap();
    scheduler
        .enqueue_input_event("stream_b", RuntimeStreamInputEvent::new("event_b", [3], 1))
        .unwrap();

    let step = scheduler.schedule_batch_step(budget(2)).unwrap();

    assert_eq!(step.batches.len(), 2);
    assert_eq!(
        step.batches[0].kind,
        RuntimeStreamActivationBatchKind::PrefillChunk {
            execution_class_id: "default".to_string(),
            token_count: 2
        }
    );
    assert_eq!(
        step.batches[1].kind,
        RuntimeStreamActivationBatchKind::PrefillChunk {
            execution_class_id: "default".to_string(),
            token_count: 1
        }
    );
    assert_eq!(step.batches[0].activations.len(), 1);
    assert_eq!(step.batches[1].activations.len(), 1);
}

#[test]
fn scheduler_batch_step_keeps_different_execution_classes_separate() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler
        .add_stream_with_execution_class("stream_a", "package_a")
        .unwrap();
    scheduler
        .add_stream_with_execution_class("stream_b", "package_b")
        .unwrap();
    scheduler
        .enqueue_input_event("stream_a", RuntimeStreamInputEvent::new("event_a", [1], 1))
        .unwrap();
    scheduler
        .enqueue_input_event("stream_b", RuntimeStreamInputEvent::new("event_b", [2], 1))
        .unwrap();

    let step = scheduler.schedule_batch_step(budget(2)).unwrap();

    assert_eq!(step.batches.len(), 2);
    assert_eq!(
        step.batches[0].kind,
        RuntimeStreamActivationBatchKind::PrefillChunk {
            execution_class_id: "package_a".to_string(),
            token_count: 1,
        }
    );
    assert_eq!(
        step.batches[1].kind,
        RuntimeStreamActivationBatchKind::PrefillChunk {
            execution_class_id: "package_b".to_string(),
            token_count: 1,
        }
    );
}

#[test]
fn scheduler_batch_step_admits_waiting_prefill_alongside_running_prefill_when_capacity_exists() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("long_stream").unwrap();
    scheduler.add_stream("new_stream").unwrap();
    scheduler
        .enqueue_input_event(
            "long_stream",
            RuntimeStreamInputEvent::new("long_event", [1, 2, 3, 4, 5], 1),
        )
        .unwrap();

    let first = scheduler.schedule_batch_step(budget(1)).unwrap();
    assert_eq!(first.batches.len(), 1);
    assert_eq!(
        first.batches[0].activations[0].kind,
        RuntimeStreamActivationKind::PrefillChunk {
            token_offset: 0,
            token_ids: vec![1, 2],
            remaining_prompt_token_count: 3,
        }
    );
    scheduler
        .complete_activation(
            first.batches[0].activations[0].id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();

    scheduler
        .enqueue_input_event(
            "new_stream",
            RuntimeStreamInputEvent::new("new_event", [9, 10], 1),
        )
        .unwrap();

    let next = scheduler.schedule_batch_step(budget(2)).unwrap();

    assert_eq!(next.batches.len(), 1);
    assert_eq!(
        next.batches[0].kind,
        RuntimeStreamActivationBatchKind::PrefillChunk {
            execution_class_id: "default".to_string(),
            token_count: 2,
        }
    );
    let scheduled = next.batches[0]
        .activations
        .iter()
        .map(|activation| {
            (
                activation.stream_id.as_str(),
                activation.kind.clone(),
                activation.state_reservations.len(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        scheduled,
        vec![
            (
                "long_stream",
                RuntimeStreamActivationKind::PrefillChunk {
                    token_offset: 2,
                    token_ids: vec![3, 4],
                    remaining_prompt_token_count: 1,
                },
                0,
            ),
            (
                "new_stream",
                RuntimeStreamActivationKind::PrefillChunk {
                    token_offset: 0,
                    token_ids: vec![9, 10],
                    remaining_prompt_token_count: 0,
                },
                0,
            ),
        ]
    );
}

#[test]
fn scheduler_batch_run_requires_exact_outcomes_for_every_activation() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler.add_stream("stream_b").unwrap();
    scheduler
        .enqueue_input_event("stream_a", RuntimeStreamInputEvent::new("event_a", [1], 1))
        .unwrap();
    scheduler
        .enqueue_input_event("stream_b", RuntimeStreamInputEvent::new("event_b", [2], 1))
        .unwrap();

    let error = scheduler
        .run_batches_until_idle_with(budget(2), 1, |batch| {
            Ok(vec![RuntimeStreamBatchActivationOutcome {
                activation_id: batch.activations[0].id,
                outcome: RuntimeStreamActivationOutcome::prefill_complete(),
            }])
        })
        .unwrap_err();

    assert!(
        error
            .0
            .contains("batch executor did not return an outcome for activation")
    );
}

#[test]
fn scheduler_batch_run_drives_multiple_streams_until_idle() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler.add_stream("stream_b").unwrap();
    scheduler
        .enqueue_input_event("stream_a", RuntimeStreamInputEvent::new("event_a", [1], 2))
        .unwrap();
    scheduler
        .enqueue_input_event("stream_b", RuntimeStreamInputEvent::new("event_b", [2], 2))
        .unwrap();

    let run = scheduler
        .run_batches_until_idle_with(
            RuntimeStreamSchedulerBudget::new(2, 2, 16).with_max_decode_tokens_per_activation(2),
            8,
            |batch| {
                Ok(batch
                    .activations
                    .iter()
                    .map(|activation| {
                        let outcome = match activation.kind {
                            RuntimeStreamActivationKind::PrefillChunk { .. } => {
                                RuntimeStreamActivationOutcome::prefill_complete()
                            }
                            RuntimeStreamActivationKind::DecodeFeedback { .. } => {
                                RuntimeStreamActivationOutcome::generated_tokens(
                                    [activation.id as u32, activation.id as u32 + 100],
                                    false,
                                )
                            }
                        };
                        RuntimeStreamBatchActivationOutcome {
                            activation_id: activation.id,
                            outcome,
                        }
                    })
                    .collect())
            },
        )
        .unwrap();

    assert_eq!(
        run.stop_condition,
        RuntimeStreamSchedulerRunStopCondition::Idle
    );
    assert_eq!(run.final_snapshot.active_stream_count, 0);
    assert_eq!(run.completed_activations.len(), 4);
    assert!(
        run.final_snapshot
            .streams
            .iter()
            .all(|stream| stream.completed_input_event_count == 1)
    );
    assert_eq!(
        run.final_snapshot
            .streams
            .iter()
            .map(|stream| stream.generated_token_count)
            .sum::<usize>(),
        4
    );
}

#[test]
fn scheduler_decode_activation_can_cover_a_feedback_window() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler
        .enqueue_input_event("stream_a", RuntimeStreamInputEvent::new("event_0", [1], 5))
        .unwrap();
    let windowed_budget =
        RuntimeStreamSchedulerBudget::new(1, 2, 16).with_max_decode_tokens_per_activation(4);

    let prefill = scheduler
        .schedule_step(windowed_budget.clone())
        .unwrap()
        .activations[0]
        .clone();
    scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();

    let decode = scheduler
        .schedule_step(windowed_budget)
        .unwrap()
        .activations[0]
        .clone();

    assert_eq!(
        decode.kind,
        RuntimeStreamActivationKind::DecodeFeedback {
            feedback_depth: 0,
            max_tokens: 4,
        }
    );
    let done = scheduler
        .complete_activation(
            decode.id,
            RuntimeStreamActivationOutcome::generated_tokens([10, 11, 12, 13], true),
        )
        .unwrap();
    assert_eq!(done.generated_token_count, 4);
    assert_eq!(done.status, RuntimeStreamStatus::Active);

    let final_decode = scheduler
        .schedule_step(
            RuntimeStreamSchedulerBudget::new(1, 2, 16).with_max_decode_tokens_per_activation(4),
        )
        .unwrap()
        .activations[0]
        .clone();
    assert_eq!(
        final_decode.kind,
        RuntimeStreamActivationKind::DecodeFeedback {
            feedback_depth: 4,
            max_tokens: 1,
        }
    );
}

#[test]
fn scheduler_commits_only_the_feedback_ticks_that_actually_executed() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler
        .add_stream_with_state_declarations("stream_a", [(state_key(), state_shape())])
        .unwrap();
    scheduler
        .enqueue_input_event("stream_a", RuntimeStreamInputEvent::new("event_0", [1], 5))
        .unwrap();
    let windowed_budget =
        RuntimeStreamSchedulerBudget::new(1, 2, 16).with_max_decode_tokens_per_activation(4);

    let prefill = scheduler
        .schedule_step(windowed_budget.clone())
        .unwrap()
        .activations[0]
        .clone();
    scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete(),
        )
        .unwrap();
    let decode = scheduler
        .schedule_step(windowed_budget)
        .unwrap()
        .activations[0]
        .clone();
    assert_eq!(
        scheduler
            .stream_transient_state_snapshot("stream_a")
            .unwrap()
            .logical_activation_count,
        6
    );

    scheduler
        .complete_activation(
            decode.id,
            RuntimeStreamActivationOutcome::generated_tokens([10, 11], false)
                .with_processed_state_activations(2),
        )
        .unwrap();

    let state = scheduler
        .stream_transient_state_snapshot("stream_a")
        .unwrap();
    assert_eq!(state.logical_activation_count, 3);
    assert_eq!(state.block_count, 2);
    let arena = scheduler.transient_state_arena_snapshot().unwrap();
    assert_eq!(arena.live_block_count, 2);
    assert_eq!(arena.free_block_count, 1);
}

#[test]
fn scheduler_rejects_a_processed_tick_count_larger_than_the_reservation() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler
        .add_stream_with_state_declarations("stream_a", [(state_key(), state_shape())])
        .unwrap();
    scheduler
        .enqueue_input_event("stream_a", RuntimeStreamInputEvent::new("event_0", [1], 5))
        .unwrap();
    let prefill = scheduler.schedule_step(budget(1)).unwrap().activations[0].clone();
    let reserved_state = scheduler
        .stream_transient_state_snapshot("stream_a")
        .unwrap();

    let error = scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::prefill_complete().with_processed_state_activations(3),
        )
        .unwrap_err();

    assert!(error.to_string().contains("exceeding its reservation"));
    assert_eq!(
        scheduler
            .stream_transient_state_snapshot("stream_a")
            .unwrap(),
        reserved_state
    );
    assert_eq!(scheduler.snapshot().in_flight_activation_count, 1);
}

#[test]
fn scheduler_prefill_completion_can_emit_initial_feedback_token() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler
        .enqueue_input_event(
            "stream_a",
            RuntimeStreamInputEvent::new("event_0", [1, 2], 3),
        )
        .unwrap();

    let prefill = scheduler
        .schedule_step(RuntimeStreamSchedulerBudget::new(1, 8, 8))
        .unwrap()
        .activations[0]
        .clone();
    assert_eq!(
        prefill.kind,
        RuntimeStreamActivationKind::PrefillChunk {
            token_offset: 0,
            token_ids: vec![1, 2],
            remaining_prompt_token_count: 0,
        }
    );
    let active = scheduler
        .complete_activation(
            prefill.id,
            RuntimeStreamActivationOutcome::generated(99, true),
        )
        .unwrap();
    assert_eq!(active.generated_token_count, 1);

    let decode = scheduler
        .schedule_step(
            RuntimeStreamSchedulerBudget::new(1, 8, 8).with_max_decode_tokens_per_activation(8),
        )
        .unwrap()
        .activations[0]
        .clone();
    assert_eq!(
        decode.kind,
        RuntimeStreamActivationKind::DecodeFeedback {
            feedback_depth: 1,
            max_tokens: 2,
        }
    );
}

#[test]
fn scheduler_run_preserves_in_flight_work_when_step_budget_expires() {
    let mut scheduler = RuntimeStreamScheduler::new();
    scheduler.add_stream("stream_a").unwrap();
    scheduler
        .enqueue_input_event("stream_a", RuntimeStreamInputEvent::new("event_0", [1], 1))
        .unwrap();

    let run = scheduler
        .run_until_idle_with(budget(1), 1, |activation| match activation.kind {
            RuntimeStreamActivationKind::PrefillChunk { .. } => {
                Ok(RuntimeStreamActivationOutcome::prefill_complete())
            }
            RuntimeStreamActivationKind::DecodeFeedback { .. } => {
                Ok(RuntimeStreamActivationOutcome::generated(42, true))
            }
        })
        .unwrap();

    assert_eq!(
        run.stop_condition,
        RuntimeStreamSchedulerRunStopCondition::StepBudget
    );
    assert_eq!(run.completed_activations.len(), 1);
    assert_eq!(run.final_snapshot.active_stream_count, 1);
    assert_eq!(run.final_snapshot.streams[0].completed_input_event_count, 0);
}
