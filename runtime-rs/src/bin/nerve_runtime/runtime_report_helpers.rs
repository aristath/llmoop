fn elapsed_nanos_u64(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn average_nanos(total_ns: u64, count: usize) -> Option<u64> {
    if count == 0 {
        None
    } else {
        Some(total_ns / u64::try_from(count).unwrap_or(u64::MAX))
    }
}

fn runtime_prompt_timing_report(
    setup_time_ns: u64,
    run_time_ns: u64,
    generated_token_count: usize,
    tick_count: usize,
    scheduler_turn_count: usize,
) -> RuntimePromptTimingReport {
    RuntimePromptTimingReport {
        setup_time_ns,
        run_time_ns,
        total_time_ns: setup_time_ns.saturating_add(run_time_ns),
        generated_token_count,
        tick_count,
        scheduler_turn_count,
        average_generated_token_time_ns: average_nanos(run_time_ns, generated_token_count),
        average_tick_time_ns: average_nanos(run_time_ns, tick_count),
        average_scheduler_turn_time_ns: average_nanos(run_time_ns, scheduler_turn_count),
    }
}

fn tokenizer_options_report(args: &Args) -> RuntimeTokenizerOptionsReport {
    RuntimeTokenizerOptionsReport {
        add_special_tokens: args.add_special_tokens,
        skip_special_tokens: args.skip_special_tokens,
    }
}

fn runtime_patch_report(args: &Args) -> RuntimePatchControls {
    RuntimePatchControls {
        default_device_id: args.default_device_id.clone(),
        pedal_devices: args.pedal_devices.clone(),
        source_chain: args.source_chain.as_ref().map(|source_chain| {
            source_chain
                .iter()
                .map(
                    |(instance_id, source_pedal_id)| RuntimePatchSourceChainEntry {
                        instance_id: instance_id.clone(),
                        source_pedal_id: source_pedal_id.clone(),
                    },
                )
                .collect::<Vec<_>>()
        }),
        duplicate_after: args
            .duplicate_after
            .iter()
            .map(
                |(after_instance_id, new_instance_id)| RuntimePatchDuplicateAfterControl {
                    after_instance_id: after_instance_id.clone(),
                    new_instance_id: new_instance_id.clone(),
                },
            )
            .collect::<Vec<_>>(),
    }
}

fn runtime_device_bindings_report(
    args: &Args,
    logical_device_ids: &[String],
) -> RuntimeDeviceBindings {
    let all_logical_devices_are_explicitly_bound = logical_device_ids
        .iter()
        .all(|device_id| args.device_bindings.contains_key(device_id));
    let default_physical_device_index = if all_logical_devices_are_explicitly_bound {
        args.vulkan_device_index.or_else(|| {
            args.default_device_id
                .as_deref()
                .and_then(|device_id| parse_vulkan_physical_device_ref(device_id).ok().flatten())
        })
    } else {
        runtime_report_default_vulkan_physical_device_index(args)
    };
    RuntimeDeviceBindings::from_vulkan_targets(
        logical_device_ids,
        &args.device_bindings,
        default_physical_device_index,
        resolve_runtime_vulkan_physical_device_ref,
    )
}

fn choose_runtime_context_size(
    package_manifest: &Path,
    requested_context_size: Option<usize>,
    minimum_context_size: usize,
) -> Result<usize, Box<dyn Error>> {
    let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
    Ok(resolve_runtime_context_size(
        manifest.max_context_activations,
        requested_context_size,
        minimum_context_size,
    )?)
}

fn resolve_runtime_context_size(
    max_context_size: usize,
    requested_context_size: Option<usize>,
    minimum_context_size: usize,
) -> io::Result<usize> {
    if max_context_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "compiled package declares a zero maximum context size",
        ));
    }

    let context_size = requested_context_size.unwrap_or(max_context_size);
    if context_size > max_context_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "requested context size {context_size} exceeds the model maximum ({max_context_size})"
            ),
        ));
    }
    if context_size < minimum_context_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "context size {context_size} cannot hold the {minimum_context_size}-token prompt"
            ),
        ));
    }

    Ok(context_size)
}

fn choose_chat_runtime_context_size(
    package_manifest: &Path,
    requested_context_size: Option<usize>,
) -> Result<usize, Box<dyn Error>> {
    choose_runtime_context_size(package_manifest, requested_context_size, 0)
}

