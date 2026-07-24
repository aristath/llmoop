fn parse_args() -> Result<Args, String> {
    parse_args_from(std::env::args().skip(1))
}

fn parse_args_from(raw: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut raw = raw.into_iter();

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--package" => {
                parsed.package_manifest = Some(PathBuf::from(next_value(&mut raw, &arg)?));
            }
            "--prompt" => {
                parsed.prompt = Some(next_value(&mut raw, "--prompt")?);
            }
            "--chat" => {
                parsed.chat = true;
            }
            "--inspect-runtime" => {
                parsed.inspect_runtime = true;
            }
            "--inspect-package" => {
                parsed.inspect_package = true;
            }
            "--inspect-graph" => {
                parsed.inspect_graph = true;
            }
            "--inspect-placement" => {
                parsed.inspect_placement = true;
            }
            "--inspect-device-slice" => {
                parsed.inspect_device_slice = Some(next_value(&mut raw, "--inspect-device-slice")?);
            }
            "--inspect-devices" => {
                parsed.inspect_devices = true;
            }
            "--device" => {
                let device_id = next_value(&mut raw, &arg)?;
                if parsed.default_device_id.replace(device_id).is_some() {
                    return Err("--device may only be supplied once".to_string());
                }
            }
            "--place-node" => {
                let assignment = next_value(&mut raw, &arg)?;
                let (component_id, device_id) = parse_node_device_assignment(&assignment)?;
                if parsed
                    .node_devices
                    .insert(component_id.clone(), device_id)
                    .is_some()
                {
                    return Err(format!(
                        "duplicate runtime placement for node {component_id:?}"
                    ));
                }
            }
            "--bind-device" => {
                let assignment = next_value(&mut raw, &arg)?;
                let (device_id, target) = parse_device_binding_assignment(&assignment)?;
                if parsed
                    .device_bindings
                    .insert(device_id.clone(), target)
                    .is_some()
                {
                    return Err(format!(
                        "duplicate runtime device binding for logical device {device_id:?}"
                    ));
                }
            }
            "--duplicate-after" => {
                let assignment = next_value(&mut raw, "--duplicate-after")?;
                parsed
                    .duplicate_after
                    .push(parse_duplicate_after_assignment(&assignment)?);
            }
            "--chain" => {
                let chain = parse_source_chain(&next_value(&mut raw, &arg)?)?;
                if parsed.source_chain.replace(chain).is_some() {
                    return Err("--chain may only be supplied once".to_string());
                }
            }
            "--chat-template-var" => {
                let assignment = next_value(&mut raw, &arg)?;
                let (name, value) = parse_chat_template_variable(&assignment)?;
                if parsed
                    .chat_template_variables
                    .insert(name.clone(), value)
                    .is_some()
                {
                    return Err(format!("duplicate chat template variable {name:?}"));
                }
            }
            "--max-new-tokens" => {
                parsed.max_new_tokens = parse_next(&mut raw, "--max-new-tokens")?;
            }
            "--speculative-draft-tokens" => {
                parsed.speculative_draft_tokens =
                    parse_next(&mut raw, "--speculative-draft-tokens")?;
            }
            "--context-size" => {
                parsed.context_size = Some(parse_next(&mut raw, "--context-size")?);
            }
            "--vulkan-device-index" => {
                parsed.vulkan_device_index = Some(parse_next(&mut raw, "--vulkan-device-index")?);
            }
            "--seed" => {
                parsed.random_seed = parse_next(&mut raw, "--seed")?;
            }
            "--temperature" => {
                parsed.temperature = Some(parse_next(&mut raw, "--temperature")?);
            }
            "--top-k" => {
                parsed.top_k = Some(parse_next(&mut raw, "--top-k")?);
            }
            "--top-p" => {
                parsed.top_p = Some(parse_next(&mut raw, "--top-p")?);
            }
            "--min-p" => {
                parsed.min_p = Some(parse_next(&mut raw, "--min-p")?);
            }
            "--presence-penalty" => {
                parsed.presence_penalty = Some(parse_next(&mut raw, "--presence-penalty")?);
            }
            "--repetition-penalty" => {
                parsed.repetition_penalty = Some(parse_next(&mut raw, "--repetition-penalty")?);
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
    let inspect_mode_count = usize::from(parsed.inspect_runtime)
        + usize::from(parsed.inspect_package)
        + usize::from(parsed.inspect_graph)
        + usize::from(parsed.inspect_placement)
        + usize::from(parsed.inspect_device_slice.is_some())
        + usize::from(parsed.inspect_devices);
    if inspect_mode_count > 1 {
        return Err(
            "--inspect-runtime, --inspect-package, --inspect-graph, --inspect-placement, --inspect-device-slice, and --inspect-devices are mutually exclusive"
                .to_string(),
        );
    }
    if parsed.chat && inspect_mode_count > 0 {
        return Err("--chat cannot be combined with inspect modes".to_string());
    }
    if parsed.chat && parsed.json {
        return Err("--json is not supported with --chat yet".to_string());
    }
    if !parsed.chat && !parsed.chat_template_variables.is_empty() {
        return Err("--chat-template-var requires --chat".to_string());
    }
    if matches!(parsed.inspect_device_slice.as_deref(), Some("")) {
        return Err("--inspect-device-slice must not be empty".to_string());
    }
    if matches!(parsed.default_device_id.as_deref(), Some("")) {
        return Err("--device must not be empty".to_string());
    }
    if parsed.max_new_tokens == 0 {
        return Err("--max-new-tokens must be at least 1".to_string());
    }
    if matches!(parsed.context_size, Some(0)) {
        return Err("--context-size must be at least 1".to_string());
    }
    if parsed
        .temperature
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("--temperature must be finite and greater than zero".to_string());
    }
    if parsed.top_k == Some(0) {
        return Err("--top-k must be at least 1".to_string());
    }
    if parsed
        .top_p
        .is_some_and(|value| !(0.0..=1.0).contains(&value) || value == 0.0)
    {
        return Err("--top-p must be finite and in (0, 1]".to_string());
    }
    if parsed
        .min_p
        .is_some_and(|value| !(0.0..=1.0).contains(&value))
    {
        return Err("--min-p must be finite and in [0, 1]".to_string());
    }
    if parsed
        .presence_penalty
        .is_some_and(|value| !value.is_finite())
    {
        return Err("--presence-penalty must be finite".to_string());
    }
    if parsed
        .repetition_penalty
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err("--repetition-penalty must be finite and greater than zero".to_string());
    }

    Ok(parsed)
}

fn parse_chat_template_variable(raw: &str) -> Result<(String, serde_json::Value), String> {
    let (name, encoded_value) = raw
        .split_once('=')
        .ok_or_else(|| format!("invalid chat template variable {raw:?}; expected NAME=JSON"))?;
    let name = name.trim();
    if name.is_empty()
        || !name
            .chars()
            .next()
            .is_some_and(|character| character == '_' || character.is_ascii_alphabetic())
        || !name
            .chars()
            .all(|character| character == '_' || character.is_ascii_alphanumeric())
    {
        return Err(format!(
            "invalid chat template variable {raw:?}; name must be a Jinja identifier"
        ));
    }
    let encoded_value = encoded_value.trim();
    if encoded_value.is_empty() {
        return Err(format!(
            "invalid chat template variable {raw:?}; JSON value must not be empty"
        ));
    }
    let value = serde_json::from_str(encoded_value).map_err(|error| {
        format!("invalid JSON value for chat template variable {name:?}: {error}")
    })?;
    Ok((name.to_string(), value))
}

fn parse_node_device_assignment(raw: &str) -> Result<(String, String), String> {
    let (component_id, device_id) = raw
        .split_once('=')
        .ok_or_else(|| format!("invalid runtime placement {raw:?}; expected NODE_ID=DEVICE_ID"))?;
    let component_id = component_id.trim();
    let device_id = device_id.trim();
    if component_id.is_empty() {
        return Err(format!(
            "invalid runtime placement {raw:?}; node id must not be empty"
        ));
    }
    if device_id.is_empty() {
        return Err(format!(
            "invalid runtime placement {raw:?}; device id must not be empty"
        ));
    }
    Ok((component_id.to_string(), device_id.to_string()))
}

fn parse_device_binding_assignment(raw: &str) -> Result<(String, String), String> {
    let (device_id, target) = raw.split_once('=').ok_or_else(|| {
        format!("invalid runtime device binding {raw:?}; expected LOGICAL_DEVICE_ID=TARGET")
    })?;
    let device_id = device_id.trim();
    let target = target.trim();
    if device_id.is_empty() {
        return Err(format!(
            "invalid runtime device binding {raw:?}; logical device id must not be empty"
        ));
    }
    if target.is_empty() {
        return Err(format!(
            "invalid runtime device binding {raw:?}; target must not be empty"
        ));
    }
    validate_runtime_device_target_syntax(target)?;
    Ok((device_id.to_string(), target.to_string()))
}

fn validate_runtime_device_target_syntax(raw: &str) -> Result<(), String> {
    if raw.starts_with("vulkan-uuid:") {
        parse_vulkan_device_uuid_ref(raw)?;
    } else if raw.starts_with("vulkan") {
        if parse_vulkan_physical_device_ref(raw)?.is_none() {
            return Err(format!(
                "invalid Vulkan physical device reference {raw:?}; expected vulkan:N"
            ));
        }
    } else if raw.starts_with("cpu") {
        parse_cpu_runtime_device_ref(raw)?;
    }
    Ok(())
}

fn resolve_runtime_vulkan_physical_device_ref(raw: &str) -> Result<Option<usize>, String> {
    if let Some(index) = parse_vulkan_physical_device_ref(raw)? {
        return Ok(Some(index));
    }
    let device_uuid = parse_vulkan_device_uuid_ref(raw)?;
    let cpu_ordinal = parse_cpu_runtime_device_ref(raw)?;
    if device_uuid.is_none() && cpu_ordinal.is_none() {
        return Ok(None);
    }
    let available_devices = VulkanComputeDevice::available_compute_devices()
        .map_err(|error| format!("failed to discover Vulkan devices: {error}"))?;
    resolve_runtime_vulkan_physical_device_ref_in(raw, &available_devices)
}

fn resolve_runtime_vulkan_physical_device_ref_in(
    raw: &str,
    available_devices: &[VulkanComputeDeviceInfo],
) -> Result<Option<usize>, String> {
    if let Some(index) = parse_vulkan_physical_device_ref(raw)? {
        return Ok(Some(index));
    }
    if let Some(device_uuid) = parse_vulkan_device_uuid_ref(raw)? {
        return available_devices
            .iter()
            .find(|device| device.device_uuid == device_uuid)
            .map(|device| Some(device.physical_device_index))
            .ok_or_else(|| format!("Vulkan device reference {raw:?} is not available"));
    }
    if let Some(cpu_ordinal) = parse_cpu_runtime_device_ref(raw)? {
        return available_devices
            .iter()
            .filter(|device| device.device_type == "cpu")
            .nth(cpu_ordinal)
            .map(|device| Some(device.physical_device_index))
            .ok_or_else(|| format!("CPU runtime device cpu{cpu_ordinal} is not available"));
    }
    Ok(None)
}

fn parse_vulkan_device_uuid_ref(raw: &str) -> Result<Option<[u8; 16]>, String> {
    let Some(encoded) = raw.strip_prefix("vulkan-uuid:") else {
        return Ok(None);
    };
    if encoded.len() != 32 {
        return Err(format!(
            "invalid Vulkan device UUID reference {raw:?}; expected vulkan-uuid followed by 32 hexadecimal digits"
        ));
    }
    let mut device_uuid = [0u8; 16];
    for (index, byte) in device_uuid.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&encoded[offset..offset + 2], 16)
            .map_err(|error| format!("invalid Vulkan device UUID reference {raw:?}: {error}"))?;
    }
    Ok(Some(device_uuid))
}

fn parse_vulkan_physical_device_ref(raw: &str) -> Result<Option<usize>, String> {
    if let Some(index) = raw.strip_prefix("vulkan:") {
        if index.is_empty() {
            return Err(format!(
                "invalid Vulkan physical device reference {raw:?}; expected vulkan:N"
            ));
        }
        return index
            .parse::<usize>()
            .map(Some)
            .map_err(|error| format!("invalid Vulkan physical device reference {raw:?}: {error}"));
    }
    Ok(None)
}

fn parse_cpu_runtime_device_ref(raw: &str) -> Result<Option<usize>, String> {
    if raw == "cpu" {
        return Ok(Some(0));
    }
    if let Some(index) = raw.strip_prefix("cpu:") {
        if index.is_empty() {
            return Err(format!(
                "invalid CPU runtime device reference {raw:?}; expected cpuN or cpu:N"
            ));
        }
        return index
            .parse::<usize>()
            .map(Some)
            .map_err(|error| format!("invalid CPU runtime device reference {raw:?}: {error}"));
    }
    if let Some(index) = raw.strip_prefix("cpu") {
        if index.is_empty() {
            return Err(format!(
                "invalid CPU runtime device reference {raw:?}; expected cpuN or cpu:N"
            ));
        }
        if index.chars().all(|ch| ch.is_ascii_digit()) {
            return index
                .parse::<usize>()
                .map(Some)
                .map_err(|error| format!("invalid CPU runtime device reference {raw:?}: {error}"));
        }
        return Err(format!(
            "invalid CPU runtime device reference {raw:?}; expected cpuN or cpu:N"
        ));
    }
    Ok(None)
}

fn parse_duplicate_after_assignment(raw: &str) -> Result<(String, String), String> {
    let (after_instance_id, new_instance_id) = raw.split_once('=').ok_or_else(|| {
        format!("invalid runtime duplicate {raw:?}; expected AFTER_INSTANCE_ID=NEW_INSTANCE_ID")
    })?;
    let after_instance_id = after_instance_id.trim();
    let new_instance_id = new_instance_id.trim();
    if after_instance_id.is_empty() {
        return Err(format!(
            "invalid runtime duplicate {raw:?}; source instance id must not be empty"
        ));
    }
    if new_instance_id.is_empty() {
        return Err(format!(
            "invalid runtime duplicate {raw:?}; new instance id must not be empty"
        ));
    }
    Ok((after_instance_id.to_string(), new_instance_id.to_string()))
}

fn parse_source_chain(raw: &str) -> Result<Vec<(String, String)>, String> {
    let separator = if raw.contains("->") { "->" } else { "," };
    let mut chain = Vec::new();
    let mut instance_ids = std::collections::BTreeSet::new();

    for raw_item in raw.split(separator) {
        let raw_item = raw_item.trim();
        if raw_item.is_empty() {
            return Err(format!(
                "invalid runtime chain {raw:?}; chain items must not be empty"
            ));
        }
        let (instance_id, source_component_id) =
            if let Some((instance_id, source_component_id)) = raw_item.split_once('=') {
                (instance_id.trim(), source_component_id.trim())
            } else {
                (raw_item, raw_item)
            };
        if instance_id.is_empty() {
            return Err(format!(
                "invalid runtime chain item {raw_item:?}; instance id must not be empty"
            ));
        }
        if source_component_id.is_empty() {
            return Err(format!(
                "invalid runtime chain item {raw_item:?}; source component id must not be empty"
            ));
        }
        if !instance_ids.insert(instance_id.to_string()) {
            return Err(format!(
                "invalid runtime chain {raw:?}; duplicate instance id {instance_id:?}"
            ));
        }
        chain.push((instance_id.to_string(), source_component_id.to_string()));
    }

    if chain.is_empty() {
        return Err("runtime chain must contain at least one component".to_string());
    }

    Ok(chain)
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
