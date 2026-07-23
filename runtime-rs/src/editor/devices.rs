pub fn discover_runtime_devices(
    default_device_id: &str,
    selected_vulkan_device_index: Option<usize>,
) -> Vec<RuntimeAvailableDevice> {
    match VulkanComputeDevice::available_compute_devices() {
        Ok(devices) if devices.is_empty() => vec![unavailable_device(
            default_device_id,
            "no compute-capable Vulkan physical devices were found",
            None,
        )],
        Ok(devices) => {
            let mut cpu_device_ordinal = 0usize;
            devices
                .iter()
                .map(|device| {
                    let selected_by_runtime = selected_vulkan_device_index
                        .map(|index| index == device.physical_device_index)
                        .unwrap_or(device.selected_by_default);
                    let cpu_runtime_device_id = if device.device_type == "cpu" {
                        let runtime_device_id = format!("cpu{cpu_device_ordinal}");
                        cpu_device_ordinal += 1;
                        Some(runtime_device_id)
                    } else {
                        None
                    };
                    let runtime_device_id = selected_by_runtime
                        .then(|| default_device_id.to_string())
                        .or(cpu_runtime_device_id.clone());
                    let device_id = runtime_device_id
                        .clone()
                        .unwrap_or_else(|| device.physical_device_id.clone());
                    RuntimeAvailableDevice {
                        device_id,
                        backend: "vulkan_compute".to_string(),
                        available: true,
                        runtime_device_id,
                        physical_device_id: Some(device.physical_device_id.clone()),
                        physical_device_index: Some(device.physical_device_index),
                        device_name: Some(device.device_name.clone()),
                        device_type: Some(device.device_type.clone()),
                        vendor_id: Some(device.vendor_id),
                        raw_device_id: Some(device.device_id),
                        api_version: Some(device.api_version),
                        driver_version: Some(device.driver_version),
                        compute_queue_family_indices: Some(
                            device.compute_queue_family_indices.clone(),
                        ),
                        memory_heaps: Some(
                            device
                                .memory_heaps
                                .iter()
                                .map(|heap| RuntimeAvailableMemoryHeap {
                                    heap_index: heap.heap_index,
                                    size_bytes: heap.size_bytes,
                                    device_local: heap.device_local,
                                })
                                .collect(),
                        ),
                        selected_by_default: Some(device.selected_by_default),
                        selected_by_runtime: Some(selected_by_runtime),
                        runtime_binding: Some(if selected_by_runtime {
                            "default_local_vulkan_target".to_string()
                        } else {
                            "inventory_only".to_string()
                        }),
                        can_host_runtime_pedals_on_physical_device: Some(true),
                        notes: if selected_by_runtime {
                            vec!["default target for unbound pedal instances".to_string()]
                        } else if let Some(cpu_runtime_device_id) = cpu_runtime_device_id {
                            vec![format!(
                                "CPU runtime target {cpu_runtime_device_id} backed by {}",
                                device.physical_device_id
                            )]
                        } else {
                            vec!["available runtime placement target".to_string()]
                        },
                        error: None,
                    }
                })
                .collect()
        }
        Err(error) => vec![unavailable_device(
            default_device_id,
            "Vulkan device discovery failed",
            Some(error.to_string()),
        )],
    }
}

fn unavailable_device(
    device_id: &str,
    note: &str,
    error: Option<String>,
) -> RuntimeAvailableDevice {
    RuntimeAvailableDevice {
        device_id: device_id.to_string(),
        backend: "vulkan_compute".to_string(),
        available: false,
        runtime_device_id: None,
        physical_device_id: None,
        physical_device_index: None,
        device_name: None,
        device_type: None,
        vendor_id: None,
        raw_device_id: None,
        api_version: None,
        driver_version: None,
        compute_queue_family_indices: None,
        memory_heaps: None,
        selected_by_default: None,
        selected_by_runtime: None,
        runtime_binding: None,
        can_host_runtime_pedals_on_physical_device: None,
        notes: vec![note.to_string()],
        error,
    }
}
