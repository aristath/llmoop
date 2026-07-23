fn placement_device_ids(components: &[ComponentPlacement]) -> Vec<String> {
    let mut device_ids = components
        .iter()
        .map(|component| component.device_id.clone())
        .collect::<Vec<_>>();
    device_ids.sort();
    device_ids.dedup();
    device_ids
}

fn runtime_model_placement(
    manifest_dir: &Path,
    runtime_model: &VulkanResidentRuntimeModel,
) -> Result<nerve_runtime::StreamCircuitPlacementPlan, Box<dyn Error>> {
    let graph = runtime_model.resolved_graph(manifest_dir.to_path_buf())?;
    graph
        .placement_plan(&runtime_model.placement)
        .map_err(|error| Box::new(error) as Box<dyn Error>)
}

fn tokenizer_dir_from_package(package_manifest: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
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

fn runtime_model(
    args: &Args,
    package_manifest: &Path,
) -> Result<VulkanResidentRuntimeModel, Box<dyn Error>> {
    let manifest = VulkanResidentModelPackageManifest::from_json_file(package_manifest)?;
    Ok(manifest.mount_runtime_graph_controls(
        args.default_device_id.as_deref(),
        &args.node_devices,
        &args.duplicate_after,
        args.source_chain.as_deref(),
    )?)
}

struct RuntimeBoundVulkanDevices {
    devices: BTreeMap<String, Rc<VulkanComputeDevice>>,
    physical_device_indices: BTreeMap<String, usize>,
    physical_device_ids: BTreeMap<String, String>,
}

fn runtime_physical_device_bindings_in(
    args: &Args,
    logical_device_ids: &[String],
    available_devices: &[VulkanComputeDeviceInfo],
) -> Result<BTreeMap<String, usize>, io::Error> {
    let default_physical_device_index = if let Some(index) = args.vulkan_device_index {
        index
    } else {
        available_devices
            .iter()
            .find(|device| device.selected_by_default)
            .or_else(|| available_devices.first())
            .map(|device| device.physical_device_index)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "no Vulkan compute-capable physical devices are available",
                )
            })?
    };
    let mut logical_device_ids = logical_device_ids.to_vec();
    logical_device_ids.sort();
    logical_device_ids.dedup();
    logical_device_ids
        .into_iter()
        .map(|logical_device_id| {
            let physical_device_index = runtime_mount_physical_device_index(
                args,
                &logical_device_id,
                default_physical_device_index,
                available_devices,
            )?;
            Ok((logical_device_id, physical_device_index))
        })
        .collect()
}

fn runtime_bound_vulkan_devices(
    args: &Args,
    logical_device_ids: &[String],
) -> Result<RuntimeBoundVulkanDevices, Box<dyn Error>> {
    let device_catalog = VulkanComputeDeviceCatalog::discover()?;
    let available_devices = device_catalog.available_compute_devices();
    let requested_bindings =
        runtime_physical_device_bindings_in(args, logical_device_ids, available_devices)?;
    let mut devices = BTreeMap::new();
    let mut physical_devices: BTreeMap<usize, Rc<VulkanComputeDevice>> = BTreeMap::new();
    let mut physical_device_indices = BTreeMap::new();
    let mut physical_device_ids = BTreeMap::new();

    for (logical_device_id, physical_device_index) in requested_bindings {
        let available_device = available_devices
            .iter()
            .find(|device| device.physical_device_index == physical_device_index)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "Vulkan physical device index {physical_device_index} is not available"
                    ),
                )
            })?;
        let device = if let Some(device) = physical_devices.get(&physical_device_index) {
            Rc::clone(device)
        } else {
            let device = Rc::new(device_catalog.open_physical_device_index(physical_device_index)?);
            physical_devices.insert(physical_device_index, Rc::clone(&device));
            device
        };
        devices.insert(logical_device_id.clone(), device);
        physical_device_indices.insert(logical_device_id.clone(), physical_device_index);
        physical_device_ids.insert(
            logical_device_id.clone(),
            available_device.physical_device_id.clone(),
        );
    }

    Ok(RuntimeBoundVulkanDevices {
        devices,
        physical_device_indices,
        physical_device_ids,
    })
}

fn runtime_default_vulkan_physical_device_index() -> Result<usize, Box<dyn Error>> {
    let devices = VulkanComputeDevice::available_compute_devices()?;
    devices
        .iter()
        .find(|device| device.selected_by_default)
        .or_else(|| devices.first())
        .map(|device| device.physical_device_index)
        .ok_or_else(|| {
            Box::new(io::Error::new(
                io::ErrorKind::NotFound,
                "no Vulkan compute-capable physical devices are available",
            )) as Box<dyn Error>
        })
}

fn bound_devices_report(bound_devices: &RuntimeBoundVulkanDevices) -> Vec<RuntimeBoundDevice> {
    bound_devices
        .devices
        .iter()
        .map(|(logical_device_id, device)| {
            let physical_device_index = bound_devices
                .physical_device_indices
                .get(logical_device_id)
                .copied();
            RuntimeBoundDevice {
                device_id: logical_device_id.clone(),
                target: bound_devices
                    .physical_device_ids
                    .get(logical_device_id)
                    .cloned(),
                physical_device_index,
                device_name: device.device_name().to_string(),
            }
        })
        .collect::<Vec<_>>()
}

fn runtime_edge_routes_report(args: &Args, edges: &[ComponentEdgePlacement]) -> RuntimeEdgeRoutes {
    RuntimeEdgeRoutes::from_edges(edges, |device_id| {
        runtime_target_for_logical_device(args, device_id)
    })
}

fn bound_edge_routes_report(
    bound_devices: &RuntimeBoundVulkanDevices,
    edges: &[ComponentEdgePlacement],
) -> RuntimeEdgeRoutes {
    RuntimeEdgeRoutes::from_edges(edges, |device_id| {
        let physical_device_index = bound_devices
            .physical_device_indices
            .get(device_id)
            .copied();
        RuntimeEdgeRouteTarget {
            target: bound_devices.physical_device_ids.get(device_id).cloned(),
            physical_device_index,
            binding_source: "mounted".to_string(),
        }
    })
}

fn runtime_mount_physical_device_index(
    args: &Args,
    logical_device_id: &str,
    default_physical_device_index: usize,
    available_devices: &[VulkanComputeDeviceInfo],
) -> Result<usize, io::Error> {
    if let Some(target) = args.device_bindings.get(logical_device_id) {
        return resolve_runtime_vulkan_physical_device_ref_in(target, available_devices)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "logical device {logical_device_id:?} is bound to unsupported target {target:?}; local mounted execution supports vulkan:N or cpuN targets"
                    ),
                )
            });
    }
    match resolve_runtime_vulkan_physical_device_ref_in(logical_device_id, available_devices)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?
    {
        Some(index) => Ok(index),
        None if logical_device_id.contains(':') => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "logical device id {logical_device_id:?} looks like an unsupported direct runtime target; local mounted execution supports vulkan:N or cpuN targets"
            ),
        )),
        None => Ok(default_physical_device_index),
    }
}

fn runtime_target_for_logical_device(
    args: &Args,
    logical_device_id: &str,
) -> RuntimeEdgeRouteTarget {
    if let Some(target) = args.device_bindings.get(logical_device_id) {
        let physical_device_index = resolve_runtime_vulkan_physical_device_ref(target)
            .ok()
            .flatten();
        return RuntimeEdgeRouteTarget {
            target: Some(target.clone()),
            physical_device_index,
            binding_source: "explicit".to_string(),
        };
    }
    match resolve_runtime_vulkan_physical_device_ref(logical_device_id) {
        Ok(Some(index)) => RuntimeEdgeRouteTarget {
            target: Some(logical_device_id.to_string()),
            physical_device_index: Some(index),
            binding_source: "device_id".to_string(),
        },
        Ok(None) | Err(_) if logical_device_id.contains(':') => RuntimeEdgeRouteTarget {
            target: Some(logical_device_id.to_string()),
            physical_device_index: None,
            binding_source: "device_id".to_string(),
        },
        Ok(None) | Err(_) => {
            let default_physical_device_index =
                runtime_report_default_vulkan_physical_device_index(args);
            let target = default_physical_device_index.map(|index| format!("vulkan:{index}"));
            RuntimeEdgeRouteTarget {
                physical_device_index: default_physical_device_index,
                target,
                binding_source: if args.vulkan_device_index.is_some() {
                    "process_default".to_string()
                } else {
                    "runtime_default".to_string()
                },
            }
        }
    }
}

fn runtime_report_default_vulkan_physical_device_index(args: &Args) -> Option<usize> {
    args.vulkan_device_index
        .or_else(|| {
            args.default_device_id.as_deref().and_then(|device_id| {
                resolve_runtime_vulkan_physical_device_ref(device_id)
                    .ok()
                    .flatten()
            })
        })
        .or_else(|| runtime_default_vulkan_physical_device_index().ok())
}
