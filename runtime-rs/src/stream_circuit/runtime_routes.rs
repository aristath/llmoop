#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeEdgeRouteTarget {
    pub target: Option<String>,
    pub physical_device_index: Option<usize>,
    pub binding_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeEdgeRouteKind {
    LogicalLocal,
    SamePhysicalTarget,
    CrossPhysicalTarget,
    UnresolvedRuntimeTarget,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeEdgeRoute {
    pub edge_index: usize,
    pub signal: String,
    pub shape: Vec<usize>,
    pub source_component_id: String,
    pub source_device_id: String,
    pub source_target: Option<String>,
    pub source_physical_device_index: Option<usize>,
    pub source_binding: String,
    pub destination_component_id: String,
    pub destination_device_id: String,
    pub destination_target: Option<String>,
    pub destination_physical_device_index: Option<usize>,
    pub destination_binding: String,
    pub route_kind: RuntimeEdgeRouteKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeEdgeRoutes {
    pub schema: String,
    pub edge_count: usize,
    pub logical_local_edge_count: usize,
    pub logical_cross_device_edge_count: usize,
    pub same_physical_target_edge_count: usize,
    pub cross_physical_target_edge_count: usize,
    pub unresolved_target_edge_count: usize,
    pub routes: Vec<RuntimeEdgeRoute>,
}

impl RuntimeEdgeRoutes {
    pub fn from_edges<F>(edges: &[ComponentEdgePlacement], mut target_for: F) -> Self
    where
        F: FnMut(&str) -> RuntimeEdgeRouteTarget,
    {
        let mut logical_local_edge_count = 0usize;
        let mut logical_cross_device_edge_count = 0usize;
        let mut same_physical_target_edge_count = 0usize;
        let mut cross_physical_target_edge_count = 0usize;
        let mut unresolved_target_edge_count = 0usize;

        let routes = edges
            .iter()
            .map(|edge| {
                let source_target = target_for(&edge.source_device_id);
                let destination_target = target_for(&edge.destination_device_id);
                let is_logical_local = edge.source_device_id == edge.destination_device_id;
                let route_kind = if is_logical_local {
                    logical_local_edge_count += 1;
                    RuntimeEdgeRouteKind::LogicalLocal
                } else {
                    logical_cross_device_edge_count += 1;
                    match (&source_target.target, &destination_target.target) {
                        (Some(source), Some(destination)) if source == destination => {
                            same_physical_target_edge_count += 1;
                            RuntimeEdgeRouteKind::SamePhysicalTarget
                        }
                        (Some(_), Some(_)) => {
                            cross_physical_target_edge_count += 1;
                            RuntimeEdgeRouteKind::CrossPhysicalTarget
                        }
                        _ => {
                            unresolved_target_edge_count += 1;
                            RuntimeEdgeRouteKind::UnresolvedRuntimeTarget
                        }
                    }
                };

                RuntimeEdgeRoute {
                    edge_index: edge.edge_index,
                    signal: edge.signal.clone(),
                    shape: edge.shape.clone(),
                    source_component_id: edge.source_component_id.clone(),
                    source_device_id: edge.source_device_id.clone(),
                    source_target: source_target.target,
                    source_physical_device_index: source_target.physical_device_index,
                    source_binding: source_target.binding_source,
                    destination_component_id: edge.destination_component_id.clone(),
                    destination_device_id: edge.destination_device_id.clone(),
                    destination_target: destination_target.target,
                    destination_physical_device_index: destination_target.physical_device_index,
                    destination_binding: destination_target.binding_source,
                    route_kind,
                }
            })
            .collect::<Vec<_>>();

        Self {
            schema: RUNTIME_EDGE_ROUTES_SCHEMA.to_string(),
            edge_count: edges.len(),
            logical_local_edge_count,
            logical_cross_device_edge_count,
            same_physical_target_edge_count,
            cross_physical_target_edge_count,
            unresolved_target_edge_count,
            routes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeLogicalDeviceBinding {
    pub device_id: String,
    pub target: Option<String>,
    pub binding_source: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeDeviceBindings {
    pub schema: String,
    pub process_vulkan_device_index: Option<usize>,
    pub requested_vulkan_device_indices: Vec<usize>,
    pub default_vulkan_device_index: Option<usize>,
    pub explicit_bindings: BTreeMap<String, String>,
    pub logical_devices: Vec<RuntimeLogicalDeviceBinding>,
    pub can_mount_in_process: bool,
    pub mounting_model: String,
    pub unsupported_targets: Vec<String>,
    pub notes: Vec<String>,
}

impl RuntimeDeviceBindings {
    pub fn from_vulkan_targets<F>(
        logical_device_ids: &[String],
        explicit_bindings: &BTreeMap<String, String>,
        default_vulkan_device_index: Option<usize>,
        mut vulkan_physical_device_index_for_target: F,
    ) -> Self
    where
        F: FnMut(&str) -> Result<Option<usize>, String>,
    {
        let mut logical_ids = logical_device_ids.to_vec();
        for logical_device_id in explicit_bindings.keys() {
            if !logical_ids.contains(logical_device_id) {
                logical_ids.push(logical_device_id.clone());
            }
        }
        logical_ids.sort();
        logical_ids.dedup();

        let mut vulkan_indices = Vec::new();
        let mut unsupported_targets = Vec::new();
        if let Some(index) = default_vulkan_device_index {
            vulkan_indices.push(index);
        }
        for (logical_device_id, target) in explicit_bindings {
            match vulkan_physical_device_index_for_target(target) {
                Ok(Some(index)) => vulkan_indices.push(index),
                Ok(None) => unsupported_targets.push(format!("{logical_device_id}={target}")),
                Err(error) => {
                    unsupported_targets.push(format!("{logical_device_id}={target} ({error})"))
                }
            }
        }
        for logical_device_id in &logical_ids {
            if explicit_bindings.contains_key(logical_device_id) {
                continue;
            }
            match vulkan_physical_device_index_for_target(logical_device_id) {
                Ok(Some(index)) => vulkan_indices.push(index),
                Ok(None) if logical_device_id.contains(':') => {
                    unsupported_targets.push(logical_device_id.clone())
                }
                Err(error) => unsupported_targets.push(format!("{logical_device_id} ({error})")),
                Ok(None) => {}
            }
        }
        vulkan_indices.sort_unstable();
        vulkan_indices.dedup();
        unsupported_targets.sort();
        unsupported_targets.dedup();

        let logical_devices = logical_ids
            .iter()
            .map(|logical_device_id| {
                let explicit_target = explicit_bindings.get(logical_device_id);
                let direct_target = if explicit_target.is_none() {
                    match vulkan_physical_device_index_for_target(logical_device_id) {
                        Ok(Some(index)) => Some(format!("vulkan:{index}")),
                        Ok(None) | Err(_) if logical_device_id.contains(':') => {
                            Some(logical_device_id.clone())
                        }
                        Ok(None) | Err(_) => None,
                    }
                } else {
                    None
                };
                let has_direct_target = direct_target.is_some();
                let target = explicit_target
                    .cloned()
                    .or(direct_target)
                    .or_else(|| default_vulkan_device_index.map(|index| format!("vulkan:{index}")));
                let binding_source = if explicit_target.is_some() {
                    "explicit"
                } else if has_direct_target {
                    "device_id"
                } else if default_vulkan_device_index.is_some() {
                    "process_default"
                } else {
                    "runtime_default"
                };
                RuntimeLogicalDeviceBinding {
                    device_id: logical_device_id.clone(),
                    target,
                    binding_source: binding_source.to_string(),
                }
            })
            .collect::<Vec<_>>();

        let can_mount_in_process = unsupported_targets.is_empty();

        Self {
            schema: RUNTIME_DEVICE_BINDINGS_SCHEMA.to_string(),
            process_vulkan_device_index: default_vulkan_device_index,
            requested_vulkan_device_indices: vulkan_indices,
            default_vulkan_device_index,
            explicit_bindings: explicit_bindings.clone(),
            logical_devices,
            can_mount_in_process,
            mounting_model: if can_mount_in_process {
                "local_vulkan_device_pool".to_string()
            } else {
                "unsupported_targets".to_string()
            },
            unsupported_targets,
            notes: if can_mount_in_process {
                vec![
                    "mounted logical device slices can use distinct local Vulkan physical devices in this runtime process"
                        .to_string(),
                ]
            } else {
                vec![
                    "only local vulkan:N targets are mountable by this runtime process".to_string(),
                ]
            },
        }
    }
}

