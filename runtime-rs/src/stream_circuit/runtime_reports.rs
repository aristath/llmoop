#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeBoundDevice {
    pub device_id: String,
    pub target: Option<String>,
    pub physical_device_index: Option<usize>,
    pub device_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeAvailableMemoryHeap {
    pub heap_index: u32,
    pub size_bytes: u64,
    pub device_local: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeAvailableDevice {
    pub device_id: String,
    pub backend: String,
    pub available: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_device_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_device_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_device_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_queue_family_indices: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_heaps: Option<Vec<RuntimeAvailableMemoryHeap>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_by_default: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_by_runtime: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_binding: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub can_host_runtime_components_on_physical_device: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeComponentPortSummary {
    pub id: String,
    pub signal: String,
    pub shape: Vec<usize>,
    pub source: Option<String>,
    pub component_port: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSourceComponent {
    pub component_index: usize,
    pub component_id: String,
    pub operator_type: String,
    pub runtime_role: CircuitRuntimeRole,
    pub implementation: String,
    pub behavioral_role: String,
    pub source_layer_index: Option<usize>,
    pub circuit_id: String,
    pub input_ports: Vec<RuntimeComponentPortSummary>,
    pub output_ports: Vec<RuntimeComponentPortSummary>,
    pub state_port_count: usize,
    pub parameter_ref_count: usize,
    pub node_count: usize,
    pub kernel_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeGraphSourceChainEntry {
    pub instance_id: String,
    pub source_component_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeGraphDuplicateAfterControl {
    pub after_instance_id: String,
    pub new_instance_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeGraphControls {
    pub default_device_id: Option<String>,
    pub node_devices: BTreeMap<String, String>,
    pub source_chain: Option<Vec<RuntimeGraphSourceChainEntry>>,
    pub duplicate_after: Vec<RuntimeGraphDuplicateAfterControl>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeCompiledExecutionGraphSummary {
    pub topology: String,
    pub source_component_count: usize,
    pub source_components: Vec<RuntimeSourceComponent>,
    pub max_context_activations: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeEffectiveExecutionGraphTopology {
    pub topology: String,
    pub component_count: usize,
    pub edge_count: usize,
    pub local_edge_count: usize,
    pub cross_device_edge_count: usize,
    pub device_count: usize,
    pub device_ids: Vec<String>,
    pub device_bindings: RuntimeDeviceBindings,
    pub edge_routes: RuntimeEdgeRoutes,
    pub components: Vec<ComponentPlacement>,
    pub edges: Vec<ComponentEdgePlacement>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeTopologyReport {
    pub ok: bool,
    pub schema: String,
    pub package_manifest: PathBuf,
    pub package_root: PathBuf,
    pub package_id: String,
    pub compiled_schema: String,
    pub config_path: String,
    pub tokenizer: Value,
    pub available_devices: Vec<RuntimeAvailableDevice>,
    pub compiled: RuntimeCompiledExecutionGraphSummary,
    pub runtime_graph_controls: RuntimeGraphControls,
    pub runtime_graph: StreamCircuitRuntimeGraph,
    pub effective: RuntimeEffectiveExecutionGraphTopology,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimePackageInspectionReport {
    pub ok: bool,
    pub package_manifest: PathBuf,
    pub package_root: PathBuf,
    pub schema: String,
    pub package_id: String,
    pub config_path: String,
    pub tokenizer: Value,
    pub compiled_topology: String,
    pub runtime_graph: RuntimeGraphControls,
    pub device_bindings: RuntimeDeviceBindings,
    pub max_context_activations: usize,
    pub source_component_count: usize,
    pub source_components: Vec<RuntimeSourceComponent>,
    pub available_devices: Vec<RuntimeAvailableDevice>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeGraphPlacementReport {
    pub schema: String,
    pub topology: String,
    pub local_edge_count: usize,
    pub cross_device_edge_count: usize,
    pub runtime_routes: RuntimeEdgeRoutes,
    pub components: Vec<ComponentPlacement>,
    pub edges: Vec<ComponentEdgePlacement>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeGraphInspectionReport {
    pub ok: bool,
    pub package_manifest: PathBuf,
    pub package_root: PathBuf,
    pub package_id: String,
    pub compiled_source_component_count: usize,
    pub runtime_graph_controls: RuntimeGraphControls,
    pub runtime_graph: StreamCircuitRuntimeGraph,
    pub device_bindings: RuntimeDeviceBindings,
    pub effective_component_count: usize,
    pub effective_edge_count: usize,
    pub placement: RuntimeGraphPlacementReport,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeLocalEdgeBufferReport {
    pub edge_index: usize,
    pub signal: String,
    pub source_component_id: String,
    pub destination_component_id: String,
    pub device_id: String,
    pub byte_capacity: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeRemoteEdgeBufferReport {
    pub edge_index: usize,
    pub signal: String,
    pub source_device_id: String,
    pub source_component_id: String,
    pub destination_device_id: String,
    pub destination_component_id: String,
    pub byte_capacity: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeDeviceTickPlanReport {
    pub stage_count: usize,
    pub receive_stage_count: usize,
    pub dispatch_stage_count: usize,
    pub publish_stage_count: usize,
    pub local_edge_read_count: usize,
    pub local_edge_write_count: usize,
    pub incoming_edge_read_count: usize,
    pub outgoing_edge_write_count: usize,
    pub model_input_read_count: usize,
    pub model_output_write_count: usize,
    pub can_execute: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeDeviceSliceReport {
    pub ok: bool,
    pub package_manifest: PathBuf,
    pub device_name: String,
    pub device_id: String,
    pub context_window_activations: usize,
    pub hosted_components: Vec<String>,
    pub local_edges: Vec<RuntimeLocalEdgeBufferReport>,
    pub incoming_edges: Vec<RuntimeRemoteEdgeBufferReport>,
    pub outgoing_edges: Vec<RuntimeRemoteEdgeBufferReport>,
    pub hosted_component_count: usize,
    pub incoming_edge_count: usize,
    pub outgoing_edge_count: usize,
    pub permanent_parameter_count: usize,
    pub permanent_parameter_bytes: usize,
    pub reusable_kernel_word_count: usize,
    pub loaded_kernel_artifact_count: usize,
    pub dispatch_count: usize,
    pub descriptor_count: usize,
    pub model_boundary_descriptor_count: usize,
    pub incoming_edge_descriptor_count: usize,
    pub outgoing_edge_descriptor_count: usize,
    pub tick_plan: RuntimeDeviceTickPlanReport,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacementReport {
    pub ok: bool,
    pub package_manifest: PathBuf,
    pub context_window_activations: usize,
    pub runtime_graph: RuntimeGraphControls,
    pub device_bindings: RuntimeDeviceBindings,
    pub bound_devices: Vec<RuntimeBoundDevice>,
    pub edge_routes: RuntimeEdgeRoutes,
    pub device_count: usize,
    pub device_ids: Vec<String>,
    pub devices: Vec<RuntimeDeviceSliceReport>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeTokenizerOptionsReport {
    pub add_special_tokens: bool,
    pub skip_special_tokens: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedTransportStatsReport {
    pub pending_packet_count: usize,
    pub pending_byte_count: usize,
    pub pending_direct_edge_count: usize,
    pub pending_direct_byte_count: usize,
    pub published_packet_count: usize,
    pub published_byte_count: usize,
    pub received_packet_count: usize,
    pub received_byte_count: usize,
    pub direct_copy_count: usize,
    pub direct_copy_byte_count: usize,
    pub direct_receive_count: usize,
    pub direct_receive_byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedTransportReport {
    pub published_packet_count: usize,
    pub published_byte_count: usize,
    pub received_packet_count: usize,
    pub received_byte_count: usize,
    pub direct_copy_count: usize,
    pub direct_copy_byte_count: usize,
    pub direct_receive_count: usize,
    pub direct_receive_byte_count: usize,
    pub by_tick: Vec<RuntimePlacedTransportStatsReport>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePromptTimingReport {
    pub setup_time_ns: u64,
    pub run_time_ns: u64,
    pub total_time_ns: u64,
    pub prefill_token_count: usize,
    pub decode_token_count: usize,
    pub generated_token_count: usize,
    pub scheduler_step_count: usize,
    pub activation_batch_count: usize,
    pub max_activation_batch_width: usize,
    pub prefill_activation_count: usize,
    pub decode_activation_count: usize,
    pub prefill_time_ns: u64,
    pub decode_time_ns: u64,
    pub tick_count: usize,
    pub scheduler_turn_count: usize,
    pub average_generated_token_time_ns: Option<u64>,
    pub average_prefill_activation_time_ns: Option<u64>,
    pub average_decode_activation_time_ns: Option<u64>,
    pub average_tick_time_ns: Option<u64>,
    pub average_scheduler_turn_time_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedComponentDispatchTimingReport {
    pub dispatch_index: usize,
    pub kernel_id: String,
    pub node_id: String,
    pub op: String,
    pub reusable_family_id: String,
    pub run_time_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedComponentTimingReport {
    pub stream_tick: u64,
    pub device_id: String,
    pub component_id: String,
    pub dispatch_count: usize,
    pub run_time_ns: u64,
    pub average_dispatch_time_ns: Option<u64>,
    pub dispatches: Vec<RuntimePlacedComponentDispatchTimingReport>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedComponentTimingSummaryReport {
    pub device_id: String,
    pub component_id: String,
    pub tick_count: usize,
    pub dispatch_count: usize,
    pub total_run_time_ns: u64,
    pub average_tick_time_ns: Option<u64>,
    pub average_dispatch_time_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimePromptBenchmarkU64MetricReport {
    pub total: u64,
    pub min: u64,
    pub max: u64,
    pub average: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimePromptBenchmarkUsizeMetricReport {
    pub total: usize,
    pub min: usize,
    pub max: usize,
    pub average: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePromptBenchmarkTransportTotalsReport {
    pub published_packet_count: usize,
    pub published_byte_count: usize,
    pub received_packet_count: usize,
    pub received_byte_count: usize,
    pub direct_copy_count: usize,
    pub direct_copy_byte_count: usize,
    pub direct_receive_count: usize,
    pub direct_receive_byte_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimePromptBenchmarkRunReport {
    pub run_index: usize,
    pub execution_mode: String,
    pub stop_reason: String,
    pub generated_token_count: usize,
    pub tick_count: usize,
    pub scheduler_turn_count: usize,
    pub setup_time_ns: u64,
    pub run_time_ns: u64,
    pub total_time_ns: u64,
    pub generated_tokens_per_second: Option<f64>,
    pub transport: Option<RuntimePromptBenchmarkTransportTotalsReport>,
    pub component_timing_summaries: Vec<RuntimePlacedComponentTimingSummaryReport>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimePromptBenchmarkReport {
    pub ok: bool,
    pub execution_mode: String,
    pub package_manifest: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub runtime_graph: RuntimeGraphControls,
    pub device_bindings: RuntimeDeviceBindings,
    pub device_count: usize,
    pub device_ids: Vec<String>,
    pub profile_runs: usize,
    pub prompt_text: String,
    pub prompt_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub setup_time_ns: RuntimePromptBenchmarkU64MetricReport,
    pub run_time_ns: RuntimePromptBenchmarkU64MetricReport,
    pub total_time_ns: RuntimePromptBenchmarkU64MetricReport,
    pub generated_token_count: RuntimePromptBenchmarkUsizeMetricReport,
    pub tick_count: RuntimePromptBenchmarkUsizeMetricReport,
    pub scheduler_turn_count: RuntimePromptBenchmarkUsizeMetricReport,
    pub generated_tokens_per_second: Option<f64>,
    pub stop_reasons: BTreeMap<String, usize>,
    pub transport_totals: Option<RuntimePromptBenchmarkTransportTotalsReport>,
    pub component_timing_summaries: Vec<RuntimePlacedComponentTimingSummaryReport>,
    pub runs: Vec<RuntimePromptBenchmarkRunReport>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePlacedPromptRunReport {
    pub ok: bool,
    pub execution_mode: String,
    pub package_manifest: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub input_device_id: String,
    pub output_device_id: String,
    pub device_count: usize,
    pub device_ids: Vec<String>,
    pub bound_devices: Vec<RuntimeBoundDevice>,
    pub edge_routes: RuntimeEdgeRoutes,
    pub runtime_graph: RuntimeGraphControls,
    pub device_bindings: RuntimeDeviceBindings,
    pub hosted_component_count: usize,
    pub context_window_activations: usize,
    pub scheduled_token_activations: usize,
    pub tokenizer: RuntimeTokenizerOptionsReport,
    pub prompt_text: String,
    pub prompt_ids: Vec<u32>,
    pub generated_ids: Vec<u32>,
    pub generated_text: String,
    pub output_text: String,
    pub stop_reason: String,
    pub tick_count: usize,
    pub scheduler_turns: usize,
    pub completed_stage_deltas: Vec<usize>,
    pub transport: RuntimePlacedTransportReport,
    pub timing: RuntimePromptTimingReport,
    pub component_timings: Vec<RuntimePlacedComponentTimingReport>,
    pub component_timing_summaries: Vec<RuntimePlacedComponentTimingSummaryReport>,
    pub speculative_cycle_count: usize,
    pub proposed_draft_token_count: usize,
    pub accepted_draft_token_count: usize,
    pub speculative_emitted_token_count: usize,
    pub speculative_draft_time_ns: u64,
    pub speculative_target_verification_time_ns: u64,
    pub speculative_draft_catch_up_time_ns: u64,
}
