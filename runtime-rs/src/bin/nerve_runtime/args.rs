use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use chrono::{DateTime, FixedOffset, Local};
use nerve_runtime::{
    CircuitPort, ComponentEdgePlacement, ComponentPlacement, RUNTIME_DEFAULT_LOGICAL_DEVICE_ID,
    RUNTIME_TOPOLOGY_SCHEMA, RuntimeAvailableDevice, RuntimeBoundDevice, RuntimeEdgeRouteTarget,
    RuntimeEdgeRoutes, RuntimeCompiledExecutionGraphSummary, RuntimeDeviceBindings,
    RuntimeDeviceSliceReport, RuntimeDeviceTickPlanReport, RuntimeEffectiveExecutionGraphTopology,
    RuntimeFeedbackExecutionReport, RuntimeLocalEdgeBufferReport, RuntimePackageInspectionReport,
    RuntimeGraphControls,
    RuntimeGraphDuplicateAfterControl, RuntimeGraphInspectionReport, RuntimeGraphPlacementReport,
    RuntimeGraphSourceChainEntry, RuntimeComponentPortSummary, RuntimePlacedComponentTimingSummaryReport,
    RuntimePlacedPromptRunReport, RuntimePlacedTransportReport, RuntimePlacementReport,
    RuntimePromptTimingReport, RuntimeRemoteEdgeBufferReport, RuntimeSourceComponent,
    RuntimeTokenizerOptionsReport, RuntimeTopologyReport, VulkanComputeDevice,
    VulkanComputeDeviceCatalog, VulkanComputeDeviceInfo, VulkanResidentExecutionCounters,
    VulkanComputeTargetCapabilities,
    VulkanResidentFeedbackExecutionStats,
    VulkanResidentHfTokenizerTextCodec, VulkanResidentInProcessPlacedPromptEngine,
    VulkanResidentInProcessPlacedPromptStream, VulkanResidentModelPackageDeviceSlice,
    VulkanResidentModelPackageManifest, VulkanResidentRuntimeModel,
    VulkanResidentPlacedPrefixStateCacheStats,
    VulkanResidentSamplerRuntimeConfig, VulkanResidentTokenInputEvent,
    VulkanResidentTokenTextCodec, VulkanReusableKernelArtifactManifest, discover_runtime_devices,
    reset_vulkan_resident_execution_counters, vulkan_resident_execution_counters,
};
use minijinja::{Environment, Error as TemplateError, ErrorKind as TemplateErrorKind};
use serde::Serialize;

#[derive(Clone, Debug, PartialEq)]
struct Args {
    package_manifest: Option<PathBuf>,
    prompt: Option<String>,
    chat: bool,
    inspect_runtime: bool,
    inspect_package: bool,
    inspect_graph: bool,
    inspect_placement: bool,
    inspect_device_slice: Option<String>,
    inspect_devices: bool,
    default_device_id: Option<String>,
    node_devices: BTreeMap<String, String>,
    device_bindings: BTreeMap<String, String>,
    duplicate_after: Vec<(String, String)>,
    source_chain: Option<Vec<(String, String)>>,
    chat_template_variables: BTreeMap<String, serde_json::Value>,
    max_new_tokens: usize,
    speculative_draft_tokens: usize,
    context_size: Option<usize>,
    vulkan_device_index: Option<usize>,
    random_seed: u32,
    temperature: Option<f32>,
    top_k: Option<u32>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    generated_only: bool,
    json: bool,
}

struct PromptRunContext<'a> {
    args: &'a Args,
    package_manifest: &'a Path,
    manifest_dir: &'a Path,
    tokenizer_dir: &'a Path,
    prompt: &'a str,
    prompt_ids: &'a [u32],
    scheduled_token_activations: usize,
    capacity: usize,
    codec: &'a VulkanResidentHfTokenizerTextCodec,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            package_manifest: None,
            prompt: None,
            chat: false,
            inspect_runtime: false,
            inspect_package: false,
            inspect_graph: false,
            inspect_placement: false,
            inspect_device_slice: None,
            inspect_devices: false,
            default_device_id: None,
            node_devices: BTreeMap::new(),
            device_bindings: BTreeMap::new(),
            duplicate_after: Vec::new(),
            source_chain: None,
            chat_template_variables: BTreeMap::new(),
            max_new_tokens: 65_536,
            speculative_draft_tokens: 0,
            context_size: None,
            vulkan_device_index: None,
            random_seed: 0,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            presence_penalty: None,
            repetition_penalty: None,
            add_special_tokens: true,
            skip_special_tokens: true,
            generated_only: false,
            json: false,
        }
    }
}

#[derive(Debug, Serialize)]
struct RuntimeDeviceCapabilitiesReport {
    ok: bool,
    schema: &'static str,
    devices: Vec<VulkanComputeTargetCapabilities>,
}

fn sampler_runtime_config(args: &Args) -> VulkanResidentSamplerRuntimeConfig {
    VulkanResidentSamplerRuntimeConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        presence_penalty: args.presence_penalty,
        repetition_penalty: args.repetition_penalty,
    }
}
