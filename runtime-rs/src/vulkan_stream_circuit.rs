use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use smallvec::SmallVec;

use crate::execution_schedule::{
    RuntimeExecutionCost, RuntimeExecutionQuantumCalibrator, RuntimeExecutionRegion,
};
use crate::stream_circuit::{
    CircuitNode, CircuitParamsArtifact, CircuitRuntimeRole, CircuitStateArtifact,
    ComponentEdgePlacement, EdgeTransport, LOWERED_EXECUTION_GRAPH_SCHEMA, LoweredCircuitRef,
    LoweredExecutionGraph, LoweredExecutionGraphGraph, LoweredExecutionGraphSource,
    LoweredExecutionGraphSummary, RUNTIME_DEFAULT_LOGICAL_DEVICE_ID, ResolvedCircuitArtifact,
    ResolvedLoweredExecutionGraph, StreamCircuit, StreamCircuitGraphBoundary,
    StreamCircuitNodeInstanceStatePolicy, StreamCircuitPlacementPlan, StreamCircuitPlacementSpec,
    StreamCircuitRuntimeGraph,
};
use crate::stream_plan::{
    CircuitActivationPlan, PlannedNode, PlannedParameterResource, PlannedPort, SignalProducer,
    SignalStorage, StreamCircuitExecutionPlan, StreamCircuitResourcePlan, TensorIndex,
};
use crate::stream_runtime::{
    RuntimeStreamActivation, RuntimeStreamActivationBatch, RuntimeStreamActivationBatchKind,
    RuntimeStreamActivationKind, RuntimeStreamActivationOutcome, RuntimeStreamInputEvent,
    RuntimeStreamScheduler, RuntimeStreamSchedulerBudget, RuntimeStreamSchedulerError,
    RuntimeStreamSchedulerSnapshot, RuntimeStreamStateReservation, RuntimeStreamStatus,
};
use crate::stream_state::{
    TransientStateBlockId, TransientStateBlockShape, TransientStateKey, TransientStateSlot,
};
use crate::tensor_storage::TensorStorage;
use crate::vulkan::{DEFAULT_COMPUTE_LOCAL_SIZE_X, DEFAULT_SPIRV_ENTRY_POINT, read_spirv_words};
use crate::vulkan_compute::{
    VULKAN_RESIDENT_INDIRECT_DISPATCH_BYTE_COUNT, VulkanComputeDevice, VulkanError,
    VulkanResidentBuffer, VulkanResidentBufferCopy, VulkanResidentBufferCopyBatch,
    VulkanResidentBufferRangeCopy, VulkanResidentKernelBufferAccess,
    VulkanResidentKernelBufferBinding, VulkanResidentKernelDispatch, VulkanResidentKernelSequence,
    VulkanResidentKernelSequenceInputCopy, VulkanResidentKernelSequenceSnapshotCopy,
    VulkanResidentKernelSequenceStep, VulkanResidentMappedBufferCopy,
    VulkanResidentQueueSubmissionBatch, VulkanResidentQueueSubmissionTemplate, VulkanShaderFeature,
    VulkanSubgroupOperation, VulkanTimelineSemaphore, VulkanTimelineSemaphorePoint,
    record_vulkan_execution_quantum_measurement, vulkan_spirv_requirements,
};
use crate::vulkan_distributed::{
    VulkanDistributedActivationBufferPlan, VulkanDistributedActivationBuffers,
    VulkanDistributedDispatchDistribution, VulkanDistributedDispatchGroup,
    VulkanDistributedDispatchRunnerError, VulkanDistributedDispatchRunners,
    VulkanDistributedDispatchSubmission, VulkanDistributedExecutionPlan,
    VulkanDistributedParameterAllocationPlan, VulkanDistributedParameterBuffers,
    VulkanDistributedParameterExclusionPlan,
};

mod package;
pub use package::*;

pub const VULKAN_STREAM_CIRCUIT_BACKEND_ID: &str = "vulkan_stream_circuit_ir";
pub const VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA: &str =
    "nerve.vulkan_reusable_kernel_artifacts.v1";
pub const VULKAN_RESIDENT_MODEL_PACKAGE_MANIFEST_SCHEMA: &str =
    "nerve.vulkan_resident_model_package.v3";
pub const VULKAN_PACKAGE_COMPILER_FINGERPRINT: &str = env!("NERVE_PACKAGE_COMPILER_FINGERPRINT");
const CONTRACT_DIGEST_ALGORITHM: &str = "nerve.json_tree_sha256.v1";
const VULKAN_STREAM_CONTROL_BYTE_CAPACITY: usize = 5 * std::mem::size_of::<u32>();
const VULKAN_STREAM_CONTROL_TOKEN_BYTE_CAPACITY: usize = std::mem::size_of::<u32>();
const VULKAN_STREAM_CONTROL_METADATA_OFFSET: usize = VULKAN_STREAM_CONTROL_TOKEN_BYTE_CAPACITY;
const VULKAN_COMPONENT_BATCH_WIDTH_CONTROL_BYTE_CAPACITY: u32 = std::mem::size_of::<u32>() as u32;
const VULKAN_COMPONENT_BATCH_CONTROL_BYTE_CAPACITY: u32 = 4 * std::mem::size_of::<u32>() as u32;
const VULKAN_SAMPLER_HISTORY_RECORD_BYTE_CAPACITY: usize = 4 * std::mem::size_of::<u32>();
pub const VULKAN_BACKEND_LOOP_MAX_WINDOW: usize = 64;
const VULKAN_BACKEND_LOOP_MIN_TRANSACTION_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const VULKAN_BACKEND_LOOP_TRANSACTION_HEAP_FRACTION_DIVISOR: usize = 8;

include!("vulkan_stream_circuit/resident_plan_buffers.rs");
include!("vulkan_stream_circuit/transient_state_pages.rs");
include!("vulkan_stream_circuit/edge_plan.rs");
include!("vulkan_stream_circuit/edge_buffers.rs");
include!("vulkan_stream_circuit/edge_transport.rs");
include!("vulkan_stream_circuit/circuit_binding.rs");
include!("vulkan_stream_circuit/circuit_mount.rs");
include!("vulkan_stream_circuit/input_transducer.rs");
include!("vulkan_stream_circuit/output_transducer.rs");
include!("vulkan_stream_circuit/sampler.rs");
include!("vulkan_stream_circuit/resident_feedback_control.rs");
include!("vulkan_stream_circuit/batched_output_projection.rs");
include!("vulkan_stream_circuit/multi_stream_batch_runner.rs");
include!("vulkan_stream_circuit/single_token_tick.rs");
include!("vulkan_stream_circuit/feedback_loop.rs");
include!("vulkan_stream_circuit/speculative_decode.rs");
include!("vulkan_stream_circuit/state_transaction.rs");
include!("vulkan_stream_circuit/component_batch_buffers.rs");
include!("vulkan_stream_circuit/component_batch_kernel_selection.rs");
include!("vulkan_stream_circuit/component_batch_slice_runner.rs");
include!("vulkan_stream_circuit/component_batch_distributed.rs");
include!("vulkan_stream_circuit/component_batch_temporal.rs");
include!("vulkan_stream_circuit/placed_component_batch_runner.rs");
include!("vulkan_stream_circuit/stream_processor.rs");
include!("vulkan_stream_circuit/token_stream.rs");
include!("vulkan_stream_circuit/token_runtime.rs");
include!("vulkan_stream_circuit/token_engine.rs");
include!("vulkan_stream_circuit/resident_package_slices.rs");
include!("vulkan_stream_circuit/placed_feedback_devices.rs");
include!("vulkan_stream_circuit/placed_model_package_loader.rs");
include!("vulkan_stream_circuit/placed_stream_processor.rs");
include!("vulkan_stream_circuit/placed_prompt_event.rs");
include!("vulkan_stream_circuit/placed_prompt_session.rs");
include!("vulkan_stream_circuit/placed_prompt_stream.rs");
include!("vulkan_stream_circuit/placed_prompt_scheduled_activation.rs");
include!("vulkan_stream_circuit/placed_prompt_engine.rs");
include!("vulkan_stream_circuit/placed_prompt_device.rs");
include!("vulkan_stream_circuit/placed_runtime_error.rs");
include!("vulkan_stream_circuit/resident_model_package.rs");
include!("vulkan_stream_circuit/resident_package_execution_contract.rs");
include!("vulkan_stream_circuit/resident_package_planning.rs");
include!("vulkan_stream_circuit/resident_package_resource_loading.rs");
include!("vulkan_stream_circuit/resident_package_kernel_loading.rs");
include!("vulkan_stream_circuit/token_engine_codec.rs");
include!("vulkan_stream_circuit/mounted_component.rs");
include!("vulkan_stream_circuit/mounted_execution_graph_runner.rs");
include!("vulkan_stream_circuit/dispatch_segment_runner.rs");
include!("vulkan_stream_circuit/stream_tick_execution_plan.rs");
include!("vulkan_stream_circuit/stream_control_bytes.rs");
include!("vulkan_stream_circuit/kernel_interface.rs");
include!("vulkan_stream_circuit/descriptor_resources.rs");
include!("vulkan_stream_circuit/reusable_kernels.rs");
include!("vulkan_stream_circuit/dispatch_binding_plan.rs");
include!("vulkan_stream_circuit/tick_plan.rs");
include!("vulkan_stream_circuit/tick_cursor.rs");
include!("vulkan_stream_circuit/in_process_submission.rs");
include!("vulkan_stream_circuit/placed_tick_execution.rs");
include!("vulkan_stream_circuit/stream_tick_errors.rs");
include!("vulkan_stream_circuit/bound_dispatch.rs");
include!("vulkan_stream_circuit/kernel_descriptor_signature.rs");
include!("vulkan_stream_circuit/circuit_binding_builder.rs");
include!("vulkan_stream_circuit/resident_plan_math.rs");
