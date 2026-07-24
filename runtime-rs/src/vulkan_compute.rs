use crate::execution_schedule::{
    RuntimeExecutionQuantum, RuntimeExecutionQuantumBudget, RuntimeExecutionQuantumCalibrator,
    RuntimeExecutionRegion, RuntimeExecutionSchedule,
};

include!("vulkan_compute/features.rs");
include!("vulkan_compute/device_types.rs");
include!("vulkan_compute/resident_buffers.rs");
include!("vulkan_compute/kernel_sequence.rs");
include!("vulkan_compute/buffer_copies.rs");
include!("vulkan_compute/device_catalog.rs");
include!("vulkan_compute/compute_device_construction.rs");
include!("vulkan_compute/compute_device_memory.rs");
include!("vulkan_compute/compute_device_dispatch.rs");
include!("vulkan_compute/compute_device_sequence.rs");
include!("vulkan_compute/compute_device_pipelines.rs");
include!("vulkan_compute/physical_device_capabilities.rs");
include!("vulkan_compute/tests.rs");
