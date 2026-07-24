use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::ffi::{CStr, CString, c_void};
use std::fmt::{Display, Formatter};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};
#[cfg(test)]
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

use ash::{Entry, vk};
use serde::{Deserialize, Serialize};

const VK_EXT_SHADER_FLOAT8_NAME: &CStr = c"VK_EXT_shader_float8";
const VK_KHR_SHADER_BFLOAT16_NAME: &CStr = c"VK_KHR_shader_bfloat16";
const VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_NAME: &CStr =
    c"VK_VALVE_shader_mixed_float_dot_product";
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT: i32 = 1_000_567_000;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR: i32 = 1_000_141_000;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE: i32 =
    1_000_673_000;
const VK_COMPONENT_TYPE_BFLOAT16_KHR: i32 = 1_000_141_000;
const VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT: i32 = 1_000_491_002;
const VULKAN_SHARED_HOST_MEMORY_HANDLE_TYPE: vk::ExternalMemoryHandleTypeFlags =
    vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
const VULKAN_PERSISTENT_CROSS_DEVICE_SYNC_HANDLE_TYPE: vk::ExternalSemaphoreHandleTypeFlags =
    vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD;
const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_OP_MEMORY_MODEL: u16 = 14;
const SPIRV_OP_CAPABILITY: u16 = 17;
const SPIRV_MEMORY_MODEL_VULKAN: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanShaderFeature {
    ShaderFloat16,
    ShaderFloat64,
    ShaderInt8,
    ShaderInt16,
    ShaderInt64,
    ShaderIntegerDotProduct,
    StorageBuffer16BitAccess,
    UniformAndStorageBuffer16BitAccess,
    StoragePushConstant16,
    StorageInputOutput16,
    StorageBuffer8BitAccess,
    UniformAndStorageBuffer8BitAccess,
    StoragePushConstant8,
    ShaderFloat8,
    ShaderFloat8CooperativeMatrix,
    ShaderBfloat16Type,
    ShaderBfloat16DotProduct,
    ShaderBfloat16CooperativeMatrix,
    ShaderMixedFloatDotProductBfloat16Acc,
    ShaderMixedFloatDotProductFloat8AccFloat32,
    VulkanMemoryModel,
    VulkanMemoryModelDeviceScope,
    CooperativeMatrix,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VulkanSubgroupOperation {
    Basic,
    Vote,
    Arithmetic,
    Ballot,
    Shuffle,
    ShuffleRelative,
    Clustered,
    Quad,
}

impl VulkanSubgroupOperation {
    pub fn label(self) -> &'static str {
        match self {
            Self::Basic => "basic",
            Self::Vote => "vote",
            Self::Arithmetic => "arithmetic",
            Self::Ballot => "ballot",
            Self::Shuffle => "shuffle",
            Self::ShuffleRelative => "shuffle_relative",
            Self::Clustered => "clustered",
            Self::Quad => "quad",
        }
    }

    fn flag(self) -> vk::SubgroupFeatureFlags {
        match self {
            Self::Basic => vk::SubgroupFeatureFlags::BASIC,
            Self::Vote => vk::SubgroupFeatureFlags::VOTE,
            Self::Arithmetic => vk::SubgroupFeatureFlags::ARITHMETIC,
            Self::Ballot => vk::SubgroupFeatureFlags::BALLOT,
            Self::Shuffle => vk::SubgroupFeatureFlags::SHUFFLE,
            Self::ShuffleRelative => vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE,
            Self::Clustered => vk::SubgroupFeatureFlags::CLUSTERED,
            Self::Quad => vk::SubgroupFeatureFlags::QUAD,
        }
    }
}

impl VulkanShaderFeature {
    pub fn label(self) -> &'static str {
        match self {
            Self::ShaderFloat16 => "shader_float16",
            Self::ShaderFloat64 => "shader_float64",
            Self::ShaderInt8 => "shader_int8",
            Self::ShaderInt16 => "shader_int16",
            Self::ShaderInt64 => "shader_int64",
            Self::ShaderIntegerDotProduct => "shader_integer_dot_product",
            Self::StorageBuffer16BitAccess => "storage_buffer16_bit_access",
            Self::UniformAndStorageBuffer16BitAccess => "uniform_and_storage_buffer16_bit_access",
            Self::StoragePushConstant16 => "storage_push_constant16",
            Self::StorageInputOutput16 => "storage_input_output16",
            Self::StorageBuffer8BitAccess => "storage_buffer8_bit_access",
            Self::UniformAndStorageBuffer8BitAccess => "uniform_and_storage_buffer8_bit_access",
            Self::StoragePushConstant8 => "storage_push_constant8",
            Self::ShaderFloat8 => "shader_float8",
            Self::ShaderFloat8CooperativeMatrix => "shader_float8_cooperative_matrix",
            Self::ShaderBfloat16Type => "shader_bfloat16_type",
            Self::ShaderBfloat16DotProduct => "shader_bfloat16_dot_product",
            Self::ShaderBfloat16CooperativeMatrix => "shader_bfloat16_cooperative_matrix",
            Self::ShaderMixedFloatDotProductBfloat16Acc => {
                "shader_mixed_float_dot_product_bfloat16_acc"
            }
            Self::ShaderMixedFloatDotProductFloat8AccFloat32 => {
                "shader_mixed_float_dot_product_float8_acc_float32"
            }
            Self::VulkanMemoryModel => "vulkan_memory_model",
            Self::VulkanMemoryModelDeviceScope => "vulkan_memory_model_device_scope",
            Self::CooperativeMatrix => "cooperative_matrix",
        }
    }
}

impl Display for VulkanShaderFeature {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct VulkanSpirvRequirements {
    pub shader_features: BTreeSet<VulkanShaderFeature>,
    pub subgroup_operations: BTreeSet<VulkanSubgroupOperation>,
    uses_subgroups: bool,
    vulkan_memory_model_capability: bool,
    vulkan_memory_model_device_scope_capability: bool,
    memory_model: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct VulkanShaderFloat8Support {
    shader_float8: bool,
    shader_float8_cooperative_matrix: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct VulkanShaderBfloat16Support {
    shader_bfloat16_type: bool,
    shader_bfloat16_dot_product: bool,
    shader_bfloat16_cooperative_matrix: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct VulkanShaderMixedFloatDotProductSupport {
    shader_bfloat16_acc: bool,
    shader_float8_acc_float32: bool,
}

// ash 0.38 is generated from Vulkan 1.3.281 headers, while shader-float8 was
// added later. Keep this ABI-compatible definition local until ash publishes
// bindings generated from current Vulkan headers.
#[repr(C)]
struct VulkanPhysicalDeviceShaderFloat8FeaturesExt {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    shader_float8: vk::Bool32,
    shader_float8_cooperative_matrix: vk::Bool32,
}

impl VulkanPhysicalDeviceShaderFloat8FeaturesExt {
    fn disabled() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT,
            ),
            p_next: std::ptr::null_mut(),
            shader_float8: vk::FALSE,
            shader_float8_cooperative_matrix: vk::FALSE,
        }
    }
}

// VK_KHR_shader_bfloat16 was published after the Vulkan headers used by the
// latest ash release. This definition mirrors the current Vulkan 1.4 ABI.
#[repr(C)]
struct VulkanPhysicalDeviceShaderBfloat16FeaturesKhr {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    shader_bfloat16_type: vk::Bool32,
    shader_bfloat16_dot_product: vk::Bool32,
    shader_bfloat16_cooperative_matrix: vk::Bool32,
}

impl VulkanPhysicalDeviceShaderBfloat16FeaturesKhr {
    fn disabled() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
            ),
            p_next: std::ptr::null_mut(),
            shader_bfloat16_type: vk::FALSE,
            shader_bfloat16_dot_product: vk::FALSE,
            shader_bfloat16_cooperative_matrix: vk::FALSE,
        }
    }
}

// VK_VALVE_shader_mixed_float_dot_product is newer than the Vulkan headers
// used by the latest ash release. Keep the current extension ABI local until
// upstream bindings include it.
#[repr(C)]
struct VulkanPhysicalDeviceShaderMixedFloatDotProductFeaturesValve {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    shader_float16_acc_float32: vk::Bool32,
    shader_float16_acc_float16: vk::Bool32,
    shader_bfloat16_acc: vk::Bool32,
    shader_float8_acc_float32: vk::Bool32,
}

impl VulkanPhysicalDeviceShaderMixedFloatDotProductFeaturesValve {
    fn disabled() -> Self {
        Self {
            s_type: vk::StructureType::from_raw(
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MIXED_FLOAT_DOT_PRODUCT_FEATURES_VALVE,
            ),
            p_next: std::ptr::null_mut(),
            shader_float16_acc_float32: vk::FALSE,
            shader_float16_acc_float16: vk::FALSE,
            shader_bfloat16_acc: vk::FALSE,
            shader_float8_acc_float32: vk::FALSE,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanError(pub String);

impl Display for VulkanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for VulkanError {}
