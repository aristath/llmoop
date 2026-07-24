#[cfg(test)]
pub(crate) fn compile_test_shader_words() -> Option<Vec<u32>> {
    use std::sync::atomic::{AtomicU64, Ordering};

    const SOURCE: &str = r#"#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint values[];
} data;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < data.values.length()) {
        data.values[index] = data.values[index] + 1;
    }
}
"#;

    static SOURCE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let source_id = SOURCE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let source_path = std::env::temp_dir().join(format!(
        "nerve-test-increment-{}-{source_id}.comp",
        std::process::id()
    ));
    std::fs::write(&source_path, SOURCE).ok()?;
    let words = compile_shader_words_from_source_path(&source_path);
    let _ = std::fs::remove_file(source_path);
    words
}

#[cfg(test)]
pub(crate) fn compile_shader_words_from_source(shader_file: &str) -> Option<Vec<u32>> {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let shader_path = manifest_dir.join("shaders").join(shader_file);
    if shader_path.exists() {
        return compile_shader_words_from_source_path(&shader_path);
    }

    let shape = shader_file
        .strip_prefix("linear_bf16_")?
        .strip_suffix(".comp")?;
    let (input_size, output_size) = shape.split_once('x')?;
    if !input_size.bytes().all(|byte| byte.is_ascii_digit())
        || !output_size.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }

    let template = std::fs::read_to_string(
        manifest_dir
            .join("shaders")
            .join("linear_bf16.comp.template"),
    )
    .ok()?;
    let rendered = template
        .replace("{{INPUT_SIZE}}", input_size)
        .replace("{{OUTPUT_SIZE}}", output_size);
    static SOURCE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let source_id = SOURCE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let rendered_path = std::env::temp_dir().join(format!(
        "nerve-linear-{input_size}x{output_size}-{}-{source_id}.comp",
        std::process::id()
    ));
    std::fs::write(&rendered_path, rendered).ok()?;
    let words = compile_shader_words_from_source_path(&rendered_path);
    let _ = std::fs::remove_file(rendered_path);
    words
}

#[cfg(test)]
pub(crate) fn compile_shader_words_from_source_path(shader: &Path) -> Option<Vec<u32>> {
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COMPILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    let compile_id = COMPILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let shader_file = shader
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .unwrap_or("shader");
    let output = std::env::temp_dir().join(format!(
        "nerve-{}-{}-{}.spv",
        shader_file.replace(['/', '.'], "-"),
        std::process::id(),
        compile_id
    ));
    let compiled = if test_command_exists("glslangValidator") {
        Command::new("glslangValidator")
            .arg("-V")
            .arg("--target-env")
            .arg("vulkan1.4")
            .arg(shader)
            .arg("-o")
            .arg(&output)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .ok()?
            .success()
    } else if test_command_exists("glslc") {
        Command::new("glslc")
            .arg("--target-env=vulkan1.4")
            .arg(shader)
            .arg("-o")
            .arg(&output)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .ok()?
            .success()
    } else {
        return None;
    };
    if !compiled {
        return None;
    }
    let bytes = std::fs::read(&output).ok()?;
    let _ = std::fs::remove_file(&output);
    if bytes.len() % 4 != 0 {
        return None;
    }
    let words = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Some(words)
}

#[cfg(test)]
pub(crate) fn compile_test_shader_words_from_source(shader_file: &str) -> Option<Vec<u32>> {
    compile_shader_words_from_source(shader_file)
}

#[cfg(test)]
fn test_command_exists(command: &str) -> bool {
    use std::process::{Command, Stdio};

    Command::new(command)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle as _;

    use super::*;

    fn queue_family(
        queue_flags: vk::QueueFlags,
        queue_count: u32,
    ) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags,
            queue_count,
            ..Default::default()
        }
    }

    #[test]
    fn compute_queue_selection_prefers_a_non_graphics_family() {
        let queue_families = vec![
            queue_family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 1),
            queue_family(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER, 1),
        ];

        assert_eq!(
            preferred_compute_queue_family_indices(&queue_families),
            vec![1, 0]
        );
    }

    #[test]
    fn compute_queue_selection_falls_back_to_a_universal_family() {
        let queue_families = vec![
            queue_family(vk::QueueFlags::TRANSFER, 1),
            queue_family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 1),
        ];

        assert_eq!(
            preferred_compute_queue_family_indices(&queue_families),
            vec![1]
        );
    }

    #[test]
    fn compute_queue_selection_ignores_families_without_queues() {
        let queue_families = vec![
            queue_family(vk::QueueFlags::COMPUTE, 0),
            queue_family(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, 1),
        ];

        assert_eq!(
            preferred_compute_queue_family_indices(&queue_families),
            vec![1]
        );
    }

    fn buffer_access(
        buffer: u64,
        access: VulkanResidentKernelBufferAccess,
    ) -> VulkanResidentKernelBufferAccessRecord {
        VulkanResidentKernelBufferAccessRecord {
            buffer: vk::Buffer::from_raw(buffer),
            access,
        }
    }

    #[test]
    fn semantic_timestamp_labels_expose_component_and_op_fields() {
        let label = "kernel=linear_00 component=block_00 node=attn_qkv op=parallel_linear_2way lane=3";

        assert_eq!(semantic_label_field(label, "component"), Some("block_00"));
        assert_eq!(
            semantic_label_field(label, "op"),
            Some("parallel_linear_2way")
        );
        assert_eq!(semantic_label_field(label, "node"), Some("attn_qkv"));
        assert_eq!(semantic_label_field(label, "missing"), None);
    }

    #[test]
    fn resident_kernel_dependencies_synchronize_only_conflicting_buffers() {
        let mut pending = vec![
            buffer_access(1, VulkanResidentKernelBufferAccess::Write),
            buffer_access(2, VulkanResidentKernelBufferAccess::Read),
        ];
        let current = [
            buffer_access(1, VulkanResidentKernelBufferAccess::Read),
            buffer_access(2, VulkanResidentKernelBufferAccess::Read),
        ];

        let dependencies = take_resident_kernel_buffer_dependencies(&mut pending, &current);

        assert_eq!(
            dependencies,
            vec![VulkanResidentKernelBufferDependency {
                buffer: vk::Buffer::from_raw(1),
            }]
        );
        assert_eq!(
            pending,
            vec![buffer_access(2, VulkanResidentKernelBufferAccess::Read)]
        );
    }

    #[test]
    fn resident_kernel_dependencies_preserve_read_after_read_without_a_barrier() {
        let access = buffer_access(1, VulkanResidentKernelBufferAccess::Read);
        let mut pending = vec![access];

        let dependencies = take_resident_kernel_buffer_dependencies(&mut pending, &[access]);

        assert!(dependencies.is_empty());
        assert_eq!(pending, vec![access]);
    }

    #[test]
    fn resident_kernel_access_merge_coalesces_each_buffer() {
        let mut pending = vec![buffer_access(1, VulkanResidentKernelBufferAccess::Read)];
        merge_resident_kernel_buffer_accesses(
            &mut pending,
            &[
                buffer_access(1, VulkanResidentKernelBufferAccess::Write),
                buffer_access(2, VulkanResidentKernelBufferAccess::Write),
            ],
        );

        assert_eq!(pending.len(), 2);
        assert_eq!(
            pending[0],
            buffer_access(1, VulkanResidentKernelBufferAccess::ReadWrite)
        );
        assert_eq!(
            pending[1],
            buffer_access(2, VulkanResidentKernelBufferAccess::Write)
        );
    }

    fn spirv_test_module(capabilities: &[u32], memory_model: u32) -> Vec<u32> {
        let mut words = vec![SPIRV_MAGIC, 0x0001_0600, 0, 1, 0];
        for capability in capabilities {
            words.extend([(2u32 << 16) | u32::from(SPIRV_OP_CAPABILITY), *capability]);
        }
        words.extend([
            (3u32 << 16) | u32::from(SPIRV_OP_MEMORY_MODEL),
            0,
            memory_model,
        ]);
        words
    }

    #[test]
    fn spirv_contract_extracts_every_feature_used_by_cooperative_bfloat16() {
        let words = spirv_test_module(&[1, 9, 22, 4433, 5116, 5118, 5345, 6022], 3);

        let requirements = vulkan_spirv_requirements(&words).unwrap();

        assert_eq!(
            requirements.shader_features,
            BTreeSet::from([
                VulkanShaderFeature::ShaderFloat16,
                VulkanShaderFeature::ShaderInt16,
                VulkanShaderFeature::StorageBuffer16BitAccess,
                VulkanShaderFeature::ShaderBfloat16Type,
                VulkanShaderFeature::ShaderBfloat16CooperativeMatrix,
                VulkanShaderFeature::VulkanMemoryModel,
                VulkanShaderFeature::CooperativeMatrix,
            ])
        );
    }

    #[test]
    fn spirv_contract_extracts_native_fp8_dot_product_feature() {
        let words = spirv_test_module(&[1, 4212, 6915], 1);

        let requirements = vulkan_spirv_requirements(&words).unwrap();

        assert_eq!(
            requirements.shader_features,
            BTreeSet::from([
                VulkanShaderFeature::ShaderFloat8,
                VulkanShaderFeature::ShaderMixedFloatDotProductFloat8AccFloat32,
            ])
        );
    }

    #[test]
    fn spirv_contract_extracts_native_integer_dot_product_feature() {
        let words = spirv_test_module(&[1, 39, 6018, 6019], 1);

        let requirements = vulkan_spirv_requirements(&words).unwrap();

        assert_eq!(
            requirements.shader_features,
            BTreeSet::from([
                VulkanShaderFeature::ShaderInt8,
                VulkanShaderFeature::ShaderIntegerDotProduct,
            ])
        );
    }

    #[test]
    fn spirv_contract_rejects_missing_device_features_before_gpu_submission() {
        let words = spirv_test_module(&[1, 5345], 3);

        let error = validate_spirv_device_contract(
            &words,
            &BTreeSet::new(),
            vk::ShaderStageFlags::COMPUTE,
            vk::SubgroupFeatureFlags::empty(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            VulkanError(
                "shader artifact requires Vulkan features that were not enabled on the logical device: vulkan_memory_model"
                    .to_string()
            )
        );
    }

    #[test]
    fn spirv_contract_accepts_a_fully_provisioned_device_contract() {
        let words = spirv_test_module(&[1, 61, 63, 5345], 3);

        validate_spirv_device_contract(
            &words,
            &BTreeSet::from([VulkanShaderFeature::VulkanMemoryModel]),
            vk::ShaderStageFlags::COMPUTE,
            vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC,
        )
        .unwrap();
    }

    #[test]
    fn spirv_contract_rejects_unsupported_subgroup_operations() {
        let words = spirv_test_module(&[1, 61, 63], 1);

        let error = validate_spirv_device_contract(
            &words,
            &BTreeSet::new(),
            vk::ShaderStageFlags::COMPUTE,
            vk::SubgroupFeatureFlags::BASIC,
        )
        .unwrap_err();

        assert!(error.0.contains("arithmetic"));
    }

    #[test]
    fn package_capability_names_match_the_compiler_contract() {
        assert_eq!(
            serde_json::to_string(&VulkanShaderFeature::VulkanMemoryModel).unwrap(),
            "\"vulkan_memory_model\""
        );
        assert_eq!(
            serde_json::to_string(&VulkanShaderFeature::StorageBuffer16BitAccess).unwrap(),
            "\"storage_buffer16_bit_access\""
        );
        assert_eq!(
            serde_json::to_string(&VulkanSubgroupOperation::ShuffleRelative).unwrap(),
            "\"shuffle_relative\""
        );
    }

    #[test]
    fn spirv_contract_rejects_inconsistent_memory_model_declarations() {
        let vulkan_without_capability = spirv_test_module(&[1], 3);
        let capability_without_vulkan = spirv_test_module(&[1, 5345], 1);

        assert!(vulkan_spirv_requirements(&vulkan_without_capability).is_err());
        assert!(vulkan_spirv_requirements(&capability_without_vulkan).is_err());
    }

    #[test]
    fn spirv_contract_fails_closed_for_unmodeled_capabilities() {
        let words = spirv_test_module(&[1, 65_535], 1);

        assert_eq!(
            vulkan_spirv_requirements(&words).unwrap_err(),
            VulkanError(
                "shader artifact declares SPIR-V capability 65535, but the runtime has no device contract for it"
                    .to_string()
            )
        );
    }

    #[test]
    fn spirv_contract_rejects_truncated_instructions() {
        let mut words = spirv_test_module(&[1], 1);
        words.push((4u32 << 16) | 54);

        assert!(vulkan_spirv_requirements(&words).is_err());
    }

    #[test]
    fn timeline_replay_offsets_preserve_values_and_reject_overflow() {
        assert_eq!(offset_timeline_value(17, 64).unwrap(), 81);
        assert_eq!(offset_timeline_value(u64::MAX, 0).unwrap(), u64::MAX);
        assert!(offset_timeline_value(u64::MAX, 1).is_err());
    }

    #[test]
    fn cooperative_bfloat16_matrix_shader_preserves_matrix_orientation() {
        let (Some(shader_path), Some(device_index)) = (
            std::env::var_os("NERVE_TEST_COOPERATIVE_BFLOAT16_SHADER"),
            std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
        ) else {
            eprintln!("skipping cooperative BF16 matrix test: explicit shader/device unset");
            return;
        };
        let bytes = std::fs::read(shader_path).unwrap();
        let spirv_words = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
        assert!(device.supports_cooperative_bfloat16_shape(16, 16, 16));
        assert_eq!(device.subgroup_size(), 64);
        assert!(device.supports_compute_local_size_x(256));

        let input_values = (0..256)
            .map(|index| f32_to_bf16_bits((index % 16) as f32 + 1.0))
            .collect::<Vec<_>>();
        let row_major_weight = (0..256)
            .map(|index| {
                let row = index / 16;
                let column = index % 16;
                f32_to_bf16_bits(if row == column { 2.0 } else { 0.0 })
            })
            .collect::<Vec<_>>();
        let input = device.create_resident_buffer(512).unwrap();
        let output = device.create_resident_buffer(512).unwrap();
        let weight = device.create_resident_buffer(512).unwrap();
        input.write_bytes(&u16_bytes(&input_values)).unwrap();
        output.write_bytes(&vec![0; 512]).unwrap();
        weight.write_bytes(&u16_bytes(&row_major_weight)).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[
                    VulkanResidentKernelBufferBinding::new(0, &input, 512),
                    VulkanResidentKernelBufferBinding::new(1, &output, 512),
                    VulkanResidentKernelBufferBinding::new(2, &weight, 512),
                ],
                1,
                256,
                4,
            )
            .unwrap();
        device
            .run_resident_kernel_dispatch(&dispatch, &16u32.to_le_bytes())
            .unwrap();

        let expected = input_values
            .iter()
            .map(|value| f32_to_bf16_bits(bf16_bits_to_f32(*value) * 2.0))
            .collect::<Vec<_>>();
        assert_eq!(output.read_bytes(512).unwrap(), u16_bytes(&expected));
    }

    fn f32_to_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let lsb = (bits >> 16) & 1;
        ((bits + 0x7fff + lsb) >> 16) as u16
    }

    fn bf16_bits_to_f32(value: u16) -> f32 {
        f32::from_bits(u32::from(value) << 16)
    }

    fn u16_bytes(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn u32_bytes(values: &[u32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(std::mem::size_of::<u32>())
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn persistently_mapped_copy_moves_exact_bound_bytes() {
        let source = [1u8, 2, 3, 4, 5, 6];
        let mut destination = [0u8; 6];
        let copy = VulkanResidentMappedBufferCopy {
            source_address: source.as_ptr() as usize,
            destination_address: destination.as_mut_ptr() as usize,
            byte_len: source.len(),
        };

        copy.run(source.len()).unwrap();

        assert_eq!(destination, source);
        assert!(copy.run(source.len() - 1).is_err());
    }

    #[test]
    fn resident_byte_buffer_can_be_reused_for_raw_model_memory() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(16).unwrap();

        buffer.write_bytes(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(buffer.byte_capacity(), 16);
        assert_eq!(buffer.read_bytes(5).unwrap(), vec![1, 2, 3, 4, 5]);

        buffer.write_bytes(&[10, 20, 30]).unwrap();
        assert_eq!(buffer.read_bytes(3).unwrap(), vec![10, 20, 30]);
        assert!(buffer.read_bytes(17).is_err());
        assert!(buffer.write_bytes(&[0; 17]).is_err());
    }

    #[test]
    fn generic_resident_kernel_dispatch_runs_on_raw_byte_buffer() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(12).unwrap();
        buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let binding = VulkanResidentKernelBufferBinding::new(0, &buffer, 12);

        let dispatch = device
            .create_resident_kernel_dispatch(&spirv_words, &[binding], 1, 64, 0)
            .unwrap();
        device.run_resident_kernel_dispatch(&dispatch, &[]).unwrap();

        assert_eq!(dispatch.descriptor_count(), 1);
        assert_eq!(dispatch.workgroup_count_x(), 1);
        assert_eq!(dispatch.push_constant_byte_count(), 0);
        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![2, 3, 42]
        );
    }

    #[test]
    fn resident_kernel_sequence_records_and_replays_composed_dispatches() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(12).unwrap();
        buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let binding = VulkanResidentKernelBufferBinding::new(0, &buffer, 12);
        let dispatch = device
            .create_resident_kernel_dispatch(&spirv_words, &[binding], 1, 64, 0)
            .unwrap();
        let sequence = device.create_resident_kernel_sequence().unwrap();
        assert!(!sequence.has_recorded_commands());
        assert!(
            device
                .run_recorded_resident_kernel_sequence(&sequence)
                .is_err()
        );

        device
            .run_resident_kernel_sequence(
                &sequence,
                &[
                    VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
                    VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
                ],
            )
            .unwrap();
        assert!(sequence.has_recorded_commands());

        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![3, 4, 43]
        );

        device
            .run_recorded_resident_kernel_sequence(&sequence)
            .unwrap();
        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![5, 6, 45]
        );
    }

    #[test]
    fn separate_resident_sequences_publish_compute_writes_to_the_next_sequence() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan sequence boundary test: no GLSL to SPIR-V compiler found");
            return;
        };
        let Some(device_index) = std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        else {
            eprintln!("skipping Vulkan sequence boundary test: explicit Vulkan device index unset");
            return;
        };
        let device = VulkanComputeDevice::new_for_physical_device_index(device_index).unwrap();
        let buffer = device.create_resident_buffer(12).unwrap();
        buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let dispatch = device
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &buffer, 12)],
                1,
                64,
                0,
            )
            .unwrap();
        let producer = device.create_resident_kernel_sequence().unwrap();
        let consumer = device.create_resident_kernel_sequence().unwrap();

        device
            .run_resident_kernel_sequence(
                &producer,
                &[VulkanResidentKernelSequenceStep::new(&dispatch, &[])],
            )
            .unwrap();
        device
            .run_resident_kernel_sequence(
                &consumer,
                &[VulkanResidentKernelSequenceStep::new(&dispatch, &[])],
            )
            .unwrap();

        assert_eq!(
            bytes_to_u32(&buffer.read_bytes(12).unwrap()),
            vec![3, 4, 43]
        );
    }

    #[test]
    fn cross_device_shared_host_memory_reuses_persistent_semaphore_dependencies() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping cross-device Vulkan test: no GLSL to SPIR-V compiler found");
            return;
        };
        let (Some(owner_index), Some(worker_index)) = (
            std::env::var("NERVE_TEST_VULKAN_DEVICE_INDEX")
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
            std::env::var("NERVE_TEST_VULKAN_SECONDARY_DEVICE_INDEX")
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
        ) else {
            eprintln!("skipping cross-device Vulkan test: explicit device pair unset");
            return;
        };
        assert_ne!(owner_index, worker_index);

        let owner = VulkanComputeDevice::new_for_physical_device_index(owner_index).unwrap();
        let worker = VulkanComputeDevice::new_for_physical_device_index(worker_index).unwrap();
        assert!(owner.supports_shared_host_memory());
        assert!(worker.supports_shared_host_memory());
        assert!(owner.supports_opaque_fd_timeline_semaphores());
        assert!(worker.supports_opaque_fd_timeline_semaphores());

        let allocation = owner.create_shared_host_allocation(&[&worker], 12).unwrap();
        let owner_buffer = owner
            .import_shared_host_buffer(Arc::clone(&allocation))
            .unwrap();
        let worker_buffer = worker.import_shared_host_buffer(allocation).unwrap();
        owner_buffer.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();

        let owner_dispatch = owner
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(0, &owner_buffer, 12)],
                1,
                64,
                0,
            )
            .unwrap();
        let worker_dispatch = worker
            .create_resident_kernel_dispatch(
                &spirv_words,
                &[VulkanResidentKernelBufferBinding::new(
                    0,
                    &worker_buffer,
                    12,
                )],
                1,
                64,
                0,
            )
            .unwrap();
        let owner_first = owner.create_resident_kernel_sequence().unwrap();
        owner
            .record_resident_kernel_sequence(
                &owner_first,
                &[VulkanResidentKernelSequenceStep::new(&owner_dispatch, &[])],
            )
            .unwrap();
        let worker_sequence = worker.create_resident_kernel_sequence().unwrap();
        worker
            .record_resident_kernel_sequence(
                &worker_sequence,
                &[VulkanResidentKernelSequenceStep::new(&worker_dispatch, &[])],
            )
            .unwrap();
        let owner_last = owner.create_resident_kernel_sequence().unwrap();
        owner
            .record_resident_kernel_sequence(
                &owner_last,
                &[VulkanResidentKernelSequenceStep::new(&owner_dispatch, &[])],
            )
            .unwrap();

        let ready_source = owner
            .create_opaque_fd_exportable_timeline_semaphore(0)
            .unwrap();
        let ready_wait = worker.create_timeline_semaphore(0).unwrap();
        worker
            .import_timeline_semaphore_opaque_fd(
                &ready_wait,
                owner
                    .export_timeline_semaphore_opaque_fd(&ready_source)
                    .unwrap(),
            )
            .unwrap();
        let done_source = worker
            .create_opaque_fd_exportable_timeline_semaphore(0)
            .unwrap();
        let done_wait = owner.create_timeline_semaphore(0).unwrap();
        owner
            .import_timeline_semaphore_opaque_fd(
                &done_wait,
                worker
                    .export_timeline_semaphore_opaque_fd(&done_source)
                    .unwrap(),
            )
            .unwrap();

        for dependency_value in 1..=2 {
            owner
                .submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    &owner_first,
                    &[],
                    &[VulkanTimelineSemaphorePoint::new(
                        &ready_source,
                        dependency_value,
                    )],
                )
                .unwrap();
            worker
                .submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    &worker_sequence,
                    &[VulkanTimelineSemaphorePoint::new(
                        &ready_wait,
                        dependency_value,
                    )],
                    &[VulkanTimelineSemaphorePoint::new(
                        &done_source,
                        dependency_value,
                    )],
                )
                .unwrap();
            owner
                .submit_recorded_resident_kernel_sequence_with_timeline_semaphores(
                    &owner_last,
                    &[VulkanTimelineSemaphorePoint::new(
                        &done_wait,
                        dependency_value,
                    )],
                    &[],
                )
                .unwrap();
            owner.wait_resident_kernel_sequence(&owner_last).unwrap();
        }

        assert_eq!(
            bytes_to_u32(&owner_buffer.read_bytes(12).unwrap()),
            vec![7, 8, 47]
        );
    }

    #[test]
    fn resident_kernel_sequence_snapshots_state_between_dispatch_groups() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let state = device.create_resident_buffer(12).unwrap();
        state.write_bytes(&u32_bytes(&[1, 2, 41])).unwrap();
        let snapshots = device.create_host_visible_resident_buffer(24).unwrap();
        let binding = VulkanResidentKernelBufferBinding::new(0, &state, 12);
        let dispatch = device
            .create_resident_kernel_dispatch(&spirv_words, &[binding], 1, 64, 0)
            .unwrap();
        let sequence = device.create_resident_kernel_sequence().unwrap();
        let steps = [
            VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
            VulkanResidentKernelSequenceStep::new(&dispatch, &[]),
        ];
        let copies = [
            VulkanResidentKernelSequenceSnapshotCopy::new(0, &state, &snapshots, 0, 0, 12).unwrap(),
            VulkanResidentKernelSequenceSnapshotCopy::new(1, &state, &snapshots, 0, 12, 12)
                .unwrap(),
        ];

        device
            .run_resident_kernel_sequence_with_snapshot_copies(&sequence, &steps, &copies)
            .unwrap();

        assert_eq!(
            bytes_to_u32(&snapshots.read_bytes(24).unwrap()),
            vec![2, 3, 42, 3, 4, 43]
        );
    }

    #[test]
    fn generic_resident_kernel_dispatch_validates_push_constant_size() {
        let Some(spirv_words) = compile_test_shader_words() else {
            eprintln!("skipping Vulkan smoke: no GLSL to SPIR-V compiler found");
            return;
        };
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let buffer = device.create_resident_buffer(4).unwrap();
        buffer.write_bytes(&u32_bytes(&[10])).unwrap();
        let binding = VulkanResidentKernelBufferBinding::new(0, &buffer, 4);
        let dispatch = device
            .create_resident_kernel_dispatch(&spirv_words, &[binding], 1, 64, 4)
            .unwrap();

        let error = device
            .run_resident_kernel_dispatch(&dispatch, &[])
            .unwrap_err();

        assert_eq!(
            error,
            VulkanError(
                "resident kernel sequence step 0 expects 4 push-constant bytes, got 0".to_string()
            )
        );
    }

    #[test]
    fn resident_byte_buffers_can_copy_on_device() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let source = device.create_resident_buffer(8).unwrap();
        let destination = device.create_resident_buffer(8).unwrap();
        source.write_bytes(&[1, 2, 3, 4, 5, 6]).unwrap();
        destination.write_bytes(&[0, 0, 0, 0, 0, 0]).unwrap();

        device
            .copy_resident_buffer_bytes(&source, &destination, 6)
            .unwrap();

        assert_eq!(destination.read_bytes(6).unwrap(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn resident_byte_copy_binding_can_be_reused() {
        let device = match VulkanComputeDevice::new() {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                return;
            }
        };
        let source = device.create_resident_buffer(8).unwrap();
        let destination = device.create_resident_buffer(8).unwrap();
        let binding = device
            .create_resident_buffer_copy(&source, &destination, 6)
            .unwrap();

        source.write_bytes(&[1, 2, 3, 4, 5, 6]).unwrap();
        device.run_resident_buffer_copy(&binding, 6).unwrap();
        assert_eq!(destination.read_bytes(6).unwrap(), vec![1, 2, 3, 4, 5, 6]);

        source.write_bytes(&[10, 20, 30, 40, 50, 60]).unwrap();
        device.run_resident_buffer_copy(&binding, 6).unwrap();
        assert_eq!(
            destination.read_bytes(6).unwrap(),
            vec![10, 20, 30, 40, 50, 60]
        );
        assert_eq!(binding.byte_len(), 6);
    }
}
