fn resident_package_reusable_kernel_manifest(
    placed_plan: &VulkanPlacedStreamCircuitPlan,
) -> VulkanReusableKernelArtifactManifest {
    VulkanReusableKernelArtifactManifest::new(
        placed_plan
            .reusable_kernel_plan
            .families
            .iter()
            .map(|family| {
                VulkanReusableKernelArtifact::from_family(
                    family,
                    format!("kernels/{}.spv", family.family_id),
                )
            })
            .collect(),
    )
}

fn resident_package_loaded_kernel_manifest_for_slice_plans(
    slice_plans: &[VulkanResidentModelPackageDeviceSlicePlan],
) -> Result<VulkanLoadedReusableKernelArtifactManifest, VulkanResidentTokenModelPackageError> {
    let mut artifacts_by_family = BTreeMap::<String, VulkanLoadedReusableKernelArtifact>::new();
    for slice in slice_plans {
        for artifact in &slice.loaded_manifest.artifacts {
            if let Some(existing) = artifacts_by_family.get(&artifact.artifact.family_id) {
                let mut existing_contract = existing.artifact.clone();
                existing_contract.path.clear();
                let mut candidate_contract = artifact.artifact.clone();
                candidate_contract.path.clear();
                if existing_contract != candidate_contract || existing.words != artifact.words {
                    return Err(VulkanResidentTokenModelPackageError::new(format!(
                        "loaded reusable Vulkan family {:?} conflicts across device slices",
                        artifact.artifact.family_id
                    )));
                }
            } else {
                artifacts_by_family.insert(artifact.artifact.family_id.clone(), artifact.clone());
            }
        }
    }
    let artifacts = artifacts_by_family.into_values().collect::<Vec<_>>();
    let total_word_count = artifacts.iter().try_fold(0usize, |total, artifact| {
        total.checked_add(artifact.words.len()).ok_or_else(|| {
            VulkanResidentTokenModelPackageError::new(
                "combined reusable Vulkan kernel word count overflowed",
            )
        })
    })?;
    Ok(VulkanLoadedReusableKernelArtifactManifest {
        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        artifacts,
        total_word_count,
    })
}

fn resident_package_component_kernel_shader_refs(
    component_executions: &[VulkanResidentComponentExecutionSpec],
) -> Vec<VulkanResidentComponentKernelShaderRef> {
    component_executions
        .iter()
        .flat_map(|component| {
            component
                .kernels
                .iter()
                .map(|kernel| VulkanResidentComponentKernelShaderRef {
                    component_id: component.component_id.clone(),
                    node_id: kernel.node_id.clone(),
                    shader_path: kernel.shader_path.clone(),
                    local_size_x: kernel.local_size_x,
                    workgroup_count_x: kernel.workgroup_count_x,
                })
        })
        .collect()
}

fn resident_package_component_kernel_shader_refs_for_prepared_dispatches(
    component_executions: &[VulkanResidentComponentExecutionSpec],
    prepared_plan: &VulkanPreparedDispatchPlan,
) -> Vec<VulkanResidentComponentKernelShaderRef> {
    resident_package_component_kernel_shader_refs(component_executions)
        .into_iter()
        .filter(|shader| {
            prepared_plan
                .dispatch(&shader.component_id, &shader.node_id)
                .is_some()
        })
        .collect()
}

fn loaded_kernel_pack_from_package_shader_refs(
    manifest_dir: &Path,
    placed_plan: &VulkanPlacedStreamCircuitPlan,
    prepared_plan: &VulkanPreparedDispatchPlan,
    dispatch_shaders: &[VulkanResidentComponentKernelShaderRef],
) -> Result<VulkanLoadedReusableKernelArtifactManifest, VulkanResidentTokenModelPackageError> {
    let mut loaded_artifacts = Vec::new();
    let mut loaded_families = BTreeSet::new();
    let mut total_word_count = 0usize;

    for shader in dispatch_shaders {
        let dispatch = prepared_plan
            .dispatch(&shader.component_id, &shader.node_id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "mounted dispatch {}.{} declared by resident model package is missing",
                    shader.component_id, shader.node_id
                ))
            })?;
        if !loaded_families.insert(dispatch.reusable_family_id.clone()) {
            continue;
        }
        let spirv_words =
            load_required_resident_model_package_shader(manifest_dir, &shader.shader_path)?;
        total_word_count = total_word_count
            .checked_add(spirv_words.len())
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(
                    "reusable kernel artifact word count overflowed",
                )
            })?;
        let family = placed_plan
            .reusable_kernel_plan
            .family(&dispatch.reusable_family_id)
            .ok_or_else(|| {
                VulkanResidentTokenModelPackageError::new(format!(
                    "reusable kernel family {:?} declared by mounted dispatch {}.{} is missing",
                    dispatch.reusable_family_id, shader.component_id, shader.node_id
                ))
            })?;
        loaded_artifacts.push(VulkanLoadedReusableKernelArtifact {
            artifact: VulkanReusableKernelArtifact::from_family(family, shader.shader_path.clone())
                .with_local_size_x(shader.local_size_x)
                .with_workgroup_count_x(shader.workgroup_count_x),
            resolved_path: resolve_resident_model_package_path(manifest_dir, &shader.shader_path),
            words: spirv_words,
        });
    }

    let required_families: BTreeSet<&str> = placed_plan
        .reusable_kernel_plan
        .families
        .iter()
        .map(|family| family.family_id.as_str())
        .collect();
    let loaded_family_ids: BTreeSet<&str> = loaded_artifacts
        .iter()
        .map(|artifact| artifact.artifact.family_id.as_str())
        .collect();
    let missing_families = required_families
        .difference(&loaded_family_ids)
        .copied()
        .collect::<Vec<_>>();
    if !missing_families.is_empty() {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package is missing shaders for reusable kernel families: {}",
            missing_families.join(", ")
        )));
    }

    Ok(VulkanLoadedReusableKernelArtifactManifest {
        schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
        backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
        artifacts: loaded_artifacts,
        total_word_count,
    })
}

fn load_resident_component_batch_kernels(
    device: &VulkanComputeDevice,
    manifest_dir: &Path,
    component_executions: &[VulkanResidentComponentExecutionSpec],
    prepared_plan: &VulkanPreparedDispatchPlan,
) -> Result<Vec<VulkanResidentComponentBatchKernelArtifact>, VulkanResidentTokenModelPackageError> {
    let mut artifacts = Vec::new();
    for component in component_executions {
        for kernel in &component.kernels {
            if !matches!(
                kernel.batch_mode,
                VulkanResidentComponentKernelBatchMode::WeightShared
                    | VulkanResidentComponentKernelBatchMode::CausalScan
            ) || prepared_plan
                .dispatch(&component.component_id, &kernel.node_id)
                .is_none()
            {
                continue;
            }
            let supported = kernel
                .batch_implementations
                .iter()
                .filter(|implementation| batch_implementation_is_supported(device, implementation))
                .collect::<Vec<_>>();
            if supported.is_empty() {
                return Err(VulkanResidentTokenModelPackageError::new(format!(
                    "component kernel {}.{} has no batch implementation compatible with Vulkan device {:?}",
                    component.component_id,
                    kernel.node_id,
                    device.device_name(),
                )));
            }
            for implementation in supported {
                artifacts.push(VulkanResidentComponentBatchKernelArtifact {
                    component_id: component.component_id.clone(),
                    node_id: kernel.node_id.clone(),
                    execution_domain: implementation.execution_domain,
                    batch_mode: kernel.batch_mode,
                    lane_tile_width: implementation.lane_tile_width as usize,
                    independent_candidate_compatible: implementation
                        .independent_candidate_compatible,
                    causal_sequence_compatible: implementation.causal_sequence_compatible,
                    device_requirements: implementation.device_requirements.clone(),
                    stages: implementation
                        .stages
                        .iter()
                        .map(|stage| {
                            Ok(VulkanResidentComponentBatchStageArtifact {
                                shader_path: stage.shader_path.clone(),
                                spirv_words: load_required_resident_model_package_shader(
                                    manifest_dir,
                                    &stage.shader_path,
                                )?,
                                local_size_x: stage.local_size_x,
                                workgroup_count_x: stage.workgroup_count_x,
                            })
                        })
                        .collect::<Result<Vec<_>, VulkanResidentTokenModelPackageError>>()?,
                });
            }
        }
    }
    Ok(artifacts)
}

fn batch_implementation_is_supported(
    device: &VulkanComputeDevice,
    implementation: &VulkanResidentComponentBatchImplementationSpec,
) -> bool {
    batch_device_requirements_are_supported(
        device,
        &implementation.device_requirements,
        implementation.stages.iter().map(|stage| stage.local_size_x),
    )
}

fn batch_kernel_artifact_is_supported(
    device: &VulkanComputeDevice,
    artifact: &VulkanResidentComponentBatchKernelArtifact,
) -> bool {
    batch_device_requirements_are_supported(
        device,
        &artifact.device_requirements,
        artifact.stages.iter().map(|stage| stage.local_size_x),
    )
}

fn batch_device_requirements_are_supported(
    device: &VulkanComputeDevice,
    requirements: &VulkanResidentVulkanDeviceRequirements,
    local_size_x_values: impl IntoIterator<Item = u32>,
) -> bool {
    local_size_x_values
        .into_iter()
        .all(|local_size_x| device.supports_compute_local_size_x(local_size_x))
        && requirements
            .vulkan_device_extensions
            .iter()
            .all(|extension| device.has_enabled_device_extension(extension))
        && requirements
            .vulkan_features
            .iter()
            .all(|feature| device.has_enabled_shader_feature(*feature))
        && requirements
            .subgroup_operations
            .iter()
            .all(|operation| device.supports_subgroup_operation(*operation))
        && requirements
            .cooperative_bfloat16_shape
            .is_none_or(|[m, n, k]| device.supports_cooperative_bfloat16_shape(m, n, k))
        && requirements
            .cooperative_float8_e4m3_shape
            .is_none_or(|[m, n, k]| device.supports_cooperative_float8_e4m3_shape(m, n, k))
        && requirements
            .subgroup_size
            .is_none_or(|subgroup_size| device.subgroup_size() == subgroup_size)
}

fn load_required_resident_model_package_shader(
    manifest_dir: &Path,
    shader_path: &str,
) -> Result<Vec<u32>, VulkanResidentTokenModelPackageError> {
    let resolved_path = resolve_resident_model_package_path(manifest_dir, shader_path);
    if resolved_path
        .extension()
        .and_then(|extension| extension.to_str())
        != Some("spv")
    {
        return Err(VulkanResidentTokenModelPackageError::new(format!(
            "resident model package shader {:?} is not a compiled SPIR-V artifact",
            resolved_path
        )));
    }
    read_spirv_words(&resolved_path).map_err(|error| {
        VulkanResidentTokenModelPackageError::new(format!(
            "failed to load compiled Vulkan shader {:?}: {error}",
            resolved_path
        ))
    })
}

fn load_resident_sampler_kernels(
    manifest_dir: &Path,
    package: &VulkanResidentSamplerPackageSpec,
) -> Result<Vec<VulkanResidentSamplerKernelArtifact>, VulkanResidentTokenModelPackageError> {
    package
        .kernels
        .iter()
        .map(|kernel| {
            Ok(VulkanResidentSamplerKernelArtifact {
                role: kernel.role.clone(),
                spirv_words: load_required_resident_model_package_shader(
                    manifest_dir,
                    &kernel.shader_path,
                )?,
                local_size_x: kernel.local_size_x,
                workgroup_count_x: kernel.workgroup_count_x,
            })
        })
        .collect()
}
