fn validate_behavioral_validation_artifact(
    manifest_path: &Path,
    manifest: &VulkanResidentModelPackageManifest,
    raw_manifest: &Value,
) -> io::Result<()> {
    let relative_path = &manifest.behavioral_validation_path;
    if relative_path.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "resident model package does not declare behavioral validation evidence",
        ));
    }
    let relative = Path::new(relative_path);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("behavioral validation path {relative_path:?} must stay inside the package"),
        ));
    }
    let path = manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(relative_path);
    let evidence: Value = serde_json::from_slice(&fs::read(&path)?).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid behavioral validation evidence {}: {error}",
                path.display()
            ),
        )
    })?;
    if evidence.get("schema").and_then(Value::as_str) != Some("nerve.behavioral_validation.v1")
        || evidence.get("status").and_then(Value::as_str) != Some("passed")
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "behavioral validation evidence {} has not passed",
                path.display()
            ),
        ));
    }
    let candidate_kind = evidence
        .get("candidate_kind")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "behavioral validation evidence {} has no candidate kind",
                    path.display()
                ),
            )
        })?;
    if candidate_kind != "exact_reference" && candidate_kind != "approximate" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "behavioral validation evidence {} has unsupported candidate kind {candidate_kind:?}",
                path.display()
            ),
        ));
    }
    if evidence
        .get("candidate_contract_digest_algorithm")
        .and_then(Value::as_str)
        != Some(CONTRACT_DIGEST_ALGORITHM)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "behavioral validation evidence {} uses an unsupported candidate contract digest algorithm",
                path.display()
            ),
        ));
    }
    let source_oracle = evidence
        .get("source_oracle")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "behavioral validation evidence {} has no source oracle contract",
                    path.display()
                ),
            )
        })?;
    if source_oracle
        .get("model_contract_digest")
        .and_then(Value::as_str)
        .is_none_or(|digest| !is_lower_hex_sha256(digest))
        || ["tensor_count", "parameter_count", "byte_count"]
            .iter()
            .any(|field| source_oracle.get(*field).and_then(Value::as_u64).is_none())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "behavioral validation evidence {} has an incomplete source oracle contract",
                path.display()
            ),
        ));
    }

    for mode in ["teacher_forced", "free_running"] {
        let mode_evidence = evidence
            .get(mode)
            .and_then(Value::as_object)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "behavioral validation evidence {} lacks {mode} validation",
                        path.display()
                    ),
                )
            })?;
        if candidate_kind == "exact_reference" {
            if mode_evidence.get("status").and_then(Value::as_str) != Some("not_required") {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "exact candidate evidence {} must mark {mode} validation not_required",
                        path.display()
                    ),
                ));
            }
        } else if mode_evidence.get("status").and_then(Value::as_str) != Some("passed")
            || mode_evidence
                .get("sample_count")
                .and_then(Value::as_u64)
                .is_none_or(|count| count == 0)
            || mode_evidence
                .get("metrics")
                .and_then(Value::as_object)
                .is_none_or(|metrics| {
                    metrics.is_empty()
                        || metrics
                            .values()
                            .any(|value| value.as_f64().is_none_or(|number| !number.is_finite()))
                })
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "approximate candidate evidence {} lacks measured passing {mode} validation",
                    path.display()
                ),
            ));
        }
    }

    let raw_components = raw_manifest
        .pointer("/circuit_graph/components")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "resident model package has no raw circuit graph components",
            )
        })?;
    let mut all_raw_components = raw_components.iter().collect::<Vec<_>>();
    if let Some(decoders) = raw_manifest
        .get("speculative_decoders")
        .and_then(Value::as_array)
    {
        for decoder in decoders {
            let components = decoder
                .pointer("/circuit_graph/components")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "resident model package speculative decoder has no raw circuit graph components",
                    )
                })?;
            all_raw_components.extend(components);
        }
    }
    let mut expected_components = BTreeMap::new();
    for component in all_raw_components {
        let component_id = component
            .get("component_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "resident model package contains a raw circuit without a component id",
                )
            })?;
        let circuit = component.get("circuit").ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("resident model package component {component_id:?} has no raw circuit"),
            )
        })?;
        let node_count = circuit
            .get("nodes")
            .and_then(Value::as_array)
            .map(Vec::len)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("resident model package component {component_id:?} has no raw circuit nodes"),
                )
            })?;
        if expected_components
            .insert(component_id, (node_count, circuit))
            .is_some()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("resident model package repeats raw component {component_id:?}"),
            ));
        }
    }
    let circuits = evidence
        .get("circuits")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "behavioral validation evidence {} has no circuit proofs",
                    path.display()
                ),
            )
        })?;
    let mut proven_components = BTreeSet::new();
    let mut approximate_proof_count = 0usize;
    for circuit in circuits {
        let component_id = circuit
            .get("component_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "behavioral validation evidence {} contains a proof without a component id",
                        path.display()
                    ),
                )
            })?;
        if !proven_components.insert(component_id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "behavioral validation evidence {} repeats component {component_id:?}",
                    path.display()
                ),
            ));
        }
        let (expected_node_count, candidate_circuit) =
            expected_components.get(component_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "behavioral validation evidence {} proves unknown component {component_id:?}",
                        path.display()
                    ),
                )
            })?;
        let expected_contract_digest = json_tree_sha256(candidate_circuit)?;
        let candidate_node_count = circuit
            .get("candidate_node_count")
            .and_then(Value::as_u64)
            .and_then(|count| usize::try_from(count).ok());
        let proof_kind = circuit.get("candidate_kind").and_then(Value::as_str);
        if circuit.get("status").and_then(Value::as_str) != Some("passed")
            || !matches!(proof_kind, Some("exact_reference" | "approximate"))
            || (candidate_kind == "exact_reference" && proof_kind != Some("exact_reference"))
            || candidate_node_count != Some(*expected_node_count)
            || circuit
                .get("candidate_contract_digest")
                .and_then(Value::as_str)
                != Some(expected_contract_digest.as_str())
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "behavioral validation evidence {} has an incomplete or stale proof for component {component_id:?}",
                    path.display()
                ),
            ));
        }
        if proof_kind == Some("approximate") {
            approximate_proof_count += 1;
        } else if circuit.get("source_node_count").and_then(Value::as_u64)
            != circuit
                .get("covered_source_node_count")
                .and_then(Value::as_u64)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "behavioral validation evidence {} does not completely cover component {component_id:?}",
                    path.display()
                ),
            ));
        }
    }
    if candidate_kind == "approximate" && approximate_proof_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "approximate behavioral validation evidence {} contains no approximate component proof",
                path.display()
            ),
        ));
    }
    if proven_components != expected_components.keys().copied().collect() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "behavioral validation evidence {} does not prove every packaged component",
                path.display()
            ),
        ));
    }
    Ok(())
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn json_tree_sha256(value: &Value) -> io::Result<String> {
    let mut digest = Sha256::new();
    update_json_tree_digest(&mut digest, value)?;
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn update_json_tree_digest(digest: &mut Sha256, value: &Value) -> io::Result<()> {
    match value {
        Value::Null => digest.update(b"n"),
        Value::Bool(false) => digest.update(b"f"),
        Value::Bool(true) => digest.update(b"t"),
        Value::Number(number) if number.is_i64() => {
            digest.update(b"i");
            update_digest_length_prefixed(digest, number.as_i64().unwrap().to_string().as_bytes());
        }
        Value::Number(number) if number.is_u64() => {
            digest.update(b"i");
            update_digest_length_prefixed(digest, number.as_u64().unwrap().to_string().as_bytes());
        }
        Value::Number(number) => {
            let number = number.as_f64().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "contract digest cannot encode a non-f64 JSON number",
                )
            })?;
            if !number.is_finite() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "contract digest cannot encode a non-finite number",
                ));
            }
            digest.update(b"d");
            digest.update(number.to_be_bytes());
        }
        Value::String(value) => {
            digest.update(b"s");
            update_digest_length_prefixed(digest, value.as_bytes());
        }
        Value::Array(values) => {
            digest.update(b"l");
            digest.update((values.len() as u64).to_be_bytes());
            for value in values {
                update_json_tree_digest(digest, value)?;
            }
        }
        Value::Object(values) => {
            digest.update(b"o");
            digest.update((values.len() as u64).to_be_bytes());
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                update_json_tree_digest(digest, &Value::String(key.clone()))?;
                update_json_tree_digest(digest, &values[key])?;
            }
        }
    }
    Ok(())
}

fn update_digest_length_prefixed(digest: &mut Sha256, payload: &[u8]) {
    digest.update((payload.len() as u64).to_be_bytes());
    digest.update(payload);
}

fn validate_resident_package_relative_path(label: &str, value: &str) -> io::Result<()> {
    let path = Path::new(value);
    if value.is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir
                    | std::path::Component::RootDir
                    | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("resident model package {label} path {value:?} must stay inside the package"),
        ));
    }
    Ok(())
}

fn validate_resident_package_paths(
    manifest: &VulkanResidentModelPackageManifest,
) -> io::Result<()> {
    for (label, path) in [
        ("tensor index", manifest.tensor_index_path.as_str()),
        (
            "behavioral validation",
            manifest.behavioral_validation_path.as_str(),
        ),
        ("config", manifest.config_path.as_str()),
        ("tokenizer", manifest.tokenizer.path.as_str()),
        (
            "input transducer shader",
            manifest.input_transducer.shader_path.as_str(),
        ),
        (
            "batched input transducer shader",
            manifest.input_transducer.batch_shader_path.as_str(),
        ),
        (
            "output embedding norm shader",
            manifest
                .output_transducer
                .embedding_norm_shader_path
                .as_str(),
        ),
        (
            "batched output embedding norm shader",
            manifest
                .output_transducer
                .embedding_norm_batch_shader_path
                .as_str(),
        ),
        (
            "output projection shader",
            manifest.output_transducer.projection_shader_path.as_str(),
        ),
        (
            "batched output projection shader",
            manifest
                .output_transducer
                .projection_batch_shader_path
                .as_str(),
        ),
    ] {
        validate_resident_package_relative_path(label, path)?;
    }
    for kernel in &manifest.sampler.kernels {
        validate_resident_package_relative_path("sampler kernel shader", &kernel.shader_path)?;
    }
    for path in &manifest.tokenizer.files {
        validate_resident_package_relative_path("tokenizer file", path)?;
    }
    for execution in &manifest.component_executions {
        for kernel in &execution.kernels {
            validate_resident_package_relative_path("component kernel shader", &kernel.shader_path)?;
            for implementation in &kernel.batch_implementations {
                for stage in &implementation.stages {
                    validate_resident_package_relative_path(
                        "component batch implementation stage shader",
                        &stage.shader_path,
                    )?;
                }
            }
        }
    }
    for decoder in &manifest.speculative_decoders {
        validate_resident_package_relative_path(
            "draft output norm shader",
            &decoder.output_transducer.norm_shader_path,
        )?;
        validate_resident_package_relative_path(
            "draft output projection shader",
            &decoder.output_transducer.projection_shader_path,
        )?;
        for execution in &decoder.component_executions {
            for kernel in &execution.kernels {
                validate_resident_package_relative_path(
                    "draft component kernel shader",
                    &kernel.shader_path,
                )?;
                for implementation in &kernel.batch_implementations {
                    for stage in &implementation.stages {
                        validate_resident_package_relative_path(
                            "draft component batch implementation stage shader",
                            &stage.shader_path,
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_resident_package_artifact_integrity(
    manifest_path: &Path,
    manifest: &VulkanResidentModelPackageManifest,
) -> io::Result<()> {
    let integrity = &manifest.artifact_integrity;
    if integrity.schema != "nerve.package_artifact_integrity.v1"
        || integrity.algorithm != "sha256"
        || integrity.files.is_empty()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "resident model package artifact integrity contract is invalid",
        ));
    }
    let tokenizer_files = manifest.tokenizer.files.iter().map(|file| {
        Path::new(&manifest.tokenizer.path)
            .join(file)
            .to_string_lossy()
            .into_owned()
    });
    let kernel_shaders = manifest.component_executions.iter().flat_map(|execution| {
        execution.kernels.iter().flat_map(|kernel| {
            std::iter::once(kernel.shader_path.clone()).chain(
                kernel
                    .batch_implementations
                    .iter()
                    .flat_map(|implementation| {
                        implementation
                            .stages
                            .iter()
                            .map(|stage| stage.shader_path.clone())
                    }),
            )
        })
    });
    let draft_shaders = manifest.speculative_decoders.iter().flat_map(|decoder| {
        decoder
            .component_executions
            .iter()
            .flat_map(|execution| {
                execution.kernels.iter().flat_map(|kernel| {
                    std::iter::once(kernel.shader_path.clone()).chain(
                        kernel
                            .batch_implementations
                            .iter()
                            .flat_map(|implementation| {
                                implementation
                                    .stages
                                    .iter()
                                    .map(|stage| stage.shader_path.clone())
                            }),
                    )
                })
            })
            .chain([
                decoder.output_transducer.norm_shader_path.clone(),
                decoder.output_transducer.projection_shader_path.clone(),
            ])
    });
    let sampler_shaders = manifest
        .sampler
        .kernels
        .iter()
        .map(|kernel| kernel.shader_path.clone());
    let required = [
        manifest.tensor_index_path.clone(),
        manifest.behavioral_validation_path.clone(),
        manifest.config_path.clone(),
        manifest.input_transducer.shader_path.clone(),
        manifest.input_transducer.batch_shader_path.clone(),
        manifest
            .output_transducer
            .embedding_norm_shader_path
            .clone(),
        manifest.output_transducer.projection_shader_path.clone(),
        manifest
            .output_transducer
            .projection_batch_shader_path
            .clone(),
    ]
    .into_iter()
    .chain(tokenizer_files)
    .chain(sampler_shaders)
    .chain(kernel_shaders)
    .chain(draft_shaders)
    .collect::<BTreeSet<_>>();
    if integrity.files.keys().cloned().collect::<BTreeSet<_>>() != required {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "resident model package artifact integrity contract does not cover its declared non-weight artifacts",
        ));
    }

    let package_root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    for (relative_path, contract) in &integrity.files {
        validate_resident_package_relative_path("integrity artifact", relative_path)?;
        if !is_lower_hex_sha256(&contract.sha256) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("resident model package artifact {relative_path:?} has an invalid SHA-256"),
            ));
        }
        let path = package_root.join(relative_path);
        let payload = fs::read(&path).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "failed to read resident model package integrity artifact {}: {error}",
                    path.display()
                ),
            )
        })?;
        let actual_sha256 = Sha256::digest(&payload)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if payload.len() != contract.byte_count || actual_sha256 != contract.sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "resident model package artifact {relative_path:?} does not match its integrity contract"
                ),
            ));
        }
    }
    Ok(())
}

fn resident_package_spirv_requirements<'a>(
    package_root: &Path,
    shader_paths: impl IntoIterator<Item = &'a str>,
) -> io::Result<(
    BTreeSet<VulkanShaderFeature>,
    BTreeSet<VulkanSubgroupOperation>,
)> {
    let mut features = BTreeSet::new();
    let mut subgroup_operations = BTreeSet::new();
    for shader_path in shader_paths {
        let resolved = resolve_resident_model_package_path(package_root, shader_path);
        let words = read_spirv_words(&resolved).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "failed to inspect compiled Vulkan shader {:?}: {error}",
                    resolved
                ),
            )
        })?;
        let requirements = vulkan_spirv_requirements(&words).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "compiled Vulkan shader {:?} has no valid runtime device contract: {error}",
                    resolved
                ),
            )
        })?;
        features.extend(requirements.shader_features);
        subgroup_operations.extend(requirements.subgroup_operations);
    }
    Ok((features, subgroup_operations))
}

fn validate_resident_package_spirv_requirements(
    package_root: &Path,
    manifest: &VulkanResidentModelPackageManifest,
) -> io::Result<()> {
    let executions = manifest.component_executions.iter().chain(
        manifest
            .speculative_decoders
            .iter()
            .flat_map(|decoder| decoder.component_executions.iter()),
    );
    let mut mandatory_shader_paths = BTreeSet::from([
        manifest.input_transducer.shader_path.as_str(),
        manifest.input_transducer.batch_shader_path.as_str(),
        manifest
            .output_transducer
            .embedding_norm_shader_path
            .as_str(),
        manifest
            .output_transducer
            .embedding_norm_batch_shader_path
            .as_str(),
        manifest.output_transducer.projection_shader_path.as_str(),
        manifest
            .output_transducer
            .projection_batch_shader_path
            .as_str(),
    ]);
    mandatory_shader_paths.extend(
        manifest
            .sampler
            .kernels
            .iter()
            .map(|kernel| kernel.shader_path.as_str()),
    );
    for decoder in &manifest.speculative_decoders {
        mandatory_shader_paths.insert(decoder.output_transducer.norm_shader_path.as_str());
        mandatory_shader_paths.insert(decoder.output_transducer.projection_shader_path.as_str());
    }

    for execution in executions {
        for kernel in &execution.kernels {
            mandatory_shader_paths.insert(kernel.shader_path.as_str());
            for implementation in &kernel.batch_implementations {
                let (actual_features, actual_subgroup_operations) =
                    resident_package_spirv_requirements(
                        package_root,
                        implementation
                            .stages
                            .iter()
                            .map(|stage| stage.shader_path.as_str()),
                    )?;
                let declared_features = implementation
                    .device_requirements
                    .vulkan_features
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                let declared_subgroup_operations = implementation
                    .device_requirements
                    .subgroup_operations
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                if declared_features != actual_features
                    || declared_subgroup_operations != actual_subgroup_operations
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "compiled batch implementation {}.{} does not declare the Vulkan requirements of its SPIR-V artifacts",
                            execution.component_id, kernel.node_id
                        ),
                    ));
                }
            }
        }
    }

    let (actual_features, actual_subgroup_operations) =
        resident_package_spirv_requirements(package_root, mandatory_shader_paths)?;
    let declared_features = manifest
        .required_vulkan_features
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let declared_subgroup_operations = manifest
        .required_vulkan_subgroup_operations
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if declared_features != actual_features
        || declared_subgroup_operations != actual_subgroup_operations
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "resident model package {:?} does not declare the Vulkan requirements of its mandatory SPIR-V artifacts",
                manifest.package_id
            ),
        ));
    }
    Ok(())
}

