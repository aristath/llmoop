#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanReusableKernelPlan {
    pub backend_id: String,
    pub total_command_count: usize,
    pub families: Vec<VulkanReusableKernelFamily>,
}

impl VulkanReusableKernelPlan {
    pub fn from_dispatch_plan(dispatch_plan: &VulkanKernelDispatchPlan) -> Self {
        let mut grouped: BTreeMap<VulkanReusableKernelKey, Vec<VulkanKernelDispatchRef>> =
            BTreeMap::new();

        for command in &dispatch_plan.commands {
            grouped
                .entry(VulkanReusableKernelKey::from_command(command))
                .or_default()
                .push(VulkanKernelDispatchRef::from_command(command));
        }

        let families = grouped
            .into_iter()
            .map(|(key, command_refs)| VulkanReusableKernelFamily {
                family_id: key.family_id(),
                op: key.op,
                descriptor_signature: key.descriptor_signature,
                push_constants: key.push_constants,
                uses_stream_tick: key.uses_stream_tick,
                command_refs,
            })
            .collect();

        Self {
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            total_command_count: dispatch_plan.total_dispatch_count(),
            families,
        }
    }

    pub fn total_family_count(&self) -> usize {
        self.families.len()
    }

    pub fn reusable_family_count(&self) -> usize {
        self.families
            .iter()
            .filter(|family| family.command_refs.len() > 1)
            .count()
    }

    pub fn family(&self, family_id: &str) -> Option<&VulkanReusableKernelFamily> {
        self.families
            .iter()
            .find(|family| family.family_id == family_id)
    }

    pub fn families_for_op(&self, op: &str) -> Vec<&VulkanReusableKernelFamily> {
        self.families
            .iter()
            .filter(|family| family.op == op)
            .collect()
    }

    pub fn coverage_report<I, S>(
        &self,
        available_family_ids: I,
    ) -> VulkanReusableKernelCoverageReport
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let available_family_ids: BTreeSet<String> = available_family_ids
            .into_iter()
            .map(|id| id.as_ref().to_string())
            .collect();
        let mut families = Vec::with_capacity(self.families.len());
        let mut available_family_count = 0usize;
        let mut covered_command_count = 0usize;

        for family in &self.families {
            let available = available_family_ids.contains(&family.family_id);
            if available {
                available_family_count += 1;
                covered_command_count += family.command_refs.len();
            }
            families.push(VulkanReusableKernelFamilyCoverage {
                family_id: family.family_id.clone(),
                op: family.op.clone(),
                command_count: family.command_refs.len(),
                available,
            });
        }

        VulkanReusableKernelCoverageReport {
            backend_id: self.backend_id.clone(),
            required_family_count: self.families.len(),
            available_family_count,
            missing_family_count: self.families.len() - available_family_count,
            required_command_count: self.total_command_count,
            covered_command_count,
            missing_command_count: self.total_command_count - covered_command_count,
            families,
        }
    }

    pub fn link_artifacts(
        &self,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> VulkanLinkedReusableKernelPlan {
        VulkanLinkedReusableKernelPlan::from_reusable_plan_and_manifest(self, manifest)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanReusableKernelFamily {
    pub family_id: String,
    pub op: String,
    pub descriptor_signature: Vec<VulkanKernelDescriptorSlotSignature>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub uses_stream_tick: bool,
    pub command_refs: Vec<VulkanKernelDispatchRef>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanReusableKernelCoverageReport {
    pub backend_id: String,
    pub required_family_count: usize,
    pub available_family_count: usize,
    pub missing_family_count: usize,
    pub required_command_count: usize,
    pub covered_command_count: usize,
    pub missing_command_count: usize,
    pub families: Vec<VulkanReusableKernelFamilyCoverage>,
}

impl VulkanReusableKernelCoverageReport {
    pub fn all_available(&self) -> bool {
        self.missing_family_count == 0
    }

    pub fn missing_families(&self) -> Vec<&VulkanReusableKernelFamilyCoverage> {
        self.families
            .iter()
            .filter(|family| !family.available)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanReusableKernelFamilyCoverage {
    pub family_id: String,
    pub op: String,
    pub command_count: usize,
    pub available: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanReusableKernelArtifactManifest {
    pub schema: String,
    pub backend_id: String,
    pub artifacts: Vec<VulkanReusableKernelArtifact>,
}

impl VulkanReusableKernelArtifactManifest {
    pub fn new(artifacts: Vec<VulkanReusableKernelArtifact>) -> Self {
        Self {
            schema: VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA.to_string(),
            backend_id: VULKAN_STREAM_CIRCUIT_BACKEND_ID.to_string(),
            artifacts,
        }
    }

    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn with_artifact(mut self, artifact: VulkanReusableKernelArtifact) -> Self {
        self.artifacts.push(artifact);
        self
    }

    pub fn family_ids(&self) -> Vec<&str> {
        self.artifacts
            .iter()
            .map(|artifact| artifact.family_id.as_str())
            .collect()
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let manifest: Self = serde_json::from_slice(&bytes)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if manifest.schema != VULKAN_REUSABLE_KERNEL_ARTIFACT_MANIFEST_SCHEMA {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported reusable kernel manifest schema {:?}",
                    manifest.schema
                ),
            ));
        }
        Ok(manifest)
    }

    pub fn write_json_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let bytes = serde_json::to_vec_pretty(self)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        fs::write(path, bytes)
    }

    pub fn load_artifacts(
        &self,
        artifact_root: impl AsRef<Path>,
    ) -> io::Result<VulkanLoadedReusableKernelArtifactManifest> {
        VulkanLoadedReusableKernelArtifactManifest::from_manifest(self, artifact_root)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanReusableKernelArtifact {
    pub family_id: String,
    pub op: String,
    pub path: String,
    pub entry_point: String,
    pub local_size_x: u32,
    pub workgroup_count_x: u32,
    pub descriptor_signature: Vec<VulkanKernelDescriptorSlotSignature>,
    pub push_constants: Vec<VulkanKernelScalarBinding>,
    pub uses_stream_tick: bool,
}

impl VulkanReusableKernelArtifact {
    pub fn from_family(family: &VulkanReusableKernelFamily, path: impl Into<String>) -> Self {
        Self {
            family_id: family.family_id.clone(),
            op: family.op.clone(),
            path: path.into(),
            entry_point: DEFAULT_SPIRV_ENTRY_POINT.to_string(),
            local_size_x: DEFAULT_COMPUTE_LOCAL_SIZE_X,
            workgroup_count_x: 1,
            descriptor_signature: family.descriptor_signature.clone(),
            push_constants: family.push_constants.clone(),
            uses_stream_tick: family.uses_stream_tick,
        }
    }

    pub fn with_entry_point(mut self, entry_point: impl Into<String>) -> Self {
        self.entry_point = entry_point.into();
        self
    }

    pub fn with_local_size_x(mut self, local_size_x: u32) -> Self {
        self.local_size_x = local_size_x;
        self
    }

    pub fn with_workgroup_count_x(mut self, workgroup_count_x: u32) -> Self {
        self.workgroup_count_x = workgroup_count_x;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanLoadedReusableKernelArtifactManifest {
    pub schema: String,
    pub backend_id: String,
    pub artifacts: Vec<VulkanLoadedReusableKernelArtifact>,
    pub total_word_count: usize,
}

impl VulkanLoadedReusableKernelArtifactManifest {
    pub fn from_manifest(
        manifest: &VulkanReusableKernelArtifactManifest,
        artifact_root: impl AsRef<Path>,
    ) -> io::Result<Self> {
        let artifact_root = artifact_root.as_ref();
        let mut artifacts = Vec::with_capacity(manifest.artifacts.len());
        let mut total_word_count = 0usize;

        for artifact in &manifest.artifacts {
            let resolved_path =
                resolve_reusable_kernel_artifact_path(artifact_root, &artifact.path);
            let words = read_spirv_words(&resolved_path)?;
            total_word_count = total_word_count.checked_add(words.len()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "loaded reusable kernel word count overflowed",
                )
            })?;
            artifacts.push(VulkanLoadedReusableKernelArtifact {
                artifact: artifact.clone(),
                resolved_path,
                words,
            });
        }

        Ok(Self {
            schema: manifest.schema.clone(),
            backend_id: manifest.backend_id.clone(),
            artifacts,
            total_word_count,
        })
    }

    pub fn artifact(&self, family_id: &str) -> Option<&VulkanLoadedReusableKernelArtifact> {
        self.artifacts
            .iter()
            .find(|artifact| artifact.artifact.family_id == family_id)
    }

    pub fn family_ids(&self) -> Vec<&str> {
        self.artifacts
            .iter()
            .map(|artifact| artifact.artifact.family_id.as_str())
            .collect()
    }

    pub fn artifact_manifest(&self) -> VulkanReusableKernelArtifactManifest {
        VulkanReusableKernelArtifactManifest::new(
            self.artifacts
                .iter()
                .map(|loaded| loaded.artifact.clone())
                .collect(),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanLoadedReusableKernelArtifact {
    pub artifact: VulkanReusableKernelArtifact,
    pub resolved_path: PathBuf,
    pub words: Vec<u32>,
}

fn resolve_reusable_kernel_artifact_path(artifact_root: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        artifact_root.join(path)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanLinkedReusableKernelPlan {
    pub backend_id: String,
    pub manifest_schema: String,
    pub manifest_backend_id: String,
    pub required_family_count: usize,
    pub linked_family_count: usize,
    pub missing_family_count: usize,
    pub incompatible_family_count: usize,
    pub required_command_count: usize,
    pub linked_command_count: usize,
    pub missing_command_count: usize,
    pub incompatible_command_count: usize,
    pub families: Vec<VulkanLinkedReusableKernelFamily>,
    pub issues: Vec<VulkanReusableKernelLinkIssue>,
}

impl VulkanLinkedReusableKernelPlan {
    pub fn from_reusable_plan_and_manifest(
        reusable_plan: &VulkanReusableKernelPlan,
        manifest: &VulkanReusableKernelArtifactManifest,
    ) -> Self {
        let mut artifacts_by_family_id: BTreeMap<&str, Vec<&VulkanReusableKernelArtifact>> =
            BTreeMap::new();
        for artifact in &manifest.artifacts {
            artifacts_by_family_id
                .entry(artifact.family_id.as_str())
                .or_default()
                .push(artifact);
        }

        let mut families = Vec::with_capacity(reusable_plan.families.len());
        let mut issues = Vec::new();
        let mut linked_family_count = 0usize;
        let mut missing_family_count = 0usize;
        let mut incompatible_family_count = 0usize;
        let mut linked_command_count = 0usize;
        let mut missing_command_count = 0usize;
        let mut incompatible_command_count = 0usize;

        for family in &reusable_plan.families {
            let command_count = family.command_refs.len();
            let artifacts = artifacts_by_family_id
                .get(family.family_id.as_str())
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let mut family_issues = Vec::new();

            if artifacts.is_empty() {
                family_issues.push(VulkanReusableKernelLinkIssue {
                    family_id: family.family_id.clone(),
                    op: family.op.clone(),
                    problem: VulkanReusableKernelLinkProblem::MissingArtifact,
                });
            } else if artifacts.len() > 1 {
                family_issues.push(VulkanReusableKernelLinkIssue {
                    family_id: family.family_id.clone(),
                    op: family.op.clone(),
                    problem: VulkanReusableKernelLinkProblem::DuplicateArtifact {
                        count: artifacts.len(),
                    },
                });
            }

            let artifact = artifacts.first().copied();
            if let Some(artifact) = artifact {
                family_issues.extend(link_compatibility_issues(family, artifact));
            }

            let (status, artifact_path) = if artifacts.is_empty() {
                missing_family_count += 1;
                missing_command_count += command_count;
                (VulkanReusableKernelLinkStatus::Missing, None)
            } else if family_issues.is_empty() {
                linked_family_count += 1;
                linked_command_count += command_count;
                (
                    VulkanReusableKernelLinkStatus::Linked,
                    artifact.map(|artifact| artifact.path.clone()),
                )
            } else {
                incompatible_family_count += 1;
                incompatible_command_count += command_count;
                (
                    VulkanReusableKernelLinkStatus::Incompatible,
                    artifact.map(|artifact| artifact.path.clone()),
                )
            };

            issues.extend(family_issues.iter().cloned());
            families.push(VulkanLinkedReusableKernelFamily {
                family_id: family.family_id.clone(),
                op: family.op.clone(),
                command_count,
                status,
                artifact_path,
                issues: family_issues,
            });
        }

        Self {
            backend_id: reusable_plan.backend_id.clone(),
            manifest_schema: manifest.schema.clone(),
            manifest_backend_id: manifest.backend_id.clone(),
            required_family_count: reusable_plan.families.len(),
            linked_family_count,
            missing_family_count,
            incompatible_family_count,
            required_command_count: reusable_plan.total_command_count,
            linked_command_count,
            missing_command_count,
            incompatible_command_count,
            families,
            issues,
        }
    }

    pub fn is_fully_linked(&self) -> bool {
        self.missing_family_count == 0
            && self.incompatible_family_count == 0
            && self.linked_command_count == self.required_command_count
    }

    pub fn family(&self, family_id: &str) -> Option<&VulkanLinkedReusableKernelFamily> {
        self.families
            .iter()
            .find(|family| family.family_id == family_id)
    }

    pub fn missing_families(&self) -> Vec<&VulkanLinkedReusableKernelFamily> {
        self.families
            .iter()
            .filter(|family| family.status == VulkanReusableKernelLinkStatus::Missing)
            .collect()
    }

    pub fn incompatible_families(&self) -> Vec<&VulkanLinkedReusableKernelFamily> {
        self.families
            .iter()
            .filter(|family| family.status == VulkanReusableKernelLinkStatus::Incompatible)
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanLinkedReusableKernelFamily {
    pub family_id: String,
    pub op: String,
    pub command_count: usize,
    pub status: VulkanReusableKernelLinkStatus,
    pub artifact_path: Option<String>,
    pub issues: Vec<VulkanReusableKernelLinkIssue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanReusableKernelLinkStatus {
    Linked,
    Missing,
    Incompatible,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanReusableKernelLinkIssue {
    pub family_id: String,
    pub op: String,
    pub problem: VulkanReusableKernelLinkProblem,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VulkanReusableKernelLinkProblem {
    MissingArtifact,
    DuplicateArtifact { count: usize },
    OpMismatch { found: String },
    DescriptorSignatureMismatch,
    PushConstantSignatureMismatch,
    StreamTickUsageMismatch { found: bool },
    EmptySpirvPath,
    UnsupportedEntryPoint { found: String },
    InvalidLocalSizeX { found: u32 },
}

fn link_compatibility_issues(
    family: &VulkanReusableKernelFamily,
    artifact: &VulkanReusableKernelArtifact,
) -> Vec<VulkanReusableKernelLinkIssue> {
    let mut issues = Vec::new();
    let family_id = family.family_id.clone();
    let op = family.op.clone();

    if artifact.op != family.op {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::OpMismatch {
                found: artifact.op.clone(),
            },
        });
    }
    if artifact.descriptor_signature != family.descriptor_signature {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::DescriptorSignatureMismatch,
        });
    }
    if artifact.push_constants != family.push_constants {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::PushConstantSignatureMismatch,
        });
    }
    if artifact.uses_stream_tick != family.uses_stream_tick {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::StreamTickUsageMismatch {
                found: artifact.uses_stream_tick,
            },
        });
    }
    if artifact.path.is_empty() {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::EmptySpirvPath,
        });
    }
    if artifact.entry_point != DEFAULT_SPIRV_ENTRY_POINT {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id: family_id.clone(),
            op: op.clone(),
            problem: VulkanReusableKernelLinkProblem::UnsupportedEntryPoint {
                found: artifact.entry_point.clone(),
            },
        });
    }
    if artifact.local_size_x == 0 {
        issues.push(VulkanReusableKernelLinkIssue {
            family_id,
            op,
            problem: VulkanReusableKernelLinkProblem::InvalidLocalSizeX {
                found: artifact.local_size_x,
            },
        });
    }

    issues
}

