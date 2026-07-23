impl VulkanResidentInProcessPlacedStreamProcessorDevice {
    pub fn mounted(&self) -> &VulkanMountedPlacedStreamCircuit {
        &self.mounted
    }

    pub fn loaded_manifest(&self) -> &VulkanLoadedReusableKernelArtifactManifest {
        self.package_slice.loaded_manifest()
    }
}
