use crate::vulkan_compute::{
    VulkanComputeDevice, VulkanError, VulkanU32PedalRun, VulkanU32ResidentBuffer,
    VulkanU32ShaderPedal,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanU32PedalboardRun {
    pub device_name: String,
    pub input: Vec<u32>,
    pub output: Vec<u32>,
    pub steps: Vec<VulkanU32PedalRun>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanU32Pedalboard {
    pedals: Vec<VulkanU32ShaderPedal>,
}

impl VulkanU32Pedalboard {
    pub fn new(pedals: Vec<VulkanU32ShaderPedal>) -> Self {
        Self { pedals }
    }

    pub fn bypass() -> Self {
        Self { pedals: Vec::new() }
    }

    pub fn pedals(&self) -> &[VulkanU32ShaderPedal] {
        &self.pedals
    }

    pub fn pedal_ids(&self) -> Vec<&str> {
        self.pedals.iter().map(|pedal| pedal.pedal_id()).collect()
    }

    pub fn duplicate_pedal(&mut self, index: usize) -> Result<(), VulkanError> {
        let pedal = self
            .pedals
            .get(index)
            .ok_or_else(|| VulkanError(format!("no pedal at index {index}")))?
            .clone();
        self.pedals.insert(index + 1, pedal);
        Ok(())
    }

    pub fn remove_pedal(&mut self, index: usize) -> Result<VulkanU32ShaderPedal, VulkanError> {
        if index >= self.pedals.len() {
            return Err(VulkanError(format!("no pedal at index {index}")));
        }
        Ok(self.pedals.remove(index))
    }

    pub fn install(&self, device: &VulkanComputeDevice) -> Result<(), VulkanError> {
        for pedal in &self.pedals {
            pedal.install_on_device(device)?;
        }
        Ok(())
    }

    pub fn process(
        &self,
        device: &VulkanComputeDevice,
        input: &[u32],
    ) -> Result<VulkanU32PedalboardRun, VulkanError> {
        let mut current = input.to_vec();
        let mut steps = Vec::with_capacity(self.pedals.len());
        for pedal in &self.pedals {
            let step = pedal.process(device, &current)?;
            current = step.output.clone();
            steps.push(step);
        }
        Ok(VulkanU32PedalboardRun {
            device_name: device.device_name().to_string(),
            input: input.to_vec(),
            output: current,
            steps,
        })
    }

    pub fn process_resident(
        &self,
        device: &VulkanComputeDevice,
        buffer: &VulkanU32ResidentBuffer,
        len: usize,
    ) -> Result<VulkanU32PedalboardRun, VulkanError> {
        let input = buffer.read(len)?;
        let mut steps = Vec::with_capacity(self.pedals.len());
        for pedal in &self.pedals {
            steps.push(pedal.process_resident(device, buffer, len)?);
        }
        Ok(VulkanU32PedalboardRun {
            device_name: device.device_name().to_string(),
            input,
            output: buffer.read(len)?,
            steps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan_compute::compile_test_shader_words;

    fn test_device() -> Option<VulkanComputeDevice> {
        match VulkanComputeDevice::new() {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping Vulkan smoke: {error}");
                None
            }
        }
    }

    fn add_one_pedal(id: &str) -> Option<VulkanU32ShaderPedal> {
        let spirv_words = compile_test_shader_words()?;
        VulkanU32ShaderPedal::new(id, "u32_add_one", spirv_words, 64).ok()
    }

    #[test]
    fn series_pedals_execute_in_order_on_one_device() {
        let Some(device) = test_device() else {
            return;
        };
        let Some(first) = add_one_pedal("add_one_a") else {
            return;
        };
        let Some(second) = add_one_pedal("add_one_b") else {
            return;
        };
        let board = VulkanU32Pedalboard::new(vec![first, second]);
        board.install(&device).unwrap();

        assert_eq!(device.pipeline_cache_stats().u32_storage_pipelines, 1);

        let run = board.process(&device, &[0, 10, 40]).unwrap();

        assert_eq!(board.pedal_ids(), vec!["add_one_a", "add_one_b"]);
        assert_eq!(run.input, vec![0, 10, 40]);
        assert_eq!(run.output, vec![2, 12, 42]);
        assert_eq!(run.steps.len(), 2);
        assert_eq!(run.steps[0].output, vec![1, 11, 41]);
        assert_eq!(run.steps[1].input, vec![1, 11, 41]);
    }

    #[test]
    fn pedal_can_be_duplicated_or_removed_from_board() {
        let Some(device) = test_device() else {
            return;
        };
        let Some(pedal) = add_one_pedal("add_one") else {
            return;
        };
        let mut board = VulkanU32Pedalboard::new(vec![pedal]);
        board.duplicate_pedal(0).unwrap();

        let duplicated = board.process(&device, &[5]).unwrap();
        let removed = board.remove_pedal(1).unwrap();
        let single = board.process(&device, &[5]).unwrap();

        assert_eq!(removed.pedal_id(), "add_one");
        assert_eq!(duplicated.output, vec![7]);
        assert_eq!(single.output, vec![6]);
        assert_eq!(board.pedal_ids(), vec!["add_one"]);
    }

    #[test]
    fn empty_board_is_a_bypass() {
        let Some(device) = test_device() else {
            return;
        };
        let board = VulkanU32Pedalboard::bypass();

        let run = board.process(&device, &[9, 10]).unwrap();

        assert_eq!(board.pedals(), &[]);
        assert_eq!(run.input, vec![9, 10]);
        assert_eq!(run.output, vec![9, 10]);
        assert!(run.steps.is_empty());
    }

    #[test]
    fn series_pedals_can_process_one_resident_signal_buffer() {
        let Some(device) = test_device() else {
            return;
        };
        let Some(first) = add_one_pedal("add_one_a") else {
            return;
        };
        let Some(second) = add_one_pedal("add_one_b") else {
            return;
        };
        let buffer = device.create_u32_resident_buffer(3).unwrap();
        buffer.write(&[10, 20, 30]).unwrap();
        let board = VulkanU32Pedalboard::new(vec![first, second]);
        board.install(&device).unwrap();

        let run = board.process_resident(&device, &buffer, 3).unwrap();

        assert_eq!(run.input, vec![10, 20, 30]);
        assert_eq!(run.output, vec![12, 22, 32]);
        assert_eq!(run.steps.len(), 2);
        assert_eq!(buffer.read(3).unwrap(), vec![12, 22, 32]);
    }
}
