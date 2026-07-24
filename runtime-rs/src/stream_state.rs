use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransientStateBlockId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransientStateKey {
    pub node_instance_id: String,
    pub state_id: String,
}

impl TransientStateKey {
    pub fn new(node_instance_id: impl Into<String>, state_id: impl Into<String>) -> Self {
        Self {
            node_instance_id: node_instance_id.into(),
            state_id: state_id.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransientStateBlockShape {
    pub bytes_per_activation: usize,
    pub activation_capacity: usize,
    pub maximum_activation_count: Option<usize>,
    pub retention: TransientStateRetention,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TransientStateRetention {
    Append,
    MutableSingleton,
}

impl TransientStateBlockShape {
    pub fn new(
        bytes_per_activation: usize,
        activation_capacity: usize,
    ) -> Result<Self, TransientStateError> {
        if bytes_per_activation == 0 {
            return Err(TransientStateError(
                "transient state block bytes_per_activation must be positive".to_string(),
            ));
        }
        if activation_capacity == 0 {
            return Err(TransientStateError(
                "transient state block activation_capacity must be positive".to_string(),
            ));
        }
        Ok(Self {
            bytes_per_activation,
            activation_capacity,
            maximum_activation_count: None,
            retention: TransientStateRetention::Append,
        })
    }

    pub fn with_maximum_activation_count(
        mut self,
        maximum_activation_count: usize,
    ) -> Result<Self, TransientStateError> {
        if maximum_activation_count == 0 {
            return Err(TransientStateError(
                "transient state maximum activation count must be positive".to_string(),
            ));
        }
        self.maximum_activation_count = Some(maximum_activation_count);
        Ok(self)
    }

    pub fn mutable_singleton(byte_capacity: usize) -> Result<Self, TransientStateError> {
        let mut shape = Self::new(byte_capacity, 1)?;
        shape.retention = TransientStateRetention::MutableSingleton;
        Ok(shape)
    }

    pub fn byte_capacity(&self) -> Result<usize, TransientStateError> {
        self.bytes_per_activation
            .checked_mul(self.activation_capacity)
            .ok_or_else(|| {
                TransientStateError("transient state block byte capacity overflow".to_string())
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransientStateSlot {
    pub key: TransientStateKey,
    pub logical_activation_index: usize,
    pub block_id: TransientStateBlockId,
    pub block_activation_offset: usize,
    pub block_activation_capacity: usize,
    pub copy_from_block_id: Option<TransientStateBlockId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransientStateBlockSnapshot {
    pub block_id: TransientStateBlockId,
    pub shape: TransientStateBlockShape,
    pub ref_count: usize,
    pub byte_capacity: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransientStateArenaSnapshot {
    pub allocated_block_count: usize,
    pub free_block_count: usize,
    pub live_block_count: usize,
    pub blocks: Vec<TransientStateBlockSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TransientStateBlock {
    id: TransientStateBlockId,
    shape: TransientStateBlockShape,
    ref_count: usize,
}

#[derive(Default)]
pub struct TransientStateArena {
    next_block_id: u64,
    blocks: BTreeMap<TransientStateBlockId, TransientStateBlock>,
    free_blocks: BTreeMap<TransientStateBlockShape, Vec<TransientStateBlockId>>,
}

impl TransientStateArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allocate_block(
        &mut self,
        shape: TransientStateBlockShape,
    ) -> Result<TransientStateBlockId, TransientStateError> {
        shape.byte_capacity()?;
        if let Some(free_for_shape) = self.free_blocks.get_mut(&shape)
            && let Some(block_id) = free_for_shape.pop()
        {
            if free_for_shape.is_empty() {
                self.free_blocks.remove(&shape);
            }
            let block = self.block_mut(block_id)?;
            block.ref_count = 1;
            return Ok(block_id);
        }

        let block_id = TransientStateBlockId(self.next_block_id);
        self.next_block_id = self
            .next_block_id
            .checked_add(1)
            .ok_or_else(|| TransientStateError("transient state block id overflow".to_string()))?;
        self.blocks.insert(
            block_id,
            TransientStateBlock {
                id: block_id,
                shape,
                ref_count: 1,
            },
        );
        Ok(block_id)
    }

    pub fn retain_block(
        &mut self,
        block_id: TransientStateBlockId,
    ) -> Result<(), TransientStateError> {
        let block = self.block_mut(block_id)?;
        block.ref_count = block.ref_count.checked_add(1).ok_or_else(|| {
            TransientStateError("transient state block refcount overflow".to_string())
        })?;
        Ok(())
    }

    pub fn release_block(
        &mut self,
        block_id: TransientStateBlockId,
    ) -> Result<(), TransientStateError> {
        let block = self.block_mut(block_id)?;
        if block.ref_count == 0 {
            return Err(TransientStateError(format!(
                "transient state block {:?} is already free",
                block_id
            )));
        }
        block.ref_count -= 1;
        let freed_shape = (block.ref_count == 0).then(|| block.shape.clone());
        if let Some(shape) = freed_shape {
            self.free_blocks.entry(shape).or_default().push(block_id);
        }
        Ok(())
    }

    pub fn ref_count(&self, block_id: TransientStateBlockId) -> Result<usize, TransientStateError> {
        Ok(self.block(block_id)?.ref_count)
    }

    pub fn snapshot(&self) -> Result<TransientStateArenaSnapshot, TransientStateError> {
        let mut blocks = self
            .blocks
            .values()
            .map(|block| {
                Ok(TransientStateBlockSnapshot {
                    block_id: block.id,
                    shape: block.shape.clone(),
                    ref_count: block.ref_count,
                    byte_capacity: block.shape.byte_capacity()?,
                })
            })
            .collect::<Result<Vec<_>, TransientStateError>>()?;
        blocks.sort_by_key(|block| block.block_id);
        Ok(TransientStateArenaSnapshot {
            allocated_block_count: self.blocks.len(),
            free_block_count: blocks.iter().filter(|block| block.ref_count == 0).count(),
            live_block_count: blocks.iter().filter(|block| block.ref_count > 0).count(),
            blocks,
        })
    }

    fn block(
        &self,
        block_id: TransientStateBlockId,
    ) -> Result<&TransientStateBlock, TransientStateError> {
        self.blocks.get(&block_id).ok_or_else(|| {
            TransientStateError(format!("unknown transient state block {:?}", block_id))
        })
    }

    fn block_mut(
        &mut self,
        block_id: TransientStateBlockId,
    ) -> Result<&mut TransientStateBlock, TransientStateError> {
        self.blocks.get_mut(&block_id).ok_or_else(|| {
            TransientStateError(format!("unknown transient state block {:?}", block_id))
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransientStateEntrySnapshot {
    pub key: TransientStateKey,
    pub shape: TransientStateBlockShape,
    pub logical_activation_count: usize,
    pub block_ids: Vec<TransientStateBlockId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransientStateTableSnapshot {
    pub stream_id: String,
    pub entry_count: usize,
    pub logical_activation_count: usize,
    pub block_count: usize,
    pub entries: Vec<TransientStateEntrySnapshot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TransientStateEntry {
    shape: TransientStateBlockShape,
    logical_activation_count: usize,
    block_ids: Vec<TransientStateBlockId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransientStateTable {
    stream_id: String,
    entries: BTreeMap<TransientStateKey, TransientStateEntry>,
}

impl TransientStateTable {
    pub fn new(stream_id: impl Into<String>) -> Result<Self, TransientStateError> {
        let stream_id = stream_id.into();
        if stream_id.is_empty() {
            return Err(TransientStateError(
                "transient state table stream id must not be empty".to_string(),
            ));
        }
        Ok(Self {
            stream_id,
            entries: BTreeMap::new(),
        })
    }

    pub fn stream_id(&self) -> &str {
        &self.stream_id
    }

    pub fn declare_state(
        &mut self,
        key: TransientStateKey,
        shape: TransientStateBlockShape,
    ) -> Result<(), TransientStateError> {
        if key.node_instance_id.is_empty() || key.state_id.is_empty() {
            return Err(TransientStateError(
                "transient state key node_instance_id and state_id must not be empty".to_string(),
            ));
        }
        if let Some(existing) = self.entries.get(&key) {
            if existing.shape != shape {
                return Err(TransientStateError(format!(
                    "transient state {:?}.{} was already declared with a different block shape",
                    key.node_instance_id, key.state_id
                )));
            }
            return Ok(());
        }
        self.entries.insert(
            key,
            TransientStateEntry {
                shape,
                logical_activation_count: 0,
                block_ids: Vec::new(),
            },
        );
        Ok(())
    }

    pub fn append_activations(
        &mut self,
        arena: &mut TransientStateArena,
        key: &TransientStateKey,
        activation_count: usize,
    ) -> Result<Vec<TransientStateSlot>, TransientStateError> {
        if activation_count == 0 {
            return Ok(Vec::new());
        }
        let entry = self.entry_mut(key)?;
        if entry.shape.retention == TransientStateRetention::MutableSingleton {
            return reserve_mutable_singleton(arena, key, entry);
        }
        let mut slots = Vec::with_capacity(activation_count);
        for _ in 0..activation_count {
            let resident_activation_index = entry
                .shape
                .maximum_activation_count
                .map(|maximum| entry.logical_activation_count % maximum)
                .unwrap_or(entry.logical_activation_count);
            let logical_page_index = resident_activation_index / entry.shape.activation_capacity;
            let block_activation_offset =
                resident_activation_index % entry.shape.activation_capacity;
            let mut copy_from_block_id = None;
            if logical_page_index == entry.block_ids.len() {
                let block_id = arena.allocate_block(entry.shape.clone())?;
                entry.block_ids.push(block_id);
            } else {
                let block_id = entry.block_ids[logical_page_index];
                if arena.ref_count(block_id)? > 1 {
                    let replacement = arena.allocate_block(entry.shape.clone())?;
                    arena.release_block(block_id)?;
                    entry.block_ids[logical_page_index] = replacement;
                    copy_from_block_id = Some(block_id);
                }
            }
            let logical_activation_index = entry.logical_activation_count;
            let block_id = entry.block_ids[logical_page_index];
            slots.push(TransientStateSlot {
                key: key.clone(),
                logical_activation_index,
                block_id,
                block_activation_offset,
                block_activation_capacity: entry.shape.activation_capacity,
                copy_from_block_id,
            });
            entry.logical_activation_count = entry.logical_activation_count.saturating_add(1);
        }
        Ok(slots)
    }

    pub fn reset_state(
        &mut self,
        arena: &mut TransientStateArena,
        key: &TransientStateKey,
    ) -> Result<(), TransientStateError> {
        let entry = self.entry_mut(key)?;
        release_unique_blocks(arena, &entry.block_ids)?;
        entry.block_ids.clear();
        entry.logical_activation_count = 0;
        Ok(())
    }

    pub fn reset_all(
        &mut self,
        arena: &mut TransientStateArena,
    ) -> Result<(), TransientStateError> {
        let block_ids = self
            .entries
            .values()
            .flat_map(|entry| entry.block_ids.iter().copied())
            .collect::<Vec<_>>();
        release_unique_blocks(arena, &block_ids)?;
        for entry in self.entries.values_mut() {
            entry.block_ids.clear();
            entry.logical_activation_count = 0;
        }
        Ok(())
    }

    pub fn fork(
        &self,
        arena: &mut TransientStateArena,
        new_stream_id: impl Into<String>,
    ) -> Result<Self, TransientStateError> {
        let new_stream_id = new_stream_id.into();
        if new_stream_id.is_empty() {
            return Err(TransientStateError(
                "forked transient state table stream id must not be empty".to_string(),
            ));
        }
        for block_id in unique_block_ids(
            self.entries
                .values()
                .flat_map(|entry| entry.block_ids.iter().copied()),
        ) {
            arena.retain_block(block_id)?;
        }
        Ok(Self {
            stream_id: new_stream_id,
            entries: self.entries.clone(),
        })
    }

    pub fn share_state_from(
        &mut self,
        arena: &mut TransientStateArena,
        source: &Self,
        key: &TransientStateKey,
    ) -> Result<(), TransientStateError> {
        self.share_states_from(arena, source, [key])
    }

    pub fn share_states_from<'a, I>(
        &mut self,
        arena: &mut TransientStateArena,
        source: &Self,
        keys: I,
    ) -> Result<(), TransientStateError>
    where
        I: IntoIterator<Item = &'a TransientStateKey>,
    {
        let shares = keys
            .into_iter()
            .map(|key| {
                let source_entry = source.entry(key)?;
                if let Some(existing) = self.entries.get(key) {
                    if existing.shape != source_entry.shape {
                        return Err(TransientStateError(format!(
                            "cannot share transient state {:?}.{} into a table with a different shape",
                            key.node_instance_id, key.state_id
                        )));
                    }
                    if !existing.block_ids.is_empty() || existing.logical_activation_count != 0 {
                        return Err(TransientStateError(format!(
                            "cannot replace non-empty transient state {:?}.{} with shared blocks",
                            key.node_instance_id, key.state_id
                        )));
                    }
                }
                Ok((key.clone(), source_entry.clone()))
            })
            .collect::<Result<Vec<_>, TransientStateError>>()?;
        for block_id in unique_block_ids(
            shares
                .iter()
                .flat_map(|(_, entry)| entry.block_ids.iter().copied()),
        ) {
            arena.retain_block(block_id)?;
        }
        for (key, source_entry) in shares {
            self.entries.insert(key, source_entry);
        }
        Ok(())
    }

    pub fn snapshot(&self) -> TransientStateTableSnapshot {
        let mut entries = self
            .entries
            .iter()
            .map(|(key, entry)| TransientStateEntrySnapshot {
                key: key.clone(),
                shape: entry.shape.clone(),
                logical_activation_count: entry.logical_activation_count,
                block_ids: entry.block_ids.clone(),
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.key.cmp(&right.key));
        TransientStateTableSnapshot {
            stream_id: self.stream_id.clone(),
            entry_count: entries.len(),
            logical_activation_count: entries
                .iter()
                .map(|entry| entry.logical_activation_count)
                .sum(),
            block_count: entries.iter().map(|entry| entry.block_ids.len()).sum(),
            entries,
        }
    }

    pub fn state_keys(&self) -> Vec<TransientStateKey> {
        self.entries.keys().cloned().collect()
    }

    fn entry(&self, key: &TransientStateKey) -> Result<&TransientStateEntry, TransientStateError> {
        self.entries.get(key).ok_or_else(|| {
            TransientStateError(format!(
                "stream {:?} has no transient state {:?}.{}",
                self.stream_id, key.node_instance_id, key.state_id
            ))
        })
    }

    fn entry_mut(
        &mut self,
        key: &TransientStateKey,
    ) -> Result<&mut TransientStateEntry, TransientStateError> {
        self.entries.get_mut(key).ok_or_else(|| {
            TransientStateError(format!(
                "stream {:?} has no transient state {:?}.{}",
                self.stream_id, key.node_instance_id, key.state_id
            ))
        })
    }
}

fn reserve_mutable_singleton(
    arena: &mut TransientStateArena,
    key: &TransientStateKey,
    entry: &mut TransientStateEntry,
) -> Result<Vec<TransientStateSlot>, TransientStateError> {
    let mut copy_from_block_id = None;
    let block_id = match entry.block_ids.first().copied() {
        Some(block_id) if arena.ref_count(block_id)? > 1 => {
            let replacement = arena.allocate_block(entry.shape.clone())?;
            arena.release_block(block_id)?;
            entry.block_ids[0] = replacement;
            copy_from_block_id = Some(block_id);
            replacement
        }
        Some(block_id) => block_id,
        None => {
            let block_id = arena.allocate_block(entry.shape.clone())?;
            entry.block_ids.push(block_id);
            entry.logical_activation_count = 1;
            block_id
        }
    };
    Ok(vec![TransientStateSlot {
        key: key.clone(),
        logical_activation_index: 0,
        block_id,
        block_activation_offset: 0,
        block_activation_capacity: 1,
        copy_from_block_id,
    }])
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransientStateError(pub String);

impl Display for TransientStateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for TransientStateError {}

fn release_unique_blocks(
    arena: &mut TransientStateArena,
    block_ids: &[TransientStateBlockId],
) -> Result<(), TransientStateError> {
    for block_id in unique_block_ids(block_ids.iter().copied()) {
        arena.release_block(block_id)?;
    }
    Ok(())
}

fn unique_block_ids(
    block_ids: impl IntoIterator<Item = TransientStateBlockId>,
) -> Vec<TransientStateBlockId> {
    block_ids
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> TransientStateBlockShape {
        TransientStateBlockShape::new(16, 4).unwrap()
    }

    fn key() -> TransientStateKey {
        TransientStateKey::new("layer_00", "kv")
    }

    #[test]
    fn transient_state_appends_slots_across_fixed_size_blocks() {
        let mut arena = TransientStateArena::new();
        let mut table = TransientStateTable::new("stream_a").unwrap();
        table.declare_state(key(), shape()).unwrap();

        let slots = table.append_activations(&mut arena, &key(), 6).unwrap();

        assert_eq!(slots.len(), 6);
        assert_eq!(slots[0].logical_activation_index, 0);
        assert_eq!(slots[0].block_activation_offset, 0);
        assert_eq!(slots[3].block_activation_offset, 3);
        assert_ne!(slots[3].block_id, slots[4].block_id);
        assert_eq!(slots[4].block_activation_offset, 0);
        assert_eq!(table.snapshot().block_count, 2);
        assert_eq!(arena.snapshot().unwrap().live_block_count, 2);
    }

    #[test]
    fn transient_state_reset_releases_and_reuses_matching_blocks() {
        let mut arena = TransientStateArena::new();
        let mut first = TransientStateTable::new("stream_a").unwrap();
        first.declare_state(key(), shape()).unwrap();
        let allocated = first
            .append_activations(&mut arena, &key(), 5)
            .unwrap()
            .into_iter()
            .map(|slot| slot.block_id)
            .collect::<BTreeSet<_>>();
        first.reset_state(&mut arena, &key()).unwrap();
        assert_eq!(arena.snapshot().unwrap().free_block_count, 2);

        let mut second = TransientStateTable::new("stream_b").unwrap();
        second.declare_state(key(), shape()).unwrap();
        let reused = second
            .append_activations(&mut arena, &key(), 5)
            .unwrap()
            .into_iter()
            .map(|slot| slot.block_id)
            .collect::<BTreeSet<_>>();

        assert_eq!(allocated, reused);
        assert_eq!(arena.snapshot().unwrap().allocated_block_count, 2);
        assert_eq!(arena.snapshot().unwrap().live_block_count, 2);
    }

    #[test]
    fn transient_state_fork_shares_blocks_until_each_table_resets() {
        let mut arena = TransientStateArena::new();
        let mut parent = TransientStateTable::new("parent").unwrap();
        parent.declare_state(key(), shape()).unwrap();
        let slots = parent.append_activations(&mut arena, &key(), 3).unwrap();
        let block_id = slots[0].block_id;

        let mut child = parent.fork(&mut arena, "child").unwrap();
        assert_eq!(arena.ref_count(block_id).unwrap(), 2);

        parent.reset_all(&mut arena).unwrap();
        assert_eq!(arena.ref_count(block_id).unwrap(), 1);
        assert_eq!(arena.snapshot().unwrap().free_block_count, 0);

        child.reset_all(&mut arena).unwrap();
        assert_eq!(arena.ref_count(block_id).unwrap(), 0);
        assert_eq!(arena.snapshot().unwrap().free_block_count, 1);
    }

    #[test]
    fn transient_state_fork_copies_a_shared_partial_append_block_before_writing() {
        let mut arena = TransientStateArena::new();
        let mut parent = TransientStateTable::new("parent").unwrap();
        parent.declare_state(key(), shape()).unwrap();
        let shared_block = parent.append_activations(&mut arena, &key(), 3).unwrap()[0].block_id;
        let mut child = parent.fork(&mut arena, "child").unwrap();

        let child_slot = child.append_activations(&mut arena, &key(), 1).unwrap()[0].clone();

        assert_ne!(child_slot.block_id, shared_block);
        assert_eq!(child_slot.copy_from_block_id, Some(shared_block));
        assert_eq!(child_slot.block_activation_offset, 3);
        assert_eq!(parent.snapshot().entries[0].block_ids, vec![shared_block]);
        assert_eq!(arena.ref_count(shared_block).unwrap(), 1);
        assert_eq!(arena.ref_count(child_slot.block_id).unwrap(), 1);
    }

    #[test]
    fn bounded_append_state_reuses_its_physical_block_after_wrapping() {
        let mut arena = TransientStateArena::new();
        let mut table = TransientStateTable::new("stream").unwrap();
        let bounded = shape().with_maximum_activation_count(4).unwrap();
        table.declare_state(key(), bounded).unwrap();
        let first_block = table.append_activations(&mut arena, &key(), 4).unwrap()[0].block_id;

        let wrapped = table.append_activations(&mut arena, &key(), 1).unwrap();

        assert_eq!(wrapped[0].logical_activation_index, 4);
        assert_eq!(wrapped[0].block_activation_offset, 0);
        assert_eq!(wrapped[0].block_id, first_block);
        assert_eq!(wrapped[0].copy_from_block_id, None);
        assert_eq!(table.snapshot().block_count, 1);
        assert_eq!(arena.snapshot().unwrap().allocated_block_count, 1);
        assert_eq!(arena.snapshot().unwrap().live_block_count, 1);
    }

    #[test]
    fn bounded_append_state_copies_a_shared_page_before_wrapped_write() {
        let mut arena = TransientStateArena::new();
        let mut parent = TransientStateTable::new("parent").unwrap();
        let bounded = shape().with_maximum_activation_count(4).unwrap();
        parent.declare_state(key(), bounded).unwrap();
        let shared_block = parent.append_activations(&mut arena, &key(), 4).unwrap()[0].block_id;
        let mut child = parent.fork(&mut arena, "child").unwrap();

        let wrapped = child.append_activations(&mut arena, &key(), 1).unwrap()[0].clone();

        assert_eq!(wrapped.logical_activation_index, 4);
        assert_eq!(wrapped.block_activation_offset, 0);
        assert_ne!(wrapped.block_id, shared_block);
        assert_eq!(wrapped.copy_from_block_id, Some(shared_block));
        assert_eq!(parent.snapshot().entries[0].block_ids, vec![shared_block]);
        assert_eq!(arena.ref_count(shared_block).unwrap(), 1);
        assert_eq!(arena.ref_count(wrapped.block_id).unwrap(), 1);
    }

    #[test]
    fn mutable_singleton_state_reuses_one_block_and_copies_on_forked_write() {
        let mut arena = TransientStateArena::new();
        let mut parent = TransientStateTable::new("parent").unwrap();
        let mutable = TransientStateBlockShape::mutable_singleton(128).unwrap();
        parent.declare_state(key(), mutable).unwrap();

        let first = parent.append_activations(&mut arena, &key(), 64).unwrap();
        assert_eq!(first.len(), 1);
        let shared_block = first[0].block_id;
        assert_eq!(parent.snapshot().logical_activation_count, 1);

        let mut child = parent.fork(&mut arena, "child").unwrap();
        let child_slot = child.append_activations(&mut arena, &key(), 64).unwrap()[0].clone();
        assert_ne!(child_slot.block_id, shared_block);
        assert_eq!(child_slot.copy_from_block_id, Some(shared_block));
        assert_eq!(child.snapshot().logical_activation_count, 1);

        let next = child.append_activations(&mut arena, &key(), 64).unwrap();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].block_id, child_slot.block_id);
        assert_eq!(next[0].copy_from_block_id, None);
    }

    #[test]
    fn transient_state_can_share_one_component_state_between_streams() {
        let mut arena = TransientStateArena::new();
        let mut source = TransientStateTable::new("source").unwrap();
        let kv = TransientStateKey::new("layer_02", "kv_memory");
        let conv = TransientStateKey::new("layer_02", "conv_state");
        source.declare_state(kv.clone(), shape()).unwrap();
        source
            .declare_state(conv.clone(), TransientStateBlockShape::new(8, 2).unwrap())
            .unwrap();
        let kv_block = source.append_activations(&mut arena, &kv, 1).unwrap()[0].block_id;
        let conv_block = source.append_activations(&mut arena, &conv, 1).unwrap()[0].block_id;

        let mut target = TransientStateTable::new("target").unwrap();
        target.share_state_from(&mut arena, &source, &kv).unwrap();

        assert_eq!(target.snapshot().entry_count, 1);
        assert_eq!(arena.ref_count(kv_block).unwrap(), 2);
        assert_eq!(arena.ref_count(conv_block).unwrap(), 1);
    }
}
