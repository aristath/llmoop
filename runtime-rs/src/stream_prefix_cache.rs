use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::stream_state::{TransientStateArena, TransientStateKey, TransientStateTable};

const FNV64_OFFSET: u64 = 0xcbf29ce484222325;
const FNV64_PRIME: u64 = 0x100000001b3;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RuntimePrefixStateCacheKey {
    pub execution_class_id: String,
    pub runtime_graph_id: String,
    pub token_count: usize,
    pub token_hash: u64,
    pub runtime_modifier_hash: u64,
    pub state_keys: Vec<TransientStateKey>,
}

impl RuntimePrefixStateCacheKey {
    pub fn from_token_prefix<I>(
        execution_class_id: impl Into<String>,
        runtime_graph_id: impl Into<String>,
        token_ids: &[u32],
        runtime_modifier_bytes: &[u8],
        state_keys: I,
    ) -> Result<Self, RuntimePrefixStateCacheError>
    where
        I: IntoIterator<Item = TransientStateKey>,
    {
        let execution_class_id = execution_class_id.into();
        if execution_class_id.is_empty() {
            return Err(RuntimePrefixStateCacheError(
                "prefix cache execution class id must not be empty".to_string(),
            ));
        }
        let runtime_graph_id = runtime_graph_id.into();
        if runtime_graph_id.is_empty() {
            return Err(RuntimePrefixStateCacheError(
                "prefix cache runtime graph id must not be empty".to_string(),
            ));
        }
        if token_ids.is_empty() {
            return Err(RuntimePrefixStateCacheError(
                "prefix cache token prefix must not be empty".to_string(),
            ));
        }
        let mut state_keys = state_keys.into_iter().collect::<Vec<_>>();
        state_keys.sort();
        state_keys.dedup();
        if state_keys.is_empty() {
            return Err(RuntimePrefixStateCacheError(
                "prefix cache state key set must not be empty".to_string(),
            ));
        }
        Ok(Self {
            execution_class_id,
            runtime_graph_id,
            token_count: token_ids.len(),
            token_hash: stable_token_prefix_hash(token_ids),
            runtime_modifier_hash: stable_runtime_modifier_hash(runtime_modifier_bytes),
            state_keys,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimePrefixStateCacheEntrySnapshot {
    pub key: RuntimePrefixStateCacheKey,
    pub cached_stream_id: String,
    pub block_count: usize,
    pub logical_activation_count: usize,
    pub last_used_tick: u64,
    pub use_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimePrefixStateCacheSnapshot {
    pub capacity_entries: usize,
    pub entry_count: usize,
    pub entries: Vec<RuntimePrefixStateCacheEntrySnapshot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimePrefixStateCacheEntry {
    cached_stream_id: String,
    table: TransientStateTable,
    last_used_tick: u64,
    use_count: usize,
}

#[derive(Default)]
pub struct RuntimePrefixStateCache {
    capacity_entries: usize,
    entries: BTreeMap<RuntimePrefixStateCacheKey, RuntimePrefixStateCacheEntry>,
    next_tick: u64,
    next_cached_stream_id: u64,
}

impl RuntimePrefixStateCache {
    pub fn new(capacity_entries: usize) -> Self {
        Self {
            capacity_entries,
            entries: BTreeMap::new(),
            next_tick: 0,
            next_cached_stream_id: 0,
        }
    }

    pub fn capacity_entries(&self) -> usize {
        self.capacity_entries
    }

    pub fn insert(
        &mut self,
        arena: &mut TransientStateArena,
        key: RuntimePrefixStateCacheKey,
        source: &TransientStateTable,
    ) -> Result<(), RuntimePrefixStateCacheError> {
        if self.capacity_entries == 0 {
            return Ok(());
        }
        let cached_stream_id = self.next_cached_stream_id()?;
        let cached_table = source.fork(arena, cached_stream_id.clone())?;
        if let Some(mut old) = self.entries.remove(&key) {
            old.table.reset_all(arena)?;
        }
        let tick = self.next_tick()?;
        self.entries.insert(
            key,
            RuntimePrefixStateCacheEntry {
                cached_stream_id,
                table: cached_table,
                last_used_tick: tick,
                use_count: 0,
            },
        );
        self.evict_until_within_capacity(arena)?;
        Ok(())
    }

    pub fn restore_into(
        &mut self,
        arena: &mut TransientStateArena,
        key: &RuntimePrefixStateCacheKey,
        target: &mut TransientStateTable,
    ) -> Result<bool, RuntimePrefixStateCacheError> {
        let tick = self.next_tick()?;
        let Some(entry) = self.entries.get_mut(key) else {
            return Ok(false);
        };
        target.share_states_from(arena, &entry.table, key.state_keys.iter())?;
        entry.last_used_tick = tick;
        entry.use_count = entry.use_count.saturating_add(1);
        Ok(true)
    }

    pub fn snapshot(&self) -> RuntimePrefixStateCacheSnapshot {
        let mut entries = self
            .entries
            .iter()
            .map(|(key, entry)| {
                let state = entry.table.snapshot();
                RuntimePrefixStateCacheEntrySnapshot {
                    key: key.clone(),
                    cached_stream_id: entry.cached_stream_id.clone(),
                    block_count: state.block_count,
                    logical_activation_count: state.logical_activation_count,
                    last_used_tick: entry.last_used_tick,
                    use_count: entry.use_count,
                }
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.key.cmp(&right.key));
        RuntimePrefixStateCacheSnapshot {
            capacity_entries: self.capacity_entries,
            entry_count: entries.len(),
            entries,
        }
    }

    fn evict_until_within_capacity(
        &mut self,
        arena: &mut TransientStateArena,
    ) -> Result<(), RuntimePrefixStateCacheError> {
        while self.entries.len() > self.capacity_entries {
            let eviction_key = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| (entry.last_used_tick, entry.use_count))
                .map(|(key, _)| key.clone())
                .ok_or_else(|| {
                    RuntimePrefixStateCacheError(
                        "prefix cache eviction could not find an entry".to_string(),
                    )
                })?;
            let mut evicted = self.entries.remove(&eviction_key).ok_or_else(|| {
                RuntimePrefixStateCacheError("prefix cache eviction entry disappeared".to_string())
            })?;
            evicted.table.reset_all(arena)?;
        }
        Ok(())
    }

    fn next_tick(&mut self) -> Result<u64, RuntimePrefixStateCacheError> {
        let tick = self.next_tick;
        self.next_tick = self.next_tick.checked_add(1).ok_or_else(|| {
            RuntimePrefixStateCacheError("prefix cache tick overflow".to_string())
        })?;
        Ok(tick)
    }

    fn next_cached_stream_id(&mut self) -> Result<String, RuntimePrefixStateCacheError> {
        let id = self.next_cached_stream_id;
        self.next_cached_stream_id =
            self.next_cached_stream_id.checked_add(1).ok_or_else(|| {
                RuntimePrefixStateCacheError("prefix cache stream id overflow".to_string())
            })?;
        Ok(format!("prefix_cache_{id}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePrefixStateCacheError(pub String);

impl Display for RuntimePrefixStateCacheError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for RuntimePrefixStateCacheError {}

impl From<crate::stream_state::TransientStateError> for RuntimePrefixStateCacheError {
    fn from(error: crate::stream_state::TransientStateError) -> Self {
        Self(error.to_string())
    }
}

pub fn stable_token_prefix_hash(token_ids: &[u32]) -> u64 {
    let mut hash = FNV64_OFFSET;
    hash = stable_hash_bytes(hash, b"nerve:token-prefix:v1");
    hash = stable_hash_bytes(hash, &(token_ids.len() as u64).to_le_bytes());
    for token_id in token_ids {
        hash = stable_hash_bytes(hash, &token_id.to_le_bytes());
    }
    hash
}

pub fn stable_runtime_modifier_hash(bytes: &[u8]) -> u64 {
    let mut hash = FNV64_OFFSET;
    hash = stable_hash_bytes(hash, b"nerve:runtime-modifiers:v1");
    hash = stable_hash_bytes(hash, &(bytes.len() as u64).to_le_bytes());
    stable_hash_bytes(hash, bytes)
}

fn stable_hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV64_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream_state::{TransientStateArena, TransientStateBlockShape, TransientStateTable};

    fn key(state_id: &str) -> TransientStateKey {
        TransientStateKey::new("layer_00", state_id)
    }

    fn shape() -> TransientStateBlockShape {
        TransientStateBlockShape::new(16, 2).unwrap()
    }

    fn cache_key(tokens: &[u32], state_keys: Vec<TransientStateKey>) -> RuntimePrefixStateCacheKey {
        RuntimePrefixStateCacheKey::from_token_prefix(
            "package_a",
            "graph_a",
            tokens,
            b"reasoning=true",
            state_keys,
        )
        .unwrap()
    }

    #[test]
    fn prefix_cache_key_normalizes_state_keys_and_hashes_prefix_contents() {
        let left = cache_key(&[1, 2, 3], vec![key("v"), key("k"), key("k")]);
        let right = cache_key(&[1, 2, 3], vec![key("k"), key("v")]);
        let different_tokens = cache_key(&[1, 2, 4], vec![key("k"), key("v")]);

        assert_eq!(left, right);
        assert_ne!(left.token_hash, different_tokens.token_hash);
        assert_eq!(left.state_keys, vec![key("k"), key("v")]);
    }

    #[test]
    fn prefix_cache_eviction_releases_retained_state_blocks() {
        let mut arena = TransientStateArena::new();
        let mut first = TransientStateTable::new("source_a").unwrap();
        let mut second = TransientStateTable::new("source_b").unwrap();
        first.declare_state(key("kv"), shape()).unwrap();
        second.declare_state(key("kv"), shape()).unwrap();
        first.append_activations(&mut arena, &key("kv"), 2).unwrap();
        second
            .append_activations(&mut arena, &key("kv"), 2)
            .unwrap();
        let first_block = first.snapshot().entries[0].block_ids[0];
        let second_block = second.snapshot().entries[0].block_ids[0];
        let mut cache = RuntimePrefixStateCache::new(1);

        cache
            .insert(&mut arena, cache_key(&[1, 2], vec![key("kv")]), &first)
            .unwrap();
        assert_eq!(arena.ref_count(first_block).unwrap(), 2);
        cache
            .insert(&mut arena, cache_key(&[3, 4], vec![key("kv")]), &second)
            .unwrap();

        assert_eq!(cache.snapshot().entry_count, 1);
        assert_eq!(arena.ref_count(first_block).unwrap(), 1);
        assert_eq!(arena.ref_count(second_block).unwrap(), 2);
    }
}
