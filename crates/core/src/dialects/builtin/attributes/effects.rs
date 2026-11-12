use crate::effects::MemoryEffect;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryEffectDescriptor {
    pub effect: MemoryEffect,
    pub argument: Option<u8>,
    pub result: Option<u8>,
}
