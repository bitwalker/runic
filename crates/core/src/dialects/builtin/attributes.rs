mod array;
mod boolean;
mod bytes;
mod data_layout;
pub(crate) mod effects;
mod integer;
mod list;
mod local_variable;
mod location;
mod overflow;
pub(crate) mod signature;
mod signedness;
mod string;
mod symbol_ref;
mod r#type;
mod unit;
mod visibility;

pub use self::{
    array::{Array, I32ArrayAttr, LocalVariableArrayAttr, TypeArrayAttr},
    boolean::BoolAttr,
    bytes::{Bytes, BytesAttr},
    data_layout::DataLayoutAttr,
    effects::{DeclaredEffects, DeclaredEffectsAttr, MemoryEffectDescriptor},
    integer::*,
    list::{List, ListAttr},
    local_variable::{LocalVariable, LocalVariableAttr},
    location::{Location, LocationAttr},
    overflow::{Overflow, OverflowAttr},
    signature::{
        AbiParam, ArgumentExtension, InvalidSignature, SextAttr, Signature, SignatureAttr,
        SretAttr, ZextAttr,
    },
    signedness::{Signedness, SignednessAttr},
    string::StringAttr,
    symbol_ref::{SymbolRef, SymbolRefAttr},
    r#type::TypeAttr,
    unit::UnitAttr,
    visibility::{Visibility, VisibilityAttr},
};
pub use crate::ir::IdentAttr;
