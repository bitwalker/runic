pub mod attributes;
mod builders;
mod ops;
pub mod types;

pub use self::{
    builders::{BuiltinOpBuilder, ComponentBuilder, FunctionBuilder, ModuleBuilder, WorldBuilder},
    ops::*,
};
use crate::{
    DialectInfo,
    derive::{Dialect, DialectRegistration},
};

#[derive(Dialect, DialectRegistration, Debug)]
pub struct BuiltinDialect {
    #[dialect(info)]
    info: DialectInfo,
}
