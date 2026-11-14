use alloc::{boxed::Box, format, rc::Rc};

use super::AttributeValue;
use crate::{
    Attribute, Context, DialectRegistration, Report, Type, UnsafeIntrusiveEntityRef,
    attributes::AttributeElements, interner, traits::TraitInfo,
};

pub trait AttributeRegistration: Attribute {
    type Dialect: DialectRegistration;
    /// The value of the attribute. What it holds is walked through [AttributeElements].
    type Value: AttributeValue + AttributeElements;

    /// False for an attribute kind without a default value, which is always written with a body
    /// (`#[attribute(.., no_default)]`); its [Self::create_default] panics. An operation property
    /// marked `#[default]` whose attribute kind has none fails to compile.
    const HAS_DEFAULT: bool = true;

    /// The name of the dialect this attribute is declared part of
    fn dialect_name() -> interner::Symbol {
        interner::Symbol::intern(
            <<Self as AttributeRegistration>::Dialect as DialectRegistration>::NAMESPACE,
        )
    }
    /// The name of the attribute
    fn name() -> interner::Symbol;
    /// The fully-qualified name of the attribute (i.e. `<dialect>.<name>`)
    fn full_name() -> interner::Symbol {
        interner::Symbol::intern(format!(
            "{}.{}",
            Self::dialect_name(),
            <Self as AttributeRegistration>::name()
        ))
    }
    /// The set of statically known traits for this attribute
    fn traits() -> Box<[TraitInfo]>;
    /// Create a new instance of this attribute in the given context
    fn create<V>(context: &Rc<Context>, value: V, ty: Type) -> UnsafeIntrusiveEntityRef<Self>
    where
        Self::Value: From<V>;
    /// Create a new default-valued instance of this attribute in the given context
    fn create_default(context: &Rc<Context>) -> UnsafeIntrusiveEntityRef<Self>;
    /// Get a reference to the concrete value of this attribute
    fn underlying_value(attr: &Self) -> &Self::Value;
    /// Get a mutable reference the concrete value of this attribute
    fn underlying_value_mut(attr: &mut Self) -> &mut Self::Value;
    /// Whether `value`, with the attribute type `ty`, is well formed by itself. The default accepts.
    ///
    /// [Operation::verify](crate::Operation::verify) calls this (through
    /// [Attribute::verify](crate::Attribute::verify)) for every attribute an operation holds, at
    /// any depth, so it catches a value edited after it was built. It checks the value alone:
    /// what depends on the operation holding the attribute is that operation's business. The
    /// types the value holds need not be checked for ownership or openness; the verifier checks
    /// every type an operation holds. A kind may still check them itself when it can say more
    /// precisely where the fault is, as `SignatureAttr` does with "parameter N".
    ///
    /// The error's message describes the attribute as a noun phrase, which the verifier puts
    /// after "is": "an integer constant of type `i8` holding a 32-bit value, but ..". A kind
    /// states its check with `#[attribute(.., verify = "path")]`, naming a function of this
    /// signature.
    fn verify(value: &Self::Value, ty: Type, context: &Context) -> Result<(), Report> {
        let _ = (value, ty, context);
        Ok(())
    }
}
