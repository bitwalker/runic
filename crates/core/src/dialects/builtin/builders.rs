mod component;
mod function;
mod module;
mod world;

pub use self::{component::*, function::*, module::*, world::*};
use super::{
    attributes::{LocalVariable, Signature},
    ops::*,
};
use crate::{
    AsCallableSymbolRef, Builder, BuilderExt, Ident, Int, OpBuilder, OpRegistration, Report,
    SourceSpan, Spanned, SymbolName, SymbolTable, Type, UnsafeIntrusiveEntityRef, ValueRef,
    Visibility, constants::ConstantData,
};

pub trait BuiltinOpBuilder<'f, B: ?Sized + Builder> {
    /// Create a [Module] named by the raw segment `name`, spelled as [Module::canonical_name]
    /// does.
    ///
    /// Returns an error when the spelling rules reject `name`, or it contains `::`, or the
    /// module cannot be defined where it is inserted.
    fn create_module(&mut self, name: Ident) -> Result<ModuleRef, Report> {
        let name = Ident::new(Module::canonical_name(name.name)?, name.span);
        ensure_definable::<Module, _>(self.builder(), name.name)?;
        let op_builder = self.builder_mut().create::<Module, (_,)>(name.span());
        op_builder(name)
    }

    /// Create a [Function] named `name`.
    ///
    /// Returns an error when the function cannot be defined where it is inserted.
    fn create_function(
        &mut self,
        name: Ident,
        visibility: Visibility,
        signature: Signature,
    ) -> Result<FunctionRef, Report> {
        ensure_definable::<Function, _>(self.builder(), name.name)?;
        let op_builder = self.builder_mut().create::<Function, (_, _, _)>(name.span());
        op_builder(name, visibility, signature)
    }

    /// Create a [FunctionAlias] referring to `target`.
    ///
    /// Returns an error when the alias cannot be defined where it is inserted.
    fn create_function_alias<C: AsCallableSymbolRef>(
        &mut self,
        name: Ident,
        visibility: Visibility,
        target: C,
    ) -> Result<FunctionAliasRef, Report> {
        ensure_definable::<FunctionAlias, _>(self.builder(), name.name)?;
        let op_builder = self.builder_mut().create::<FunctionAlias, (_, _, C)>(name.span());
        op_builder(name, visibility, target)
    }

    /// Create a [GlobalVariable] named `name`.
    ///
    /// Returns an error when the global cannot be defined where it is inserted.
    fn create_global_variable(
        &mut self,
        name: Ident,
        visibility: Visibility,
        ty: Type,
    ) -> Result<GlobalVariableRef, Report> {
        ensure_definable::<GlobalVariable, _>(self.builder(), name.name)?;
        let op_builder = self.builder_mut().create::<GlobalVariable, (_, _, _)>(name.span());
        op_builder(name, visibility, ty)
    }

    fn create_data_segment(
        &mut self,
        offset: u64,
        data: impl Into<ConstantData>,
        readonly: bool,
        span: SourceSpan,
    ) -> Result<UnsafeIntrusiveEntityRef<Segment>, Report> {
        let id = self.builder().context().create_constant(data);
        let data = self.builder().context().get_constant(id);
        let op_builder = self.builder_mut().create::<Segment, (_, _, _)>(span);
        // The offset is held in an `I64Attr`, as the bits of the `u64`
        op_builder(offset as i64, data, readonly)
    }

    /// Create a [GlobalSymbol] of `global`: its address, a `ptr<T>` in address space 0, where `T`
    /// is the declared type of the global.
    fn global_symbol(
        &mut self,
        global: GlobalVariableRef,
        span: SourceSpan,
    ) -> Result<UnsafeIntrusiveEntityRef<GlobalSymbol>, Report> {
        let op_builder = self.builder_mut().create::<GlobalSymbol, (GlobalVariableRef,)>(span);
        op_builder(global)
    }

    /// Create a [Bytes] constant holding `data`, an `array<i8; N>` of its `N` bytes.
    fn bytes(
        &mut self,
        data: impl Into<ConstantData>,
        span: SourceSpan,
    ) -> Result<BytesRef, Report> {
        let id = self.builder().context().create_constant(data);
        let data = self.builder().context().get_constant(id);
        let op_builder = self.builder_mut().create::<Bytes, (_,)>(span);
        op_builder(data)
    }

    fn unrealized_conversion_cast(
        &mut self,
        value: ValueRef,
        ty: Type,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<UnrealizedConversionCast, (_, _)>(span);
        let op = op_builder(value, ty)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// Create a [Call] of `callee` with `signature`, passing `args`.
    ///
    /// The call has one result per result of `signature`. Nothing checks here that `args` agree
    /// with `signature`, or that `signature` is the callee's: the call's verifier does.
    fn call<C, A>(
        &mut self,
        callee: C,
        signature: Signature,
        args: A,
        span: SourceSpan,
    ) -> Result<CallRef, Report>
    where
        C: AsCallableSymbolRef,
        A: IntoIterator<Item = ValueRef>,
    {
        let op_builder = self.builder_mut().create::<Call, (C, _, A)>(span);
        op_builder(callee, signature, args)
    }

    /// Create a [FunctionReference] to `callee`, a value of the function type of the signature of
    /// the callable `callee` resolves to.
    ///
    /// Returns an error when `callee` does not resolve to a callable.
    fn function_ref<C>(
        &mut self,
        callee: C,
        span: SourceSpan,
    ) -> Result<FunctionReferenceRef, Report>
    where
        C: AsCallableSymbolRef,
    {
        let signature = callee
            .as_callable_symbol_ref()
            .resolve_callable()
            .map_err(|err| Report::msg(alloc::format!("cannot reference the callee: {err}")))?
            .signature();
        let ty = super::ops::function_type_of(self.builder().context(), &signature);
        let op_builder = self.builder_mut().create::<FunctionReference, (C, Type)>(span);
        op_builder(callee, ty)
    }

    /// Create a [CallIndirect] of the function value `callee` with `signature`, passing `args`.
    ///
    /// The call has one result per result of `signature`. Nothing checks here that `callee` and
    /// `args` agree with `signature`: the call's verifier does.
    fn call_indirect<A>(
        &mut self,
        callee: ValueRef,
        signature: Signature,
        args: A,
        span: SourceSpan,
    ) -> Result<CallIndirectRef, Report>
    where
        A: IntoIterator<Item = ValueRef>,
    {
        let op_builder = self.builder_mut().create::<CallIndirect, (_, A, _)>(span);
        op_builder(callee, args, signature)
    }

    fn ret<I>(
        &mut self,
        returning: I,
        span: SourceSpan,
    ) -> Result<UnsafeIntrusiveEntityRef<Ret>, Report>
    where
        I: IntoIterator<Item = ValueRef>,
    {
        let op_builder = self.builder_mut().create::<Ret, (I,)>(span);
        op_builder(returning)
    }

    fn ret_imm(
        &mut self,
        arg: Int,
        span: SourceSpan,
    ) -> Result<UnsafeIntrusiveEntityRef<RetImm>, Report> {
        let op_builder = self.builder_mut().create::<RetImm, _>(span);
        op_builder(arg)
    }

    /// Create a [LoadLocal] of `local`, and return the value read, of the local's type.
    fn load_local(&mut self, local: LocalVariable, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<LoadLocal, _>(span);
        let op = op_builder(local)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// Create a [StoreLocal] of `value` to `local`.
    ///
    /// Nothing checks here that `value` has the local's type: the store's verifier does.
    fn store_local(
        &mut self,
        local: LocalVariable,
        value: ValueRef,
        span: SourceSpan,
    ) -> Result<StoreLocalRef, Report> {
        let op_builder = self.builder_mut().create::<StoreLocal, _>(span);
        op_builder(local, value)
    }

    fn builder(&self) -> &B;
    fn builder_mut(&mut self) -> &mut B;
}

/// Returns an error when a `T` named `name` cannot be defined in the symbol table `builder`
/// inserts into: a symbol there has that name or the same path (see
/// `<dyn SymbolTable>::can_define`), or the new symbol would be at or below the
/// path of a world-level component (see [World::reject_symbol_shadowed_by_component]).
///
/// A builder without a symbol table at its insertion point defines anything. Inserting a symbol
/// that conflicts would otherwise fail an assertion.
fn ensure_definable<T: OpRegistration, B: ?Sized + Builder>(
    builder: &B,
    name: SymbolName,
) -> Result<(), Report> {
    let Some(owner) = builder.insertion_block().and_then(|block| block.borrow().parent_op()) else {
        return Ok(());
    };
    // The table the op is registered in on insertion, see `Operation::nearest_symbol_table`
    let table = if owner.name().implements::<dyn SymbolTable>() {
        owner
    } else {
        let Some(table) = owner.borrow().nearest_symbol_table() else {
            return Ok(());
        };
        table
    };
    let kind = builder
        .context()
        .get_or_register_dialect::<T::Dialect>()
        .expect_registered_name::<T>();
    table
        .borrow()
        .as_trait::<dyn SymbolTable>()
        .expect("a symbol table")
        .can_define(name, &kind)
        .map_err(Report::msg)?;
    World::reject_symbol_shadowed_by_component(table, name, &kind)
}

impl<'f, B: ?Sized + Builder> BuiltinOpBuilder<'f, B> for FunctionBuilder<'f, B> {
    #[inline(always)]
    fn builder(&self) -> &B {
        FunctionBuilder::builder(self)
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut B {
        FunctionBuilder::builder_mut(self)
    }
}

impl<'f> BuiltinOpBuilder<'f, OpBuilder> for &'f mut OpBuilder {
    #[inline(always)]
    fn builder(&self) -> &OpBuilder {
        self
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut OpBuilder {
        self
    }
}

impl<B: ?Sized + Builder> BuiltinOpBuilder<'_, B> for B {
    #[inline(always)]
    fn builder(&self) -> &B {
        self
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut B {
        self
    }
}
