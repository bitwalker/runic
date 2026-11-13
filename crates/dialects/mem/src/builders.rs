use runic_core::{
    Builder, BuilderExt, OpBuilder, Report, SourceSpan, Type, UnsafeIntrusiveEntityRef, ValueRef,
    dialects::builtin::FunctionBuilder,
};

use crate::*;

pub trait MemOpBuilder<'f, B: ?Sized + Builder> {
    /// A `mem.load` through the typed pointer `addr`, of the pointee's type, at its natural
    /// alignment.
    fn load(&mut self, addr: ValueRef, span: SourceSpan) -> Result<ValueRef, Report> {
        self.load_aligned(addr, Alignment::Natural, span)
    }

    /// A `mem.load` through the typed pointer `addr`, of the pointee's type, assuming `align`.
    fn load_aligned(
        &mut self,
        addr: ValueRef,
        align: Alignment,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let unknown = self.builder().context().unknown_type();
        build_load(self.builder_mut(), addr, unknown, align, span)
    }

    /// A `mem.load` of a `ty` through `addr`, an opaque pointer, at its natural alignment.
    fn load_as(&mut self, addr: ValueRef, ty: Type, span: SourceSpan) -> Result<ValueRef, Report> {
        self.load_as_aligned(addr, ty, Alignment::Natural, span)
    }

    /// A `mem.load` of a `ty` through `addr`, an opaque pointer, assuming `align`.
    fn load_as_aligned(
        &mut self,
        addr: ValueRef,
        ty: Type,
        align: Alignment,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        build_load(self.builder_mut(), addr, ty, align, span)
    }

    /// A `mem.store` of `value` to `addr`, at its natural alignment.
    fn store(
        &mut self,
        value: ValueRef,
        addr: ValueRef,
        span: SourceSpan,
    ) -> Result<UnsafeIntrusiveEntityRef<Store>, Report> {
        self.store_aligned(value, addr, Alignment::Natural, span)
    }

    /// A `mem.store` of `value` to `addr`, assuming `align`.
    fn store_aligned(
        &mut self,
        value: ValueRef,
        addr: ValueRef,
        align: Alignment,
        span: SourceSpan,
    ) -> Result<UnsafeIntrusiveEntityRef<Store>, Report> {
        let op_builder = self.builder_mut().create::<Store, (_, _, _)>(span);
        op_builder(value, addr, align)
    }

    /// A `mem.alloca` of storage for a `ty`, at its natural alignment; returns its `ptr<ty>`.
    fn alloca(&mut self, ty: Type, span: SourceSpan) -> Result<ValueRef, Report> {
        self.alloca_aligned(ty, Alignment::Natural, span)
    }

    /// A `mem.alloca` of storage for a `ty`, aligned to `align`; returns its `ptr<ty>`.
    fn alloca_aligned(
        &mut self,
        ty: Type,
        align: Alignment,
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Alloca, (_, _)>(span);
        let op = op_builder(ty, align)?;
        Ok(op.borrow().result().as_value_ref())
    }

    fn builder(&self) -> &B;
    fn builder_mut(&mut self) -> &mut B;
}

/// A `mem.load` through `addr` with the property `ty` (unknown for a typed pointer) and `align`.
fn build_load<B: ?Sized + Builder>(
    builder: &mut B,
    addr: ValueRef,
    ty: Type,
    align: Alignment,
    span: SourceSpan,
) -> Result<ValueRef, Report> {
    let op_builder = builder.create::<Load, (_, _, _)>(span);
    let op = op_builder(addr, ty, align)?;
    Ok(op.borrow().result().as_value_ref())
}

impl<'f, B: ?Sized + Builder> MemOpBuilder<'f, B> for FunctionBuilder<'f, B> {
    #[inline(always)]
    fn builder(&self) -> &B {
        FunctionBuilder::builder(self)
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut B {
        FunctionBuilder::builder_mut(self)
    }
}

impl<'f> MemOpBuilder<'f, OpBuilder> for &'f mut OpBuilder {
    #[inline(always)]
    fn builder(&self) -> &OpBuilder {
        self
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut OpBuilder {
        self
    }
}

impl<B: ?Sized + Builder> MemOpBuilder<'_, B> for B {
    #[inline(always)]
    fn builder(&self) -> &B {
        self
    }

    #[inline(always)]
    fn builder_mut(&mut self) -> &mut B {
        self
    }
}
