use runic_core::{
    Builder, BuilderExt, OpBuilder, Report, SourceSpan, Type, UnsafeIntrusiveEntityRef, ValueRef,
    dialects::builtin::{FunctionBuilder, attributes::Array},
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

    /// A `mem.gep` from the typed pointer `base`: `index` (an `isize`, if given) steps over whole
    /// pointees, then `path` selects an element inside the value reached; returns the address of
    /// the element.
    fn gep(
        &mut self,
        base: ValueRef,
        index: Option<ValueRef>,
        path: &[u32],
        span: SourceSpan,
    ) -> Result<ValueRef, Report> {
        let path = path
            .iter()
            .map(|step| i32::try_from(*step))
            .collect::<Result<Array<i32>, _>>()
            .map_err(|_| Report::msg("a gep path step is at most i32::MAX"))?;
        let op_builder = self.builder_mut().create::<Gep, (_, Option<ValueRef>, _)>(span);
        let op = op_builder(base, index, path)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// A `mem.sizeof` of `ty`; returns the `isize` size.
    fn sizeof(&mut self, ty: Type, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Sizeof, (_,)>(span);
        let op = op_builder(ty)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// A `mem.alignof` of `ty`; returns the `isize` alignment.
    fn alignof(&mut self, ty: Type, span: SourceSpan) -> Result<ValueRef, Report> {
        let op_builder = self.builder_mut().create::<Alignof, (_,)>(span);
        let op = op_builder(ty)?;
        Ok(op.borrow().result().as_value_ref())
    }

    /// A `mem.offsetof` of the element `path` selects in `ty`; returns the `isize` offset.
    fn offsetof(&mut self, ty: Type, path: &[u32], span: SourceSpan) -> Result<ValueRef, Report> {
        let path = path
            .iter()
            .map(|step| i32::try_from(*step))
            .collect::<Result<Array<i32>, _>>()
            .map_err(|_| Report::msg("an offsetof path step is at most i32::MAX"))?;
        let op_builder = self.builder_mut().create::<Offsetof, (_, _)>(span);
        let op = op_builder(ty, path)?;
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
