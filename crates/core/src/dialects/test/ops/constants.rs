use crate::{
    derive::{EffectOpInterface, operation},
    dialects::test::TestDialect,
    effects::*,
    traits::*,
    *,
};

/// An operation for expressing constant immediate values.
///
/// This is used to materialize folded constants for the HIR dialect.
#[derive(EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    traits(ConstantLike),
    implements(InferTypeOpInterface, Foldable, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Constant {
    #[attr(hidden)]
    value: ImmediateAttr,
    #[result]
    result: AnyInteger,
}

impl InferTypeOpInterface for Constant {
    fn infer_return_types(&mut self, _context: &Context) -> Result<(), Report> {
        let ty = self.value().ty();
        self.result_mut().set_type(ty);

        Ok(())
    }
}

impl Foldable for Constant {
    #[inline]
    fn fold(&self, results: &mut smallvec::SmallVec<[OpFoldResult; 1]>) -> FoldResult {
        results.push(OpFoldResult::Attribute(self.value));
        FoldResult::Ok(())
    }

    #[inline(always)]
    fn fold_with(
        &self,
        _operands: &[Option<AttributeRef>],
        results: &mut smallvec::SmallVec<[OpFoldResult; 1]>,
    ) -> FoldResult {
        self.fold(results)
    }
}

impl OpPrinter for Constant {
    fn print(&self, printer: &mut print::AsmPrinter<'_>) {
        printer.print_space();
        printer.print_decimal_integer(*self.get_value());
        printer.print_space();
        printer.print_colon_type(self.result().ty());
    }
}

impl OpParser for Constant {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        use alloc::format;

        use crate::{attributes::AttrParser, parse::ParserError};

        let start = parser.current_location();
        let imm = ImmediateAttr::parse(parser)?;
        let mut imm = imm.try_downcast_attr::<ImmediateAttr>().unwrap();

        let (ty_span, ty) = parser.parse_colon_type()?.into_parts();
        let span = SourceSpan::session(
            start.source().id(),
            crate::diagnostics::TextRange::new(start.range().start(), ty_span.range().end())
                .expect("valid range"),
        );

        let mut imm_mut = imm.borrow_mut();
        if ty.integer_width().is_none() {
            return Err(ParserError::InvalidOperationType {
                span,
                ty_span,
                reason: format!("expected integer type, got '{ty}'"),
            });
        }
        let new_value = imm_mut.as_value().coerced_to(ty);

        if let Some(new_value) = new_value {
            imm_mut.set_type(ty);
            *imm_mut.as_value_mut() = new_value;
        } else {
            return Err(ParserError::InvalidAttributeValue {
                span,
                reason: format!("could not convert value to type {ty}: value is out of range"),
            });
        }

        state.add_attribute("value", imm);
        state.results.push(ty);

        Ok(())
    }
}
