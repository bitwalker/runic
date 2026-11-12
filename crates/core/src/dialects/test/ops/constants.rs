use crate::{
    derive::{EffectOpInterface, operation},
    dialects::{builtin::attributes::IntAttr, test::TestDialect},
    effects::*,
    traits::*,
    *,
};

/// An integer constant, `test.constant -1 : i8`, whose result has the type of its value.
///
/// The test dialect materializes folded constants as this operation.
#[derive(EffectOpInterface)]
#[operation(
    dialect = TestDialect,
    traits(ConstantLike),
    implements(InferTypeOpInterface, Foldable, MemoryEffectOpInterface, OpPrinter)
)]
pub struct Constant {
    #[attr(hidden)]
    value: IntAttr,
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
        printer.print_int(*self.get_value());
        printer.print_space();
        printer.print_colon_type(self.result().ty());
    }
}

impl OpParser for Constant {
    fn parse(state: &mut OperationState, parser: &mut dyn OpAsmParser<'_>) -> ParseResult {
        use alloc::format;

        use crate::parse::{ParserError, ParserExt};

        let (literal_span, literal) = parser.parse_integer_literal()?.into_parts();
        let (ty_span, ty) = parser.parse_colon_type()?.into_parts();
        let span = SourceSpan::session(
            literal_span.source().id(),
            crate::diagnostics::TextRange::new(literal_span.range().start(), ty_span.range().end())
                .expect("valid range"),
        );

        if !ty.is_integer() {
            return Err(ParserError::InvalidOperationType {
                span,
                ty_span,
                reason: format!("expected integer type, got '{ty}'"),
            });
        }
        let value = literal.to_int(ty, literal_span)?;
        let attr = parser.context_rc().int_attr(value, ty);

        state.add_attribute("value", attr);
        state.results.push(ty);

        Ok(())
    }
}
