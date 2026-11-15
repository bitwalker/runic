use crate::{
    Context, Ident, IdentAttr, Op, OpParser, OpPrinter, Operation, RegionKind, RegionKindInterface,
    Report, Symbol, SymbolManager, SymbolManagerMut, SymbolMap, SymbolNameComponent, SymbolPath,
    SymbolRef, SymbolTable, SymbolTableKind, SymbolUseList, UnsafeIntrusiveEntityRef, Usable,
    Verify, Visibility,
    derive::operation,
    dialects::builtin::{BuiltinDialect, attributes::VisibilityAttr},
    interner,
    layout::DataLayoutOpInterface,
    print::AsmPrinter,
    traits::{
        GraphRegionNoTerminator, HasOnlyGraphRegion, IsolatedFromAbove, NoRegionArguments,
        NoTerminator, SingleBlock, SingleRegion,
    },
};

pub type ComponentRef = UnsafeIntrusiveEntityRef<Component>;

/// A [Component] is a modular abstraction operation, i.e. it is designed to model shared-nothing
/// boundaries between groups of shared-everything modules in a system.
///
/// Components can contain the following entities:
///
/// * [super::Function] used to export standalone component-level functions, e.g. a program
///   entrypoint, or component initializer. These functions always have `Public` visibility, and
///   must be representable using the Canonical ABI.
/// * [super::Module], used to implement the functionality exported by a component-level
///   [super::Function]. Modules may not have `Public` visibility. All modules
///   within a [Component] are within the same shared-everything boundary, so conflicting data
///   segment declarations are not allowed. Additionally, global variables within the same
///   shared-everything boundary are allocated in the same linear memory address space.
///
/// Externally-defined functions are represented as declarations, and must be referenced using their
/// fully-qualified name in order to resolve them.
///
/// ## Linking
///
/// NOTE: Components always have `Public` visibility.
#[operation(
    dialect = BuiltinDialect,
    traits(
        SingleRegion,
        SingleBlock,
        NoRegionArguments,
        NoTerminator,
        HasOnlyGraphRegion,
        GraphRegionNoTerminator,
        IsolatedFromAbove,
    ),
    implements(RegionKindInterface, SymbolTable, Symbol, DataLayoutOpInterface, OpPrinter)
)]
pub struct Component {
    #[attr]
    name: IdentAttr,
    #[attr]
    #[default]
    visibility: VisibilityAttr,
    #[region]
    body: RegionRef,
    #[default]
    symbols: SymbolMap,
    #[default]
    uses: SymbolUseList,
}

impl OpPrinter for Component {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_space();
        printer.print_keyword(self.get_visibility().as_str());
        printer.print_space();
        // Printed one `@`-prefixed segment per segment of the name (`@acme::@a::@b`,
        // `@"acme::pkg"::@app`), the symbol-path form the parser reads back and re-joins.
        let path = SymbolPath::from_iter(
            SymbolPath::segments_of(Symbol::name(self))
                .into_iter()
                .map(SymbolNameComponent::Component),
        );
        printer.print_symbol_path(&path);
        printer.print_space();
        if self.op.has_attributes() {
            *printer += crate::formatter::const_text("attributes ");
            printer.print_attribute_dictionary(
                self.op.attributes().iter().map(|attr| *attr.as_named_attribute()),
            );
            printer.print_space();
        }
        printer.print_region(&self.body());
    }
}

impl OpParser for Component {
    fn parse(
        state: &mut crate::OperationState,
        parser: &mut dyn crate::OpAsmParser<'_>,
    ) -> crate::ParseResult {
        use crate::parse::Token;

        let context = parser.context_rc();
        let visibility = parser
            .parse_keyword_from(&[
                Token::BareIdent("public"),
                Token::BareIdent("private"),
                Token::BareIdent("internal"),
            ])?
            .into_inner()
            .parse::<Visibility>()
            .expect("one or more of these visibilities are no longer valid");
        state
            .add_attribute("visibility", context.create_attribute::<VisibilityAttr, _>(visibility));

        // The name is the canonical spelling of the whole (possibly multi-segment) path of raw
        // segments (see `SymbolPath::to_symbol_name`)
        let (span, path) = parser.parse_symbol_path()?.into_parts();
        let name = path
            .to_symbol_name()
            .map_err(|err| crate::parse::ParserError::invalid_symbol_table_name(span, err))?;
        state.add_attribute(
            "name",
            context.create_attribute::<IdentAttr, _>(Ident::new(name, span)),
        );

        parser.parse_optional_attribute_dict_with_keyword(&mut state.attrs)?;

        let region = parser.context().create_region();
        parser.parse_region(region, &[], true)?;
        state.add_region(region);

        Ok(())
    }
}

/// Components are built through the builtin builders and the parser, which spell and validate
/// their names; a name set afterwards is not, so it is checked here as well (see
/// [SymbolPath::segments_of]).
///
/// The error is located at the component.
impl Verify<dyn SymbolTable> for Component {
    fn verify(&self, _context: &Context) -> Result<(), Report> {
        use alloc::string::ToString;

        use crate::Spanned;

        SymbolPath::validate_table_name(SymbolTableKind::Component, Symbol::name(self)).map_err(|err| {
            crate::diagnostics::report!(
                message: (err.to_string()),
                labels: [primary(self.as_operation().span(), "this component's name is not valid")],
            )
        })
    }
}

/// Reads its specification from its attribute dictionary.
impl DataLayoutOpInterface for Component {}

impl RegionKindInterface for Component {
    #[inline(always)]
    fn kind(&self) -> RegionKind {
        RegionKind::Graph
    }
}

impl Usable for Component {
    type Use = crate::SymbolUse;

    #[inline(always)]
    fn uses(&self) -> &SymbolUseList {
        &self.uses
    }

    #[inline(always)]
    fn uses_mut(&mut self) -> &mut SymbolUseList {
        &mut self.uses
    }
}

impl Symbol for Component {
    #[inline(always)]
    fn as_symbol_operation(&self) -> &Operation {
        &self.op
    }

    #[inline(always)]
    fn as_symbol_operation_mut(&mut self) -> &mut Operation {
        &mut self.op
    }

    fn name(&self) -> interner::Symbol {
        self.get_name().as_symbol()
    }

    fn set_name(&mut self, name: interner::Symbol) {
        self.name_mut().name = name;
    }

    fn visibility(&self) -> Visibility {
        *self.get_visibility()
    }

    fn set_visibility(&mut self, visibility: Visibility) {
        *self.get_visibility_mut() = visibility;
    }
}

impl SymbolTable for Component {
    #[inline(always)]
    fn as_symbol_table_operation(&self) -> &Operation {
        &self.op
    }

    #[inline(always)]
    fn as_symbol_table_operation_mut(&mut self) -> &mut Operation {
        &mut self.op
    }

    fn symbol_manager(&self) -> SymbolManager<'_> {
        SymbolManager::new(&self.op, crate::Symbols::Borrowed(&self.symbols))
    }

    fn symbol_manager_mut(&mut self) -> SymbolManagerMut<'_> {
        SymbolManagerMut::new(&mut self.op, crate::SymbolsMut::Borrowed(&mut self.symbols))
    }

    #[inline]
    fn get(&self, name: interner::Symbol) -> Option<SymbolRef> {
        self.symbols.get(name)
    }
}

impl Component {
    /// Name of the optional operation attribute (a `BoolAttr`) marking a component the compiler
    /// invented to wrap a bare core module, rather than one an author wrote.
    ///
    /// This is a marker rather than a name comparison because the wrapper has no name of its own:
    /// it is named by the target namespace (or by the wrapped module when there is none), which is
    /// a name an author may equally give a component of theirs, with the module visibility they
    /// declared.
    pub const SYNTHETIC_WRAPPER_ATTR: &'static str = "synthetic_wrapper";

    /// Returns the name a component given `name` gets: `name` itself when it already is the
    /// canonical spelling of its segments (`acme::a::b`, `"acme::pkg"::app`), otherwise the
    /// spelling of `name` read as one raw segment (`masm-dep` is named `"masm-dep"`).
    ///
    /// Returns an error when the spelling rules reject a segment (see
    /// [SymbolPath::is_spellable]).
    pub fn canonical_name(name: interner::Symbol) -> Result<interner::Symbol, Report> {
        SymbolPath::table_name(SymbolTableKind::Component, name)
    }

    /// Returns the absolute path this component is rooted at, with one component per segment of
    /// its name (see [SymbolPath::segments_of]); every symbol inside the component has it as a
    /// prefix.
    ///
    /// `Symbol::path()` of the component itself has the same segments, its last one being a
    /// `Leaf`.
    pub fn namespace_path(&self) -> SymbolPath {
        SymbolPath::from_symbol_name(Symbol::name(self))
    }

    /// Mark this component as the compiler's wrapper around a bare core module.
    ///
    /// The marker is a plain attribute, but setting it by hand is three lines of attribute
    /// plumbing that say nothing about what is being asserted, and it is asserted from the
    /// frontend that invents the wrapper and from every test standing in for that frontend. This
    /// is the counterpart of [`Component::is_synthetic_wrapper`], and the two belong together:
    /// whatever the marker's representation is, both ends of it should change at once.
    pub fn mark_synthetic_wrapper(&mut self) {
        let marker = self
            .as_operation()
            .context_rc()
            .create_attribute::<crate::dialects::builtin::attributes::BoolAttr, _>(true);
        self.as_operation_mut().set_attribute(Self::SYNTHETIC_WRAPPER_ATTR, marker);
    }

    /// Returns true if this component is the compiler's wrapper around a bare core module.
    pub fn is_synthetic_wrapper(&self) -> bool {
        self.as_operation()
            .get_typed_attribute::<crate::dialects::builtin::attributes::BoolAttr>(
                Self::SYNTHETIC_WRAPPER_ATTR,
            )
            .is_some_and(|attr| **attr.borrow())
    }

    #[inline(always)]
    pub fn as_component_ref(&self) -> ComponentRef {
        unsafe { ComponentRef::from_raw(self) }
    }
}
