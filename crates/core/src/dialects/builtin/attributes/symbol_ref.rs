use core::fmt;

use crate::{
    AttrPrinter, AttributeRef, SymbolNameComponent, SymbolPath, SymbolUseRef,
    attributes::AttrParser, derive::DialectAttribute, dialects::builtin::BuiltinDialect,
    print::AsmPrinter,
};

/// A reference to a symbol by its path.
///
/// Two references are equal, and hash alike, when their paths are: the use that links a reference
/// to its symbol belongs to the operation holding the reference, so two operations that name the
/// same symbol hold equal references (and may be merged, as CSE merges equal operations).
///
/// A path identifies one symbol only under one nearest symbol table: equal references held by
/// operations under two different symbol tables may name two different symbols. Operations holding
/// references are therefore comparable only under one nearest symbol table; a consumer that compares
/// operations across symbol tables must resolve the references.
///
/// As an attribute ([SymbolRefAttr]), a reference holds no element: its path is data, its use is
/// its link to the symbol, and it has no type of its own. A replacement of an attribute's
/// elements ([crate::Attribute::replace_elements]) therefore never rebuilds a `SymbolRefAttr`,
/// and the reference stays linked to its symbol.
#[derive(DialectAttribute, Debug, Clone)]
#[attribute(name = "symbol", dialect = BuiltinDialect, implements(AttrPrinter))]
pub struct SymbolRef {
    /// The referenced path
    path: SymbolPath,
    /// The SymbolUse reference, established when we've linked the referenced operation to this use
    user: Option<SymbolUseRef>,
}

impl PartialEq for SymbolRef {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl Eq for SymbolRef {}

impl core::hash::Hash for SymbolRef {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.path.hash(state);
    }
}

impl Default for SymbolRef {
    fn default() -> Self {
        Self {
            path: SymbolPath::new([SymbolNameComponent::Root]).unwrap(),
            user: None,
        }
    }
}

impl AsRef<SymbolPath> for SymbolRef {
    fn as_ref(&self) -> &SymbolPath {
        &self.path
    }
}

impl AsMut<SymbolPath> for SymbolRef {
    fn as_mut(&mut self) -> &mut SymbolPath {
        &mut self.path
    }
}

impl SymbolRef {
    pub const fn new(path: SymbolPath, user: Option<SymbolUseRef>) -> Self {
        Self { path, user }
    }

    #[inline(always)]
    pub const fn path(&self) -> &SymbolPath {
        &self.path
    }

    #[inline(always)]
    pub fn path_mut(&mut self) -> &mut SymbolPath {
        &mut self.path
    }

    #[inline]
    pub fn set_path(&mut self, path: SymbolPath) {
        self.path = path;
    }

    /// The use this reference was linked through, or `None` if it was never linked to a symbol,
    /// as the parser leaves a reference that names no symbol.
    #[inline(always)]
    pub const fn try_user(&self) -> Option<SymbolUseRef> {
        self.user
    }

    pub fn set_user(&mut self, user: SymbolUseRef) {
        assert!(
            self.user.is_none_or(|user| !user.is_linked()),
            "attempted to replace symbol use without unlinking the previously used symbol first"
        );
        self.user = Some(user);
    }
}

impl fmt::Display for SymbolRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path)
    }
}

impl crate::formatter::PrettyPrint for SymbolRef {
    fn render(&self) -> crate::formatter::Document {
        crate::formatter::display(self)
    }
}

impl SymbolRefAttr {
    #[inline(always)]
    pub const fn path(&self) -> &SymbolPath {
        self.value.path()
    }

    #[inline]
    pub fn set_path(&mut self, path: SymbolPath) {
        self.value.set_path(path);
    }

    /// The use this reference was linked through, or `None` if it was never linked to a symbol,
    /// as the parser leaves a reference that names no symbol.
    #[inline(always)]
    pub const fn try_user(&self) -> Option<SymbolUseRef> {
        self.value.try_user()
    }

    pub fn set_user(&mut self, user: SymbolUseRef) {
        self.value.set_user(user);
    }

    /// The symbol this reference is linked to.
    ///
    /// Returns `None` if the reference has no tracked use, or its use is not linked to a symbol: it
    /// never was (the parser leaves a reference that names no symbol unlinked), or the symbol was
    /// removed from its symbol table or erased, or the operation holding the reference was erased.
    /// The path is not looked up again; in IR that verifies, the linked symbol is the one the path
    /// names, as [Operation::verify_symbol_refs](crate::Operation::verify_symbol_refs) checks.
    ///
    /// This resolves the *named* symbol only, without following aliases: if the path names a
    /// [`FunctionAlias`], the alias itself is returned. Use [Self::resolve_callable] to obtain both
    /// the named symbol and its canonical callable, or [`SymbolRef::resolve_canonical`] for other
    /// symbols.
    ///
    /// This immutably borrows the tracked use. Following the result borrows the symbol.
    ///
    /// [`FunctionAlias`]: crate::dialects::builtin::FunctionAlias
    /// [`SymbolRef::resolve_canonical`]: crate::SymbolRef::resolve_canonical
    pub fn resolve(&self) -> Option<crate::SymbolRef> {
        self.try_user()?.linked_symbol()
    }

    /// Resolve this reference as a callable, retaining its named symbol and canonical target.
    ///
    /// The named symbol is the one the reference is linked to ([Self::resolve]); aliases from it on
    /// are followed by path, each in the symbol table of the alias.
    pub fn resolve_callable(
        &self,
    ) -> Result<crate::ResolvedSymbolCallee, crate::SymbolResolutionError> {
        let user =
            self.try_user().ok_or_else(|| crate::SymbolResolutionError::UntrackedSymbol {
                path: self.path().clone(),
            })?;
        user.linked_symbol()
            .ok_or_else(|| crate::SymbolResolutionError::UnknownSymbol {
                path: self.path().clone(),
            })?
            .resolve_callable()
    }

    /// Unlinks the tracked use from the symbol it is linked to, if there is a use and it is linked.
    pub fn unlink(&mut self) {
        if let Some(user) = self.try_user() {
            user.unlink_from_symbol();
        }
    }

    /// Links the tracked use to `symbol` without modifying the stored path.
    ///
    /// # Panics
    ///
    /// Panics if this reference has no tracked use: a use records the operation that owns the
    /// reference, which the attribute alone does not know.
    pub fn link(&mut self, symbol: crate::SymbolRef) {
        self.try_user()
            .expect("a symbol reference without a tracked use cannot be linked")
            .link_to_symbol(symbol);
    }

    /// Updates this symbol reference to point at `symbol`, mutating and relinking the existing use.
    ///
    /// # Panics
    ///
    /// Panics if this reference has no tracked use; see [Self::link].
    pub fn set_symbol(&mut self, symbol: crate::SymbolRef) {
        self.unlink();
        self.set_path(symbol.borrow().path());
        self.link(symbol);
    }
}

impl AttrPrinter for SymbolRefAttr {
    fn print(&self, printer: &mut AsmPrinter<'_>) {
        printer.print_symbol_path(&self.path);
    }
}

impl AttrParser for SymbolRefAttr {
    fn parse(
        parser: &mut dyn crate::parse::Parser<'_>,
    ) -> crate::parse::ParseResult<crate::AttributeRef> {
        parser.parse_symbol_ref().map(|spanned| spanned.into_inner() as AttributeRef)
    }
}
