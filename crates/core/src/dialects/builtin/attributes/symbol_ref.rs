use core::fmt;

use crate::{
    AttrPrinter, AttributeRef, SymbolNameComponent, SymbolPath, SymbolUseRef,
    attributes::AttrParser, derive::DialectAttribute, dialects::builtin::BuiltinDialect,
    print::AsmPrinter,
};

#[derive(DialectAttribute, Debug, Clone, PartialEq, Eq, Hash)]
#[attribute(name = "symbol", dialect = BuiltinDialect, implements(AttrPrinter))]
pub struct SymbolRef {
    /// The referenced path
    path: SymbolPath,
    /// The SymbolUse reference, established when we've linked the referenced operation to this use
    user: Option<SymbolUseRef>,
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

    /// Resolves this symbol reference relative to the operation that owns the tracked use.
    ///
    /// Returns `None` if the reference has no tracked use or does not resolve.
    ///
    /// This resolves the *named* symbol only, without following aliases: if the path names a
    /// [`FunctionAlias`], the alias itself is returned. Use [Self::resolve_callable] to obtain both
    /// the named symbol and its canonical callable, or [`SymbolRef::resolve_canonical`] for other
    /// symbols.
    ///
    /// This immutably borrows the tracked user while resolving the containing symbol table. Callers
    /// must not hold a mutable borrow of that user when invoking this API, or the borrow check will
    /// report an aliasing violation.
    ///
    /// [`FunctionAlias`]: crate::dialects::builtin::FunctionAlias
    /// [`SymbolRef::resolve_canonical`]: crate::SymbolRef::resolve_canonical
    pub fn resolve(&self) -> Option<crate::SymbolRef> {
        self.try_user()?.resolve_symbol(self.path())
    }

    /// Resolve this reference as a callable, retaining its named symbol and canonical target.
    /// This has the same borrowing requirements as [Self::resolve].
    pub fn resolve_callable(
        &self,
    ) -> Result<crate::ResolvedSymbolCallee, crate::SymbolResolutionError> {
        let user =
            self.try_user().ok_or_else(|| crate::SymbolResolutionError::UntrackedSymbol {
                path: self.path().clone(),
            })?;
        let owner = user.borrow().owner;
        let table = owner
            .nearest_symbol_table()
            .ok_or(crate::SymbolResolutionError::NoSymbolTable)?;
        table
            .borrow()
            .as_symbol_table()
            .ok_or(crate::SymbolResolutionError::NoSymbolTable)?
            .resolve_callable(self.path())
    }

    /// Unlinks the tracked use from its current symbol, if there is a use and it is linked.
    pub fn unlink(&mut self) {
        if let Some(user) = self.try_user() {
            user.unlink_from_symbol(self.path());
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
