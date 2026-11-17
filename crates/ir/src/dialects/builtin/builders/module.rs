use super::BuiltinOpBuilder;
use crate::{
    AsCallableSymbolRef, Builder, Ident, Op, OpBuilder, Report, ResolvedSymbolCallee, SourceSpan,
    SymbolName, SymbolNameComponent, SymbolPath, SymbolResolutionError, SymbolTable, Type,
    UnsafeIntrusiveEntityRef, Visibility,
    constants::ConstantData,
    dialects::builtin::{
        Function, FunctionAlias, FunctionAliasRef, FunctionRef, GlobalVariable, GlobalVariableRef,
        Module, ModuleRef, Segment, attributes::Signature,
    },
};

/// A specialized builder for constructing/modifying [crate::dialects::builtin::Module]
pub struct ModuleBuilder {
    pub module: ModuleRef,
    builder: OpBuilder,
}
impl ModuleBuilder {
    /// Create a builder over `module`
    pub fn new(module: ModuleRef) -> Self {
        let module_ref = module.borrow();
        let context = module_ref.as_operation().context_rc();
        let mut builder = OpBuilder::new(context);

        {
            let body = module_ref.body();

            if let Some(current_block) = body.entry_block_ref() {
                builder.set_insertion_point_to_end(current_block);
            } else {
                let body_ref = body.as_region_ref();
                drop(body);
                builder.create_block(body_ref, None, &[]);
            }
        }

        Self { module, builder }
    }

    /// Get the underlying [OpBuilder]
    pub fn builder(&mut self) -> &mut OpBuilder {
        &mut self.builder
    }

    /// Declare a new [crate::dialects::builtin::Function] in this module with the given name and
    /// signature.
    ///
    /// The returned [FunctionRef] can be used to construct a [super::FunctionBuilder] to define the
    /// body of the function.
    pub fn define_function(
        &mut self,
        name: Ident,
        visibility: Visibility,
        signature: Signature,
    ) -> Result<FunctionRef, Report> {
        self.builder.create_function(name, visibility, signature)
    }

    /// Define a [crate::dialects::builtin::FunctionAlias] in this module.
    pub fn define_function_alias<C: AsCallableSymbolRef>(
        &mut self,
        name: Ident,
        visibility: Visibility,
        target: C,
    ) -> Result<FunctionAliasRef, Report> {
        self.builder.create_function_alias(name, visibility, target)
    }

    /// Declare a new [GlobalVariable] in this module with the given name, visibility, and type.
    ///
    /// The global is a declaration: the returned [UnsafeIntrusiveEntityRef] can be used to
    /// construct a builder over the body of the global variable initializer region, which ends
    /// with a [Yield](crate::dialects::builtin::Yield) of a value of the global's type.
    pub fn define_global_variable(
        &mut self,
        name: Ident,
        visibility: Visibility,
        ty: Type,
    ) -> Result<UnsafeIntrusiveEntityRef<GlobalVariable>, Report> {
        let global_var_ref = self.builder.create_global_variable(name, visibility, ty)?;
        Ok(global_var_ref)
    }

    pub fn define_data_segment(
        &mut self,
        offset: u64,
        data: impl Into<ConstantData>,
        readonly: bool,
        span: SourceSpan,
    ) -> Result<UnsafeIntrusiveEntityRef<Segment>, Report> {
        self.builder.create_data_segment(offset, data, readonly, span)
    }

    /// Get the function named `name`, without resolving aliases.
    ///
    /// Returns `None` if the name is missing or refers to another kind of symbol.
    pub fn get_function(&self, name: &str) -> Option<FunctionRef> {
        let symbol = SymbolName::intern(name);
        let symbol_ref = self.module.borrow().get(symbol)?;
        let op = symbol_ref.borrow();
        op.as_symbol_operation()
            .downcast_ref::<Function>()
            .map(Function::as_function_ref)
    }

    /// Resolve `name` to a function, following any function aliases.
    ///
    /// Returns `None` if the name or an alias target cannot be resolved, the alias chain is
    /// cyclic or too deep, or the resolved symbol is not a function.
    pub fn resolve_function(&self, name: &str) -> Option<FunctionRef> {
        let symbol = SymbolName::intern(name);
        let symbol_ref = self.module.borrow().get(symbol)?;
        symbol_ref.resolve_function().ok()
    }

    /// Resolve a callable name, retaining both the named symbol and its canonical callable.
    pub fn resolve_callable(
        &self,
        name: &str,
    ) -> Result<ResolvedSymbolCallee, SymbolResolutionError> {
        let path = SymbolPath::from_iter([SymbolNameComponent::Leaf(SymbolName::intern(name))]);
        self.module.borrow().resolve_callable(&path)
    }

    /// Get the alias op itself, without resolving its target.
    pub fn get_function_alias(&self, name: &str) -> Option<FunctionAliasRef> {
        let symbol = SymbolName::intern(name);
        let symbol_ref = self.module.borrow().get(symbol)?;
        let op = symbol_ref.borrow();
        op.as_symbol_operation()
            .downcast_ref::<FunctionAlias>()
            .map(|alias| alias.as_function_alias_ref())
    }

    pub fn get_global_var(&self, name: SymbolName) -> Option<GlobalVariableRef> {
        self.module.borrow().get(name).and_then(|gv_symbol| {
            let op_ref = gv_symbol.borrow().as_operation_ref();
            op_ref
                .borrow()
                .downcast_ref::<GlobalVariable>()
                .map(|gv| gv.as_global_var_ref())
        })
    }

    /// Declare a new nested module named by the raw segment `name`, see
    /// [BuiltinOpBuilder::create_module].
    pub fn declare_module(&mut self, name: Ident) -> Result<ModuleRef, Report> {
        self.builder.create_module(name)
    }

    /// Resolve the nested module named by the raw segment `name` (see [Module::canonical_name]),
    /// if declared/defined
    pub fn find_module(&self, name: SymbolName) -> Option<ModuleRef> {
        let name = Module::canonical_name(name).ok()?;
        (&*self.module.borrow() as &dyn SymbolTable).find::<Module>(name)
    }
}
