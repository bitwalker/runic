use alloc::{rc::Rc, string::String};
use core::{cell::RefCell, ops::AddAssign};

use super::*;
use crate::{
    AsValueRange, Attribute, Block, BlockRef, Context, EntityList, FunctionType, FxHashMap, Int,
    Location, NamedAttribute, SuccessorOperands, SymbolPath, Type, ValueRef,
    attributes::Marker,
    formatter::{Document, concat_balanced},
    interner,
    types::{Aliased, TypeAliases, collect_rec_groups},
};

/// [AsmPrinter] provides utilities for pretty-printing an operation in either the generic
/// or custom formats. It provides access to the current printer flags, and manages the output
/// document to ensure that custom printers adhere to the requirements expected of all printers.
///
/// The aliases of recursion-group members are kept in a table shared by every printer of one
/// print: a printer that is created while printing, to print a part of the output separately,
/// must be made with [Self::nested].
///
/// A printer made with [Self::root] is the *root* of its print: [Self::finish] writes the
/// declarations of the recursion groups the print refers to in front of the printed text, so
/// that the text parses on its own. A printer made with [Self::nested] or [Self::fragment]
/// never writes declarations.
pub struct AsmPrinter<'a> {
    /// The position of each block in its region's stored order, for the regions whose order this
    /// print has needed (see [Self::block_position]); shared by every printer of one print.
    block_positions: Rc<RefCell<FxHashMap<BlockRef, usize>>>,
    context: Rc<Context>,
    flags: &'a OpPrintingFlags,
    document: Pieces,
    aliases: Rc<RefCell<TypeAliases>>,
    /// Whether [Self::finish] writes the print's recursion-group declarations.
    root: bool,
}

impl<'a> AsmPrinter<'a> {
    /// Construct a new printer with the given [OpPrintingFlags], starting a print of its own whose
    /// root it is: [Self::finish] precedes the printed text with its recursion-group declarations.
    ///
    /// A printer created while printing, to print a part of the output separately, is made with
    /// [Self::nested] instead, so that it shares this print's alias table.
    pub fn root(context: Rc<Context>, flags: &'a OpPrintingFlags) -> Self {
        Self {
            block_positions: Rc::default(),
            context,
            flags,
            document: Pieces::default(),
            aliases: Rc::default(),
            root: true,
        }
    }

    /// Construct a new printer that starts a print of its own, like [Self::root], but prints a
    /// fragment: [Self::finish] writes no declarations, and the recursion-group members in its
    /// text are aliases that it does not declare.
    pub fn fragment(context: Rc<Context>, flags: &'a OpPrintingFlags) -> Self {
        Self {
            root: false,
            ..Self::root(context, flags)
        }
    }

    /// A printer with an empty document that is part of the same print as this one: it has the
    /// same context and flags, shares this printer's alias table, and is not a root.
    pub fn nested(&self) -> AsmPrinter<'a> {
        Self {
            block_positions: self.block_positions.clone(),
            context: self.context.clone(),
            flags: self.flags,
            document: Pieces::default(),
            aliases: self.aliases.clone(),
            root: false,
        }
    }

    #[inline(always)]
    pub const fn flags(&self) -> &OpPrintingFlags {
        self.flags
    }

    /// The position of `block` in the stored order of the blocks of its region, which is the order
    /// they are printed in; `0` for a block in no region.
    ///
    /// The positions of a region's blocks are computed together, the first time one of them is
    /// asked for, and kept for the rest of the print: the IR must not change while it is printed.
    pub(super) fn block_position(&self, block: BlockRef) -> usize {
        if let Some(position) = self.block_positions.borrow().get(&block) {
            return *position;
        }
        let Some(region) = block.borrow().parent() else {
            return 0;
        };
        let mut positions = self.block_positions.borrow_mut();
        for (position, block) in region.borrow().body().iter().enumerate() {
            positions.insert(block.as_block_ref(), position);
        }
        positions.get(&block).copied().unwrap_or(0)
    }

    /// Get a reference to the [Context] this printer was instantiated with
    #[inline(always)]
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Get a reference to the [Context] this printer was instantiated with
    #[inline]
    pub fn context_rc(&self) -> Rc<Context> {
        self.context.clone()
    }

    /// Render the current printer state as a [Document], and reset the printer to an empty buffer.
    ///
    /// This can be used to emit multiple separate documents using a single [AsmPrinter] instance.
    /// The result is always a fragment: it never holds recursion-group declarations, even on a
    /// root printer, so that the parts of a text assembled from several renders do not have
    /// declarations in the middle. A root writes its declarations only in [Self::finish].
    #[inline]
    pub fn render(&mut self) -> Document {
        self.document.take()
    }

    /// Consume the printer, and render its state as a [Document].
    ///
    /// On a root printer (see [Self::root]) whose print refers to recursion groups, the document is
    /// [Self::rec_group_declarations], a blank line, then the printed text.
    pub fn finish(mut self) -> Document {
        use crate::formatter::nl;

        if !self.root || self.aliases.borrow().groups().is_empty() {
            return self.document.take();
        }
        let declarations = self.rec_group_declarations();
        declarations + nl() + nl() + self.document.take()
    }
}

impl<'a> AsmPrinter<'a> {
    /// Print `op` using its custom format if available, falling back to the generic format.
    ///
    /// See the [print](crate::print) module docs for the details of the generic format.
    pub fn print_operation(&mut self, op: impl AsRef<Operation>) {
        op.as_ref().print(self);
    }

    /// Print `op` using the generic assembly format
    ///
    /// See the [print](crate::print) module docs for the details of the generic format.
    pub fn print_operation_generic(&mut self, op: &Operation) {
        use crate::formatter::*;

        let context = op.context();
        self.print_results(op.results().all());

        self.document += text(format!("\"{}\"", op.name()));

        // The operand list holds the operands of every group that no successor forwards, in group
        // order; each successor prints the arguments of its own group. The signature divides the
        // operand types into those groups when there is more than one.
        let mut forwarded = alloc::vec![false; op.operands().num_groups()];
        for successor in op.successors().all().iter() {
            if let Some(group) = forwarded.get_mut(successor.successor_operand_group()) {
                *group = true;
            }
        }
        let own_groups = (0..op.operands().num_groups()).filter(|&g| !forwarded[g]);
        let mut operands = crate::SmallVec::<[ValueRef; 4]>::new();
        let mut operand_types = crate::SmallVec::<[crate::SmallVec<[Type; 4]>; 1]>::new();
        for group in own_groups {
            let group = op.operands().group(group);
            operands.extend(group.iter().map(|operand| operand.borrow().as_value_ref()));
            operand_types.push(group.iter().map(|operand| operand.borrow().ty()).collect());
        }
        let operands: ValueRange<'static, 4> = operands.into();
        self.document += const_text("(");
        self.print_value_uses(operands);
        self.document += const_text(")");
        if op.has_successors() {
            self.document += const_text(" ");
            self.print_generic_successors(op);
        }

        // Print properties
        if op.has_properties() {
            self.document += const_text(" <");
            self.print_attribute_dictionary(op.properties());
            self.document += const_text(">");
        }

        // Print regions, in the order they are stored
        if op.has_regions() {
            self.document += const_text(" (");
            let mut printer = self.nested();
            let doc = join_balanced(
                op.regions().iter().map(|region| {
                    printer.print_region_in_form(&region, /*generic=*/ true);
                    printer.render()
                }),
                || const_text(", "),
            );
            self.document += doc;
            self.document += const_text(")");
        }

        // Print attributes
        let attrs = op.attributes();
        if !attrs.is_empty() {
            self.document += const_text(" ");
            self.print_attribute_dictionary(
                op.attributes().iter().map(|attr| *attr.as_named_attribute()),
            );
        }

        // Print operation type
        self.document += const_text(" : ");
        // A lone operand type prints without parentheses, unless it is a function type, which
        // would otherwise run into the `->` of the operation's type. Groups are separated by `|`.
        match operand_types.as_slice() {
            [group] => match group.as_slice() {
                [ty] if !ty.is_function() => self.print_type(*ty),
                _ => self.print_type_list(group.iter().copied()),
            },
            groups => {
                self.document += const_text("(");
                for (index, group) in groups.iter().enumerate() {
                    if index > 0 {
                        self.document += if group.is_empty() {
                            const_text(" |")
                        } else {
                            const_text(" | ")
                        };
                    }
                    self.print_types(group.iter().copied());
                }
                self.document += const_text(")");
            }
        }

        self.document += const_text(" ");
        self.print_arrow_type_list(
            /*elide_single_type_parens=*/ true,
            op.results().all().iter().map(|result| result.borrow().ty()),
        );

        // Add source location if requested, separated from the type as in the custom form
        if self.flags.print_source_locations {
            let loc = Location::from_span(op.span, context);
            self.print_space();
            self.print_trailing_location_specifier(&loc);
        }

        self.document += const_text(";");
    }

    /// Print the successors of `op` in the generic form, in `[` `]`: each with its key, if it has
    /// one, and the arguments it forwards with their types, the successor groups separated by
    /// `|` when there is more than one.
    fn print_generic_successors(&mut self, op: &Operation) {
        use crate::formatter::*;

        let successors = op.successors();
        self.document += const_text("[");
        let mut groups = alloc::vec::Vec::with_capacity(successors.num_groups());
        let mut printer = self.nested();
        for group in successors.groups() {
            let doc = join_balanced(
                group.iter().map(|successor| {
                    if let Some(key) = successor.key {
                        printer.print_attribute_value(&*key.borrow());
                        printer.document += const_text(": ");
                    }
                    printer.document += display(successor.successor().borrow().id());
                    printer.print_successor_arguments(successor.successor_operands());
                    printer.render()
                }),
                || const_text(", "),
            );
            groups.push(doc);
        }
        for (index, group) in groups.into_iter().enumerate() {
            if index > 0 {
                self.document += const_text(" |");
            }
            if !group.is_empty() {
                self.document += const_text(" ") + group;
            }
        }
        self.document += const_text(" ]");
    }

    /// Prints operation results ungrouped, i.e. `%0, %1, %2 =`
    ///
    /// If an operation has no results, this is a no-op.
    pub fn print_results(&mut self, results: OpResultRange<'_>) {
        use crate::formatter::*;

        if results.is_empty() {
            return;
        }

        let doc = join_balanced(results.iter().map(|result| display(result.borrow().id())), || {
            const_text(", ")
        });

        if doc.is_empty() {
            self.document += doc;
        } else {
            self.document += doc + const_text(" = ")
        }
    }

    /// Prints operation operands in parentheses, i.e. `(%0, %1)`
    pub fn print_operand_list(&mut self, operands: OpOperandRange<'_>) {
        use crate::formatter::*;

        self.document += const_text("(");
        self.print_value_uses(operands.as_value_range());
        self.document += const_text(")");
    }

    /// Prints zero or more comma-separated operands (i.e. used values)
    pub fn print_value_uses<const N: usize>(&mut self, values: ValueRange<'_, N>) {
        use crate::formatter::*;

        let doc = join_balanced(values.iter().map(|value| display(value.borrow().id())), || {
            const_text(", ")
        });
        self.document += doc;
    }

    /// Prints a successor argument list as an `ssa-use-and-type-list`, i.e. the operand uses, a
    /// colon, then their types: `(%0, %1 : u32, u32)`. This matches the grammar accepted by
    /// `parse_successor_and_use_list`.
    ///
    /// Prints nothing when `values` is empty, as the parser treats the entire list as optional.
    pub fn print_successor_arguments<const N: usize>(&mut self, values: ValueRange<'_, N>) {
        use crate::formatter::*;

        if values.is_empty() {
            return;
        }
        let types = values
            .iter()
            .map(|value| value.borrow().ty())
            .collect::<crate::SmallVec<[_; 4]>>();
        self.document += const_text("(");
        self.print_value_uses(values);
        self.document += const_text(" : ");
        self.print_types(types);
        self.document += const_text(")");
    }

    /// Prints zero or more comma-separated value ids and their types in parentheses.
    ///
    /// This is intended for use in parameter lists (e.g. block arguments).
    ///
    /// See [`Self::print_value_ids_and_types`] for more information on correct usage of this method.
    pub fn print_value_id_and_type_list(
        &mut self,
        values: impl ExactSizeIterator<Item = ValueRef>,
    ) {
        use crate::formatter::*;

        self.document += const_text("(");
        self.print_value_ids_and_types(values);
        self.document += const_text(")");
    }

    /// Prints zero or more comma-separated value ids and their types.
    ///
    /// This differs from [`Self::print_value_uses`] in that it does not allow value indexing, e.g.
    /// `%value#0` or value packing, e.g. `%value:2`. Any attempt to print such values with this
    /// method will panic.
    pub fn print_value_ids_and_types(&mut self, values: impl ExactSizeIterator<Item = ValueRef>) {
        use crate::formatter::*;

        let doc = join_balanced(
            values.map(|value| {
                let value = value.borrow();
                display(value.id()) + const_text(": ") + self.format_type(value.ty())
            }),
            || const_text(", "),
        );
        self.document += doc;
    }

    /// Prints zero or more types in parentheses, i.e. `(i1, i32)`
    pub fn print_type_list(&mut self, types: impl IntoIterator<Item = Type>) {
        use crate::formatter::*;

        self.document += const_text("(");
        self.print_types(types);
        self.document += const_text(")");
    }

    /// Prints a colon and then zero or more types in parentheses, i.e. `: (i1, i32)`
    pub fn print_colon_type_list(&mut self, types: impl IntoIterator<Item = Type>) {
        use crate::formatter::*;

        self.document += const_text(": ");
        self.print_type_list(types);
    }

    /// Prints a colon and then a single type
    pub fn print_colon_type(&mut self, ty: Type) {
        use crate::formatter::*;

        self.document += const_text(": ");
        self.print_type(ty);
    }

    /// Prints an arrow and then zero or more types in parentheses, i.e. `-> (i1, i32)`.
    ///
    /// If `elide_single_type_parens` is true, then if the input collection only has a single type,
    /// then the parentheses will be elided, unless that type is a function type: `-> (() -> i32)`
    /// keeps them, so that the result cannot be read as the start of a longer function type.
    pub fn print_arrow_type_list(
        &mut self,
        elide_single_type_parens: bool,
        types: impl IntoIterator<Item = Type>,
    ) {
        use crate::formatter::*;

        self.document += const_text("-> ");
        let types = types.into_iter().collect::<crate::SmallVec<[Type; 4]>>();
        match types.as_slice() {
            [ty] if elide_single_type_parens && !ty.is_function() => self.print_type(*ty),
            _ => self.print_type_list(types),
        }
    }

    /// Prints zero or more comma-separated types
    pub fn print_types(&mut self, types: impl IntoIterator<Item = Type>) {
        use crate::formatter::*;

        let doc =
            join_balanced(types.into_iter().map(|ty| self.format_type(ty)), || const_text(", "));
        self.document += doc;
    }

    /// Print a single type
    pub fn print_type(&mut self, ty: Type) {
        self.document += self.format_type(ty);
    }

    /// `ty` as text, with its recursion-group members written as their aliases.
    fn format_type(&self, ty: Type) -> Document {
        crate::formatter::text(alloc::format!(
            "{}",
            Aliased {
                ty,
                aliases: &self.aliases,
                definition: false,
            }
        ))
    }

    /// The declarations of the recursion groups whose members this print has written so far, and
    /// of every group they refer to, as `rec { type !alias = definition; .. }` blocks separated
    /// by newlines; an empty document if there are none.
    ///
    /// The lines are joined as a balanced tree, so that the size of a group or the number of
    /// groups does not make a deep document.
    ///
    /// Each group comes after every group its members refer to, so that the text parses in one
    /// pass; a group that is only referred to is named here, before any declaration is written,
    /// so that no name already printed changes. Members are declared in index order.
    ///
    /// The groups are all those the shared alias table has named, not only those named since an
    /// earlier call: a second call, or [Self::finish] after a call, writes every group again. A
    /// printer used for several roots in turn therefore declares, for each, the groups of every
    /// root before it as well; print each root with a printer of its own to avoid that.
    pub fn rec_group_declarations(&self) -> Document {
        use crate::formatter::*;

        let queued = self.aliases.borrow().groups().to_vec();
        let order = collect_rec_groups(queued.iter().flat_map(|group| group.members().to_vec()));
        {
            let mut aliases = self.aliases.borrow_mut();
            for group in &order {
                aliases.alias_of(group.members()[0]);
            }
        }

        if order.is_empty() {
            return Document::Empty;
        }
        // One text node per line: a text node must not contain a newline.
        let mut lines = alloc::vec::Vec::new();
        for group in &order {
            lines.push(const_text("rec {"));
            for member in group.members() {
                let definition = Aliased {
                    ty: *member,
                    aliases: &self.aliases,
                    definition: true,
                };
                lines.push(text(alloc::format!(
                    "  type !{} = {definition};",
                    self.alias_of(*member)
                )));
            }
            lines.push(const_text("}"));
        }
        let mut parts = alloc::vec::Vec::with_capacity(lines.len() * 2);
        for (i, line) in lines.into_iter().enumerate() {
            if i > 0 {
                parts.push(nl());
            }
            parts.push(line);
        }
        concat_balanced(&mut parts)
    }

    /// The alias of the recursion-group member `member` in this print's alias table.
    fn alias_of(&self, member: Type) -> String {
        String::from(self.aliases.borrow_mut().alias_of(member))
    }

    /// Print a function type (i.e. `(arg0, arg1) -> (result0, result1)`)
    ///
    /// The printed type will elide parens around single-result types
    pub fn print_function_type(&mut self, ty: &FunctionType) {
        self.print_function_type_parts(ty.params().iter().copied(), ty.results().iter().copied());
    }

    /// Print a function type (i.e. `(arg0, arg1) -> (result0, result1)`) given its component parts.
    ///
    /// The printed type will elide parens around single-result types
    pub fn print_function_type_parts(
        &mut self,
        params: impl IntoIterator<Item = Type>,
        results: impl IntoIterator<Item = Type>,
    ) {
        self.print_type_list(params);
        self.print_space();
        self.print_arrow_type_list(/*elide_single_type_parens=*/ true, results);
    }

    /// Print a single region in `{` `}`, as an operation's custom form writes it.
    ///
    /// The label of the entry block is left out when the operation's custom form gives the
    /// block's arguments (a callable's parameters, say) or the block has none; see
    /// [Self::print_region_in_form].
    pub fn print_region(&mut self, region: &Region) {
        self.print_region_in_form(region, /*generic=*/ false);
    }

    /// Print a single region in `{` `}`, for an operation printed in the generic form when
    /// `generic` is set, and in its custom form otherwise.
    ///
    /// In the generic form the entry block is labelled whenever it has arguments, since nothing
    /// else in the text gives them. In either form, a region whose one block has no arguments
    /// and no operations, and needs no label, is printed as `{ }`, as a region parsed from that
    /// text is.
    fn print_region_in_form(&mut self, region: &Region, generic: bool) {
        use crate::formatter::*;

        if region.is_empty() {
            self.document += const_text("{ }");
            return;
        }

        let labelled_target = region
            .entry_block_ref()
            .is_some_and(|entry| Self::is_labelled_entry_target(&entry.borrow()));
        let elide_entry_block_label = if self.flags.print_entry_block_headers || labelled_target {
            false
        } else if generic {
            region.entry_block_ref().is_some_and(|entry| !entry.borrow().has_arguments())
        } else {
            // This is the entry region - if we have the parent operation, and that operation
            // has operands, then we need to print the block labels
            let region_ref = region.as_region_ref();
            region.parent().is_some_and(|op| {
                let op = op.borrow();
                if let Some(branch_interface) = op.as_trait::<dyn crate::RegionBranchOpInterface>()
                {
                    let has_no_region_arguments = branch_interface
                        .get_entry_successor_operands(crate::RegionBranchPoint::Parent)
                        .is_empty();
                    let mut entries =
                        branch_interface.get_successor_regions(crate::RegionBranchPoint::Parent);
                    has_no_region_arguments
                        && entries
                            .any(|r| r.region_graph_successor().is_some_and(|r| r == region_ref))
                } else if let Some(callable) = op.as_trait::<dyn crate::CallableOpInterface>() {
                    callable.get_callable_region().is_some_and(|r| r == region_ref)
                } else if let Some(entry) = region.entry_block_ref() {
                    entry.borrow().arguments().is_empty()
                } else {
                    false
                }
            })
        };

        let mut blocks = region.body().iter();
        if elide_entry_block_label
            && let (Some(only), None) = (blocks.next(), blocks.next())
            && !only.has_arguments()
            && only.body().is_empty()
        {
            self.document += const_text("{ }");
            return;
        }

        self.document += const_text("{");
        let mut printer = self.nested();
        let mut parts = alloc::vec::Vec::new();
        for (i, block) in region.body().iter().enumerate() {
            if i > 0 || !elide_entry_block_label {
                parts.push(nl());
                printer.print_block_label_and_arguments(&block);
            }
            printer.print_block_body(block.body());
            parts.push(printer.render());
        }
        self.document += concat_balanced(&mut parts);
        self.document += nl() + const_text("}");
    }

    /// Returns true if `block` is an entry block that branches target, and so needs its label for
    /// the printed form to parse back.
    ///
    /// The parser only accepts a labelled entry block without arguments, so an entry block with
    /// arguments stays unlabelled even when it is a branch target.
    fn is_labelled_entry_target(block: &Block) -> bool {
        block.is_entry_block() && crate::Usable::is_used(block) && !block.has_arguments()
    }

    /// Print a single block
    ///
    /// An entry block is printed without its label unless the flags request it, or the block has
    /// no arguments and is a branch target.
    pub fn print_block(&mut self, block: &Block) {
        let print_label = !block.is_entry_block()
            || self.flags.print_entry_block_headers
            || Self::is_labelled_entry_target(block);

        if print_label {
            self.print_block_label_and_arguments(block);
        }
        self.print_block_body(block.body());
    }

    /// Print the body of a block, i.e. a sequence of newline-separated operations, indented by 4
    /// spaces.
    ///
    /// NOTE: This method inserts a newline before printing the first operation, in order to trigger
    /// indentation, so it is not required to emit one yourself first.
    pub fn print_block_body(&mut self, ops: &EntityList<Operation>) {
        use crate::formatter::*;

        let body = join_balanced(
            ops.iter().map(|op| {
                let mut printer = self.nested();
                op.print(&mut printer);
                printer.finish()
            }),
            nl,
        );
        self.document += indent(4, nl() + body);
    }

    /// Print the block label and argument list for a block, i.e. `^block0(%0: i32):`
    ///
    /// With [OpPrintingFlags::print_source_locations], each argument is followed by its location
    /// (`^block0(%0: i32 loc("f.runic":1:1)):`), which the parser gives the argument back.
    ///
    /// No leading or trailing newlines are emitted by this method.
    pub fn print_block_label_and_arguments(&mut self, block: &Block) {
        use crate::formatter::*;

        self.document += display(block.id());
        if block.has_arguments() {
            if self.flags.print_source_locations {
                self.document += const_text("(");
                for (index, argument) in block.argument_values().enumerate() {
                    if index > 0 {
                        self.document += const_text(", ");
                    }
                    let argument = argument.borrow();
                    self.document += display(argument.id()) + const_text(": ");
                    self.print_type(argument.ty());
                    self.print_argument_location(&*argument);
                }
                self.document += const_text(")");
            } else {
                self.print_value_id_and_type_list(block.argument_values());
            }
        }
        self.document += const_text(":");
    }

    /// With [OpPrintingFlags::print_source_locations], a space and the location of `argument`, a
    /// block argument, as its definition in a block label or a callable's parameter list writes
    /// it; nothing otherwise.
    pub fn print_argument_location(&mut self, argument: &dyn crate::Value) {
        if self.flags.print_source_locations {
            let loc = Location::from_span(argument.span(), &self.context);
            self.print_space();
            self.print_trailing_location_specifier(&loc);
        }
    }

    /// Print an attribute value: `#dialect.name<body>`, or `#dialect.name` for a marker; a list
    /// is `[a, b, ..]`, each entry printed as an attribute value.
    pub fn print_attribute_value(&mut self, value: &dyn Attribute) {
        use crate::formatter::*;

        let attr = value.as_attr();
        if let Some(list) = attr.downcast_ref::<crate::dialects::builtin::attributes::ListAttr>() {
            list.print(self);
            return;
        }
        self.document += text(format!("#{}", attr.name()));
        if attr.implements::<dyn Marker>() {
            return;
        }
        self.document += const_text("<");
        if let Some(value) = attr.as_trait::<dyn AttrPrinter>() {
            value.print(self);
        } else {
            self.print_string(format!("{value:?}"));
        }
        self.document += const_text(">");
    }

    /// Print an attribute dictionary in `{` `}`
    pub fn print_attribute_dictionary(&mut self, attrs: impl IntoIterator<Item = NamedAttribute>) {
        use crate::formatter::*;

        self.document += const_text("{ ");
        let mut printer = self.nested();
        let doc = join_balanced(
            attrs.into_iter().map(|NamedAttribute { name, value }| {
                printer.print_identifier(name);
                printer.document += const_text(" = ");
                printer.print_attribute_value(&*value.borrow());
                printer.render()
            }),
            || const_text(", "),
        );
        self.document += doc;
        self.document += const_text(" }");
    }

    /// Print the optional trailing location specifier, i.e. `loc("file":1:1)`
    pub fn print_trailing_location_specifier(&mut self, loc: &Location) {
        use crate::formatter::*;

        self.document += text(format!("loc({loc})"));
    }

    /// Print an identifier as either a bare identifier or a quoted, escaped string if it contains
    /// characters which are not valid in bare identifiers.
    ///
    /// This is only valid to call in positions where both bare identifiers and strings are valid.
    pub fn print_identifier(&mut self, ident: interner::Symbol) {
        use crate::formatter::*;

        let id = ident.as_str();
        if is_valid_bare_identifier(id) {
            self.document += const_text(id);
        } else {
            self.print_string(id);
        }
    }

    /// Print an identifier bare.
    ///
    /// This function will panic if `ident` is not a valid bare identifier.
    pub fn print_bare_identifier(&mut self, ident: interner::Symbol) {
        use crate::formatter::*;

        let id = ident.as_str();
        assert!(is_valid_bare_identifier(id));
        self.document += const_text(id);
    }

    /// Print a possibly multi-component symbol path, i.e. `@foo::@bar`
    pub fn print_symbol_path(&mut self, path: &SymbolPath) {
        use crate::{SymbolNameComponent, formatter::*};
        let is_absolute = path.is_absolute();
        for (i, component) in path.components().enumerate() {
            if (is_absolute && i != 1) || (!is_absolute && i > 0) {
                self.document += const_text("::");
            }
            match component {
                SymbolNameComponent::Component(sym) | SymbolNameComponent::Leaf(sym) => {
                    self.print_symbol_name(sym);
                }
                SymbolNameComponent::Root => (),
            }
        }
    }

    /// Print a single-component symbol name, i.e. `@foo`, or `@"foo:bar"` if `name` is not a valid
    /// bare identifier.
    ///
    /// A quoted name escapes `"` and `\` with a `\`, which is exactly what the lexer unescapes.
    pub fn print_symbol_name(&mut self, name: interner::Symbol) {
        use crate::formatter::*;

        let name = name.as_str();
        if is_valid_bare_identifier(name) {
            self.document += text(format!("@{name}"));
        } else {
            let mut quoted = String::from("@");
            crate::ir::symbols::push_quoted(&mut quoted, name);
            self.document += text(quoted);
        }
    }

    /// Print a custom keyword.
    ///
    /// Keywords must be valid bare identifiers. This method will panic if the given keyword is
    /// not valid printed bare.
    pub fn print_keyword(&mut self, keyword: &'static str) {
        use crate::formatter::*;

        assert!(is_valid_bare_identifier(keyword));

        self.document += const_text(keyword);
    }

    /// Print an integer value as a signed decimal at its width (`0` or `1` for an `i1`)
    pub fn print_int(&mut self, value: Int) {
        use crate::formatter::*;

        self.document += display(value);
    }

    /// Print a Rust integer in decimal, as it is (signed if its type is signed)
    pub fn print_decimal_integer<I: PrimitiveInteger>(&mut self, value: I) {
        use crate::formatter::*;

        self.document += display(value);
    }

    /// Print an integer value as `0x` followed by its unsigned value in lowercase hexadecimal
    pub fn print_hex_integer(&mut self, value: Int) {
        use crate::formatter::*;

        self.document += text(format!("{value:#x}"));
    }

    /// Print a boolean value
    pub fn print_bool(&mut self, value: bool) {
        use crate::formatter::*;

        if value {
            self.document += const_text("true");
        } else {
            self.document += const_text("false");
        }
    }

    /// Print a literal string value
    pub fn print_string(&mut self, string: impl AsRef<str>) {
        use crate::formatter::*;

        self.document += text(format!("\"{}\"", string.as_ref().escape_default()));
    }

    /// Print a single '('
    pub fn print_lparen(&mut self) {
        use crate::formatter::*;
        self.document += const_text("(");
    }

    /// Print a single ')'
    pub fn print_rparen(&mut self) {
        use crate::formatter::*;
        self.document += const_text(")");
    }

    /// Print a `->`
    pub fn print_arrow(&mut self) {
        use crate::formatter::*;
        self.document += const_text("->");
    }

    /// Print a single space
    pub fn print_space(&mut self) {
        use crate::formatter::*;
        self.document += const_text(" ");
    }

    /// Print a single newline
    pub fn print_newline(&mut self) {
        use crate::formatter::*;
        self.document += nl();
    }
}

/// The pieces a printer has been given, in order. They are joined as a balanced tree when the
/// document is taken ([AsmPrinter::render], [AsmPrinter::finish]), so that a printer which appends
/// one piece per item of a long list does not build a document as deep as the list is long.
#[derive(Default)]
struct Pieces(alloc::vec::Vec<Document>);

impl Pieces {
    fn take(&mut self) -> Document {
        let mut pieces = core::mem::take(&mut self.0);
        concat_balanced(&mut pieces)
    }
}

impl AddAssign<Document> for Pieces {
    fn add_assign(&mut self, rhs: Document) {
        if rhs.is_empty() {
            return;
        }
        self.0.push(rhs);
    }
}

impl AddAssign<Document> for AsmPrinter<'_> {
    fn add_assign(&mut self, rhs: Document) {
        self.document += rhs;
    }
}

/// Returns true if `id` reads back as one bare identifier, mirroring the lexer's
/// `lex_identifier`: an ASCII letter or `_`, followed by ASCII letters, digits, `_`, `$` and `.`.
fn is_valid_bare_identifier(id: &str) -> bool {
    let mut chars = id.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '$'))
}

#[cfg(test)]
mod int_tests {
    use alloc::{format, rc::Rc};

    use super::*;
    use crate::print::OpPrintingFlags;

    #[test]
    fn print_int_writes_the_display_form() {
        let flags = OpPrintingFlags::default();
        for (value, expected) in [(Int::from(-1i8), "-1"), (Int::from(true), "1")] {
            let mut printer = AsmPrinter::fragment(Rc::new(Context::default()), &flags);
            printer.print_int(value);
            assert_eq!(format!("{}", printer.finish()), expected);
        }
    }

    #[test]
    fn print_hex_integer_writes_the_prefix() {
        let flags = OpPrintingFlags::default();
        let mut printer = AsmPrinter::fragment(Rc::new(Context::default()), &flags);
        printer.print_hex_integer(Int::from(255u8));
        assert_eq!(format!("{}", printer.finish()), "0xff");
    }

    #[test]
    fn print_decimal_integer_writes_rust_integers_as_they_are() {
        let flags = OpPrintingFlags::default();
        let mut printer = AsmPrinter::fragment(Rc::new(Context::default()), &flags);
        printer.print_decimal_integer(-1i8);
        printer.print_decimal_integer(u32::MAX);
        assert_eq!(format!("{}", printer.finish()), "-14294967295");
    }
}

/// The Rust integer types, which [AsmPrinter::print_decimal_integer] writes as they are.
pub trait PrimitiveInteger: sealed::Sealed + Copy + core::fmt::Display {}

mod sealed {
    pub trait Sealed {}
}

macro_rules! primitive_integer {
    ($($t:ty),*) => {
        $(
            impl sealed::Sealed for $t {}
            impl PrimitiveInteger for $t {}
        )*
    };
}

primitive_integer!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize);
