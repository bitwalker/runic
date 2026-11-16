//! The `calc` language and its frontend.
//!
//! A file holds definitions of functions over 32-bit integers:
//!
//! ```text
//! # a comment runs to the end of the line
//! fn add(a, b) = a + b;
//! fn twice(x) = add(x, x) * 2;
//! ```
//!
//! An expression is made of integer literals, the parameters of the function, calls of functions
//! defined earlier in the same file, parentheses and `+`, `-` and `*`, which associate to the left
//! and bind as usual.

use core::{fmt, ops::Range};
use std::collections::HashMap;

use runic_dialect_arith::ArithOpBuilder;
use runic_ir::{
    Builder, Ident, Op, Report, Signedness, SourceSpan, Type, ValueRef, Visibility,
    diagnostics::{SourceId, SourceKey, SourceProvider, report},
    dialects::builtin::{
        BuiltinOpBuilder, FunctionBuilder, FunctionRef, ModuleBuilder, WorldBuilder, WorldRef,
        attributes::Signature,
    },
    interner::Symbol,
};

use crate::{
    Artifact, CheckpointDecl, CheckpointId, Checkpoints, Flow, Frontend, Input, OptionsDecl,
    Render, Session,
};

/// The checkpoint at which the parsed files are held.
const AST: CheckpointId = CheckpointId::new("calc.ast");

/// What the options of the `calc` frontend say; a request carries them with
/// [CompilationRequest::set_options](crate::CompilationRequest::set_options).
#[derive(clap::Args, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CalcOptions {
    /// Wrap `+`, `-` and `*` around when the result does not fit in 32 bits, instead of trapping
    /// on overflow of signed numbers.
    #[arg(long)]
    pub wrap_on_overflow: bool,
}

/// The frontend for files of the `calc` language (`.calc`).
///
/// Each input becomes a public `builtin.module` named after the stem of the input, and each
/// definition in it a public `builtin.function` over `i32`, built from `arith` operations,
/// `builtin.call` and `builtin.ret`. The first error in a file fails the compilation with a report
/// that labels the place in the file. [CalcOptions] choose the overflow mode of the arithmetic.
///
/// The frontend reaches `calc.ast` (also `ast`, in files with the extension `ast`) once every
/// input is parsed and before any IR is built.
#[derive(Debug, Default, Clone, Copy)]
pub struct CalcFrontend;

const CHECKPOINTS: &[CheckpointDecl] = &[CheckpointDecl {
    id: AST,
    aliases: &["ast"],
    extension: "ast",
    render: render_ast as Render,
}];

impl Frontend for CalcFrontend {
    fn name(&self) -> &'static str {
        "calc"
    }

    fn extensions(&self) -> &'static [&'static str] {
        &["calc"]
    }

    fn checkpoints(&self) -> &'static [CheckpointDecl] {
        CHECKPOINTS
    }

    fn translate(
        &self,
        session: &Session<'_>,
        inputs: &[Input],
        world: WorldRef,
        checkpoints: &mut Checkpoints<'_>,
    ) -> Result<Flow<()>, Report> {
        let mut files = Vec::with_capacity(inputs.len());
        for input in inputs {
            files.push(parse_input(session, input)?);
        }
        let ast = match checkpoints.reach(AST, CalcAst { files })? {
            Flow::Continue(ast) => ast,
            Flow::Break(stopped) => return Ok(Flow::Break(stopped)),
        };
        let options = session.options::<CalcOptions>().copied().ok_or_else(|| {
            Report::msg("the options of the calc frontend were not stored in the session")
        })?;
        for file in &ast.files {
            build(session, world, file, options)?;
        }
        Ok(Flow::Continue(()))
    }

    fn options(&self) -> Option<OptionsDecl> {
        Some(OptionsDecl::of::<CalcOptions>())
    }
}

/// The parsed files of a compilation: what `calc.ast` holds.
///
/// Displayed, it is an indented tree, one file after another:
///
/// ```text
/// file demo
///   fn add(a, b)
///     add
///       param a
///       param b
/// ```
#[derive(Debug)]
pub struct CalcAst {
    files: Vec<File>,
}

impl fmt::Display for CalcAst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for file in &self.files {
            writeln!(f, "file {}", file.stem)?;
            for function in &file.functions {
                writeln!(f, "  fn {}({})", function.name, function.params.join(", "))?;
                function.body.write(f, 1)?;
            }
        }
        Ok(())
    }
}

fn render_ast(
    artifact: &Artifact,
    _session: &Session<'_>,
    out: &mut dyn std::io::Write,
) -> Result<(), Report> {
    let checkpoint = artifact.checkpoint();
    let Some(ast) = artifact.downcast_ref::<CalcAst>() else {
        return Err(Report::msg(format!(
            "the artifact at checkpoint '{checkpoint}' is not a parsed file of the calc language"
        )));
    };
    write!(out, "{ast}").map_err(|error| {
        Report::msg(format!("could not render checkpoint '{checkpoint}': {error}"))
    })
}

/// One parsed input.
#[derive(Debug)]
struct File {
    /// The stem of the input: the name of the module.
    stem: String,
    functions: Vec<Function>,
}

#[derive(Debug)]
struct Function {
    name: String,
    span: SourceSpan,
    params: Vec<String>,
    body: Expr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
}

impl BinOp {
    fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
        }
    }
}

/// An expression whose names are resolved: a parameter is its index, a callee a function of the
/// file that was defined before the expression.
#[derive(Debug)]
enum Expr {
    Int {
        value: i32,
        span: SourceSpan,
    },
    Param {
        index: usize,
        name: String,
        span: SourceSpan,
    },
    Call {
        callee: String,
        args: Vec<Expr>,
        span: SourceSpan,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: SourceSpan,
    },
}

impl Expr {
    fn write(&self, f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
        let indent = "  ".repeat(depth + 1);
        match self {
            Self::Int { value, .. } => writeln!(f, "{indent}int {value}"),
            Self::Param { name, .. } => writeln!(f, "{indent}param {name}"),
            Self::Call { callee, args, .. } => {
                writeln!(f, "{indent}call {callee}")?;
                args.iter().try_for_each(|arg| arg.write(f, depth + 1))
            }
            Self::Binary { op, lhs, rhs, .. } => {
                writeln!(f, "{indent}{}", op.name())?;
                lhs.write(f, depth + 1)?;
                rhs.write(f, depth + 1)
            }
        }
    }

    fn span(&self) -> SourceSpan {
        match self {
            Self::Int { span, .. }
            | Self::Param { span, .. }
            | Self::Call { span, .. }
            | Self::Binary { span, .. } => *span,
        }
    }
}

// Lexing and parsing

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Ident(String),
    Int(String),
    Punct(char),
}

impl Token {
    fn describe(&self) -> String {
        match self {
            Self::Ident(name) => format!("'{name}'"),
            Self::Int(digits) => format!("'{digits}'"),
            Self::Punct(c) => format!("'{c}'"),
        }
    }
}

/// A token and where it is in the text.
type Spanned = (Token, Range<usize>);

struct Source {
    key: SourceKey,
    len: usize,
}

impl Source {
    fn span(&self, range: Range<usize>) -> SourceSpan {
        SourceSpan::try_from_range(self.key, None, range).unwrap_or_default()
    }

    fn error(&self, range: Range<usize>, message: String, label: String) -> Report {
        report!(
            message: (message),
            labels: [primary(self.span(range), label)],
        )
    }
}

fn lex(source: &Source, text: &str) -> Result<Vec<Spanned>, Report> {
    let mut tokens = Vec::new();
    let mut chars = text.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        if c.is_whitespace() {
            continue;
        }
        if c == '#' {
            while chars.next_if(|&(_, c)| c != '\n').is_some() {}
            continue;
        }
        let mut end = start + c.len_utf8();
        if c.is_ascii_digit() {
            while let Some((at, c)) = chars.next_if(|&(_, c)| c.is_ascii_alphanumeric() || c == '_')
            {
                end = at + c.len_utf8();
            }
            tokens.push((Token::Int(text[start..end].to_owned()), start..end));
        } else if c.is_ascii_alphabetic() || c == '_' {
            while let Some((at, c)) = chars.next_if(|&(_, c)| c.is_ascii_alphanumeric() || c == '_')
            {
                end = at + c.len_utf8();
            }
            tokens.push((Token::Ident(text[start..end].to_owned()), start..end));
        } else if "()+-*,=;".contains(c) {
            tokens.push((Token::Punct(c), start..end));
        } else {
            return Err(source.error(
                start..end,
                format!("unexpected character '{c}'"),
                "the calc language has no such character".to_owned(),
            ));
        }
    }
    Ok(tokens)
}

struct Parser<'a> {
    source: &'a Source,
    tokens: Vec<Spanned>,
    at: usize,
    /// The functions defined so far in this file and their arities.
    defined: HashMap<String, usize>,
    params: Vec<String>,
    /// Where the body of the function being read starts, as an index into `tokens`.
    body_start: usize,
}

/// The most tokens the body of one function may have. Every level of nesting takes a token, so
/// this also bounds how deep the parser and everything that walks an expression recurse.
const MAX_BODY_TOKENS: usize = 128;

impl Parser<'_> {
    fn peek(&self) -> Option<&Spanned> {
        self.tokens.get(self.at)
    }

    fn peek_punct(&self, c: char) -> bool {
        matches!(self.peek(), Some((Token::Punct(p), _)) if *p == c)
    }

    /// Where the end of the text is, as a place a label can point to.
    fn end_range(&self) -> Range<usize> {
        self.source.len.saturating_sub(1)..self.source.len
    }

    fn unexpected(&self, wanted: &str) -> Report {
        match self.peek() {
            Some((token, range)) => self.source.error(
                range.clone(),
                format!("expected {wanted}, found {}", token.describe()),
                format!("expected {wanted} here"),
            ),
            None => self.source.error(
                self.end_range(),
                format!("expected {wanted}, found the end of the file"),
                format!("expected {wanted} after this"),
            ),
        }
    }

    fn punct(&mut self, c: char) -> Result<Range<usize>, Report> {
        if self.peek_punct(c) {
            self.at += 1;
            Ok(self.tokens[self.at - 1].1.clone())
        } else {
            Err(self.unexpected(&format!("'{c}'")))
        }
    }

    fn ident(&mut self, wanted: &str) -> Result<(String, Range<usize>), Report> {
        match self.peek() {
            Some((Token::Ident(name), range)) => {
                let found = (name.clone(), range.clone());
                self.at += 1;
                Ok(found)
            }
            _ => Err(self.unexpected(wanted)),
        }
    }

    fn file(&mut self) -> Result<Vec<Function>, Report> {
        let mut functions = Vec::new();
        while self.peek().is_some() {
            let function = self.function()?;
            self.defined.insert(function.name.clone(), function.params.len());
            functions.push(function);
        }
        Ok(functions)
    }

    fn function(&mut self) -> Result<Function, Report> {
        let (keyword, range) = self.ident("'fn' to start a definition")?;
        if keyword != "fn" {
            return Err(self.source.error(
                range,
                format!("expected 'fn' to start a definition, found '{keyword}'"),
                "a definition starts with 'fn'".to_owned(),
            ));
        }
        let (name, name_range) = self.ident("the name of the function")?;
        if self.defined.contains_key(&name) {
            return Err(self.source.error(
                name_range,
                format!("the function '{name}' is defined twice"),
                format!("'{name}' is already defined in this file"),
            ));
        }
        self.punct('(')?;
        self.params.clear();
        if !self.peek_punct(')') {
            loop {
                let (param, range) = self.ident("the name of a parameter")?;
                if self.params.contains(&param) {
                    return Err(self.source.error(
                        range,
                        format!("the parameter '{param}' of '{name}' is named twice"),
                        format!("'{param}' is already a parameter of '{name}'"),
                    ));
                }
                self.params.push(param);
                if self.peek_punct(',') {
                    self.at += 1;
                } else {
                    break;
                }
            }
        }
        self.punct(')')?;
        self.punct('=')?;
        self.body_start = self.at;
        let body = self.expr()?;
        self.punct(';')?;
        Ok(Function {
            span: self.source.span(name_range),
            name,
            params: core::mem::take(&mut self.params),
            body,
        })
    }

    fn expr(&mut self) -> Result<Expr, Report> {
        let mut lhs = self.term()?;
        loop {
            let op = match self.peek() {
                Some((Token::Punct('+'), _)) => BinOp::Add,
                Some((Token::Punct('-'), _)) => BinOp::Sub,
                _ => return Ok(lhs),
            };
            self.at += 1;
            let rhs = self.term()?;
            lhs = self.binary(op, lhs, rhs);
        }
    }

    fn term(&mut self) -> Result<Expr, Report> {
        let mut lhs = self.factor()?;
        while self.peek_punct('*') {
            self.at += 1;
            let rhs = self.factor()?;
            lhs = self.binary(BinOp::Mul, lhs, rhs);
        }
        Ok(lhs)
    }

    fn binary(&self, op: BinOp, lhs: Expr, rhs: Expr) -> Expr {
        let span = self.join(lhs.span(), rhs.span());
        Expr::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            span,
        }
    }

    /// The span from the start of `first` to the end of `last`.
    fn join(&self, first: SourceSpan, last: SourceSpan) -> SourceSpan {
        let (first, last) = (first.range(), last.range());
        self.source.span(first.start() as usize..last.end() as usize)
    }

    fn factor(&mut self) -> Result<Expr, Report> {
        if self.at - self.body_start >= MAX_BODY_TOKENS
            && let Some((_, range)) = self.peek()
        {
            return Err(self.source.error(
                range.clone(),
                format!(
                    "the body of this function is too long: a body has at most {MAX_BODY_TOKENS} \
                     tokens"
                ),
                String::from("the body is too long from here"),
            ));
        }
        match self.peek().cloned() {
            Some((Token::Int(digits), range)) => {
                self.at += 1;
                match digits.parse::<i32>() {
                    Ok(value) => Ok(Expr::Int {
                        value,
                        span: self.source.span(range),
                    }),
                    Err(_) => Err(self.source.error(
                        range,
                        format!("the literal '{digits}' is not a 32-bit integer"),
                        format!("a literal is a number from 0 to {}", i32::MAX),
                    )),
                }
            }
            Some((Token::Ident(name), range)) => {
                self.at += 1;
                if self.peek_punct('(') {
                    self.call(name, range)
                } else if let Some(index) = self.params.iter().position(|param| *param == name) {
                    Ok(Expr::Param {
                        index,
                        name,
                        span: self.source.span(range),
                    })
                } else {
                    Err(self.source.error(
                        range,
                        format!("the name '{name}' is not a parameter of this function"),
                        format!("'{name}' is not a parameter"),
                    ))
                }
            }
            Some((Token::Punct('('), _)) => {
                self.at += 1;
                let inner = self.expr()?;
                self.punct(')')?;
                Ok(inner)
            }
            _ => Err(self.unexpected("an expression")),
        }
    }

    fn call(&mut self, callee: String, name_range: Range<usize>) -> Result<Expr, Report> {
        let Some(&arity) = self.defined.get(&callee) else {
            return Err(self.source.error(
                name_range,
                format!("the function '{callee}' is called but not defined before this call"),
                format!("no function '{callee}' is defined above"),
            ));
        };
        self.punct('(')?;
        let mut args = Vec::new();
        if !self.peek_punct(')') {
            loop {
                args.push(self.expr()?);
                if self.peek_punct(',') {
                    self.at += 1;
                } else {
                    break;
                }
            }
        }
        let close = self.punct(')')?;
        if args.len() != arity {
            return Err(self.source.error(
                name_range.start..close.end,
                format!(
                    "the function '{callee}' takes {arity} argument{}, but the call gives {}",
                    if arity == 1 { "" } else { "s" },
                    args.len()
                ),
                format!("'{callee}' takes {arity}"),
            ));
        }
        Ok(Expr::Call {
            span: self.source.span(name_range.start..close.end),
            callee,
            args,
        })
    }
}

fn parse_input(session: &Session<'_>, input: &Input) -> Result<File, Report> {
    let stem = input.stem().map(str::to_owned).ok_or_else(|| {
        Report::msg(format!(
            "the input '{}' has no file stem to name its module after",
            input.name()
        ))
    })?;
    let id: SourceId = session.load_source(input)?;
    let sources = session.context().sources();
    let text = sources.get(id).and_then(|source| source.text).ok_or_else(|| {
        Report::msg(format!("the text of input '{}' is not available", input.name()))
    })?;
    let source = Source {
        key: SourceKey::Session(id),
        len: text.len(),
    };
    let mut parser = Parser {
        tokens: lex(&source, text)?,
        source: &source,
        at: 0,
        defined: HashMap::new(),
        params: Vec::new(),
        body_start: 0,
    };
    let functions = parser.file()?;
    Ok(File { stem, functions })
}

// Building IR

/// Adds a module for `file` to `world`, with a function for each definition.
fn build(
    session: &Session<'_>,
    world: WorldRef,
    file: &File,
    options: CalcOptions,
) -> Result<(), Report> {
    let context = session.context();
    let mut module = WorldBuilder::new(world)
        .declare_module(Ident::with_empty_span(Symbol::intern(file.stem.as_str())))
        .map_err(|error| {
            Report::msg(format!("cannot define the module '{}' for its file: {error}", file.stem))
        })?;
    module.borrow_mut().set_visibility(Visibility::Public);
    let i32_type = context.i32();
    let mut defined: HashMap<&str, FunctionRef> = HashMap::new();
    for function in &file.functions {
        let signature = Signature::new(
            context,
            core::iter::repeat_n(i32_type, function.params.len()),
            [i32_type],
        );
        // A builder of the module for each function: the one of a function body points into it.
        let mut module_builder = ModuleBuilder::new(module);
        let func = module_builder.define_function(
            Ident::new(Symbol::intern(function.name.as_str()), function.span),
            Visibility::Public,
            signature,
        )?;
        let mut builder = FunctionBuilder::new(func, module_builder.builder());
        let arguments: Vec<ValueRef> = builder
            .current_block()
            .borrow()
            .arguments()
            .iter()
            .map(|argument| *argument as ValueRef)
            .collect();
        let result = emit(&mut builder, &function.body, &arguments, &defined, i32_type, options)?;
        builder.ret([result], function.span)?;
        defined.insert(&function.name, func);
    }
    Ok(())
}

fn emit(
    builder: &mut FunctionBuilder<'_, runic_ir::OpBuilder>,
    expr: &Expr,
    arguments: &[ValueRef],
    defined: &HashMap<&str, FunctionRef>,
    ty: Type,
    options: CalcOptions,
) -> Result<ValueRef, Report> {
    match expr {
        Expr::Int { value, span } => Ok(ArithOpBuilder::i32(builder, *value, *span)),
        Expr::Param { index, .. } => Ok(arguments[*index]),
        Expr::Call { callee, args, span } => {
            let args = args
                .iter()
                .map(|arg| emit(builder, arg, arguments, defined, ty, options))
                .collect::<Result<Vec<_>, _>>()?;
            let target = defined[callee.as_str()];
            let context = builder.builder().context_rc();
            let signature = Signature::new(&context, vec![ty; args.len()], [ty]);
            let call = BuiltinOpBuilder::call(builder, target, signature, args, *span)?;
            let result = call.borrow().as_operation().results()[0].borrow().as_value_ref();
            Ok(result)
        }
        Expr::Binary { op, lhs, rhs, span } => {
            let lhs = emit(builder, lhs, arguments, defined, ty, options)?;
            let rhs = emit(builder, rhs, arguments, defined, ty, options)?;
            let signed = Signedness::Signed;
            match (op, options.wrap_on_overflow) {
                (BinOp::Add, true) => ArithOpBuilder::add(builder, lhs, rhs, *span),
                (BinOp::Add, false) => {
                    ArithOpBuilder::add_checked(builder, lhs, rhs, signed, *span)
                }
                (BinOp::Sub, true) => ArithOpBuilder::sub(builder, lhs, rhs, *span),
                (BinOp::Sub, false) => {
                    ArithOpBuilder::sub_checked(builder, lhs, rhs, signed, *span)
                }
                (BinOp::Mul, true) => ArithOpBuilder::mul(builder, lhs, rhs, *span),
                (BinOp::Mul, false) => {
                    ArithOpBuilder::mul_checked(builder, lhs, rhs, signed, *span)
                }
            }
        }
    }
}
