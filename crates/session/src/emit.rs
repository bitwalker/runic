#[cfg(not(feature = "std"))]
use alloc::string::ToString;
use alloc::{boxed::Box, sync::Arc};

use runic_core::{
    Op, OpPrinter, Symbol as RSymbol, SymbolPath,
    diagnostics::{IntoDiagnostic, Report},
    dialects::builtin::{Component, Module},
    print::{AsmPrinter, OpPrintingFlags},
};
use runic_symbol::Symbol;

use crate::{OutputMode, OutputType, Session};

pub trait Emit {
    /// The name of this item, if applicable
    fn name(&self) -> Option<Symbol>;
    /// The output type associated with this item and the given `mode`
    fn output_type(&self, mode: OutputMode) -> OutputType;
    /// Write this item to the given [std::io::Write] handle, using `mode` to determine the output
    /// type
    fn write_to<W: Writer>(
        &self,
        writer: W,
        mode: OutputMode,
        _session: &Session,
    ) -> Result<(), Report>;
}

#[cfg(feature = "std")]
pub trait EmitExt: Emit {
    /// Write this item to standard output, inferring the best [OutputMode] based on whether or not
    /// stdout is a tty or not
    fn write_to_stdout(&self, session: &Session) -> Result<(), Report>;
    /// Write this item to the given file path, using `mode` to determine the output type
    fn write_to_file(
        &self,
        path: &std::path::Path,
        mode: OutputMode,
        session: &Session,
    ) -> Result<(), Report>;
}

#[cfg(feature = "std")]
impl<T: ?Sized + Emit> EmitExt for T {
    default fn write_to_stdout(&self, session: &Session) -> Result<(), Report> {
        use std::io::IsTerminal;
        let stdout = std::io::stdout().lock();
        let mode = if stdout.is_terminal() {
            OutputMode::Text
        } else {
            OutputMode::Binary
        };
        self.write_to(stdout, mode, session)
    }

    default fn write_to_file(
        &self,
        path: &std::path::Path,
        mode: OutputMode,
        session: &Session,
    ) -> Result<(), Report> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).into_diagnostic()?;
        }
        let file = std::fs::File::create(path).into_diagnostic()?;
        self.write_to(file, mode, session)
    }
}

/// A trait that provides a subset of the [std::io::Write] functionality that is usable in no-std
/// contexts.
pub trait Writer {
    fn write_fmt(&mut self, fmt: core::fmt::Arguments<'_>) -> Result<(), Report>;
    fn write_all(&mut self, buf: &[u8]) -> Result<(), Report>;
}

#[cfg(feature = "std")]
impl<W: ?Sized + std::io::Write> Writer for W {
    fn write_fmt(&mut self, fmt: core::fmt::Arguments<'_>) -> Result<(), Report> {
        <W as std::io::Write>::write_fmt(self, fmt).into_diagnostic()
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Report> {
        <W as std::io::Write>::write_all(self, buf).into_diagnostic()
    }
}

#[cfg(not(feature = "std"))]
impl Writer for alloc::vec::Vec<u8> {
    fn write_fmt(&mut self, fmt: core::fmt::Arguments<'_>) -> Result<(), Report> {
        if let Some(s) = fmt.as_str() {
            self.extend(s.as_bytes());
        } else {
            let formatted = fmt.to_string();
            self.extend(formatted.as_bytes());
        }
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Report> {
        self.extend(buf);
        Ok(())
    }
}

#[cfg(not(feature = "std"))]
impl Writer for alloc::string::String {
    fn write_fmt(&mut self, fmt: core::fmt::Arguments<'_>) -> Result<(), Report> {
        if let Some(s) = fmt.as_str() {
            self.push_str(s);
        } else {
            let formatted = fmt.to_string();
            self.push_str(&formatted);
        }
        Ok(())
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<(), Report> {
        let s = core::str::from_utf8(buf).into_diagnostic()?;
        self.push_str(s);
        Ok(())
    }
}

impl<T: Emit> Emit for &T {
    #[inline]
    fn name(&self) -> Option<Symbol> {
        (**self).name()
    }

    #[inline]
    fn output_type(&self, mode: OutputMode) -> OutputType {
        (**self).output_type(mode)
    }

    #[inline]
    fn write_to<W: Writer>(
        &self,
        writer: W,
        mode: OutputMode,
        session: &Session,
    ) -> Result<(), Report> {
        (**self).write_to(writer, mode, session)
    }
}

impl<T: Emit> Emit for &mut T {
    #[inline]
    fn name(&self) -> Option<Symbol> {
        (**self).name()
    }

    #[inline]
    fn output_type(&self, mode: OutputMode) -> OutputType {
        (**self).output_type(mode)
    }

    #[inline]
    fn write_to<W: Writer>(
        &self,
        writer: W,
        mode: OutputMode,
        session: &Session,
    ) -> Result<(), Report> {
        (**self).write_to(writer, mode, session)
    }
}

impl<T: Emit> Emit for Box<T> {
    #[inline]
    fn name(&self) -> Option<Symbol> {
        (**self).name()
    }

    #[inline]
    fn output_type(&self, mode: OutputMode) -> OutputType {
        (**self).output_type(mode)
    }

    #[inline]
    fn write_to<W: Writer>(
        &self,
        writer: W,
        mode: OutputMode,
        session: &Session,
    ) -> Result<(), Report> {
        (**self).write_to(writer, mode, session)
    }
}

impl<T: Emit> Emit for Arc<T> {
    #[inline]
    fn name(&self) -> Option<Symbol> {
        (**self).name()
    }

    #[inline]
    fn output_type(&self, mode: OutputMode) -> OutputType {
        (**self).output_type(mode)
    }

    #[inline]
    fn write_to<W: Writer>(
        &self,
        writer: W,
        mode: OutputMode,
        session: &Session,
    ) -> Result<(), Report> {
        (**self).write_to(writer, mode, session)
    }
}

impl Emit for alloc::string::String {
    fn name(&self) -> Option<Symbol> {
        None
    }

    fn output_type(&self, _mode: OutputMode) -> OutputType {
        OutputType::Ir
    }

    fn write_to<W: Writer>(
        &self,
        mut writer: W,
        _mode: OutputMode,
        _session: &Session,
    ) -> Result<(), Report> {
        writer.write_fmt(format_args!("{self}\n"))
    }
}

impl Emit for Module {
    /// The raw segment of the module name, without the quotes of its path spelling
    /// (`masm-dep` for `"masm-dep"`). It may hold any spellable character, `/` and `:` included.
    fn name(&self) -> Option<Symbol> {
        Some(SymbolPath::raw_segment(RSymbol::name(self)))
    }

    fn output_type(&self, _mode: OutputMode) -> OutputType {
        OutputType::Ir
    }

    fn write_to<W: Writer>(
        &self,
        mut writer: W,
        _mode: OutputMode,
        _session: &Session,
    ) -> Result<(), Report> {
        let flags = OpPrintingFlags::default();
        let mut printer = AsmPrinter::root(self.as_operation().context_rc(), &flags);
        <Self as OpPrinter>::print(self, &mut printer);
        let document = printer.finish();
        writer.write_fmt(format_args!("{document}"))
    }
}

impl Emit for Component {
    /// The raw last segment of the component name, without the quotes of its path spelling.
    /// It may hold any spellable character, `/` and `:` included.
    fn name(&self) -> Option<Symbol> {
        SymbolPath::segments_of(RSymbol::name(self)).last().copied()
    }

    fn output_type(&self, _mode: OutputMode) -> OutputType {
        OutputType::Ir
    }

    fn write_to<W: Writer>(
        &self,
        mut writer: W,
        _mode: OutputMode,
        _session: &Session,
    ) -> Result<(), Report> {
        let flags = OpPrintingFlags::default();
        let mut printer = AsmPrinter::root(self.as_operation().context_rc(), &flags);
        <Self as OpPrinter>::print(self, &mut printer);
        writer.write_fmt(format_args!("{}", printer.finish()))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::{rc::Rc, string::String, vec::Vec};

    use runic_core::{
        BuilderExt, Context, Ident, SourceSpan, StructType, Visibility,
        dialects::builtin::{
            ComponentBuilder, ModuleBuilder, World, WorldBuilder, attributes::Signature,
        },
        interner,
    };
    use tempfile::TempDir;

    use super::*;
    use crate::{InputFile, InputType, Options};

    #[test]
    fn an_emitted_module_declares_the_recursion_groups_it_refers_to() {
        let context = Rc::new(Context::default());
        let body = context.struct_type(StructType::named(
            interner::Symbol::intern("list"),
            [context.i32().into(), context.ptr_type(context.rec_var(0)).into()],
        ));
        let list = context.rec_group(&[body]).unwrap()[0];

        let world = context.clone().builder().create::<World, ()>(SourceSpan::UNKNOWN)().unwrap();
        let component =
            WorldBuilder::new(world).define_component(Ident::from("acme::app")).unwrap();
        let module = ComponentBuilder::new(component).define_module(Ident::from("m")).unwrap();
        let signature = Signature::new(&context, [context.ptr_type(list)], []);
        ModuleBuilder::new(module)
            .define_function(Ident::from("f"), Visibility::Public, signature)
            .unwrap();

        let temp = TempDir::new().unwrap();
        let options = Options {
            current_dir: temp.path().to_path_buf(),
            target_dir: temp.path().join("target"),
            ..Options::default()
        };
        let input = InputFile::new(InputType::Real("m.runic".into()));
        let session = Session::new(input, Box::new(options)).unwrap();

        let mut out = Vec::new();
        module.borrow().write_to(&mut out, OutputMode::Text, &session).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.starts_with("rec {\n  type !list = struct list<i32, ptr<!list>>;\n}\n\n"),
            "unexpected emitted text:\n{text}"
        );
        assert_eq!(text.matches("rec {").count(), 1, "unexpected emitted text:\n{text}");
    }
}
