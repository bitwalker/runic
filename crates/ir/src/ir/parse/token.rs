use alloc::string::String;
use core::fmt;

use crate::CompactString;

/// The token type produced by [super::Lexer], and consumed by the parser.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Token<'input> {
    // Markers
    Comment,
    Eof,

    // Identifiers
    /// foo
    BareIdent(&'input str),
    /// @foo
    AtIdent(&'input str),
    /// #foo
    HashIdent(&'input str),
    /// %foo
    PercentIdent(&'input str),
    /// ^foo
    CaretIdent(&'input str),
    /// !foo
    BangIdent(&'input str),

    // Primitives
    /// "foo"
    String(&'input str),
    /// 128
    Int(&'input str),
    /// 0x0F
    Hex(&'input str),
    /// 0b00001111
    Binary(&'input str),

    // Value Keywords
    True,
    False,

    // Attribute Keywords
    Loc,

    // Type Keywords
    /// `iN`: an integer type of `N` bits, written as `i` followed by a decimal width without a
    /// leading zero. A width too large for a `u32` is not lexed as a type, but as an identifier.
    IntType(u32),
    Isize,
    F32,
    F64,
    Never,
    Ptr,
    Struct,
    Enum,
    Array,

    // Delimiters
    /// <
    Langle,
    /// >
    Rangle,
    /// {
    Lbrace,
    /// }
    Rbrace,
    /// [
    Lbracket,
    /// ]
    Rbracket,
    /// (
    Lparen,
    /// )
    Rparen,

    // Punctuation
    /// @
    At,
    /// !
    Bang,
    /// ,
    Comma,
    /// ::
    ColonColon,
    /// :
    Colon,
    /// .
    Dot,
    /// $
    Dollar,
    /// ...
    Ellipsis,
    /// =
    Equal,
    /// #
    Hash,
    /// %
    Percent,
    /// |
    Pipe,
    /// +
    Plus,
    /// ;
    Semicolon,
    /// /
    Slash,
    /// *
    Star,
    /// -
    Minus,
    /// ->
    Rstab,
    /// ?
    Question,
}

impl Token<'_> {
    pub fn into_compact_string(&self) -> CompactString {
        match self {
            Self::BareIdent(s)
            | Self::AtIdent(s)
            | Self::HashIdent(s)
            | Self::PercentIdent(s)
            | Self::CaretIdent(s)
            | Self::BangIdent(s)
            | Self::String(s)
            | Self::Int(s)
            | Self::Hex(s)
            | Self::Binary(s) => CompactString::from(*s),
            Self::Comment => CompactString::const_new("comment"),
            Self::Eof => CompactString::const_new("end of input"),
            Self::True => CompactString::const_new("true"),
            Self::False => CompactString::const_new("false"),
            Self::Loc => CompactString::const_new("loc"),
            Self::IntType(width) => CompactString::from(alloc::format!("i{width}")),
            Self::Isize => CompactString::const_new("isize"),
            Self::F32 => CompactString::const_new("f32"),
            Self::F64 => CompactString::const_new("f64"),
            Self::Never => CompactString::const_new("never"),
            Self::Ptr => CompactString::const_new("ptr"),
            Self::Struct => CompactString::const_new("struct"),
            Self::Enum => CompactString::const_new("enum"),
            Self::Array => CompactString::const_new("array"),
            Self::Langle => CompactString::const_new("<"),
            Self::Rangle => CompactString::const_new(">"),
            Self::Lbrace => CompactString::const_new("{"),
            Self::Rbrace => CompactString::const_new("}"),
            Self::Lbracket => CompactString::const_new("["),
            Self::Rbracket => CompactString::const_new("]"),
            Self::Lparen => CompactString::const_new("("),
            Self::Rparen => CompactString::const_new(")"),
            Self::At => CompactString::const_new("@"),
            Self::Bang => CompactString::const_new("!"),
            Self::Comma => CompactString::const_new(","),
            Self::ColonColon => CompactString::const_new("::"),
            Self::Colon => CompactString::const_new(":"),
            Self::Dot => CompactString::const_new("."),
            Self::Dollar => CompactString::const_new("$"),
            Self::Ellipsis => CompactString::const_new("..."),
            Self::Equal => CompactString::const_new("="),
            Self::Hash => CompactString::const_new("#"),
            Self::Percent => CompactString::const_new("%"),
            Self::Pipe => CompactString::const_new("|"),
            Self::Plus => CompactString::const_new("+"),
            Self::Semicolon => CompactString::const_new(";"),
            Self::Slash => CompactString::const_new("/"),
            Self::Star => CompactString::const_new("*"),
            Self::Minus => CompactString::const_new("-"),
            Self::Rstab => CompactString::const_new("->"),
            Self::Question => CompactString::const_new("?"),
        }
    }
}

impl fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use core::fmt::Write;
        match self {
            // A token that carries text is shown as it is written
            Self::BareIdent(name) => f.write_str(name),
            // The token holds a quoted name unescaped: it is shown quoted again
            Self::AtIdent(name) if crate::ir::print::is_valid_bare_identifier(name) => {
                write!(f, "@{name}")
            }
            Self::AtIdent(name) => {
                let mut quoted = String::from("@");
                crate::ir::symbols::push_quoted(&mut quoted, name);
                f.write_str(&quoted)
            }
            Self::HashIdent(name) => write!(f, "#{name}"),
            Self::PercentIdent(name) => write!(f, "%{name}"),
            Self::CaretIdent(name) => write!(f, "^{name}"),
            Self::BangIdent(name) => write!(f, "!{name}"),
            // The token holds the text as written; an escaped line break is shown as `\n` (and
            // a carriage return or a tab as `\r` or `\t`), so that the token is shown on one line
            Self::String(text) => {
                f.write_char('"')?;
                let mut chars = text.chars().peekable();
                while let Some(c) = chars.next() {
                    match c {
                        '\\' if chars.peek() == Some(&'\n') => {
                            chars.next();
                            f.write_str("\\n")?;
                        }
                        '\n' => f.write_str("\\n")?,
                        '\r' => f.write_str("\\r")?,
                        '\t' => f.write_str("\\t")?,
                        c => f.write_char(c)?,
                    }
                }
                f.write_char('"')
            }
            Self::Int(digits) => f.write_str(digits),
            Self::Hex(digits) => write!(f, "0x{digits}"),
            Self::Binary(digits) => write!(f, "0b{digits}"),
            Self::Comment => f.write_str("comment"),
            Self::Eof => f.write_str("end of input"),
            Self::True => f.write_str("true"),
            Self::False => f.write_str("false"),
            Self::Loc => f.write_str("loc"),
            Self::IntType(width) => write!(f, "i{width}"),
            Self::Isize => f.write_str("isize"),
            Self::F32 => f.write_str("f32"),
            Self::F64 => f.write_str("f64"),
            Self::Never => f.write_str("never"),
            Self::Ptr => f.write_str("ptr"),
            Self::Struct => f.write_str("struct"),
            Self::Enum => f.write_str("enum"),
            Self::Array => f.write_str("array"),
            Self::Langle => f.write_char('<'),
            Self::Rangle => f.write_char('>'),
            Self::Lbrace => f.write_char('{'),
            Self::Rbrace => f.write_char('}'),
            Self::Lbracket => f.write_char('['),
            Self::Rbracket => f.write_char(']'),
            Self::Lparen => f.write_char('('),
            Self::Rparen => f.write_char(')'),
            Self::At => f.write_char('@'),
            Self::Bang => f.write_char('!'),
            Self::Comma => f.write_char(','),
            Self::ColonColon => f.write_str("::"),
            Self::Colon => f.write_char(':'),
            Self::Dot => f.write_char('.'),
            Self::Dollar => f.write_char('$'),
            Self::Ellipsis => f.write_str("..."),
            Self::Equal => f.write_char('='),
            Self::Hash => f.write_char('#'),
            Self::Percent => f.write_char('%'),
            Self::Pipe => f.write_char('|'),
            Self::Plus => f.write_char('+'),
            Self::Semicolon => f.write_char(';'),
            Self::Slash => f.write_char('/'),
            Self::Star => f.write_char('*'),
            Self::Minus => f.write_char('-'),
            Self::Rstab => f.write_str("->"),
            Self::Question => f.write_char('?'),
        }
    }
}

impl<'input> Token<'input> {
    /// Returns true if this token represents the name of an type or a type-related keyword.
    ///
    /// This is used to simplify diagnostic output related to expected tokens so as not to
    /// overwhelm the user with a ton of possible expected tokens.
    pub fn is_type_keyword(&self) -> bool {
        matches!(
            self,
            Token::IntType(_)
                | Token::Isize
                | Token::F32
                | Token::F64
                | Token::Never
                | Token::Ptr
                | Token::Array
                | Token::Struct
                | Token::Enum
        )
    }

    pub fn is_keyword(&self) -> bool {
        matches!(self, Token::True | Token::False | Token::Loc) || self.is_type_keyword()
    }

    /// Returns true if this token represents a known delimiter, e.g. `(`
    pub fn is_delimiter(&self) -> bool {
        matches!(
            self,
            Self::Lparen
                | Self::Rparen
                | Self::Lbracket
                | Self::Rbracket
                | Self::Langle
                | Self::Rangle
                | Self::Lbrace
                | Self::Rbrace
        )
    }

    /// Returns an appropriate [Token] depending on whether the given string is a keyword or an
    /// identifier.
    pub fn from_keyword_or_ident(s: &'input str) -> Self {
        match s {
            "true" => Self::True,
            "false" => Self::False,
            "loc" => Self::Loc,
            "isize" => Self::Isize,
            "f32" => Self::F32,
            "f64" => Self::F64,
            "never" => Self::Never,
            "ptr" => Self::Ptr,
            "struct" => Self::Struct,
            "enum" => Self::Enum,
            "array" => Self::Array,
            _ => match Self::int_type_width(s) {
                Some(width) => Self::IntType(width),
                None => Self::BareIdent(s),
            },
        }
    }

    /// The width of the integer type `s` names, if `s` is `i` followed by a decimal width with no
    /// leading zero, and that width fits in a `u32`. Any other word, such as `i99999999999`, is an
    /// identifier that keeps its spelling.
    fn int_type_width(s: &str) -> Option<u32> {
        let digits = s.strip_prefix('i')?;
        let mut chars = digits.chars();
        if !chars.next().is_some_and(|c| matches!(c, '1'..='9'))
            || !chars.all(|c| c.is_ascii_digit())
        {
            return None;
        }
        digits.parse::<u32>().ok()
    }
}
