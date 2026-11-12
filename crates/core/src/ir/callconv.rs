//! Calling conventions, which are open names rather than a closed set.

use core::fmt;

use crate::interner::Symbol;

/// A calling convention, identified by its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CallConv(Symbol);

impl CallConv {
    pub fn new(name: impl AsRef<str>) -> Self {
        Self(Symbol::intern(name))
    }

    /// The C calling convention, `C`.
    pub fn c() -> Self {
        Self::new("C")
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl Default for CallConv {
    fn default() -> Self {
        Self::c()
    }
}

impl From<&str> for CallConv {
    fn from(name: &str) -> Self {
        Self::new(name)
    }
}

impl fmt::Display for CallConv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calling_conventions_are_open_names() {
        assert_eq!(CallConv::default(), CallConv::c());
        assert_eq!(CallConv::c().as_str(), "C");
        assert_eq!(CallConv::new("fast"), CallConv::from("fast"));
        assert_ne!(CallConv::new("fast"), CallConv::c());
        assert_eq!(alloc::format!("{}", CallConv::new("my-frontend-abi")), "my-frontend-abi");
    }
}
