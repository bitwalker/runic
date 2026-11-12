//! The builtin dialect's types, which print without a `!builtin.` prefix.

mod float;
mod integer;
mod special;

pub use self::{
    float::FloatType,
    integer::{IntegerType, IsizeType},
    special::{NeverType, UnknownType},
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    #[test]
    fn scalar_types_print_bare_and_are_cached() {
        let context = Context::default();
        for (ty, text) in [
            (context.i1(), "i1"),
            (context.i32(), "i32"),
            (context.int_type(256), "i256"),
            (context.int_type(24), "i24"),
            (context.isize_type(), "isize"),
            (context.f32_type(), "f32"),
            (context.f64_type(), "f64"),
            (context.unknown_type(), "?"),
            (context.never_type(), "never"),
        ] {
            assert_eq!(alloc::format!("{ty}"), text);
        }
        assert_eq!(context.i32(), context.int_type(32));
        assert_eq!(context.i32(), context.intern_type(IntegerType::new(32).unwrap()));
        assert_ne!(context.i32(), context.i64());
    }

    #[test]
    fn integer_queries_are_signless_and_isize_has_no_fixed_width() {
        let context = Context::default();
        assert!(context.i64().is_integer());
        assert_eq!(context.i64().integer_width(), Some(64));
        assert!(context.isize_type().is_integer() && context.isize_type().is_isize());
        assert_eq!(context.isize_type().integer_width(), None);
        assert!(!context.f64_type().is_integer() && context.f64_type().is_float());
        assert!(context.unknown_type().is_unknown() && context.never_type().is_never());
        assert_eq!(IntegerType::new(0), None);
        assert_eq!(IntegerType::new(IntegerType::MAX_WIDTH + 1), None);
        assert_eq!(context.try_int_type(0), None);
    }
}
