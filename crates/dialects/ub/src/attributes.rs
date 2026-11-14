mod poison;

pub use self::poison::PoisonAttr;

#[cfg(test)]
mod tests {
    use alloc::{rc::Rc, vec};

    use runic_core::{
        Context,
        testing::elements::{Element, Row, check_every_kind},
    };

    use super::*;

    /// Every attribute kind of `ub` lists and replaces what it holds: a poison value, its type.
    #[test]
    fn every_attribute_kind_lists_and_replaces_what_it_holds() {
        let context = Rc::new(Context::default());
        let i32 = context.i32();
        let rows = vec![Row::new(
            "poison: its type",
            context.create_attribute::<PoisonAttr, _>(i32),
            [Element::Type(i32)],
        )];
        check_every_kind(&context, &["ub"], &rows);
    }
}
