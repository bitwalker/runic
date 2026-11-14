use alloc::{format, rc::Rc, string::String, vec::Vec};
use core::cell::RefCell;

use smallvec::SmallVec;

use crate::{Context, FxHashMap, Report, Type, ValueRef};

type TypeConversionFn = Rc<dyn Fn(&Context, Type) -> Option<TypeConversion>>;
type ValueConversionFn = Rc<dyn Fn(&Context, ValueRef) -> Option<TypeConversion>>;

/// What a conversion callback answers for one type or value: the types it becomes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeConversion {
    /// The value becomes one value of this type.
    One(Type),
    /// The value becomes one value of each of these types, in order. A list of one type means
    /// the same as [Self::One], and an empty list the same as [Self::Drop].
    Many(SmallVec<[Type; 2]>),
    /// The value becomes no value.
    Drop,
}

impl TypeConversion {
    /// The value becomes one value of type `ty`.
    #[inline]
    pub fn one(ty: Type) -> Self {
        Self::One(ty)
    }

    /// The value becomes one value of each of `types`, in order.
    #[inline]
    pub fn many(types: impl IntoIterator<Item = Type>) -> Self {
        Self::Many(types.into_iter().collect())
    }

    /// The value becomes no value.
    #[inline]
    pub const fn drop() -> Self {
        Self::Drop
    }

    /// The types the value becomes.
    fn into_types(self) -> SmallVec<[Type; 2]> {
        match self {
            Self::One(ty) => smallvec::smallvec![ty],
            Self::Many(types) => types,
            Self::Drop => SmallVec::new(),
        }
    }
}

/// Says what types the values of a conversion become.
///
/// A converter is flat: it is asked about the type of a value (an operand, a result, a block
/// argument), never about a type held inside another type. A callback that wants a deep map
/// builds a [TypeMapper](crate::types::TypeMapper) in the call.
///
/// Type callbacks are tried in the order they were added, and the first that answers wins; a type
/// no callback answers for converts to itself. Value callbacks are tried before type callbacks.
/// Each callback is given the context of the IR being converted, in which it builds the types it
/// converts to.
///
/// The converter remembers what the type callbacks answered for a type, so each type is asked
/// about once. The memo is shared by the clones of a converter, and a clone that is given another
/// callback starts a memo of its own. An answer of a value callback is not remembered.
#[derive(Clone, Default)]
pub struct TypeConverter {
    conversions: Vec<TypeConversionFn>,
    value_conversions: Vec<ValueConversionFn>,
    memo: Rc<RefCell<FxHashMap<Type, SmallVec<[Type; 2]>>>>,
}

impl TypeConverter {
    /// A converter under which every type converts to itself.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a type conversion callback.
    ///
    /// Return `Some` to answer for the type, or `None` to let later callbacks try. Callbacks are
    /// tried in the order they were added, and are given the context the type belongs to.
    pub fn add_conversion<F>(&mut self, callback: F) -> &mut Self
    where
        F: Fn(&Context, Type) -> Option<TypeConversion> + 'static,
    {
        self.conversions.push(Rc::new(callback));
        self.forget_answers();
        self
    }

    /// Add a value conversion callback.
    ///
    /// Value callbacks are tried before type callbacks and may look at the value's defining
    /// operation, its uses, or its type. What they answer is not remembered.
    pub fn add_value_conversion<F>(&mut self, callback: F) -> &mut Self
    where
        F: Fn(&Context, ValueRef) -> Option<TypeConversion> + 'static,
    {
        self.value_conversions.push(Rc::new(callback));
        self.forget_answers();
        self
    }

    /// The types a value of type `ty` becomes: itself alone when it is legal, several, or none.
    pub fn convert_type(&self, context: &Context, ty: Type) -> SmallVec<[Type; 2]> {
        if let Some(types) = self.memo.borrow().get(&ty) {
            return types.clone();
        }
        let types = self
            .conversions
            .iter()
            .find_map(|convert| convert(context, ty))
            .map_or_else(|| smallvec::smallvec![ty], TypeConversion::into_types);
        self.memo.borrow_mut().insert(ty, types.clone());
        types
    }

    /// The types `value` becomes: what the first value callback answers, otherwise what its type
    /// becomes ([Self::convert_type]).
    pub fn convert_value(&self, context: &Context, value: ValueRef) -> SmallVec<[Type; 2]> {
        match self.value_conversions.iter().find_map(|convert| convert(context, value)) {
            Some(conversion) => conversion.into_types(),
            None => self.convert_type(context, value.borrow().ty()),
        }
    }

    /// Whether `ty` converts to exactly itself.
    #[inline]
    pub fn is_legal_type(&self, context: &Context, ty: Type) -> bool {
        self.convert_type(context, ty).as_slice() == [ty]
    }

    /// The one type `ty` converts to, or an error when it converts to several or none.
    pub fn convert_type_1_to_1(&self, context: &Context, ty: Type) -> Result<Type, Report> {
        exactly_one(self.convert_type(context, ty), || format!("type '{ty}'"))
    }

    /// The one type `value` converts to ([Self::convert_value]), or an error when it converts to
    /// several or none.
    pub fn convert_value_1_to_1(&self, context: &Context, value: ValueRef) -> Result<Type, Report> {
        exactly_one(self.convert_value(context, value), || {
            format!("value '{}' of type '{}'", value.borrow(), value.borrow().ty())
        })
    }

    /// Start a memo of this converter's own: its callbacks are no longer those of its clones.
    fn forget_answers(&mut self) {
        self.memo = Rc::default();
    }
}

fn exactly_one(
    types: SmallVec<[Type; 2]>,
    describe: impl FnOnce() -> String,
) -> Result<Type, Report> {
    match types.as_slice() {
        [ty] => Ok(*ty),
        [] => Err(Report::msg(format!(
            "{} is dropped by the type converter, but a 1:1 conversion is required",
            describe()
        ))),
        types => Err(Report::msg(format!(
            "{} converts to {} types, but a 1:1 conversion is required",
            describe(),
            types.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use smallvec::smallvec;

    use super::*;
    use crate::{Context, testing::Test};

    #[test]
    fn defaults_to_identity_conversion() {
        let context = Context::default();
        let converter = TypeConverter::new();

        assert_eq!(converter.convert_type_1_to_1(&context, context.i32()).unwrap(), context.i32());
        assert_eq!(converter.convert_type(&context, context.i32()).as_slice(), [context.i32()]);
        assert!(converter.is_legal_type(&context, context.i32()));
    }

    #[test]
    fn applies_registered_type_conversion() {
        let context = Context::default();
        let (i32, i64) = (context.i32(), context.i64());
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| {
            if ty == i32 {
                Some(TypeConversion::One(i64))
            } else {
                None
            }
        });

        assert_eq!(converter.convert_type_1_to_1(&context, i32).unwrap(), i64);
        assert!(!converter.is_legal_type(&context, i32));
    }

    #[test]
    fn a_conversion_can_build_types_in_the_context_it_is_given() {
        let context = Context::default();
        let mut converter = TypeConverter::new();
        converter.add_conversion(|context, ty| {
            (!ty.is_pointer()).then(|| TypeConversion::One(context.ptr_type(ty)))
        });

        let i32 = context.i32();
        assert_eq!(converter.convert_type_1_to_1(&context, i32).unwrap(), context.ptr_type(i32));
        assert!(converter.is_legal_type(&context, context.opaque_ptr_type()));
    }

    /// `Many` gives its types in order; `Many` of one type is `One`, and of none `Drop`.
    #[test]
    fn a_type_converts_to_several_one_or_no_types() {
        let context = Context::default();
        let (i8, i16, i32, i64, i128) =
            (context.i8(), context.i16(), context.i32(), context.i64(), context.i128());
        let mut converter = TypeConverter::new();
        converter.add_conversion(move |_, ty| match ty {
            ty if ty == i128 => Some(TypeConversion::many([i64, i32])),
            ty if ty == i16 => Some(TypeConversion::many([i32])),
            ty if ty == i8 => Some(TypeConversion::many([])),
            ty if ty == i64 => Some(TypeConversion::Drop),
            _ => None,
        });

        assert_eq!(converter.convert_type(&context, i128).as_slice(), [i64, i32]);
        assert_eq!(converter.convert_type(&context, i16).as_slice(), [i32]);
        assert_eq!(converter.convert_type_1_to_1(&context, i16).unwrap(), i32);
        assert!(converter.convert_type(&context, i8).is_empty());
        assert!(converter.convert_type(&context, i64).is_empty());
        assert!(!converter.is_legal_type(&context, i8));
        assert!(converter.is_legal_type(&context, i32));
    }

    #[test]
    fn rejects_many_and_drop_for_1_to_1_queries() {
        let context = Context::default();
        let (i32, i64) = (context.i32(), context.i64());
        let mut many = TypeConverter::new();
        many.add_conversion(move |_, ty| {
            if ty == i32 {
                Some(TypeConversion::Many(smallvec![i64, i64]))
            } else {
                None
            }
        });
        assert!(
            format!("{}", many.convert_type_1_to_1(&context, i32).unwrap_err()).contains("1:1")
        );

        let mut drop = TypeConverter::new();
        drop.add_conversion(move |_, ty| {
            if ty == i32 {
                Some(TypeConversion::Drop)
            } else {
                None
            }
        });
        assert!(
            format!("{}", drop.convert_type_1_to_1(&context, i32).unwrap_err()).contains("1:1")
        );
    }

    /// A type callback is asked about a type once, however many values have it; the clones of a
    /// converter share what it answered, and a clone given another callback asks again.
    #[test]
    fn a_type_is_asked_about_once_until_a_conversion_is_added() {
        let test = Test::new("f", |cx| ([cx.i128(), cx.i128(), cx.i128()], []));
        let context = test.context_rc();
        let (i64, i128) = (context.i64(), context.i128());
        let values = {
            let entry = test.entry_block();
            let entry = entry.borrow();
            entry.arguments().iter().map(|arg| *arg as ValueRef).collect::<Vec<_>>()
        };

        let calls = Rc::new(Cell::new(0));
        let mut converter = TypeConverter::new();
        {
            let calls = Rc::clone(&calls);
            converter.add_conversion(move |_, ty| {
                calls.set(calls.get() + 1);
                (ty == i128).then(|| TypeConversion::many([i64, i64]))
            });
        }

        for value in &values {
            assert_eq!(converter.convert_value(&context, *value).as_slice(), [i64, i64]);
        }
        assert_eq!(calls.get(), 1, "one type, asked about once");

        let mut clone = converter.clone();
        assert_eq!(clone.convert_type(&context, i128).as_slice(), [i64, i64]);
        assert_eq!(calls.get(), 1, "a clone shares the answers");

        clone.add_conversion(|_, _| None);
        assert_eq!(clone.convert_type(&context, i128).as_slice(), [i64, i64]);
        assert_eq!(calls.get(), 2, "a clone given another callback asks again");
        assert_eq!(converter.convert_type(&context, i128).as_slice(), [i64, i64]);
        assert_eq!(calls.get(), 2, "the original keeps its answers");
    }

    /// A value callback is asked every time; a value it does not answer for falls back to its
    /// type.
    #[test]
    fn a_value_callback_is_not_remembered() {
        let test = Test::new("f", |cx| ([cx.i32(), cx.i32()], []));
        let context = test.context_rc();
        let (first, second) = {
            let entry = test.entry_block();
            let entry = entry.borrow();
            (entry.arguments()[0] as ValueRef, entry.arguments()[1] as ValueRef)
        };
        let calls = Rc::new(Cell::new(0));
        let mut converter = TypeConverter::new();
        {
            let calls = Rc::clone(&calls);
            converter.add_value_conversion(move |_, value| {
                calls.set(calls.get() + 1);
                (value == first).then(TypeConversion::drop)
            });
        }

        assert!(converter.convert_value(&context, first).is_empty());
        assert!(converter.convert_value(&context, first).is_empty());
        assert_eq!(converter.convert_value(&context, second).as_slice(), [context.i32()]);
        assert_eq!(calls.get(), 3);
    }
}
