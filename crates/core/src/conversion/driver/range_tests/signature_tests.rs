//! A block's arguments are converted through a signature conversion: new arguments are appended,
//! the old ones erased, and an old argument that still has users is replaced by a cast at the
//! start of the block.

use super::*;
use crate::{
    BlockRef, EntityWithId, OpPrinter,
    conversion::{InputMapping, SignatureConversion},
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::builtin::attributes::AbiParam,
    effects::MemoryEffectOpInterface,
    traits::{BranchOpInterface, Terminator},
};

/// A branch that forwards its operands to its one successor: `test.forward_br ^bb1(%0 : i32);`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(Terminator),
    implements(BranchOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct ForwardBr {
    #[successor]
    target: Successor,
}

impl BranchOpInterface for ForwardBr {}

/// The block after the entry block of `function`.
fn second_block(function: OperationRef) -> BlockRef {
    let function = function.try_downcast_op::<Function>().unwrap();
    let entry = function.borrow().entry_block();
    entry.next().expect("a second block")
}

/// A rewriter under [wide_converter], outside a conversion.
fn wide_rewriter(context: &Rc<Context>) -> ConversionPatternRewriter {
    ConversionPatternRewriter::new_with_type_converter(
        context.clone(),
        Some(wide_converter(context)),
    )
}

/// `@f`'s arguments are available in its second block, `^bb1(%3: i128, %4: i32)`, whose arguments
/// a `test.sink` uses.
const SIGNATURE: &str = "(%0: i64, %1: i64, %2: i32)";
const BODY: &str = "    builtin.ret;
^bb1(%3: i128, %4: i32):
    test.sink %3, %4 : (i128, i32);
    builtin.ret;";

fn arguments(block: BlockRef) -> Vec<ValueRef> {
    block.borrow().argument_values().collect()
}

/// A block `(i128, i32)` under the converter becomes `(i64, i64, i32)`: the user of the old
/// `i128` reads a cast of the two new arguments placed first in the block, and the `i32` is
/// replaced by its new argument directly.
#[test]
fn an_argument_that_becomes_two_is_read_through_a_cast_at_the_start_of_the_block() {
    let (context, function) = parse_function(SIGNATURE, BODY);
    let block = second_block(function);
    let mut rewriter = wide_rewriter(&context);
    let conversion = SignatureConversion::for_block(block, &wide_converter(&context)).unwrap();
    let (i32, i64) = (context.i32(), context.i64());
    assert_eq!(conversion.converted_types(), [i64, i64, i32]);
    assert_eq!(
        conversion.mapping(0),
        Some(&InputMapping::Arguments {
            new_index: 0,
            count: 2
        })
    );
    assert!(!conversion.is_identity(block));

    let new = rewriter.apply_signature_conversion(block, &conversion).unwrap();

    assert_eq!(new.as_slice(), arguments(block).as_slice());
    assert_eq!(types(&new), [i64, i64, i32]);
    assert!(rewriter.mutation_count() > 0);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            SIGNATURE,
            "    builtin.ret;
^block2(%5: i64, %6: i64, %7: i32):
    %8 = builtin.unrealized_conversion_cast %5, %6 : (i64, i64) to i128;
    test.sink %8, %7 : (i128, i32);
    builtin.ret;"
        )
    );
}

/// New arguments that replace nothing, added between two mapped arguments, land between them.
#[test]
fn new_inputs_land_where_they_are_added() {
    let (context, function) = parse_function(SIGNATURE, BODY);
    let block = second_block(function);
    let (i1, i32, i64) = (context.i1(), context.i32(), context.i64());
    let mut conversion = SignatureConversion::new(2);
    conversion.add_inputs(0, &[i64, i64]).unwrap();
    conversion.add_new_inputs(&[i1]);
    conversion.add_inputs(1, &[i32]).unwrap();

    let new = wide_rewriter(&context).apply_signature_conversion(block, &conversion).unwrap();

    assert_eq!(types(&new), [i64, i64, i1, i32]);
    assert_eq!(
        conversion.mapping(1),
        Some(&InputMapping::Arguments {
            new_index: 3,
            count: 1
        })
    );
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            SIGNATURE,
            "    builtin.ret;
^block2(%5: i64, %6: i64, %7: i1, %8: i32):
    %9 = builtin.unrealized_conversion_cast %5, %6 : (i64, i64) to i128;
    test.sink %9, %8 : (i128, i32);
    builtin.ret;"
        )
    );
}

/// An argument remapped to one value of its own type is replaced by it directly; to two values,
/// by a cast of them; to none, by a cast of no inputs when it has a user, and by nothing when it
/// has none.
#[test]
fn a_remapped_argument_is_replaced_by_its_values() {
    type Remap = fn(&mut SignatureConversion, &[ValueRef], &Context);
    let cases: [(&str, &str, Remap, &str); 4] = [
        (
            "one value of its own type",
            BODY,
            |conversion, entry, context| {
                conversion.add_inputs(0, &[context.i64(), context.i64()]).unwrap();
                conversion.remap_input(1, &[entry[2]]).unwrap();
            },
            "    builtin.ret;
^block2(%5: i64, %6: i64):
    %7 = builtin.unrealized_conversion_cast %5, %6 : (i64, i64) to i128;
    test.sink %7, %2 : (i128, i32);
    builtin.ret;",
        ),
        (
            "two values",
            BODY,
            |conversion, entry, context| {
                conversion.remap_input(0, &[entry[0], entry[1]]).unwrap();
                conversion.add_inputs(1, &[context.i32()]).unwrap();
            },
            "    builtin.ret;
^block2(%5: i32):
    %6 = builtin.unrealized_conversion_cast %0, %1 : (i64, i64) to i128;
    test.sink %6, %5 : (i128, i32);
    builtin.ret;",
        ),
        (
            "no value, with a user",
            BODY,
            |conversion, _, context| {
                conversion.remap_input(0, &[]).unwrap();
                conversion.add_inputs(1, &[context.i32()]).unwrap();
            },
            "    builtin.ret;
^block2(%5: i32):
    %6 = builtin.unrealized_conversion_cast : () to i128;
    test.sink %6, %5 : (i128, i32);
    builtin.ret;",
        ),
        (
            "no value, without a user",
            "    builtin.ret;
^bb1(%3: i128, %4: i32):
    test.sink %4 : (i32);
    builtin.ret;",
            |conversion, _, context| {
                conversion.remap_input(0, &[]).unwrap();
                conversion.add_inputs(1, &[context.i32()]).unwrap();
            },
            "    builtin.ret;
^block2(%5: i32):
    test.sink %5 : (i32);
    builtin.ret;",
        ),
    ];
    for (name, body, remap, expected) in cases {
        let (context, function) = parse_function(SIGNATURE, body);
        let entry = defined_values(function);
        let block = second_block(function);
        let mut conversion = SignatureConversion::new(2);
        remap(&mut conversion, &entry, &context);

        wide_rewriter(&context)
            .apply_signature_conversion(block, &conversion)
            .unwrap_or_else(|err| panic!("{name}: {err}"));

        assert_str_eq!(function.borrow().to_string(), function_text(SIGNATURE, expected), "{name}");
    }
}

/// Mapping an argument twice, or one that does not exist, is an error of the conversion.
#[test]
fn an_argument_is_mapped_once() {
    let context = Context::default();
    let i32 = context.i32();
    let mut conversion = SignatureConversion::new(2);
    conversion.add_inputs(0, &[i32]).unwrap();
    let errors = [
        conversion.add_inputs(0, &[i32]).unwrap_err(),
        conversion.remap_input(0, &[]).unwrap_err(),
        conversion.add_inputs(2, &[i32]).unwrap_err(),
        conversion.add_inputs(1, &[]).unwrap_err(),
    ];
    let errors = errors.iter().map(|err| format!("{err}")).collect::<Vec<_>>();
    assert_eq!(
        errors,
        [
            "old argument 0 is already mapped",
            "old argument 0 is already mapped",
            "old argument 2 does not exist: the conversion has 2 old arguments",
            "old argument 1 must become at least one new argument; an argument that becomes no \
             value is remapped to none",
        ]
    );
    assert_eq!(conversion.mapping(1), None);
}

/// A conversion that leaves an argument unmapped, maps one to types the converter does not answer
/// for it, remaps one to an argument of the block itself, or is applied to a block already
/// converted, is an error raised before anything changes: the block prints as before.
#[test]
fn a_conversion_that_cannot_be_applied_changes_nothing() {
    type Build = fn(&Context, &[ValueRef]) -> SignatureConversion;
    let cases: [(&str, Build, &str); 4] = [
        (
            "unmapped",
            |context, _| {
                let mut conversion = SignatureConversion::new(2);
                conversion.add_inputs(1, &[context.i32()]).unwrap();
                conversion
            },
            "the signature conversion of a block of 'builtin.function' does not map argument 0",
        ),
        (
            "too few inputs",
            |_, _| SignatureConversion::new(1),
            "a signature conversion of 1 inputs cannot be applied to a block of \
             'builtin.function', which has 2 arguments",
        ),
        (
            "not what the converter answers",
            |context, _| {
                let mut conversion = SignatureConversion::new(2);
                conversion.add_inputs(0, &[context.i64()]).unwrap();
                conversion.add_inputs(1, &[context.i32()]).unwrap();
                conversion
            },
            "argument 0 of a block of 'builtin.function' has type 'i128', which the type \
             converter converts to (i64, i64), but the signature conversion gives it types (i64)",
        ),
        (
            "remapped to an argument of the block",
            |_, values| {
                let mut conversion = SignatureConversion::new(2);
                conversion.remap_input(0, &[]).unwrap();
                conversion.remap_input(1, &[values[4]]).unwrap();
                conversion
            },
            "argument 1 of a block of 'builtin.function' is remapped to '%4: i32', an argument of \
             the same block, which the conversion erases",
        ),
    ];
    for (name, build, message) in cases {
        let (context, function) = parse_function(SIGNATURE, BODY);
        let before = function.borrow().to_string();
        let block = second_block(function);
        let mut values = defined_values(function);
        values.extend(arguments(block));
        let conversion = build(&context, &values);
        let mut rewriter = wide_rewriter(&context);

        let err = rewriter.apply_signature_conversion(block, &conversion).unwrap_err();

        assert_eq!(format!("{err}"), message, "{name}");
        assert_str_eq!(function.borrow().to_string(), before, "{name}");
        assert_eq!(rewriter.mutation_count(), 0, "{name}");
    }

    // Applied twice by the rewriters of one conversion.
    let (context, function) = parse_function(SIGNATURE, BODY);
    let block = second_block(function);
    let mut rewriter = wide_rewriter(&context);
    let conversion = SignatureConversion::for_block(block, &wide_converter(&context)).unwrap();
    rewriter.apply_signature_conversion(block, &conversion).unwrap();
    let after = function.borrow().to_string();
    let again = SignatureConversion::for_block(block, &wide_converter(&context)).unwrap();
    let err = rewriter.apply_signature_conversion(block, &again).unwrap_err();
    assert_eq!(
        format!("{err}"),
        "the arguments of a block of 'builtin.function' were already converted in this conversion"
    );
    assert_str_eq!(function.borrow().to_string(), after);
}

/// `for_block` maps an argument that converts to no type to no value, and a block whose arguments
/// all convert to themselves is an identity.
#[test]
fn for_block_asks_the_converter_about_each_argument() {
    let (context, function) = parse_function(
        "()",
        "    builtin.ret;
^bb1(%0: i32, %1: array<i8; 0>):
    builtin.ret;
^bb2(%2: i32, %3: i64):
    builtin.ret;",
    );
    let converter = wide_converter(&context);
    let first = second_block(function);
    let second = first.next().unwrap();

    let conversion = SignatureConversion::for_block(first, &converter).unwrap();
    assert_eq!(conversion.converted_types(), [context.i32()]);
    assert_eq!(conversion.mapping(1), Some(&InputMapping::Values(SmallVec::new())));
    assert!(!conversion.is_identity(first));

    let identity = SignatureConversion::for_block(second, &converter).unwrap();
    assert!(identity.is_identity(second));
    assert!(!identity.is_identity(first));
}

/// The `builtin.function` whose signature has a type [wide_converter] changes: its signature is
/// set to the converted types, in place, and the blocks of its body are converted.
fn convert_function(context: &Rc<Context>, converter: &TypeConverter) -> Rule {
    Rule::new::<Function>(
        context,
        "convert-function",
        Some(converter.clone()),
        [],
        |op, _operands, rewriter| {
            if function_is_converted(&op.borrow()) {
                return Ok(false);
            }
            let context = op.borrow().context_rc();
            let converter = wide_converter(&context);
            rewriter.modify_op_in_place(op, |op| {
                let function = op.downcast_mut::<Function>().unwrap();
                let mut signature = function.get_signature_mut();
                let params = signature
                    .params()
                    .iter()
                    .flat_map(|param| converter.convert_type(&context, param.ty))
                    .map(AbiParam::new)
                    .collect::<Vec<_>>();
                signature.params_mut().clear();
                signature.params_mut().extend(params);
            })?;
            let body = op.borrow().regions().front().as_pointer().unwrap();
            rewriter.convert_region_types(body)?;
            Ok(true)
        },
    )
}

/// Whether the signature of the function `op` has no parameter of a type [wide_converter]
/// changes.
fn function_is_converted(op: &Operation) -> bool {
    let i128 = op.context().i128();
    let function = op.downcast_ref::<Function>().unwrap();
    !function.get_signature().params().iter().any(|param| param.ty == i128)
}

/// A `test.forward_br` that forwards a type [wide_converter] changes forwards the values its
/// operands became, flattened.
fn flatten_branch(context: &Rc<Context>, converter: &TypeConverter) -> Rule {
    Rule::new::<ForwardBr>(
        context,
        "flatten-branch",
        Some(converter.clone()),
        [registered::<ForwardBr>(context)],
        |op, operands, rewriter| {
            if has_no_wide_types(&op.borrow()) {
                return Ok(false);
            }
            let flat = operands.iter().flatten().copied().collect::<Vec<_>>();
            let dest = op.borrow().successor(0).successor();
            let span = op.borrow().span();
            rewriter.set_insertion_point_before(op);
            rewriter.create_op::<ForwardBr, _>(span, (dest, flat))?;
            rewriter.erase_op(op)?;
            Ok(true)
        },
    )
}

/// [wide_target], where a function is legal once its signature is converted, and a
/// `test.forward_br` once it forwards no type the converter changes (or always, when
/// `branch_is_legal`).
fn function_target(context: &Rc<Context>, branch_is_legal: bool) -> ConversionTarget {
    let mut target = wide_target(context);
    target.add_dynamically_legal_op::<Function, _>(|op: &Operation| {
        DynamicLegalityResult::legal_if(function_is_converted(op))
    });
    if !branch_is_legal {
        target.add_dynamically_legal_op::<ForwardBr, _>(|op: &Operation| {
            DynamicLegalityResult::legal_if(has_no_wide_types(op))
        });
    }
    target
}

const TWO_BLOCKS: &str = "    test.forward_br ^bb1(%0 : i128);
^bb1(%1: i128):
    test.sink %1 : (i128);
    builtin.ret;";

/// A two-block function converted by a pattern on the function that converts its body's types,
/// with a pattern for its branch that forwards the flattened operands: the branch forwards the new
/// arguments, the sink is given them, no cast remains, and the result verifies.
#[test]
fn a_function_whose_blocks_and_branch_are_converted_verifies() {
    let (context, function) = parse_function("(%0: i128)", TWO_BLOCKS);
    let converter = wide_converter(&context);
    let sinks = Rc::new(RefCell::new(vec![]));

    convert(
        function,
        function_target(&context, false),
        [
            convert_function(&context, &converter),
            flatten_branch(&context, &converter),
            flatten_sink(&context, &converter, sinks.clone()),
        ],
    )
    .unwrap_or_else(|err| panic!("{}", describe(&err)));

    let block = second_block(function);
    assert_eq!(*sinks.borrow(), [vec![vec![arguments(block)], vec![]]]);
    assert_str_eq!(
        function.borrow().to_string(),
        function_text(
            "(%2: i64, %3: i64)",
            "    test.forward_br ^block2(%2, %3 : i64, i64);
^block2(%5: i64, %6: i64):
    test.sink %5, %6 : (i64, i64);
    builtin.ret;"
        )
    );
}

/// Without a pattern for the branch, the converted block's predecessor still forwards the old
/// value: the conversion fails verification, and says which edge.
#[test]
fn a_converted_block_whose_branch_is_not_converted_fails_verification() {
    let (context, function) = parse_function("(%0: i128)", TWO_BLOCKS);
    let mut converter = wide_converter(&context);
    join_halves(&mut converter, Rc::default());

    let err = convert(
        function,
        function_target(&context, true),
        [
            convert_function(&context, &converter),
            flatten_sink(&context, &converter, Rc::default()),
        ],
    )
    .unwrap_err();

    let err = format!("{err}");
    assert!(err.contains("successor 0 (^block2"), "{err}");
    assert!(
        err.contains("takes 2 block argument(s), but test.forward_br gives it 1"),
        "{err}"
    );
}

/// A conversion that needs a cast for an argument with users, applied by a rewriter without a type
/// converter, is an error raised before anything changes.
#[test]
fn a_needed_cast_without_a_type_converter_is_an_error() {
    let (context, function) = parse_function(SIGNATURE, BODY);
    let before = function.borrow().to_string();
    let block = second_block(function);
    let (i32, i64) = (context.i32(), context.i64());
    let mut conversion = SignatureConversion::new(2);
    conversion.add_inputs(0, &[i64, i64]).unwrap();
    conversion.add_inputs(1, &[i32]).unwrap();
    let mut rewriter = ConversionPatternRewriter::new(context.clone());

    let err = rewriter.apply_signature_conversion(block, &conversion).unwrap_err();

    assert_eq!(
        format!("{err}"),
        "argument 0 of a block of 'builtin.function' of type 'i128' becomes values of types (i64, \
         i64), which needs a cast, and the pattern has no type converter"
    );
    assert_str_eq!(function.borrow().to_string(), before);
    assert_eq!(rewriter.mutation_count(), 0);
}

/// Converting the arguments of a block that has no parent operation counts as a mutation.
#[test]
fn converting_a_block_without_a_parent_operation_is_a_mutation() {
    let context = Rc::new(Context::default());
    let block = context.create_block_with_params([context.i128()]);
    let conversion = SignatureConversion::for_block(block, &wide_converter(&context)).unwrap();
    let mut rewriter = wide_rewriter(&context);

    let new = rewriter.apply_signature_conversion(block, &conversion).unwrap();

    let i64 = context.i64();
    assert_eq!(types(&new), [i64, i64]);
    assert_eq!(new.as_slice(), arguments(block).as_slice());
    assert_eq!(rewriter.mutation_count(), 1);
}
