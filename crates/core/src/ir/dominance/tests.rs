//! The tables of a dominator tree, and of the Semi-NCA construction that builds it, hold the
//! blocks of the tree's region only: building trees for many functions of one context costs time
//! in proportion to their blocks, not to every block the context has created.
//!
//! The tests assert on the size of the tables rather than on time, which would be flaky.

use alloc::{format, rc::Rc, string::ToString, vec::Vec};

use super::{DominanceTree, SemiNCA};
use crate::{
    BlockRef, Context, EntityId, EntityWithId, OpPrinter, Operation, RegionRef,
    derive::{EffectOpInterface, OpParser, OpPrinter, operation},
    dialects::test::TestDialect,
    effects::MemoryEffectOpInterface,
    parse::{ParserConfig, parse_any},
    testing::text::describe,
    traits::{BranchOpInterface, Terminator},
};

/// A branch to either of two blocks: `test.tree_either ^bb1, ^bb2;`.
#[derive(EffectOpInterface, OpPrinter, OpParser)]
#[operation(
    dialect = TestDialect,
    traits(Terminator),
    implements(BranchOpInterface, MemoryEffectOpInterface, OpPrinter)
)]
pub struct TreeEither {
    #[successor]
    first: Successor,
    #[successor]
    second: Successor,
}

impl BranchOpInterface for TreeEither {}

/// How many functions the module holds.
const FUNCTIONS: usize = 2000;

/// A module of `FUNCTIONS` functions, each a diamond: an entry that branches to `^left` or
/// `^right`, which both go to `^join`. Returns the body of each function.
fn diamonds(context: &Rc<Context>) -> Vec<RegionRef> {
    let mut source = alloc::string::String::from("builtin.module public @m {\n");
    for index in 0..FUNCTIONS {
        source.push_str(&format!(
            "builtin.function public extern(\"C\") @f{index}() {{
    test.tree_either ^left, ^right;
^left:
    test.tree_either ^join, ^join;
^right:
    test.tree_either ^join, ^join;
^join:
    builtin.ret;
}};
"
        ));
    }
    source.push_str("};\n");
    let config = ParserConfig::new(context.clone()).verify_after_parse(false);
    let module = parse_any(config, "trees.runic", source)
        .into_result()
        .unwrap_or_else(|err| panic!("the input parses: {}", describe(&err)));
    let mut bodies = Vec::new();
    module.borrow().prewalk_all(|op: &Operation| {
        if op.name().to_string() == "builtin.function" {
            bodies.push(op.regions().front().as_pointer().unwrap());
        }
    });
    assert_eq!(bodies.len(), FUNCTIONS);
    bodies
}

/// The blocks of `region`, in order.
fn blocks(region: RegionRef) -> Vec<BlockRef> {
    let region = region.borrow();
    let mut blocks = Vec::new();
    let mut cursor = region.body().front();
    while let Some(block) = cursor.as_pointer() {
        cursor.move_next();
        blocks.push(block);
    }
    blocks
}

#[test]
fn a_dominator_tree_is_sized_by_its_region() {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    let bodies = diamonds(&context);
    // The last function's blocks have ids near the end of every block the context created.
    let last = blocks(*bodies.last().unwrap());
    assert!(last[0].borrow().id().as_usize() >= 4 * (FUNCTIONS - 1), "{}", last[0]);

    for body in bodies {
        let [entry, left, right, join] = blocks(body)[..] else {
            panic!("a diamond has four blocks");
        };

        let tree = DominanceTree::new(body).unwrap();
        assert_eq!(tree.table_len(), 4);
        assert_eq!(tree.len(), 4);
        assert!(tree.properly_dominates(Some(entry), Some(join)));
        assert!(!tree.dominates(Some(left), Some(join)));
        assert!(!tree.dominates(Some(right), Some(join)));
        assert_eq!(tree.get(Some(join)).unwrap().idom().unwrap().block(), Some(entry));
        assert_eq!(tree.root_node().unwrap().num_children(), 3);
        assert!(tree.is_reachable_from_entry(join));
    }
}

#[test]
fn the_construction_is_sized_by_the_blocks_it_visits() {
    let context = Rc::new(Context::default());
    context.get_or_register_dialect::<TestDialect>();
    let bodies = diamonds(&context);
    let body = *bodies.last().unwrap();
    let tree = DominanceTree::new(body).unwrap();

    // One entry per block, and one for the virtual node.
    let mut snca = SemiNCA::<false>::new(None);
    snca.do_full_dfs_walk(&tree, |_, _| true);
    snca.run();
    assert_eq!(snca.table_len(), 5);
    let [entry, _, _, join] = blocks(body)[..] else {
        panic!("a diamond has four blocks");
    };
    assert_eq!(snca.idom(Some(join)), Some(entry));
}
