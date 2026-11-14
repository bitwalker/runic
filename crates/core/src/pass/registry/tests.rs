use alloc::{string::ToString, vec::Vec};

use super::*;
use crate::{EntityMut, Operation, OperationName};

macro_rules! test_pass {
    ($ty:ident, $argument:literal) => {
        test_pass!($ty, stringify!($ty), $argument);
    };
    ($ty:ident, $name:expr, $argument:literal) => {
        #[derive(Default)]
        struct $ty;

        impl Pass for $ty {
            type Target = Operation;

            fn name(&self) -> &'static str {
                $name
            }

            fn argument(&self) -> &'static str {
                $argument
            }

            fn description(&self) -> &'static str {
                "a pass of the registry tests"
            }

            fn can_schedule_on(&self, _name: &OperationName) -> bool {
                true
            }

            fn run_on_operation(
                &mut self,
                _op: EntityMut<'_, Operation>,
                _state: &mut PassExecutionState,
            ) -> Result<(), Report> {
                Ok(())
            }
        }
    };
}

test_pass!(FirstOwner, "registry-test-owned");
test_pass!(SecondOwner, "registry-test-owned");
test_pass!(Lone, "registry-test-lone");
test_pass!(Twin, "registry-test-twin", "registry-test-twin");
test_pass!(OtherTwin, "registry-test-twin", "registry-test-twin");

fn entry<P: Pass + Default>(arg: &'static str) -> PassRegistryEntry {
    PassInfo::new::<P>(arg, "a pass of the registry tests").0
}

#[test]
fn a_dynamic_registration_of_a_taken_name_is_an_error() {
    register_pass(default_instance::<FirstOwner>).unwrap();
    // The same type again is not a second registration.
    register_pass(default_instance::<FirstOwner>).unwrap();

    let error = register_pass(default_instance::<SecondOwner>).unwrap_err().to_string();
    assert!(error.contains("registry-test-owned"), "{error}");
    assert!(error.contains("FirstOwner") && error.contains("SecondOwner"), "{error}");

    // The first registration is untouched.
    let info = PassInfo::lookup("registry-test-owned").unwrap();
    assert_eq!(info.0.type_id, Some(TypeId::of::<FirstOwner>()));
}

#[test]
fn two_dynamic_passes_of_one_name_are_two_passes() {
    // `Twin` and `OtherTwin` are different types that call themselves by one name.
    register_pass(default_instance::<Twin>).unwrap();
    let error = register_pass(default_instance::<OtherTwin>).unwrap_err().to_string();
    assert_eq!(
        error,
        "cannot register pass 'registry-test-twin' for 'registry-test-twin': the name is already \
         registered for 'registry-test-twin'"
    );
    let info = PassInfo::lookup("registry-test-twin").unwrap();
    assert_eq!(info.0.type_id, Some(TypeId::of::<Twin>()));
}

#[test]
fn a_registered_pass_is_listed_by_name() {
    register_pass(default_instance::<Lone>).unwrap();
    let names = PassInfo::registered().iter().map(PassInfo::argument).collect::<Vec<_>>();
    assert!(names.contains(&"registry-test-lone"));
    assert!(names.is_sorted());
    let info = PassInfo::lookup("registry-test-lone").unwrap();
    assert_eq!(info.description(), "a pass of the registry tests");
}

#[test]
fn building_a_registry_from_two_passes_of_one_name_is_an_error() {
    let error = PassRegistry::from_entries(
        [entry::<FirstOwner>("shared"), entry::<SecondOwner>("shared")],
        [],
    )
    .err()
    .expect("a duplicate name")
    .to_string();
    assert!(error.contains("'shared'"), "{error}");
    assert!(error.contains("FirstOwner") && error.contains("SecondOwner"), "{error}");

    let Ok(registry) =
        PassRegistry::from_entries([entry::<FirstOwner>("a"), entry::<SecondOwner>("b")], [])
    else {
        panic!("distinct names are not a duplicate");
    };
    assert_eq!(registry.passes().iter().map(PassInfo::argument).collect::<Vec<_>>(), ["a", "b"]);
}

#[test]
fn a_pass_submitted_by_two_copies_of_its_crate_is_one_pass() {
    // Two copies of a crate define the pass as two types of one path.
    let mut copy = entry::<FirstOwner>("copied");
    copy.type_id = Some(TypeId::of::<Lone>());
    let Ok(registry) = PassRegistry::from_entries([entry::<FirstOwner>("copied"), copy], []) else {
        panic!("two copies of one pass are not a duplicate");
    };
    assert_eq!(registry.passes().len(), 1);
}

#[test]
fn a_dynamic_pipeline_under_a_taken_name_is_an_error() {
    let registry = PassRegistry::from_entries([], []).ok().unwrap();
    registry
        .register_pipeline(PassPipelineInfo::new("pipe", "a pipeline", default_instance::<Lone>))
        .unwrap();
    // Even with the same builder: a pipeline records no type to tell two pipelines apart.
    for builder in [
        default_instance::<Lone> as PassAllocatorFunction,
        default_instance::<FirstOwner>,
    ] {
        let error = registry
            .register_pipeline(PassPipelineInfo::new("pipe", "a pipeline", builder))
            .unwrap_err()
            .to_string();
        assert!(error.contains("pass pipeline 'pipe'"), "{error}");
    }
    assert_eq!(registry.pipelines().len(), 1);
}

#[test]
fn submitted_pipelines_of_one_name_are_one_pipeline() {
    // Two copies of a crate submit its pipelines twice, and a pipeline records no type.
    let first = PassPipelineInfo::new("pipe", "a pipeline", default_instance::<Lone>).0;
    let second = PassPipelineInfo::new("pipe", "a pipeline", default_instance::<FirstOwner>).0;
    let Ok(registry) = PassRegistry::from_entries([], [first, second]) else {
        panic!("two submissions of one pipeline name are one pipeline");
    };
    assert_eq!(registry.pipelines().len(), 1);
}

#[test]
fn the_edit_distance_counts_single_character_edits() {
    assert_eq!(edit_distance("", ""), 0);
    assert_eq!(edit_distance("cse", "cse"), 0);
    assert_eq!(edit_distance("cze", "cse"), 1);
    assert_eq!(edit_distance("cs", "cse"), 1);
    assert_eq!(edit_distance("csee", "cse"), 1);
    assert_eq!(edit_distance("", "dce"), 3);
    assert_eq!(edit_distance("kitten", "sitting"), 3);
}
