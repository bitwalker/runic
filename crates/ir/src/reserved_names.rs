//! Names of the procedures the backend generates in the modules of a component.
//!
//! No function a frontend defines may take one of [`RESERVED_PROCEDURE_NAMES`] in a module the
//! backend lowers.

/// The component initializer codegen emits into a component's root module, next to the exports;
/// `<namespace>::init` is therefore reserved and no export may use it.
pub const COMPONENT_INIT_PROCEDURE: &str = "init";

/// Every name in this module except [`COMPONENT_INIT_PROCEDURE`].
///
/// The initializer only exists in the root module, where the procedures a frontend defines are the
/// exports, which the frontend already keeps off that name; core functions live in nested modules,
/// so a core function named `init` collides with nothing.
pub const RESERVED_PROCEDURE_NAMES: &[&str] = &[];
