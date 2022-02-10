mod wordmaps;
mod symspell;
mod composition;
mod edit_distance;
mod suggestion;

pub use wordmaps::WordRepr;
pub use symspell::{SymSpell, Verbosity};
pub use suggestion::Suggestion;
pub use composition::Composition;
