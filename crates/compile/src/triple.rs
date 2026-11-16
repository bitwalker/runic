use core::{fmt, str::FromStr};

use runic_ir::Report;

/// The name of a target, in triple form: `wasm32-unknown-unknown`, `aarch64-apple-darwin`,
/// `x86_64-unknown-linux-gnu`.
///
/// A triple has two to four parts separated by `-`. Each part is one or more ASCII letters,
/// digits, `_` or `.`. The first part is the architecture. The other parts are not interpreted:
/// the names in use do not agree on what stands where (`wasm32-wasip2` has no vendor,
/// `thumbv7em-none-eabihf` has an environment and no vendor), so what they mean is for the
/// backend that supports the target to say.
///
/// Two triples are the same target when their text is the same. Nothing is normalized:
/// `x86_64-linux-gnu` and `x86_64-unknown-linux-gnu` are different triples.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Triple(Box<str>);

impl Triple {
    /// The most parts a triple has.
    const MAX_PARTS: usize = 4;
    /// The fewest parts a triple has.
    const MIN_PARTS: usize = 2;

    /// The triple as it was written.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The architecture: the first part.
    pub fn arch(&self) -> &str {
        self.parts().next().expect("a triple has at least two parts")
    }

    /// The parts, in order, the architecture first.
    pub fn parts(&self) -> impl Iterator<Item = &str> {
        self.0.split('-')
    }
}

impl FromStr for Triple {
    type Err = Report;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text.is_empty() {
            return Err(Report::msg("invalid target triple: it is empty"));
        }
        let mut parts = 0;
        for part in text.split('-') {
            parts += 1;
            if part.is_empty() {
                return Err(Report::msg(format!(
                    "invalid target triple '{text}': part {parts} is empty"
                )));
            }
            if let Some(invalid) =
                part.chars().find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.')))
            {
                return Err(Report::msg(format!(
                    "invalid target triple '{text}': '{invalid}' cannot appear in a triple (its \
                     parts are ASCII letters, digits, '_' and '.', separated by '-')"
                )));
            }
        }
        if !(Self::MIN_PARTS..=Self::MAX_PARTS).contains(&parts) {
            return Err(Report::msg(format!(
                "invalid target triple '{text}': a triple has {} to {} parts separated by '-', \
                 but this has {parts}",
                Self::MIN_PARTS,
                Self::MAX_PARTS
            )));
        }
        Ok(Self(text.into()))
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triple(text: &str) -> Triple {
        text.parse()
            .unwrap_or_else(|error| panic!("'{text}' should be a triple: {error}"))
    }

    fn rejection(text: &str) -> String {
        match text.parse::<Triple>() {
            Ok(triple) => panic!("'{text}' should not be a triple, but is {triple:?}"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn triples_of_two_to_four_parts_are_accepted() {
        for text in [
            "wasm32-wasip2",
            "wasm32-unknown-unknown",
            "aarch64-apple-darwin",
            "thumbv7em-none-eabihf",
            "x86_64-unknown-linux-gnu",
            "riscv32imac-unknown-none-elf",
            "armv7-unknown-linux-gnueabi.2.31",
        ] {
            assert_eq!(triple(text).as_str(), text);
        }
    }

    #[test]
    fn an_architecture_no_other_tool_knows_is_a_triple() {
        let triple = triple("stackvm-unknown-unknown");
        assert_eq!(triple.arch(), "stackvm");
    }

    #[test]
    fn the_architecture_is_the_first_part_and_the_parts_come_in_order() {
        let triple = triple("x86_64-unknown-linux-gnu");
        assert_eq!(triple.arch(), "x86_64");
        assert_eq!(triple.parts().collect::<Vec<_>>(), ["x86_64", "unknown", "linux", "gnu"]);
        assert_eq!(self::triple("wasm32-wasip2").parts().collect::<Vec<_>>(), ["wasm32", "wasip2"]);
    }

    #[test]
    fn a_triple_prints_as_it_was_written_and_reads_back() {
        for text in ["wasm32-wasip2", "aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
            let triple = triple(text);
            assert_eq!(triple.to_string(), text);
            assert_eq!(self::triple(&triple.to_string()), triple);
        }
    }

    #[test]
    fn triples_are_compared_as_text_and_nothing_is_normalized() {
        assert_eq!(triple("wasm32-unknown-unknown"), triple("wasm32-unknown-unknown"));
        assert_ne!(triple("x86_64-linux-gnu"), triple("x86_64-unknown-linux-gnu"));
        assert_ne!(triple("WASM32-unknown-unknown"), triple("wasm32-unknown-unknown"));
    }

    #[test]
    fn the_empty_string_is_rejected() {
        assert_eq!(rejection(""), "invalid target triple: it is empty");
    }

    #[test]
    fn an_empty_part_is_rejected_with_its_position() {
        assert_eq!(
            rejection("wasm32--unknown"),
            "invalid target triple 'wasm32--unknown': part 2 is empty"
        );
        assert_eq!(
            rejection("-unknown-unknown"),
            "invalid target triple '-unknown-unknown': part 1 is empty"
        );
        assert_eq!(
            rejection("wasm32-unknown-"),
            "invalid target triple 'wasm32-unknown-': part 3 is empty"
        );
    }

    #[test]
    fn a_character_outside_the_alphabet_is_rejected_and_named() {
        let message = rejection("wasm32-unknown/unknown");
        assert!(message.starts_with("invalid target triple 'wasm32-unknown/unknown': '/' cannot"));
        assert!(rejection("wasm32 -unknown").contains("' ' cannot"));
        assert!(rejection("wäsm32-unknown").contains("'ä' cannot"));
    }

    #[test]
    fn too_few_and_too_many_parts_are_rejected_with_the_count() {
        assert_eq!(
            rejection("wasm32"),
            "invalid target triple 'wasm32': a triple has 2 to 4 parts separated by '-', but this \
             has 1"
        );
        assert!(rejection("a-b-c-d-e").ends_with("but this has 5"));
    }
}
