use super::FilterOp;

#[derive(Debug, Clone)]
pub struct KvFilter {
    pub key: String,
    pub patterns: Vec<KvFilterOp>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct KvFilterOp {
    pub value: FilterOp,
    pub negated: bool,
}

impl KvFilter {
    pub fn matches(&self, kv: &dyn log::kv::Source) -> Option<bool> {
        let value = kv.get(log::kv::Key::from_str(&self.key))?;
        let value = value.to_borrowed_str()?;

        let mut was_matched = false;
        for pattern in self.patterns.iter() {
            if pattern.value.is_match(value) {
                if pattern.negated {
                    return Some(false);
                }
                was_matched = true;
            }
        }
        Some(was_matched)
    }

    /// Adds `pattern` to the patterns of this key; an error when `pattern` is not a valid pattern
    /// (with the `regex` feature, a regular expression), described by the message.
    pub fn insert_value_filter(&mut self, pattern: &str, negated: bool) -> Result<(), String> {
        if self
            .patterns
            .iter()
            .any(|p| p.value.as_str() == pattern && p.negated == negated)
        {
            return Ok(());
        }
        self.patterns.push(KvFilterOp {
            value: FilterOp::new(pattern)?,
            negated,
        });
        Ok(())
    }
}
