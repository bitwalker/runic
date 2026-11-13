use core::{
    fmt,
    hash::{Hash, Hasher},
};

use smallvec::SmallVec;

use crate::dialects::builtin::types::{FloatType, IntegerType};

/// The order of the bytes of a value wider than a byte in memory.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Endianness {
    /// The least significant byte is at the lowest address.
    Little,
    /// The most significant byte is at the lowest address.
    Big,
}

impl Endianness {
    /// The keyword of this endianness in the text of a specification: `little` or `big`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Little => "little",
            Self::Big => "big",
        }
    }
}

impl fmt::Display for Endianness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A power of two number of bytes, 1 ..= 32768.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Align(/* log2 */ u8);

impl Align {
    /// The largest alignment, in bytes.
    pub const MAX_BYTES: u64 = 32768;
    /// One byte: no alignment requirement.
    pub const ONE: Align = Align(0);

    /// The alignment of `bytes` bytes, or `None` unless `bytes` is a power of two from 1 to
    /// [Self::MAX_BYTES].
    pub fn from_bytes(bytes: u64) -> Option<Align> {
        (bytes.is_power_of_two() && bytes <= Self::MAX_BYTES)
            .then(|| Align(bytes.trailing_zeros() as u8))
    }

    /// This alignment in bytes.
    pub fn bytes(self) -> u64 {
        1 << self.0
    }

    /// The smallest multiple of this alignment that is at least `offset`, or `None` if it does
    /// not fit a `u64`.
    pub fn align_up(self, offset: u64) -> Option<u64> {
        let mask = self.bytes() - 1;
        offset.checked_add(mask).map(|end| end & !mask)
    }
}

impl fmt::Display for Align {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.bytes())
    }
}

/// The representation of a pointer in one address space.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PointerSpec {
    /// The size of a pointer, in bits: a multiple of 8 from 8 to 64.
    pub size_bits: u32,
    /// The alignment of a pointer.
    pub align: Align,
    /// The width of an index into memory of this address space, in bits: 1 to `size_bits`.
    /// The index width of address space 0 is the width of `isize`.
    pub index_bits: u32,
}

impl PointerSpec {
    /// A pointer of `size_bits` bits, aligned to its size in bytes, with an index as wide as
    /// itself: what `ptr = N` states. `None` if `size_bits / 8` is not an alignment (`size_bits`
    /// is not 8 times a power of two); the size itself is checked when the pointer enters a
    /// specification.
    pub fn sized(size_bits: u32) -> Option<PointerSpec> {
        if !size_bits.is_multiple_of(8) {
            return None;
        }
        let align = Align::from_bytes(u64::from(size_bits / 8))?;
        Some(PointerSpec {
            size_bits,
            align,
            index_bits: size_bits,
        })
    }

    /// True if a pointer may be `size_bits` bits: a multiple of 8 from 8 to 64.
    pub(crate) fn is_valid_size(size_bits: u32) -> bool {
        size_bits.is_multiple_of(8) && (8..=64).contains(&size_bits)
    }

    /// The alignment `ptr = N` implies for a pointer of this size, if it implies one.
    pub(crate) fn shorthand_align(&self) -> Option<Align> {
        PointerSpec::sized(self.size_bits).map(|sized| sized.align)
    }
}

/// Why a [DataLayoutSpec] cannot be built as asked.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidDataLayout {
    /// The pointer of `addrspace` is `bits` bits, which is not a multiple of 8 from 8 to 64.
    PointerSize { addrspace: u32, bits: u32 },
    /// The index of `addrspace` is `bits` bits, which is 0 or wider than its pointer of `size`
    /// bits.
    IndexWidth {
        addrspace: u32,
        bits: u32,
        size: u32,
    },
    /// An integer alignment is given for `width` bits, which is not the width of an integer type.
    IntegerWidth { width: u32 },
}

impl fmt::Display for InvalidDataLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::PointerSize { addrspace, bits } => write!(
                f,
                "a pointer of address space {addrspace} is {bits} bits: expected a multiple of 8 \
                 from 8 to 64"
            ),
            Self::IndexWidth {
                addrspace,
                bits,
                size,
            } => write!(
                f,
                "an index of address space {addrspace} is {bits} bits: expected 1 to the pointer \
                 size, {size}"
            ),
            Self::IntegerWidth { width } => write!(
                f,
                "an integer alignment is given for {width} bits: expected a width from 1 to {}",
                IntegerType::MAX_WIDTH
            ),
        }
    }
}

impl core::error::Error for InvalidDataLayout {}

/// The integer alignments of every specification unless it states otherwise, by width.
const DEFAULT_INTEGER_ALIGNS: [(u32, Align); 5] =
    [(8, Align(0)), (16, Align(1)), (32, Align(2)), (64, Align(3)), (128, Align(4))];
/// The alignment of `f32` unless a specification states otherwise.
const DEFAULT_F32_ALIGN: Align = Align(2);
/// The alignment of `f64` unless a specification states otherwise.
const DEFAULT_F64_ALIGN: Align = Align(3);

type PointerTable = SmallVec<[(u32, PointerSpec); 2]>;
type IntegerTable = SmallVec<[(u32, Align); 8]>;

/// How values of the builtin types are laid out in byte-addressed memory: the endianness, the
/// pointer of each address space, and the alignments of integers and floats.
///
/// A specification is a plain value; an operation states one with a
/// [DataLayoutAttr](crate::dialects::builtin::attributes::DataLayoutAttr). It holds its answers
/// with the defaults filled in:
///
/// * an address space without an entry has the pointer of address space 0;
/// * an integer of a width without an entry has the alignment of the smallest listed width at
///   least as wide, else of the widest listed; the listed widths start as
///   `i8 = 1, i16 = 2, i32 = 4, i64 = 8, i128 = 16` (bytes);
/// * `f32` and `f64` are aligned to 4 and 8 bytes.
///
/// Two specifications are equal, and hash alike, when they answer every query alike, whatever
/// entries were given to build them: an entry equal to what the specification answers without it
/// makes no difference.
#[derive(Clone, Debug)]
pub struct DataLayoutSpec {
    endianness: Endianness,
    /// Sorted by address space; the first entry is address space 0.
    pointers: PointerTable,
    /// Sorted by width, one entry per width, never empty.
    integers: IntegerTable,
    f32_align: Align,
    f64_align: Align,
}

impl DataLayoutSpec {
    /// A specification of `endianness` whose address space 0 has `pointer`, with the default
    /// alignments.
    pub fn new(endianness: Endianness, pointer: PointerSpec) -> Result<Self, InvalidDataLayout> {
        validate_pointer(0, pointer)?;
        let mut pointers = PointerTable::new();
        pointers.push((0, pointer));
        Ok(Self {
            endianness,
            pointers,
            integers: IntegerTable::from_slice(&DEFAULT_INTEGER_ALIGNS),
            f32_align: DEFAULT_F32_ALIGN,
            f64_align: DEFAULT_F64_ALIGN,
        })
    }

    /// This specification with `pointer` in `addrspace`, replacing its entry if it has one.
    ///
    /// Setting address space 0 also changes every address space without an entry of its own.
    pub fn with_pointer(
        mut self,
        addrspace: u32,
        pointer: PointerSpec,
    ) -> Result<Self, InvalidDataLayout> {
        validate_pointer(addrspace, pointer)?;
        match self.pointers.binary_search_by_key(&addrspace, |(space, _)| *space) {
            Ok(index) => self.pointers[index].1 = pointer,
            Err(index) => self.pointers.insert(index, (addrspace, pointer)),
        }
        Ok(self)
    }

    /// This specification with integers of `width` bits aligned to `align`, replacing the entry
    /// for `width` if it has one.
    pub fn with_integer_align(
        mut self,
        width: u32,
        align: Align,
    ) -> Result<Self, InvalidDataLayout> {
        if !(1..=IntegerType::MAX_WIDTH).contains(&width) {
            return Err(InvalidDataLayout::IntegerWidth { width });
        }
        match self.integers.binary_search_by_key(&width, |(w, _)| *w) {
            Ok(index) => self.integers[index].1 = align,
            Err(index) => self.integers.insert(index, (width, align)),
        }
        Ok(self)
    }

    /// This specification with `float` aligned to `align`.
    pub fn with_float_align(mut self, float: FloatType, align: Align) -> Self {
        match float {
            FloatType::F32 => self.f32_align = align,
            FloatType::F64 => self.f64_align = align,
        }
        self
    }

    pub fn endianness(&self) -> Endianness {
        self.endianness
    }

    /// The pointer of `addrspace`; that of address space 0 if it has no entry of its own.
    pub fn pointer(&self, addrspace: u32) -> PointerSpec {
        match self.pointers.binary_search_by_key(&addrspace, |(space, _)| *space) {
            Ok(index) => self.pointers[index].1,
            Err(_) => self.pointers[0].1,
        }
    }

    /// The alignment of an integer of `width` bits: that of the smallest listed width at least
    /// `width`, else that of the widest listed.
    pub fn integer_align(&self, width: u32) -> Align {
        let index = self.integers.partition_point(|(w, _)| *w < width);
        match self.integers.get(index) {
            Some((_, align)) => *align,
            None => self.integers.last().expect("the integer table is never empty").1,
        }
    }

    pub fn float_align(&self, float: FloatType) -> Align {
        match float {
            FloatType::F32 => self.f32_align,
            FloatType::F64 => self.f64_align,
        }
    }

    /// The width of `isize`, in bits: the index width of address space 0.
    pub fn isize_width(&self) -> u32 {
        self.pointers[0].1.index_bits
    }

    /// The pointer entries that differ from address space 0, after address space 0 itself.
    pub(crate) fn distinct_pointers(&self) -> impl Iterator<Item = (u32, PointerSpec)> + '_ {
        let default = self.pointers[0].1;
        self.pointers
            .iter()
            .copied()
            .enumerate()
            .filter(move |(index, (_, pointer))| *index == 0 || *pointer != default)
            .map(|(_, entry)| entry)
    }

    /// The fewest integer entries that answer as this specification does.
    ///
    /// Of a run of neighbouring entries with one alignment, the widest answers for the run; the
    /// last run also answers for every wider width, so its narrowest entry is enough.
    fn canonical_integers(&self) -> IntegerTable {
        let mut canonical = IntegerTable::new();
        let mut runs = self.integers.chunk_by(|(_, a), (_, b)| a == b).peekable();
        while let Some(run) = runs.next() {
            let entry = if runs.peek().is_some() {
                run[run.len() - 1]
            } else {
                run[0]
            };
            canonical.push(entry);
        }
        canonical
    }

    /// The integer entries the text of this specification lists, by ascending width: this
    /// specification's alignment at each width of its canonical table and of the defaults,
    /// leaving out an entry equal to the default for its width.
    ///
    /// Parsing them over the defaults gives a specification equal to this one: each entry answers
    /// as this specification does at its width, and an entry agreeing with a table changes none
    /// of its answers. They depend only on the answers, so equal specifications list the same.
    pub(crate) fn printed_integers(&self) -> IntegerTable {
        let mut widths: SmallVec<[u32; 16]> = self
            .canonical_integers()
            .iter()
            .chain(DEFAULT_INTEGER_ALIGNS.iter())
            .map(|(width, _)| *width)
            .collect();
        widths.sort_unstable();
        widths.dedup();
        widths
            .into_iter()
            .map(|width| (width, self.integer_align(width)))
            .filter(|entry| !DEFAULT_INTEGER_ALIGNS.contains(entry))
            .collect()
    }

    /// The float entries the text of this specification lists: those that differ from the
    /// defaults, `f32` first.
    pub(crate) fn printed_floats(&self) -> impl Iterator<Item = (FloatType, Align)> {
        [
            (FloatType::F32, self.f32_align, DEFAULT_F32_ALIGN),
            (FloatType::F64, self.f64_align, DEFAULT_F64_ALIGN),
        ]
        .into_iter()
        .filter(|(_, align, default)| align != default)
        .map(|(float, align, _)| (float, align))
    }
}

/// Checks the size and index width of the pointer of `addrspace`.
fn validate_pointer(addrspace: u32, pointer: PointerSpec) -> Result<(), InvalidDataLayout> {
    let PointerSpec {
        size_bits,
        index_bits,
        ..
    } = pointer;
    if !PointerSpec::is_valid_size(size_bits) {
        return Err(InvalidDataLayout::PointerSize {
            addrspace,
            bits: size_bits,
        });
    }
    if !(1..=size_bits).contains(&index_bits) {
        return Err(InvalidDataLayout::IndexWidth {
            addrspace,
            bits: index_bits,
            size: size_bits,
        });
    }
    Ok(())
}

impl PartialEq for DataLayoutSpec {
    fn eq(&self, other: &Self) -> bool {
        self.endianness == other.endianness
            && self.f32_align == other.f32_align
            && self.f64_align == other.f64_align
            && self.distinct_pointers().eq(other.distinct_pointers())
            && self.canonical_integers() == other.canonical_integers()
    }
}

impl Eq for DataLayoutSpec {}

impl Hash for DataLayoutSpec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.endianness.hash(state);
        self.f32_align.hash(state);
        self.f64_align.hash(state);
        for entry in self.distinct_pointers() {
            entry.hash(state);
        }
        self.canonical_integers().hash(state);
    }
}
