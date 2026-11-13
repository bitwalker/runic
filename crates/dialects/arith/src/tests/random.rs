//! A deterministic generator of random operands, widths and functions for the randomized tests.
//!
//! The tests run three fixed seeds, or the one in the environment variable `ARITH_TEST_SEED`, with
//! [DEFAULT_CASES] cases per operation variant and seed, or `ARITH_TEST_CASES`.

use alloc::vec::Vec;

use super::model::M;

/// The number of random cases per operation variant and seed when `ARITH_TEST_CASES` is not set.
const DEFAULT_CASES: usize = 400;

/// A splitmix64 generator: the tests are deterministic for a seed.
pub(super) struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    pub fn bits(&mut self) -> u128 {
        (self.next() as u128) << 64 | self.next() as u128
    }

    /// A value of `w` bits, weighted toward the edges: 0, 1, -1, MIN, MAX, their neighbours, the
    /// width and its neighbours and multiples, single bits.
    pub fn edge(&mut self, w: u32) -> M {
        let width = w as u128;
        match self.below(18) {
            0 => M::zero(w),
            1 => M::new(w, 1),
            2 => M::ones(w),
            3 => M::smin(w),
            4 => M::smax(w),
            5 => M::new(w, width),
            6 => M::new(w, width.wrapping_sub(1)),
            7 => M::new(w, width + 1),
            8 => M::new(w, 2),
            9 => M::new(w, M::smin(w).bits.wrapping_add(1)),
            10 => M::new(w, M::ones(w).bits.wrapping_sub(1)),
            11 => M::new(w, 1u128 << self.below(w as u64)),
            12 => M::new(w, self.below(2 * w as u64 + 3) as u128),
            13 => M::new(w, (width * self.below(5) as u128).wrapping_add(self.below(3) as u128)),
            14 => M::new(w, M::smax(w).bits.wrapping_sub(1)),
            _ => M::new(w, self.bits()),
        }
    }

    /// A 64-bit `isize` constant, weighted toward the values on which 32- and 64-bit targets
    /// part: an edge of 32 bits in the low half under a high half that is zero, all ones, one, the
    /// sign extension of the low half, or random.
    pub fn isize_edge(&mut self) -> M {
        if self.below(4) == 0 {
            return self.edge(64);
        }
        let low = self.edge(32);
        let high: u128 = match self.below(5) {
            0 => 0,
            1 => 0xffff_ffff,
            2 => 1,
            3 => {
                if low.sign() {
                    0xffff_ffff
                } else {
                    0
                }
            }
            _ => self.next() as u32 as u128,
        };
        M::new(64, high << 32 | low.bits)
    }

    /// A fixed width in `1..=128`, weighted toward the widths around the limb sizes.
    pub fn width(&mut self) -> u32 {
        const EDGES: [u32; 15] = [1, 2, 3, 7, 8, 16, 31, 32, 33, 63, 64, 65, 96, 127, 128];
        if self.below(2) == 0 {
            EDGES[self.below(EDGES.len() as u64) as usize]
        } else {
            1 + self.below(128) as u32
        }
    }
}

/// The seeds a randomized test runs: `ARITH_TEST_SEED` if set, else three fixed ones.
pub(super) fn seeds() -> Vec<u64> {
    match std::env::var("ARITH_TEST_SEED") {
        Ok(seed) => alloc::vec![seed.parse().expect("ARITH_TEST_SEED is a u64")],
        Err(_) => alloc::vec![0x5eed_0001, 0x5eed_0002, 0x5eed_0003],
    }
}

/// How many random cases each operation variant gets per seed: `ARITH_TEST_CASES`, else
/// [DEFAULT_CASES].
pub(super) fn cases_per_variant() -> usize {
    std::env::var("ARITH_TEST_CASES")
        .ok()
        .map(|n| n.parse().expect("ARITH_TEST_CASES is a number"))
        .unwrap_or(DEFAULT_CASES)
}
