pub trait FromStrRadix: Sized {
    type Error: core::error::Error;

    fn try_from_str_radix(source: &str, radix: u32) -> Result<Self, Self::Error>;
}

macro_rules! from_str_radix_impl {
    ($($t:ty),*) => {
        $(
            impl FromStrRadix for $t {
                type Error = core::num::ParseIntError;

                #[inline]
                fn try_from_str_radix(source: &str, radix: u32) -> Result<Self, Self::Error> {
                    <$t>::from_str_radix(source, radix)
                }
            }
        )*
    }
}

from_str_radix_impl!(u8, i8, u16, i16, u32, i32, u64, i64, u128, i128, usize, isize);
