//! The layout of a frame of `wasm-lower-frames`.

use alloc::vec;

use runic_ir::Align;

use crate::frames::Frame;

/// The alignment of `bytes` bytes.
fn align(bytes: u64) -> Align {
    Align::from_bytes(bytes).expect("a power of two")
}

#[test]
fn a_frame_of_no_slot_is_empty_and_16_byte_aligned() {
    assert_eq!(
        Frame::of(&[]),
        Some(Frame {
            offsets: vec![],
            size: 0,
            align: 16
        })
    );
}

#[test]
fn one_slot_is_at_offset_0_and_the_size_is_rounded_up_to_16() {
    let frame = Frame::of(&[(4, align(4))]).unwrap();
    assert_eq!(frame.offsets, [0]);
    assert_eq!(frame.size, 16);
    assert_eq!(frame.align, 16);
}

#[test]
fn each_slot_is_at_the_first_offset_its_alignment_divides_after_the_one_before() {
    // An `i8`, an `i64`, an `i16` aligned to 4, an `i32`, and 3 bytes.
    let frame =
        Frame::of(&[(1, align(1)), (8, align(8)), (2, align(4)), (4, align(4)), (3, align(1))])
            .unwrap();
    assert_eq!(frame.offsets, [0, 8, 16, 20, 24]);
    assert_eq!(frame.size, 32);
    assert_eq!(frame.align, 16);
}

#[test]
fn a_frame_that_fills_16_bytes_exactly_is_not_rounded_further() {
    let frame = Frame::of(&[(8, align(8)), (8, align(8))]).unwrap();
    assert_eq!(frame.offsets, [0, 8]);
    assert_eq!(frame.size, 16);
}

#[test]
fn an_over_aligned_slot_aligns_its_offset_and_the_frame() {
    let frame = Frame::of(&[(1, align(1)), (40, align(64)), (4, align(4))]).unwrap();
    assert_eq!(frame.offsets, [0, 64, 104]);
    assert_eq!(frame.size, 112);
    assert_eq!(frame.align, 64);
}

#[test]
fn a_slot_of_size_0_takes_no_room() {
    let frame = Frame::of(&[(0, align(8)), (4, align(4))]).unwrap();
    assert_eq!(frame.offsets, [0, 0]);
    assert_eq!(frame.size, 16);
}

#[test]
fn a_frame_larger_than_32_bits_hold_is_refused() {
    assert_eq!(Frame::of(&[(u64::from(u32::MAX), align(1))]), None);
    assert!(Frame::of(&[(u64::from(u32::MAX) - 15, align(1))]).is_some());
    assert_eq!(Frame::of(&[(u64::MAX, align(1)), (1, align(1))]), None);
}
