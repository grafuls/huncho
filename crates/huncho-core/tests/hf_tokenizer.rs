//! Byte-identical tokenization (CORE-02) via the Hugging Face `tokenizers`
//! crate, behind the `tokenizers` feature.
//!
//! The fixture `tests/fixtures/minimal_tokenizer.json` was generated with the
//! reference `tokenizers` Python library; this test asserts the Rust crate
//! produces the exact same ids for the same inputs, so prompt building can be
//! byte-identical to the reference implementation.
//!
//! Run with:
//!   cargo test -p huncho-core --features tokenizers --test hf_tokenizer

#![cfg(feature = "tokenizers")]

use huncho_core::tokenizer::{HfTokenizer, Tokenizer};

#[test]
fn hf_tokenizer_matches_reference_ids() {
    let tk = HfTokenizer::from_file("tests/fixtures/minimal_tokenizer.json").unwrap();

    // Reference ids (from the Python `tokenizers` 0.23 build of this fixture):
    // BOS(1) hello(7) world(8) EOS(2).
    assert_eq!(tk.encode("hello world", true).unwrap(), vec![1, 7, 8, 2]);
    assert_eq!(tk.encode("hello world", false).unwrap(), vec![7, 8]);

    // The synthetic option markers are real vocabulary entries (CORE-02 F1).
    assert_eq!(tk.id_for("<option:0>"), Some(18));
    assert_eq!(tk.id_for("<option:1>"), Some(19));
    assert_eq!(tk.id_for("refund"), Some(9));
}

#[test]
fn hf_tokenizer_decodes_round_trip() {
    let tk = HfTokenizer::from_file("tests/fixtures/minimal_tokenizer.json").unwrap();
    let s = tk.decode(&[7, 8]).unwrap();
    assert_eq!(s.trim(), "hello world");
}

#[test]
fn hf_tokenizer_builds_deterministic_prompt() {
    // A realistic F1 prompt with option markers; token ids must be stable across
    // calls so the head positions are reproducible.
    let tk = HfTokenizer::from_file("tests/fixtures/minimal_tokenizer.json").unwrap();
    let mk = tk.id_for("<option:0>").unwrap();
    let text = "hello world <option:0> refund";
    let a = tk.encode(text, true).unwrap();
    let b = tk.encode(text, true).unwrap();
    assert_eq!(a, b);
    // Reference: BOS(1) hello(7) world(8) <option:0>(mk) refund(9) EOS(2).
    assert_eq!(a, vec![1, 7, 8, mk, 9, 2]);
    let pos = a.iter().position(|&t| t == mk).unwrap();
    assert!(pos > 0);
}
