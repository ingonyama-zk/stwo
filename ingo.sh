cargo test; 
cargo bench --bench accumulate; 
cargo bench --bench bit_rev;
cargo test --features icicle_poc;
cargo bench --bench accumulate --features icicle_poc;
cargo bench --bench bit_rev --features icicle_poc;