cargo test; 
cargo bench --bench accumulate; 
cargo bench --bench bit_rev;
cargo test --features icicle;
cargo bench --bench accumulate --features icicle;
cargo bench --bench bit_rev --features icicle;