.PHONY: build install lint fmt test check clean help

help:
	@echo "Makefile commands:"
	@echo "  build    - Build the release binary in target/release/osslag"
	@echo "  install  - Install osslag into ~/.cargo/bin"
	@echo "  lint     - Lint the code with clippy"
	@echo "  fmt      - Format the code with rustfmt"
	@echo "  test     - Run the tests"
	@echo "  check    - Run every check CI and the pre-push hook run"
	@echo "  clean    - Remove build output"

build:
	cargo build --release --locked

install:
	cargo install --path . --locked

lint:
	cargo clippy --all-targets --locked -- -D warnings

fmt:
	cargo fmt

test:
	cargo test --locked

check:
	cargo fmt --check
	cargo clippy --all-targets --locked -- -D warnings
	cargo test --locked

clean:
	cargo clean
