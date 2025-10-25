build:
	cargo build -r

check:
	cargo check -r --all

test-all:
	cargo test -r --workspace --all-features

test:
	cargo test -r --workspace

accept:
	cargo insta accept --all

fmt:
	cargo fmt --all

clippy-all:
	cargo clippy --workspace --all-features --benches --examples --tests -- -D warnings

clippy:
	cargo clippy --workspace -- -D warnings

fix:
	cargo clippy --workspace --all-features --benches --examples --tests --fix --allow-dirty

# SDK helpers
.PHONY: sdk-node-build sdk-python-develop sdk-python-build sdks-build-all

sdk-node-build:
	cd sdks/node && npm install && npm run build

sdk-python-develop:
	cd sdks/python && python -m venv .venv && . .venv/Scripts/Activate.ps1 && pip install -U pip maturin && maturin develop

sdk-python-build:
	cd sdks/python && python -m venv .venv && . .venv/Scripts/Activate.ps1 && pip install -U pip maturin && maturin build

sdks-build-all: sdk-node-build sdk-python-build
