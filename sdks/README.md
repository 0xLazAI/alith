# Alith SDKs

This folder contains the language SDKs that wrap the core Rust crate(s) to provide first‑class developer experiences in Node.js/TypeScript and Python.

- `sdks/node/` – Node.js bindings built with napi‑rs, plus TypeScript typings and helper utilities.
- `sdks/python/` – Python bindings built with PyO3 + maturin, plus Pythonic wrappers and examples.
- `sdks/rust/` – Thin Rust client for direct use from Rust projects.

## How the bindings work

The core logic lives in the workspace Rust crates (see `crates/` and `sdks/*/Cargo.toml`). Each language SDK compiles a Rust `cdylib` and exposes idiomatic APIs.

### Node.js (napi‑rs)
- Rust crate: `sdks/node/Cargo.toml` (`crate-type = ["cdylib"]`).
- Binding tool: [`napi-rs`](https://napi.rs). We export Node‑API functions via `napi` and `napi-derive`.
- Build entry: `sdks/node/build.rs` and package `scripts` in `sdks/node/package.json`.
- TypeScript: sources under `sdks/node/src/` are compiled to `dist/` and published with the native `.node` binary.

Build locally:
```powershell
cd sdks/node
npm install
npm run build   # builds the Rust addon and TypeScript to dist/
```
Notes:
- On Windows, ensure MSVC Build Tools and Rust toolchain are installed (\"rustup update\").
- To build platform‑specific artifacts in CI we use `napi build --platform`.

### Python (PyO3 + maturin)
- Rust crate: `sdks/python/Cargo.toml` (`crate-type = ["cdylib"]`).
- Binding tool: [`PyO3`](https://pyo3.rs) with [`maturin`](https://www.maturin.rs/) as the build backend.
- Python package metadata is in `sdks/python/pyproject.toml`.
- The compiled extension module is exposed as `alith._alith`; pure‑Python code lives under `sdks/python/alith/`.

Build locally:
```powershell
cd sdks/python
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -U pip maturin
maturin develop  # builds and installs the local wheel into the venv
```
Run examples:
```powershell
python examples/agent_groq.py
```

## Publishing (high level)
- Node: `npm run prepublishOnly` prepares artifacts; CI builds per‑platform `.node` and runs `npm publish` from `sdks/node`.
- Python: `maturin build` (or `maturin publish`) builds wheels for the active platform; CI can produce manylinux/macos/windows wheels.

## Environment and models
Both SDKs delegate inference to remote providers. Set the appropriate env vars:
- `GROQ_API_KEY` (and optionally `GROQ_MODEL`), `BASE_URL` where applicable.
- Use a `.env` with `dotenv` (Node) or `python‑dotenv` (Python) during development. Do not commit `.env`.

## Troubleshooting
- Rust build errors: run `rustup update` and ensure the target toolchains (e.g., MSVC on Windows) are installed.
- Node build cannot find `.node`: run `npm run build` from `sdks/node`, ensure `NAPI_TARGET` matches your platform when cross‑compiling.
- Python `numpy/pandas` ABI errors: create a clean venv and reinstall pins compatible with your Python version.
- API model deprecations: if you see `model_decommissioned`, switch to a current model (e.g., `mixtral-8x7b-32768`) and retry.

## Repo layout quick reference
```
sdks/
  node/
    Cargo.toml       # napi‑rs addon
    build.rs
    package.json     # scripts build both Rust and TS
    src/             # TS sources
    dist/            # built JS/DTs
  python/
    Cargo.toml       # PyO3 crate
    pyproject.toml   # maturin config
    alith/           # Python package sources
    examples/
```

If you want, I can also wire a `make` target at the repo root to build all SDKs in one go.