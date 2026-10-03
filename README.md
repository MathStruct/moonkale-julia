# moonkale-julia

Julia for [Moonkale](https://github.com/MathStruct/Moonkale) — today one crate, **`moonkale-ext-lux`**: a [Lux.jl](https://lux.csail.mit.edu) block library for Moonkale's flow editor. Layers (`Dense`, `Conv`, `MaxPool`, `Flatten`, `Dropout`, `BatchNorm`), activations, an input, a loss and an optimiser, with tensor-shape ports (a shape mismatch is a red wire), and code generation to a Julia file (`Lux.Chain(…)` plus a training scaffold) that runs with `julia model.jl`.

It lived in the Moonkale repository as `packages/extensions/lux` until Milestone 18 (phase 6.4): it is what its author needs, and for everyone else an opt-in extension, so it is off by default and built only when asked for.

## Using it

Moonkale pulls it behind an off-by-default feature of its distribution:

```sh
# in the Moonkale repository
cd packages/desktop && dx serve --features julia      # or: cargo build -p desktop --features julia
cd packages/web && dx serve --features julia
```

then switch it on in **Settings → Extensions → Lux.jl library**; the flow editor's palette gets the Lux blocks, and **Generate** writes the Julia file. The generated code needs `Lux`, `Optimisers`, `Zygote` and `Random` in your Julia environment; the extension itself has no Julia dependency.

## Developing it

It depends on Moonkale's extension contract by **git tag** (`lib-v1`; see Moonkale's `packages/ext-api/CHANGELOG.md`), so `cargo test` here builds against that tag. To work on both at once, point Moonkale's distribution at a local checkout (`[patch."https://github.com/MathStruct/moonkale-julia"]` in Moonkale's `Cargo.toml`).

Notes on the code: `lux.md`. License: MIT.
