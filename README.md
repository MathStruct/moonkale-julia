# moonkale-julia

Julia for [Moonkale](https://github.com/MathStruct/Moonkale) — today one crate, **`moonkale-ext-lux`**, with three block libraries for Moonkale's flow editor:

- **Lux.jl** ([lux.csail.mit.edu](https://lux.csail.mit.edu)): layers (`Dense`, `Conv`, `MaxPool`, `Flatten`, `Dropout`, `BatchNorm`), activations, an input, a loss and an optimiser, with tensor-shape ports (a shape mismatch is a red wire), and code generation to a Julia file (`Lux.Chain(…)` plus a training scaffold) that runs with `julia model.jl`.
- **ModelingToolkit** (since v0.2.0): acausal, domain-typed components — Resistor, Capacitor, Inductor, Voltage, Ground; Mass, Spring, Damper, Fixed — and code generation to `circuit.jl`: `@named` components, one `connect` per net, `structural_simplify` and a solved ODE problem.

- **RxInfer** (since v0.4.0, Moonkale ADR-0022): a bipartite factor graph — variables (data makes one observed) and Beta, Bernoulli, Normal, Gamma and Poisson factors, each channel holding one variable — and code generation to `inference.jl`: an `@model` whose arguments are the observed variables, and an `infer` run that prints every latent posterior (exact message passing; mean-field when a factor has two latent inputs).
- **Symbols** (since v0.5.0, Moonkale ADR-0024): every kind draws as itself in the flow editor. ModelingToolkit parts are schematics (a resistor box, a capacitor's plates, ground's bars, a spring, a dashpot) with each pin at the end of its lead. RxInfer factors are framed plots of their densities. Since v0.6.0 every part also carries a caption: its value and unit (`1.0 Ω`), or a factor as written (`θ ~ Beta(2, 7)`).

The Lux and ModelingToolkit generators understand **subflows** (since v0.3.0, Moonkale ADR-0021): an MTK subsystem flattens into the parent circuit, its components named per instance (`rc_filter_1_resistor_1`) and its boundary pins merging the inner and outer nets; a Lux subflow with one input and one output splices its layer path into the parent `Chain`.

It lived in the Moonkale repository as `packages/extensions/lux` until Milestone 18 (phase 6.4): it is what its author needs, and for everyone else an opt-in extension, so it is off by default and built only when asked for.

## Using it

Moonkale pulls it behind an off-by-default feature of its distribution:

```sh
# in the Moonkale repository
cd packages/desktop && dx serve --features julia      # or: cargo build -p desktop --features julia
cd packages/web && dx serve --features julia
```

then switch it on in **Settings → Extensions → Lux.jl library**; the flow editor's palette gets the Lux blocks, and **Generate** writes the Julia file. The generated Lux code needs `Lux`, `Optimisers`, `Zygote` and `Random` in your Julia environment, the circuit `ModelingToolkit`, `ModelingToolkitStandardLibrary` and `OrdinaryDiffEq`, the factor graph `RxInfer`; the extension itself has no Julia dependency.

## Developing it

It depends on Moonkale's extension contract by **git tag** (`ext-api-YYYY-MM-DD`, today `ext-api-2026-10-14-2`; see Moonkale's `packages/ext-api/CHANGELOG.md`), so `cargo test` here builds against that tag. To work on both at once, point Moonkale's distribution at a local checkout (`[patch."https://github.com/MathStruct/moonkale-julia"]` in Moonkale's `Cargo.toml`).

Notes on the code: `lux.md`. License: MIT.
