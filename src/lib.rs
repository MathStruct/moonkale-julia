//! # moonkale-ext-lux
//!
//! Moonkale's Julia block libraries for the flow editor (their own repository
//! since Milestone 18 phase 6.4 — this crate is the `julia` feature):
//!
//! - **Lux.jl** (Milestone 6): layers (`Dense`, `Conv`, `MaxPool`, `Flatten`,
//!   `Dropout`, `BatchNorm`), activations, an `Input`, a `Loss` and an
//!   `Optimiser`, with tensor-shape ports so a shape mismatch is a red wire,
//!   and **codegen** to `model.jl` (`Lux.Chain(...)` plus a training scaffold).
//! - **ModelingToolkit** (ADR-0020's first consumer): acausal components —
//!   electrical (Resistor, Capacitor, Inductor, Voltage, Ground) and
//!   mechanical (Mass, Spring, Damper, Fixed) — with **domain-typed pins**
//!   (`electrical`, `mechanical`), grouped into nets for one `connect` per
//!   net, and **codegen** to `circuit.jl` (`@named` components, an
//!   `ODESystem`, `structural_simplify`, an `ODEProblem` and `solve`).
//!
//! - **RxInfer** (Moonkale ADR-0022): a bipartite factor graph — variables
//!   (circles; data makes one observed) and distribution factors (Beta,
//!   Bernoulli, Normal, Gamma, Poisson) whose channels hold one variable
//!   each — and **codegen** to `inference.jl` (an `@model`, the observed
//!   variables as its data, `infer`, every latent posterior printed).
//!
//! All three extensions are **opt-in**: they contribute nothing until enabled in
//! Settings → Extensions (Daniel's rule for niche/heavy features). Neither
//! has a Julia dependency of its own; the generated files need their
//! packages (`Lux`/`Optimisers`/`Zygote`/`Random`; `ModelingToolkit`, its
//! StandardLibrary and `OrdinaryDiffEq`; `RxInfer`) in the user's Julia environment.
//!
//! Vault: `editors/Flow Editor.md`.

mod codegen;
mod library;
mod mtk;
mod mtk_codegen;
mod rx_codegen;
mod rxinfer;

use dioxus::prelude::*;
use moonkale_ext_api::prelude::*;

pub use library::library;
pub use mtk::library as mtk_library;
pub use rxinfer::library as rxinfer_library;

pub struct LuxExtension;

impl Extension for LuxExtension {
    fn manifest(&self) -> Manifest {
        Manifest::opt_in(
            "dev.moonkale.ext-lux",
            "Lux.jl library",
            "Blocks for building Lux.jl models in the flow editor and generating a Julia training script.",
        )
    }

    fn panels(&self, _ws: Workspace) -> Vec<PanelContribution> {
        Vec::new()
    }

    fn render(&self, panel_id: &str, _ws: Workspace) -> Element {
        rsx! { "unknown panel {panel_id}" }
    }

    fn flow_libraries(&self) -> Vec<moonkale_ext_api::flow::FlowLibrary> {
        vec![library()]
    }
}

/// The ModelingToolkit library: acausal, domain-typed components whose
/// wires are equations (ADR-0020), generating a runnable Julia ODE problem.
pub struct MtkExtension;

impl Extension for MtkExtension {
    fn manifest(&self) -> Manifest {
        Manifest::opt_in(
            "dev.moonkale.ext-mtk",
            "ModelingToolkit library",
            "Acausal ModelingToolkit components for the flow editor — pins grouped into connect nets, generating a Julia ODE problem.",
        )
    }

    fn panels(&self, _ws: Workspace) -> Vec<PanelContribution> {
        Vec::new()
    }

    fn render(&self, panel_id: &str, _ws: Workspace) -> Element {
        rsx! { "unknown panel {panel_id}" }
    }

    fn flow_libraries(&self) -> Vec<moonkale_ext_api::flow::FlowLibrary> {
        vec![mtk::library()]
    }
}

/// The RxInfer library: a bipartite factor graph of variables and
/// distribution factors (ADR-0022), generating an `@model` and its
/// inference run.
pub struct RxInferExtension;

impl Extension for RxInferExtension {
    fn manifest(&self) -> Manifest {
        Manifest::opt_in(
            "dev.moonkale.ext-rxinfer",
            "RxInfer library",
            "Factor graphs for RxInfer in the flow editor — variables and distribution factors, generating an @model and its inference run.",
        )
    }

    fn panels(&self, _ws: Workspace) -> Vec<PanelContribution> {
        Vec::new()
    }

    fn render(&self, panel_id: &str, _ws: Workspace) -> Element {
        rsx! { "unknown panel {panel_id}" }
    }

    fn flow_libraries(&self) -> Vec<moonkale_ext_api::flow::FlowLibrary> {
        vec![rxinfer::library()]
    }
}
