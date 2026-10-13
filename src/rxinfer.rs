//! The RxInfer block kinds (Moonkale ADR-0022 stage B): a **factor graph**,
//! bipartite and undirected. A variable is a small circle with one pin of
//! any degree. A factor is a distribution whose channels are pins: `out`
//! (the variable it generates) plus its parameters. A wire always joins a
//! variable and a factor, and a factor's channel holds one variable; the
//! editor enforces both through `BlockKind.role`.
//!
//! A parameter channel left unwired takes the block's parameter of the
//! same name as a constant (`Beta(2.0, 7.0)`). A variable is *observed*
//! when it carries data (comma-separated numbers; more than one is an
//! i.i.d. data vector); the generated model takes the observed variables
//! as its data arguments.

use moonkale_ext_api::flow::{BlockKind, FlowLibrary, Param, ParamKind, Port, PortType, Role};

/// A channel or a variable's pin. `out` must be wired; a parameter channel
/// falls back to its constant.
fn pin(name: &str, required: bool) -> Port {
    Port {
        name: name.into(),
        ty: PortType::Any,
        required,
    }
}

fn float(name: &str, default: &str) -> Param {
    Param {
        name: name.into(),
        kind: ParamKind::Float,
        default: default.into(),
    }
}

/// A distribution factor: `out` plus one channel per parameter, each
/// channel's constant a parameter of the same name.
fn factor(id: &str, name: &str, description: &str, params: &[(&str, &str)]) -> BlockKind {
    let mut connections = vec![pin("out", true)];
    connections.extend(params.iter().map(|(n, _)| pin(n, false)));
    BlockKind {
        id: id.into(),
        name: name.into(),
        category: "factors".into(),
        description: description.into(),
        connections,
        params: params.iter().map(|(n, d)| float(n, d)).collect(),
        role: Some(Role::Factor),
        ..Default::default()
    }
}

pub fn library() -> FlowLibrary {
    FlowLibrary {
        id: "rxinfer",
        name: "RxInfer",
        language: "julia",
        codegen: Some(crate::rx_codegen::generate),
        blocks: vec![
            BlockKind {
                id: "variable".into(),
                name: "Variable".into(),
                category: "variables".into(),
                description: "A random variable: name it; give it data to observe it (one number, or several for an i.i.d. vector).".into(),
                connections: vec![pin("x", true)],
                params: vec![
                    Param {
                        name: "name".into(),
                        kind: ParamKind::Text,
                        default: "x".into(),
                    },
                    Param {
                        name: "data".into(),
                        kind: ParamKind::Text,
                        default: String::new(),
                    },
                ],
                role: Some(Role::Variable),
                ..Default::default()
            },
            factor(
                "beta",
                "Beta",
                "Beta(a, b): a prior on a probability.",
                &[("a", "1.0"), ("b", "1.0")],
            ),
            factor(
                "bernoulli",
                "Bernoulli",
                "Bernoulli(p): a coin flip, 1 with probability p.",
                &[("p", "0.5")],
            ),
            factor(
                "normal",
                "Normal",
                "Normal(mean, variance).",
                &[("mean", "0.0"), ("variance", "1.0")],
            ),
            factor(
                "normal_precision",
                "Normal (precision)",
                "Normal(mean, precision): the form a Gamma precision prior conjugates with.",
                &[("mean", "0.0"), ("precision", "1.0")],
            ),
            factor(
                "gamma",
                "Gamma",
                "Gamma(shape, rate): a prior on a positive quantity (a rate, a precision).",
                &[("shape", "1.0"), ("rate", "1.0")],
            ),
            factor(
                "poisson",
                "Poisson",
                "Poisson(rate): a count.",
                &[("rate", "1.0")],
            ),
        ],
    }
}
