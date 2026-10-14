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

use moonkale_ext_api::flow::{
    BlockKind, FlowLibrary, Param, ParamKind, Port, PortType, Role, Symbol,
};

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

/// A factor's drawing (Moonkale ADR-0024): a framed plot of its density
/// or mass function over a baseline, 44 × 32. The pins keep the editor's
/// default for factors — the generated variable `out` on the right, the
/// parameters in from the left.
fn plot(glyph: &str) -> Symbol {
    Symbol {
        width: 44.0,
        height: 32.0,
        path: format!("M 0 0 H 44 V 32 H 0 Z M 6 26 H 38 {glyph}"),
        anchors: vec![],
    }
}

/// A distribution factor: `out` plus one channel per parameter, each
/// channel's constant a parameter of the same name.
fn factor(
    id: &str,
    name: &str,
    description: &str,
    params: &[(&str, &str)],
    glyph: &str,
) -> BlockKind {
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
        symbol: Some(plot(glyph)),
        ..Default::default()
    }
}

/// The line under each factor (Moonkale ADR-0024 §6): the factor as it
/// is written, `θ ~ Beta(2.0, 7.0)` — a wired channel shows its
/// variable's name, an unwired one its constant.
fn caption(id: &str) -> Option<&'static str> {
    Some(match id {
        "beta" => "{out} ~ Beta({a}, {b})",
        "bernoulli" => "{out} ~ Bernoulli({p})",
        "normal" => "{out} ~ Normal({mean}, {variance})",
        "normal_precision" => "{out} ~ NormalMP({mean}, {precision})",
        "gamma" => "{out} ~ Gamma({shape}, {rate})",
        "poisson" => "{out} ~ Poisson({rate})",
        _ => return None,
    })
}

pub fn library() -> FlowLibrary {
    let mut lib = kinds();
    for k in lib.blocks.iter_mut() {
        k.caption = caption(&k.id).map(String::from);
    }
    lib
}

fn kinds() -> FlowLibrary {
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
                // A hump on [0, 1].
                "M 8 26 C 12 4, 24 4, 36 26",
            ),
            factor(
                "bernoulli",
                "Bernoulli",
                "Bernoulli(p): a coin flip, 1 with probability p.",
                &[("p", "0.5")],
                // Two bars: 0 and 1.
                "M 15 26 V 16 M 29 26 V 9",
            ),
            factor(
                "normal",
                "Normal",
                "Normal(mean, variance).",
                &[("mean", "0.0"), ("variance", "1.0")],
                // The bell.
                "M 6 26 C 15 26, 16 6, 22 6 C 28 6, 29 26, 38 26",
            ),
            factor(
                "normal_precision",
                "Normal (precision)",
                "Normal(mean, precision): the form a Gamma precision prior conjugates with.",
                &[("mean", "0.0"), ("precision", "1.0")],
                // The bell, narrow: a precision.
                "M 6 26 C 17 26, 18 5, 22 5 C 26 5, 27 26, 38 26",
            ),
            factor(
                "gamma",
                "Gamma",
                "Gamma(shape, rate): a prior on a positive quantity (a rate, a precision).",
                &[("shape", "1.0"), ("rate", "1.0")],
                // Skewed right, a long tail.
                "M 6 26 C 9 6, 15 6, 21 16 C 26 23, 31 25, 38 25",
            ),
            factor(
                "poisson",
                "Poisson",
                "Poisson(rate): a count.",
                &[("rate", "1.0")],
                // Counts: bars that rise and fall.
                "M 10 26 V 17 M 16 26 V 10 M 22 26 V 12 M 28 26 V 19 M 34 26 V 23",
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_factor_captions_itself_with_its_own_channels() {
        for k in library().blocks {
            if k.role != Some(Role::Factor) {
                assert!(k.caption.is_none(), "{}", k.id);
                continue;
            }
            let c = k
                .caption
                .as_ref()
                .unwrap_or_else(|| panic!("{} has no caption", k.id));
            assert!(c.starts_with("{out} ~ "), "{c}");
            for hole in c.split('{').skip(1) {
                let name = hole.split('}').next().unwrap();
                assert!(
                    k.connections.iter().any(|p| p.name == name),
                    "{}: {{{name}}}",
                    k.id
                );
            }
        }
    }
}
