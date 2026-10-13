//! RxInfer codegen (Moonkale ADR-0022 stage B): a factor graph becomes an
//! `@model` plus an `infer` call that prints every latent posterior.
//!
//! - Each factor is one `~` statement: `out ~ Kind(channels…)`, a wired
//!   channel naming its variable and an unwired one its constant.
//! - Observed variables (those with data) are the model's arguments and
//!   the `data` of the run. A data vector is i.i.d.: its factor's
//!   statement runs `for i in eachindex(y)` over `y[i]`.
//! - Statements follow the graph: a factor comes after the factors that
//!   generate its inputs. A loop through the factors is refused (loopy
//!   models need a schedule, which isn't generated yet).
//! - Inference is exact message passing. Only when a factor has two or
//!   more latent inputs (an unknown mean *and* precision) is the
//!   posterior not conjugate, and the run becomes mean-field variational:
//!   `MeanField()`, an initial marginal per latent variable, 30
//!   iterations, the last posterior printed.
//!
//! Anything this cannot generate is an error naming the block, never a
//! silently wrong file.

use moonkale_ext_api::flow::{Block, Flow, FlowLibrary, Generated};
use std::collections::{BTreeMap, BTreeSet};

fn param<'a>(b: &'a Block, name: &str, default: &'a str) -> &'a str {
    b.params.get(name).map(|s| s.as_str()).unwrap_or(default)
}

/// A Julia float literal: finite, and always with a decimal point
/// (`2` → `2.0`), so a constant never turns a distribution integer.
fn float_lit(v: &str) -> Option<String> {
    let v = v.trim();
    let f = v.parse::<f64>().ok().filter(|f| f.is_finite())?;
    Some(if v.contains(['.', 'e', 'E']) {
        v.to_string()
    } else {
        format!("{f:.1}")
    })
}

/// A Julia identifier a user can type: a letter or `_`, then letters,
/// digits or `_` (Unicode letters included — `θ`, `μ`).
fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    c.next().is_some_and(|f| f.is_alphabetic() || f == '_')
        && c.all(|ch| ch.is_alphanumeric() || ch == '_')
        && !matches!(
            s,
            "end"
                | "for"
                | "if"
                | "in"
                | "function"
                | "begin"
                | "let"
                | "data"
                | "model"
                | "result"
        )
}

/// One factor kind: its Julia call and parameter channels in order;
/// `keywords` writes `name = value` (RxInfer's `Normal`/`Gamma` need it).
struct Dist {
    call: &'static str,
    channels: &'static [&'static str],
    keywords: bool,
    /// The initial marginal when the run is mean-field (`None`: a variable
    /// this factor generates cannot be initialised — refuse).
    init: Option<&'static str>,
    /// The data must be integers (a count).
    integer: bool,
}

fn dist(kind: &str) -> Option<Dist> {
    Some(match kind {
        "beta" => Dist {
            call: "Beta",
            channels: &["a", "b"],
            keywords: false,
            init: Some("Beta(1.0, 1.0)"),
            integer: false,
        },
        "bernoulli" => Dist {
            call: "Bernoulli",
            channels: &["p"],
            keywords: false,
            init: Some("Bernoulli(0.5)"),
            integer: false,
        },
        "normal" => Dist {
            call: "Normal",
            channels: &["mean", "variance"],
            keywords: true,
            init: Some("vague(NormalMeanVariance)"),
            integer: false,
        },
        "normal_precision" => Dist {
            call: "Normal",
            channels: &["mean", "precision"],
            keywords: true,
            init: Some("vague(NormalMeanVariance)"),
            integer: false,
        },
        "gamma" => Dist {
            call: "Gamma",
            channels: &["shape", "rate"],
            keywords: true,
            init: Some("vague(GammaShapeRate)"),
            integer: false,
        },
        "poisson" => Dist {
            call: "Poisson",
            channels: &["rate"],
            keywords: false,
            init: None,
            integer: true,
        },
        _ => return None,
    })
}

/// A variable as the generator sees it.
struct Var {
    name: String,
    /// The data, already Julia literals; empty = latent.
    data: Vec<String>,
}

impl Var {
    fn observed(&self) -> bool {
        !self.data.is_empty()
    }
    fn vector(&self) -> bool {
        self.data.len() > 1
    }
}

pub fn generate(flow: &Flow, lib: &FlowLibrary) -> Result<Generated, String> {
    let prefix = format!("{}/", lib.id);
    let kind_of = |b: &Block| b.kind.strip_prefix(&prefix).map(str::to_string);
    if !flow.subflows.is_empty() || flow.blocks.iter().any(|b| b.subflow.is_some()) {
        return Err("a subsystem in a factor graph is not generated yet (ADR-0022 stage C)".into());
    }

    // The variables: named, unique, their data parsed.
    let mut vars: BTreeMap<&str, Var> = BTreeMap::new();
    let mut taken: BTreeSet<String> = BTreeSet::new();
    let mut factors: Vec<(&Block, Dist)> = Vec::new();
    for b in &flow.blocks {
        let Some(k) = kind_of(b) else {
            return Err(format!(
                "{} is not an {} block — one library per flow for now",
                b.kind, lib.name
            ));
        };
        if k == "variable" {
            let name = param(b, "name", "x").trim().to_string();
            if !is_ident(&name) {
                return Err(format!("{}: {name:?} is not a Julia variable name", b.id));
            }
            if !taken.insert(name.clone()) {
                return Err(format!("two variables are named {name}"));
            }
            let raw = param(b, "data", "");
            let data = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| float_lit(s).ok_or(format!("{name}: data value {s:?} is not a number")))
                .collect::<Result<Vec<_>, _>>()?;
            vars.insert(&b.id, Var { name, data });
        } else if let Some(d) = dist(&k) {
            factors.push((b, d));
        } else {
            return Err(format!("unknown kind {}", b.kind));
        }
    }

    // Each factor's channels → the variable wired to it (either drag
    // order: a pin wire's ends are just its two ends).
    let wired = |f: &Block, ch: &str| -> Result<Option<&Var>, String> {
        let mut found = flow.wires.iter().filter_map(|w| {
            if w.from_block == f.id && w.from_port == ch {
                Some(w.to_block.as_str())
            } else if w.to_block == f.id && w.to_port == ch {
                Some(w.from_block.as_str())
            } else {
                None
            }
        });
        let first = found.next();
        if found.next().is_some() {
            return Err(format!(
                "channel {ch} of {} carries two wires — a channel attaches to one variable",
                f.id
            ));
        }
        match first {
            None => Ok(None),
            Some(id) => vars.get(id).map(Some).ok_or(format!(
                "{}.{ch} is wired to {id}, which is not a variable — factors connect to variables",
                f.id
            )),
        }
    };

    // Statements, and the dependencies between them.
    struct Stmt<'a> {
        out: &'a Var,
        line: String,
        inputs: Vec<&'a Var>,
        id: &'a str,
        dist: &'a Dist,
    }
    let mut stmts: Vec<Stmt> = Vec::new();
    let mut generated_by: BTreeMap<&str, &str> = BTreeMap::new();
    let mut counts: BTreeSet<&str> = BTreeSet::new();
    for (f, d) in &factors {
        let out = wired(f, "out")?.ok_or(format!(
            "the out channel of {} is not connected — wire it to the variable it generates",
            f.id
        ))?;
        if let Some(other) = generated_by.insert(&out.name, &f.id) {
            return Err(format!(
                "{} is generated by both {other} and {} — a variable has one generating factor",
                out.name, f.id
            ));
        }
        let mut args = Vec::new();
        let mut inputs = Vec::new();
        for ch in d.channels {
            let value = match wired(f, ch)? {
                Some(v) => {
                    if v.vector() {
                        return Err(format!(
                            "{}: the data vector {} can only be a factor's out (an i.i.d. likelihood)",
                            f.id, v.name
                        ));
                    }
                    inputs.push(v);
                    v.name.clone()
                }
                None => {
                    // The block's constant, else its kind's default.
                    let default = kind_of(f)
                        .and_then(|k| lib.block(&k))
                        .and_then(|k| k.params.iter().find(|p| p.name == *ch))
                        .map(|p| p.default.as_str())
                        .unwrap_or("1.0");
                    let v = param(f, ch, default);
                    float_lit(v).ok_or(format!("{}.{ch} = {v:?} is not a number", f.id))?
                }
            };
            args.push(if d.keywords {
                format!("{ch} = {value}")
            } else {
                value
            });
        }
        if d.integer
            && out
                .data
                .iter()
                .any(|v| v.parse::<f64>().map_or(true, |x| x.fract() != 0.0))
        {
            return Err(format!(
                "{}: {} data must be whole numbers (counts)",
                f.id, out.name
            ));
        }
        if d.integer {
            counts.insert(&out.name);
        }
        let target = if out.vector() {
            format!("{}[i]", out.name)
        } else {
            out.name.clone()
        };
        stmts.push(Stmt {
            out,
            line: format!("{target} ~ {}({})", d.call, args.join(", ")),
            inputs,
            id: &f.id,
            dist: d,
        });
    }

    // Every latent variable needs a generating factor (its prior).
    for v in vars.values() {
        if !v.observed() && !generated_by.contains_key(v.name.as_str()) {
            return Err(format!(
                "{} has no prior — wire it to the out channel of a factor, or give it data",
                v.name
            ));
        }
    }
    let observed: Vec<&Var> = flow
        .blocks
        .iter()
        .filter_map(|b| vars.get(b.id.as_str()))
        .filter(|v| v.observed())
        .collect();
    if observed.is_empty() {
        return Err("nothing is observed — give a variable data".into());
    }

    // Order: a statement after those generating its inputs (Kahn, stable
    // in flow order).
    let mut done: BTreeSet<&str> = observed
        .iter()
        .filter(|v| !generated_by.contains_key(v.name.as_str()))
        .map(|v| v.name.as_str())
        .collect();
    let mut order: Vec<&Stmt> = Vec::new();
    let mut left: Vec<&Stmt> = stmts.iter().collect();
    while !left.is_empty() {
        let Some(i) = left
            .iter()
            .position(|s| s.inputs.iter().all(|v| done.contains(v.name.as_str())))
        else {
            return Err(format!(
                "the factor graph loops through {} — a loopy model needs a schedule, which is not generated yet",
                left[0].id
            ));
        };
        let s = left.remove(i);
        done.insert(&s.out.name);
        order.push(s);
    }

    // Mean-field when a factor has two or more latent inputs.
    let mean_field = order
        .iter()
        .any(|s| s.inputs.iter().filter(|v| !v.observed()).count() >= 2);
    let latent: Vec<&Stmt> = order
        .iter()
        .copied()
        .filter(|s| !s.out.observed())
        .collect();

    let mut text = String::new();
    text.push_str("# Generated by Moonkale's RxInfer library — a .flow.json's factor graph.\n");
    text.push_str("# Run: julia inference.jl (needs RxInfer).\n");
    text.push_str("using RxInfer\n\n");
    let args: Vec<&str> = observed.iter().map(|v| v.name.as_str()).collect();
    text.push_str(&format!(
        "@model function factor_graph({})\n",
        args.join(", ")
    ));
    for s in &order {
        if s.out.vector() {
            text.push_str(&format!("    for i in eachindex({})\n", s.out.name));
            text.push_str(&format!("        {}\n", s.line));
            text.push_str("    end\n");
        } else {
            text.push_str(&format!("    {}\n", s.line));
        }
    }
    text.push_str("end\n\n");
    if mean_field {
        text.push_str("init = @initialization begin\n");
        for s in &latent {
            let init = s.dist.init.ok_or(format!(
                "{}: {} cannot be initialised for a mean-field run",
                s.id, s.out.name
            ))?;
            text.push_str(&format!("    q({}) = {init}\n", s.out.name));
        }
        text.push_str("end\n\n");
    }
    text.push_str("result = infer(\n    model = factor_graph(),\n");
    let data: Vec<String> = observed
        .iter()
        .map(|v| {
            // Counts are integers in Julia (`3`, not `3.0`).
            let lits: Vec<String> = if counts.contains(v.name.as_str()) {
                v.data
                    .iter()
                    .map(|x| format!("{}", x.parse::<f64>().unwrap_or(0.0) as i64))
                    .collect()
            } else {
                v.data.clone()
            };
            let value = if v.vector() {
                format!("[{}]", lits.join(", "))
            } else {
                lits[0].clone()
            };
            format!("{} = {value}", v.name)
        })
        .collect();
    text.push_str(&format!("    data = ({},),\n", data.join(", ")));
    if mean_field {
        text.push_str(
            "    constraints = MeanField(),\n    initialization = init,\n    iterations = 30,\n",
        );
    }
    text.push_str(")\n");
    let last = if mean_field { "[end]" } else { "" };
    for s in &latent {
        let n = &s.out.name;
        text.push_str(&format!(
            "println(\"{n} = \", result.posteriors[:{n}]{last})\n"
        ));
        text.push_str(&format!(
            "println(\"mean({n}) = \", mean(result.posteriors[:{n}]{last}))\n"
        ));
    }

    Ok(Generated {
        file_name: "inference.jl".into(),
        text,
        run_hint: "julia inference.jl".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use moonkale_ext_api::flow::{validate, Wire};

    fn block(id: &str, kind: &str, params: &[(&str, &str)]) -> Block {
        Block {
            id: id.into(),
            kind: format!("rxinfer/{kind}"),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..Default::default()
        }
    }

    fn wire(id: &str, a: &str, ap: &str, b: &str, bp: &str) -> Wire {
        Wire {
            id: id.into(),
            from_block: a.into(),
            from_port: ap.into(),
            to_block: b.into(),
            to_port: bp.into(),
        }
    }

    /// θ ~ Beta(2, 7); ten coin flips y ~ Bernoulli(θ).
    pub(super) fn coin() -> Flow {
        let mut f = Flow::new();
        f.blocks = vec![
            block("prior", "beta", &[("a", "2"), ("b", "7.0")]),
            block("theta", "variable", &[("name", "θ")]),
            block("flip", "bernoulli", &[]),
            block(
                "y",
                "variable",
                &[("name", "y"), ("data", "1, 1, 0, 1, 1, 1, 0, 1, 0, 1")],
            ),
        ];
        f.wires = vec![
            wire("w1", "prior", "out", "theta", "x"),
            // Drawn from the variable to the channel: either order.
            wire("w2", "theta", "x", "flip", "p"),
            wire("w3", "flip", "out", "y", "x"),
        ];
        f
    }

    #[test]
    fn a_coin_generates_a_model_its_data_and_the_posterior() {
        let lib = crate::rxinfer::library();
        let f = coin();
        assert_eq!(validate(&f, std::slice::from_ref(&lib)), vec![]);
        let out = generate(&f, &lib).expect("generates");
        assert_eq!(out.file_name, "inference.jl");
        assert!(
            out.text.contains("@model function factor_graph(y)\n"),
            "{}",
            out.text
        );
        assert!(
            out.text.contains("    θ ~ Beta(2.0, 7.0)\n"),
            "{}",
            out.text
        );
        assert!(
            out.text
                .contains("    for i in eachindex(y)\n        y[i] ~ Bernoulli(θ)\n    end\n"),
            "{}",
            out.text
        );
        assert!(
            out.text
                .contains("data = (y = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],)"),
            "{}",
            out.text
        );
        assert!(out.text.contains("mean(θ) = "), "{}", out.text);
        assert!(
            !out.text.contains("MeanField"),
            "a conjugate model is exact"
        );
    }

    #[test]
    fn an_unset_constant_is_the_kind_default() {
        // A Normal without a mean parameter means 0.0 (its kind's
        // default), not some generic fallback.
        let lib = crate::rxinfer::library();
        let mut f = coin();
        f.blocks.push(block("n", "normal", &[("variance", "10")]));
        f.blocks
            .push(block("z", "variable", &[("name", "z"), ("data", "0.5")]));
        f.wires.push(wire("w4", "n", "out", "z", "x"));
        let out = generate(&f, &lib).unwrap();
        assert!(
            out.text.contains("z ~ Normal(mean = 0.0, variance = 10.0)"),
            "{}",
            out.text
        );
    }

    #[test]
    fn statements_follow_the_graph_not_the_canvas() {
        // The likelihood placed first on the canvas still comes after
        // its prior.
        let lib = crate::rxinfer::library();
        let mut f = coin();
        f.blocks.rotate_left(2);
        let out = generate(&f, &lib).unwrap();
        let prior = out.text.find("θ ~ Beta").unwrap();
        let like = out.text.find("y[i] ~ Bernoulli").unwrap();
        assert!(prior < like, "{}", out.text);
    }

    #[test]
    fn two_latent_inputs_make_the_run_mean_field() {
        let lib = crate::rxinfer::library();
        let mut f = Flow::new();
        f.blocks = vec![
            block("pm", "normal", &[("mean", "0.0"), ("variance", "100.0")]),
            block("mu", "variable", &[("name", "μ")]),
            block("pt", "gamma", &[]),
            block("tau", "variable", &[("name", "τ")]),
            block("lik", "normal_precision", &[]),
            block("y", "variable", &[("name", "y"), ("data", "1, 2, 3")]),
        ];
        f.wires = vec![
            wire("w1", "pm", "out", "mu", "x"),
            wire("w2", "pt", "out", "tau", "x"),
            wire("w3", "mu", "x", "lik", "mean"),
            wire("w4", "tau", "x", "lik", "precision"),
            wire("w5", "lik", "out", "y", "x"),
        ];
        let out = generate(&f, &lib).unwrap();
        assert!(
            out.text.contains("y[i] ~ Normal(mean = μ, precision = τ)"),
            "{}",
            out.text
        );
        assert!(
            out.text.contains("q(μ) = vague(NormalMeanVariance)"),
            "{}",
            out.text
        );
        assert!(
            out.text.contains("q(τ) = vague(GammaShapeRate)"),
            "{}",
            out.text
        );
        assert!(
            out.text.contains("constraints = MeanField(),"),
            "{}",
            out.text
        );
        assert!(
            out.text.contains("mean(result.posteriors[:μ][end])"),
            "{}",
            out.text
        );
    }

    #[test]
    fn errors_name_the_block() {
        let lib = crate::rxinfer::library();
        // A latent variable without a prior.
        let mut f = coin();
        f.wires.retain(|w| w.id != "w1");
        f.blocks.retain(|b| b.id != "prior");
        let err = generate(&f, &lib).unwrap_err();
        assert!(err.contains("θ has no prior"), "{err}");
        // Nothing observed.
        let mut f = coin();
        f.blocks[3].params.insert("data".into(), String::new());
        let err = generate(&f, &lib).unwrap_err();
        assert!(
            err.contains("nothing is observed") || err.contains("no prior"),
            "{err}"
        );
        // Two variables of one name, a bad name, bad data.
        let mut f = coin();
        f.blocks[1].params.insert("name".into(), "y".into());
        assert!(generate(&f, &lib)
            .unwrap_err()
            .contains("two variables are named y"));
        let mut f = coin();
        f.blocks[1].params.insert("name".into(), "1x".into());
        assert!(generate(&f, &lib)
            .unwrap_err()
            .contains("not a Julia variable name"));
        let mut f = coin();
        f.blocks[3].params.insert("data".into(), "1, heads".into());
        assert!(generate(&f, &lib).unwrap_err().contains("is not a number"));
        // A variable generated twice.
        let mut f = coin();
        f.blocks.push(block("prior2", "beta", &[]));
        f.wires.push(wire("w4", "prior2", "out", "theta", "x"));
        assert!(generate(&f, &lib)
            .unwrap_err()
            .contains("generated by both"));
        // A data vector as a factor's input.
        let mut f = coin();
        f.blocks.push(block("n2", "normal", &[]));
        f.blocks
            .push(block("z", "variable", &[("name", "z"), ("data", "0.5")]));
        f.wires.push(wire("w5", "y", "x", "n2", "mean"));
        f.wires.push(wire("w6", "n2", "out", "z", "x"));
        assert!(generate(&f, &lib)
            .unwrap_err()
            .contains("i.i.d. likelihood"));
        // Poisson counts must be whole.
        let mut f = Flow::new();
        f.blocks = vec![
            block("g", "gamma", &[]),
            block("l", "variable", &[("name", "λ")]),
            block("p", "poisson", &[]),
            block("y", "variable", &[("name", "y"), ("data", "3, 4.5")]),
        ];
        f.wires = vec![
            wire("w1", "g", "out", "l", "x"),
            wire("w2", "l", "x", "p", "rate"),
            wire("w3", "p", "out", "y", "x"),
        ];
        assert!(generate(&f, &lib).unwrap_err().contains("whole numbers"));
    }

    #[test]
    fn the_editor_rule_holds_for_these_kinds() {
        // Factor – factor is a validate issue (the bipartite rule of
        // ADR-0022, from the roles this library sets).
        let lib = crate::rxinfer::library();
        let mut f = coin();
        f.wires.push(wire("w9", "prior", "out", "flip", "p"));
        let issues = validate(&f, std::slice::from_ref(&lib));
        assert!(
            issues
                .iter()
                .any(|i| i.about == "w9" && i.message.contains("not to factors")),
            "{issues:?}"
        );
    }
}
