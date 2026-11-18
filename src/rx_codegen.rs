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

use moonkale_ext_api::flow::{Block, Flow, FlowLibrary, Generated, Role, Wire, BOUNDARY};
use std::collections::{BTreeMap, BTreeSet};

fn param<'a>(b: &'a Block, name: &str, default: &'a str) -> &'a str {
    b.params.get(name).map(|s| s.as_str()).unwrap_or(default)
}

/// A Julia float literal: finite, and always with a decimal point
/// (`2` → `2.0`), so a constant never turns a distribution integer.
fn float_lit(v: &str) -> Option<String> {
    let v = v.trim();
    v.parse::<f64>().ok().filter(|f| f.is_finite())?;
    // Keep the digits as typed: going through f64 would round integers
    // beyond 2^53 (J-BUG-03, Gemini's review 2026-11-07).
    Some(if v.contains(['.', 'e', 'E']) {
        v.to_string()
    } else {
        format!("{v}.0")
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
            // Julia's reserved words, and the generated file's own names.
            "end"
                | "else"
                | "elseif"
                | "while"
                | "return"
                | "true"
                | "false"
                | "try"
                | "catch"
                | "finally"
                | "do"
                | "struct"
                | "mutable"
                | "abstract"
                | "primitive"
                | "type"
                | "module"
                | "baremodule"
                | "import"
                | "using"
                | "export"
                | "const"
                | "global"
                | "local"
                | "break"
                | "continue"
                | "quote"
                | "macro"
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

/// One end of a wire, after flattening: a variable is one node whatever
/// its pin; a factor's channel and a subsystem instance's pin are each
/// their own.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum End {
    Var(String),
    Channel(String, String),
    Pin(String, String),
}

/// The factor graph with every subsystem inlined (ADR-0022 stage C,
/// flattened like ADR-0021 §6): blocks in flow order (an instance's
/// content where its block stands), wires with flat ids, and the
/// subsystem pins seen from each side.
#[derive(Default)]
struct Flat<'a> {
    /// `(flat id, block, name prefix)` of every variable and factor.
    blocks: Vec<(String, &'a Block, String)>,
    /// Every scope's wires as flat ends.
    wires: Vec<(End, End)>,
    /// Instance flat id → its subsystem's declared pins.
    pins: BTreeMap<String, Vec<String>>,
    /// Pin ends wired from the parent's side (the rest come from inside).
    outer: Vec<End>,
}

/// A name part that is a Julia identifier: lowercase ASCII letters,
/// digits and `_` (never leading with a digit).
fn sanitize(name: &str) -> String {
    let s: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    match s.chars().next() {
        None => "subsystem".into(),
        Some(c) if c.is_ascii_digit() => format!("s{s}"),
        Some(_) => s,
    }
}

/// Inline one scope (the root's, or an instance's) into `out`. `host`
/// is the instance block's flat id, where the scope's boundary ends land.
#[allow(clippy::too_many_arguments)]
fn expand<'a>(
    flow: &'a Flow,
    blocks: &'a [Block],
    wires: &'a [Wire],
    host: &str,
    id_prefix: &str,
    name_prefix: &str,
    counters: &mut BTreeMap<String, usize>,
    lib: &FlowLibrary,
    out: &mut Flat<'a>,
    depth: usize,
) -> Result<(), String> {
    let prefix = format!("{}/", lib.id);
    // `validate` refuses a subflow that contains itself; this bounds the
    // longer cycles too, so a hand-edited file cannot recurse away.
    if depth > 16 {
        return Err(format!("{host} nests subsystems too deep (a cycle?)"));
    }
    let mut kinds: BTreeMap<&str, End> = BTreeMap::new();
    // A block's pins, so a wire to a port it lacks is an error, never
    // silently dropped (review 2026-11-18, V2).
    let mut pins_of: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for b in blocks {
        let flat = format!("{id_prefix}{}", b.id);
        if let Some(sub_id) = &b.subflow {
            let sub = flow.subflow(sub_id).ok_or_else(|| {
                format!(
                    "{} references the subflow {sub_id}, which this document does not hold",
                    b.id
                )
            })?;
            if sub.role == Some(Role::Variable) {
                return Err(format!(
                    "subflow {sub_id} has the role variable — a subsystem is a factor"
                ));
            }
            if !sub.inputs.is_empty() || !sub.outputs.is_empty() {
                return Err(format!(
                    "subflow {sub_id} declares directed ports — a factor's boundary is its pins (channels)"
                ));
            }
            // Numbered per subsystem name, so `noisy_1` is unique
            // document-wide and two instances never share a variable.
            let base = sanitize(&sub.name);
            let n = counters.entry(base.clone()).or_insert(0);
            *n += 1;
            let instance = format!("{base}_{n}");
            out.pins.insert(
                flat.clone(),
                sub.connections.iter().map(|p| p.name.clone()).collect(),
            );
            kinds.insert(&b.id, End::Pin(flat.clone(), String::new()));
            expand(
                flow,
                &sub.blocks,
                &sub.wires,
                &flat,
                &format!("{flat}/"),
                &format!("{name_prefix}{instance}_"),
                counters,
                lib,
                out,
                depth + 1,
            )?;
        } else {
            let Some(k) = b.kind.strip_prefix(&prefix) else {
                return Err(format!(
                    "{} is not an RxInfer block — one library per flow for now",
                    b.kind
                ));
            };
            if let Some(kind) = lib.block(k) {
                pins_of.insert(
                    &b.id,
                    kind.connections.iter().map(|p| p.name.as_str()).collect(),
                );
            }
            kinds.insert(
                &b.id,
                if k == "variable" {
                    End::Var(flat.clone())
                } else {
                    End::Channel(flat.clone(), String::new())
                },
            );
            out.blocks.push((flat, b, name_prefix.to_string()));
        }
    }
    let end = |id: &str, port: &str| -> Result<(End, bool), String> {
        if let Some(pin) = id.strip_prefix(BOUNDARY) {
            if host.is_empty() {
                return Err(format!(
                    "boundary pin {pin} outside a subsystem — the root has no boundary"
                ));
            }
            if !out
                .pins
                .get(host)
                .is_some_and(|p| p.iter().any(|p| p == pin))
            {
                return Err(format!(
                    "boundary pin {pin} of {host} is not a declared pin"
                ));
            }
            return Ok((End::Pin(host.to_string(), pin.to_string()), false));
        }
        if let Some(pins) = pins_of.get(id) {
            if !pins.contains(&port) {
                return Err(format!(
                    "{id} has no pin {port} — the wire cannot be generated"
                ));
            }
        }
        match kinds.get(id) {
            Some(End::Var(v)) => Ok((End::Var(v.clone()), false)),
            Some(End::Channel(f, _)) => Ok((End::Channel(f.clone(), port.to_string()), false)),
            Some(End::Pin(h, _)) => {
                if !out.pins.get(h).is_some_and(|p| p.iter().any(|p| p == port)) {
                    return Err(format!("{port} is not a declared pin of {h}"));
                }
                Ok((End::Pin(h.clone(), port.to_string()), true))
            }
            None => Err(format!("a wire names {id}, which this scope does not hold")),
        }
    };
    for w in wires {
        let (a, a_outer) = end(&w.from_block, &w.from_port)?;
        let (b, b_outer) = end(&w.to_block, &w.to_port)?;
        if a_outer {
            out.outer.push(a.clone());
        }
        if b_outer {
            out.outer.push(b.clone());
        }
        out.wires.push((a, b));
    }
    Ok(())
}

/// Union-find over flat ends: a variable, the channels wired to it and
/// the subsystem pins between them are one class.
#[derive(Default)]
struct Classes {
    parent: BTreeMap<End, End>,
}

impl Classes {
    fn find(&mut self, x: &End) -> End {
        let p = self.parent.get(x).cloned().unwrap_or_else(|| x.clone());
        if &p == x {
            return p;
        }
        let root = self.find(&p);
        self.parent.insert(x.clone(), root.clone());
        root
    }
    fn union(&mut self, a: &End, b: &End) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent.insert(ra, rb);
        }
    }
}

pub fn generate(flow: &Flow, lib: &FlowLibrary) -> Result<Generated, String> {
    let prefix = format!("{}/", lib.id);
    let kind_of = |b: &Block| b.kind.strip_prefix(&prefix).map(str::to_string);

    // Every subsystem inlined: its variables carry its instance's prefix,
    // its boundary pins continue the parent's wires (ADR-0022 stage C).
    let mut flat = Flat::default();
    expand(
        flow,
        &flow.blocks,
        &flow.wires,
        "",
        "",
        "",
        &mut BTreeMap::new(),
        lib,
        &mut flat,
        0,
    )?;
    // A subsystem pin wired inside continues the variable wired to it
    // outside; with none there, nothing outside defines it (review
    // 2026-11-18, V1: an inner variable must not stand in for it).
    for (a, b) in &flat.wires {
        for e in [a, b] {
            if let End::Pin(h, p) = e {
                if !flat.outer.contains(e) {
                    return Err(format!(
                        "pin {p} of {h} is wired inside but not to a variable outside — a subsystem pin continues the variable wired to it"
                    ));
                }
            }
        }
    }

    // The variables: named, unique, their data parsed.
    let mut vars: BTreeMap<String, Var> = BTreeMap::new();
    let mut taken: BTreeSet<String> = BTreeSet::new();
    let mut factors: Vec<(&str, &Block, Dist)> = Vec::new();
    for (id, b, name_prefix) in &flat.blocks {
        let k = kind_of(b).unwrap_or_default();
        if k == "variable" {
            let short = param(b, "name", "x").trim().to_string();
            if !is_ident(&short) {
                return Err(format!("{id}: {short:?} is not a Julia variable name"));
            }
            let name = format!("{name_prefix}{short}");
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
            vars.insert(id.clone(), Var { name, data });
        } else if let Some(d) = dist(&k) {
            factors.push((id, b, d));
        } else {
            return Err(format!("unknown kind {}", b.kind));
        }
    }

    // A channel attaches to one variable — a factor's, and a subsystem's
    // seen from outside (either drag order: a pin wire's ends are just
    // its two ends).
    let mut on: BTreeMap<&End, usize> = BTreeMap::new();
    for e in flat
        .wires
        .iter()
        .flat_map(|(a, b)| [a, b])
        .filter(|e| matches!(e, End::Channel(..)))
        .chain(&flat.outer)
    {
        *on.entry(e).or_insert(0) += 1;
    }
    for (e, n) in &on {
        if *n > 1 {
            let (End::Channel(f, ch) | End::Pin(f, ch)) = e else {
                continue;
            };
            return Err(format!(
                "channel {ch} of {f} carries two wires — a channel attaches to one variable"
            ));
        }
    }
    let mut classes = Classes::default();
    for (a, b) in &flat.wires {
        classes.union(a, b);
    }
    // Each class holds at most one variable: a subsystem pin continues
    // one variable, it never joins two.
    let mut var_of: BTreeMap<End, &str> = BTreeMap::new();
    for id in vars.keys() {
        let root = classes.find(&End::Var(id.clone()));
        if let Some(other) = var_of.insert(root, id) {
            return Err(format!(
                "{} and {} are joined through a subsystem pin — a pin continues one variable",
                vars[other].name, vars[id].name
            ));
        }
    }
    // Every wired end → its class.
    let ends: BTreeSet<End> = flat
        .wires
        .iter()
        .flat_map(|(a, b)| [a.clone(), b.clone()])
        .collect();
    let class_of: BTreeMap<End, End> = ends
        .into_iter()
        .map(|e| {
            let root = classes.find(&e);
            (e, root)
        })
        .collect();
    let wired = |f: &str, ch: &str| -> Result<Option<&Var>, String> {
        let Some(root) = class_of.get(&End::Channel(f.to_string(), ch.to_string())) else {
            return Ok(None); // unwired: the constant
        };
        if let Some(v) = var_of.get(root) {
            return Ok(Some(&vars[*v]));
        }
        // Wired, but no variable at the far end: a subsystem pin left
        // open outside, or a factor straight onto a factor.
        let open = class_of
            .iter()
            .find(|(e, r)| matches!(e, End::Pin(..)) && *r == root);
        Err(match open {
            Some((End::Pin(h, p), _)) => format!(
                "{f}.{ch} reaches pin {p} of the subsystem {h}, which is not wired to a variable outside"
            ),
            _ => format!(
                "{f}.{ch} is wired to a factor, not a variable — factors connect to variables"
            ),
        })
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
    for (fid, f, d) in &factors {
        let fid: &str = fid;
        let out = wired(fid, "out")?.ok_or(format!(
            "the out channel of {} is not connected — wire it to the variable it generates",
            fid
        ))?;
        if let Some(other) = generated_by.insert(&out.name, fid) {
            return Err(format!(
                "{} is generated by both {other} and {} — a variable has one generating factor",
                out.name, fid
            ));
        }
        let mut args = Vec::new();
        let mut inputs = Vec::new();
        for ch in d.channels {
            let value = match wired(fid, ch)? {
                Some(v) => {
                    if v.vector() {
                        return Err(format!(
                            "{}: the data vector {} can only be a factor's out (an i.i.d. likelihood)",
                            fid, v.name
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
                    float_lit(v).ok_or(format!("{}.{ch} = {v:?} is not a number", fid))?
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
                fid, out.name
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
            id: fid,
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
    let observed: Vec<&Var> = flat
        .blocks
        .iter()
        .filter_map(|(id, _, _)| vars.get(id))
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
    #[test]
    fn float_literals_keep_their_digits() {
        assert_eq!(float_lit("2").as_deref(), Some("2.0"));
        assert_eq!(float_lit(" 3 ").as_deref(), Some("3.0"));
        assert_eq!(float_lit("1e-3").as_deref(), Some("1e-3"));
        assert_eq!(
            float_lit("12345678901234567890").as_deref(),
            Some("12345678901234567890.0")
        );
        assert_eq!(float_lit("NaN"), None);
        assert_eq!(float_lit("inf"), None);
        assert_eq!(float_lit("x"), None);
    }

    // ---- ADR-0022 stage C: a subsystem as one factor ---------------

    fn pin(n: &str) -> moonkale_ext_api::flow::Port {
        moonkale_ext_api::flow::Port {
            name: n.into(),
            ty: moonkale_ext_api::flow::PortType::Any,
            required: false,
        }
    }

    fn instance(id: &str, sub: &str) -> Block {
        Block {
            id: id.into(),
            kind: "subflow".into(),
            subflow: Some(sub.into()),
            ..Default::default()
        }
    }

    /// The coin with its likelihood as one factor: a subsystem "Coin"
    /// whose pins are `out` and `p`, a Bernoulli inside.
    pub(super) fn nested_coin() -> Flow {
        let mut f = coin();
        f.blocks.retain(|b| b.id != "flip");
        f.blocks.insert(2, instance("coin-1", "coin"));
        f.wires = vec![
            wire("w1", "prior", "out", "theta", "x"),
            wire("w2", "theta", "x", "coin-1", "p"),
            wire("w3", "coin-1", "out", "y", "x"),
        ];
        f.subflows = vec![moonkale_ext_api::flow::Subflow {
            id: "coin".into(),
            name: "Coin".into(),
            blocks: vec![block("flip", "bernoulli", &[])],
            wires: vec![
                wire("iw1", "boundary/p", "p", "flip", "p"),
                wire("iw2", "flip", "out", "boundary/out", "out"),
            ],
            connections: vec![pin("out"), pin("p")],
            role: Some(Role::Factor),
            ..Default::default()
        }];
        f
    }

    #[test]
    fn a_subsystem_as_one_factor_generates_the_flat_model() {
        let lib = crate::rxinfer::library();
        let f = nested_coin();
        assert_eq!(validate(&f, std::slice::from_ref(&lib)), vec![]);
        // The flatten is exact: the same file as the coin drawn flat.
        assert_eq!(
            generate(&f, &lib).unwrap().text,
            generate(&coin(), &lib).unwrap().text
        );
        // One level deeper: the Coin inside a "Trial" subsystem — the
        // boundary continues through both.
        let mut g = f.clone();
        g.blocks[2] = instance("trial-1", "trial");
        g.wires[1].to_block = "trial-1".into();
        g.wires[2].from_block = "trial-1".into();
        g.subflows.push(moonkale_ext_api::flow::Subflow {
            id: "trial".into(),
            name: "Trial".into(),
            blocks: vec![instance("inner", "coin")],
            wires: vec![
                wire("tw1", "boundary/p", "p", "inner", "p"),
                wire("tw2", "inner", "out", "boundary/out", "out"),
            ],
            connections: vec![pin("out"), pin("p")],
            role: Some(Role::Factor),
            ..Default::default()
        });
        assert_eq!(validate(&g, std::slice::from_ref(&lib)), vec![]);
        assert_eq!(
            generate(&g, &lib).unwrap().text,
            generate(&coin(), &lib).unwrap().text
        );
    }

    #[test]
    fn two_instances_keep_their_inner_variables_apart() {
        // A "Noisy" measurement: inside, its own precision τ ~ Gamma and
        // out ~ Normal(mean, precision τ). Two instances measure one
        // latent μ ~ Normal(0, 100); each has its own τ.
        let lib = crate::rxinfer::library();
        let mut f = Flow::new();
        f.blocks = vec![
            block("prior", "normal", &[("mean", "0"), ("variance", "100")]),
            block("mu", "variable", &[("name", "μ")]),
            instance("n-a", "noisy"),
            instance("n-b", "noisy"),
            block("a", "variable", &[("name", "a"), ("data", "1.2, 0.8, 1.1")]),
            block("b", "variable", &[("name", "b"), ("data", "0.9, 1.3")]),
        ];
        f.wires = vec![
            wire("w1", "prior", "out", "mu", "x"),
            wire("w2", "mu", "x", "n-a", "mean"),
            wire("w3", "mu", "x", "n-b", "mean"),
            wire("w4", "n-a", "out", "a", "x"),
            wire("w5", "n-b", "out", "b", "x"),
        ];
        f.subflows = vec![moonkale_ext_api::flow::Subflow {
            id: "noisy".into(),
            name: "Noisy".into(),
            blocks: vec![
                block("g", "gamma", &[("shape", "2"), ("rate", "1")]),
                block("tau", "variable", &[("name", "τ")]),
                block("n", "normal_precision", &[]),
            ],
            wires: vec![
                wire("iw1", "g", "out", "tau", "x"),
                wire("iw2", "tau", "x", "n", "precision"),
                wire("iw3", "boundary/mean", "mean", "n", "mean"),
                wire("iw4", "n", "out", "boundary/out", "out"),
            ],
            connections: vec![pin("out"), pin("mean")],
            role: Some(Role::Factor),
            ..Default::default()
        }];
        assert_eq!(validate(&f, std::slice::from_ref(&lib)), vec![]);
        let text = generate(&f, &lib).unwrap().text;
        for line in [
            "@model function factor_graph(a, b)",
            "    μ ~ Normal(mean = 0.0, variance = 100.0)",
            "    noisy_1_τ ~ Gamma(shape = 2.0, rate = 1.0)",
            "        a[i] ~ Normal(mean = μ, precision = noisy_1_τ)",
            "    noisy_2_τ ~ Gamma(shape = 2.0, rate = 1.0)",
            "        b[i] ~ Normal(mean = μ, precision = noisy_2_τ)",
            // Two latent inputs: mean-field, each τ initialised.
            "    q(noisy_1_τ) = vague(GammaShapeRate)",
            "    q(noisy_2_τ) = vague(GammaShapeRate)",
            "println(\"mean(noisy_2_τ) = \", mean(result.posteriors[:noisy_2_τ][end]))",
        ] {
            assert!(text.contains(line), "missing {line:?} in\n{text}");
        }
    }

    #[test]
    fn a_subsystem_pin_is_checked_like_a_channel() {
        let lib = crate::rxinfer::library();
        // Wired inside, open outside.
        let mut f = nested_coin();
        f.wires.retain(|w| w.id != "w2");
        let err = generate(&f, &lib).unwrap_err();
        assert!(
            err.contains("pin p of coin-1 is wired inside but not to a variable outside"),
            "{err}"
        );
        // Two variables on one channel of the subsystem.
        let mut f = nested_coin();
        f.blocks
            .push(block("z", "variable", &[("name", "z"), ("data", "1")]));
        f.wires.push(wire("w4", "coin-1", "out", "z", "x"));
        let err = generate(&f, &lib).unwrap_err();
        assert!(
            err.contains("channel out of coin-1 carries two wires"),
            "{err}"
        );
        // A pin joining two variables (an inner variable on a boundary
        // pin — `validate` refuses it too).
        let mut f = nested_coin();
        f.subflows[0]
            .blocks
            .push(block("v", "variable", &[("name", "v")]));
        f.subflows[0]
            .wires
            .push(wire("iw3", "v", "x", "boundary/p", "p"));
        let err = generate(&f, &lib).unwrap_err();
        assert!(err.contains("are joined through a subsystem pin"), "{err}");
        // A subsystem is never a variable; a factor's boundary is pins.
        let mut f = nested_coin();
        f.subflows[0].role = Some(Role::Variable);
        assert!(generate(&f, &lib).unwrap_err().contains("is a factor"));
        let mut f = nested_coin();
        f.subflows[0].inputs = vec![pin("u")];
        assert!(generate(&f, &lib).unwrap_err().contains("directed ports"));
        // A boundary pin that is not declared.
        let mut f = nested_coin();
        f.subflows[0].wires[0].from_block = "boundary/q".into();
        assert!(generate(&f, &lib)
            .unwrap_err()
            .contains("not a declared pin"));
        // A subflow that references itself ends, with an error.
        let mut f = nested_coin();
        f.subflows[0].blocks.push(instance("again", "coin"));
        assert!(generate(&f, &lib).unwrap_err().contains("too deep"));
    }

    /// Review 2026-11-18 (Mistral, V1/V2/S2): an inner variable standing
    /// in for an unwired pin, a wire to a port the block lacks, and a
    /// Julia keyword as a name are refused, never a silently wrong file.
    #[test]
    fn review_findings_are_refused() {
        let lib = crate::rxinfer::library();
        // V1: pin p open outside, an inner variable on boundary/p.
        let mut f = nested_coin();
        f.wires.retain(|w| w.id != "w2");
        f.subflows[0]
            .blocks
            .push(block("v", "variable", &[("name", "v"), ("data", "0.5")]));
        f.subflows[0]
            .wires
            .push(wire("iw3", "v", "x", "boundary/p", "p"));
        let err = generate(&f, &lib).unwrap_err();
        assert!(err.contains("pin p of coin-1"), "{err}");
        // V2: a wire onto a channel the Bernoulli does not have.
        let mut f = coin();
        f.wires[1].to_port = "bogus".into();
        let err = generate(&f, &lib).unwrap_err();
        assert!(err.contains("bogus"), "{err}");
        // S2: a Julia keyword is not a variable name.
        let mut f = coin();
        f.blocks[1].params.insert("name".into(), "while".into());
        assert!(generate(&f, &lib)
            .unwrap_err()
            .contains("not a Julia variable name"));
    }
}
