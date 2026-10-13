//! Flow → `model.jl`. The layer chain is the path from `Input` to `Loss`
//! (blocks in topological order along the wires); everything else is
//! reported as an error so a half-wired flow never produces a silently wrong
//! file.
//!
//! A subflow block in the chain is a sub-`Chain` (ADR-0021 §6): the inner
//! layer path splices between the boundary input and the boundary
//! output — exactly one of each, because a chain has one path — and
//! nested subflows recurse. Only layers may sit inside; a Loss or an
//! Optimiser there is an error, not a layer.

use moonkale_ext_api::flow::{Block, Flow, FlowLibrary, Generated, Subflow};

/// The boundary pseudo-block id prefix (ADR-0021 §3), as in
/// `mtk_codegen.rs` — ext-api's own `flow::BOUNDARY` is still above
/// the tag this crate pins.
const BOUNDARY: &str = "boundary/";

fn param<'a>(b: &'a Block, name: &str, default: &'a str) -> &'a str {
    b.params.get(name).map(|s| s.as_str()).unwrap_or(default)
}

/// Julia integer/float literal from a parameter, validated.
fn num(b: &Block, name: &str, default: &str) -> Result<String, String> {
    let v = param(b, name, default).trim();
    v.parse::<f64>()
        .map(|_| v.to_string())
        .map_err(|_| format!("{}.{name} = {v:?} is not a number", b.id))
}

/// The chain's running shape: the input features or channels the next
/// layer sees (whatever the last layer left behind).
struct ChainState {
    in_features: Option<String>,
    in_channels: Option<String>,
}

/// One block's layer, spliced into `layers`, updating the running
/// shape. Only layer kinds belong here; a chain ender (Loss,
/// Optimiser) or an unknown kind is an error — the outer walk handles
/// the chain's own ending.
fn emit_layer(
    b: &Block,
    prefix: &str,
    st: &mut ChainState,
    layers: &mut Vec<String>,
) -> Result<(), String> {
    let kind = b.kind.strip_prefix(prefix).unwrap_or(&b.kind);
    match kind {
        "dense" => {
            let out = num(b, "out", "128")?;
            let act = param(b, "activation", "relu");
            let inp = st.in_features.clone().ok_or_else(|| {
                format!(
                    "{}: input size unknown — put a Flatten before a Dense after image layers",
                    b.id
                )
            })?;
            layers.push(format!("Dense({inp} => {out}, {act})"));
            st.in_features = Some(out);
        }
        "conv" => {
            let ch = num(b, "channels", "16")?;
            let k = num(b, "kernel", "3")?;
            let pad = num(b, "pad", "1")?;
            let act = param(b, "activation", "relu");
            let inp = st
                .in_channels
                .clone()
                .ok_or_else(|| format!("{}: channel count unknown", b.id))?;
            layers.push(format!("Conv(({k}, {k}), {inp} => {ch}, {act}; pad={pad})"));
            st.in_channels = Some(ch);
            st.in_features = None;
        }
        "maxpool" => {
            let k = num(b, "kernel", "2")?;
            layers.push(format!("MaxPool(({k}, {k}))"));
        }
        "flatten" => {
            layers.push("FlattenLayer()".into());
            // Features after flatten depend on the spatial size; let Lux
            // infer them lazily.
            st.in_features = Some("Lux.NoInputSize".into());
        }
        "dropout" => layers.push(format!("Dropout({})", num(b, "p", "0.5")?)),
        "batchnorm" => {
            let ch = st
                .in_channels
                .clone()
                .or(st.in_features.clone())
                .ok_or("BatchNorm: size unknown")?;
            layers.push(format!("BatchNorm({ch})"));
        }
        other => {
            return Err(format!(
                "{}: block kind {other} cannot be part of the chain",
                b.id
            ))
        }
    }
    Ok(())
}

/// Splice a subflow block's inner path into the chain (ADR-0021 §6):
/// from the boundary input, block by block, until the boundary
/// output. A nested subflow recurses.
fn splice_subflow(
    flow: &Flow,
    sub: &Subflow,
    prefix: &str,
    st: &mut ChainState,
    layers: &mut Vec<String>,
    depth: usize,
) -> Result<(), String> {
    if depth > 16 {
        return Err(format!("subflow {} nests too deep", sub.id));
    }
    // A chain has one path: one boundary input, one boundary output.
    let (in_port, out_port) = match (sub.inputs.first(), sub.outputs.first()) {
        (Some(i), Some(o)) if sub.inputs.len() == 1 && sub.outputs.len() == 1 => {
            (i.name.clone(), o.name.clone())
        }
        _ => {
            return Err(format!(
                "subflow {} must declare exactly one input and one output to sit in a chain",
                sub.id
            ))
        }
    };
    let first = sub
        .wires
        .iter()
        .find(|w| w.from_block == format!("{BOUNDARY}{in_port}"));
    // A pass-through subflow (input wired straight to output) splices
    // nothing.
    if first.is_some_and(|w| w.to_block == format!("{BOUNDARY}{out_port}")) {
        return Ok(());
    }
    let mut cur = first
        .and_then(|w| sub.blocks.iter().find(|b| b.id == w.to_block))
        .ok_or_else(|| {
            format!(
                "subflow {} does not feed its boundary input {in_port}",
                sub.id
            )
        })? as &Block;
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 256 {
            return Err(format!(
                "subflow {subid}: the inner path is too long",
                subid = sub.id
            ));
        }
        if let Some(nested) = &cur.subflow {
            let sub = flow.subflow(nested).ok_or_else(|| {
                format!(
                    "{} references the subflow {nested}, which this document does not hold",
                    cur.id
                )
            })?;
            splice_subflow(flow, sub, prefix, st, layers, depth + 1)?;
        } else {
            emit_layer(cur, prefix, st, layers)?;
        }
        let wire = sub
            .wires
            .iter()
            .find(|w| w.from_block == cur.id)
            .ok_or_else(|| {
                format!(
                    "subflow {subid}: the path from {in_port} breaks at {}",
                    cur.id,
                    subid = sub.id
                )
            })?;
        if wire.to_block == format!("{BOUNDARY}{out_port}") {
            return Ok(());
        }
        cur = sub
            .blocks
            .iter()
            .find(|b| b.id == wire.to_block)
            .ok_or_else(|| {
                format!(
                    "subflow {subid}: {} wires to {} which the subflow does not hold",
                    cur.id,
                    wire.to_block,
                    subid = sub.id
                )
            })?;
    }
}

pub fn generate(flow: &Flow, lib: &FlowLibrary) -> Result<Generated, String> {
    let prefix = format!("{}/", lib.id);
    let kind = |b: &Block| b.kind.strip_prefix(&prefix).unwrap_or(&b.kind).to_string();
    let order = flow.topological()?;
    let input = order
        .iter()
        .find(|b| kind(b) == "input")
        .ok_or("the flow needs an Input block")?;
    // Follow the single output wire from block to block.
    let next = |from: &Block| -> Option<&Block> {
        flow.wires
            .iter()
            .find(|w| w.from_block == from.id)
            .and_then(|w| flow.block(&w.to_block))
    };
    let mut layers: Vec<String> = Vec::new();
    let mut cur = next(input);
    let mut loss: Option<&Block> = None;
    let mut optimiser: Option<&Block> = None;
    let mut st = ChainState {
        in_features: shape_features(param(input, "shape", "28,28,1")),
        in_channels: shape_channels(param(input, "shape", "28,28,1")),
    };
    let mut guard = 0;
    while let Some(b) = cur {
        guard += 1;
        if guard > 256 {
            return Err("chain too long (cycle?)".into());
        }
        // A subflow block is a sub-Chain: its inner path splices in
        // between the boundary input and output (ADR-0021 §6).
        if let Some(sub_id) = &b.subflow {
            let sub = flow.subflow(sub_id).ok_or_else(|| {
                format!(
                    "{} references the subflow {sub_id}, which this document does not hold",
                    b.id
                )
            })?;
            splice_subflow(flow, sub, &prefix, &mut st, &mut layers, 0)?;
            cur = next(b);
            continue;
        }
        if kind(b) == "loss" {
            loss = Some(b);
            cur = next(b);
            if let Some(o) = cur {
                if kind(o) == "optimiser" {
                    optimiser = Some(o);
                }
            }
            break;
        }
        emit_layer(b, &prefix, &mut st, &mut layers)?;
        cur = next(b);
    }
    let loss = loss.ok_or("the chain must end in a Loss block")?;
    if layers.is_empty() {
        return Err("the flow has no layers between Input and Loss".into());
    }
    // `Lux.NoInputSize` is a placeholder for "use a lazy Dense": Lux needs the
    // size, so we compute it by running the front of the chain once.
    let lazy = layers.iter().any(|l| l.contains("Lux.NoInputSize"));
    let chain = layers
        .iter()
        .map(|l| l.replace("Lux.NoInputSize", "flat_features"))
        .collect::<Vec<_>>()
        .join(",\n    ");
    let shape = param(input, "shape", "28,28,1")
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(", ");
    let batch = num(input, "batch", "32")?;
    let loss_kind = param(loss, "kind", "CrossEntropyLoss");
    let classes = num(loss, "classes", "10")?;
    let (rule, lr, epochs) = match optimiser {
        Some(o) => (
            param(o, "rule", "Adam").to_string(),
            num(o, "lr", "0.001")?,
            num(o, "epochs", "5")?,
        ),
        None => ("Adam".into(), "0.001".into(), "5".into()),
    };
    let mut text = String::new();
    text.push_str("# Generated by Moonkale's Lux.jl extension from the flow editor.\n");
    text.push_str("# Needs: using Pkg; Pkg.add([\"Lux\", \"Optimisers\", \"Zygote\"])\n");
    text.push_str("using Lux, Optimisers, Zygote, Random\n\n");
    text.push_str(&format!(
        "const INPUT_SHAPE = ({shape})\nconst BATCH = {batch}\nconst CLASSES = {classes}\n\n"
    ));
    if lazy {
        text.push_str("# Features after FlattenLayer, computed once from the input shape.\n");
        let front: Vec<&String> = layers
            .iter()
            .take_while(|l| !l.contains("Lux.NoInputSize"))
            .collect();
        text.push_str("const flat_features = let\n    front = Chain(\n        ");
        text.push_str(
            &front
                .iter()
                .map(|l| l.as_str())
                .collect::<Vec<_>>()
                .join(",\n        "),
        );
        text.push_str("\n    )\n    ps, st = Lux.setup(Random.default_rng(), front)\n    y, _ = front(randn(Float32, INPUT_SHAPE..., 1), ps, st)\n    prod(size(y)[1:end-1])\nend\n\n");
    }
    text.push_str(&format!("model = Chain(\n    {chain}\n)\n\n"));
    text.push_str("rng = Random.default_rng()\nps, st = Lux.setup(rng, model)\n\n");
    text.push_str("# Replace with real data: x is INPUT_SHAPE × BATCH, y is one-hot CLASSES × BATCH.\nx = randn(Float32, INPUT_SHAPE..., BATCH)\ny = Lux.onehotbatch(rand(rng, 1:CLASSES, BATCH), 1:CLASSES)\n\n");
    text.push_str(&format!("loss_fn = {loss_kind}()\nopt = Optimisers.{rule}({lr})\ntrain_state = Lux.Training.TrainState(model, ps, st, opt)\n\n"));
    text.push_str(&format!("for epoch in 1:{epochs}\n    _, loss, _, train_state = Lux.Training.single_train_step!(AutoZygote(), loss_fn, (x, y), train_state)\n    println(\"epoch \", epoch, \" loss \", loss)\nend\n"));
    Ok(Generated {
        file_name: "model.jl".into(),
        text,
        run_hint: "julia model.jl".into(),
    })
}

fn shape_dims(shape: &str) -> Vec<String> {
    shape
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}
fn shape_features(shape: &str) -> Option<String> {
    let d = shape_dims(shape);
    (d.len() == 1).then(|| d[0].clone())
}
fn shape_channels(shape: &str) -> Option<String> {
    let d = shape_dims(shape);
    (d.len() == 3).then(|| d[2].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use moonkale_ext_api::flow::{Block, Port, PortType, Subflow, Wire};
    use std::collections::BTreeMap;

    fn b(id: &str, kind: &str, params: &[(&str, &str)]) -> Block {
        Block {
            id: id.into(),
            kind: format!("lux/{kind}"),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>(),
            ..Default::default()
        }
    }
    fn w(a: &str, ap: &str, b: &str, bp: &str) -> Wire {
        Wire {
            id: format!("{a}-{b}"),
            from_block: a.into(),
            from_port: ap.into(),
            to_block: b.into(),
            to_port: bp.into(),
        }
    }

    #[test]
    fn cnn_generates_a_chain_with_lazy_flatten_size() {
        let lib = crate::library();
        let mut f = Flow::new();
        f.blocks = vec![
            b("in", "input", &[("shape", "28,28,1")]),
            b("c1", "conv", &[("channels", "8"), ("kernel", "3")]),
            b("p1", "maxpool", &[]),
            b("fl", "flatten", &[]),
            b("d1", "dense", &[("out", "10"), ("activation", "identity")]),
            b("l", "loss", &[]),
            b("o", "optimiser", &[("epochs", "2")]),
        ];
        f.wires = vec![
            w("in", "out", "c1", "in"),
            w("c1", "out", "p1", "in"),
            w("p1", "out", "fl", "in"),
            w("fl", "out", "d1", "in"),
            w("d1", "out", "l", "prediction"),
            w("l", "loss", "o", "loss"),
        ];
        assert!(moonkale_ext_api::flow::validate(&f, std::slice::from_ref(&lib)).is_empty());
        let g = generate(&f, &lib).unwrap();
        assert_eq!(g.file_name, "model.jl");
        assert!(
            g.text.contains("Conv((3, 3), 1 => 8, relu; pad=1)"),
            "{}",
            g.text
        );
        assert!(g.text.contains("Dense(flat_features => 10, identity)"));
        assert!(g.text.contains("const flat_features = let"));
        assert!(g.text.contains("for epoch in 1:2"));
    }

    #[test]
    fn errors_are_specific() {
        let lib = crate::library();
        let mut f = Flow::new();
        f.blocks = vec![
            b("in", "input", &[("shape", "28,28,1")]),
            b("d", "dense", &[]),
            b("l", "loss", &[]),
        ];
        f.wires = vec![w("in", "out", "d", "in"), w("d", "out", "l", "prediction")];
        let e = generate(&f, &lib).unwrap_err();
        assert!(e.contains("Flatten"), "{e}");
        f.blocks[1].params.insert("out".into(), "many".into());
        f.blocks[0].params.insert("shape".into(), "784".into());
        assert!(generate(&f, &lib).unwrap_err().contains("not a number"));
    }
    // ---- ADR-0021 stage C: the sub-Chain splice -------------------

    #[test]
    fn a_subflow_splices_its_inner_path_into_the_chain() {
        let mut f = Flow::new();
        f.subflows = vec![Subflow {
            id: "encoder".into(),
            name: "Encoder".into(),
            // One path in, one path out — a chain has no branches.
            inputs: vec![Port {
                name: "x".into(),
                ty: PortType::Any,
                required: false,
            }],
            outputs: vec![Port {
                name: "y".into(),
                ty: PortType::Any,
                required: false,
            }],
            connections: vec![],
            blocks: vec![
                Block {
                    id: "conv-in".into(),
                    kind: "lux/conv".into(),
                    params: [
                        ("channels".to_string(), "8".to_string()),
                        ("kernel".to_string(), "3".to_string()),
                        ("pad".to_string(), "1".to_string()),
                        ("activation".to_string(), "relu".to_string()),
                    ]
                    .into_iter()
                    .collect(),
                    ..Default::default()
                },
                Block {
                    id: "fl-in".into(),
                    kind: "lux/flatten".into(),
                    ..Default::default()
                },
            ],
            wires: vec![
                Wire {
                    id: "sw1".into(),
                    from_block: "boundary/x".into(),
                    from_port: "x".into(),
                    to_block: "conv-in".into(),
                    to_port: "in".into(),
                },
                Wire {
                    id: "sw2".into(),
                    from_block: "conv-in".into(),
                    from_port: "out".into(),
                    to_block: "fl-in".into(),
                    to_port: "in".into(),
                },
                Wire {
                    id: "sw3".into(),
                    from_block: "fl-in".into(),
                    from_port: "out".into(),
                    to_block: "boundary/y".into(),
                    to_port: "y".into(),
                },
            ],
        }];
        // Input → the subsystem → Loss: the inner Conv and Flatten
        // splice in where the subsystem sits.
        f.blocks = vec![
            Block {
                id: "in1".into(),
                kind: "lux/input".into(),
                params: [("shape".to_string(), "28,28,1".to_string())]
                    .into_iter()
                    .collect(),
                ..Default::default()
            },
            Block {
                id: "sub".into(),
                kind: "subflow".into(),
                subflow: Some("encoder".into()),
                ..Default::default()
            },
            Block {
                id: "loss".into(),
                kind: "lux/loss".into(),
                ..Default::default()
            },
        ];
        f.wires = vec![
            Wire {
                id: "w1".into(),
                from_block: "in1".into(),
                from_port: "out".into(),
                to_block: "sub".into(),
                to_port: "x".into(),
            },
            Wire {
                id: "w2".into(),
                from_block: "sub".into(),
                from_port: "y".into(),
                to_block: "loss".into(),
                to_port: "prediction".into(),
            },
        ];
        let lib = crate::library::library();
        let out = generate(&f, &lib).expect("splices");
        assert!(
            out.text.contains("Conv((3, 3), 1 => 8, relu; pad=1)"),
            "the inner Conv spliced in: {}",
            out.text
        );
        assert!(
            out.text.contains("FlattenLayer()"),
            "the inner Flatten spliced in: {}",
            out.text
        );
    }

    #[test]
    fn a_subflow_without_a_single_path_is_refused() {
        let mut f = Flow::new();
        f.subflows = vec![Subflow {
            id: "two_ins".into(),
            name: "Two inputs".into(),
            inputs: vec![
                Port {
                    name: "a".into(),
                    ty: PortType::Any,
                    required: false,
                },
                Port {
                    name: "b".into(),
                    ty: PortType::Any,
                    required: false,
                },
            ],
            outputs: vec![],
            connections: vec![],
            blocks: vec![],
            wires: vec![],
        }];
        f.blocks = vec![
            Block {
                id: "in1".into(),
                kind: "lux/input".into(),
                ..Default::default()
            },
            Block {
                id: "sub".into(),
                kind: "subflow".into(),
                subflow: Some("two_ins".into()),
                ..Default::default()
            },
            Block {
                id: "loss".into(),
                kind: "lux/loss".into(),
                ..Default::default()
            },
        ];
        f.wires = vec![
            Wire {
                id: "w1".into(),
                from_block: "in1".into(),
                from_port: "out".into(),
                to_block: "sub".into(),
                to_port: "a".into(),
            },
            Wire {
                id: "w2".into(),
                from_block: "sub".into(),
                from_port: "b".into(),
                to_block: "loss".into(),
                to_port: "prediction".into(),
            },
        ];
        let lib = crate::library::library();
        let err = generate(&f, &lib).unwrap_err();
        assert!(err.contains("exactly one input and one output"), "{err}");
    }
}

#[cfg(test)]
mod julia_parse {
    //! The generated file must at least parse with a real `julia` (running
    //! it needs Lux installed, which is the user's environment).
    use super::tests_support::cnn_flow;

    #[test]
    fn generated_file_parses_in_julia() {
        let Ok(julia) = which_julia() else {
            eprintln!("julia not found; skipping");
            return;
        };
        let lib = crate::library();
        let g = super::generate(&cnn_flow(), &lib).unwrap();
        let dir = std::env::temp_dir().join("moonkale-lux-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.jl");
        std::fs::write(&path, &g.text).unwrap();
        let out = std::process::Command::new(julia)
            .args([
                "-e",
                &format!(
                    "Meta.parseall(read({:?}, String)); println(\"parsed\")",
                    path.display()
                ),
            ])
            .output()
            .unwrap();
        assert!(
            out.status.success() && String::from_utf8_lossy(&out.stdout).contains("parsed"),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn which_julia() -> Result<String, ()> {
        std::process::Command::new("julia")
            .arg("--version")
            .output()
            .map(|_| "julia".to_string())
            .map_err(|_| ())
    }
}

#[cfg(test)]
pub(crate) mod tests_support {
    use moonkale_ext_api::flow::{Block, Flow, Wire};
    use std::collections::BTreeMap;

    pub fn cnn_flow() -> Flow {
        let b = |id: &str, kind: &str, params: &[(&str, &str)]| Block {
            id: id.into(),
            kind: format!("lux/{kind}"),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>(),
            ..Default::default()
        };
        let w = |a: &str, ap: &str, b: &str, bp: &str| Wire {
            id: format!("{a}-{b}"),
            from_block: a.into(),
            from_port: ap.into(),
            to_block: b.into(),
            to_port: bp.into(),
        };
        let mut f = Flow::new();
        f.blocks = vec![
            b("in", "input", &[("shape", "28,28,1")]),
            b("c1", "conv", &[("channels", "8")]),
            b("p1", "maxpool", &[]),
            b("fl", "flatten", &[]),
            b("d1", "dense", &[("out", "10"), ("activation", "identity")]),
            b("l", "loss", &[]),
            b("o", "optimiser", &[]),
        ];
        f.wires = vec![
            w("in", "out", "c1", "in"),
            w("c1", "out", "p1", "in"),
            w("p1", "out", "fl", "in"),
            w("fl", "out", "d1", "in"),
            w("d1", "out", "l", "prediction"),
            w("l", "loss", "o", "loss"),
        ];
        f
    }
}
