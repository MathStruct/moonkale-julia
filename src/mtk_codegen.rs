//! Flow → `circuit.jl` for the ModelingToolkit library (ADR-0020's first
//! consumer). Pins are equations: the wires group into **nets** (a pin may
//! carry several wires — a bus is one `connect` with all its ends), every
//! pin must be wired (a floating pin is a broken circuit, refused here —
//! `validate` flags the required ones earlier), and the components become
//! `@named` declarations in flow order. One library per flow for now:
//! a block from another library is an error, not a silent skip.
//!
//! Subflows flatten (ADR-0021 §6): an instance's inner components join
//! the parent's under per-instance prefixes
//! (`rc_filter_1_resistor_1`, unique per instance — two instances of
//! one subflow cannot collide), and a boundary
//! pin's inner net merges with whatever the parent wires to it — a
//! boundary pin is just a net continuation, so `connect` never cared
//! where a component came from. Nothing generates a nested Julia
//! structure; `structural_simplify` flattens anyway. Only boundary
//! *pins* join nets here — an MTK circuit is acausal, so a declared
//! dataflow port is an error, not a pin.
//!
//! The scaffold simulates `t ∈ (0, 5)` and prints every state at the end
//! (a capacitor's `v`, an inductor's `i`, a mass's `v` — all starting at
//! 0). An RC low-pass with τ = 1 (R = 1 Ω, C = 1 F) reads its capacitor at
//! 99.3 % of the source after those five time constants.

use moonkale_ext_api::flow::{Block, Flow, FlowLibrary, Generated, Wire};

/// The boundary pseudo-block id prefix (ADR-0021 §3): the reserved
/// prefix the panel writes into an inner wire that targets a declared
/// port. ext-api's `flow::BOUNDARY` is the same string; it is still
/// above the tag this crate pins, so the constant is local until the
/// next one.
const BOUNDARY: &str = "boundary/";
use std::collections::BTreeMap;

/// The pins of each block's kind — from the library, not the document.
fn pins_of(lib: &FlowLibrary, kind: &str) -> Option<Vec<String>> {
    lib.blocks
        .iter()
        .find(|k| format!("{}/{}", lib.id, k.id) == kind)
        .map(|k| k.connections.iter().map(|p| p.name.clone()).collect())
}

fn param<'a>(b: &'a Block, name: &str, default: &'a str) -> &'a str {
    b.params.get(name).map(|s| s.as_str()).unwrap_or(default)
}

fn num(b: &Block, name: &str, default: &str) -> Result<String, String> {
    let v = param(b, name, default).trim();
    v.parse::<f64>()
        .ok()
        .filter(|f| f.is_finite())
        .map(|_| v.to_string())
        .ok_or(format!("{}.{} = {v:?} is not a number", b.id, name))
}

/// The block kind without its `mtk/` prefix.
fn kind<'a>(prefix: &str, b: &'a Block) -> &'a str {
    b.kind.strip_prefix(prefix).unwrap_or(&b.kind)
}

/// Union-find over `(block, pin)` endpoints.
struct Nets {
    parent: BTreeMap<(String, String), (String, String)>,
}

impl Nets {
    fn find(&mut self, x: (String, String)) -> (String, String) {
        let mut root = x.clone();
        while let Some(p) = self.parent.get(&root).cloned() {
            if p == root {
                break;
            }
            root = p;
        }
        let mut cur = x;
        while let Some(p) = self.parent.get(&cur).cloned() {
            if p == root {
                break;
            }
            self.parent.insert(cur, root.clone());
            cur = p;
        }
        root
    }
    fn union(&mut self, a: (String, String), b: (String, String)) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent.insert(ra, rb);
        }
    }
}

/// The flattened view the generator works on (ADR-0021 §6): ordinary
/// components from every scope, one wire list with boundary ends
/// remapped onto the instance block's declared pins, Julia names
/// assigned deterministically (per-instance `_` prefixes — the golden
/// file is byte-stable), and each instance's declared pins for the
/// floating checks. Block ids never reach the file.
#[derive(Default)]
struct Flat {
    /// Ordinary component blocks, root and inlined (subflow blocks are
    /// not components — their content is here, prefixed).
    blocks: Vec<Block>,
    /// Every scope's wires, ends remapped: inner ids prefixed, a
    /// boundary end `(boundary/<port>, …)` becomes the instance
    /// block's `(id, port)` — the parent's wire on that port unions
    /// the same net.
    wires: Vec<Wire>,
    /// Flat block id → Julia name (flow order, sequential per kind per
    /// naming scope — an inner component carries its instance's
    /// prefix, `rc_filter_1_resistor_1`).
    names: BTreeMap<String, String>,
    /// Instance block id (flat) → its subflow's declared pins.
    sub_pins: BTreeMap<String, Vec<String>>,
    /// `(instance, pin)` → how many inner wire ends land on that
    /// boundary pin; the rest of the wires on it are the parent's.
    inner_ends: BTreeMap<(String, String), usize>,
}

/// An identifier-safe, lowercase, compact name part.
fn sanitize(name: &str) -> String {
    let s: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.is_empty() {
        "subsystem".into()
    } else {
        s
    }
}

/// Inline one scope (the root's, or an instance's) into `out`. The
/// instance block itself is `host`: its flat id is where inner
/// boundary ends land. `id_prefix`/`name_prefix` carry the nesting
/// path (the root's are empty); `instance_counters` numbers same-named
/// subflows so two instances never collide.
#[allow(clippy::too_many_arguments)]
fn expand(
    flow: &Flow,
    blocks: &[Block],
    wires: &[Wire],
    host: &str,
    id_prefix: &str,
    name_prefix: &str,
    instance_counters: &mut BTreeMap<String, usize>,
    kind_counters: &mut BTreeMap<String, usize>,
    out: &mut Flat,
    prefix: &str,
    depth: usize,
) -> Result<(), String> {
    // `validate` refuses a subflow that contains itself; this bounds
    // the longer cycles too, so a hand-edited file cannot recurse away.
    if depth > 16 {
        return Err(format!("{host} nests subflows too deep (a cycle?)"));
    }
    for b in blocks {
        if let Some(sub_id) = &b.subflow {
            // The instance: named after its subflow, numbered per
            // subflow name so `rc_filter_1` is unique document-wide.
            let sub = flow.subflow(sub_id).ok_or_else(|| {
                format!(
                    "{} references the subflow {sub_id}, which this document does not hold",
                    b.id
                )
            })?;
            let base = sanitize(&sub.name);
            let n = instance_counters.entry(base.clone()).or_insert(0);
            *n += 1;
            let instance = format!("{base}_{n}");
            let flat_id = format!("{id_prefix}{}", b.id);
            out.sub_pins.insert(
                flat_id.clone(),
                sub.connections.iter().map(|p| p.name.clone()).collect(),
            );
            // The per-instance separator is `_`, not MTK's own `₊`:
            // ModelingToolkit parses `₊` as its namespace separator at
            // runtime, so a component named `rc_filter_1₊resistor_1`
            // makes it look for a system `rc_filter_1` that does not
            // exist ("System circuit: variable rc_filter_1 does not
            // exist"). The prefix stays per-instance either way — the
            // collision the ADR guards against.
            let id_prefix = format!("{id_prefix}{instance}_");
            let name_prefix = format!("{name_prefix}{instance}_");
            expand(
                flow,
                &sub.blocks,
                &sub.wires,
                &flat_id,
                &id_prefix,
                &name_prefix,
                instance_counters,
                kind_counters,
                out,
                prefix,
                depth + 1,
            )?;
        } else {
            // An ordinary component: named per kind per naming scope.
            let k = kind(prefix, b);
            let n = kind_counters
                .entry(format!("{name_prefix}{k}"))
                .or_insert(0);
            *n += 1;
            let flat_id = format!("{id_prefix}{}", b.id);
            out.names
                .insert(flat_id.clone(), format!("{name_prefix}{k}_{n}"));
            out.blocks.push(Block {
                id: flat_id,
                kind: b.kind.clone(),
                params: b.params.clone(),
                ..Default::default()
            });
        }
    }
    // This scope's wires, ends remapped. An inner id names a block of
    // this scope (prefixed below) — except a boundary pseudo-block,
    // which becomes the instance block's own pin: `(host, <port>)`, the
    // port read from the pseudo-block's id, and declared on the host.
    let end = |id: &str, port: &str| -> Result<(String, String, bool), String> {
        let Some(pin) = id.strip_prefix(BOUNDARY) else {
            return Ok((format!("{id_prefix}{id}"), port.to_string(), false));
        };
        if host.is_empty() {
            return Err(format!(
                "boundary port {pin} outside a subflow — the root has no boundary"
            ));
        }
        if !out
            .sub_pins
            .get(host)
            .is_some_and(|pins| pins.iter().any(|p| p == pin))
        {
            return Err(format!(
                "boundary pin {pin} of {host} is not a declared pin"
            ));
        }
        Ok((host.to_string(), pin.to_string(), true))
    };
    let mut remapped = Vec::new();
    for w in wires {
        let (from_block, from_port, from_inner) = end(&w.from_block, &w.from_port)?;
        let (to_block, to_port, to_inner) = end(&w.to_block, &w.to_port)?;
        remapped.push((
            Wire {
                id: w.id.clone(),
                from_block,
                from_port,
                to_block,
                to_port,
            },
            from_inner,
            to_inner,
        ));
    }
    for (w, from_inner, to_inner) in remapped {
        if from_inner {
            *out.inner_ends
                .entry((w.from_block.clone(), w.from_port.clone()))
                .or_insert(0) += 1;
        }
        if to_inner {
            *out.inner_ends
                .entry((w.to_block.clone(), w.to_port.clone()))
                .or_insert(0) += 1;
        }
        out.wires.push(w);
    }
    Ok(())
}

/// Flatten the flow (ADR-0021 §6): every subflow instance inlined.
fn flatten(flow: &Flow, prefix: &str) -> Result<Flat, String> {
    let mut out = Flat::default();
    let mut instance_counters: BTreeMap<String, usize> = BTreeMap::new();
    let mut kind_counters: BTreeMap<String, usize> = BTreeMap::new();
    expand(
        flow,
        &flow.blocks,
        &flow.wires,
        "",
        "",
        "",
        &mut instance_counters,
        &mut kind_counters,
        &mut out,
        prefix,
        0,
    )?;
    Ok(out)
}

pub fn generate(flow: &Flow, lib: &FlowLibrary) -> Result<Generated, String> {
    let prefix = format!("{}/", lib.id);

    // Subflow instances flatten first (ADR-0021 §6): the generator below
    // sees one flat component list, as if the circuit was drawn open —
    // so the one-library rule covers every scope, a subflow's inside
    // included (a subflow block itself is not a component).
    let flat = flatten(flow, &prefix)?;
    for b in &flat.blocks {
        if !b.kind.starts_with(&prefix) {
            return Err(format!(
                "{} is not a {} block — one library per flow for now",
                b.kind, lib.name
            ));
        }
    }
    let name_of = &flat.names;

    // The components. A `Voltage` is driven: this standard library's source
    // takes its level on a control input, so each one gets a `Constant`
    // block of its own (the modern API — the old `Voltage(V = …)` is gone).
    let mut drives = Vec::new();
    let mut components = Vec::new();
    let mut systems = Vec::new();
    for b in &flat.blocks {
        let name = &name_of[&b.id];
        systems.push(name.clone());
        let line = match kind(&prefix, b) {
            "resistor" => format!("{name} = Resistor(R = {})", num(b, "R", "1.0")?),
            "capacitor" => format!("{name} = Capacitor(C = {})", num(b, "C", "1e-3")?),
            "inductor" => format!("{name} = Inductor(L = {})", num(b, "L", "1e-3")?),
            "voltage" => {
                let v = num(b, "V", "1.0")?;
                components.push(format!("@named {name}_level = Blocks.Constant(k = {v})"));
                systems.push(format!("{name}_level"));
                drives.push(format!("connect({name}.V, {name}_level.output)"));
                format!("{name} = Voltage()")
            }
            "ground" => format!("{name} = Ground()"),
            "mass" => format!("{name} = Mass(m = {})", num(b, "m", "1.0")?),
            "spring" => format!("{name} = Spring(k = {})", num(b, "k", "100.0")?),
            "damper" => format!("{name} = Damper(d = {})", num(b, "d", "10.0")?),
            "fixed" => format!("{name} = Fixed()"),
            other => return Err(format!("unknown {other} component")),
        };
        components.push(format!("@named {line}"));
    }

    // Nets: every wire unions its two ends; every pin of every block —
    // and every declared boundary pin of every instance — must be wired
    // (a floating pin is refused — `validate` flags the required ones
    // first, the codegen is strict about the rest).
    let mut nets = Nets {
        parent: BTreeMap::new(),
    };
    for b in &flat.blocks {
        for pin in pins_of(lib, &b.kind).ok_or_else(|| format!("unknown kind {}", b.kind))? {
            let key = (b.id.clone(), pin);
            nets.parent.insert(key.clone(), key);
        }
    }
    for (id, pins) in &flat.sub_pins {
        for pin in pins {
            let key = (id.clone(), pin.clone());
            nets.parent.insert(key.clone(), key);
        }
    }
    // A wire end is a flat block's pin or an instance's declared pin.
    let end_is_pin = |block: &str, port: &str| -> bool {
        flat.blocks
            .iter()
            .find(|b| b.id == block)
            .and_then(|b| pins_of(lib, &b.kind))
            .is_some_and(|pins| pins.iter().any(|p| p == port))
            || flat
                .sub_pins
                .get(block)
                .is_some_and(|pins| pins.iter().any(|p| p == port))
    };
    for w in &flat.wires {
        if !end_is_pin(&w.from_block, &w.from_port) {
            return Err(format!("{}.{} is not a pin", w.from_block, w.from_port));
        }
        if !end_is_pin(&w.to_block, &w.to_port) {
            return Err(format!("{}.{} is not a pin", w.to_block, w.to_port));
        }
        nets.union(
            (w.from_block.clone(), w.from_port.clone()),
            (w.to_block.clone(), w.to_port.clone()),
        );
    }
    let mut net_order: Vec<(String, String)> = Vec::new();
    let mut net_members: BTreeMap<(String, String), Vec<(String, String)>> = BTreeMap::new();
    for w in &flat.wires {
        let root = nets.find((w.from_block.clone(), w.from_port.clone()));
        if !net_order.contains(&root) {
            net_order.push(root.clone());
            net_members.insert(root.clone(), Vec::new());
        }
        for end in [
            (w.from_block.clone(), w.from_port.clone()),
            (w.to_block.clone(), w.to_port.clone()),
        ] {
            let members = net_members.get_mut(&root).unwrap();
            if !members.contains(&end) {
                members.push(end);
            }
        }
    }
    // The voltage drives come first, then the nets — deterministic. A
    // boundary end is a net continuation, not a Julia pin: the parent
    // side and the inner side join in one connect, the boundary itself
    // never appears (nothing generates a nested Julia structure).
    let mut connects = drives;
    for root in &net_order {
        let members = &net_members[root];
        let joined = members
            .iter()
            .filter(|(b, _)| !flat.sub_pins.contains_key(b))
            .map(|(b, p)| format!("{}.{}", name_of[b.as_str()], p))
            .collect::<Vec<_>>()
            .join(", ");
        if !joined.is_empty() {
            connects.push(format!("connect({joined})"));
        }
    }
    // A pin in no net (or in a net of one) floats — an instance's
    // declared pin needs both sides: a wire inside *and* a wire of the
    // parent (a boundary pin is a net continuation, not a dead end).
    for b in &flat.blocks {
        for pin in pins_of(lib, &b.kind).unwrap_or_default() {
            let root = nets.find((b.id.clone(), pin.clone()));
            let size = net_members.get(&root).map(|m| m.len()).unwrap_or(0);
            if size < 2 {
                return Err(format!("{}.{} is not connected", b.id, pin));
            }
        }
    }
    for (id, pins) in &flat.sub_pins {
        for pin in pins {
            // Both sides, counted at the boundary itself: at least one
            // wire from inside (a remapped inner wire) and one of the
            // parent. A bus is fine — more than one on either side.
            let all = flat
                .wires
                .iter()
                .filter(|w| {
                    (w.from_block == *id && w.from_port == *pin)
                        || (w.to_block == *id && w.to_port == *pin)
                })
                .count();
            let inner = flat
                .inner_ends
                .get(&(id.clone(), pin.clone()))
                .copied()
                .unwrap_or(0);
            if inner == 0 || all == inner {
                return Err(format!(
                    "boundary pin {pin} of {id} is not connected on both sides"
                ));
            }
        }
    }
    if connects.is_empty() {
        return Err("the circuit has no connections".into());
    }

    // The state and its initial conditions: a capacitor's v (0), an
    // inductor's i (0), a mass's s and v (s from its s0, v at 0), a
    // spring's elongation delta_s (from its s0 — a released mass needs
    // it consistent with the mass's s0, or the initialization is
    // underdetermined and the system does not move). A damper has no
    // state of its own.
    let mut u0 = Vec::new();
    let mut prints = Vec::new();
    let sample = |name: &str, state: &str, prints: &mut Vec<String>| {
        for t in ["0.5", "1.0", "5.0"] {
            prints.push(format!(
                "println(\"{name}.{state} at t={t}: \", sol({t}; idxs = {name}.{state}))"
            ));
        }
    };
    for b in &flat.blocks {
        let name = &name_of[&b.id];
        match kind(&prefix, b) {
            "capacitor" => {
                u0.push(format!("{name}.v => 0.0"));
                sample(name, "v", &mut prints);
            }
            "inductor" => {
                u0.push(format!("{name}.i => 0.0"));
                sample(name, "i", &mut prints);
            }
            "mass" => {
                u0.push(format!("{name}.s => {}", num(b, "s0", "0.0")?));
                u0.push(format!("{name}.v => 0.0"));
                sample(name, "s", &mut prints);
                sample(name, "v", &mut prints);
            }
            "spring" => {
                u0.push(format!("{name}.delta_s => {}", num(b, "s0", "0.0")?));
            }
            _ => {}
        }
    }
    // The @named components and their level drivers, as subsystems —
    // connect() needs them on the system it builds onto (flow order).
    let systems = systems.join(", ");

    if u0.is_empty() {
        return Err("the circuit has no state — add a Capacitor, Inductor or Mass".into());
    }

    let text = format!(
        "# Generated by Moonkale's {name} library — a .flow.json's circuit.\n\
         # Run: julia circuit.jl (needs ModelingToolkit, its StandardLibrary and OrdinaryDiffEq).\n\
         using ModelingToolkit\n\
         using ModelingToolkitStandardLibrary.Electrical\n\
         using ModelingToolkitStandardLibrary.Mechanical.Translational\n\
         using ModelingToolkitStandardLibrary.Blocks\n\
         using OrdinaryDiffEq\n\
         \n\
         @independent_variables t\n\
         \n\
         {}\n\
         \n\
         @named circuit = ODESystem([\n\
         {}\n\
         ], t; systems = [{}])\n\
         circuit = structural_simplify(circuit)\n\
         \n\
         prob = ODEProblem(circuit, [{}], (0.0, 5.0))\n\
         sol = solve(prob, Tsit5())\n\
         {}\n",
        components.join("\n"),
        connects
            .iter()
            .map(|c| format!("    {c},"))
            .collect::<Vec<_>>()
            .join("\n"),
        systems,
        u0.join(", "),
        prints.join("\n"),
        name = lib.name,
    );
    Ok(Generated {
        file_name: "circuit.jl".into(),
        text,
        run_hint: "julia circuit.jl".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use moonkale_ext_api::flow::{Block, Port, PortType, Subflow, Wire};

    pub(super) fn rc() -> (Flow, FlowLibrary) {
        let mut f = Flow::new();
        f.blocks = vec![
            Block {
                id: "src-aa".into(),
                kind: "mtk/voltage".into(),
                x: 0.0,
                params: [("V".to_string(), "1.0".to_string())].into_iter().collect(),
                ..Default::default()
            },
            Block {
                id: "res-bb".into(),
                kind: "mtk/resistor".into(),
                x: 1.0,
                params: [("R".to_string(), "1.0".to_string())].into_iter().collect(),
                ..Default::default()
            },
            Block {
                id: "cap-cc".into(),
                kind: "mtk/capacitor".into(),
                x: 2.0,
                params: [("C".to_string(), "1.0".to_string())].into_iter().collect(),
                ..Default::default()
            },
            Block {
                id: "gnd-dd".into(),
                kind: "mtk/ground".into(),
                x: 3.0,
                ..Default::default()
            },
        ];
        f.wires = vec![
            Wire {
                id: "w1".into(),
                from_block: "src-aa".into(),
                from_port: "p".into(),
                to_block: "res-bb".into(),
                to_port: "p".into(),
            },
            Wire {
                id: "w2".into(),
                from_block: "res-bb".into(),
                from_port: "n".into(),
                to_block: "cap-cc".into(),
                to_port: "p".into(),
            },
            Wire {
                id: "w3".into(),
                from_block: "cap-cc".into(),
                from_port: "n".into(),
                to_block: "src-aa".into(),
                to_port: "n".into(),
            },
            // The ground rides the capacitor's negative pin — a bus.
            Wire {
                id: "w4".into(),
                from_block: "cap-cc".into(),
                from_port: "n".into(),
                to_block: "gnd-dd".into(),
                to_port: "g".into(),
            },
        ];
        (f, crate::mtk::library())
    }

    #[test]
    fn an_rc_low_pass_generates_named_components_and_grouped_nets() {
        let (f, lib) = rc();
        let g = generate(&f, &lib).expect("generates");
        assert_eq!(g.file_name, "circuit.jl");
        assert_eq!(g.run_hint, "julia circuit.jl");
        assert!(g
            .text
            .contains("@named voltage_1_level = Blocks.Constant(k = 1.0)"));
        assert!(g.text.contains("@named voltage_1 = Voltage()"));
        assert!(g.text.contains("@named resistor_1 = Resistor(R = 1.0)"));
        assert!(g.text.contains("@named capacitor_1 = Capacitor(C = 1.0)"));
        assert!(g.text.contains("@named ground_1 = Ground()"));
        // The drive first, then the three nets; the ground's is the bus.
        assert!(g
            .text
            .contains("connect(voltage_1.V, voltage_1_level.output),"));
        assert!(g.text.contains("connect(voltage_1.p, resistor_1.p),"));
        assert!(g.text.contains("connect(resistor_1.n, capacitor_1.p),"));
        assert!(g
            .text
            .contains("connect(capacitor_1.n, voltage_1.n, ground_1.g),"));
        // The components ride along as subsystems, in flow order.
        assert!(g.text.contains(
            "], t; systems = [voltage_1, voltage_1_level, resistor_1, capacitor_1, ground_1])"
        ));
        // The state and the readout.
        assert!(g
            .text
            .contains("prob = ODEProblem(circuit, [capacitor_1.v => 0.0], (0.0, 5.0))"));
        assert!(g.text.contains("sol(5.0; idxs = capacitor_1.v)"));
        // Deterministic: the block ids (with their uuid suffixes) never appear.
        assert!(!g.text.contains("src-aa"));
    }

    #[test]
    fn a_floating_pin_and_a_foreign_block_are_refused() {
        let (f, lib) = rc();
        let mut broken = f.clone();
        broken.wires.pop(); // the ground floats off the bus
        let err = generate(&broken, &lib).unwrap_err();
        assert_eq!(err, "gnd-dd.g is not connected");

        let mut mixed = f.clone();
        mixed.blocks.push(Block {
            id: "l".into(),
            kind: "lux/dense".into(),
            ..Default::default()
        });
        let err = generate(&mixed, &lib).unwrap_err();
        assert!(err.contains("one library per flow"), "{err}");

        // A wire into a non-pin is refused, not silently drawn.
        let mut bad = f.clone();
        bad.wires.push(Wire {
            id: "w5".into(),
            from_block: "src-aa".into(),
            from_port: "nowhere".into(),
            to_block: "res-bb".into(),
            to_port: "p".into(),
        });
        assert!(generate(&bad, &lib).unwrap_err().contains("is not a pin"));
    }

    #[test]
    fn a_released_mass_oscillates_in_the_generated_problem() {
        // RC + a spring-mass in one circuit: the Translational import, the
        // flange pins, and the initial conditions that make the mass move —
        // s from the mass's s0, delta_s from the spring's (Claude's #1-3).
        let (f, lib) = rc();
        let mut mixed = f;
        mixed.blocks.push(Block {
            id: "wall".into(),
            kind: "mtk/fixed".into(),
            x: 4.0,
            ..Default::default()
        });
        mixed.blocks.push(Block {
            id: "spr".into(),
            kind: "mtk/spring".into(),
            x: 5.0,
            params: [
                ("k".to_string(), "100.0".to_string()),
                ("s0".to_string(), "0.1".to_string()),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        });
        mixed.blocks.push(Block {
            id: "ms".into(),
            kind: "mtk/mass".into(),
            x: 6.0,
            params: [
                ("m".to_string(), "1.0".to_string()),
                ("s0".to_string(), "0.1".to_string()),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        });
        mixed.wires.push(Wire {
            id: "w5".into(),
            from_block: "wall".into(),
            from_port: "flange".into(),
            to_block: "spr".into(),
            to_port: "flange_a".into(),
        });
        mixed.wires.push(Wire {
            id: "w6".into(),
            from_block: "spr".into(),
            from_port: "flange_b".into(),
            to_block: "ms".into(),
            to_port: "flange".into(),
        });
        let g = generate(&mixed, &lib).unwrap();
        assert!(g
            .text
            .contains("using ModelingToolkitStandardLibrary.Mechanical.Translational"));
        assert!(g
            .text
            .contains("connect(fixed_1.flange, spring_1.flange_a),"));
        assert!(g
            .text
            .contains("connect(spring_1.flange_b, mass_1.flange),"));
        // The initial conditions: the mass's position and the spring's
        // elongation from their s0 — without both, MTK leaves the mass still.
        assert!(g.text.contains("prob = ODEProblem(circuit, [capacitor_1.v => 0.0, spring_1.delta_s => 0.1, mass_1.s => 0.1, mass_1.v => 0.0], (0.0, 5.0))"));
        assert!(g.text.contains("sol(0.5; idxs = mass_1.v)"));
    }

    #[test]
    fn non_finite_parameters_are_refused() {
        // Rust's f64 parse accepts inf and NaN; Julia must not see them
        // (the MTK review's #4).
        let (f, lib) = rc();
        let mut bad = f;
        bad.blocks[1]
            .params
            .insert("R".to_string(), "inf".to_string());
        let err = generate(&bad, &lib).unwrap_err();
        assert!(err.contains("is not a number"), "{err}");
        bad.blocks[1]
            .params
            .insert("R".to_string(), "NaN".to_string());
        let err = generate(&bad, &lib).unwrap_err();
        assert!(err.contains("is not a number"), "{err}");
    }

    #[test]
    fn a_stateless_circuit_is_refused() {
        let (f, lib) = rc();
        let mut r_only = f;
        // Only the source and the resistor: every pin wired, no state.
        r_only
            .blocks
            .retain(|b| b.kind == "mtk/voltage" || b.kind == "mtk/resistor");
        r_only.wires = vec![
            Wire {
                id: "w1".into(),
                from_block: "src-aa".into(),
                from_port: "p".into(),
                to_block: "res-bb".into(),
                to_port: "p".into(),
            },
            Wire {
                id: "w6".into(),
                from_block: "res-bb".into(),
                from_port: "n".into(),
                to_block: "src-aa".into(),
                to_port: "n".into(),
            },
        ];
        let err = generate(&r_only, &lib).unwrap_err();
        assert!(err.contains("no state"), "{err}");
    }

    // ---- ADR-0021 stage C: flattening -----------------------------

    /// The RC filter as a subsystem: R and C inside `rc_filter`, the
    /// source and the ground outside, joined through two boundary pins.
    fn rc_subsystem() -> (Flow, FlowLibrary) {
        let (_, lib) = rc();
        let mut f = Flow::new();
        let pin = |n: &str| Port {
            name: n.into(),
            ty: PortType::Named {
                name: "electrical".into(),
            },
            required: false,
        };
        let b = |id: &str, kind: &str, params: &[(&str, &str)]| Block {
            id: id.into(),
            kind: format!("mtk/{kind}"),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..Default::default()
        };
        let w = |id: &str, a: &str, ap: &str, bb: &str, bp: &str| Wire {
            id: id.into(),
            from_block: a.into(),
            from_port: ap.into(),
            to_block: bb.into(),
            to_port: bp.into(),
        };
        f.subflows = vec![Subflow {
            id: "rc_filter".into(),
            name: "RC filter".into(),
            inputs: vec![],
            outputs: vec![],
            connections: vec![pin("in"), pin("out")],
            blocks: vec![
                b("res-x", "resistor", &[("R", "1.0")]),
                b("cap-x", "capacitor", &[("C", "1.0")]),
            ],
            wires: vec![
                w("iw1", "boundary/in", "in", "res-x", "p"),
                w("iw2", "res-x", "n", "cap-x", "p"),
                w("iw3", "cap-x", "n", "boundary/out", "out"),
            ],
        }];
        f.blocks = vec![
            b("src-aa", "voltage", &[("V", "1.0")]),
            Block {
                id: "rc-1".into(),
                kind: "subflow".into(),
                subflow: Some("rc_filter".into()),
                ..Default::default()
            },
            b("gnd-bb", "ground", &[]),
        ];
        f.wires = vec![
            w("w1", "src-aa", "p", "rc-1", "in"),
            w("w2", "rc-1", "out", "gnd-bb", "g"),
            w("w3", "rc-1", "out", "src-aa", "n"),
        ];
        (f, lib)
    }

    #[test]
    fn an_rc_filter_in_a_subsystem_flattens_into_one_circuit() {
        let (f, lib) = rc_subsystem();
        let out = generate(&f, &lib).expect("flattens");
        // The inner components join the parent's under per-instance
        // prefixes (ADR-0021 §6, `_` not `₊`).
        assert!(
            out.text
                .contains("@named rc_filter_1_resistor_1 = Resistor(R = 1.0)"),
            "{}",
            out.text
        );
        assert!(
            out.text
                .contains("@named rc_filter_1_capacitor_1 = Capacitor(C = 1.0)"),
            "{}",
            out.text
        );
        // The boundary never appears: the source's pin connects straight
        // to the inner resistor, the net continues (the flatten merged
        // the nets through both boundary pins).
        assert!(
            out.text
                .contains("connect(rc_filter_1_resistor_1.p, voltage_1.p)"),
            "{}",
            out.text
        );
        assert!(
            out.text
                .contains("connect(rc_filter_1_capacitor_1.n, ground_1.g, voltage_1.n)"),
            "{}",
            out.text
        );
        assert!(
            out.text.contains("rc_filter_1_capacitor_1.v => 0.0"),
            "the inner state carries the instance prefix: {}",
            out.text
        );
    }

    #[test]
    fn two_instances_of_one_subflow_keep_their_names_apart() {
        let (f, lib) = rc_subsystem();
        let mut f = f;
        // A second instance of the same subflow: a divider of two halves.
        f.subflows.push(Subflow {
            id: "half".into(),
            name: "Half".into(),
            inputs: vec![],
            outputs: vec![],
            connections: vec![
                Port {
                    name: "in".into(),
                    ty: PortType::Named {
                        name: "electrical".into(),
                    },
                    required: false,
                },
                Port {
                    name: "out".into(),
                    ty: PortType::Named {
                        name: "electrical".into(),
                    },
                    required: false,
                },
            ],
            blocks: vec![Block {
                id: "res-y".into(),
                kind: "mtk/resistor".into(),
                params: [("R".to_string(), "1.0".to_string())].into_iter().collect(),
                ..Default::default()
            }],
            wires: vec![
                Wire {
                    id: "hw1".into(),
                    from_block: "boundary/in".into(),
                    from_port: "in".into(),
                    to_block: "res-y".into(),
                    to_port: "p".into(),
                },
                Wire {
                    id: "hw2".into(),
                    from_block: "res-y".into(),
                    from_port: "n".into(),
                    to_block: "boundary/out".into(),
                    to_port: "out".into(),
                },
            ],
        });
        f.blocks.push(Block {
            id: "rc-2".into(),
            kind: "subflow".into(),
            subflow: Some("half".into()),
            ..Default::default()
        });
        f.blocks.push(Block {
            id: "rc-3".into(),
            kind: "subflow".into(),
            subflow: Some("half".into()),
            ..Default::default()
        });
        // Wire the two halves in series between the first subsystem's
        // out and the ground.
        f.wires.retain(|w| w.id != "w2");
        f.wires.push(Wire {
            id: "w4".into(),
            from_block: "rc-1".into(),
            from_port: "out".into(),
            to_block: "rc-2".into(),
            to_port: "in".into(),
        });
        f.wires.push(Wire {
            id: "w5".into(),
            from_block: "rc-2".into(),
            from_port: "out".into(),
            to_block: "rc-3".into(),
            to_port: "in".into(),
        });
        f.wires.push(Wire {
            id: "w6".into(),
            from_block: "rc-3".into(),
            from_port: "out".into(),
            to_block: "gnd-bb".into(),
            to_port: "g".into(),
        });
        let out = generate(&f, &lib).expect("flattens");
        assert!(out.text.contains("half_1_resistor_1"), "{}", out.text);
        assert!(out.text.contains("half_2_resistor_1"), "{}", out.text);
        assert!(
            out.text
                .contains("connect(half_1_resistor_1.n, half_2_resistor_1.p)"),
            "the two instances chain without colliding: {}",
            out.text
        );
    }

    #[test]
    fn a_boundary_pin_without_both_sides_is_refused() {
        let (f, lib) = rc_subsystem();
        let mut f = f;
        // Nothing inside feeds the boundary out (the capacitor returns
        // into an inner ground instead): the parent's wire on the
        // boundary is a dead end, and every other pin stays wired, so
        // the one-sided boundary is the error that surfaces.
        f.subflows[0].blocks.push(Block {
            id: "gnd-in".into(),
            kind: "mtk/ground".into(),
            ..Default::default()
        });
        f.subflows[0].wires[2] = Wire {
            id: "iw3".into(),
            from_block: "cap-x".into(),
            from_port: "n".into(),
            to_block: "gnd-in".into(),
            to_port: "g".into(),
        };
        // The source's return goes straight to the parent's ground —
        // only one wire touches the boundary out.
        f.wires.retain(|w| w.id != "w3");
        f.wires.push(Wire {
            id: "w7".into(),
            from_block: "src-aa".into(),
            from_port: "n".into(),
            to_block: "gnd-bb".into(),
            to_port: "g".into(),
        });
        let err = generate(&f, &lib).unwrap_err();
        assert!(err.contains("not connected on both sides"), "{err}");
        // A declared dataflow input is not a pin in an acausal circuit.
        f.subflows[0].inputs = vec![Port {
            name: "x".into(),
            ty: PortType::Any,
            required: false,
        }];
        f.wires.push(Wire {
            id: "w9".into(),
            from_block: "src-aa".into(),
            from_port: "p".into(),
            to_block: "rc-1".into(),
            to_port: "x".into(),
        });
        let err = generate(&f, &lib).unwrap_err();
        assert!(err.contains("is not a pin"), "{err}");
    }

    #[test]
    fn a_nested_subsystem_flattens_through_both_boundaries() {
        // `rc_filter` inside `stage`: the source reaches the inner
        // resistor through two boundaries, the names nest.
        let (mut f, lib) = rc_subsystem();
        let pins = f.subflows[0].connections.clone();
        f.subflows.push(Subflow {
            id: "stage".into(),
            name: "Stage".into(),
            inputs: vec![],
            outputs: vec![],
            connections: pins,
            blocks: vec![Block {
                id: "inner-rc".into(),
                kind: "subflow".into(),
                subflow: Some("rc_filter".into()),
                ..Default::default()
            }],
            wires: vec![
                Wire {
                    id: "sw1".into(),
                    from_block: "boundary/in".into(),
                    from_port: "in".into(),
                    to_block: "inner-rc".into(),
                    to_port: "in".into(),
                },
                Wire {
                    id: "sw2".into(),
                    from_block: "inner-rc".into(),
                    from_port: "out".into(),
                    to_block: "boundary/out".into(),
                    to_port: "out".into(),
                },
            ],
        });
        f.blocks[1].subflow = Some("stage".into());
        let out = generate(&f, &lib).expect("flattens");
        assert!(
            out.text
                .contains("connect(stage_1_rc_filter_1_resistor_1.p, voltage_1.p)"),
            "{}",
            out.text
        );
        assert!(
            out.text
                .contains("connect(stage_1_rc_filter_1_capacitor_1.n, ground_1.g, voltage_1.n)"),
            "{}",
            out.text
        );
    }

    #[test]
    fn the_inside_of_a_subsystem_obeys_the_same_rules() {
        // A foreign block inside is the one-library error, as outside.
        let (mut f, lib) = rc_subsystem();
        f.subflows[0].blocks[0].kind = "lux/dense".into();
        let err = generate(&f, &lib).unwrap_err();
        assert!(err.contains("one library per flow"), "{err}");
        // An inner wire to a boundary pin the subflow does not declare.
        let (mut f, lib) = rc_subsystem();
        f.subflows[0].wires[2].to_block = "boundary/nope".into();
        let err = generate(&f, &lib).unwrap_err();
        assert!(err.contains("not a declared pin"), "{err}");
        // Two inner wires on a boundary pin and none of the parent's is
        // still one-sided: the count tells the sides apart.
        let (mut f, lib) = rc_subsystem();
        f.subflows[0].wires.push(Wire {
            id: "iw4".into(),
            from_block: "res-x".into(),
            from_port: "p".into(),
            to_block: "boundary/in".into(),
            to_port: "in".into(),
        });
        f.wires.retain(|w| w.id != "w1");
        f.wires.push(Wire {
            id: "w8".into(),
            from_block: "src-aa".into(),
            from_port: "p".into(),
            to_block: "gnd-bb".into(),
            to_port: "g".into(),
        });
        let err = generate(&f, &lib).unwrap_err();
        assert!(err.contains("not connected on both sides"), "{err}");
    }
}
