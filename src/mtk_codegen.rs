//! Flow → `circuit.jl` for the ModelingToolkit library (ADR-0020's first
//! consumer). Pins are equations: the wires group into **nets** (a pin may
//! carry several wires — a bus is one `connect` with all its ends), every
//! pin must be wired (a floating pin is a broken circuit, refused here —
//! `validate` flags the required ones earlier), and the components become
//! `@named` declarations in flow order. One library per flow for now:
//! a block from another library is an error, not a silent skip.
//!
//! The scaffold simulates `t ∈ (0, 5)` and prints every state at the end
//! (a capacitor's `v`, an inductor's `i`, a mass's `v` — all starting at
//! 0). An RC low-pass with τ = 1 (R = 1 Ω, C = 1 F) reads its capacitor at
//! 99.3 % of the source after those five time constants.

use moonkale_ext_api::flow::{Block, Flow, FlowLibrary, Generated};
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

pub fn generate(flow: &Flow, lib: &FlowLibrary) -> Result<Generated, String> {
    let prefix = format!("{}/", lib.id);
    for b in &flow.blocks {
        if !b.kind.starts_with(&prefix) {
            return Err(format!(
                "{} is not a {} block — one library per flow for now",
                b.kind, lib.name
            ));
        }
    }

    // Julia identifiers: sequential per kind, in flow order, so the golden
    // file is byte-stable (block ids carry uuid suffixes — never emitted).
    let mut counters: BTreeMap<&str, usize> = BTreeMap::new();
    let mut name_of: BTreeMap<&str, String> = BTreeMap::new();
    for b in &flow.blocks {
        let k = kind(&prefix, b);
        let n = counters.entry(k).or_insert(0);
        *n += 1;
        name_of.insert(b.id.as_str(), format!("{}_{}", k, n));
    }

    // The components. A `Voltage` is driven: this standard library's source
    // takes its level on a control input, so each one gets a `Constant`
    // block of its own (the modern API — the old `Voltage(V = …)` is gone).
    let mut drives = Vec::new();
    let mut components = Vec::new();
    let mut systems = Vec::new();
    for b in &flow.blocks {
        let name = &name_of[b.id.as_str()];
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

    // Nets: every wire unions its two ends; every pin of every block must
    // be wired (a floating pin is refused — `validate` flags the required
    // ones first, the codegen is strict about the rest).
    let mut nets = Nets {
        parent: BTreeMap::new(),
    };
    for b in &flow.blocks {
        for pin in pins_of(lib, &b.kind).ok_or_else(|| format!("unknown kind {}", b.kind))? {
            let key = (b.id.clone(), pin);
            nets.parent.insert(key.clone(), key);
        }
    }
    for w in &flow.wires {
        let from_pins = pins_of(
            lib,
            &flow
                .block(&w.from_block)
                .map(|b| b.kind.clone())
                .unwrap_or_default(),
        )
        .unwrap_or_default();
        if !from_pins.iter().any(|p| p == &w.from_port) {
            return Err(format!("{}.{} is not a pin", w.from_block, w.from_port));
        }
        let to_kind = flow
            .block(&w.to_block)
            .map(|b| b.kind.clone())
            .unwrap_or_default();
        let to_pins = pins_of(lib, &to_kind).unwrap_or_default();
        if !to_pins.iter().any(|p| p == &w.to_port) {
            return Err(format!("{}.{} is not a pin", w.to_block, w.to_port));
        }
        nets.union(
            (w.from_block.clone(), w.from_port.clone()),
            (w.to_block.clone(), w.to_port.clone()),
        );
    }
    let mut net_order: Vec<(String, String)> = Vec::new();
    let mut net_members: BTreeMap<(String, String), Vec<(String, String)>> = BTreeMap::new();
    for w in &flow.wires {
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
    // The voltage drives come first, then the nets — deterministic.
    let mut connects = drives;
    for root in &net_order {
        let members = &net_members[root];
        let joined = members
            .iter()
            .map(|(b, p)| format!("{}.{}", name_of[b.as_str()], p))
            .collect::<Vec<_>>()
            .join(", ");
        connects.push(format!("connect({joined})"));
    }
    // A pin in no net (or in a net of one) floats.
    for b in &flow.blocks {
        for pin in pins_of(lib, &b.kind).unwrap_or_default() {
            let root = nets.find((b.id.clone(), pin.clone()));
            let size = net_members.get(&root).map(|m| m.len()).unwrap_or(0);
            if size < 2 {
                return Err(format!("{}.{} is not connected", b.id, pin));
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
    for b in &flow.blocks {
        let name = &name_of[b.id.as_str()];
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
    use moonkale_ext_api::flow::{Block, Wire};

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
}
