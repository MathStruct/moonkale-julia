//! The ModelingToolkit block kinds (ADR-0020's first consumer): acausal
//! components — electrical (Resistor, Capacitor, Inductor, Voltage, Ground)
//! and mechanical (Mass, Spring, Damper, Fixed) — whose pins are
//! **domain-typed** (`electrical`, `mechanical`), so a wire between domains
//! is refused before any code exists. Parameters carry their units in the
//! description; the generated file uses plain numbers.
//!
//! Pins are undirected: `connect` is symmetric, buses are legal (a pin may
//! carry several wires), and loops are equations, not cycles.

use moonkale_ext_api::flow::{
    Anchor, BlockKind, FlowLibrary, Param, ParamKind, Port, PortType, Side, Symbol,
};

fn pin(name: &str, domain: &str) -> Port {
    Port {
        name: name.into(),
        ty: PortType::Named {
            name: domain.into(),
        },
        required: true,
    }
}

/// A float parameter. The unit lives in the block description (R in Ω);
/// the generated literal is a plain number.
fn p(name: &str, default: &str) -> Param {
    Param {
        name: name.into(),
        kind: ParamKind::Float,
        default: default.into(),
    }
}

/// A schematic symbol (Moonkale ADR-0024): SVG path data in a
/// `width × height` box, and each pin's side. A pin sits at its side's
/// middle unless `offset` says otherwise.
fn symbol(width: f32, height: f32, path: &str, anchors: &[(&str, Side, f32)]) -> Symbol {
    Symbol {
        width,
        height,
        path: path.into(),
        anchors: anchors
            .iter()
            .map(|&(port, side, offset)| Anchor {
                port: port.into(),
                side,
                offset,
            })
            .collect(),
    }
}

/// A two-terminal part, 80 × 32: its leads run along the middle to a
/// pin on each side, so it lies in its circuit's line.
fn two_terminal(path: &str, a: &str, b: &str) -> Symbol {
    symbol(
        80.0,
        32.0,
        path,
        &[(a, Side::Left, 0.5), (b, Side::Right, 0.5)],
    )
}

/// An acausal component: no inputs, no outputs, only pins.
fn acausal(
    id: &str,
    name: &str,
    category: &str,
    description: &str,
    pins: Vec<Port>,
    params: Vec<Param>,
    drawing: Symbol,
) -> BlockKind {
    BlockKind {
        id: id.into(),
        name: name.into(),
        category: category.into(),
        description: description.into(),
        inputs: vec![],
        connections: pins,
        params,
        symbol: Some(drawing),
        // `outputs` (and any future field) comes from Default — the pins
        // review's #2 keeps this literal non-breaking.
        ..Default::default()
    }
}

pub fn library() -> FlowLibrary {
    FlowLibrary {
        id: "mtk",
        name: "ModelingToolkit",
        language: "julia",
        codegen: Some(crate::mtk_codegen::generate),
        blocks: vec![
            acausal(
                "resistor",
                "Resistor",
                "electrical",
                "A resistor: R in Ω.",
                vec![pin("p", "electrical"), pin("n", "electrical")],
                vec![p("R", "1.0")],
                // DIN/IEC: a box between two leads.
                two_terminal("M 0 16 H 20 M 20 9 H 60 V 23 H 20 Z M 60 16 H 80", "p", "n"),
            ),
            acausal(
                "capacitor",
                "Capacitor",
                "electrical",
                "A capacitor: C in F; its voltage v is a state, starting at 0.",
                vec![pin("p", "electrical"), pin("n", "electrical")],
                vec![p("C", "1e-3")],
                // Two plates.
                two_terminal("M 0 16 H 35 M 35 4 V 28 M 45 4 V 28 M 45 16 H 80", "p", "n"),
            ),
            acausal(
                "inductor",
                "Inductor",
                "electrical",
                "An inductor: L in H; its current i is a state, starting at 0.",
                vec![pin("p", "electrical"), pin("n", "electrical")],
                vec![p("L", "1e-3")],
                // Four turns.
                two_terminal(
                    "M 0 16 H 16 A 6 6 0 0 1 28 16 A 6 6 0 0 1 40 16 A 6 6 0 0 1 52 16 A 6 6 0 0 1 64 16 H 80",
                    "p",
                    "n",
                ),
            ),
            acausal(
                "voltage",
                "Voltage",
                "electrical",
                "A constant voltage source: V in V.",
                vec![pin("p", "electrical"), pin("n", "electrical")],
                vec![p("V", "1.0")],
                // A circle, + on the p side.
                two_terminal(
                    "M 0 16 H 28 M 52 16 H 80 M 52 16 A 12 12 0 1 1 28 16 A 12 12 0 1 1 52 16 M 31 16 H 37 M 34 13 V 19 M 43 16 H 49",
                    "p",
                    "n",
                ),
            ),
            acausal(
                "ground",
                "Ground",
                "electrical",
                "The electrical ground — wire it into one net of the circuit.",
                vec![pin("g", "electrical")],
                vec![],
                // Three shrinking bars under its pin.
                symbol(
                    32.0,
                    28.0,
                    "M 16 0 V 12 M 4 12 H 28 M 9 18 H 23 M 13 24 H 19",
                    &[("g", Side::Top, 0.5)],
                ),
            ),
            acausal(
                "mass",
                "Mass",
                "mechanical",
                "A translating mass: m in kg. Its position s and velocity v are states — s starts at s0 (m), v at 0.",
                vec![pin("flange", "mechanical")],
                vec![p("m", "1.0"), p("s0", "0.0")],
                // A cart on two wheels, its flange on the left.
                symbol(
                    64.0,
                    34.0,
                    "M 0 14 H 16 M 16 4 H 60 V 24 H 16 Z M 29 28 A 3 3 0 1 1 23 28 A 3 3 0 1 1 29 28 M 53 28 A 3 3 0 1 1 47 28 A 3 3 0 1 1 53 28",
                    &[("flange", Side::Left, 14.0 / 34.0)],
                ),
            ),
            acausal(
                "spring",
                "Spring",
                "mechanical",
                "A spring: k in N/m. Its elongation delta_s is a state, starting at s0 (m) — for a released mass, set it to the mass's s0.",
                vec![pin("flange_a", "mechanical"), pin("flange_b", "mechanical")],
                vec![p("k", "100.0"), p("s0", "0.0")],
                // A zigzag between its flanges.
                two_terminal(
                    "M 0 16 H 14 L 18 5 L 26 27 L 34 5 L 42 27 L 50 5 L 58 27 L 62 16 H 80",
                    "flange_a",
                    "flange_b",
                ),
            ),
            acausal(
                "damper",
                "Damper",
                "mechanical",
                "A damper: d in N·s/m. No state of its own — it needs no initial condition.",
                vec![pin("flange_a", "mechanical"), pin("flange_b", "mechanical")],
                vec![p("d", "10.0")],
                // A dashpot: a piston in an open cylinder.
                two_terminal(
                    "M 0 16 H 36 M 36 9 V 23 M 26 5 H 52 V 27 H 26 M 52 16 H 80",
                    "flange_a",
                    "flange_b",
                ),
            ),
            acausal(
                "fixed",
                "Fixed",
                "mechanical",
                "A fixed wall for the mechanical domain, like Ground for the electrical.",
                vec![pin("flange", "mechanical")],
                vec![],
                // A hatched wall, its flange on the right.
                symbol(
                    28.0,
                    36.0,
                    "M 14 4 V 32 M 14 6 L 7 13 M 14 13 L 7 20 M 14 20 L 7 27 M 14 27 L 7 34 M 14 18 H 28",
                    &[("flange", Side::Right, 0.5)],
                ),
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_component_draws_a_symbol_that_anchors_all_its_pins() {
        for k in library().blocks {
            let s = k
                .symbol
                .as_ref()
                .unwrap_or_else(|| panic!("{} has no symbol", k.id));
            assert!(s.width > 0.0 && s.height > 0.0, "{}", k.id);
            for p in &k.connections {
                let a = s.anchors.iter().find(|a| a.port == p.name);
                let a = a.unwrap_or_else(|| panic!("{}.{} is not anchored", k.id, p.name));
                assert!((0.0..=1.0).contains(&a.offset), "{}.{}", k.id, p.name);
            }
            // Path data only: commands, numbers, separators.
            assert!(
                s.path.chars().all(|c| "MLHVACZ0123456789. -".contains(c)),
                "{}: {}",
                k.id,
                s.path
            );
            // Every point stays inside the box: M/L pairs, H/V values, and
            // an arc's end point (its last two numbers).
            let tokens: Vec<&str> = s.path.split_whitespace().collect();
            let mut i = 0;
            let num = |t: &str| t.parse::<f32>().unwrap();
            let inside = |x: f32, y: f32| {
                assert!(
                    (0.0..=s.width).contains(&x) && (0.0..=s.height).contains(&y),
                    "{}: ({x}, {y}) leaves the box",
                    k.id
                )
            };
            while i < tokens.len() {
                let n = match tokens[i] {
                    "M" | "L" => {
                        inside(num(tokens[i + 1]), num(tokens[i + 2]));
                        2
                    }
                    "H" => {
                        inside(num(tokens[i + 1]), 0.0);
                        1
                    }
                    "V" => {
                        inside(0.0, num(tokens[i + 1]));
                        1
                    }
                    "A" => {
                        inside(num(tokens[i + 6]), num(tokens[i + 7]));
                        7
                    }
                    "Z" => 0,
                    t => panic!("{}: unexpected {t}", k.id),
                };
                i += n + 1;
            }
        }
    }
}
