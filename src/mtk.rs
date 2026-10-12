//! The ModelingToolkit block kinds (ADR-0020's first consumer): acausal
//! components — electrical (Resistor, Capacitor, Inductor, Voltage, Ground)
//! and mechanical (Mass, Spring, Damper, Fixed) — whose pins are
//! **domain-typed** (`electrical`, `mechanical`), so a wire between domains
//! is refused before any code exists. Parameters carry their units in the
//! description; the generated file uses plain numbers.
//!
//! Pins are undirected: `connect` is symmetric, buses are legal (a pin may
//! carry several wires), and loops are equations, not cycles.

use moonkale_ext_api::flow::{BlockKind, FlowLibrary, Param, ParamKind, Port, PortType};

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

/// An acausal component: no inputs, no outputs, only pins.
fn acausal(
    id: &str,
    name: &str,
    category: &str,
    description: &str,
    pins: Vec<Port>,
    params: Vec<Param>,
) -> BlockKind {
    BlockKind {
        id: id.into(),
        name: name.into(),
        category: category.into(),
        description: description.into(),
        inputs: vec![],
        connections: pins,
        params,
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
            ),
            acausal(
                "capacitor",
                "Capacitor",
                "electrical",
                "A capacitor: C in F; its voltage v is a state, starting at 0.",
                vec![pin("p", "electrical"), pin("n", "electrical")],
                vec![p("C", "1e-3")],
            ),
            acausal(
                "inductor",
                "Inductor",
                "electrical",
                "An inductor: L in H; its current i is a state, starting at 0.",
                vec![pin("p", "electrical"), pin("n", "electrical")],
                vec![p("L", "1e-3")],
            ),
            acausal(
                "voltage",
                "Voltage",
                "electrical",
                "A constant voltage source: V in V.",
                vec![pin("p", "electrical"), pin("n", "electrical")],
                vec![p("V", "1.0")],
            ),
            acausal(
                "ground",
                "Ground",
                "electrical",
                "The electrical ground — wire it into one net of the circuit.",
                vec![pin("g", "electrical")],
                vec![],
            ),
            acausal(
                "mass",
                "Mass",
                "mechanical",
                "A translating mass: m in kg; its velocity v is a state, starting at 0.",
                vec![pin("flange", "mechanical")],
                vec![p("m", "1.0")],
            ),
            acausal(
                "spring",
                "Spring",
                "mechanical",
                "A spring: k in N/m.",
                vec![pin("a", "mechanical"), pin("b", "mechanical")],
                vec![p("k", "100.0")],
            ),
            acausal(
                "damper",
                "Damper",
                "mechanical",
                "A damper: d in N·s/m.",
                vec![pin("a", "mechanical"), pin("b", "mechanical")],
                vec![p("d", "10.0")],
            ),
            acausal(
                "fixed",
                "Fixed",
                "mechanical",
                "A fixed wall for the mechanical domain, like Ground for the electrical.",
                vec![pin("flange", "mechanical")],
                vec![],
            ),
        ],
    }
}
