pub mod analysis;
pub mod app;
pub mod automata;
pub mod discrete_automata;
pub mod quantization;
pub mod types;

use crate::app::VCEMApp;
use eframe::{NativeOptions, Result, run_native};

fn main() -> Result<()> {
    let options = NativeOptions::default();

    run_native(
        "VCEM Analyser",
        options,
        Box::new(|cc| Ok(Box::new(VCEMApp::new(cc)))),
    )
}
