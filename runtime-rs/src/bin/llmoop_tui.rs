fn main() {
    if let Err(error) = llmoop_runtime::tui::run() {
        eprintln!("llmoop TUI error: {error}");
        std::process::exit(1);
    }
}
