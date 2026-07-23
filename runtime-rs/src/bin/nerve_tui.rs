fn main() {
    if let Err(error) = nerve_runtime::tui::run() {
        eprintln!("NERVE TUI error: {error}");
        std::process::exit(1);
    }
}
