use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let simplified = zhconv::zhconv(&line, zhconv::Variant::ZhCN);
        if writeln!(stdout, "{}", simplified).is_err() || stdout.flush().is_err() {
            break;
        }
    }
}
