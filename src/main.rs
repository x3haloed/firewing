use firewing::verify_tokenizer_fixture;
use std::env;
use std::fs;
use std::path::Path;

fn usage() -> ! {
    eprintln!("usage: firewing verify-tokenizer CHECKPOINT_DIR FIXTURE_JSON [REPORT_JSON]");
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if !(4..=5).contains(&args.len()) || args[1] != "verify-tokenizer" {
        usage();
    }
    match verify_tokenizer_fixture(Path::new(&args[2]), Path::new(&args[3])) {
        Ok(report) => {
            let serialized = serde_json::to_string_pretty(&report)
                .expect("report serialization cannot fail")
                + "\n";
            if let Some(output) = args.get(4) {
                let path = Path::new(output);
                if let Some(parent) = path.parent()
                    && let Err(error) = fs::create_dir_all(parent)
                {
                    eprintln!("firewing: cannot create {}: {error}", parent.display());
                    std::process::exit(1);
                }
                let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
                if let Err(error) =
                    fs::write(&temporary, &serialized).and_then(|()| fs::rename(&temporary, path))
                {
                    let _ = fs::remove_file(&temporary);
                    eprintln!("firewing: cannot write {}: {error}", path.display());
                    std::process::exit(1);
                }
            }
            print!("{serialized}");
        }
        Err(error) => {
            eprintln!("firewing: {error}");
            std::process::exit(1);
        }
    }
}
