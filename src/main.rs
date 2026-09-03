use firewing::{verify_ngram_fixture, verify_tokenizer_fixture};
use std::env;
use std::fs;
use std::path::Path;

fn usage() -> ! {
    eprintln!(
        "usage:\n  firewing verify-tokenizer CHECKPOINT_DIR FIXTURE_JSON [REPORT_JSON]\n  firewing verify-ngram CHECKPOINT_DIR MODEL_LOCK FIXTURE_JSON [REPORT_JSON]"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (result, output) = match args.get(1).map(String::as_str) {
        Some("verify-tokenizer") if (4..=5).contains(&args.len()) => (
            verify_tokenizer_fixture(Path::new(&args[2]), Path::new(&args[3]))
                .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(4),
        ),
        Some("verify-ngram") if (5..=6).contains(&args.len()) => (
            verify_ngram_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(5),
        ),
        _ => usage(),
    };
    match result {
        Ok(report) => {
            let serialized = serde_json::to_string_pretty(&report)
                .expect("report serialization cannot fail")
                + "\n";
            if let Some(output) = output {
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
