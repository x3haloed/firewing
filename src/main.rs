use firewing::{
    benchmark_expert_acquisition, benchmark_ngram_transport, verify_attention_residual_fixture,
    verify_decoder_layer_fixture, verify_decoder_layer1_fixture, verify_decoder_layer3_fixture,
    verify_deltanet_fixture, verify_expert_fixture, verify_full_attention_fixture,
    verify_full_attention_residual_fixture, verify_hyper_connection_fixture,
    verify_mixture_fixture, verify_ngram_fixture, verify_ngram_rows,
    verify_ple_attention_residual_fixture, verify_ple_fixture, verify_router_fixture,
    verify_sparse_moe_fixture, verify_tokenizer_fixture,
};
use std::env;
use std::fs;
use std::path::Path;

fn usage() -> ! {
    eprintln!(
        "usage:\n  firewing verify-tokenizer CHECKPOINT_DIR FIXTURE_JSON [REPORT_JSON]\n  firewing verify-ngram CHECKPOINT_DIR MODEL_LOCK FIXTURE_JSON [REPORT_JSON]\n  firewing verify-ngram-rows CHECKPOINT_DIR MODEL_LOCK ADDRESS_FIXTURE ROW_FIXTURE [REPORT_JSON]\n  firewing bench-ngram-transport CHECKPOINT_DIR MODEL_LOCK ADDRESS_FIXTURE ROW_FIXTURE COMMIT [REPORT_JSON]\n  firewing bench-expert-acquisition CHECKPOINT_DIR MODEL_LOCK FIXTURE_JSON COMMIT [REPORT_JSON]\n  firewing verify-router CHECKPOINT_DIR MODEL_LOCK FIXTURE_JSON [REPORT_JSON]\n  firewing verify-expert CHECKPOINT_DIR MODEL_LOCK ROUTER_FIXTURE EXPERT_FIXTURE [REPORT_JSON]\n  firewing verify-mixture CHECKPOINT_DIR MODEL_LOCK ROUTER_FIXTURE EXPERT_FIXTURE MIXTURE_FIXTURE [REPORT_JSON]\n  firewing verify-sparse-moe CHECKPOINT_DIR MODEL_LOCK ROUTER_FIXTURE EXPERT_FIXTURE MIXTURE_FIXTURE SPARSE_MOE_FIXTURE [REPORT_JSON]\n  firewing verify-hyper-connection CHECKPOINT_DIR MODEL_LOCK FIXTURE_JSON [REPORT_JSON]\n  firewing verify-deltanet CHECKPOINT_DIR MODEL_LOCK FIXTURE_JSON [REPORT_JSON]\n  firewing verify-attention-residual CHECKPOINT_DIR MODEL_LOCK HYPER_FIXTURE DELTANET_FIXTURE FIXTURE_JSON [REPORT_JSON]\n  firewing verify-decoder-layer CHECKPOINT_DIR MODEL_LOCK HYPER_FIXTURE DELTANET_FIXTURE ATTENTION_FIXTURE SPARSE_MOE_FIXTURE FIXTURE_JSON [REPORT_JSON]\n  firewing verify-ple CHECKPOINT_DIR MODEL_LOCK NGRAM_FIXTURE NGRAM_ROW_FIXTURE PLE_FIXTURE [REPORT_JSON]\n  firewing verify-ple-attention-residual CHECKPOINT_DIR MODEL_LOCK NGRAM_FIXTURE NGRAM_ROW_FIXTURE PLE_FIXTURE FIXTURE_JSON [REPORT_JSON]\n  firewing verify-decoder-layer1 CHECKPOINT_DIR MODEL_LOCK NGRAM_FIXTURE NGRAM_ROW_FIXTURE PLE_FIXTURE ATTENTION_RESIDUAL_FIXTURE FIXTURE_JSON [REPORT_JSON]\n  firewing verify-full-attention CHECKPOINT_DIR MODEL_LOCK FIXTURE_JSON [REPORT_JSON]\n  firewing verify-full-attention-residual CHECKPOINT_DIR MODEL_LOCK FULL_ATTENTION_FIXTURE FIXTURE_JSON [REPORT_JSON]\n  firewing verify-decoder-layer3 CHECKPOINT_DIR MODEL_LOCK FULL_ATTENTION_FIXTURE ATTENTION_RESIDUAL_FIXTURE FIXTURE_JSON [REPORT_JSON]"
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
        Some("verify-ngram-rows") if (6..=7).contains(&args.len()) => (
            verify_ngram_rows(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(6),
        ),
        Some("bench-ngram-transport") if (7..=8).contains(&args.len()) => (
            benchmark_ngram_transport(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                &args[6],
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(7),
        ),
        Some("bench-expert-acquisition") if (6..=7).contains(&args.len()) => (
            benchmark_expert_acquisition(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                &args[5],
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(6),
        ),
        Some("verify-router") if (5..=6).contains(&args.len()) => (
            verify_router_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(5),
        ),
        Some("verify-hyper-connection") if (5..=6).contains(&args.len()) => (
            verify_hyper_connection_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(5),
        ),
        Some("verify-deltanet") if (5..=6).contains(&args.len()) => (
            verify_deltanet_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(5),
        ),
        Some("verify-attention-residual") if (7..=8).contains(&args.len()) => (
            verify_attention_residual_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(7),
        ),
        Some("verify-decoder-layer") if (9..=10).contains(&args.len()) => (
            verify_decoder_layer_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
                Path::new(&args[7]),
                Path::new(&args[8]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(9),
        ),
        Some("verify-ple") if (7..=8).contains(&args.len()) => (
            verify_ple_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(7),
        ),
        Some("verify-ple-attention-residual") if (8..=9).contains(&args.len()) => (
            verify_ple_attention_residual_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
                Path::new(&args[7]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(8),
        ),
        Some("verify-decoder-layer1") if (9..=10).contains(&args.len()) => (
            verify_decoder_layer1_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
                Path::new(&args[7]),
                Path::new(&args[8]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(9),
        ),
        Some("verify-full-attention") if (5..=6).contains(&args.len()) => (
            verify_full_attention_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(5),
        ),
        Some("verify-full-attention-residual") if (6..=7).contains(&args.len()) => (
            verify_full_attention_residual_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(6),
        ),
        Some("verify-decoder-layer3") if (7..=8).contains(&args.len()) => (
            verify_decoder_layer3_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(7),
        ),
        Some("verify-expert") if (6..=7).contains(&args.len()) => (
            verify_expert_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(6),
        ),
        Some("verify-mixture") if (7..=8).contains(&args.len()) => (
            verify_mixture_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(7),
        ),
        Some("verify-sparse-moe") if (8..=9).contains(&args.len()) => (
            verify_sparse_moe_fixture(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
                Path::new(&args[7]),
            )
            .and_then(|report| serde_json::to_value(report).map_err(|error| error.to_string())),
            args.get(8),
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
