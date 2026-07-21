use apex_ue::gemma4_e2b::{format_bytes, load_config, Gemma4Plan};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma4_e2b/gemma4_e2b_config.json"));
    let cfg = load_config(&config_path)?;
    let plan = Gemma4Plan::from_config(&cfg);
    let fi = &cfg.file_info;
    let model = &cfg.model;

    println!("Gemma 4 E2B Apex plan");
    println!("  config        : {}", config_path.display());
    println!("  HF repo       : {}", cfg.paths.hf_model_repo);
    println!(
        "  dimensions    : layers={}, hidden={}, vocab={}, group={}, head_dim={} / sliding={}",
        fi.num_layers,
        fi.hidden_size,
        fi.embedding_vocab,
        fi.group_size,
        fi.head_dim,
        fi.head_dim_sliding
    );
    println!(
        "  context       : max={}, prefill={}, sliding_window={}",
        model.max_context_size,
        model
            .max_prefill_seq_len
            .or(model.prefill_max_seq_len)
            .unwrap_or(model.max_context_size),
        model.sliding_window
    );
    println!(
        "  attention     : full/global layers {:?}",
        model.full_attention_layers
    );
    println!("  q bytes/token : {}", plan.q_bytes);
    println!("  k bytes/token : {}", plan.k_bytes);
    println!(
        "  KV no sharing : {} across {} layers",
        format_bytes(plan.kv_no_share_bytes),
        fi.num_layers
    );
    println!(
        "  KV estimate   : {} unique slots, {} resident, {} saved",
        plan.estimated_unique_kv_slots,
        format_bytes(plan.estimated_kv_bytes),
        format_bytes(plan.estimated_kv_saved_bytes)
    );
    println!(
        "  config max off: 0x{:08x} (file offset high water, not direct DRAM residency)",
        plan.config_weight_end,
    );
    println!(
        "  runtime params: {} ({} headroom in 1600 MiB params region)",
        format_bytes(plan.runtime_params_bytes),
        format_signed_bytes(plan.runtime_params_headroom_bytes)
    );

    println!();
    println!("DRAM layout");
    for section in &plan.dram_sections {
        println!(
            "  {:18} 0x{:08x}..0x{:08x} {:>9}  {}",
            section.name,
            section.start,
            section.end,
            format_bytes(section.size()),
            section.purpose
        );
    }

    println!();
    println!("SRAM strategy");
    for note in &plan.sram_notes {
        println!("  - {note}");
    }

    println!();
    println!(
        "Feasibility: {}",
        if plan.possible_on_4gib {
            "possible with the current quantized Apex layout, pending hardware validation"
        } else {
            "blocked by the current layout; inspect the figures above"
        }
    );
    println!(
        "Caveat: the combined weight bin may contain host-side sections that are not all accelerator-resident at once."
    );

    Ok(())
}

fn format_signed_bytes(bytes: i64) -> String {
    if bytes >= 0 {
        format_bytes(bytes as u64)
    } else {
        format!("-{}", format_bytes(bytes.unsigned_abs()))
    }
}
