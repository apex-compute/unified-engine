use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

pub const FOUR_GIB: u64 = 0x1_0000_0000;
pub const SRAM_SECTION_BYTES: u64 = 0x80000;
pub const SRAM_SECTION_ELEMENTS_BF16: u64 = SRAM_SECTION_BYTES / 2;
pub const SRAM_TOTAL_BYTES: u64 = SRAM_SECTION_BYTES * 2;
pub const SAFE_ELTWISE_CHUNK_ELEMENTS: u64 = 131_072;

#[derive(Debug, Deserialize)]
pub struct Gemma4Config {
    pub file_info: FileInfo,
    pub model: ModelInfo,
    pub paths: PathsInfo,
    #[serde(default)]
    pub regions: HashMap<String, Region>,
    #[serde(default)]
    pub non_layer_regions: HashMap<String, Region>,
    pub special: SpecialInfo,
}

#[derive(Debug, Deserialize)]
pub struct FileInfo {
    pub layer_size: u64,
    pub num_layers: u64,
    pub head_dim: u64,
    pub head_dim_sliding: u64,
    pub hidden_size: u64,
    pub embedding_vocab: u64,
    pub group_size: u64,
    pub mlp_elements: u64,
    #[serde(default)]
    pub mlp_elements_wide: Option<u64>,
    pub bytes_per_element: u64,
    pub per_layer_input_dim: u64,
}

#[derive(Debug, Deserialize)]
pub struct ModelInfo {
    pub max_context_size: u64,
    #[serde(default)]
    pub prefill_max_seq_len: Option<u64>,
    #[serde(default)]
    pub max_prefill_seq_len: Option<u64>,
    pub sliding_window: u64,
    pub end_of_turn_token_id: u64,
    pub rope_global_layers: Vec<u64>,
    pub full_attention_layers: Vec<u64>,
    #[serde(default)]
    pub double_wide_mlp_first_layer: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct PathsInfo {
    pub weights_bin: String,
    pub hf_model_dir: String,
    pub hf_model_repo: String,
}

#[derive(Debug, Deserialize)]
pub struct Region {
    #[serde(deserialize_with = "deserialize_u64ish")]
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Deserialize)]
pub struct SpecialInfo {
    pub embedding: EmbeddingInfo,
    pub rope: RopeInfo,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingInfo {
    #[serde(deserialize_with = "deserialize_u64ish")]
    pub token_embd_offset: u64,
    #[serde(deserialize_with = "deserialize_u64ish")]
    pub token_embd_size: u64,
}

#[derive(Debug, Deserialize)]
pub struct RopeInfo {
    pub num_positions: u64,
    pub theta: f64,
    pub local_base: f64,
    pub partial_rotary_factor_global: f64,
}

fn deserialize_u64ish<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    struct U64ishVisitor;

    impl<'de> Visitor<'de> for U64ishVisitor {
        type Value = u64;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a u64 integer or a decimal/hex string")
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
            Ok(value)
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            u64::try_from(value).map_err(|_| E::custom("negative offset is invalid"))
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            if let Some(hex) = value
                .strip_prefix("0x")
                .or_else(|| value.strip_prefix("0X"))
            {
                u64::from_str_radix(hex, 16).map_err(E::custom)
            } else {
                value.parse::<u64>().map_err(E::custom)
            }
        }
    }

    deserializer.deserialize_any(U64ishVisitor)
}

pub fn load_config(path: impl AsRef<Path>) -> Result<Gemma4Config, Box<dyn std::error::Error>> {
    let data = fs::read_to_string(path)?;
    let cfg = serde_json::from_str(&data)?;
    Ok(cfg)
}

#[derive(Debug, Clone)]
pub struct DramSection {
    pub name: &'static str,
    pub start: u64,
    pub end: u64,
    pub purpose: &'static str,
}

impl DramSection {
    pub fn size(&self) -> u64 {
        self.end - self.start
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4Plan {
    pub q_bytes: u64,
    pub k_bytes: u64,
    pub kv_slot_bytes: u64,
    pub kv_no_share_bytes: u64,
    pub estimated_unique_kv_slots: u64,
    pub estimated_kv_bytes: u64,
    pub estimated_kv_saved_bytes: u64,
    pub config_weight_end: u64,
    pub runtime_params_bytes: u64,
    pub runtime_params_headroom_bytes: i64,
    pub dram_sections: Vec<DramSection>,
    pub possible_on_4gib: bool,
    pub sram_notes: Vec<String>,
}

impl Gemma4Plan {
    pub fn from_config(cfg: &Gemma4Config) -> Self {
        let fi = &cfg.file_info;
        let q_bytes = fi.head_dim * fi.group_size * fi.bytes_per_element;
        let k_bytes = fi.head_dim * fi.bytes_per_element;
        let kv_slot_bytes = cfg.model.max_context_size * k_bytes * 2;
        let kv_no_share_bytes = fi.num_layers * kv_slot_bytes;

        // The public Python runner learns the exact shared map from the HF model
        // manifest. For config-only planning, layer 15 is the first special/shared
        // MLP layer, and the code comments say L15-34 share existing KV slots.
        let estimated_unique_kv_slots = cfg
            .model
            .double_wide_mlp_first_layer
            .unwrap_or(fi.num_layers)
            .min(fi.num_layers);
        let estimated_kv_bytes = estimated_unique_kv_slots * kv_slot_bytes;
        let estimated_kv_saved_bytes = kv_no_share_bytes.saturating_sub(estimated_kv_bytes);

        let config_weight_end = max_config_weight_end(cfg);
        let weight_lm_end = 0x6400_0000_u64;
        let runtime_params_bytes = runtime_params_usage(cfg);
        let runtime_params_headroom_bytes = weight_lm_end as i64 - runtime_params_bytes as i64;

        let dram_sections = fixed_dram_sections();
        let no_section_overlaps = sections_do_not_overlap(&dram_sections);
        let all_sections_fit = dram_sections.iter().all(|section| section.end <= FOUR_GIB);
        let kv_region_size = 0x1000_0000_u64;
        let possible_on_4gib = all_sections_fit
            && no_section_overlaps
            && runtime_params_headroom_bytes >= 0
            && estimated_kv_bytes <= kv_region_size;

        let mlp_wide = fi.mlp_elements_wide.unwrap_or(fi.mlp_elements);
        let rows_per_chunk_hidden = SAFE_ELTWISE_CHUNK_ELEMENTS / fi.hidden_size;
        let rows_per_chunk_mlp_wide = SAFE_ELTWISE_CHUNK_ELEMENTS / mlp_wide;
        let sram_notes = vec![
            format!(
                "Two SRAM/URAM sections expose {} bytes total: A={} bytes and B={} bytes.",
                SRAM_TOTAL_BYTES, SRAM_SECTION_BYTES, SRAM_SECTION_BYTES
            ),
            format!(
                "Treat SRAM as a streaming tile store, not model storage; one Gemma layer region is {}.",
                format_bytes(fi.layer_size)
            ),
            format!(
                "Safe eltwise chunk is {} bf16 elements ({}), leaving room for both operands.",
                SAFE_ELTWISE_CHUNK_ELEMENTS,
                format_bytes(SAFE_ELTWISE_CHUNK_ELEMENTS * fi.bytes_per_element)
            ),
            format!(
                "That chunk covers about {} hidden rows or {} wide-MLP rows at Gemma 4 E2B dimensions.",
                rows_per_chunk_hidden, rows_per_chunk_mlp_wide
            ),
            "Keep K/V cache in DRAM and stream K, V, Q, MLP, and residual tiles through SRAM.".to_string(),
        ];

        Self {
            q_bytes,
            k_bytes,
            kv_slot_bytes,
            kv_no_share_bytes,
            estimated_unique_kv_slots,
            estimated_kv_bytes,
            estimated_kv_saved_bytes,
            config_weight_end,
            runtime_params_bytes,
            runtime_params_headroom_bytes,
            dram_sections,
            possible_on_4gib,
            sram_notes,
        }
    }
}

pub fn fixed_dram_sections() -> Vec<DramSection> {
    vec![
        DramSection {
            name: "Weight LM",
            start: 0x0000_0000,
            end: 0x6400_0000,
            purpose: "quantized language weights and embedding section",
        },
        DramSection {
            name: "Weight Vision",
            start: 0x6400_0000,
            end: 0x6c00_0000,
            purpose: "vision encoder weights",
        },
        DramSection {
            name: "Weight Audio",
            start: 0x6c00_0000,
            end: 0x7800_0000,
            purpose: "audio encoder weights",
        },
        DramSection {
            name: "Activation Scratch",
            start: 0x7800_0000,
            end: 0x8800_0000,
            purpose: "stage scratch and temporary tensors",
        },
        DramSection {
            name: "Activation KV",
            start: 0x8800_0000,
            end: 0x9800_0000,
            purpose: "K/V cache region",
        },
        DramSection {
            name: "ISA Audio",
            start: 0x9800_0000,
            end: 0xa000_0000,
            purpose: "audio instruction subregion",
        },
        DramSection {
            name: "ISA Unified",
            start: 0xa000_0000,
            end: 0x1_0000_0000,
            purpose: "unified programs.bin region",
        },
    ]
}

pub fn max_config_weight_end(cfg: &Gemma4Config) -> u64 {
    let fi = &cfg.file_info;
    let mut max_end = 0_u64;
    for region in cfg.regions.values() {
        max_end = max_end.max(region.offset + (fi.num_layers - 1) * fi.layer_size + region.size);
    }
    for region in cfg.non_layer_regions.values() {
        max_end = max_end.max(region.offset + region.size);
    }
    max_end.max(cfg.special.embedding.token_embd_offset + cfg.special.embedding.token_embd_size)
}

pub fn runtime_params_usage(cfg: &Gemma4Config) -> u64 {
    let fi = &cfg.file_info;
    let layer_bytes = fi.num_layers * fi.layer_size;
    let non_layer_bytes: u64 = [
        "OUTPUT_NORM_WEIGHT",
        "PER_LAYER_MODEL_PROJ_WEIGHT",
        "PER_LAYER_PROJ_NORM_WEIGHT",
        "LM_HEAD_WEIGHT_SCALE",
        "LM_HEAD_WEIGHT_DATA",
    ]
    .iter()
    .map(|key| {
        cfg.non_layer_regions
            .get(*key)
            .map(|region| region.size)
            .unwrap_or(0)
    })
    .sum();
    let rope_bytes: u64 = ["ROPE_LOCAL", "ROPE_GLOBAL"]
        .iter()
        .map(|key| {
            cfg.non_layer_regions
                .get(*key)
                .map(|region| region.size)
                .unwrap_or(0)
        })
        .sum();
    let identity_cache_bytes = 64 * 64 * fi.bytes_per_element;

    layer_bytes + non_layer_bytes + rope_bytes + identity_cache_bytes
}

pub fn sections_do_not_overlap(sections: &[DramSection]) -> bool {
    let mut sorted = sections.to_vec();
    sorted.sort_by_key(|section| section.start);
    sorted.windows(2).all(|pair| pair[0].end <= pair[1].start)
}

pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    let value = bytes as f64;
    if value >= GIB {
        format!("{:.2} GiB", value / GIB)
    } else if value >= MIB {
        format!("{:.1} MiB", value / MIB)
    } else if value >= KIB {
        format!("{:.1} KiB", value / KIB)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn config_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/gemma4_e2b/gemma4_e2b_config.json")
    }

    #[test]
    fn parses_hex_offsets_and_builds_expected_plan() {
        let cfg = load_config(config_path()).expect("load gemma4_e2b config");
        let plan = Gemma4Plan::from_config(&cfg);

        assert_eq!(cfg.file_info.num_layers, 35);
        assert_eq!(cfg.file_info.hidden_size, 1536);
        assert_eq!(plan.q_bytes, 8192);
        assert_eq!(plan.k_bytes, 1024);
        assert_eq!(plan.estimated_unique_kv_slots, 15);
        assert_eq!(plan.estimated_kv_bytes, 31_457_280);
        assert_eq!(plan.runtime_params_bytes, 0x6046_dd00);
        assert!(plan.possible_on_4gib);
        assert!(sections_do_not_overlap(&plan.dram_sections));
    }

    #[test]
    fn reports_known_file_and_runtime_weight_usage() {
        let cfg = load_config(config_path()).expect("load gemma4_e2b config");
        assert_eq!(max_config_weight_end(&cfg), 0x9046_bd00);
        assert_eq!(runtime_params_usage(&cfg), 0x6046_dd00);
    }
}
