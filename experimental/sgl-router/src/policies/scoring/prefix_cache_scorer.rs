// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! [`PrefixCacheScorer`] — prefer the worker that already holds the deepest
//! prefix of the request in its KV cache (the cache term of the fused score).
//!
//! Unlike the binary hit/miss used by the existing `cache_aware_zmq`
//! fast-path, this scorer returns a *continuous* per-worker hit depth in
//! `[0, 1]`: the fraction of the request's block-hash chain that the worker
//! is known to hold, read from the KV-event-fed [`HashTree`]. A worker that
//! holds the whole prefix scores `1.0`; one that holds half scores `~0.5`;
//! one that holds none scores `0.0`.
//!
//! This is a weighted, continuous prefix-overlap term composed into the fused
//! score; the default configuration makes it the primary term. The block-hash
//! source is this router's own kv_events `HashTree`.
//!
//! ## Per-worker depth
//! [`HashTree::match_prefix`] only reports the workers holding the *deepest*
//! matched node for a query. To recover each worker's *own* prefix depth we
//! probe increasing prefixes `hashes[..k]` and record the largest `k` at
//! which this worker is still in the matched set. The walk is monotonic (a
//! worker that diverges at depth `k` can never reappear deeper on the fixed
//! request path), so we stop at the first miss. This costs up to `n` read-
//! locked matches per scored worker; it is correct and deterministic. A
//! future optimization would precompute the per-worker depth map once per
//! request (request-scoped cache) instead of per-`score()` call.
//!
//! ## No-signal convention
//! When the request cannot be hashed at all (no body, no tokenizer for the
//! model, no worker-published block size yet, or an empty block chain) the
//! scorer is *inert*: it returns the neutral `1.0` for every worker so it
//! neither rewards nor penalizes, and the fused decision falls to the other
//! scorers (e.g. load). This matches the module-level score convention.

use super::{ScoreContext, Scorer};
use crate::discovery::ModelId;
use crate::policies::kv_events::{compute_block_hashes, BlockSizeOracle, HashTree};
use crate::tokenizer::{adapter, TokenizerRegistry};
use crate::workers::Worker;
use std::sync::Arc;

/// Neutral score returned when there is no prefix-cache signal for the
/// request (see module docs). Higher = better, so a neutral scorer returns
/// the top of the range and lets other scorers decide.
const NEUTRAL: f32 = 1.0;

/// Scores workers by continuous KV-cache prefix-hit depth in `[0, 1]`.
pub struct PrefixCacheScorer {
    weight: f32,
    /// KV-event-fed hash tree (read-only from here). Shared `Arc` with the
    /// `KvEventIndex` pump that writes it.
    tree: Arc<HashTree>,
    /// Per-model tokenizers; the request prompt is encoded with the
    /// tokenizer for `ctx.selection().model()`.
    tokenizers: Arc<TokenizerRegistry>,
    /// Worker-published block size — the router must hash at the same block
    /// size the engines publish at, or the hashes won't line up.
    block_size_oracle: Arc<BlockSizeOracle>,
}

impl std::fmt::Debug for PrefixCacheScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixCacheScorer")
            .field("weight", &self.weight)
            .field("tree_nodes", &self.tree.node_count())
            .finish()
    }
}

impl PrefixCacheScorer {
    /// `weight` is the non-negative multiplier applied in the fused
    /// weighted sum. The default configuration makes prefix-cache the
    /// primary term, with load weighted lower as a corrective.
    pub fn new(
        weight: f32,
        tree: Arc<HashTree>,
        tokenizers: Arc<TokenizerRegistry>,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Self {
        Self {
            weight,
            tree,
            tokenizers,
            block_size_oracle,
        }
    }

    /// Extract a prompt-text candidate from a JSON request body. Mirrors
    /// the shapes `cache_aware_zmq` accepts (prompt string / array, chat
    /// `messages[*].content`, SGLang `text`). Returns `None` when the body
    /// is not routable — the scorer then falls back to neutral.
    fn extract_prompt_text(body: &[u8]) -> Option<String> {
        let v: serde_json::Value = serde_json::from_slice(body).ok()?;
        if let Some(s) = v.get("prompt").and_then(|p| p.as_str()) {
            return Some(s.to_string());
        }
        if let Some(arr) = v.get("prompt").and_then(|p| p.as_array()) {
            let parts: Vec<&str> = arr.iter().filter_map(|x| x.as_str()).collect();
            if !parts.is_empty() {
                return Some(parts.join("\n"));
            }
        }
        if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
            let mut buf = String::new();
            for m in msgs {
                match m.get("content") {
                    Some(serde_json::Value::String(s)) => {
                        if !buf.is_empty() {
                            buf.push('\n');
                        }
                        buf.push_str(s);
                    }
                    Some(serde_json::Value::Array(parts)) => {
                        for part in parts {
                            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                                if !buf.is_empty() {
                                    buf.push('\n');
                                }
                                buf.push_str(t);
                            }
                        }
                    }
                    _ => {}
                }
            }
            if !buf.is_empty() {
                return Some(buf);
            }
        }
        if let Some(s) = v.get("text").and_then(|t| t.as_str()) {
            return Some(s.to_string());
        }
        None
    }

    /// Tokenize `text` for `model_id`. `None` on missing tokenizer / encode
    /// error / empty output — the scorer then falls back to neutral.
    fn tokenize(&self, model_id: &ModelId, text: &str) -> Option<Vec<u32>> {
        let tokenizer = self.tokenizers.get(&model_id.0)?;
        match adapter::encode(&tokenizer, text) {
            Ok(ids) if !ids.is_empty() => Some(ids),
            _ => None,
        }
    }

    /// Compute the request's block-hash chain, or `None` when the request
    /// cannot be hashed (no body / tokenizer / block size / empty chain).
    fn request_block_hashes(&self, ctx: &ScoreContext<'_>) -> Option<Vec<i64>> {
        let body = ctx.selection().request_body().filter(|b| !b.is_empty())?;
        let text = Self::extract_prompt_text(body)?;
        let tokens = self.tokenize(ctx.selection().model(), &text)?;
        let block_size = self.block_size_oracle.get()? as usize;
        let hashes = compute_block_hashes(&tokens, block_size);
        if hashes.is_empty() {
            None
        } else {
            Some(hashes)
        }
    }

    /// This worker's own prefix-hit depth against `hashes`: the largest `k`
    /// such that the worker is in `match_prefix(hashes[..k])`'s deepest set.
    /// Monotonic — stop at the first miss.
    fn worker_depth(&self, url: &str, hashes: &[i64]) -> usize {
        let mut depth = 0usize;
        for k in 1..=hashes.len() {
            let m = self.tree.match_prefix(None, &hashes[..k]);
            // `matched_blocks < k` means nobody reaches depth k on this
            // path (the returned workers belong to a shallower node), so
            // this worker can't be counted at depth k either.
            if m.matched_blocks == k && m.workers.iter().any(|w| w.url == url) {
                depth = k;
            } else {
                break;
            }
        }
        depth
    }
}

impl Scorer for PrefixCacheScorer {
    fn score(&self, worker: &Arc<Worker>, ctx: &ScoreContext<'_>) -> f32 {
        let Some(hashes) = self.request_block_hashes(ctx) else {
            return NEUTRAL;
        };
        let depth = self.worker_depth(&worker.url, &hashes);
        depth as f32 / hashes.len() as f32
    }

    fn weight(&self) -> f32 {
        self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::worker_with_load;
    use super::*;
    use crate::policies::kv_events::tree::KvWorkerId;
    use crate::policies::SelectionContext;

    /// Tiny-tokenizer registry mirroring the cache_aware_zmq test fixture.
    fn tokenizer_registry_with_tiny() -> Arc<TokenizerRegistry> {
        let cfg = crate::config::Config {
            server: crate::config::ServerConfig {
                host: "0".into(),
                port: 0,
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                id: "tiny".into(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
                policy: crate::config::PolicyKind::RoundRobin,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
            },
            discovery: crate::config::DiscoveryBackend::StaticUrls(
                crate::config::StaticUrlsDiscoveryConfig {
                    urls: vec!["http://placeholder:0".into()],
                },
            ),
            proxy: crate::config::ProxyConfig::default(),
            active_load: crate::config::ActiveLoadConfig::default(),
        };
        Arc::new(TokenizerRegistry::load_from_config(&cfg).expect("load tiny tokenizer"))
    }

    fn oracle(block_size: u32) -> Arc<BlockSizeOracle> {
        let o = BlockSizeOracle::new();
        o.try_set(block_size).expect("fresh oracle accepts set");
        o
    }

    /// Hash the canonical test prompt through the tiny tokenizer so the
    /// tree state and the request path use identical block hashes.
    fn hashes_for(registry: &TokenizerRegistry, text: &str, block_size: u32) -> Vec<i64> {
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let h = compute_block_hashes(&ids, block_size as usize);
        assert!(!h.is_empty(), "tiny tokenizer must produce ≥1 full block");
        h
    }

    fn score_one(
        scorer: &PrefixCacheScorer,
        worker: &Arc<Worker>,
        candidates: &[Arc<Worker>],
        body: &[u8],
    ) -> f32 {
        let model = ModelId("tiny".into());
        let sel = SelectionContext::new(&model, Some(body));
        let ctx = ScoreContext::new(&sel, candidates);
        scorer.score(worker, &ctx)
    }

    #[test]
    fn full_holder_scores_one() {
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let bs = 4u32;
        let hashes = hashes_for(&registry, text, bs);
        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://a".into(), 0), None, &hashes);

        let scorer = PrefixCacheScorer::new(3.0, tree, registry, oracle(bs));
        let a = worker_with_load("http://a", 0);
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();
        let s = score_one(&scorer, &a, &[a.clone()], &body);
        assert!((s - 1.0).abs() < 1e-6, "full holder → 1.0, got {s}");
    }

    #[test]
    fn non_holder_scores_zero() {
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let bs = 4u32;
        let hashes = hashes_for(&registry, text, bs);
        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://a".into(), 0), None, &hashes);

        let scorer = PrefixCacheScorer::new(3.0, tree, registry, oracle(bs));
        // Worker "b" holds nothing.
        let b = worker_with_load("http://b", 0);
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();
        let s = score_one(&scorer, &b, &[b.clone()], &body);
        assert_eq!(s, 0.0, "non-holder → 0.0");
    }

    #[test]
    fn partial_holder_scores_between_zero_and_one() {
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world hello world";
        let bs = 4u32;
        let hashes = hashes_for(&registry, text, bs);
        assert!(hashes.len() >= 2, "need ≥2 blocks for a partial hit");
        let partial_len = hashes.len() / 2;
        let tree = Arc::new(HashTree::new());
        // b holds only the first half of the chain.
        tree.insert(
            &KvWorkerId::new("http://b".into(), 0),
            None,
            &hashes[..partial_len],
        );

        let scorer = PrefixCacheScorer::new(3.0, tree, registry, oracle(bs));
        let b = worker_with_load("http://b", 0);
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();
        let s = score_one(&scorer, &b, &[b.clone()], &body);
        let expected = partial_len as f32 / hashes.len() as f32;
        assert!(
            (s - expected).abs() < 1e-6,
            "partial holder → {expected}, got {s}",
        );
        assert!(s > 0.0 && s < 1.0, "partial hit must be strictly in (0,1)");
    }

    #[test]
    fn no_body_is_neutral() {
        let registry = tokenizer_registry_with_tiny();
        let tree = Arc::new(HashTree::new());
        let scorer = PrefixCacheScorer::new(3.0, tree, registry, oracle(4));
        let a = worker_with_load("http://a", 0);
        let model = ModelId("tiny".into());
        let sel = SelectionContext::new(&model, None);
        let ctx = ScoreContext::new(&sel, std::slice::from_ref(&a));
        assert_eq!(scorer.score(&a, &ctx), NEUTRAL, "no body → neutral 1.0");
    }

    #[test]
    fn no_block_size_is_neutral() {
        let registry = tokenizer_registry_with_tiny();
        let tree = Arc::new(HashTree::new());
        // Oracle never set → cannot hash → inert.
        let scorer = PrefixCacheScorer::new(3.0, tree, registry, BlockSizeOracle::new());
        let a = worker_with_load("http://a", 0);
        let body = br#"{"prompt":"hello world"}"#;
        let s = score_one(&scorer, &a, &[a.clone()], body);
        assert_eq!(s, NEUTRAL, "no block size → neutral 1.0");
    }

    #[test]
    fn weight_is_reported() {
        let registry = tokenizer_registry_with_tiny();
        let tree = Arc::new(HashTree::new());
        let scorer = PrefixCacheScorer::new(3.0, tree, registry, oracle(4));
        assert_eq!(scorer.weight(), 3.0);
    }
}
