// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Pluggable **Filter → Score → Pick** routing framework.
//!
//! Turns the compile-time `enum PolicyKind` + one monolithic `select()`
//! per policy into a composable pipeline:
//!
//! ```text
//! candidates = workers.filter(all filters)              // eligibility
//! scored     = candidates.map(w => Σ weight_i · score_i(w, ctx))
//! winner     = selector.select_one(scored)              // argmax or sampled
//! ```
//!
//! This is *additive*: the existing `round_robin` / `random` /
//! `power_of_two` / `cache_aware_zmq` policies are untouched. A new
//! [`ScoredPolicy`] composes any set of [`Filter`] / [`Scorer`] / [`Selector`]
//! plugins and implements the existing [`Policy`] trait, so it drops into
//! the same [`PolicyRegistry`].
//!
//! ## Design notes
//! - The plugin boundary keeps eligibility (filters), desirability (scorers),
//!   and selection (selectors) as independent, testable units with per-scorer
//!   configurable weights and a weighted-sum aggregation
//!   `final = Σ weight_i · score_i`. A selector is *total*: given a non-empty
//!   candidate set it always returns a worker (there is also a cold-start
//!   fallback if every worker is filtered out).
//! - [`ArgmaxSelector`] breaks ties by lowest worker URL, so all router
//!   replicas resolve an equal-score tie to the *same* worker rather than to
//!   whatever happened to arrive first in the candidate list.
//! - [`SoftmaxSelector`] offers a temperature-controlled sampled alternative
//!   to argmax, to spread load rather than herd onto the single best worker.
//!
//! ## Score convention
//! Every [`Scorer::score`] returns a value normalized to **`[0, 1]`**, where
//! **higher is better** (more desirable worker). A scorer with no signal for
//! the current request returns a neutral `1.0` (does not penalize). Weights
//! are non-negative multipliers. The default configuration treats prefix-cache
//! locality as the primary term and load as a lighter corrective term (see the
//! individual scorer modules for the exact defaults).

pub mod load_scorer;
pub mod pickers;
pub mod prefix_cache_scorer;

use super::{Policy, SelectionContext};
use crate::workers::Worker;
use std::sync::Arc;

/// Scoring-time view handed to every [`Filter`] and [`Scorer`].
///
/// Wraps the request-level [`SelectionContext`] and additionally exposes the
/// *candidate set* (workers that survived the filters) so scorers can compute
/// relative normalization — e.g. min/max load across live workers — without
/// reaching into global state.
pub struct ScoreContext<'a> {
    sel: &'a SelectionContext<'a>,
    candidates: &'a [Arc<Worker>],
}

impl<'a> ScoreContext<'a> {
    pub fn new(sel: &'a SelectionContext<'a>, candidates: &'a [Arc<Worker>]) -> Self {
        Self { sel, candidates }
    }

    /// The underlying request-level selection context (model + body).
    pub fn selection(&self) -> &SelectionContext<'a> {
        self.sel
    }

    /// Workers that survived filtering — the set being scored. Scorers that
    /// normalize relative to the field (e.g. load) read this.
    pub fn candidates(&self) -> &[Arc<Worker>] {
        self.candidates
    }
}

/// Eligibility gate: return `false` to drop a worker from candidacy before
/// scoring (e.g. role/health/capacity checks).
pub trait Filter: Send + Sync + std::fmt::Debug {
    fn keep(&self, worker: &Arc<Worker>, ctx: &ScoreContext<'_>) -> bool;
}

/// A weighted per-worker scorer. [`Self::score`] returns a value in `[0, 1]`
/// (higher = more desirable); [`Self::weight`] is a non-negative multiplier
/// applied in the weighted sum. See the module-level *Score convention*.
pub trait Scorer: Send + Sync + std::fmt::Debug {
    fn score(&self, worker: &Arc<Worker>, ctx: &ScoreContext<'_>) -> f32;
    fn weight(&self) -> f32;
}

/// Chooses a winner from `(worker, aggregated_score)` pairs. Implementations
/// must be *total*: a non-empty slice always yields `Some`.
pub trait Selector: Send + Sync + std::fmt::Debug {
    fn select_one(&self, scored: &[(Arc<Worker>, f32)]) -> Option<Arc<Worker>>;
}

/// A [`Policy`] assembled from a set of filters, weighted scorers, and a
/// selector. New scorers (prefix-cache, session affinity, KV-utilization, …)
/// plug in by pushing a `Box<dyn Scorer>` here and registering a name in
/// `factory.rs` — no change to this file.
pub struct ScoredPolicy {
    filters: Vec<Box<dyn Filter>>,
    scorers: Vec<Box<dyn Scorer>>,
    selector: Box<dyn Selector>,
}

impl std::fmt::Debug for ScoredPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScoredPolicy")
            .field("filters", &self.filters)
            .field("scorers", &self.scorers)
            .field("selector", &self.selector)
            .finish()
    }
}

impl ScoredPolicy {
    pub fn new(
        filters: Vec<Box<dyn Filter>>,
        scorers: Vec<Box<dyn Scorer>>,
        selector: Box<dyn Selector>,
    ) -> Self {
        Self {
            filters,
            scorers,
            selector,
        }
    }

    /// Builder-style: append a filter.
    pub fn with_filter(mut self, f: Box<dyn Filter>) -> Self {
        self.filters.push(f);
        self
    }

    /// Builder-style: append a scorer.
    pub fn with_scorer(mut self, s: Box<dyn Scorer>) -> Self {
        self.scorers.push(s);
        self
    }
}

impl Policy for ScoredPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        if workers.is_empty() {
            return None;
        }
        // Filter → candidate set. Cold-start fallback: if every worker is
        // filtered out, fall back to the full set rather than failing to
        // route.
        let mut candidates: Vec<Arc<Worker>> = Vec::with_capacity(workers.len());
        {
            // Build a throwaway ctx over the full set for filter evaluation;
            // filters that normalize relative to candidates see the full pool.
            let fctx = ScoreContext::new(ctx, workers);
            for w in workers {
                if self.filters.iter().all(|flt| flt.keep(w, &fctx)) {
                    candidates.push(w.clone());
                }
            }
        }
        let scored_over: Vec<Arc<Worker>> = if candidates.is_empty() {
            workers.to_vec()
        } else {
            candidates
        };

        let sctx = ScoreContext::new(ctx, &scored_over);
        let scored: Vec<(Arc<Worker>, f32)> = scored_over
            .iter()
            .map(|w| {
                let agg: f32 = self
                    .scorers
                    .iter()
                    .map(|s| s.weight() * s.score(w, &sctx))
                    .sum();
                (w.clone(), agg)
            })
            .collect();

        self.selector.select_one(&scored)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::workers::Worker;
    use std::sync::Arc;

    /// Build a plain worker with `n` in-flight requests already registered.
    pub fn worker_with_load(url: &str, load: usize) -> Arc<Worker> {
        let w = Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }));
        // active_load() reads active_requests; bump it via load guards that we
        // deliberately leak for the lifetime of the test worker.
        for _ in 0..load {
            std::mem::forget(w.load_guard());
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::load_scorer::LoadScorer;
    use super::pickers::ArgmaxSelector;
    use super::test_support::worker_with_load;
    use super::*;
    use crate::discovery::ModelId;

    #[derive(Debug)]
    struct DropAll;
    impl Filter for DropAll {
        fn keep(&self, _w: &Arc<Worker>, _ctx: &ScoreContext<'_>) -> bool {
            false
        }
    }

    #[test]
    fn scored_policy_picks_lowest_load_via_load_scorer() {
        let workers = vec![
            worker_with_load("http://a", 5),
            worker_with_load("http://b", 1),
            worker_with_load("http://c", 9),
        ];
        let policy = ScoredPolicy::new(
            vec![],
            vec![Box::new(LoadScorer::new(1.0))],
            Box::new(ArgmaxSelector::new()),
        );
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);
        let chosen = policy.select(&workers, &ctx).unwrap();
        assert_eq!(chosen.url, "http://b", "lowest-load worker must win");
    }

    #[test]
    fn empty_workers_returns_none() {
        let policy = ScoredPolicy::new(vec![], vec![], Box::new(ArgmaxSelector::new()));
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);
        assert!(policy.select(&[], &ctx).is_none());
    }

    /// Acceptance test: a *partial-hit idle* worker must beat a *full-hit
    /// busy* worker once cache and load are fused — i.e. the load term can
    /// override the cache term. Exercises the whole pipeline:
    /// PrefixCacheScorer (continuous depth) + LoadScorer, weighted-summed,
    /// argmax-picked.
    #[test]
    fn partial_hit_idle_worker_beats_full_hit_busy_worker() {
        use super::prefix_cache_scorer::PrefixCacheScorer;
        use crate::policies::kv_events::tree::KvWorkerId;
        use crate::policies::kv_events::{compute_block_hashes, BlockSizeOracle, HashTree};
        use crate::tokenizer::{adapter, TokenizerRegistry};

        // Tiny-tokenizer registry (same fixture the cache-aware tests use).
        let registry: Arc<TokenizerRegistry> = {
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
        };

        let text = "hello world hello world hello world hello world";
        let block_size = 4u32;
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, block_size as usize);
        assert!(hashes.len() >= 2, "need ≥2 blocks for a partial hit");
        let partial_len = hashes.len() / 2;

        let tree = Arc::new(HashTree::new());
        // busy worker "a" holds the FULL prefix; idle worker "b" holds HALF.
        tree.insert(&KvWorkerId::new("http://a".into(), 0), None, &hashes);
        tree.insert(
            &KvWorkerId::new("http://b".into(), 0),
            None,
            &hashes[..partial_len],
        );
        let oracle = {
            let o = BlockSizeOracle::new();
            o.try_set(block_size).unwrap();
            o
        };

        // a: full cache but heavily loaded. b: partial cache but idle.
        let a = worker_with_load("http://a", 10);
        let b = worker_with_load("http://b", 0);
        let workers = vec![a.clone(), b.clone()];
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));

        // Cache is the primary term (weight 1.0) with load as a lighter
        // corrective (weight 0.7) — yet the load delta still flips the
        // winner to the idle partial-hit worker.
        let prefix = PrefixCacheScorer::new(1.0, tree, registry, oracle);
        let fused = ScoredPolicy::new(
            vec![],
            vec![Box::new(prefix), Box::new(LoadScorer::new(0.7))],
            Box::new(ArgmaxSelector::new()),
        );
        let chosen = fused.select(&workers, &ctx).unwrap();
        assert_eq!(
            chosen.url, "http://b",
            "partial-hit idle worker must beat full-hit busy worker once load is fused",
        );

        // Control: with ONLY the cache term (no load), the full-hit busy
        // worker wins — proving it is the load term that flips the decision.
        let a2 = worker_with_load("http://a", 10);
        let b2 = worker_with_load("http://b", 0);
        let workers2 = vec![a2.clone(), b2.clone()];
        let tree2 = Arc::new(HashTree::new());
        tree2.insert(&KvWorkerId::new("http://a".into(), 0), None, &hashes);
        tree2.insert(
            &KvWorkerId::new("http://b".into(), 0),
            None,
            &hashes[..partial_len],
        );
        let oracle2 = {
            let o = BlockSizeOracle::new();
            o.try_set(block_size).unwrap();
            o
        };
        let registry2: Arc<TokenizerRegistry> = {
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
        };
        let cache_only = ScoredPolicy::new(
            vec![],
            vec![Box::new(PrefixCacheScorer::new(1.0, tree2, registry2, oracle2))],
            Box::new(ArgmaxSelector::new()),
        );
        let chosen_cache_only = cache_only.select(&workers2, &ctx).unwrap();
        assert_eq!(
            chosen_cache_only.url, "http://a",
            "cache-only must prefer the full-hit worker (control for the flip)",
        );
    }

    #[test]
    fn all_filtered_out_falls_back_to_full_set() {
        // Cold-start fallback: DropAll removes everyone, but the pipeline
        // must still route rather than drop the request.
        let workers = vec![worker_with_load("http://a", 2)];
        let policy = ScoredPolicy::new(
            vec![Box::new(DropAll)],
            vec![Box::new(LoadScorer::new(1.0))],
            Box::new(ArgmaxSelector::new()),
        );
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);
        assert!(policy.select(&workers, &ctx).is_some());
    }
}
