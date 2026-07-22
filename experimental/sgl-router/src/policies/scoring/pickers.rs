// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! [`Selector`] implementations: [`ArgmaxSelector`] (argmax) and
//! [`SoftmaxSelector`] (temperature-controlled weighted-random).
//!
//! - `ArgmaxSelector` selects the highest-scoring worker; equal scores are
//!   broken by lowest worker URL, so every router replica resolves a tie to
//!   the same worker regardless of candidate ordering.
//! - `SoftmaxSelector` samples a worker with probability
//!   ∝ `exp(score / temperature)` to avoid thundering-herd on the single
//!   best worker.
//!
//! Both are total: a non-empty input always yields `Some`.

use super::Selector;
use crate::workers::Worker;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

/// Argmax selector: returns the worker with the highest aggregated score.
/// Equal scores are broken deterministically by the lexicographically
/// smallest worker URL, so the choice is stable across router replicas and
/// independent of candidate ordering.
#[derive(Debug, Default, Clone)]
pub struct ArgmaxSelector;

impl ArgmaxSelector {
    pub fn new() -> Self {
        Self
    }
}

impl Selector for ArgmaxSelector {
    fn select_one(&self, scored: &[(Arc<Worker>, f32)]) -> Option<Arc<Worker>> {
        scored
            .iter()
            .fold(None::<&(Arc<Worker>, f32)>, |best, cur| match best {
                None => Some(cur),
                Some(b) => {
                    // Strictly-better score wins; on an exact tie prefer the
                    // smaller URL so the result does not depend on ordering.
                    let better = cur.1 > b.1 || (cur.1 == b.1 && cur.0.url < b.0.url);
                    if better {
                        Some(cur)
                    } else {
                        Some(b)
                    }
                }
            })
            .map(|(w, _)| w.clone())
    }
}

/// Temperature-controlled softmax weighted-random selector.
///
/// Probability of picking worker `i` is `exp(s_i / T) / Σ_j exp(s_j / T)`.
/// As `T → 0` the distribution collapses to argmax; larger `T` flattens it
/// toward uniform, spreading load and avoiding herd-on-best.
#[derive(Debug, Clone)]
pub struct SoftmaxSelector {
    temperature: f32,
}

impl SoftmaxSelector {
    /// `temperature` must be > 0. A non-positive temperature degenerates to
    /// argmax (equivalent to [`ArgmaxSelector`]) to keep the selector total.
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    /// Deterministic core used by tests: sample with a caller-supplied RNG.
    /// [`Selector::select_one`] wraps this with a freshly-seeded thread RNG.
    pub fn select_with_rng<R: Rng>(
        &self,
        scored: &[(Arc<Worker>, f32)],
        rng: &mut R,
    ) -> Option<Arc<Worker>> {
        if scored.is_empty() {
            return None;
        }
        // Non-positive temperature → argmax fallback (keeps the selector total).
        if self.temperature <= 0.0 || !self.temperature.is_finite() {
            return ArgmaxSelector::new().select_one(scored);
        }
        // Numerical stability: subtract max score before exponentiating.
        let max = scored
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scored
            .iter()
            .map(|(_, s)| ((s - max) / self.temperature).exp())
            .collect();
        let total: f32 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            // Degenerate (all -inf / NaN) → argmax fallback.
            return ArgmaxSelector::new().select_one(scored);
        }
        let mut threshold = rng.gen::<f32>() * total;
        for (i, w) in weights.iter().enumerate() {
            threshold -= w;
            if threshold <= 0.0 {
                return Some(scored[i].0.clone());
            }
        }
        // Floating-point residue: fall through to the last candidate.
        Some(scored[scored.len() - 1].0.clone())
    }
}

impl Selector for SoftmaxSelector {
    fn select_one(&self, scored: &[(Arc<Worker>, f32)]) -> Option<Arc<Worker>> {
        let mut rng = StdRng::from_entropy();
        self.select_with_rng(scored, &mut rng)
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::worker_with_load;
    use super::*;

    fn scored(pairs: &[(&str, f32)]) -> Vec<(Arc<Worker>, f32)> {
        pairs
            .iter()
            .map(|(url, s)| (worker_with_load(url, 0), *s))
            .collect()
    }

    #[test]
    fn argmax_picks_highest() {
        let s = scored(&[("http://a", 0.1), ("http://b", 0.9), ("http://c", 0.5)]);
        let w = ArgmaxSelector::new().select_one(&s).unwrap();
        assert_eq!(w.url, "http://b");
    }

    #[test]
    fn argmax_tie_breaks_by_lowest_url() {
        // Equal scores with the higher URL listed first: the tie-break must
        // still return the lowest URL, proving it is URL- not order-based.
        let s = scored(&[("http://b", 0.7), ("http://a", 0.7)]);
        let w = ArgmaxSelector::new().select_one(&s).unwrap();
        assert_eq!(w.url, "http://a", "ties resolve to the lowest worker URL");
    }

    #[test]
    fn argmax_empty_is_none() {
        assert!(ArgmaxSelector::new().select_one(&[]).is_none());
    }

    #[test]
    fn softmax_is_deterministic_for_fixed_seed() {
        let s = scored(&[("http://a", 0.2), ("http://b", 0.8), ("http://c", 0.5)]);
        let picker = SoftmaxSelector::new(0.5);
        let mut r1 = StdRng::seed_from_u64(42);
        let mut r2 = StdRng::seed_from_u64(42);
        let a = picker.select_with_rng(&s, &mut r1).unwrap();
        let b = picker.select_with_rng(&s, &mut r2).unwrap();
        assert_eq!(a.url, b.url, "same seed → same pick");
    }

    #[test]
    fn softmax_low_temperature_favours_best() {
        // With a tiny temperature the sampler should overwhelmingly pick the
        // top-scored worker. Count over many draws from a fixed seed.
        let s = scored(&[("http://a", 0.0), ("http://b", 1.0)]);
        let picker = SoftmaxSelector::new(0.05);
        let mut rng = StdRng::seed_from_u64(7);
        let mut b_hits = 0;
        for _ in 0..1000 {
            if picker.select_with_rng(&s, &mut rng).unwrap().url == "http://b" {
                b_hits += 1;
            }
        }
        assert!(b_hits > 950, "low-T should pick best ~always, got {b_hits}/1000");
    }

    #[test]
    fn softmax_high_temperature_spreads() {
        // High temperature flattens toward uniform: both workers get picked.
        let s = scored(&[("http://a", 0.0), ("http://b", 1.0)]);
        let picker = SoftmaxSelector::new(100.0);
        let mut rng = StdRng::seed_from_u64(7);
        let mut a_hits = 0;
        let mut b_hits = 0;
        for _ in 0..1000 {
            match picker.select_with_rng(&s, &mut rng).unwrap().url.as_str() {
                "http://a" => a_hits += 1,
                "http://b" => b_hits += 1,
                other => panic!("unexpected {other}"),
            }
        }
        assert!(a_hits > 300 && b_hits > 300, "high-T spreads: a={a_hits} b={b_hits}");
    }

    #[test]
    fn softmax_nonpositive_temperature_is_argmax() {
        let s = scored(&[("http://a", 0.2), ("http://b", 0.9)]);
        let picker = SoftmaxSelector::new(0.0);
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(picker.select_with_rng(&s, &mut rng).unwrap().url, "http://b");
    }

    #[test]
    fn softmax_empty_is_none() {
        let picker = SoftmaxSelector::new(1.0);
        let mut rng = StdRng::seed_from_u64(1);
        assert!(picker.select_with_rng(&[], &mut rng).is_none());
    }
}
