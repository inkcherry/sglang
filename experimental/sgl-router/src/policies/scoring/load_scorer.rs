// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! [`LoadScorer`] — prefer the least-loaded worker.
//!
//! Reads `Worker::active_load()` (in-flight request count) and normalizes it
//! across the candidate set so *lower load → higher score*, in `[0, 1]`.
//!
//! The load *source* here is the router-local `active_requests` counter. A
//! truer load signal (e.g. an engine-published load registry or an
//! uncached-token predictor) can be swapped in later; the scorer boundary
//! keeps that change local to this file.

use super::{Scorer, ScoreContext};
use crate::workers::Worker;
use std::sync::Arc;

/// Scores workers by (normalized, inverted) in-flight load.
#[derive(Debug, Clone)]
pub struct LoadScorer {
    weight: f32,
}

impl LoadScorer {
    /// `weight` is the non-negative multiplier applied in the weighted sum.
    /// In the default configuration load is the lighter, corrective term
    /// relative to prefix-cache locality; callers set the ratio by choosing
    /// weights across scorers.
    pub fn new(weight: f32) -> Self {
        Self { weight }
    }
}

impl Scorer for LoadScorer {
    fn score(&self, worker: &Arc<Worker>, ctx: &ScoreContext<'_>) -> f32 {
        let loads = ctx.candidates().iter().map(|w| w.active_load());
        let (mut min, mut max) = (usize::MAX, usize::MIN);
        for l in loads {
            min = min.min(l);
            max = max.max(l);
        }
        // No candidates (shouldn't happen; pipeline guards it) or all-equal
        // load → no load signal, return neutral 1.0 so this scorer neither
        // rewards nor penalizes.
        if min > max || max == min {
            return 1.0;
        }
        let load = worker.active_load();
        // Lower load → higher score. Linear map [min,max] → [1,0].
        (max - load) as f32 / (max - min) as f32
    }

    fn weight(&self) -> f32 {
        self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::worker_with_load;
    use super::super::ScoreContext;
    use super::*;
    use crate::discovery::ModelId;
    use crate::policies::SelectionContext;

    fn score_all(scorer: &LoadScorer, workers: &[Arc<Worker>]) -> Vec<f32> {
        let model = ModelId("m".into());
        let sel = SelectionContext::new(&model, None);
        let ctx = ScoreContext::new(&sel, workers);
        workers.iter().map(|w| scorer.score(w, &ctx)).collect()
    }

    #[test]
    fn least_loaded_scores_highest() {
        let workers = vec![
            worker_with_load("http://a", 0),
            worker_with_load("http://b", 5),
            worker_with_load("http://c", 10),
        ];
        let s = score_all(&LoadScorer::new(1.0), &workers);
        assert_eq!(s[0], 1.0, "load 0 (min) → 1.0");
        assert_eq!(s[2], 0.0, "load 10 (max) → 0.0");
        assert!((s[1] - 0.5).abs() < 1e-6, "load 5 (midpoint) → 0.5, got {}", s[1]);
    }

    #[test]
    fn all_equal_load_is_neutral() {
        let workers = vec![
            worker_with_load("http://a", 3),
            worker_with_load("http://b", 3),
        ];
        let s = score_all(&LoadScorer::new(1.0), &workers);
        assert_eq!(s, vec![1.0, 1.0], "flat load → neutral 1.0 for all");
    }

    #[test]
    fn weight_is_reported() {
        assert_eq!(LoadScorer::new(2.5).weight(), 2.5);
    }
}
