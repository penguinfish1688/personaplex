#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "moshi"))

from moshi.modules.attention_suppression import apply_pre_interrupt_attention_suppression


def _logits(kv_len: int = 128) -> torch.Tensor:
    return torch.zeros(1, 2, 1, kv_len)


class AttentionSuppressionTest(unittest.TestCase):
    def test_noop_before_interruption(self):
        logits = _logits()
        original = logits.clone()
        out = apply_pre_interrupt_attention_suppression(
            logits,
            current_timestep=99,
            interrupt_timestep=100,
        )
        self.assertTrue(torch.equal(out, original))

    def test_suppression_during_window(self):
        logits = _logits()
        out = apply_pre_interrupt_attention_suppression(
            logits,
            current_timestep=100,
            interrupt_timestep=100,
            kv_cache_start_timestep=0,
            k_post_interrupt=10,
            n_pre_interrupt=30,
            lambda_suppression=0.2,
        )
        expected = _logits()
        expected[..., 70:100] += math.log(0.2)
        self.assertTrue(torch.equal(out, expected))

    def test_noop_after_k(self):
        logits = _logits()
        original = logits.clone()
        out = apply_pre_interrupt_attention_suppression(
            logits,
            current_timestep=110,
            interrupt_timestep=100,
            k_post_interrupt=10,
            n_pre_interrupt=30,
            lambda_suppression=0.2,
        )
        self.assertTrue(torch.equal(out, original))

    def test_sliding_kv_cache_contiguous_indexing(self):
        logits = _logits(kv_len=80)
        out = apply_pre_interrupt_attention_suppression(
            logits,
            current_timestep=100,
            interrupt_timestep=100,
            kv_cache_start_timestep=50,
            n_pre_interrupt=30,
            lambda_suppression=0.2,
        )
        expected = _logits(kv_len=80)
        expected[..., 20:50] += math.log(0.2)
        self.assertTrue(torch.equal(out, expected))

    def test_lambda_one_noop(self):
        logits = _logits()
        original = logits.clone()
        out = apply_pre_interrupt_attention_suppression(
            logits,
            current_timestep=100,
            interrupt_timestep=100,
            lambda_suppression=1.0,
        )
        self.assertTrue(torch.equal(out, original))

    def test_invalid_lambda(self):
        for bad_lambda in (0.0, -0.1, 1.1):
            with self.subTest(bad_lambda=bad_lambda):
                with self.assertRaisesRegex(ValueError, "lambda_suppression"):
                    apply_pre_interrupt_attention_suppression(
                        _logits(),
                        current_timestep=100,
                        interrupt_timestep=100,
                        lambda_suppression=bad_lambda,
                    )


if __name__ == "__main__":
    unittest.main()
