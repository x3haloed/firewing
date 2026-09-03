from __future__ import annotations

import unittest

import torch

from tools.analyze_block4_int8_real_layers import quantized_mixture
from tools.analyze_q2_lossless_experts import AnalysisError
from tools.generate_expert_fixture import expert_forward
from tools.generate_mixture_fixture import accumulate_bf16_in_expert_order


class TensorBank:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def get_slice(self, name: str) -> torch.Tensor:
        return self.tensors[name]


class Block4Int8RealLayerTests(unittest.TestCase):
    def test_constant_tiny_mixture_is_exact_and_accounts_bytes(self) -> None:
        hidden = torch.ones(4, dtype=torch.bfloat16)
        gate_up = torch.ones((2, 8, 4), dtype=torch.bfloat16)
        down = torch.ones((2, 4, 4), dtype=torch.bfloat16)
        bank = TensorBank({"gate": gate_up, "down": down})
        references = [
            expert_forward(
                hidden,
                gate_up[expert],
                down[expert],
                torch.tensor(0.5, dtype=torch.bfloat16),
            )["weighted_down"]
            for expert in (0, 1)
        ]
        result = quantized_mixture(
            hidden,
            [1, 0],
            [0.5, 0.5],
            bank,
            bank,
            "gate",
            "down",
            accumulate_bf16_in_expert_order(references),
        )
        self.assertEqual(result["mixture"]["relative_l2"], 0.0)
        self.assertEqual(result["artifact_to_source_ratio"], 0.625)

    def test_output_channel_oriented_topology_is_exact_on_constant_fixture(self) -> None:
        hidden = torch.ones(16, dtype=torch.bfloat16)
        gate_up = torch.ones((2, 32, 16), dtype=torch.bfloat16)
        down = torch.ones((2, 16, 16), dtype=torch.bfloat16)
        bank = TensorBank({"gate": gate_up, "down": down})
        references = [
            expert_forward(
                hidden,
                gate_up[expert],
                down[expert],
                torch.tensor(0.5, dtype=torch.bfloat16),
            )["weighted_down"]
            for expert in (0, 1)
        ]
        result = quantized_mixture(
            hidden,
            [0, 1],
            [0.5, 0.5],
            bank,
            bank,
            "gate",
            "down",
            accumulate_bf16_in_expert_order(references),
            (1, 16),
        )
        self.assertEqual(result["mixture"]["relative_l2"], 0.0)
        self.assertEqual(result["artifact_to_source_ratio"], 0.625)

    def test_affine_topology_accounts_zero_point_bytes(self) -> None:
        hidden = torch.ones(16, dtype=torch.bfloat16)
        gate_up = torch.ones((2, 32, 16), dtype=torch.bfloat16)
        down = torch.ones((2, 16, 16), dtype=torch.bfloat16)
        bank = TensorBank({"gate": gate_up, "down": down})
        references = [
            expert_forward(
                hidden,
                gate_up[expert],
                down[expert],
                torch.tensor(0.5, dtype=torch.bfloat16),
            )["weighted_down"]
            for expert in (0, 1)
        ]
        result = quantized_mixture(
            hidden,
            [0, 1],
            [0.5, 0.5],
            bank,
            bank,
            "gate",
            "down",
            accumulate_bf16_in_expert_order(references),
            (8, 2),
            "affine_uint8",
        )
        self.assertEqual(result["mixture"]["relative_l2"], 0.0)
        self.assertEqual(result["artifact_to_source_ratio"], 0.65625)

    def test_affine_exact_groups_have_larger_auditable_ledger(self) -> None:
        hidden = torch.ones(16, dtype=torch.bfloat16)
        gate_up = (
            torch.linspace(-2.0, 3.0, 2 * 32 * 16)
            .reshape(2, 32, 16)
            .to(torch.bfloat16)
        )
        down = (
            torch.linspace(-1.0, 2.0, 2 * 16 * 16)
            .reshape(2, 16, 16)
            .to(torch.bfloat16)
        )
        bank = TensorBank({"gate": gate_up, "down": down})
        references = [
            expert_forward(
                hidden,
                gate_up[expert],
                down[expert],
                torch.tensor(0.5, dtype=torch.bfloat16),
            )["weighted_down"]
            for expert in (0, 1)
        ]
        result = quantized_mixture(
            hidden,
            [0, 1],
            [0.5, 0.5],
            bank,
            bank,
            "gate",
            "down",
            accumulate_bf16_in_expert_order(references),
            (4, 4),
            "affine_uint8_exact_groups",
            400,
        )
        self.assertGreater(result["artifact_to_source_ratio"], 0.65625)
        self.assertEqual(result["artifact_to_source_ratio"], 0.7265625)

    def test_duplicate_route_fails_closed(self) -> None:
        hidden = torch.ones(4, dtype=torch.bfloat16)
        reference = torch.ones(4, dtype=torch.bfloat16)
        bank = TensorBank({})
        with self.assertRaises(AnalysisError):
            quantized_mixture(
                hidden,
                [0, 0],
                [0.5, 0.5],
                bank,
                bank,
                "gate",
                "down",
                reference,
            )


if __name__ == "__main__":
    unittest.main()
