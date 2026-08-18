"""Unit tests for HAREN-CTC masking and loss helpers."""

import sys
import unittest
from pathlib import Path

import torch
from transformers import Wav2Vec2FeatureExtractor


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from haren_ctc_utils import (  # noqa: E402
    aggregate_subject_predictions,
    build_ctc_targets,
    collapse_sliding_window_predictions,
    ctc_greedy_edit_counts,
    crop_and_pad_waveform,
    crop_waveform,
    deterministic_crop_start,
    effective_ctc_weight,
    EpochSeededWeightedSampler,
    evaluation_crop_start,
    feature_output_lengths,
    lengths_to_attention_mask,
    loss_gradient_diagnostics,
    masked_kmeans_batched,
    multi_view_crop_starts,
    named_parameter_hash,
    normalized_ctc_loss,
    normalized_frame_ce_loss,
    prepare_wavlm_inputs,
    routing_initial_logits,
    sliding_window_starts,
    subject_balanced_sample_weights,
    SharedGradientRatioController,
    temporal_logit_view,
    trainable_state_hash,
    waveform_quality_metrics,
)


class HarenCtcUtilsTest(unittest.TestCase):
    def test_wavlm_crop_pads_to_fixed_window_and_tracks_true_length(self):
        short = torch.arange(12).reshape(1, 12)
        long = torch.arange(30).reshape(1, 30)
        padded_short, short_valid = crop_and_pad_waveform(short, 20)
        cropped_long, long_valid = crop_and_pad_waveform(long, 20, start=5)
        self.assertEqual(padded_short.shape[-1], 20)
        self.assertEqual(short_valid, 12)
        self.assertTrue(torch.equal(padded_short[0, 12:], torch.zeros(8)))
        self.assertEqual(cropped_long.shape[-1], 20)
        self.assertEqual(long_valid, 20)
        self.assertEqual(cropped_long[0, 0].item(), 5)

    def test_crop_without_padding_preserves_short_length(self):
        short = torch.arange(12).reshape(1, 12)
        long = torch.arange(30).reshape(1, 30)
        self.assertEqual(crop_waveform(short, 20).shape[-1], 12)
        cropped = crop_waveform(long, 20, start=5)
        self.assertEqual(cropped.shape[-1], 20)
        self.assertEqual(cropped[0, 0].item(), 5)

    def test_valid_then_pad_normalizes_only_valid_wavlm_audio(self):
        extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        generator = torch.Generator().manual_seed(7)
        valid = 0.03 * torch.randn(3200, generator=generator) + 0.01
        legacy, legacy_mask = prepare_wavlm_inputs(
            [valid],
            [valid.numel()],
            extractor,
            preprocess_policy="legacy_prepad",
            mask_policy="true_length",
            padding_policy="fixed_10s",
            max_samples=16000,
        )
        repaired, repaired_mask = prepare_wavlm_inputs(
            [valid],
            [valid.numel()],
            extractor,
            preprocess_policy="valid_then_pad",
            mask_policy="true_length",
            padding_policy="fixed_10s",
            max_samples=16000,
        )
        self.assertTrue(torch.equal(legacy_mask, repaired_mask))
        self.assertEqual(repaired_mask.sum().item(), valid.numel())
        self.assertAlmostEqual(
            float(repaired[0, : valid.numel()].mean()), 0.0, places=5
        )
        self.assertAlmostEqual(
            float(repaired[0, : valid.numel()].std(unbiased=False)),
            1.0,
            places=3,
        )
        self.assertTrue(
            torch.equal(
                repaired[0, valid.numel() :],
                torch.zeros(16000 - valid.numel()),
            )
        )
        self.assertGreater(
            float(legacy[0, : valid.numel()].std(unbiased=False)), 1.5
        )
        self.assertGreater(
            float((legacy - repaired).abs().mean()), 0.1
        )

    def test_dynamic_wavlm_padding_uses_longest_valid_wave(self):
        extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        inputs, mask = prepare_wavlm_inputs(
            [torch.arange(12).float(), torch.arange(20).float()],
            [12, 20],
            extractor,
            preprocess_policy="valid_then_pad",
            mask_policy="true_length",
            padding_policy="longest",
            max_samples=30,
        )
        self.assertEqual(inputs.shape, (2, 20))
        self.assertEqual(mask.sum(dim=1).tolist(), [12, 20])
        self.assertTrue(torch.equal(inputs[0, 12:], torch.zeros(8)))

    def test_epoch_keyed_sampler_and_crop_are_restart_stable(self):
        sampler = EpochSeededWeightedSampler(
            [1.0, 2.0, 1.0], num_samples=8, seed=123
        )
        sampler.set_epoch(4)
        first = list(sampler)
        sampler.set_epoch(4)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(5)
        self.assertNotEqual(first, list(sampler))
        start = deterministic_crop_start(
            "subject_001.wav",
            320000,
            160000,
            seed=123,
            epoch=4,
            draw=2,
        )
        self.assertEqual(
            start,
            deterministic_crop_start(
                "subject_001.wav",
                320000,
                160000,
                seed=123,
                epoch=4,
                draw=2,
            ),
        )
        self.assertGreaterEqual(start, 0)
        self.assertLessEqual(start, 160000)
        aligned = deterministic_crop_start(
            "subject_001.wav",
            320000,
            160000,
            seed=123,
            epoch=4,
            draw=2,
            alignment=320,
        )
        self.assertEqual(aligned % 320, 0)
        self.assertLessEqual(aligned, 160000)

    def test_sliding_windows_cover_audio_without_gaps(self):
        self.assertEqual(sliding_window_starts(12, 20, 10), [0])
        self.assertEqual(sliding_window_starts(23, 20, 10), [0, 3])
        self.assertEqual(
            sliding_window_starts(45, 20, 10), [0, 10, 20, 25]
        )
        with self.assertRaises(ValueError):
            sliding_window_starts(45, 20, 21)

    def test_multi_view_starts_are_unique_and_aligned(self):
        self.assertEqual(multi_view_crop_starts(50, 100), [0])
        self.assertEqual(multi_view_crop_starts(200, 100), [0, 50, 100])
        self.assertEqual(
            multi_view_crop_starts(1000, 100, alignment=320),
            [0, 320, 640],
        )

    def test_sliding_predictions_average_per_utterance_first(self):
        paths, subjects, labels, probabilities = (
            collapse_sliding_window_predictions(
                [
                    "/audio/300_1__window0000.wav",
                    "/audio/300_1__window0001.wav",
                    "/audio/300_2__window0000.wav",
                ],
                ["300", "300", "300"],
                [0, 0, 0],
                [0.2, 0.6, 0.8],
            )
        )
        self.assertEqual(paths, ["/audio/300_1.wav", "/audio/300_2.wav"])
        self.assertEqual(subjects, ["300", "300"])
        self.assertEqual(labels, [0, 0])
        self.assertEqual(probabilities, [0.4, 0.8])

    def test_ctc_uses_true_mask_while_wavlm_keeps_full_window(self):
        wavlm_mask = lengths_to_attention_mask(
            torch.tensor([160000, 160000]), max_length=160000
        )
        self.assertEqual(wavlm_mask.sum(dim=1).tolist(), [160000, 160000])
        mask = lengths_to_attention_mask(
            torch.tensor([16000, 160000]), max_length=160000
        )
        self.assertEqual(mask.shape, (2, 160000))
        self.assertEqual(mask[0].sum().item(), 16000)
        self.assertEqual(mask[1].sum().item(), 160000)
        self.assertEqual(mask[0, 15999].item(), 1)
        self.assertEqual(mask[0, 16000].item(), 0)

    def test_feature_lengths_match_wavlm_stack(self):
        kernels = [10, 3, 3, 3, 3, 2, 2]
        strides = [5, 2, 2, 2, 2, 2, 2]
        result = feature_output_lengths(
            torch.tensor([16000, 80000, 160000]), kernels, strides
        )
        self.assertEqual(result.tolist(), [49, 249, 499])

    def test_masked_kmeans_ignores_padded_tail(self):
        torch.manual_seed(7)
        valid = torch.randn(2, 12, 4)
        first = torch.cat([valid, torch.zeros(2, 8, 4)], dim=1)
        second = torch.cat([valid, torch.full((2, 8, 4), 999.0)], dim=1)
        lengths = torch.tensor([12, 12])
        labels_a = masked_kmeans_batched(first, lengths, n_clusters=3)
        labels_b = masked_kmeans_batched(second, lengths, n_clusters=3)
        self.assertTrue(torch.equal(labels_a[:, :12], labels_b[:, :12]))

    def test_targets_respect_lengths_and_modes(self):
        clusters = torch.tensor(
            [[0, 0, 1, 1, 2, 9], [2, 2, 3, 3, 4, 8]]
        )
        lengths = torch.tensor([5, 5])
        labels = torch.tensor([0, 1])
        shifted, shifted_lengths = build_ctc_targets(
            clusters, lengths, labels, 10, "label_shifted"
        )
        neutral, neutral_lengths = build_ctc_targets(
            clusters, lengths, labels, 10, "neutral"
        )
        self.assertEqual(shifted_lengths.tolist(), [3, 3])
        self.assertEqual(neutral_lengths.tolist(), [3, 3])
        self.assertEqual(shifted.tolist(), [0, 1, 2, 12, 13, 14])
        self.assertEqual(neutral.tolist(), [0, 1, 2, 2, 3, 4])
        self.assertLess(max(neutral.tolist()), 10)
        self.assertLess(max(shifted.tolist()), 20)

    def test_targets_support_stride_and_temporal_reverse_control(self):
        clusters = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]])
        targets, lengths = build_ctc_targets(
            clusters,
            torch.tensor([8]),
            torch.tensor([0]),
            10,
            "neutral",
            unit_stride=2,
        )
        reversed_targets, reversed_lengths = build_ctc_targets(
            clusters,
            torch.tensor([8]),
            torch.tensor([0]),
            10,
            "neutral",
            unit_stride=2,
            reverse=True,
        )
        self.assertEqual(targets.tolist(), [0, 1, 2, 3])
        self.assertEqual(reversed_targets.tolist(), [3, 2, 1, 0])
        self.assertEqual(lengths.tolist(), reversed_lengths.tolist())

    def test_adaptive_weight_caps_ctc_term(self):
        cls_loss = torch.tensor(0.5)
        ctc_loss = torch.tensor(10.0)
        weight = effective_ctc_weight(
            cls_loss=cls_loss,
            ctc_loss=ctc_loss,
            mode="adaptive_ratio",
            max_weight=0.05,
            warmup_factor=1.0,
            target_ratio=0.1,
        )
        self.assertAlmostEqual(weight * ctc_loss.item(), 0.05)
        warmup_weight = effective_ctc_weight(
            cls_loss=cls_loss,
            ctc_loss=ctc_loss,
            mode="adaptive_ratio",
            max_weight=0.05,
            warmup_factor=0.2,
            target_ratio=0.1,
        )
        self.assertAlmostEqual(warmup_weight * ctc_loss.item(), 0.01)

    def test_fixed_weight_uses_warmup(self):
        weight = effective_ctc_weight(
            cls_loss=torch.tensor(1.0),
            ctc_loss=torch.tensor(3.0),
            mode="fixed",
            max_weight=0.005,
            warmup_factor=0.4,
            target_ratio=0.1,
        )
        self.assertAlmostEqual(weight, 0.002)

    def test_ctc_off_weight_is_zero(self):
        weight = effective_ctc_weight(
            cls_loss=torch.tensor(1.0),
            ctc_loss=torch.tensor(100.0),
            mode="fixed",
            max_weight=0.0,
            warmup_factor=1.0,
            target_ratio=0.1,
        )
        self.assertEqual(weight, 0.0)

    def test_reseed_makes_paired_head_initialization_identical(self):
        torch.manual_seed(99)
        _ = torch.randn(100)
        torch.manual_seed(123)
        first = torch.nn.Linear(8, 3)
        first_hash = trainable_state_hash(first)
        torch.manual_seed(17)
        _ = torch.randn(250)
        torch.manual_seed(123)
        second = torch.nn.Linear(8, 3)
        self.assertEqual(first_hash, trainable_state_hash(second))
        first_shared = named_parameter_hash(
            [("weight", first.weight)]
        )
        second_shared = named_parameter_hash(
            [("weight", second.weight)]
        )
        self.assertEqual(first_shared, second_shared)

    def test_probability_correct_routing_matches_requested_probabilities(self):
        logits = routing_initial_logits(
            4, alpha=0.8, policy="probability_correct"
        )
        probabilities = torch.softmax(logits, dim=-1)[:, 0]
        self.assertTrue(
            torch.allclose(
                probabilities,
                torch.tensor([0.8, 0.64, 0.512, 0.4096]),
                atol=1e-6,
            )
        )
        self.assertGreater(float(probabilities[0]), 0.0)
        self.assertLess(float(probabilities[0]), 1.0)
        legacy = torch.softmax(
            routing_initial_logits(2, alpha=0.8, policy="legacy"), dim=-1
        )
        self.assertGreater(float(legacy[0, 0]), 0.999)

    def test_subject_balanced_weights_equalize_class_and_subject_mass(self):
        labels = [0, 0, 0, 1, 1, 1, 1]
        subjects = ["a", "a", "b", "c", "c", "c", "d"]
        weights = subject_balanced_sample_weights(labels, subjects)
        self.assertAlmostEqual(float(weights[:3].sum()), 1.0)
        self.assertAlmostEqual(float(weights[3:].sum()), 1.0)
        masses = {}
        for subject, weight in zip(subjects, weights):
            masses[subject] = masses.get(subject, 0.0) + float(weight)
        self.assertAlmostEqual(masses["a"], masses["b"])
        self.assertAlmostEqual(masses["c"], masses["d"])

    def test_subject_aggregation_supports_confidence_weighting(self):
        probabilities = [0.51, 0.99, 0.4, 0.6]
        subjects = ["10", "10", "20", "20"]
        labels = [1, 1, 0, 0]
        mean_scores, subject_labels, ordered = aggregate_subject_predictions(
            probabilities, subjects, labels, policy="mean_probability"
        )
        weighted_scores, _, _ = aggregate_subject_predictions(
            probabilities,
            subjects,
            labels,
            policy="confidence_weighted_probability",
        )
        vote_scores, _, _ = aggregate_subject_predictions(
            probabilities,
            subjects,
            labels,
            policy="confidence_weighted_vote",
        )
        self.assertEqual(ordered, ["10", "20"])
        self.assertEqual(subject_labels, [1, 0])
        self.assertAlmostEqual(mean_scores[0], 0.75)
        self.assertGreater(weighted_scores[0], mean_scores[0])
        self.assertAlmostEqual(vote_scores[0], 1.0)

    def test_gradient_diagnostics_detect_alignment_and_conflict(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        primary = parameter.square().sum()
        aligned = 2.0 * parameter.square().sum()
        aligned_result = loss_gradient_diagnostics(
            primary, aligned, [parameter]
        )
        self.assertAlmostEqual(aligned_result["gradient_cosine"], 1.0)
        conflict = -parameter.square().sum()
        conflict_result = loss_gradient_diagnostics(
            primary, conflict, [parameter]
        )
        self.assertAlmostEqual(conflict_result["gradient_cosine"], -1.0)

    def test_shared_gradient_controller_targets_ratio_and_caps_loss(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        primary = parameter.square().sum()
        auxiliary = (10.0 * parameter).square().sum()
        controller = SharedGradientRatioController(
            target_ratio=0.1,
            max_weight=1.0,
            loss_ratio_cap=0.05,
            warmup_steps=0,
            update_interval=10,
            ema_decay=0.9,
        )
        weight, diagnostics = controller.weight(
            primary_loss=primary,
            auxiliary_loss=auxiliary,
            shared_parameters=[parameter],
            step_index=0,
        )
        self.assertLessEqual(
            weight * auxiliary.item() / primary.item(), 0.05 + 1e-7
        )
        self.assertTrue(diagnostics["gradient_balance_updated"])
        restored = SharedGradientRatioController(
            target_ratio=0.1,
            max_weight=1.0,
            loss_ratio_cap=0.05,
            warmup_steps=0,
        )
        restored.load_state_dict(controller.state_dict())
        self.assertEqual(restored.state_dict(), controller.state_dict())

    def test_shared_gradient_controller_prevents_ema_overshoot(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        controller = SharedGradientRatioController(
            target_ratio=0.1,
            max_weight=100.0,
            loss_ratio_cap=1000.0,
            warmup_steps=0,
            update_interval=1,
            ema_decay=0.9,
        )
        controller.weight(
            primary_loss=100.0 * parameter,
            auxiliary_loss=parameter,
            shared_parameters=[parameter],
            step_index=0,
        )
        weight, diagnostics = controller.weight(
            primary_loss=parameter,
            auxiliary_loss=100.0 * parameter,
            shared_parameters=[parameter],
            step_index=1,
        )
        self.assertLessEqual(
            diagnostics[
                "weighted_auxiliary_to_primary_gradient_ratio"
            ],
            0.1 + 1e-7,
        )
        self.assertLessEqual(weight, 0.001 + 1e-7)

    def test_shared_gradient_controller_measures_nonupdate_batches(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        controller = SharedGradientRatioController(
            target_ratio=0.1,
            max_weight=100.0,
            loss_ratio_cap=1000.0,
            warmup_steps=0,
            update_interval=10,
            ema_decay=0.9,
        )
        controller.weight(
            primary_loss=100.0 * parameter,
            auxiliary_loss=parameter,
            shared_parameters=[parameter],
            step_index=0,
        )
        weight, diagnostics = controller.weight(
            primary_loss=parameter,
            auxiliary_loss=100.0 * parameter,
            shared_parameters=[parameter],
            step_index=1,
        )
        self.assertFalse(diagnostics["gradient_balance_updated"])
        self.assertEqual(
            diagnostics["gradient_balance_measurement_age_steps"], 0.0
        )
        self.assertEqual(diagnostics["gradient_balance_ema_age_steps"], 1.0)
        self.assertLessEqual(
            diagnostics[
                "weighted_auxiliary_to_primary_gradient_ratio"
            ],
            0.1 + 1e-7,
        )
        self.assertLessEqual(weight, 0.001 + 1e-7)

    def test_waveform_quality_reports_speech_fraction(self):
        waveform = torch.cat(
            [torch.zeros(320), torch.full((320,), 0.1)]
        )
        quality = waveform_quality_metrics(waveform)
        self.assertAlmostEqual(quality["speech_frame_fraction"], 0.5)
        self.assertGreater(quality["rms"], 0.0)

    def test_evaluation_crop_start_is_deterministic(self):
        self.assertEqual(evaluation_crop_start(200, 100, "head"), 0)
        self.assertEqual(evaluation_crop_start(200, 100, "center"), 50)
        self.assertEqual(evaluation_crop_start(200, 100, "tail"), 100)
        self.assertEqual(evaluation_crop_start(50, 100, "tail"), 0)
        self.assertEqual(
            evaluation_crop_start(1000, 100, "tail", alignment=320), 640
        )

    def test_temporal_logit_view_removes_impossible_classes(self):
        logits = torch.arange(2 * 3 * 21).reshape(2, 3, 21).float()
        neutral_ctc, blank = temporal_logit_view(
            logits, k=10, target_mode="neutral", framewise=False
        )
        self.assertEqual(neutral_ctc.shape, (2, 3, 11))
        self.assertEqual(blank, 10)
        self.assertTrue(
            torch.equal(neutral_ctc[:, :, -1], logits[:, :, -1])
        )
        neutral_frame, blank = temporal_logit_view(
            logits, k=10, target_mode="neutral", framewise=True
        )
        self.assertEqual(neutral_frame.shape, (2, 3, 10))
        self.assertIsNone(blank)
        shifted_frame, _ = temporal_logit_view(
            logits, k=10, target_mode="label_shifted", framewise=True
        )
        self.assertEqual(shifted_frame.shape, (2, 3, 20))
        compact_neutral = torch.randn(2, 3, 11)
        compact_ctc, blank = temporal_logit_view(
            compact_neutral, k=10, target_mode="neutral", framewise=False
        )
        self.assertIs(compact_ctc, compact_neutral)
        self.assertEqual(blank, 10)
        with self.assertRaises(ValueError):
            temporal_logit_view(
                compact_neutral,
                k=10,
                target_mode="label_shifted",
                framewise=False,
            )

    def test_normalized_fp32_ctc_prefers_correct_order_and_backpropagates(self):
        logits = torch.full((1, 5, 3), -4.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 0] = 4.0
            logits[0, 1, 0] = 4.0
            logits[0, 2, 2] = 4.0
            logits[0, 3, 1] = 4.0
            logits[0, 4, 1] = 4.0
        lengths = torch.tensor([5])
        correct, per_example = normalized_ctc_loss(
            logits,
            torch.tensor([0, 1]),
            lengths,
            torch.tensor([2]),
            blank=2,
        )
        reversed_loss, _ = normalized_ctc_loss(
            logits,
            torch.tensor([1, 0]),
            lengths,
            torch.tensor([2]),
            blank=2,
        )
        self.assertEqual(correct.dtype, torch.float32)
        self.assertEqual(per_example.shape, (1,))
        self.assertLess(float(correct), float(reversed_loss))
        correct.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_greedy_ctc_unit_error_collapses_repeats_and_blank(self):
        logits = torch.full((2, 6, 4), -5.0)
        first = [0, 0, 3, 1, 1, 3]
        second = [2, 2, 3, 1, 1, 3]
        for batch, sequence in enumerate((first, second)):
            for time, token in enumerate(sequence):
                logits[batch, time, token] = 5.0
        result = ctc_greedy_edit_counts(
            logits,
            torch.tensor([0, 1, 1, 2]),
            torch.tensor([6, 6]),
            torch.tensor([2, 2]),
            blank=3,
        )
        self.assertEqual(result["target_tokens"], 4)
        self.assertEqual(result["predicted_tokens"], 4)
        self.assertEqual(result["edit_distance"], 2)
        self.assertEqual(result["exact_sequences"], 1)

    def test_frame_ce_normalizes_each_utterance_before_batch_mean(self):
        logits = torch.tensor(
            [
                [[4.0, -4.0], [-4.0, 4.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            requires_grad=True,
        )
        targets = torch.tensor([[0, 1, -100], [0, 1, 0]])
        loss, per_example = normalized_frame_ce_loss(
            logits, targets, torch.tensor([2, 3])
        )
        self.assertAlmostEqual(
            float(loss), float(per_example.mean()), places=7
        )
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
