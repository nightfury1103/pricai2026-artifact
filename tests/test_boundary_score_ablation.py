import unittest

import pandas as pd

import build_boundary_focus_pack as boundary
import build_boundary_score_ablation as ablation
import build_judge_cqa_outstanding_pack as cqa


class BoundaryScoreComponentTest(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "source": "historical",
            "premise": "A child runs through a park.",
            "hypothesis": "A child is outdoors.",
            "rationale": (
                "The correct answer is entailment because a park is outdoors, "
                "so the answer is entailment."
            ),
        }

    def test_disabling_each_component_removes_only_its_weighted_contribution(self):
        full_score, contributions = boundary.candidate_quality(
            self.candidate,
            training_label="entailment",
            vote_margin=2.5,
            agreement_count=5,
            total_candidates=7,
            return_contributions=True,
        )

        self.assertEqual(set(boundary.SCORE_COMPONENTS), set(contributions))
        self.assertAlmostEqual(full_score, sum(contributions.values()))

        for component, contribution in contributions.items():
            with self.subTest(component=component):
                ablated_score = boundary.candidate_quality(
                    self.candidate,
                    training_label="entailment",
                    vote_margin=2.5,
                    agreement_count=5,
                    total_candidates=7,
                    disabled_components={component},
                )
                self.assertAlmostEqual(ablated_score, full_score - contribution)

    def test_unknown_component_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown score components"):
            boundary.candidate_quality(
                self.candidate,
                training_label="entailment",
                vote_margin=2.5,
                agreement_count=5,
                total_candidates=7,
                disabled_components={"unknown"},
            )


class PaperConfigIntegrationTest(unittest.TestCase):
    def test_esnli_boundary_scoring_uses_paper_config_objects(self):
        config = boundary.PAPER_CONFIG

        self.assertEqual(
            boundary.SCORING_COEFFICIENTS,
            config["rationale_quality_score"]["table_2_coefficients"],
        )
        self.assertEqual(
            boundary.BOUNDARY_SOURCE_PRIOR,
            config["source_prior_pi_D"]["values"],
        )
        self.assertEqual(
            boundary.ESNLI_BOUNDARY_THRESHOLDS,
            config["difficulty_band_thresholds"]["esnli_boundary_mix"],
        )
        self.assertEqual(
            boundary.ESNLI_BOUNDARY_MIX_CONFIG,
            config["boundary_mix_selection"]["esnli"],
        )


    def test_esnli_judge_priors_are_loaded_from_config(self):
        import build_judge_esnli as esnli

        config = esnli.PAPER_CONFIG
        self.assertEqual(esnli.ESNLI_SOURCE_PRIORS, config["esnli_source_priors"])
        self.assertEqual(esnli.TYPE_PRIOR, config["esnli_source_priors"]["TYPE_PRIOR"])
        self.assertEqual(
            esnli.HYBRID_MULTIVIEW_TYPE_PRIOR,
            config["esnli_source_priors"]["HYBRID_MULTIVIEW_TYPE_PRIOR"],
        )

    def test_cqa_pack_profiles_are_loaded_from_config(self):
        config = cqa.PAPER_CONFIG
        self.assertEqual(cqa.CQA_SOURCE_PRIORS, config["cqa_source_priors"])
        self.assertEqual(cqa.SUPERCLEAN_SOURCE_PRIOR, config["cqa_source_priors"]["SUPERCLEAN_SOURCE_PRIOR"])
        self.assertEqual(cqa.CQA_BASE_PROFILES, config["cqa_pack_profiles"]["base_profiles"])
        self.assertEqual(cqa.CQA_MAIN_PACK_CONFIG, config["cqa_pack_profiles"]["main_packs"])

    def test_cqa_derived_pack_settings_are_loaded_from_config(self):
        config = cqa.PAPER_CONFIG
        self.assertEqual(cqa.CQA_SHORTCUT_SCORING_PROFILE, config["cqa_shortcut_scoring_profile"])
        self.assertEqual(cqa.CQA_SHORTCUT_SECONDARY_CONFIG, config["cqa_shortcut_secondary"])
        self.assertEqual(cqa.CQA_DERIVED_PACK_CONFIG, config["cqa_derived_pack_profiles"])

    def test_esnli_strategy_settings_are_loaded_from_config(self):
        import build_judge_esnli as esnli

        config = esnli.PAPER_CONFIG
        self.assertEqual(esnli.ESNLI_LENGTH_SCORE_CONFIG, config["esnli_length_score"])
        self.assertEqual(esnli.ESNLI_SCORING_STRATEGIES, config["esnli_scoring_strategies"])
        self.assertEqual(esnli.ESNLI_KEEP_RULES, config["esnli_keep_rules"])

    def test_cqa_boundary_scoring_uses_paper_config_objects(self):
        config = cqa.PAPER_CONFIG

        self.assertEqual(
            cqa.BASE_SOURCE_PRIOR,
            config["cqa_boundary_scoring_profile"]["source_prior"],
        )
        self.assertEqual(
            cqa.CQA_BOUNDARY_PROFILE["cue_weight"],
            config["cqa_boundary_scoring_profile"]["cue_weight"],
        )
        self.assertEqual(
            cqa.CQA_BOUNDARY_THRESHOLDS,
            config["difficulty_band_thresholds"]["cqa_boundary_mix"],
        )


class BoundarySelectionChangeTest(unittest.TestCase):
    @staticmethod
    def row(example, rationale, label="entailment", source="causal"):
        return {
            "premise": example,
            "hypothesis": f"{example} hypothesis",
            "rationale": rationale,
            "LLM_answer": label,
            "judge_source": source,
        }

    def test_selection_change_counts_changed_rationales_and_examples(self):
        full = pd.DataFrame(
            [
                self.row("one", "rationale a"),
                self.row("two", "rationale b"),
            ]
        )
        variant = pd.DataFrame(
            [
                self.row("one", "rationale changed", source="if_else"),
                self.row("three", "rationale c"),
            ]
        )

        report = ablation.compare_selection(full, variant)

        self.assertEqual(report["full_rows"], 2)
        self.assertEqual(report["variant_rows"], 2)
        self.assertEqual(report["shared_selected_rows"], 0)
        self.assertEqual(report["changed_selected_rows"], 2)
        self.assertEqual(report["shared_examples"], 1)
        self.assertEqual(report["changed_examples"], 1)
        self.assertEqual(report["selection_change"], 1.0)
        self.assertEqual(report["example_change"], 0.5)

    def test_rank_reports_orders_largest_selection_change_first(self):
        reports = {
            "source": {"selection_change": 0.2, "changed_selected_rows": 20},
            "ground": {"selection_change": 0.5, "changed_selected_rows": 50},
            "cue": {"selection_change": 0.1, "changed_selected_rows": 10},
        }

        ranked = ablation.rank_reports(reports)

        self.assertEqual(["ground", "source", "cue"], [item["component"] for item in ranked])


class CqaBoundaryScoreComponentTest(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "source": "causal",
            "premise": "Where would a bird build a nest?",
            "hypothesis": "['tree', 'ocean', 'desk', 'road', 'car']",
            "rationale": "The correct answer is tree because birds build nests in trees.",
        }
        self.profile = cqa.CQA_BOUNDARY_PROFILE

    def test_cqa_disabling_each_component_removes_only_its_weighted_contribution(self):
        full_score, contributions = cqa.score_candidate(
            self.candidate,
            gold_label="tree",
            vote_margin=2.5,
            agreement_count=5,
            total_candidates=7,
            source_prior=cqa.BASE_SOURCE_PRIOR,
            profile=self.profile,
            return_contributions=True,
        )

        self.assertEqual(set(boundary.SCORE_COMPONENTS), set(contributions))
        self.assertAlmostEqual(full_score, sum(contributions.values()))

        for component, contribution in contributions.items():
            with self.subTest(component=component):
                ablated_score = cqa.score_candidate(
                    self.candidate,
                    gold_label="tree",
                    vote_margin=2.5,
                    agreement_count=5,
                    total_candidates=7,
                    source_prior=cqa.BASE_SOURCE_PRIOR,
                    profile=self.profile,
                    disabled_components={component},
                )
                self.assertAlmostEqual(ablated_score, full_score - contribution)

    def test_cqa_component_split_preserves_the_existing_full_score(self):
        full_score = cqa.score_candidate(
            self.candidate,
            gold_label="tree",
            vote_margin=2.5,
            agreement_count=5,
            total_candidates=7,
            source_prior=cqa.BASE_SOURCE_PRIOR,
            profile=self.profile,
        )
        rationale_lower = self.candidate["rationale"].lower()
        overlap = cqa.support_overlap(
            self.candidate["premise"],
            self.candidate["hypothesis"],
            self.candidate["rationale"],
        )
        legacy_cue_hits = sum(1 for cue in cqa.SHORTCUT_CUES if cue in rationale_lower)
        expected_legacy_score = (
            cqa.BASE_SOURCE_PRIOR[self.candidate["source"]]
            + self.profile["overlap_weight"] * overlap
            + self.profile["agreement_weight"] * 5
            + self.profile["agreement_ratio_weight"] * (5 / 7)
            + self.profile["margin_weight"] * 2.5
            + self.profile["explicit_weight"]
            + self.profile["cue_weight"] * legacy_cue_hits
            + self.profile["preferred_source_bonus"]
            + self.profile["brevity_positive"]
        )

        self.assertAlmostEqual(full_score, expected_legacy_score)


if __name__ == "__main__":
    unittest.main()
