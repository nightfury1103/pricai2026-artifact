import argparse
import json
import math
import os
import re
from pathlib import Path

import pandas as pd

from config_utils import load_config


API_DIR = Path(os.environ.get("ESNLI_API_DIR", "[API] ESNLI"))
DATASET_DIR = Path(os.environ.get("ESNLI_DATASET_DIR", "datasets/esnli"))
PAPER_CONFIG = load_config()
DEFAULT_SOURCES = PAPER_CONFIG["esnli_sources"]
ESNLI_SOURCE_PRIORS = PAPER_CONFIG["esnli_source_priors"]
ESNLI_PREFERRED_SOURCES = PAPER_CONFIG["esnli_preferred_sources"]
ESNLI_LENGTH_SCORE_CONFIG = PAPER_CONFIG["esnli_length_score"]
ESNLI_COMMON_SCORING = PAPER_CONFIG["esnli_common_scoring"]
ESNLI_SCORING_STRATEGIES = PAPER_CONFIG["esnli_scoring_strategies"]
ESNLI_SECONDARY_MULTIVIEW = PAPER_CONFIG["esnli_secondary_multiview"]
ESNLI_TRAINING_VOTE_MARGIN_THRESHOLDS = PAPER_CONFIG["esnli_training_vote_margin_thresholds"]
ESNLI_KEEP_RULES = PAPER_CONFIG["esnli_keep_rules"]
THESIS_PREFERRED_SOURCES = ESNLI_PREFERRED_SOURCES["THESIS_PREFERRED_SOURCES"]
AGREEMENT_PREFERRED_SOURCES = ESNLI_PREFERRED_SOURCES["AGREEMENT_PREFERRED_SOURCES"]
GUARDED_PREFERRED_SOURCES = ESNLI_PREFERRED_SOURCES["GUARDED_PREFERRED_SOURCES"]
TYPE_PRIOR = ESNLI_SOURCE_PRIORS["TYPE_PRIOR"]
THESIS_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["THESIS_TYPE_PRIOR"]
AGREEMENT_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["AGREEMENT_TYPE_PRIOR"]
GUARDED_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["GUARDED_TYPE_PRIOR"]
BALANCED_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["BALANCED_TYPE_PRIOR"]
LABEL_PRIORITY_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["LABEL_PRIORITY_TYPE_PRIOR"]
LABEL_EXPERT_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["LABEL_EXPERT_TYPE_PRIOR"]
STUDENT_SIGNAL_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["STUDENT_SIGNAL_TYPE_PRIOR"]
ELITE_MULTIVIEW_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["ELITE_MULTIVIEW_TYPE_PRIOR"]
HARDCLEAN_MULTIVIEW_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["HARDCLEAN_MULTIVIEW_TYPE_PRIOR"]
HYBRID_MULTIVIEW_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["HYBRID_MULTIVIEW_TYPE_PRIOR"]
SINGLEVIEW_SUPERCLEAN_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["SINGLEVIEW_SUPERCLEAN_TYPE_PRIOR"]
SOURCEBLEND_MULTIVIEW_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["SOURCEBLEND_MULTIVIEW_TYPE_PRIOR"]
SINGLEVIEW_DIVERSE_TYPE_PRIOR = ESNLI_SOURCE_PRIORS["SINGLEVIEW_DIVERSE_TYPE_PRIOR"]

LABEL_NORMALIZATION = {
    "entailment": "entailment",
    "entailed": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
    "contradicted": "contradiction",
}

REASONING_CUES = (
    "because",
    "therefore",
    "however",
    "but",
    "although",
    "so the answer is",
    "the correct answer",
    "implies",
    "not necessarily",
)

LABEL_TEACHING_CUES = {
    "entailment": (
        "entail",
        "supported",
        "support",
        "more general",
        "therefore",
        "so the answer is entailment",
        "means that",
    ),
    "neutral": (
        "not enough information",
        "does not specify",
        "not specified",
        "not necessarily",
        "could be",
        "might be",
        "possible",
        "unclear",
    ),
    "contradiction": (
        "contradiction",
        "contradicts",
        "cannot",
        "can't",
        "opposite",
        "incompatible",
        "different",
        "not the same",
    ),
}


def normalize_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_key(premise, hypothesis):
    return normalize_text(premise).lower() + "</s>" + normalize_text(hypothesis).lower()


def normalize_label(label):
    normalized = normalize_text(label).lower()
    normalized = normalized.replace("**", "").replace(".", "")
    if normalized in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[normalized]
    if "contrad" in normalized:
        return "contradiction"
    if "neutral" in normalized:
        return "neutral"
    if "entail" in normalized:
        return "entailment"
    return normalized


def tokenize_for_overlap(text):
    return set(re.findall(r"[a-z]+", normalize_text(text).lower()))


def rationale_jaccard(left, right):
    left_tokens = tokenize_for_overlap(left)
    right_tokens = tokenize_for_overlap(right)
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def piecewise_score(word_count, schedule):
    for rule in schedule:
        if "max_words" not in rule or word_count <= rule["max_words"]:
            return rule["score"]
    return schedule[-1]["score"]


def preferred_sources(name):
    if not name:
        return ()
    return ESNLI_PREFERRED_SOURCES[name]


def word_condition_bonus(word_count, config):
    if not config:
        return 0.0
    min_words = config.get("min_words", 0)
    max_words = config.get("max_words")
    if max_words is not None and min_words <= word_count <= max_words:
        return config["positive"]
    if "long_word_min" in config and word_count >= config["long_word_min"]:
        return config["long_penalty"]
    return 0.0


def explicit_label_count(candidate, training_label):
    return candidate["rationale"].lower().count(training_label)


def keep_clause_passes(clause, matched_count, support_margin, best, high_quality_source, training_label):
    if matched_count < clause.get("min_matched_count", 0):
        return False
    if support_margin < clause.get("min_support_margin", float("-inf")):
        return False
    if best["judge_score"] < clause.get("min_judge_score", float("-inf")):
        return False
    if clause.get("require_high_quality_source") and not high_quality_source:
        return False
    if clause.get("require_explicit_label_support") and explicit_label_count(best, training_label) < 1:
        return False
    if "word_count_min" in clause and best["word_count"] < clause["word_count_min"]:
        return False
    if "word_count_max" in clause and best["word_count"] > clause["word_count_max"]:
        return False
    return True


def keep_rule_passes(strategy, matched_count, support_margin, best, high_quality_source, training_label):
    rule = ESNLI_KEEP_RULES.get(strategy, ESNLI_KEEP_RULES["default"])
    return any(
        keep_clause_passes(clause, matched_count, support_margin, best, high_quality_source, training_label)
        for clause in rule["clauses"]
    )


def choose_secondary_multiview(primary, matching_candidates, training_label, strategy):
    alternatives = []
    if strategy == "student_multiview_hardclean_balanced":
        label_prior = HARDCLEAN_MULTIVIEW_TYPE_PRIOR.get(training_label, {})
    elif strategy == "student_multiview_hybrid_balanced":
        label_prior = HYBRID_MULTIVIEW_TYPE_PRIOR.get(training_label, {})
    elif strategy == "student_multiview_sourceblend_balanced":
        label_prior = SOURCEBLEND_MULTIVIEW_TYPE_PRIOR.get(training_label, {})
    elif strategy == "student_multiview_elite":
        label_prior = ELITE_MULTIVIEW_TYPE_PRIOR.get(training_label, {})
    else:
        label_prior = STUDENT_SIGNAL_TYPE_PRIOR.get(training_label, {})
    selection_config = ESNLI_SECONDARY_MULTIVIEW.get(strategy, ESNLI_SECONDARY_MULTIVIEW["default"])
    for candidate in matching_candidates:
        if candidate is primary:
            continue
        if candidate["source"] == primary["source"]:
            continue
        if normalize_text(candidate["rationale"]).lower() == normalize_text(primary["rationale"]).lower():
            continue
        similarity = rationale_jaccard(primary["rationale"], candidate["rationale"])
        if similarity >= selection_config["similarity_limit"]:
            continue
        if candidate.get("agreement_count", 0) < selection_config["min_agreement"]:
            continue
        if candidate.get("label_support_margin", 0.0) < selection_config.get("min_label_support_margin", float("-inf")):
            continue
        if candidate["judge_score"] < primary["judge_score"] - selection_config["max_score_gap"]:
            continue
        if candidate["judge_score"] < selection_config.get("min_judge_score", float("-inf")):
            continue
        if "word_count_min" in selection_config and candidate.get("word_count", 0) < selection_config["word_count_min"]:
            continue
        if "word_count_max" in selection_config and candidate.get("word_count", 0) > selection_config["word_count_max"]:
            continue
        source_prior = label_prior.get(candidate["source"], 0.0)
        if source_prior < selection_config.get("min_source_prior", float("-inf")):
            continue

        diversity_gain = 1.0 - similarity
        alternatives.append((diversity_gain, source_prior, candidate))

    if not alternatives:
        return None

    alternatives.sort(
        key=lambda item: (
            item[2].get("agreement_count", 0),
            item[1],
            round(item[0], 6),
            item[2]["judge_score"],
            item[2].get("label_support_margin", 0.0),
        ),
        reverse=True,
    )
    return alternatives[0][2]


def source_priority_order(label, strategy):
    if strategy == "student_multiview_hardclean_balanced":
        prior = HARDCLEAN_MULTIVIEW_TYPE_PRIOR.get(label, {})
    elif strategy == "student_multiview_hybrid_balanced":
        prior = HYBRID_MULTIVIEW_TYPE_PRIOR.get(label, {})
    elif strategy == "student_multiview_sourceblend_balanced":
        prior = SOURCEBLEND_MULTIVIEW_TYPE_PRIOR.get(label, {})
    elif strategy == "student_singleview_superclean_balanced":
        prior = SINGLEVIEW_SUPERCLEAN_TYPE_PRIOR.get(label, {})
    elif strategy == "student_singleview_diverse_balanced":
        prior = SINGLEVIEW_DIVERSE_TYPE_PRIOR.get(label, {})
    elif strategy == "student_multiview_elite":
        prior = ELITE_MULTIVIEW_TYPE_PRIOR.get(label, {})
    else:
        prior = STUDENT_SIGNAL_TYPE_PRIOR.get(label, {})
    return sorted(DEFAULT_SOURCES, key=lambda source: prior.get(source, 0.0), reverse=True)


def select_diverse_balanced_subset(judged, strategy):
    balanced_parts = []
    label_counts = judged["LLM_answer"].value_counts()
    min_count = int(label_counts.min())

    for label in ["entailment", "neutral", "contradiction"]:
        subset = judged[judged["LLM_answer"] == label].copy()
        if len(subset) == 0:
            continue

        subset["_row_id"] = range(len(subset))
        subset["example_key"] = subset.apply(
            lambda row: normalize_key(row["premise"], row["hypothesis"]),
            axis=1,
        )
        subset = subset.sort_values(
            by=["judge_view_rank", "agreement_count", "judge_score", "voted_label_margin"],
            ascending=[True, False, False, False],
        )

        per_source_rows = {
            source: subset[subset["judge_source"] == source].to_dict("records")
            for source in source_priority_order(label, strategy)
        }

        selected = []
        selected_ids = set()
        example_counts = {}

        def take_from_source(source, max_per_example):
            rows = per_source_rows.get(source, [])
            while rows:
                candidate = rows[0]
                key = candidate["example_key"]
                if example_counts.get(key, 0) >= max_per_example:
                    rows.pop(0)
                    continue
                rows.pop(0)
                selected.append(candidate)
                selected_ids.add(candidate["_row_id"])
                example_counts[key] = example_counts.get(key, 0) + 1
                return True
            return False

        while len(selected) < min_count:
            progressed = False
            for source in source_priority_order(label, strategy):
                if len(selected) >= min_count:
                    break
                progressed = take_from_source(source, max_per_example=1) or progressed
            if not progressed:
                break

        if len(selected) < min_count:
            remaining = subset[~subset["_row_id"].isin(selected_ids)].to_dict("records")
            for candidate in remaining:
                if len(selected) >= min_count:
                    break
                key = candidate["example_key"]
                if example_counts.get(key, 0) >= 2:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate["_row_id"])
                example_counts[key] = example_counts.get(key, 0) + 1

        selected_df = pd.DataFrame(selected).drop(columns=["_row_id", "example_key"], errors="ignore")
        balanced_parts.append(selected_df.head(min_count))

    return pd.concat(balanced_parts, ignore_index=True)


def load_local_gold_records():
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    paths = [
        DATASET_DIR / "esnli_train.json",
        DATASET_DIR / "esnli_valid.json",
        DATASET_DIR / "esnli_test.json",
    ]
    if not all(path.exists() for path in paths):
        return []

    records = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                records.append({
                    "key": normalize_key(row["premise"], row["hypothesis"]),
                    "premise": normalize_text(row["premise"]),
                    "hypothesis": normalize_text(row["hypothesis"]),
                    "gold_label": label_map.get(row["label"], normalize_label(row["label"])),
                    "paper_rationale": "",
                })
    return records


def load_paper_gold_records():
    paper_path = API_DIR / "paper - full.csv"
    if not paper_path.exists():
        raise FileNotFoundError(f"Missing gold anchor file: {paper_path}")

    paper = pd.read_csv(paper_path)
    records = []
    for row in paper.to_dict("records"):
        records.append({
            "key": normalize_key(row["premise"], row["hypothesis"]),
            "premise": normalize_text(row["premise"]),
            "hypothesis": normalize_text(row["hypothesis"]),
            "gold_label": normalize_label(row["LLM_answer"]),
            "paper_rationale": normalize_text(row.get("rationale", "")),
        })
    return records


def load_gold_records():
    local_records = load_local_gold_records()
    if len(local_records) >= 1000:
        return local_records, "local_esnli_json"
    return load_paper_gold_records(), "paper_full_csv"


def load_candidates(source_name):
    path = API_DIR / f"{source_name} - full.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate rationale file: {path}")

    dataframe = pd.read_csv(path)
    required_columns = {"premise", "hypothesis", "rationale", "LLM_answer"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    candidates = {}
    for row in dataframe.to_dict("records"):
        key = normalize_key(row["premise"], row["hypothesis"])
        rationale = normalize_text(row["rationale"])
        if not rationale:
            continue
        candidates[key] = {
            "source": source_name,
            "premise": normalize_text(row["premise"]),
            "hypothesis": normalize_text(row["hypothesis"]),
            "label": normalize_label(row["LLM_answer"]),
            "rationale": rationale,
            "prompt": row.get("prompt", ""),
            "split": row.get("split", ""),
            "correct_index": row.get("correct_index", ""),
        }
    return candidates


def score_candidate(candidate, gold_label, strategy):
    rationale = candidate["rationale"]
    premise = candidate["premise"]
    hypothesis = candidate["hypothesis"]
    predicted_label = candidate["label"]
    source = candidate["source"]

    label_match = predicted_label == gold_label
    rationale_lower = rationale.lower()
    word_count = len(re.findall(r"\w+", rationale))

    length_schedule = (
        ESNLI_LENGTH_SCORE_CONFIG["short"]
        if strategy in {"thesis", "guarded_short"}
        else ESNLI_LENGTH_SCORE_CONFIG["default"]
    )
    length_score = piecewise_score(word_count, length_schedule)

    support_tokens = tokenize_for_overlap(premise) | tokenize_for_overlap(hypothesis)
    rationale_tokens = tokenize_for_overlap(rationale)
    overlap_score = 0.0
    if rationale_tokens:
        overlap_score = len(rationale_tokens & support_tokens) / max(1, len(rationale_tokens))

    common_config = ESNLI_COMMON_SCORING
    cue_bonus = sum(1 for cue in REASONING_CUES if cue in rationale_lower)
    cue_bonus = min(cue_bonus * common_config["cue_bonus_weight"], common_config["cue_bonus_cap"])

    label_bonus = 0.0
    if gold_label in rationale_lower:
        label_bonus += common_config["label_mention_bonus"]
    if "the correct answer" in rationale_lower or "so the answer is" in rationale_lower:
        label_bonus += common_config["answer_format_bonus"]

    teaching_cues = LABEL_TEACHING_CUES.get(gold_label, ())
    teaching_cue_bonus = min(
        common_config["teaching_cue_weight"] * sum(1 for cue in teaching_cues if cue in rationale_lower),
        common_config["teaching_cue_cap"],
    )

    ambiguity_penalty = 0.0
    other_labels = {"entailment", "neutral", "contradiction"} - {gold_label}
    other_mentions = sum(1 for other in other_labels if other in rationale_lower)
    if other_mentions >= 2:
        ambiguity_penalty += common_config["ambiguity_two_label_penalty"]
    elif other_mentions == 1:
        ambiguity_penalty += common_config["ambiguity_one_label_penalty"]
    if any(hedge in rationale_lower for hedge in ["maybe", "perhaps", "probably", "i think"]):
        ambiguity_penalty += common_config["hedge_penalty"]

    teachability_bonus = 0.0
    if common_config["teachability_ideal_min_words"] <= word_count <= common_config["teachability_ideal_max_words"]:
        teachability_bonus += common_config["teachability_ideal_bonus"]
    elif common_config["teachability_ok_min_words"] <= word_count <= common_config["teachability_ok_max_words"]:
        teachability_bonus += common_config["teachability_ok_bonus"]
    elif word_count >= common_config["teachability_long_word_min"]:
        teachability_bonus += common_config["teachability_long_penalty"]

    if strategy in {"student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview", "all_views", "all_views_consensus"}:
        type_prior = STUDENT_SIGNAL_TYPE_PRIOR.get(gold_label, {})
    elif strategy == "student_multiview_elite":
        type_prior = ELITE_MULTIVIEW_TYPE_PRIOR.get(gold_label, {})
    elif strategy == "student_multiview_hardclean_balanced":
        type_prior = HARDCLEAN_MULTIVIEW_TYPE_PRIOR.get(gold_label, {})
    elif strategy == "student_multiview_hybrid_balanced":
        type_prior = HYBRID_MULTIVIEW_TYPE_PRIOR.get(gold_label, {})
    elif strategy == "student_singleview_superclean_balanced":
        type_prior = SINGLEVIEW_SUPERCLEAN_TYPE_PRIOR.get(gold_label, {})
    elif strategy == "student_multiview_sourceblend_balanced":
        type_prior = SOURCEBLEND_MULTIVIEW_TYPE_PRIOR.get(gold_label, {})
    elif strategy == "student_singleview_diverse_balanced":
        type_prior = SINGLEVIEW_DIVERSE_TYPE_PRIOR.get(gold_label, {})
    elif strategy == "thesis":
        type_prior = THESIS_TYPE_PRIOR
    elif strategy == "agreement":
        type_prior = AGREEMENT_TYPE_PRIOR
    elif strategy in {"guarded", "guarded_short", "guarded_hardclean", "label_priority_guarded"}:
        type_prior = GUARDED_TYPE_PRIOR
    elif strategy in {"guarded_balanced", "label_priority_guarded_balanced"}:
        type_prior = BALANCED_TYPE_PRIOR
    elif strategy == "label_priority":
        type_prior = LABEL_PRIORITY_TYPE_PRIOR
    elif strategy in {"label_expert_guarded", "label_expert_guarded_balanced"}:
        type_prior = LABEL_EXPERT_TYPE_PRIOR.get(gold_label, {})
    else:
        type_prior = TYPE_PRIOR

    source_prior = type_prior.get(source, 0.5)
    strategy_config = ESNLI_SCORING_STRATEGIES.get(strategy, ESNLI_SCORING_STRATEGIES["default"])
    label_score = strategy_config["label_match_score"] if label_match else strategy_config["label_mismatch_score"]
    preferred_bonus = (
        strategy_config.get("preferred_bonus", 0.0)
        if source in preferred_sources(strategy_config.get("preferred_sources"))
        else 0.0
    )
    diversity_bonus = (
        strategy_config.get("diversity_bonus", 0.0)
        if source in set(strategy_config.get("diversity_sources", []))
        else 0.0
    )
    specialist_bonus = (
        strategy_config.get("specialist_bonus", 0.0)
        if source in set(strategy_config.get("specialist_sources", []))
        else 0.0
    )
    agreement_bonus = (
        strategy_config.get("agreement_count_weight", 0.0) * candidate.get("agreement_count", 0)
        + strategy_config.get("agreement_ratio_weight", 0.0) * candidate.get("agreement_ratio", 0.0)
    )
    margin_bonus = strategy_config.get("margin_weight", 0.0) * candidate.get("label_support_margin", 0.0)
    strict_bonus = word_condition_bonus(word_count, strategy_config.get("strict_bonus"))

    total = (
        label_score
        + source_prior
        + preferred_bonus
        + diversity_bonus
        + specialist_bonus
        + agreement_bonus
        + margin_bonus
        + strict_bonus
        + strategy_config.get("length_weight", 0.0) * length_score
        + strategy_config.get("overlap_weight", 0.0) * overlap_score
        + cue_bonus
        + label_bonus
    )
    if strategy_config.get("include_teaching"):
        total += teaching_cue_bonus
    if strategy_config.get("include_teachability"):
        total += teachability_bonus
    if strategy_config.get("include_ambiguity"):
        total += ambiguity_penalty

    return {
        "judge_score": round(total, 6),
        "label_match": label_match,
        "word_count": word_count,
        "overlap_score": round(overlap_score, 6),
    }

def infer_gold_label(candidates, strategy):
    scores = {}
    if strategy in {"student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview", "all_views", "all_views_consensus"}:
        type_prior = TYPE_PRIOR
    elif strategy == "student_multiview_elite":
        type_prior = TYPE_PRIOR
    elif strategy == "thesis":
        type_prior = THESIS_TYPE_PRIOR
    elif strategy == "agreement":
        type_prior = AGREEMENT_TYPE_PRIOR
    elif strategy in {"guarded", "guarded_short", "guarded_hardclean", "label_priority_guarded"}:
        type_prior = GUARDED_TYPE_PRIOR
    elif strategy in {"guarded_balanced", "label_priority_guarded_balanced"}:
        type_prior = BALANCED_TYPE_PRIOR
    elif strategy == "label_priority":
        type_prior = LABEL_PRIORITY_TYPE_PRIOR
    elif strategy in {"label_expert_guarded", "label_expert_guarded_balanced"}:
        type_prior = TYPE_PRIOR
    else:
        type_prior = TYPE_PRIOR
    for candidate in candidates:
        label = normalize_label(candidate.get("label", ""))
        if label not in {"entailment", "neutral", "contradiction"}:
            continue
        scores[label] = scores.get(label, 0.0) + type_prior.get(candidate["source"], 0.5)
    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def choose_best_candidate(candidates, strategy):
    if strategy == "thesis":
        preferred = [candidate for candidate in candidates if candidate["source"] in THESIS_PREFERRED_SOURCES]
        if preferred:
            return max(preferred, key=lambda candidate: (candidate["judge_score"], THESIS_TYPE_PRIOR.get(candidate["source"], 0.0)))
    if strategy == "agreement":
        preferred = [candidate for candidate in candidates if candidate["source"] in AGREEMENT_PREFERRED_SOURCES]
        if preferred:
            return max(
                preferred,
                key=lambda candidate: (
                    candidate.get("agreement_count", 0),
                    candidate.get("agreement_ratio", 0.0),
                    candidate["judge_score"],
                    AGREEMENT_TYPE_PRIOR.get(candidate["source"], 0.0),
                ),
            )
    if strategy in {"guarded", "guarded_short", "guarded_hardclean", "label_priority_guarded"}:
        preferred = [candidate for candidate in candidates if candidate["source"] in GUARDED_PREFERRED_SOURCES]
        if preferred:
            return max(
                preferred,
                key=lambda candidate: (
                    candidate.get("agreement_count", 0),
                    candidate.get("agreement_ratio", 0.0),
                    candidate["judge_score"],
                    GUARDED_TYPE_PRIOR.get(candidate["source"], 0.0),
                ),
            )
    if strategy in {"student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview", "all_views", "all_views_consensus", "student_multiview_elite", "student_multiview_hardclean_balanced", "student_multiview_hybrid_balanced", "student_singleview_superclean_balanced", "student_multiview_sourceblend_balanced", "student_singleview_diverse_balanced"}:
        if strategy == "student_singleview_diverse_balanced":
            student_prior = SINGLEVIEW_DIVERSE_TYPE_PRIOR
        elif strategy == "student_multiview_sourceblend_balanced":
            student_prior = SOURCEBLEND_MULTIVIEW_TYPE_PRIOR
        elif strategy == "student_singleview_superclean_balanced":
            student_prior = SINGLEVIEW_SUPERCLEAN_TYPE_PRIOR
        elif strategy == "student_multiview_hardclean_balanced":
            student_prior = HARDCLEAN_MULTIVIEW_TYPE_PRIOR
        elif strategy == "student_multiview_hybrid_balanced":
            student_prior = HYBRID_MULTIVIEW_TYPE_PRIOR
        elif strategy == "student_multiview_elite":
            student_prior = ELITE_MULTIVIEW_TYPE_PRIOR
        else:
            student_prior = STUDENT_SIGNAL_TYPE_PRIOR
        return max(
            candidates,
            key=lambda candidate: (
                candidate.get("agreement_count", 0),
                candidate.get("label_support_margin", 0.0),
                candidate["judge_score"],
                student_prior.get(candidate.get("voted_label", ""), {}).get(candidate["source"], 0.0),
            ),
        )
    if strategy in {"guarded_balanced", "label_priority_guarded_balanced"}:
        return max(
            candidates,
            key=lambda candidate: (
                candidate.get("agreement_count", 0),
                candidate["judge_score"],
                BALANCED_TYPE_PRIOR.get(candidate["source"], 0.0),
            ),
        )
    if strategy == "label_priority":
        return max(
            candidates,
            key=lambda candidate: (
                candidate.get("agreement_count", 0),
                candidate.get("label_support_margin", 0.0),
                candidate["judge_score"],
                LABEL_PRIORITY_TYPE_PRIOR.get(candidate["source"], 0.0),
            ),
        )
    if strategy in {"label_expert_guarded", "label_expert_guarded_balanced"}:
        return max(
            candidates,
            key=lambda candidate: (
                candidate.get("agreement_count", 0),
                candidate.get("label_support_margin", 0.0),
                candidate["judge_score"],
                LABEL_EXPERT_TYPE_PRIOR.get(candidate.get("voted_label", ""), {}).get(candidate["source"], 0.0),
            ),
        )
    if strategy in {"student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview", "all_views", "all_views_consensus"}:
        type_prior = TYPE_PRIOR
    elif strategy == "student_multiview_elite":
        type_prior = TYPE_PRIOR
    elif strategy == "student_multiview_hardclean_balanced":
        type_prior = TYPE_PRIOR
    elif strategy == "student_multiview_hybrid_balanced":
        type_prior = TYPE_PRIOR
    elif strategy == "student_singleview_superclean_balanced":
        type_prior = TYPE_PRIOR
    elif strategy == "student_multiview_sourceblend_balanced":
        type_prior = TYPE_PRIOR
    elif strategy == "student_singleview_diverse_balanced":
        type_prior = TYPE_PRIOR
    elif strategy == "thesis":
        type_prior = THESIS_TYPE_PRIOR
    elif strategy == "agreement":
        type_prior = AGREEMENT_TYPE_PRIOR
    elif strategy in {"guarded", "guarded_short", "guarded_hardclean", "label_priority_guarded"}:
        type_prior = GUARDED_TYPE_PRIOR
    elif strategy in {"guarded_balanced", "label_priority_guarded_balanced"}:
        type_prior = BALANCED_TYPE_PRIOR
    elif strategy == "label_priority":
        type_prior = LABEL_PRIORITY_TYPE_PRIOR
    elif strategy in {"label_expert_guarded", "label_expert_guarded_balanced"}:
        type_prior = TYPE_PRIOR
    else:
        type_prior = TYPE_PRIOR
    return max(candidates, key=lambda candidate: (candidate["judge_score"], type_prior.get(candidate["source"], 0.0)))


def weighted_label_vote(candidates, strategy):
    if strategy in {"student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview", "all_views", "all_views_consensus"}:
        type_prior = None
    elif strategy == "student_multiview_elite":
        type_prior = None
    elif strategy == "student_multiview_hardclean_balanced":
        type_prior = None
    elif strategy == "student_multiview_hybrid_balanced":
        type_prior = None
    elif strategy == "student_singleview_superclean_balanced":
        type_prior = None
    elif strategy == "student_multiview_sourceblend_balanced":
        type_prior = None
    elif strategy == "student_singleview_diverse_balanced":
        type_prior = None
    elif strategy == "label_priority":
        type_prior = LABEL_PRIORITY_TYPE_PRIOR
    elif strategy == "agreement":
        type_prior = AGREEMENT_TYPE_PRIOR
    elif strategy in {"guarded", "guarded_short", "guarded_hardclean", "label_priority_guarded"}:
        type_prior = GUARDED_TYPE_PRIOR
    elif strategy in {"guarded_balanced", "label_priority_guarded_balanced"}:
        type_prior = BALANCED_TYPE_PRIOR
    elif strategy == "thesis":
        type_prior = THESIS_TYPE_PRIOR
    elif strategy in {"label_expert_guarded", "label_expert_guarded_balanced"}:
        type_prior = None
    else:
        type_prior = TYPE_PRIOR

    scores = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
    for candidate in candidates:
        label = normalize_label(candidate.get("label", ""))
        if label not in scores:
            continue
        if strategy in {"student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview", "all_views", "all_views_consensus"}:
            scores[label] += STUDENT_SIGNAL_TYPE_PRIOR.get(label, {}).get(candidate["source"], 0.0)
        elif strategy == "student_multiview_elite":
            scores[label] += ELITE_MULTIVIEW_TYPE_PRIOR.get(label, {}).get(candidate["source"], 0.0)
        elif strategy == "student_multiview_hardclean_balanced":
            scores[label] += HARDCLEAN_MULTIVIEW_TYPE_PRIOR.get(label, {}).get(candidate["source"], 0.0)
        elif strategy == "student_multiview_hybrid_balanced":
            scores[label] += HYBRID_MULTIVIEW_TYPE_PRIOR.get(label, {}).get(candidate["source"], 0.0)
        elif strategy == "student_singleview_superclean_balanced":
            scores[label] += SINGLEVIEW_SUPERCLEAN_TYPE_PRIOR.get(label, {}).get(candidate["source"], 0.0)
        elif strategy == "student_multiview_sourceblend_balanced":
            scores[label] += SOURCEBLEND_MULTIVIEW_TYPE_PRIOR.get(label, {}).get(candidate["source"], 0.0)
        elif strategy == "student_singleview_diverse_balanced":
            scores[label] += SINGLEVIEW_DIVERSE_TYPE_PRIOR.get(label, {}).get(candidate["source"], 0.0)
        elif strategy in {"label_expert_guarded", "label_expert_guarded_balanced"}:
            scores[label] += LABEL_EXPERT_TYPE_PRIOR.get(label, {}).get(candidate["source"], 0.0)
        else:
            scores[label] += type_prior.get(candidate["source"], 0.0)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winning_label, winning_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return winning_label, winning_score, runner_up_score


def build_judged_dataset(source_names, output_name, strategy):
    gold_records, gold_source = load_gold_records()
    candidate_tables = {source: load_candidates(source) for source in source_names}

    rows = []
    source_counts = {}
    fallback_count = 0
    inferred_gold_count = 0
    skipped_count = 0
    low_confidence_drop_count = 0
    voted_label_override_count = 0
    extra_view_count = 0

    for gold in gold_records:
        key = gold["key"]
        candidates = []
        for source_name, table in candidate_tables.items():
            candidate = table.get(key)
            if candidate is None:
                continue
            scored = candidate.copy()
            candidates.append(scored)

        label_counts = {}
        for candidate in candidates:
            label = normalize_label(candidate.get("label", ""))
            if label not in {"entailment", "neutral", "contradiction"}:
                continue
            label_counts[label] = label_counts.get(label, 0) + 1

        total_candidate_count = max(1, len(candidates))
        for candidate in candidates:
            label = normalize_label(candidate.get("label", ""))
            agreement_count = label_counts.get(label, 0)
            candidate["agreement_count"] = agreement_count
            candidate["agreement_ratio"] = agreement_count / total_candidate_count

        paper_gold_label = gold["gold_label"]
        voted_label, winning_support, runner_up_support = weighted_label_vote(candidates, strategy)
        vote_margin = winning_support - runner_up_support

        def append_selected_row(selected, output_label, view_rank):
            source_counts[selected["source"]] = source_counts.get(selected["source"], 0) + 1
            rows.append({
                "premise": gold["premise"],
                "hypothesis": gold["hypothesis"],
                "prompt": selected.get("prompt", ""),
                "rationale": selected["rationale"],
                "split": selected.get("split", ""),
                "correct_index": selected.get("correct_index", ""),
                "LLM_answer": output_label,
                "judge_source": selected["source"],
                "judge_score": selected["judge_score"],
                "candidate_label": selected["label"],
                "gold_label": output_label,
                "paper_gold_label": paper_gold_label,
                "voted_label": voted_label,
                "voted_label_support": winning_support,
                "voted_label_margin": vote_margin,
                "label_match": selected["label_match"],
                "word_count": selected["word_count"],
                "overlap_score": selected["overlap_score"],
                "agreement_count": selected.get("agreement_count", 0),
                "agreement_ratio": selected.get("agreement_ratio", 0.0),
                "judge_view_rank": view_rank,
            })

        if strategy in {"all_views", "all_views_consensus"}:
            if not candidates:
                skipped_count += 1
                continue
            seen_pairs = set()
            kept_for_example = 0
            for candidate in candidates:
                label = candidate["label"]
                if label not in {"entailment", "neutral", "contradiction"}:
                    continue
                if strategy == "all_views_consensus":
                    if label != voted_label:
                        continue
                    if candidate.get("agreement_count", 0) < 2:
                        continue
                    output_label = voted_label
                    score_target = voted_label
                else:
                    output_label = label
                    score_target = label
                candidate["label_support_margin"] = vote_margin if label == voted_label else -vote_margin
                candidate["voted_label"] = voted_label
                candidate.update(score_candidate(candidate, score_target, strategy))
                pair_key = (normalize_text(candidate["rationale"]).lower(), output_label)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                kept_for_example += 1
                append_selected_row(candidate, output_label, kept_for_example)
            if kept_for_example == 0:
                skipped_count += 1
                continue
            if kept_for_example > 1:
                extra_view_count += kept_for_example - 1
            continue

        training_label = paper_gold_label
        if strategy in {"label_priority", "label_priority_guarded", "label_priority_guarded_balanced", "label_expert_guarded", "label_expert_guarded_balanced", "student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview", "student_multiview_elite", "student_multiview_hardclean_balanced", "student_multiview_hybrid_balanced", "student_singleview_superclean_balanced", "student_multiview_sourceblend_balanced", "student_singleview_diverse_balanced"}:
            if winning_support <= 0:
                skipped_count += 1
                continue
            if strategy == "label_priority":
                if paper_gold_label in {"entailment", "neutral", "contradiction"}:
                    if voted_label != paper_gold_label and vote_margin >= ESNLI_TRAINING_VOTE_MARGIN_THRESHOLDS["label_priority"]:
                        training_label = voted_label
                        voted_label_override_count += 1
                else:
                    training_label = voted_label
                    inferred_gold_count += 1
            elif strategy in {"student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview", "student_multiview_elite", "student_multiview_hardclean_balanced", "student_multiview_hybrid_balanced", "student_singleview_superclean_balanced", "student_multiview_sourceblend_balanced", "student_singleview_diverse_balanced"}:
                if gold_source == "local_esnli_json" and paper_gold_label in {"entailment", "neutral", "contradiction"}:
                    training_label = paper_gold_label
                else:
                    if paper_gold_label in {"entailment", "neutral", "contradiction"} and voted_label == paper_gold_label:
                        training_label = paper_gold_label
                    elif vote_margin >= ESNLI_TRAINING_VOTE_MARGIN_THRESHOLDS.get(
                        strategy,
                        ESNLI_TRAINING_VOTE_MARGIN_THRESHOLDS["default"],
                    ):
                        training_label = voted_label
                        if paper_gold_label in {"entailment", "neutral", "contradiction"} and voted_label != paper_gold_label:
                            voted_label_override_count += 1
                    else:
                        skipped_count += 1
                        continue
            else:
                if gold_source == "local_esnli_json" and paper_gold_label in {"entailment", "neutral", "contradiction"}:
                    training_label = paper_gold_label
                else:
                    if paper_gold_label in {"entailment", "neutral", "contradiction"} and voted_label == paper_gold_label:
                        training_label = paper_gold_label
                    elif vote_margin >= ESNLI_TRAINING_VOTE_MARGIN_THRESHOLDS["fallback"]:
                        training_label = voted_label
                        if paper_gold_label in {"entailment", "neutral", "contradiction"} and voted_label != paper_gold_label:
                            voted_label_override_count += 1
                    else:
                        skipped_count += 1
                        continue
        else:
            if training_label not in {"entailment", "neutral", "contradiction"}:
                training_label = infer_gold_label(candidates, strategy)
                if training_label is None:
                    skipped_count += 1
                    continue
                inferred_gold_count += 1

        for candidate in candidates:
            candidate["label_support_margin"] = vote_margin if candidate["label"] == voted_label else -vote_margin
            candidate["voted_label"] = voted_label
            candidate.update(score_candidate(candidate, training_label, strategy))

        matching_candidates = [candidate for candidate in candidates if candidate["label_match"]]

        if matching_candidates:
            best = choose_best_candidate(matching_candidates, strategy)
        else:
            if strategy in {"guarded", "guarded_short", "guarded_balanced", "guarded_hardclean", "label_priority_guarded", "label_priority_guarded_balanced", "label_expert_guarded", "label_expert_guarded_balanced", "student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview", "student_multiview_elite", "student_multiview_hardclean_balanced", "student_multiview_hybrid_balanced", "student_singleview_superclean_balanced", "student_multiview_sourceblend_balanced", "student_singleview_diverse_balanced"}:
                skipped_count += 1
                continue
            fallback_count += 1
            best = {
                "source": "paper",
                "premise": gold["premise"],
                "hypothesis": gold["hypothesis"],
                "label": training_label,
                "rationale": gold["paper_rationale"],
                "prompt": "",
                "split": "",
                "correct_index": "",
                "judge_score": (THESIS_TYPE_PRIOR if strategy == "thesis" else TYPE_PRIOR)["paper"],
                "label_match": True,
                "word_count": len(re.findall(r"\w+", gold["paper_rationale"])),
                "overlap_score": 0.0,
                "agreement_count": 0,
                "agreement_ratio": 0.0,
                "label_support_margin": 0.0,
                "voted_label": voted_label,
            }
            if strategy == "agreement":
                best["judge_score"] = AGREEMENT_TYPE_PRIOR["paper"]
            elif strategy == "guarded_balanced":
                best["judge_score"] = BALANCED_TYPE_PRIOR["paper"]
            elif strategy == "label_priority":
                best["judge_score"] = LABEL_PRIORITY_TYPE_PRIOR["paper"]

        if strategy in {"guarded", "guarded_short", "guarded_balanced", "guarded_hardclean", "label_priority_guarded", "label_priority_guarded_balanced", "label_expert_guarded", "label_expert_guarded_balanced", "student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview", "student_multiview_elite", "student_multiview_hardclean_balanced", "student_multiview_hybrid_balanced", "student_singleview_superclean_balanced", "student_multiview_sourceblend_balanced", "student_singleview_diverse_balanced"}:
            if strategy in {"guarded_balanced", "label_priority_guarded_balanced"}:
                support_prior = BALANCED_TYPE_PRIOR
            elif strategy in {"label_expert_guarded", "label_expert_guarded_balanced"}:
                support_prior = LABEL_EXPERT_TYPE_PRIOR.get(training_label, {})
            elif strategy in {"student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview"}:
                support_prior = STUDENT_SIGNAL_TYPE_PRIOR.get(training_label, {})
            elif strategy == "student_multiview_elite":
                support_prior = ELITE_MULTIVIEW_TYPE_PRIOR.get(training_label, {})
            elif strategy == "student_multiview_hardclean_balanced":
                support_prior = HARDCLEAN_MULTIVIEW_TYPE_PRIOR.get(training_label, {})
            elif strategy == "student_multiview_hybrid_balanced":
                support_prior = HYBRID_MULTIVIEW_TYPE_PRIOR.get(training_label, {})
            elif strategy == "student_singleview_superclean_balanced":
                support_prior = SINGLEVIEW_SUPERCLEAN_TYPE_PRIOR.get(training_label, {})
            elif strategy == "student_multiview_sourceblend_balanced":
                support_prior = SOURCEBLEND_MULTIVIEW_TYPE_PRIOR.get(training_label, {})
            elif strategy == "student_singleview_diverse_balanced":
                support_prior = SINGLEVIEW_DIVERSE_TYPE_PRIOR.get(training_label, {})
            else:
                support_prior = GUARDED_TYPE_PRIOR
            winning_support = sum(
                support_prior.get(candidate["source"], 0.0)
                for candidate in candidates
                if candidate["label"] == training_label
            )
            other_supports = []
            for other_label in {"entailment", "neutral", "contradiction"} - {training_label}:
                other_supports.append(sum(
                    support_prior.get(candidate["source"], 0.0)
                    for candidate in candidates
                    if candidate["label"] == other_label
                ))
            runner_up_support = max(other_supports) if other_supports else 0.0
            support_margin = winning_support - runner_up_support
            matched_count = sum(1 for candidate in candidates if candidate["label"] == training_label)
            high_quality_source = any(candidate["source"] in GUARDED_PREFERRED_SOURCES for candidate in matching_candidates)
            keep_example = keep_rule_passes(
                strategy,
                matched_count,
                support_margin,
                best,
                high_quality_source,
                training_label,
            )
            if not keep_example:
                low_confidence_drop_count += 1
                continue

        append_selected_row(best, training_label, 1)
        if strategy in {"student_multiview", "student_multiview_elite", "student_multiview_hardclean_balanced", "student_multiview_hybrid_balanced", "student_multiview_sourceblend_balanced"}:
            secondary = choose_secondary_multiview(best, matching_candidates, training_label, strategy)
            if secondary is not None:
                append_selected_row(secondary, training_label, 2)
                extra_view_count += 1

    judged = pd.DataFrame(rows)
    if strategy in {"student_multiview_hybrid_balanced", "student_singleview_superclean_balanced", "student_multiview_sourceblend_balanced", "student_singleview_diverse_balanced"} and not judged.empty:
        judged = select_diverse_balanced_subset(judged, strategy)
    elif strategy in {"label_priority_guarded_balanced", "label_expert_guarded_balanced", "student_signal_balanced", "student_multiview_elite", "student_multiview_hardclean_balanced"} and not judged.empty:
        balanced_parts = []
        label_counts = judged["LLM_answer"].value_counts()
        min_count = int(label_counts.min())
        for label in ["entailment", "neutral", "contradiction"]:
            subset = judged[judged["LLM_answer"] == label]
            if len(subset) == 0:
                continue
            subset = subset.sort_values(
                by=["agreement_count", "judge_score", "voted_label_margin"],
                ascending=[False, False, False],
            ).head(min_count)
            balanced_parts.append(subset)
        judged = pd.concat(balanced_parts, ignore_index=True)
    judged.index.name = "Unnamed: 0"

    output_csv = API_DIR / f"{output_name} - full.csv"
    judged.to_csv(output_csv)

    report = {
        "output_csv": str(output_csv),
        "num_examples": int(len(judged)),
        "gold_source": gold_source,
        "source_counts": judged["judge_source"].value_counts().to_dict() if not judged.empty else {},
        "fallback_to_paper_count": int(fallback_count),
        "inferred_gold_count": int(inferred_gold_count),
        "skipped_count": int(skipped_count),
        "low_confidence_drop_count": int(low_confidence_drop_count),
        "voted_label_override_count": int(voted_label_override_count),
        "extra_view_count": int(extra_view_count),
        "label_match_rate": float(judged["label_match"].mean()) if not judged.empty else math.nan,
        "average_judge_score": float(judged["judge_score"].mean()) if not judged.empty else math.nan,
        "sources_considered": source_names,
        "strategy": strategy,
    }
    report_path = API_DIR / f"{output_name}_judge_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    return output_csv, report_path, report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", type=str, default="judge")
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--strategy", type=str, choices=["baseline", "thesis", "agreement", "guarded", "guarded_short", "guarded_balanced", "guarded_hardclean", "label_priority", "label_priority_guarded", "label_priority_guarded_balanced", "label_expert_guarded", "label_expert_guarded_balanced", "student_signal", "student_signal_hardclean", "student_signal_balanced", "student_multiview", "student_multiview_elite", "student_multiview_hardclean_balanced", "student_multiview_hybrid_balanced", "student_singleview_superclean_balanced", "student_multiview_sourceblend_balanced", "student_singleview_diverse_balanced", "all_views", "all_views_consensus"], default="baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    output_csv, report_path, report = build_judged_dataset(args.sources, args.output_name, args.strategy)
    print(json.dumps(report, indent=2))
    print(f"Saved judged rationale CSV to {output_csv}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
