import ast
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from config_utils import load_config


API_DIR = Path(os.environ.get("CQA_API_DIR", "[API] CQA"))
PAPER_CONFIG = load_config()
SOURCES = PAPER_CONFIG["cqa_sources"]
CQA_BOUNDARY_PROFILE = PAPER_CONFIG["cqa_boundary_scoring_profile"].copy()
CQA_BOUNDARY_PROFILE["preferred_sources"] = set(CQA_BOUNDARY_PROFILE["preferred_sources"])
BASE_SOURCE_PRIOR = CQA_BOUNDARY_PROFILE["source_prior"]
CQA_BOUNDARY_THRESHOLDS = PAPER_CONFIG["difficulty_band_thresholds"]["cqa_boundary_mix"]
CQA_BOUNDARY_SELECTION_CONFIG = PAPER_CONFIG["boundary_mix_selection"]["cqa"]
CQA_BOUNDARY_BRIDGE_CONFIG = PAPER_CONFIG["boundary_bridge_selection"]["cqa"]

CQA_SOURCE_PRIORS = PAPER_CONFIG["cqa_source_priors"]
SUPERCLEAN_SOURCE_PRIOR = CQA_SOURCE_PRIORS["SUPERCLEAN_SOURCE_PRIOR"]
DIVERSE_SOURCE_PRIOR = CQA_SOURCE_PRIORS["DIVERSE_SOURCE_PRIOR"]
EXPERT_SOURCE_PRIOR = CQA_SOURCE_PRIORS["EXPERT_SOURCE_PRIOR"]
CQA_BASE_PROFILES = PAPER_CONFIG["cqa_pack_profiles"]["base_profiles"]
CQA_MAIN_PACK_CONFIG = PAPER_CONFIG["cqa_pack_profiles"]["main_packs"]
CQA_SHORTCUT_SCORING_PROFILE = PAPER_CONFIG["cqa_shortcut_scoring_profile"]
CQA_SHORTCUT_SECONDARY_CONFIG = PAPER_CONFIG["cqa_shortcut_secondary"]
CQA_DERIVED_PACK_CONFIG = PAPER_CONFIG["cqa_derived_pack_profiles"]

FORMAT_CUES = ("so the answer is", "the correct answer")
REASONING_CUES = ("because", "therefore", "if", "then", "means")
SHORTCUT_CUES = REASONING_CUES + FORMAT_CUES
SCORE_COMPONENTS = (
    "source",
    "ground",
    "cue",
    "explicit",
    "format",
    "brief",
    "agreement_count",
    "agreement_ratio",
    "margin",
)


def normalize_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_answer(text):
    normalized = normalize_text(text).lower()
    normalized = normalized.replace("**", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" .,:;!?\"'")


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", normalize_text(text).lower()))


def parse_choices(text):
    normalized = normalize_text(text)
    if not normalized:
        return []
    try:
        parsed = ast.literal_eval(normalized)
        if isinstance(parsed, list):
            return [normalize_text(item) for item in parsed]
    except Exception:
        pass
    return [part.strip() for part in normalized.split("',") if part.strip()]


def example_key(premise, hypothesis):
    return normalize_text(premise).lower() + "</s>" + normalize_text(hypothesis).lower()


def rationale_jaccard(left, right):
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def load_gold_records():
    paper = pd.read_csv(API_DIR / "paper.csv")
    records = []
    for row in paper.to_dict("records"):
        raw_input = str(row.get("input", ""))
        if "\nAnswer Choices:\n" not in raw_input:
            continue
        question, choices_block = raw_input.split("\nAnswer Choices:\n", 1)
        choices = []
        for line in choices_block.split("\n"):
            if ") " in line:
                choices.append(normalize_text(line.split(") ", 1)[1]))
        hypothesis = str(choices)
        records.append({
            "key": example_key(question, hypothesis),
            "premise": normalize_text(question),
            "hypothesis": hypothesis,
            "gold_label": normalize_answer(row.get("label", "")),
            "paper_rationale": normalize_text(row.get("rationale", "")),
        })
    return records


def load_candidates(source_name):
    df = pd.read_csv(API_DIR / f"{source_name} - full.csv")
    table = {}
    for row in df.to_dict("records"):
        key = example_key(row["premise"], row["hypothesis"])
        table[key] = {
            "source": source_name,
            "premise": normalize_text(row["premise"]),
            "hypothesis": normalize_text(row["hypothesis"]),
            "label": normalize_answer(row["LLM_answer"]),
            "rationale": normalize_text(row["rationale"]),
            "prompt": row.get("prompt", ""),
            "split": row.get("split", ""),
            "correct_index": row.get("correct_index", ""),
        }
    return table


def weighted_vote(candidates, source_prior):
    scores = {}
    counts = Counter()
    for candidate in candidates:
        label = candidate["label"]
        if not label:
            continue
        scores[label] = scores.get(label, 0.0) + source_prior.get(candidate["source"], 0.0)
        counts[label] += 1
    if not scores:
        return "", 0.0, 0.0, counts
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, winner_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return winner, winner_score, runner_up, counts


def support_overlap(premise, hypothesis, rationale):
    support_tokens = tokenize(premise) | tokenize(hypothesis)
    rationale_tokens = tokenize(rationale)
    if not rationale_tokens:
        return 0.0
    return len(support_tokens & rationale_tokens) / len(rationale_tokens)


def shortcut_profile(premise, hypothesis, answer):
    question_tokens = tokenize(premise)
    choices = parse_choices(hypothesis)
    answer_tokens = tokenize(answer)
    answer_overlap = len(answer_tokens & question_tokens) / max(1, len(answer_tokens))
    option_overlaps = []
    for choice in choices:
        choice_tokens = tokenize(choice)
        overlap = len(choice_tokens & question_tokens) / max(1, len(choice_tokens))
        option_overlaps.append((normalize_answer(choice), overlap))
    best_other = max((score for label, score in option_overlaps if label != answer), default=0.0)
    gap = answer_overlap - best_other
    spread = sum(abs(score - answer_overlap) < 0.15 for _, score in option_overlaps)

    score = 0.0
    if best_other >= answer_overlap:
        score += 0.9
    if answer_overlap == 0.0 and best_other > 0.0:
        score += 0.8
    if gap < 0.1:
        score += 0.55
    if spread >= 3:
        score += 0.45

    if score >= 1.5:
        band = "challenge"
    elif score >= 0.8:
        band = "bridge"
    else:
        band = "easy"

    return {
        "shortcut_score": round(score, 6),
        "shortcut_band": band,
        "answer_overlap": round(answer_overlap, 6),
        "best_other_overlap": round(best_other, 6),
        "overlap_gap": round(gap, 6),
    }


def score_candidate(
    candidate,
    gold_label,
    vote_margin,
    agreement_count,
    total_candidates,
    source_prior,
    profile,
    disabled_components=None,
    return_contributions=False,
):
    disabled_components = frozenset(disabled_components or ())
    unknown_components = disabled_components.difference(SCORE_COMPONENTS)
    if unknown_components:
        raise ValueError(f"Unknown score components: {sorted(unknown_components)}")

    rationale = candidate["rationale"]
    rationale_lower = rationale.lower()
    word_count = len(re.findall(r"\w+", rationale))
    overlap = support_overlap(candidate["premise"], candidate["hypothesis"], rationale)
    explicit_answer = 1.0 if gold_label and gold_label in rationale_lower else 0.0
    cue_hits = sum(1 for cue in REASONING_CUES if cue in rationale_lower)
    format_hits = sum(1 for cue in FORMAT_CUES if cue in rationale_lower)
    brevity = profile["brevity_positive"] if profile["brief_min_words"] <= word_count <= profile["ideal_word_max"] else (profile["brevity_long_penalty"] if word_count > profile["hard_word_max"] else 0.0)
    agreement_ratio = agreement_count / max(1, total_candidates)

    contributions = {
        "source": source_prior.get(candidate["source"], 0.0) + profile["preferred_source_bonus"] * (candidate["source"] in profile["preferred_sources"]),
        "ground": profile["overlap_weight"] * overlap,
        "cue": profile["cue_weight"] * cue_hits,
        "explicit": profile["explicit_weight"] * explicit_answer,
        "format": profile["format_weight"] * format_hits,
        "brief": brevity,
        "agreement_count": profile["agreement_weight"] * agreement_count,
        "agreement_ratio": profile["agreement_ratio_weight"] * agreement_ratio,
        "margin": profile["margin_weight"] * vote_margin,
    }
    score = sum(
        contribution
        for component, contribution in contributions.items()
        if component not in disabled_components
    )
    if return_contributions:
        return score, contributions
    return score


def choose_secondary(primary, matching, profile):
    alternatives = []
    for candidate in matching:
        if candidate is primary:
            continue
        if candidate["source"] == primary["source"]:
            continue
        similarity = rationale_jaccard(primary["rationale"], candidate["rationale"])
        if similarity >= profile["similarity_limit"]:
            continue
        if candidate["judge_score"] < primary["judge_score"] - profile["max_secondary_gap"]:
            continue
        alternatives.append((1.0 - similarity, candidate))
    if not alternatives:
        return None
    alternatives.sort(key=lambda item: (item[1]["judge_score"], item[0]), reverse=True)
    return alternatives[0][1]


def append_row(rows, gold, candidate, voted_label, winner_score, vote_margin, label_counts, view_rank):
    rows.append({
        "premise": gold["premise"],
        "hypothesis": gold["hypothesis"],
        "prompt": candidate.get("prompt", ""),
        "rationale": candidate["rationale"],
        "split": candidate.get("split", ""),
        "correct_index": candidate.get("correct_index", ""),
        "LLM_answer": gold["gold_label"],
        "judge_source": candidate["source"],
        "judge_score": candidate["judge_score"],
        "candidate_label": candidate["label"],
        "gold_label": gold["gold_label"],
        "paper_gold_label": gold["gold_label"],
        "voted_label": voted_label,
        "voted_label_support": winner_score,
        "voted_label_margin": vote_margin,
        "label_match": candidate["label"] == gold["gold_label"],
        "word_count": len(candidate["rationale"].split()),
        "overlap_score": candidate["support_overlap"],
        "agreement_count": label_counts.get(gold["gold_label"], 0),
        "agreement_ratio": label_counts.get(gold["gold_label"], 0) / max(1, sum(label_counts.values())),
        "judge_view_rank": view_rank,
    })


def source_balanced_select(df, cap, max_per_example, source_order):
    if df.empty:
        return df.copy()
    df = df.copy()
    df["example_key"] = df.apply(lambda row: example_key(row["premise"], row["hypothesis"]), axis=1)
    df["row_key"] = (
        df["example_key"]
        + "||"
        + df["rationale"].fillna("").map(normalize_text).str.lower()
        + "||"
        + df["judge_source"].fillna("").astype(str).str.lower()
    )
    df = df.sort_values(
        by=["judge_score", "agreement_count", "voted_label_margin"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["row_key"], keep="first")

    target = min(cap, len(df))
    buckets = {source: df[df["judge_source"] == source].to_dict("records") for source in source_order}
    selected = []
    selected_keys = set()
    example_counts = Counter()

    while len(selected) < target:
        progressed = False
        for source in source_order:
            rows = buckets.get(source, [])
            while rows:
                row = rows.pop(0)
                if row["row_key"] in selected_keys:
                    continue
                if example_counts[row["example_key"]] >= max_per_example:
                    continue
                selected.append(row)
                selected_keys.add(row["row_key"])
                example_counts[row["example_key"]] += 1
                progressed = True
                break
            if len(selected) >= target:
                break
        if not progressed:
            break

    if len(selected) < target:
        remaining = df[~df["row_key"].isin(selected_keys)].to_dict("records")
        for row in remaining:
            if len(selected) >= target:
                break
            if example_counts[row["example_key"]] >= max_per_example:
                continue
            selected.append(row)
            selected_keys.add(row["row_key"])
            example_counts[row["example_key"]] += 1

    return pd.DataFrame(selected)


def build_family(gold_records, candidate_tables, name, profile):
    rows = []
    for gold in gold_records:
        key = gold["key"]
        candidates = []
        for source_name, table in candidate_tables.items():
            candidate = table.get(key)
            if candidate is not None and candidate["rationale"]:
                candidates.append(candidate.copy())
        if not candidates or not gold["gold_label"]:
            continue

        voted_label, winner_score, runner_up, label_counts = weighted_vote(candidates, profile["vote_prior"])
        vote_margin = winner_score - runner_up
        matching = []
        for candidate in candidates:
            if candidate["label"] != gold["gold_label"]:
                continue
            enriched = candidate.copy()
            enriched["support_overlap"] = support_overlap(candidate["premise"], candidate["hypothesis"], candidate["rationale"])
            enriched["judge_score"] = score_candidate(
                enriched,
                gold["gold_label"],
                vote_margin,
                label_counts.get(gold["gold_label"], 0),
                len(candidates),
                profile["score_prior"],
                profile,
            )
            matching.append(enriched)

        if not matching:
            continue

        matching.sort(key=lambda item: item["judge_score"], reverse=True)
        best = matching[0]
        if label_counts.get(gold["gold_label"], 0) < profile["min_agreement"]:
            continue
        if vote_margin < profile["min_margin"]:
            continue
        if best["judge_score"] < profile["min_score"]:
            continue

        append_row(rows, gold, best, voted_label, winner_score, vote_margin, label_counts, view_rank=1)

        if profile["allow_secondary"]:
            secondary = choose_secondary(best, matching, profile)
            if secondary is not None:
                append_row(rows, gold, secondary, voted_label, winner_score, vote_margin, label_counts, view_rank=2)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if profile["post_quantile"] > 0.0:
        cutoff = df["judge_score"].quantile(profile["post_quantile"])
        df = df[df["judge_score"] >= cutoff].copy()

    return source_balanced_select(
        df,
        cap=profile["cap"],
        max_per_example=profile["max_per_example"],
        source_order=profile["source_order"],
    )


def build_boundary_rows(gold_records, candidate_tables, disabled_components=None):
    rows = []
    for gold in gold_records:
        key = gold["key"]
        candidates = []
        for source_name, table in candidate_tables.items():
            candidate = table.get(key)
            if candidate is not None and candidate["rationale"]:
                candidates.append(candidate.copy())
        if not candidates or not gold["gold_label"]:
            continue

        voted_label, winner_score, runner_up, label_counts = weighted_vote(candidates, BASE_SOURCE_PRIOR)
        vote_margin = winner_score - runner_up
        matching = []
        for candidate in candidates:
            if candidate["label"] != gold["gold_label"]:
                continue
            enriched = candidate.copy()
            enriched["support_overlap"] = support_overlap(candidate["premise"], candidate["hypothesis"], candidate["rationale"])
            enriched["judge_score"] = score_candidate(
                enriched,
                gold["gold_label"],
                vote_margin,
                label_counts.get(gold["gold_label"], 0),
                len(candidates),
                BASE_SOURCE_PRIOR,
                CQA_BOUNDARY_PROFILE,
                disabled_components=disabled_components,
            )
            matching.append(enriched)
        if not matching:
            continue
        matching.sort(key=lambda item: item["judge_score"], reverse=True)
        best = matching[0]

        if vote_margin >= CQA_BOUNDARY_THRESHOLDS["easy"]["tau_easy_m"] and label_counts.get(gold["gold_label"], 0) >= CQA_BOUNDARY_THRESHOLDS["easy"]["tau_easy_a"]:
            boundary_band = "easy"
        elif vote_margin >= CQA_BOUNDARY_THRESHOLDS["boundary"]["tau_boundary_m"] and label_counts.get(gold["gold_label"], 0) >= CQA_BOUNDARY_THRESHOLDS["boundary"]["tau_boundary_a"]:
            boundary_band = "boundary"
        else:
            boundary_band = "hard"

        append_row(rows, gold, best, voted_label, winner_score, vote_margin, label_counts, view_rank=1)
        rows[-1]["boundary_band"] = boundary_band

        partner = choose_secondary(
            best,
            matching,
            CQA_BOUNDARY_BRIDGE_CONFIG,
        )
        if partner is not None and boundary_band != "hard":
            append_row(rows, gold, partner, voted_label, winner_score, vote_margin, label_counts, view_rank=2)
            rows[-1]["boundary_band"] = "bridge"

    return pd.DataFrame(rows)


def build_shortcut_rows(gold_records, candidate_tables):
    rows = []
    for gold in gold_records:
        key = gold["key"]
        candidates = []
        for source_name, table in candidate_tables.items():
            candidate = table.get(key)
            if candidate is not None and candidate["rationale"]:
                candidates.append(candidate.copy())
        if not candidates or not gold["gold_label"]:
            continue

        profile = shortcut_profile(gold["premise"], gold["hypothesis"], gold["gold_label"])
        voted_label, winner_score, runner_up, label_counts = weighted_vote(candidates, DIVERSE_SOURCE_PRIOR)
        vote_margin = winner_score - runner_up

        matching = []
        for candidate in candidates:
            if candidate["label"] != gold["gold_label"]:
                continue
            enriched = candidate.copy()
            enriched["support_overlap"] = support_overlap(candidate["premise"], candidate["hypothesis"], candidate["rationale"])
            score_profile = CQA_SHORTCUT_SCORING_PROFILE["score_profile"].copy()
            score_profile["preferred_sources"] = set(score_profile["preferred_sources"])
            score_prior_name = (
                CQA_SHORTCUT_SCORING_PROFILE["challenge_score_prior"]
                if profile["shortcut_band"] == "challenge"
                else CQA_SHORTCUT_SCORING_PROFILE["non_challenge_score_prior"]
            )
            base_score = score_candidate(
                enriched,
                gold["gold_label"],
                vote_margin,
                label_counts.get(gold["gold_label"], 0),
                len(candidates),
                resolve_prior(score_prior_name),
                score_profile,
            )
            enriched["judge_score"] = base_score + CQA_SHORTCUT_SCORING_PROFILE["shortcut_bonus_weight"] * profile["shortcut_score"]
            matching.append(enriched)
        if not matching:
            continue

        matching.sort(key=lambda item: item["judge_score"], reverse=True)
        best = matching[0]
        append_row(rows, gold, best, voted_label, winner_score, vote_margin, label_counts, view_rank=1)
        rows[-1].update(profile)

        partner = choose_secondary(
            best,
            matching,
            {
                "similarity_limit": (
                    CQA_SHORTCUT_SECONDARY_CONFIG["challenge_similarity_limit"]
                    if profile["shortcut_band"] == "challenge"
                    else CQA_SHORTCUT_SECONDARY_CONFIG["default_similarity_limit"]
                ),
                "max_secondary_gap": CQA_SHORTCUT_SECONDARY_CONFIG["max_secondary_gap"],
            },
        )
        if partner is not None and (profile["shortcut_band"] != "easy" or label_counts.get(gold["gold_label"], 0) >= 5):
            append_row(rows, gold, partner, voted_label, winner_score, vote_margin, label_counts, view_rank=2)
            rows[-1].update(profile)

    return pd.DataFrame(rows)


def band_select(df, cap, max_per_example, band_column, allowed_bands=None, band_mix=None, source_order=None):
    if df.empty:
        return df.copy()
    work = df.copy()
    if allowed_bands is not None:
        work = work[work[band_column].isin(set(allowed_bands))].copy()
    if work.empty:
        return work

    work["example_key"] = work.apply(lambda row: example_key(row["premise"], row["hypothesis"]), axis=1)
    work["row_key"] = (
        work["example_key"]
        + "||"
        + work["rationale"].fillna("").map(normalize_text).str.lower()
        + "||"
        + work["judge_source"].fillna("").astype(str).str.lower()
    )
    work = work.sort_values(
        by=["judge_score", "agreement_count", "voted_label_margin"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["row_key"], keep="first")

    target = min(cap, len(work))
    chosen = []
    chosen_keys = set()
    example_counts = Counter()

    if band_mix:
        for band, ratio in band_mix:
            subset = work[work[band_column] == band]
            desired = min(len(subset), int(math.floor(target * ratio)))
            for _, row in subset.iterrows():
                if row["row_key"] in chosen_keys:
                    continue
                if example_counts[row["example_key"]] >= max_per_example:
                    continue
                chosen.append(row)
                chosen_keys.add(row["row_key"])
                example_counts[row["example_key"]] += 1
                if len([item for item in chosen if item[band_column] == band]) >= desired:
                    break

    if len(chosen) < target:
        if source_order:
            remainder = source_balanced_select(
                work[~work["row_key"].isin(chosen_keys)],
                cap=target - len(chosen),
                max_per_example=max_per_example,
                source_order=source_order,
            )
            for _, row in remainder.iterrows():
                if example_counts[row["example_key"]] >= max_per_example:
                    continue
                chosen.append(row)
                example_counts[row["example_key"]] += 1
        else:
            remaining = work[~work["row_key"].isin(chosen_keys)]
            for _, row in remaining.iterrows():
                if len(chosen) >= target:
                    break
                if example_counts[row["example_key"]] >= max_per_example:
                    continue
                chosen.append(row)
                example_counts[row["example_key"]] += 1

    return pd.DataFrame(chosen[:target])


def add_bonus(df, bonus):
    boosted = df.copy()
    boosted["quality_score"] = boosted["judge_score"] + bonus
    return boosted


def dedupe_by_quality(df):
    if df.empty:
        return df.copy()
    work = df.copy()
    if "quality_score" not in work.columns:
        work["quality_score"] = work["judge_score"]
    work["example_key"] = work.apply(lambda row: example_key(row["premise"], row["hypothesis"]), axis=1)
    work["row_key"] = (
        work["example_key"]
        + "||"
        + work["rationale"].fillna("").map(normalize_text).str.lower()
        + "||"
        + work["judge_source"].fillna("").astype(str).str.lower()
    )
    return work.sort_values(
        by=["quality_score", "judge_score", "agreement_count", "voted_label_margin"],
        ascending=[False, False, False, False],
    ).drop_duplicates(subset=["row_key"], keep="first")


def derived_pack_rows(base_packs, input_specs):
    parts = []
    for spec in input_specs:
        rows = base_packs[spec["pack"]]
        if "view_rank" in spec:
            rows = rows[rows["judge_view_rank"] == spec["view_rank"]]
        if "filter_sources" in spec:
            rows = rows[rows["judge_source"].isin(set(spec["filter_sources"]))]
        parts.append(add_bonus(rows, spec["bonus"]))
    return pd.concat(parts, ignore_index=True)


def build_derived_packs(base_packs):
    packs = {}

    core_config = CQA_DERIVED_PACK_CONFIG["judge_hybrid_core_balanced"]
    hybrid_core = base_packs[core_config["source_pack"]].copy()
    hybrid_core["score_cutoff"] = hybrid_core["judge_score"].quantile(core_config["score_quantile"])
    hybrid_core["margin_cutoff"] = hybrid_core["voted_label_margin"].quantile(core_config["margin_quantile"])
    hybrid_core["agree_cutoff"] = hybrid_core["agreement_count"].median()
    hybrid_core = hybrid_core[
        (hybrid_core["judge_score"] >= hybrid_core["score_cutoff"])
        & (hybrid_core["voted_label_margin"] >= hybrid_core["margin_cutoff"])
        & (hybrid_core["agreement_count"] >= hybrid_core["agree_cutoff"])
    ].drop(columns=["score_cutoff", "margin_cutoff", "agree_cutoff"])
    packs["judge_hybrid_core_balanced"] = source_balanced_select(
        hybrid_core,
        core_config["cap"],
        core_config["max_per_example"],
        list(resolve_prior(core_config["source_order"])),
    )

    expert_config = CQA_DERIVED_PACK_CONFIG["judge_expert_hybrid_fusion_balanced"]
    packs["judge_expert_hybrid_fusion_balanced"] = source_balanced_select(
        dedupe_by_quality(derived_pack_rows(base_packs, expert_config["inputs"])),
        expert_config["cap"],
        expert_config["max_per_example"],
        list(resolve_prior(expert_config["source_order"])),
    )

    diverse_config = CQA_DERIVED_PACK_CONFIG["judge_precision_diverse_mix_balanced"]
    packs["judge_precision_diverse_mix_balanced"] = source_balanced_select(
        dedupe_by_quality(derived_pack_rows(base_packs, diverse_config["inputs"])),
        diverse_config["cap"],
        diverse_config["max_per_example"],
        list(resolve_prior(diverse_config["source_order"])),
    )

    cleanfusion_config = CQA_DERIVED_PACK_CONFIG["judge_student_multiview_hybrid_cleanfusion"]
    packs["judge_student_multiview_hybrid_cleanfusion"] = source_balanced_select(
        dedupe_by_quality(derived_pack_rows(base_packs, cleanfusion_config["inputs"])),
        cleanfusion_config["cap"],
        cleanfusion_config["max_per_example"],
        list(resolve_prior(cleanfusion_config["source_order"])),
    )

    return packs


def save_dataset(name, df):
    output_path = API_DIR / f"{name} - full.csv"
    clean_df = df.copy()
    clean_df.index.name = "Unnamed: 0"
    clean_df.to_csv(output_path)

    report = {
        "output_csv": str(output_path),
        "num_examples": int(len(clean_df)),
        "judge_source_counts": clean_df["judge_source"].value_counts().to_dict() if "judge_source" in clean_df else {},
        "view_rank_counts": clean_df["judge_view_rank"].value_counts().to_dict() if "judge_view_rank" in clean_df else {},
        "average_judge_score": round(float(clean_df["judge_score"].mean()), 6) if "judge_score" in clean_df and not clean_df.empty else None,
        "average_agreement_count": round(float(clean_df["agreement_count"].mean()), 6) if "agreement_count" in clean_df and not clean_df.empty else None,
        "average_voted_label_margin": round(float(clean_df["voted_label_margin"].mean()), 6) if "voted_label_margin" in clean_df and not clean_df.empty else None,
    }
    if "boundary_band" in clean_df:
        report["boundary_band_counts"] = clean_df["boundary_band"].value_counts().to_dict()
    if "shortcut_band" in clean_df:
        report["shortcut_band_counts"] = clean_df["shortcut_band"].value_counts().to_dict()
        report["average_shortcut_score"] = round(float(clean_df["shortcut_score"].mean()), 6)

    report_path = API_DIR / f"{name}_judge_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)
    return report



def resolve_prior(name):
    if isinstance(name, dict):
        return name
    return CQA_SOURCE_PRIORS[name]


def resolve_profile(profile):
    resolved = profile.copy()
    resolved["vote_prior"] = resolve_prior(resolved["vote_prior"])
    resolved["score_prior"] = resolve_prior(resolved["score_prior"])
    resolved["preferred_sources"] = set(resolved["preferred_sources"])
    resolved["source_order"] = list(resolve_prior(resolved["source_order"]))
    return resolved


def resolve_pack_config(pack_config):
    resolved = pack_config.copy()
    if "allowed_bands" in resolved:
        resolved["allowed_bands"] = set(resolved["allowed_bands"])
    if "filter_sources" in resolved:
        resolved["filter_sources"] = set(resolved["filter_sources"])
    if "band_mix" in resolved:
        resolved["band_mix"] = [tuple(item) for item in resolved["band_mix"]]
    if "source_order" in resolved:
        resolved["source_order"] = list(resolve_prior(resolved["source_order"]))
    return resolved

def main():
    gold_records = load_gold_records()
    candidate_tables = {source: load_candidates(source) for source in SOURCES}

    base_profiles = {name: resolve_profile(profile) for name, profile in CQA_BASE_PROFILES.items()}

    packs = {}
    for name, profile in base_profiles.items():
        packs[name] = build_family(gold_records, candidate_tables, name, profile)

    boundary_rows = build_boundary_rows(gold_records, candidate_tables)
    main_pack_config = {name: resolve_pack_config(config) for name, config in CQA_MAIN_PACK_CONFIG.items()}
    packs["judge_student_boundary_mix_balanced"] = band_select(
        boundary_rows,
        cap=main_pack_config["judge_student_boundary_mix_balanced"]["cap"],
        max_per_example=main_pack_config["judge_student_boundary_mix_balanced"]["max_per_example"],
        band_column=main_pack_config["judge_student_boundary_mix_balanced"]["band_column"],
        allowed_bands=main_pack_config["judge_student_boundary_mix_balanced"]["allowed_bands"],
        source_order=main_pack_config["judge_student_boundary_mix_balanced"]["source_order"],
    )
    packs["judge_student_boundary_bridge_balanced"] = band_select(
        boundary_rows,
        cap=main_pack_config["judge_student_boundary_bridge_balanced"]["cap"],
        max_per_example=main_pack_config["judge_student_boundary_bridge_balanced"]["max_per_example"],
        band_column=main_pack_config["judge_student_boundary_bridge_balanced"]["band_column"],
        allowed_bands=main_pack_config["judge_student_boundary_bridge_balanced"]["allowed_bands"],
        source_order=main_pack_config["judge_student_boundary_bridge_balanced"]["source_order"],
    )
    specialist_config = main_pack_config["judge_student_boundary_specialist_balanced"]
    packs["judge_student_boundary_specialist_balanced"] = band_select(
        boundary_rows[boundary_rows["judge_source"].isin(specialist_config["filter_sources"])],
        cap=specialist_config["cap"],
        max_per_example=specialist_config["max_per_example"],
        band_column=specialist_config["band_column"],
        allowed_bands=specialist_config["allowed_bands"],
        source_order=specialist_config["source_order"],
    )

    shortcut_rows = build_shortcut_rows(gold_records, candidate_tables)
    shortcut_config = main_pack_config["judge_student_shortcut_aware_balanced"]
    packs["judge_student_shortcut_aware_balanced"] = band_select(
        shortcut_rows,
        cap=shortcut_config["cap"],
        max_per_example=shortcut_config["max_per_example"],
        band_column=shortcut_config["band_column"],
        allowed_bands=shortcut_config["allowed_bands"],
        band_mix=shortcut_config["band_mix"],
        source_order=shortcut_config["source_order"],
    )

    packs.update(build_derived_packs(packs))

    for name, df in packs.items():
        report = save_dataset(name, df)
        print(f"{name}: {report['num_examples']} rows")


if __name__ == "__main__":
    main()
