import json
from pathlib import Path

import pandas as pd

from config_utils import load_config
from build_judge_esnli import (
    API_DIR,
    DEFAULT_SOURCES,
    LABEL_TEACHING_CUES,
    load_candidates,
    load_gold_records,
    normalize_label,
    normalize_text,
    rationale_jaccard,
    tokenize_for_overlap,
)


PAPER_CONFIG = load_config()
BOUNDARY_SOURCE_PRIOR = PAPER_CONFIG["source_prior_pi_D"]["values"]
SCORING_COEFFICIENTS = PAPER_CONFIG["rationale_quality_score"]["table_2_coefficients"]
SCORING_COMPONENT_MAP = PAPER_CONFIG["rationale_quality_score"]["code_component_mapping"]
SCORING_SIGNAL_VALUES = PAPER_CONFIG["rationale_quality_score"]["signal_values"]
ESNLI_BOUNDARY_THRESHOLDS = PAPER_CONFIG["difficulty_band_thresholds"]["esnli_boundary_mix"]
ESNLI_BOUNDARY_MIX_CONFIG = PAPER_CONFIG["boundary_mix_selection"]["esnli"]
ESNLI_BOUNDARY_BRIDGE_DATASET_CONFIG = PAPER_CONFIG["boundary_mix_selection"]["esnli_bridge"]
ESNLI_BOUNDARY_SPECIALIST_CONFIG = PAPER_CONFIG["boundary_mix_selection"]["esnli_specialist"]
ESNLI_BOUNDARY_BRIDGE_CONFIG = PAPER_CONFIG["boundary_bridge_selection"]["esnli"]

LABELS = ["entailment", "neutral", "contradiction"]

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


def build_example_key(premise, hypothesis):
    return normalize_text(premise).lower() + "</s>" + normalize_text(hypothesis).lower()


def vote_labels(candidates):
    scores = {label: 0.0 for label in LABELS}
    counts = {label: 0 for label in LABELS}
    for candidate in candidates:
        label = normalize_label(candidate["label"])
        if label not in scores:
            continue
        scores[label] += BOUNDARY_SOURCE_PRIOR[label].get(candidate["source"], 0.0)
        counts[label] += 1
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, winner_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return winner, winner_score, runner_up_score, counts


def candidate_quality(
    candidate,
    training_label,
    vote_margin,
    agreement_count,
    total_candidates,
    disabled_components=None,
    return_contributions=False,
):
    disabled_components = frozenset(disabled_components or ())
    unknown_components = disabled_components.difference(SCORE_COMPONENTS)
    if unknown_components:
        raise ValueError(f"Unknown score components: {sorted(unknown_components)}")

    rationale = candidate["rationale"]
    rationale_lower = rationale.lower()
    word_count = len(rationale.split())
    overlap_tokens = tokenize_for_overlap(candidate["premise"]) | tokenize_for_overlap(candidate["hypothesis"])
    rationale_tokens = tokenize_for_overlap(rationale)
    overlap_score = len(overlap_tokens & rationale_tokens) / max(1, len(rationale_tokens))
    source_prior = BOUNDARY_SOURCE_PRIOR.get(training_label, {}).get(candidate["source"], 0.0)
    teaching_hits = sum(1 for cue in LABEL_TEACHING_CUES.get(training_label, ()) if cue in rationale_lower)
    explicit_label = 1.0 if training_label in rationale_lower else 0.0
    format_bonus = SCORING_SIGNAL_VALUES["format_positive"] if "the correct answer" in rationale_lower or "so the answer is" in rationale_lower else 0.0
    brevity = (
        SCORING_SIGNAL_VALUES["brief_positive"]
        if SCORING_SIGNAL_VALUES["brief_min_words"] <= word_count <= SCORING_SIGNAL_VALUES["brief_ideal_word_max"]
        else (SCORING_SIGNAL_VALUES["brief_long_penalty"] if word_count > SCORING_SIGNAL_VALUES["brief_hard_word_max"] else 0.0)
    )
    agreement_ratio = agreement_count / max(1, total_candidates)
    contributions = {
        "source": SCORING_COEFFICIENTS["alpha"] * source_prior,
        "ground": SCORING_COEFFICIENTS["beta"] * overlap_score,
        "cue": SCORING_COEFFICIENTS["gamma"] * teaching_hits,
        "explicit": SCORING_COEFFICIENTS["delta"] * explicit_label,
        "format": SCORING_COEFFICIENTS["eta"] * format_bonus,
        "brief": SCORING_COEFFICIENTS["lambda"] * brevity,
        "agreement_count": SCORING_COEFFICIENTS["mu"] * agreement_count,
        "agreement_ratio": SCORING_COEFFICIENTS["nu"] * agreement_ratio,
        "margin": SCORING_COEFFICIENTS["rho"] * vote_margin,
    }
    score = sum(
        contribution
        for component, contribution in contributions.items()
        if component not in disabled_components
    )
    if return_contributions:
        return score, contributions
    return score


def choose_boundary_partner(primary, matching_candidates):
    alternatives = []
    for candidate in matching_candidates:
        if candidate is primary:
            continue
        if candidate["source"] == primary["source"]:
            continue
        similarity = rationale_jaccard(primary["rationale"], candidate["rationale"])
        if similarity >= ESNLI_BOUNDARY_BRIDGE_CONFIG["similarity_limit"]:
            continue
        alternatives.append((1.0 - similarity, candidate))
    if not alternatives:
        return None
    alternatives.sort(key=lambda item: (item[1]["quality_score"], item[0]), reverse=True)
    return alternatives[0][1]


def append_row(rows, gold, candidate, output_label, vote_margin, label_counts, winner_score, boundary_band):
    rows.append({
        "premise": gold["premise"],
        "hypothesis": gold["hypothesis"],
        "prompt": candidate.get("prompt", ""),
        "rationale": candidate["rationale"],
        "split": candidate.get("split", ""),
        "correct_index": candidate.get("correct_index", ""),
        "LLM_answer": output_label,
        "judge_source": candidate["source"],
        "judge_score": candidate["quality_score"],
        "candidate_label": candidate["label"],
        "gold_label": output_label,
        "paper_gold_label": gold["gold_label"],
        "voted_label": output_label,
        "voted_label_support": winner_score,
        "voted_label_margin": vote_margin,
        "label_match": True,
        "word_count": len(candidate["rationale"].split()),
        "overlap_score": candidate["overlap_score"],
        "agreement_count": label_counts.get(output_label, 0),
        "agreement_ratio": label_counts.get(output_label, 0) / max(1, sum(label_counts.values())),
        "judge_view_rank": candidate.get("judge_view_rank", 1),
        "boundary_band": boundary_band,
    })


def collect_examples(gold_records=None, candidate_tables=None, disabled_components=None):
    if gold_records is None:
        gold_records, _ = load_gold_records()
    if candidate_tables is None:
        candidate_tables = {source: load_candidates(source) for source in DEFAULT_SOURCES}
    rows = []

    for gold in gold_records:
        key = gold["key"]
        candidates = []
        for source_name, table in candidate_tables.items():
            candidate = table.get(key)
            if candidate is None:
                continue
            scored = candidate.copy()
            candidates.append(scored)

        if not candidates:
            continue

        voted_label, winner_score, runner_up_score, label_counts = vote_labels(candidates)
        vote_margin = winner_score - runner_up_score
        paper_label = normalize_label(gold["gold_label"])

        if paper_label in LABELS and (paper_label == voted_label or vote_margin < ESNLI_BOUNDARY_THRESHOLDS["label_override_vote_margin"]):
            training_label = paper_label
        elif voted_label in LABELS:
            training_label = voted_label
        else:
            continue

        matching = []
        for candidate in candidates:
            label = normalize_label(candidate["label"])
            if label != training_label:
                continue
            agreement_count = label_counts.get(training_label, 0)
            total_candidates = max(1, len(candidates))
            quality = candidate_quality(
                candidate,
                training_label,
                vote_margin,
                agreement_count,
                total_candidates,
                disabled_components=disabled_components,
            )
            rationale_tokens = tokenize_for_overlap(candidate["rationale"])
            support_tokens = tokenize_for_overlap(candidate["premise"]) | tokenize_for_overlap(candidate["hypothesis"])
            overlap_score = len(rationale_tokens & support_tokens) / max(1, len(rationale_tokens))
            enriched = candidate.copy()
            enriched["quality_score"] = quality
            enriched["overlap_score"] = overlap_score
            matching.append(enriched)

        if not matching:
            continue

        matching.sort(key=lambda cand: cand["quality_score"], reverse=True)
        best = matching[0]

        if vote_margin >= ESNLI_BOUNDARY_THRESHOLDS["easy"]["tau_easy_m"] and label_counts.get(training_label, 0) >= ESNLI_BOUNDARY_THRESHOLDS["easy"]["tau_easy_a"]:
            boundary_band = "easy"
        elif vote_margin >= ESNLI_BOUNDARY_THRESHOLDS["boundary"]["tau_boundary_m"] and label_counts.get(training_label, 0) >= ESNLI_BOUNDARY_THRESHOLDS["boundary"]["tau_boundary_a"]:
            boundary_band = "boundary"
        else:
            boundary_band = "hard"

        append_row(rows, gold, best, training_label, vote_margin, label_counts, winner_score, boundary_band)

        partner = choose_boundary_partner(best, matching)
        if partner is not None and boundary_band != "hard":
            partner = partner.copy()
            partner["judge_view_rank"] = 2
            append_row(rows, gold, partner, training_label, vote_margin, label_counts, winner_score, "bridge")

    return pd.DataFrame(rows)


def balance_dataset(df, cap_per_label, allowed_bands, max_per_example):
    if df.empty:
        return df.copy()

    df = df[df["boundary_band"].isin(allowed_bands)].copy()
    df["example_key"] = df.apply(lambda row: build_example_key(row["premise"], row["hypothesis"]), axis=1)
    df["row_key"] = (
        df["example_key"]
        + "||"
        + df["rationale"].fillna("").map(normalize_text).str.lower()
        + "||"
        + df["LLM_answer"].fillna("").astype(str).str.lower()
    )
    df = df.sort_values(
        by=["judge_score", "agreement_count", "voted_label_margin"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["row_key"], keep="first")

    counts = df["LLM_answer"].value_counts()
    target = min(cap_per_label, int(counts.min()))
    parts = []

    for label in LABELS:
        subset = df[df["LLM_answer"] == label].sort_values(
            by=["judge_score", "agreement_count", "voted_label_margin"],
            ascending=[False, False, False],
        )
        chosen = []
        example_counts = {}
        for _, row in subset.iterrows():
            key = row["example_key"]
            if example_counts.get(key, 0) >= max_per_example:
                continue
            chosen.append(row)
            example_counts[key] = example_counts.get(key, 0) + 1
            if len(chosen) >= target:
                break
        parts.append(pd.DataFrame(chosen))

    return pd.concat(parts, ignore_index=True)


def save_dataset(name, df):
    output_csv = API_DIR / f"{name} - full.csv"
    df = df.copy()
    df.index.name = "Unnamed: 0"
    df.to_csv(output_csv)

    report = {
        "output_csv": str(output_csv),
        "num_examples": int(len(df)),
        "label_counts": df["LLM_answer"].value_counts().to_dict(),
        "judge_source_counts": df["judge_source"].value_counts().to_dict(),
        "boundary_band_counts": df["boundary_band"].value_counts().to_dict(),
        "average_judge_score": float(df["judge_score"].mean()),
        "average_agreement_count": float(df["agreement_count"].mean()),
        "average_voted_label_margin": float(df["voted_label_margin"].mean()),
        "extra_view_count": int((df["judge_view_rank"] > 1).sum()),
    }
    report_path = API_DIR / f"{name}_judge_report.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    return output_csv, report_path, report


def main():
    rows = collect_examples()

    datasets = {
        "judge_student_boundary_mix_balanced": balance_dataset(
            rows,
            cap_per_label=ESNLI_BOUNDARY_MIX_CONFIG["cap_per_label"],
            allowed_bands=set(ESNLI_BOUNDARY_MIX_CONFIG["allowed_bands"]),
            max_per_example=ESNLI_BOUNDARY_MIX_CONFIG["max_per_example"],
        ),
        "judge_student_boundary_bridge_balanced": balance_dataset(
            rows,
            cap_per_label=ESNLI_BOUNDARY_BRIDGE_DATASET_CONFIG["cap_per_label"],
            allowed_bands=set(ESNLI_BOUNDARY_BRIDGE_DATASET_CONFIG["allowed_bands"]),
            max_per_example=ESNLI_BOUNDARY_BRIDGE_DATASET_CONFIG["max_per_example"],
        ),
        "judge_student_boundary_specialist_balanced": balance_dataset(
            rows[rows["judge_source"].isin(set(ESNLI_BOUNDARY_SPECIALIST_CONFIG["sources"]))],
            cap_per_label=ESNLI_BOUNDARY_SPECIALIST_CONFIG["cap_per_label"],
            allowed_bands=set(ESNLI_BOUNDARY_SPECIALIST_CONFIG["allowed_bands"]),
            max_per_example=ESNLI_BOUNDARY_SPECIALIST_CONFIG["max_per_example"],
        ),
    }

    for name, df in datasets.items():
        output_csv, report_path, report = save_dataset(name, df)
        print(json.dumps(report, indent=2))
        print(f"Saved judged rationale CSV to {output_csv}")
        print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
