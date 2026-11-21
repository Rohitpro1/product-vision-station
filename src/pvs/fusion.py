"""Versioned, explicit identity fusion rules (PDF p.6 evidence policy).

Every decision records the rule that fired and which observations supplied
its evidence. Raw gallery cosines are marked uncalibrated.
"""
from __future__ import annotations

from dataclasses import dataclass

from .barcode import BarcodeRead, to_gtin14
from .config import Thresholds
from .gallery import GalleryHit
from .orientation import DirectionResult
from .schema import Candidate, Evidence, Identity, IdentityLevel, ScoreType

RULES_VERSION = "fusion-v1.5"


@dataclass
class FusionInput:
    gallery: GalleryHit | None
    barcodes: list[BarcodeRead]
    template: DirectionResult | None
    held: Identity | None = None  # track-held identity (already age-checked by caller)
    held_age_s: float | None = None
    frame_id: int | None = None
    observed_at: str | None = None


def _cands(g: GalleryHit | None) -> list[Candidate]:
    if not g:
        return []
    return [Candidate(**c) for c in g.candidates if not c["sku"].startswith("negative:")]


def decide(x: FusionInput, thr: Thresholds, gtin_idx: dict[str, dict], score_type: ScoreType) -> Identity:
    ev: list[Evidence] = []
    g = x.gallery
    if g and g.top_sku:
        ev.append(Evidence(kind="gallery", value=g.top_sku, score=round(g.score, 4), score_type=score_type,
                           frame_id=x.frame_id, observed_at=x.observed_at))
    t = x.template
    if t and t.sku:
        ev.append(Evidence(kind="template", value=f"{t.sku}/{t.face} inliers={t.inliers} other_sku={t.runner_up_inliers} cues={t.cue_scores}", score=round(t.score, 4),
                           score_type=ScoreType.RULE, frame_id=x.frame_id, observed_at=x.observed_at))
    for b in x.barcodes:
        ev.append(Evidence(kind="barcode", value=b.text, score=1.0 if b.valid else 0.0, score_type=ScoreType.RULE,
                           frame_id=x.frame_id, observed_at=x.observed_at, polygon_px=b.polygon_px))
    common = dict(evidence=ev, candidates=_cands(g), observed_at=x.observed_at, age_ms=0.0)
    vis_strong = bool(g and g.top_sku and not g.top_sku.startswith("negative:") and g.score >= thr.sku_score
                      and g.margin >= thr.sku_margin)
    variant_cue = bool(t and t.sku and g and t.sku == g.top_sku)

    valid = [b for b in x.barcodes if b.valid]
    hits = {to_gtin14(b.text): gtin_idx.get(to_gtin14(b.text)) for b in valid}
    if len({k for k in hits}) > 1:
        return Identity(level=IdentityLevel.CONFLICT, rule="R0_multiple_codes", reason="conflicting_code", **common)
    if valid:
        code = next(iter(hits))
        prod = hits[code]
        if prod is None:
            if vis_strong:
                return Identity(level=IdentityLevel.CONFLICT, rule="R1_unknown_code_vs_visual",
                                reason="conflicting_code", gtin=code, **common)
            return Identity(level=IdentityLevel.UNKNOWN, rule="R1b_code_not_in_catalogue", reason="not_in_catalogue",
                            gtin=code, **common)
        if vis_strong and g and g.top_sku != prod["sku"]:
            return Identity(level=IdentityLevel.CONFLICT, rule="R1_code_vs_visual", reason="conflicting_code",
                            gtin=code, family=prod.get("family") if g.top_family == prod.get("family") else None,
                            **common)
        return Identity(level=IdentityLevel.SKU, family=prod.get("family"), sku=prod["sku"], gtin=code, score=1.0,
                        score_type=ScoreType.RULE, rule="R2_verified_gtin", **common)
    if g and g.top_sku and not g.top_sku.startswith("negative:"):
        if vis_strong and variant_cue:
            return Identity(level=IdentityLevel.SKU, family=g.top_family, sku=g.top_sku, score=round(g.score, 4),
                            score_type=score_type, margin=round(g.margin, 4), rule="R3_visual_sku_with_variant_cue",
                            **common)
        cue_best = max(t.cue_scores, key=t.cue_scores.get) if (t and t.cue_scores) else None
        if (t and t.sku and cue_best == t.sku and t.cue_margin is not None and t.cue_margin >= thr.cue_margin
                and g.score >= thr.family_score and any(c["sku"] == t.sku for c in g.candidates)
                and (g.top_family is not None and t.sku in {c["sku"] for c in g.candidates if c["family"] == g.top_family})):
            return Identity(level=IdentityLevel.SKU, family=g.top_family, sku=t.sku, score=t.cue_scores[t.sku],
                            score_type=ScoreType.RULE, margin=t.cue_margin, rule="R3b_variant_cue_region", **common)
        if g.score >= thr.sku_score and g.margin < thr.tie_margin and g.competitor and not g.competitor.startswith("negative:"):
            cands = [c for c in g.candidates if c["score"] >= g.score - thr.tie_margin and not c["sku"].startswith("negative:")]
            fams = {c["family"] for c in cands}
            if len(fams) == 1:
                return Identity(level=IdentityLevel.FAMILY, family=g.top_family, score=round(g.score, 4),
                                score_type=score_type, margin=round(g.margin, 4), rule="R4_family_tie",
                                reason="tied_candidates", **common)
            return Identity(level=IdentityLevel.AMBIGUOUS, rule="R5_tied_families", reason="tied_candidates",
                            score=round(g.score, 4), score_type=score_type, margin=round(g.margin, 4), **common)
        if g.score >= thr.family_score and (g.margin >= thr.tie_margin or _same_family(g)):
            return Identity(level=IdentityLevel.FAMILY, family=g.top_family, score=round(g.score, 4),
                            score_type=score_type, margin=round(g.margin, 4), rule="R4_visual_family",
                            reason="ok" if not vis_strong else "variant_cue_missing", **common)
    if (t and t.sku and t.family and t.reason.value == "ok" and t.inliers >= 2 * thr.template_min_inliers
            and not (g and g.top_sku and g.top_sku.startswith("negative:") and g.score >= thr.sku_score)):
        return Identity(level=IdentityLevel.FAMILY, family=t.family, score=round(t.score, 4), score_type=ScoreType.RULE,
                        rule="R4b_template_family", reason="variant_cue_missing", **common)
    contradicts = bool(g and g.top_sku and g.score >= thr.family_score and (
        g.top_sku.startswith("negative:") or (x.held is not None and g.top_family not in (None, x.held.family))))
    if x.held is not None and x.held_age_s is not None and x.held_age_s <= thr.track_hold_s \
            and x.held.level in (IdentityLevel.SKU, IdentityLevel.FAMILY) and not contradicts:
        h = x.held.model_copy(deep=True)
        h.rule = "R6_track_hold"
        h.age_ms = round(x.held_age_s * 1000, 1)
        h.evidence = h.evidence + [Evidence(kind="track_history", value=h.sku or h.family,
                                            age_ms=h.age_ms, observed_at=h.observed_at)]
        return h
    reason = "below_threshold"
    if g and g.top_sku and g.top_sku.startswith("negative:"):
        reason = "not_in_catalogue"
    return Identity(level=IdentityLevel.UNKNOWN, rule="R7_unknown", reason=reason,
                    score=round(g.score, 4) if g else None, score_type=score_type if g else ScoreType.NONE,
                    margin=round(g.margin, 4) if g else None, **common)


def _same_family(g: GalleryHit) -> bool:
    return bool(g.competitor) and any(c["sku"] == g.competitor and c["family"] == g.top_family for c in g.candidates)
