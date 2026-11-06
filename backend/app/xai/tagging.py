from pydantic import BaseModel, ConfigDict, Field, model_validator

TAGS = ("consistent", "needs_review", "uncertain", "inconclusive", "no_detection")


class TaggingConfig(BaseModel):
    """Settings for agreement and auto_tag. THE DEFAULTS ARE GUESSES: they are calibrated with real data."""
    model_config = ConfigDict(extra="forbid")

    iou_threshold: float = Field(0.5, gt=0, lt=1)          # binarization threshold of the heatmaps for the IoU
    iou_weight: float = Field(0.5, ge=0, le=1)             # weight of IoU versus Pearson in agreement_score
    agreement_high: float = Field(0.45, ge=0, le=1)
    agreement_low: float = Field(0.25, ge=0, le=1)
    conf_high: float = Field(0.6, ge=0, le=1)
    box_focus_min: float | None = Field(0.3, ge=0, le=1)   # None = disable the rule

    @model_validator(mode="after")
    def _check_order(self):
        if self.agreement_low > self.agreement_high:
            raise ValueError("agreement_low must be <= agreement_high")
        return self


def decide_tag(cfg: TaggingConfig, *, has_targets: bool, agreement: float | None,
               mean_conf: float | None, min_box_focus: float | None) -> tuple[str, str]:
    if not has_targets:
        return "no_detection", "No detections above the confidence threshold"
    if (cfg.box_focus_min is not None and min_box_focus is not None
            and min_box_focus < cfg.box_focus_min):
        return "needs_review", (f"Attention mostly outside the detected boxes "
                                f"(box_focus {min_box_focus:.2f} < {cfg.box_focus_min})")
    if agreement is None:
        return "inconclusive", "Fewer than two methods produced a heatmap, so agreement is unavailable"
    if agreement < cfg.agreement_low:
        return "needs_review", f"Methods disagree (agreement {agreement:.2f} < {cfg.agreement_low})"
    conf = mean_conf or 0.0
    if agreement >= cfg.agreement_high and conf >= cfg.conf_high:
        return "consistent", f"Methods agree ({agreement:.2f}) and confidence is high ({conf:.2f})"
    if agreement >= cfg.agreement_high:
        return "uncertain", f"Methods agree ({agreement:.2f}) but confidence is low ({conf:.2f})"
    return "uncertain", f"Agreement is in the intermediate range ({agreement:.2f})"