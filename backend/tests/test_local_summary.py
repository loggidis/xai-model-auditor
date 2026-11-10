from app.services.summary_service import local_summary


def _payload() -> dict:
    return {
        "model": {"name": "m", "architecture": "yolo_detect", "num_classes": 80},
        "dataset": {"name": "d", "images": 14},
        "config": {"xai_methods": ["gradcam_pp", "lime_image"], "target_layer": "model.18",
                   "mask": "full_image", "conf_threshold": 0.25,
                   "tagging": {"agreement_low": 0.2, "agreement_high": 0.4}},
        "report": {
            "status": "completed",
            "images": {"total": 14, "processed": 14, "failed": 0, "remaining": 0, "evaluated": 4},
            "auto_tags": {"needs_review": 7, "uncertain": 7},
            "reviewer_categories": {"correct_right_reason": 2, "correct_wrong_reason": 1,
                                    "wrong_prediction": 1, "unevaluated": 10},
            "avg_model_confidence": 0.5635, "avg_reviewer_trust": 52.5, "avg_agreement": 0.1,
            "detected_classes": [{"class_name": "apple", "detections": 64, "images": 14,
                                  "avg_confidence": 0.5888}],
        },
        "flagged": {"total": 6, "by_reason": {"auto:needs_review": 4, "user:wrong_prediction": 1,
                                              "user:correct_wrong_reason": 1},
                    "top": [{"image_index": 4}, {"image_index": 8}]},
    }


def test_local_summary_reports_numbers_and_concerns():
    text = local_summary(_payload())
    for section in ("## Overview", "## Key findings", "## Concerns", "## Recommended next steps"):
        assert section in text
    assert "needs_review 7 (50%)" in text
    assert "4 of 14 images evaluated" in text
    assert "apple (64 detections in 14 images" in text
    assert "below the low-agreement threshold" in text
    assert "marked as wrong predictions" in text
    assert "image_index 4, 8" in text


def test_local_summary_without_problems():
    p = _payload()
    p["report"].update({"auto_tags": {"consistent": 14}, "avg_agreement": 0.6,
                        "reviewer_categories": {"correct_right_reason": 14}})
    p["report"]["images"]["evaluated"] = 14
    p["flagged"] = {"total": 0, "by_reason": {}, "top": []}
    text = local_summary(p)
    assert "The automatic rules found no major concerns." in text
    assert "No action is required." in text