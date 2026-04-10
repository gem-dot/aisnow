from __future__ import annotations

import cv2

from .base import EvaluatorContext
from .casi_evaluator import CASIEvaluator
from .common import adapt_keypoints_to_dict, load_keypoint_payload
from .jsba_evaluator import JSBAEvaluator


def build_evaluator(context: EvaluatorContext):
    if context.pipeline_id == 'casi':
        return CASIEvaluator(context)
    return JSBAEvaluator(context)


def run_evaluator(context: EvaluatorContext):
    evaluator = build_evaluator(context)
    if not evaluator.streaming_required:
        return evaluator.finalize()

    payload = load_keypoint_payload(context.keypoint_cache_path)
    keypoints = payload['keypoints']
    cap = cv2.VideoCapture(str(context.input_path))
    can_read_video = cap.isOpened()

    try:
        for frame_idx in range(len(keypoints)):
            frame_image = None
            if can_read_video:
                ok, frame_image = cap.read()
                if not ok:
                    can_read_video = False
                    frame_image = None
            evaluator.process_frame(frame_image, adapt_keypoints_to_dict(keypoints[frame_idx]))
    finally:
        cap.release()

    return evaluator.finalize()
