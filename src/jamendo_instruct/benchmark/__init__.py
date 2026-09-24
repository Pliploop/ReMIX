"""ReMIX-B retrieval benchmark: graded-pool answer key + scoring.

Public surface (see docs/evaluation.md):
    grade_pool(pool)                 -> {query_id: [clip_id ranked by grade]}   (answer key)
    evaluate(predictions, pool)      -> Report                                  (scoring)
"""
from .graded_pool import grade_pool, evaluate, Report  # noqa: F401
