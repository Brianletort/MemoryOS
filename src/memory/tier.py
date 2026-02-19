"""Hot / warm / cold tier classification for indexed documents."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger("memoryos.memory")

DEFAULT_HOT_DAYS = 7
DEFAULT_WARM_DAYS = 90


def classify(created_at: datetime | str, *, hot_days: int = DEFAULT_HOT_DAYS, warm_days: int = DEFAULT_WARM_DAYS) -> str:
    """Return 'hot', 'warm', or 'cold' based on document age.

    Hot:  0 to hot_days old    -- prioritized in search results
    Warm: hot_days to warm_days -- normal weight
    Cold: older than warm_days  -- deprioritized but still searchable
    """
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    age = datetime.now() - created_at
    if age <= timedelta(days=hot_days):
        return "hot"
    if age <= timedelta(days=warm_days):
        return "warm"
    return "cold"


def reclassify_all(index: Any, *, hot_days: int = DEFAULT_HOT_DAYS, warm_days: int = DEFAULT_WARM_DAYS) -> dict[str, int]:
    """Re-classify every document in the index. Returns counts per tier.

    Call this periodically (e.g. daily) so that aging documents move from
    hot -> warm -> cold automatically.
    """
    conn = index._conn
    now = datetime.now()
    hot_cutoff = (now - timedelta(days=hot_days)).isoformat()
    warm_cutoff = (now - timedelta(days=warm_days)).isoformat()

    conn.execute("UPDATE documents SET tier = 'hot'  WHERE created_at >= ?", (hot_cutoff,))
    conn.execute("UPDATE documents SET tier = 'warm' WHERE created_at < ? AND created_at >= ?", (hot_cutoff, warm_cutoff))
    conn.execute("UPDATE documents SET tier = 'cold' WHERE created_at < ?", (warm_cutoff,))
    conn.commit()

    counts: dict[str, int] = {"hot": 0, "warm": 0, "cold": 0}
    for row in conn.execute("SELECT tier, COUNT(*) AS n FROM documents GROUP BY tier"):
        counts[row["tier"]] = row["n"]

    logger.info("Tier reclassification: %s", counts)
    return counts
