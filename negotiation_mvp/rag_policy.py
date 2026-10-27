"""Search admission policy; also embedded in standalone Langflow components."""


def search_enabled(record: dict) -> bool:
    """Allow explicit main-index entries and legacy canonical material only."""
    if "index_policy" in record:
        return record["index_policy"] == "main_index"
    return record.get("area") in ("shared", "role", "judge", "methodology")
