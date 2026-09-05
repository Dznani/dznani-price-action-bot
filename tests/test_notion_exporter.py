"""
tests/test_notion_exporter.py

Covers the Notion integration's safety contract end to end using mocks
that match the real notion-client SDK's shape (Client(auth=...),
client.pages.create(parent=..., properties=...), APIResponseError with a
.status int) — verified against the actual installed package's
constructor signatures before writing these, not guessed.
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import notion_exporter as ne  # noqa: E402


class _FakeAPIError(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__(f"fake notion error, status={status}")


def _sample_signal(**overrides):
    signal = {
        "symbol": "BTC/USDT", "signal_type": "WAIT FOR RETEST", "decision": "WAIT", "direction": "BUY",
        "entry": 63250.0, "current_price": 63845.10, "setup_grade": "A+",
        "rr": {"rr": 4.2}, "extension": {"chase_score": 32.0, "preferred_entry_low": 63100.0, "preferred_entry_high": 63400.0, "label": "EXTENDED"},
        "location": {"premium_discount": {"zone": "discount"}},
        "timestamp": "2026-08-22T10:30:00+00:00",
    }
    signal.update(overrides)
    return signal


# --------------------------------------------------------------------------- #
# signal_to_notion_properties() — pure mapping logic, no client needed
# --------------------------------------------------------------------------- #
def test_valid_signal_maps_all_fields():
    props = ne.signal_to_notion_properties(_sample_signal(signal_type="VALID", decision="VALID"))
    assert props["Symbol"]["title"][0]["text"]["content"] == "BTC/USDT"
    assert props["Signal Type"]["select"]["name"] == "VALID"
    assert props["Direction"]["select"]["name"] == "BUY"
    assert props["Entry Price"]["number"] == 63250.0
    assert props["Setup Grade"]["select"]["name"] == "A+"
    assert props["R:R"]["number"] == 4.2
    assert props["Chase Score"]["number"] == 32.0
    assert "63,100" in props["Preferred Zone"]["rich_text"][0]["text"]["content"]
    assert props["Premium/Discount"]["select"]["name"] == "Discount"
    assert props["Timestamp"]["date"]["start"] == "2026-08-22T10:30:00+00:00"


def test_no_chase_signal_maps_correctly():
    props = ne.signal_to_notion_properties(_sample_signal(signal_type="NO CHASE", decision="WAIT",
                                                            location={"premium_discount": {"zone": "premium"}}))
    assert props["Signal Type"]["select"]["name"] == "NO CHASE"
    assert props["Premium/Discount"]["select"]["name"] == "Premium"


def test_wait_for_retest_signal_maps_correctly():
    props = ne.signal_to_notion_properties(_sample_signal(signal_type="WAIT FOR RETEST"))
    assert props["Signal Type"]["select"]["name"] == "WAIT FOR RETEST"


def test_new_unseen_signal_type_is_passed_through_not_dropped():
    """spec: 'preserve any additional existing signal types instead of
    breaking them' — must not remap/validate against a fixed enum."""
    props = ne.signal_to_notion_properties(_sample_signal(signal_type="SOME FUTURE SIGNAL TYPE"))
    assert props["Signal Type"]["select"]["name"] == "SOME FUTURE SIGNAL TYPE"


def test_missing_optional_fields_does_not_raise():
    minimal_signal = {"symbol": "ETH/USDT", "signal_type": "WATCHLIST / DEVELOPING SETUP", "current_price": 4165.0}
    props = ne.signal_to_notion_properties(minimal_signal)  # must not raise
    assert props["Symbol"]["title"][0]["text"]["content"] == "ETH/USDT"
    assert "Direction" not in props   # omitted, not set to something invalid
    assert props["Entry Price"]["number"] == 4165.0  # falls back to current_price
    assert "Setup Grade" not in props
    assert "R:R" not in props
    assert "Chase Score" not in props
    assert props["Preferred Zone"]["rich_text"][0]["text"]["content"] == ""
    assert props["Premium/Discount"]["select"]["name"] == "Neutral"


def test_none_direction_is_omitted_not_errored():
    props = ne.signal_to_notion_properties(_sample_signal(direction=None))
    assert "Direction" not in props


def test_status_property_never_set():
    """Manual Status changes in Notion must never be touched — we must
    never write this property on create OR anywhere else."""
    props = ne.signal_to_notion_properties(_sample_signal())
    assert "Status" not in props


def test_priority_score_never_set():
    """Priority Score is a Notion FORMULA property — the API rejects
    writes to it. Must never appear in our payload."""
    props = ne.signal_to_notion_properties(_sample_signal())
    assert "Priority Score" not in props


# --------------------------------------------------------------------------- #
# NotionExporter — construction / safe-disable paths
# --------------------------------------------------------------------------- #
def test_disabled_by_config_never_touches_client():
    exporter = ne.NotionExporter(api_key="fake", database_id="fake-db", enabled=False)
    assert exporter.enabled is False
    assert exporter.export_signal(_sample_signal()) is False


def test_missing_credentials_disables_safely():
    exporter = ne.NotionExporter(api_key=None, database_id=None, enabled=True)
    assert exporter.enabled is False
    assert exporter.export_signal(_sample_signal()) is False  # never raises


def test_missing_package_disables_safely(monkeypatch):
    monkeypatch.setattr(ne, "_NOTION_CLIENT_AVAILABLE", False)
    exporter = ne.NotionExporter(api_key="fake", database_id="fake-db", enabled=True)
    assert exporter.enabled is False


def test_none_signal_never_raises():
    exporter = ne.NotionExporter(api_key="fake", database_id="fake-db", enabled=False)
    assert exporter.export_signal(None) is False


# --------------------------------------------------------------------------- #
# export_signal() — mocked client, real safety-contract behavior
# --------------------------------------------------------------------------- #
def _live_exporter():
    exporter = ne.NotionExporter(api_key="fake", database_id="fake-db", enabled=True, max_retries=3, retry_delay_seconds=0.001)
    exporter.enabled = True  # force past the "no real package" guard for this unit test
    exporter.client = MagicMock()
    return exporter


def test_successful_export_creates_page():
    exporter = _live_exporter()
    result = exporter.export_signal(_sample_signal())
    assert result is True
    exporter.client.pages.create.assert_called_once()
    call_kwargs = exporter.client.pages.create.call_args.kwargs
    assert call_kwargs["parent"] == {"database_id": "fake-db"}
    assert "properties" in call_kwargs


def test_duplicate_signal_within_window_is_skipped():
    exporter = _live_exporter()
    sig = _sample_signal()
    assert exporter.export_signal(sig) is True
    assert exporter.export_signal(sig) is False  # duplicate, same call, immediately after
    assert exporter.client.pages.create.call_count == 1


def test_duplicate_window_expiry_allows_re_export():
    exporter = _live_exporter()
    exporter.dedup_window_hours = 1.0
    sig = _sample_signal()
    key = ne._dedup_key(sig)
    exporter._recent_exports[key] = datetime.now(timezone.utc) - timedelta(hours=2)  # simulate an old export
    assert exporter.export_signal(sig) is True
    assert exporter.client.pages.create.call_count == 1


def test_auth_error_disables_further_attempts_without_retry_storm():
    exporter = _live_exporter()
    monkeypatch_error = _FakeAPIError(status=401)
    exporter.client.pages.create.side_effect = ne._NotionAPIError if False else None

    import notion_exporter as mod
    original_error_cls = mod._NotionAPIError
    mod._NotionAPIError = _FakeAPIError
    try:
        exporter.client.pages.create.side_effect = _FakeAPIError(status=401)
        result = exporter.export_signal(_sample_signal())
        assert result is False
        assert exporter._auth_failed is True
        assert exporter.client.pages.create.call_count == 1  # no retries burned on a bad key

        # A second call must short-circuit immediately without even trying the client.
        exporter.client.pages.create.reset_mock()
        result2 = exporter.export_signal(_sample_signal(symbol="ETH/USDT"))
        assert result2 is False
        assert exporter.client.pages.create.call_count == 0
    finally:
        mod._NotionAPIError = original_error_cls


def test_retryable_error_retries_then_succeeds():
    exporter = _live_exporter()
    import notion_exporter as mod
    original_error_cls = mod._NotionAPIError
    mod._NotionAPIError = _FakeAPIError
    try:
        exporter.client.pages.create.side_effect = [_FakeAPIError(status=503), _FakeAPIError(status=503), MagicMock()]
        result = exporter.export_signal(_sample_signal())
        assert result is True
        assert exporter.client.pages.create.call_count == 3
    finally:
        mod._NotionAPIError = original_error_cls


def test_retries_exhausted_returns_false_never_raises():
    exporter = _live_exporter()
    import notion_exporter as mod
    original_error_cls = mod._NotionAPIError
    mod._NotionAPIError = _FakeAPIError
    try:
        exporter.client.pages.create.side_effect = _FakeAPIError(status=503)
        result = exporter.export_signal(_sample_signal())  # must not raise even after exhausting retries
        assert result is False
        assert exporter.client.pages.create.call_count == exporter.max_retries
    finally:
        mod._NotionAPIError = original_error_cls


def test_completely_unexpected_exception_never_propagates():
    """Absolute safety net: even a totally unrelated exception type from
    the client (network stack, bug, whatever) must never crash the caller."""
    exporter = _live_exporter()
    exporter.client.pages.create.side_effect = RuntimeError("something totally unexpected")
    result = exporter.export_signal(_sample_signal())
    assert result is False  # never raises, just reports failure


def test_export_never_raises_even_with_malformed_signal():
    exporter = _live_exporter()
    malformed = {"symbol": None, "signal_type": None, "rr": "not-a-dict", "extension": 12345}
    result = exporter.export_signal(malformed)  # must not raise
    assert result is False
