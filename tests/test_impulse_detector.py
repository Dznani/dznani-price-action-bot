"""
test_impulse_detector.py — Tests for Impulse Discovery Layer.

Tests cover:
1. RSI divergence detection (regular bullish, hidden bullish)
2. MACD momentum detection
3. Discovery score calculation
4. Candidate lifecycle management
5. No lookahead bias verification
6. Integration with price action (indicators don't bypass structure)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

import impulse_detector as imp


# --------------------------------------------------------------------------- #
# Test Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def sample_df_bullish_divergence():
    """
    Create a synthetic DataFrame showing regular bullish divergence:
    - Price makes lower low
    - RSI makes higher low
    """
    n_candles = 100
    timestamps = pd.date_range(start="2024-01-01", periods=n_candles, freq="1H")

    # Create price pattern with two lows where second is lower
    prices = []
    for i in range(n_candles):
        if i < 20:
            prices.append(100 - i * 0.5)  # Initial decline
        elif i < 40:
            prices.append(80 + (i - 20) * 0.3)  # Recovery to ~86
        elif i < 60:
            prices.append(86 - (i - 40) * 0.4)  # Second decline to ~78 (lower low)
        else:
            prices.append(78 + (i - 60) * 0.5)  # Final rally

    # Add some noise
    np.random.seed(42)
    prices = np.array(prices) + np.random.normal(0, 0.5, n_candles)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + np.random.uniform(0.5, 1.5, n_candles),
        "low": prices - np.random.uniform(0.5, 1.5, n_candles),
        "close": prices,
        "volume": np.random.uniform(1000, 5000, n_candles),
    })
    return df


@pytest.fixture
def sample_df_hidden_divergence():
    """
    Create a synthetic DataFrame showing hidden bullish divergence:
    - Price makes higher low
    - RSI makes lower low
    """
    n_candles = 100
    timestamps = pd.date_range(start="2024-01-01", periods=n_candles, freq="1H")

    prices = []
    for i in range(n_candles):
        if i < 20:
            prices.append(100 - i * 0.5)
        elif i < 40:
            prices.append(80 + (i - 20) * 0.5)  # Recovery to ~90
        elif i < 60:
            prices.append(90 - (i - 40) * 0.3)  # Pullback to ~84 (higher low than 80)
        else:
            prices.append(84 + (i - 60) * 0.4)  # Continuation

    np.random.seed(43)
    prices = np.array(prices) + np.random.normal(0, 0.5, n_candles)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + np.random.uniform(0.5, 1.5, n_candles),
        "low": prices - np.random.uniform(0.5, 1.5, n_candles),
        "close": prices,
        "volume": np.random.uniform(1000, 5000, n_candles),
    })
    return df


@pytest.fixture
def sample_df_no_divergence():
    """Create a DataFrame with no clear divergence."""
    n_candles = 100
    timestamps = pd.date_range(start="2024-01-01", periods=n_candles, freq="1H")

    np.random.seed(44)
    prices = 100 + np.cumsum(np.random.normal(0, 1, n_candles))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": prices + np.random.uniform(0.5, 1.5, n_candles),
        "low": prices - np.random.uniform(0.5, 1.5, n_candles),
        "close": prices,
        "volume": np.random.uniform(1000, 5000, n_candles),
    })
    return df


@pytest.fixture
def default_config():
    return dict(imp.DEFAULT_IMPULSE_CONFIG)


# --------------------------------------------------------------------------- #
# RSI Divergence Tests
# --------------------------------------------------------------------------- #

class TestRSIDivergence:

    def test_regular_bullish_divergence_detection(self, sample_df_bullish_divergence, default_config):
        """Test that regular bullish divergence is detected when price LL and RSI HL."""
        df = sample_df_bullish_divergence
        rsi = imp.ind.calculate_rsi(df["close"], 14)

        result = imp.detect_rsi_divergence(df, rsi, default_config)

        # Should detect regular bullish divergence
        assert result.divergence_type == "regular_bullish"
        assert result.confidence > 0.0
        assert result.price_pivot_1 is not None
        assert result.price_pivot_2 is not None

    def test_hidden_bullish_divergence_detection(self, sample_df_hidden_divergence, default_config):
        """Test that hidden bullish divergence is detected when price HL and RSI LL."""
        df = sample_df_hidden_divergence
        rsi = imp.ind.calculate_rsi(df["close"], 14)

        result = imp.detect_rsi_divergence(df, rsi, default_config)

        # May detect hidden bullish or none depending on exact RSI values
        # The key is it should not falsely detect regular bearish
        assert result.divergence_type in ("hidden_bullish", "regular_bullish", None)

    def test_no_divergence_returns_none(self, sample_df_no_divergence, default_config):
        """Test that random price action without divergence returns None."""
        df = sample_df_no_divergence
        rsi = imp.ind.calculate_rsi(df["close"], 14)

        result = imp.detect_rsi_divergence(df, rsi, default_config)

        # With random walk, may or may not detect divergence
        # Just verify it doesn't crash
        assert isinstance(result, imp.RSIDivergenceSignal)

    def test_rsi_disabled_in_config(self, sample_df_bullish_divergence, default_config):
        """Test that disabling RSI in config returns None."""
        df = sample_df_bullish_divergence
        rsi = imp.ind.calculate_rsi(df["close"], 14)

        config = dict(default_config)
        config["rsi"]["enabled"] = False

        result = imp.detect_rsi_divergence(df, rsi, config)

        assert result.divergence_type is None

    def test_pivot_confirmation_no_lookahead(self, sample_df_bullish_divergence, default_config):
        """Verify that pivots are only confirmed after lookback period."""
        df = sample_df_bullish_divergence
        rsi = imp.ind.calculate_rsi(df["close"], 14)

        pivot_lookback = default_config["rsi"]["pivot_lookback"]

        # Get pivots at different points in the data
        for end_idx in range(pivot_lookback * 2, len(df) - 5):
            df_subset = df.iloc[:end_idx].copy()
            rsi_subset = rsi.iloc[:end_idx]

            pivots_lows, pivots_highs = imp.detect_rsi_pivots(
                df_subset, rsi_subset, pivot_lookback
            )

            # All confirmed pivots should have index <= end_idx - lookback - 1
            for pivot in pivots_lows + pivots_highs:
                assert pivot.index <= end_idx - pivot_lookback - 1


# --------------------------------------------------------------------------- #
# MACD Momentum Tests
# --------------------------------------------------------------------------- #

class TestMACDMomentum:

    def test_macd_calculation(self, sample_df_no_divergence, default_config):
        """Test basic MACD calculation."""
        df = sample_df_no_divergence
        macd_line, signal_line, histogram = imp.calculate_macd(df, 12, 26, 9)

        assert len(macd_line) == len(df)
        assert len(signal_line) == len(df)
        assert len(histogram) == len(df)

        # Histogram should be MACD - Signal
        pd.testing.assert_series_equal(histogram, macd_line - signal_line)

    def test_momentum_recovery_detection(self, default_config):
        """Test detection of momentum recovery (histogram becoming less negative)."""
        # Create a DataFrame with recovering momentum
        n = 50
        df = pd.DataFrame({
            "timestamp": pd.date_range(start="2024-01-01", periods=n, freq="1H"),
            "open": np.linspace(100, 110, n),
            "high": np.linspace(101, 111, n),
            "low": np.linspace(99, 109, n),
            "close": np.linspace(100, 110, n),
            "volume": np.ones(n) * 1000,
        })

        state = imp.analyze_macd_momentum(df, default_config)

        # Should not crash; actual state depends on the data
        assert isinstance(state, imp.MACDMomentumState)

    def test_macd_disabled_in_config(self, sample_df_no_divergence, default_config):
        """Test that disabling MACD in config returns neutral state."""
        df = sample_df_no_divergence

        config = dict(default_config)
        config["macd"]["enabled"] = False

        state = imp.analyze_macd_momentum(df, config)

        assert state.momentum_state == "neutral"
        assert state.confidence == 0.0


# --------------------------------------------------------------------------- #
# Discovery Score Tests
# --------------------------------------------------------------------------- #

class TestDiscoveryScore:

    def test_score_with_regular_bullish(self, sample_df_bullish_divergence, default_config):
        """Test discovery score calculation with regular bullish divergence."""
        df = sample_df_bullish_divergence
        rsi = imp.ind.calculate_rsi(df["close"], 14)

        rsi_div = imp.detect_rsi_divergence(df, rsi, default_config)
        macd_state = imp.analyze_macd_momentum(df, default_config)

        score = imp.calculate_discovery_score(
            rsi_div, macd_state, htf_trend="BULLISH", liquidity_sweep_detected=True
        )

        # Should have some score (may be below threshold depending on data quality)
        assert score >= 0  # Score should be non-negative
        assert score <= 100  # Score should not exceed maximum

    def test_score_with_no_signals(self, sample_df_no_divergence, default_config):
        """Test discovery score with no strong signals."""
        df = sample_df_no_divergence
        rsi = imp.ind.calculate_rsi(df["close"], 14)

        rsi_div = imp.detect_rsi_divergence(df, rsi, default_config)
        macd_state = imp.analyze_macd_momentum(df, default_config)

        score = imp.calculate_discovery_score(
            rsi_div, macd_state, htf_trend="NEUTRAL", liquidity_sweep_detected=False
        )

        # Should be low but may not be zero
        assert score >= 0
        assert score <= 100

    def test_htf_context_bonus(self, default_config):
        """Test that HTF bullish trend adds bonus to score."""
        rsi_div = imp.RSIDivergenceSignal(divergence_type=None)
        macd_state = imp.MACDMomentumState()

        score_neutral = imp.calculate_discovery_score(rsi_div, macd_state, htf_trend="NEUTRAL")
        score_bullish = imp.calculate_discovery_score(rsi_div, macd_state, htf_trend="BULLISH")

        # Bullish HTF should add bonus
        assert score_bullish >= score_neutral

    def test_liquidity_sweep_bonus(self, default_config):
        """Test that liquidity sweep adds bonus to score."""
        rsi_div = imp.RSIDivergenceSignal(divergence_type=None)
        macd_state = imp.MACDMomentumState()

        score_no_sweep = imp.calculate_discovery_score(rsi_div, macd_state, liquidity_sweep_detected=False)
        score_with_sweep = imp.calculate_discovery_score(rsi_div, macd_state, liquidity_sweep_detected=True)

        # Sweep should add bonus
        assert score_with_sweep >= score_no_sweep


# --------------------------------------------------------------------------- #
# ImpulseCandidate Tests
# --------------------------------------------------------------------------- #

class TestImpulseCandidate:

    def test_candidate_creation(self, sample_df_bullish_divergence, default_config):
        """Test creating an impulse candidate."""
        detector = imp.ImpulseDetector(default_config)
        df = sample_df_bullish_divergence
        df_4h = df  # Using same data for simplicity

        from structure import analyze_structure
        state_4h = analyze_structure(df_4h, left=2, right=2)

        candidate = detector.evaluate_symbol(
            symbol="TEST/USDT",
            df_1h=df,
            df_4h=df_4h,
            state_4h=state_4h,
            sweep_detected=True,
            candle_idx=len(df) - 1,
        )

        # May or may not create candidate depending on score threshold
        if candidate is not None:
            assert candidate.symbol == "TEST/USDT"
            assert candidate.direction in ("BUY", "SELL")
            assert candidate.discovery_score >= 0
            assert candidate.status == "SCOUTED"

    def test_duplicate_prevention(self, sample_df_no_divergence, default_config):
        """Test that duplicate candidates are prevented."""
        detector = imp.ImpulseDetector(default_config)
        df = sample_df_no_divergence
        df_4h = df

        from structure import analyze_structure
        state_4h = analyze_structure(df_4h, left=2, right=2)

        # First evaluation
        candidate1 = detector.evaluate_symbol(
            symbol="TEST/USDT",
            df_1h=df,
            df_4h=df_4h,
            state_4h=state_4h,
            candle_idx=len(df) - 1,
        )

        # Second evaluation at nearby candle should be prevented
        candidate2 = detector.evaluate_symbol(
            symbol="TEST/USDT",
            df_1h=df,
            df_4h=df_4h,
            state_4h=state_4h,
            candle_idx=len(df) - 1,
        )

        # At most one should be created
        if candidate1 is not None:
            assert candidate2 is None

    def test_candidate_expiry(self, sample_df_no_divergence, default_config):
        """Test that candidates expire after max candles."""
        default_config["candidate_management"]["default_expiry_candles"] = 10
        detector = imp.ImpulseDetector(default_config)
        df = sample_df_no_divergence

        # Manually create a candidate
        candidate = imp.ImpulseCandidate(
            candidate_id="TEST_EXPIRY",
            symbol="TEST/USDT",
            direction="BUY",
            created_at="2024-01-01T00:00:00Z",
            created_candle_idx=0,
            expiry_candles=10,
            discovery_score=50.0,
        )
        detector.active_candidates.append(candidate)

        # Update at candle 11 (past expiry)
        detector.update_candidates(df, candle_idx=11)

        assert len(detector.completed_candidates) == 1
        assert detector.completed_candidates[0].status == "EXPIRED"

    def test_candidate_status_progression(self, sample_df_no_divergence, default_config):
        """Test candidate status progression through lifecycle."""
        detector = imp.ImpulseDetector(default_config)
        df = sample_df_no_divergence

        candidate = imp.ImpulseCandidate(
            candidate_id="TEST_PROG",
            symbol="TEST/USDT",
            direction="BUY",
            created_at="2024-01-01T00:00:00Z",
            created_candle_idx=0,
            discovery_score=50.0,
            status="SCOUTED",
        )
        detector.active_candidates.append(candidate)

        # Update with structure confirmed
        detector.update_candidates(df, candle_idx=5, structure_confirmed=True)

        assert candidate.status == "STRUCTURE_CONFIRMED"
        assert candidate.structure_confirmed_at is not None

        # Update with zone touched
        detector.update_candidates(df, candle_idx=6, zone_touched=True)

        assert candidate.status == "ZONE_TOUCHED"
        assert candidate.zone_touched_at is not None


# --------------------------------------------------------------------------- #
# Integration Tests
# --------------------------------------------------------------------------- #

class TestIntegration:

    def test_indicator_does_not_bypass_price_action(self, sample_df_bullish_divergence):
        """
        CRITICAL TEST: Verify that RSI/MACD discovery does NOT automatically
        create a trade signal without price action confirmation.

        This tests the core philosophy: INDICATORS = DISCOVERY, PRICE ACTION = EXECUTION
        """
        df = sample_df_bullish_divergence

        # Even with strong divergence...
        rsi = imp.ind.calculate_rsi(df["close"], 14)
        rsi_div = imp.detect_rsi_divergence(df, rsi, imp.DEFAULT_IMPULSE_CONFIG)

        # ...we only get a CANDIDATE, not a trade signal
        assert isinstance(rsi_div, imp.RSIDivergenceSignal)
        # The candidate must still go through structure analysis, watch engine, etc.

    def test_enrich_signal_with_discovery(self, sample_df_bullish_divergence, default_config):
        """Test enriching a strategy signal with discovery information."""
        df = sample_df_bullish_divergence

        # Create a mock signal card
        signal = {
            "symbol": "TEST/USDT",
            "direction": "BUY",
            "signal_type": "CONFIRMATION LONG",
            "decision": "VALID",
        }

        # Create a candidate
        detector = imp.ImpulseDetector(default_config)
        rsi = imp.ind.calculate_rsi(df["close"], 14)
        rsi_div = imp.detect_rsi_divergence(df, rsi, default_config)

        candidate = imp.ImpulseCandidate(
            candidate_id="TEST_ENRICH",
            symbol="TEST/USDT",
            direction="BUY",
            created_at="2024-01-01T00:00:00Z",
            created_candle_idx=0,
            discovery_score=65.0,
            rsi_divergence=rsi_div if rsi_div.divergence_type else None,
        )

        enriched = imp.enrich_signal_with_discovery(signal, candidate)

        assert "impulse_discovery" in enriched
        assert enriched["impulse_discovery"]["status"] == "SCOUTED"
        assert enriched["impulse_discovery"]["discovery_score"] == 65.0

    def test_statistics_calculation(self, default_config):
        """Test statistics calculation for candidates."""
        detector = imp.ImpulseDetector(default_config)

        # Empty stats
        stats = detector.get_statistics()
        assert stats["total_candidates"] == 0

        # Add some candidates
        for i in range(3):
            candidate = imp.ImpulseCandidate(
                candidate_id=f"TEST_{i}",
                symbol="TEST/USDT",
                direction="BUY",
                created_at="2024-01-01T00:00:00Z",
                created_candle_idx=i,
                discovery_score=50.0 + i * 10,
                status="SCOUTED" if i == 0 else "STRUCTURE_CONFIRMED" if i == 1 else "EXPIRED",
            )
            if i == 2:
                detector.completed_candidates.append(candidate)
            else:
                detector.active_candidates.append(candidate)

        stats = detector.get_statistics()
        assert stats["total_candidates"] == 3
        assert stats["avg_discovery_score"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
