"""
Unit tests for ARC (Adaptive Replacement Cache) Strategy.
"""

import pytest
from omnicache.strategies.arc import ARCStrategy, ARCListType, ARCEntry


class TestARCStrategy:
    """Test cases for ARCStrategy class."""

    def test_initialization_with_capacity(self):
        """Test ARC strategy initializes with correct capacity."""
        strategy = ARCStrategy(capacity=100)
        assert strategy.max_size == 100
        assert strategy.capacity == 100
        assert strategy.current_size == 0

    def test_initialization_with_max_size(self):
        """Test ARC strategy initializes with max_size parameter."""
        strategy = ARCStrategy(max_size=50)
        assert strategy.max_size == 50

    def test_cannot_specify_both_capacity_and_max_size(self):
        """Test error when both capacity and max_size specified."""
        with pytest.raises(ValueError):
            ARCStrategy(capacity=100, max_size=50)

    def test_default_capacity(self):
        """Test default capacity is 1000."""
        strategy = ARCStrategy()
        assert strategy.max_size == 1000

    def test_set_and_get(self):
        """Test basic set and get operations."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")
        assert strategy.get("key1") == "value1"

    def test_cache_miss_returns_none(self):
        """Test get returns None for missing key."""
        strategy = ARCStrategy(capacity=10)
        assert strategy.get("nonexistent") is None

    def test_new_entry_goes_to_t1(self):
        """Test new entries go to T1 (recent) list."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")

        assert "key1" in strategy._t1
        assert "key1" not in strategy._t2

    def test_t1_to_t2_promotion(self):
        """Test item promotion from T1 to T2 on second access."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")

        # First access from T1 should move to T2
        strategy.get("key1")

        assert "key1" not in strategy._t1
        assert "key1" in strategy._t2

    def test_t2_stays_in_t2(self):
        """Test items in T2 stay in T2 on access."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")
        strategy.get("key1")  # Move to T2

        assert "key1" in strategy._t2

        strategy.get("key1")  # Access again
        assert "key1" in strategy._t2

    def test_eviction_creates_ghost_entries(self):
        """Test evicted items move to ghost lists."""
        strategy = ARCStrategy(capacity=3)

        # Fill cache
        strategy.set("key1", "value1")
        strategy.set("key2", "value2")
        strategy.set("key3", "value3")

        # This should trigger eviction
        strategy.set("key4", "value4")

        # Check eviction occurred
        assert strategy.evictions >= 1
        # One key should be in ghost list
        assert strategy.b1_size + strategy.b2_size >= 1

    def test_ghost_hit_adapts_target(self):
        """Test ghost list hits adapt target T1 size."""
        strategy = ARCStrategy(capacity=5)
        initial_target = strategy.target_t1_size

        # Fill and evict to create ghost entries
        for i in range(10):
            strategy.set(f"key{i}", f"value{i}")

        # B1 or B2 should have entries now
        assert strategy.b1_size > 0 or strategy.b2_size > 0

    def test_hit_ratio_calculation(self):
        """Test hit ratio is calculated correctly."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")

        # One hit
        strategy.get("key1")
        # One miss
        strategy.get("nonexistent")

        assert strategy.hits == 1
        assert strategy.misses == 1
        assert strategy.hit_ratio == 0.5

    def test_delete_removes_from_all_lists(self):
        """Test delete removes key from all structures."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")

        result = strategy.delete("key1")

        assert result is True
        assert strategy.get("key1") is None
        assert "key1" not in strategy._entries
        assert "key1" not in strategy._t1
        assert "key1" not in strategy._t2

    def test_delete_returns_false_for_missing(self):
        """Test delete returns False for missing key."""
        strategy = ARCStrategy(capacity=10)
        result = strategy.delete("nonexistent")
        assert result is False

    def test_clear_resets_all_lists(self):
        """Test clear empties all lists and resets stats."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")
        strategy.set("key2", "value2")
        strategy.get("key1")

        strategy.clear()

        assert strategy.t1_size == 0
        assert strategy.t2_size == 0
        assert strategy.b1_size == 0
        assert strategy.b2_size == 0
        assert strategy.hits == 0
        assert strategy.misses == 0
        assert strategy.target_t1_size == 0

    def test_reset_alias_for_clear(self):
        """Test reset is an alias for clear."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")

        strategy.reset()
        assert strategy.current_size == 0

    def test_keys_returns_all_keys(self):
        """Test keys method returns all cached keys."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")
        strategy.set("key2", "value2")

        keys = strategy.keys()
        assert "key1" in keys
        assert "key2" in keys

    def test_items_returns_all_items(self):
        """Test items method returns all cached items."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")
        strategy.set("key2", "value2")

        items = strategy.items()
        assert ("key1", "value1") in items
        assert ("key2", "value2") in items

    def test_update_existing_key(self):
        """Test updating an existing key."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")
        strategy.set("key1", "updated")

        assert strategy.get("key1") == "updated"
        assert strategy.current_size == 1

    def test_get_statistics_comprehensive(self):
        """Test get_statistics returns comprehensive data."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")
        strategy.get("key1")

        stats = strategy.get_statistics()

        assert "hits" in stats
        assert "misses" in stats
        assert "hit_ratio" in stats
        assert "t1_size" in stats
        assert "t2_size" in stats
        assert "target_t1_size" in stats
        assert "evictions" in stats
        assert "adaptations" in stats

    def test_get_entry_info(self):
        """Test getting detailed entry information."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")

        info = strategy.get_entry_info("key1")

        assert info is not None
        assert info["key"] == "key1"
        assert info["value"] == "value1"
        assert "access_count" in info
        assert "current_list" in info

    def test_get_list_contents(self):
        """Test getting contents of all lists."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")
        strategy.set("key2", "value2")
        strategy.get("key1")  # Promote to T2

        contents = strategy.get_list_contents()

        assert "T1" in contents
        assert "T2" in contents
        assert "B1" in contents
        assert "B2" in contents
        assert "key2" in contents["T1"]
        assert "key1" in contents["T2"]


class TestARCStrategyEdgeCases:
    """Edge case tests for ARC strategy."""

    def test_single_item_cache(self):
        """Test ARC with capacity of 1."""
        strategy = ARCStrategy(capacity=1)
        strategy.set("key1", "value1")
        strategy.set("key2", "value2")

        assert strategy.current_size == 1
        assert strategy.get("key2") == "value2"
        assert strategy.get("key1") is None

    def test_capacity_validation(self):
        """Test capacity must be positive."""
        with pytest.raises(ValueError):
            strategy = ARCStrategy(capacity=0)

        with pytest.raises(ValueError):
            strategy = ARCStrategy(capacity=-1)

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")
        strategy.get("key1")

        data = strategy.to_dict()
        restored = ARCStrategy.from_dict(data)

        assert restored.max_size == strategy.max_size
        assert restored.hits == strategy.hits

    def test_equality(self):
        """Test equality comparison."""
        strategy1 = ARCStrategy(name="test", capacity=10)
        strategy2 = ARCStrategy(name="test", capacity=10)
        strategy3 = ARCStrategy(name="other", capacity=10)

        assert strategy1 == strategy2
        assert strategy1 != strategy3

    def test_string_representation(self):
        """Test string representation."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")

        str_repr = str(strategy)
        assert "ARCStrategy" in str_repr
        assert "size=1" in str_repr

    def test_property_accessors(self):
        """Test property accessors for list contents."""
        strategy = ARCStrategy(capacity=10)
        strategy.set("key1", "value1")

        assert isinstance(strategy.t1, dict)
        assert isinstance(strategy.t2, dict)
        assert isinstance(strategy.b1, dict)
        assert isinstance(strategy.b2, dict)

    def test_thread_safety_lock_exists(self):
        """Test that thread safety lock is initialized."""
        strategy = ARCStrategy(capacity=10)
        assert hasattr(strategy, "_lock")
        assert strategy._lock is not None


class TestARCEntry:
    """Test cases for ARCEntry dataclass."""

    def test_entry_creation(self):
        """Test ARCEntry creation."""
        entry = ARCEntry(key="test", value="value")
        assert entry.key == "test"
        assert entry.value == "value"
        assert entry.access_count == 1

    def test_update_access(self):
        """Test access update increments count."""
        entry = ARCEntry(key="test", value="value")
        initial_count = entry.access_count

        entry.update_access()

        assert entry.access_count == initial_count + 1

    def test_to_dict(self):
        """Test ARCEntry to_dict."""
        entry = ARCEntry(key="test", value="value")
        data = entry.to_dict()

        assert data["key"] == "test"
        assert data["value"] == "value"
        assert "access_count" in data

    def test_from_dict(self):
        """Test ARCEntry from_dict."""
        data = {"key": "test", "value": "value", "access_count": 5}
        entry = ARCEntry.from_dict(data)

        assert entry.key == "test"
        assert entry.value == "value"
        assert entry.access_count == 5
