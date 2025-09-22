# Data Model: Base Caching Library Architecture

**Feature**: Base Caching Library Architecture
**Phase**: 1 - Data Model Design
**Date**: 2025-09-22

## Core Entities

### Cache

**Purpose**: Central storage abstraction that manages key-value pairs with associated metadata and strategy enforcement.

**Attributes**:
- `name: str` - Unique identifier for the cache instance
- `strategy: Strategy` - Eviction and expiration strategy
- `backend: Backend` - Storage backend implementation
- `max_size: Optional[int]` - Maximum number of entries (None = unlimited)
- `default_ttl: Optional[float]` - Default time-to-live in seconds
- `namespace: str` - Key namespace for multi-tenant scenarios
- `statistics: Statistics` - Real-time performance metrics
- `created_at: datetime` - Cache creation timestamp
- `updated_at: datetime` - Last modification timestamp

**Relationships**:
- Owns multiple CacheEntry instances
- Has one Strategy instance
- Has one Backend instance
- Has one Statistics instance

**State Transitions**:
- `INITIALIZING` → `ACTIVE` (after backend connection)
- `ACTIVE` → `DEGRADED` (backend failures with fallback)
- `ACTIVE` → `MAINTENANCE` (during cache warming/cleanup)
- `MAINTENANCE` → `ACTIVE` (after operations complete)
- Any state → `SHUTDOWN` (explicit termination)

**Validation Rules**:
- name must be unique within application scope
- max_size must be positive if specified
- default_ttl must be positive if specified
- namespace must follow naming conventions (alphanumeric + underscore)

### Strategy

**Purpose**: Defines behavior for cache operations including eviction policies, expiration rules, and storage optimization.

**Attributes**:
- `name: str` - Strategy identifier (e.g., "lru", "lfu", "ttl", "size")
- `policy: EvictionPolicy` - Algorithm for item removal
- `parameters: Dict[str, Any]` - Strategy-specific configuration
- `priority_factor: float` - Weight for multi-strategy scenarios
- `enabled: bool` - Whether strategy is active

**Relationships**:
- Belongs to one Cache instance
- May compose with other Strategy instances (hierarchical)

**State Transitions**:
- `INACTIVE` → `ACTIVE` (when enabled)
- `ACTIVE` → `INACTIVE` (when disabled)
- `ACTIVE` → `EVALUATING` (during eviction decisions)
- `EVALUATING` → `ACTIVE` (after eviction complete)

**Validation Rules**:
- name must be valid strategy type
- priority_factor must be between 0.0 and 1.0
- parameters must conform to strategy schema

### Key

**Purpose**: Unique identifier for cached data with optional namespace and tagging capabilities.

**Attributes**:
- `value: str` - The actual key string
- `namespace: str` - Logical grouping identifier
- `tags: Set[str]` - Labels for bulk operations
- `hash_value: int` - Computed hash for efficient lookups
- `created_at: datetime` - Key creation timestamp

**Relationships**:
- Associated with one CacheEntry
- May belong to multiple Tag groups

**Validation Rules**:
- value cannot be empty string
- namespace must follow naming conventions
- tags must be valid identifiers
- combined key+namespace must be unique per cache

### Value

**Purpose**: Stored data with serialization metadata and access tracking information.

**Attributes**:
- `data: Any` - The actual cached data
- `serialized_data: bytes` - Serialized representation
- `serializer_type: str` - Serialization method used
- `size_bytes: int` - Storage size for capacity management
- `content_type: str` - MIME type or data type hint
- `checksum: str` - Data integrity verification
- `version: int` - For optimistic locking scenarios

**Relationships**:
- Belongs to one CacheEntry
- May reference external resources

**Validation Rules**:
- data cannot be None (use explicit null markers)
- serializer_type must be registered serializer
- size_bytes must match actual serialized size
- checksum must be valid for data

### CacheEntry

**Purpose**: Complete cache record combining key, value, metadata, and access patterns.

**Attributes**:
- `key: Key` - Entry identifier
- `value: Value` - Stored data
- `ttl: Optional[float]` - Time-to-live in seconds
- `expires_at: Optional[datetime]` - Absolute expiration time
- `access_count: int` - Number of times accessed
- `last_accessed: datetime` - Most recent access timestamp
- `created_at: datetime` - Entry creation timestamp
- `updated_at: datetime` - Last modification timestamp
- `priority: float` - Entry importance for eviction decisions
- `locked: bool` - Whether entry is protected from eviction

**Relationships**:
- Belongs to one Cache instance
- Has one Key instance
- Has one Value instance

**State Transitions**:
- `PENDING` → `ACTIVE` (after successful storage)
- `ACTIVE` → `STALE` (after TTL expiration)
- `ACTIVE` → `EVICTED` (removed by strategy)
- `STALE` → `REFRESHING` (during background update)
- `REFRESHING` → `ACTIVE` (after successful refresh)
- Any state → `DELETED` (explicit removal)

**Validation Rules**:
- ttl must be positive if specified
- expires_at must be in future if specified
- access_count must be non-negative
- priority must be between 0.0 and 1.0

### Backend

**Purpose**: Storage implementation abstraction supporting different persistence mechanisms.

**Attributes**:
- `type: str` - Backend type (memory, redis, filesystem)
- `config: Dict[str, Any]` - Backend-specific configuration
- `connection_string: Optional[str]` - Connection details
- `health_status: HealthStatus` - Current operational state
- `capabilities: Set[str]` - Supported features
- `metrics: BackendMetrics` - Performance statistics

**Relationships**:
- Serves one or more Cache instances
- May have failover Backend instances

**State Transitions**:
- `DISCONNECTED` → `CONNECTING` (during initialization)
- `CONNECTING` → `CONNECTED` (successful connection)
- `CONNECTED` → `DEGRADED` (partial failure)
- `DEGRADED` → `CONNECTED` (recovery)
- `CONNECTED` → `DISCONNECTED` (failure or shutdown)

**Validation Rules**:
- type must be registered backend type
- config must conform to backend schema
- connection_string format must be valid for backend type

### Statistics

**Purpose**: Performance metrics and usage data for monitoring and optimization.

**Attributes**:
- `cache_name: str` - Associated cache identifier
- `hit_count: int` - Successful cache retrievals
- `miss_count: int` - Failed cache retrievals
- `eviction_count: int` - Items removed by strategy
- `error_count: int` - Operation failures
- `total_size_bytes: int` - Current storage usage
- `entry_count: int` - Current number of entries
- `avg_access_time_ms: float` - Average operation latency
- `last_reset: datetime` - When statistics were reset
- `collection_interval: float` - Metrics update frequency

**Relationships**:
- Belongs to one Cache instance
- Aggregates data from Backend metrics

**Validation Rules**:
- all count fields must be non-negative
- percentages must be between 0.0 and 100.0
- time measurements must be positive

## Data Flow

### Cache Write Operation
1. **Key** normalization and namespace application
2. **Value** serialization and size calculation
3. **Strategy** evaluation for capacity management
4. **Backend** storage with metadata
5. **Statistics** update for monitoring

### Cache Read Operation
1. **Key** lookup in backend storage
2. **Value** deserialization if found
3. **Strategy** access tracking update
4. **Statistics** hit/miss recording
5. Return data or cache miss indication

### Eviction Process
1. **Strategy** identifies candidates for removal
2. **CacheEntry** state transitions to evicted
3. **Backend** performs actual deletion
4. **Statistics** update eviction counters
5. Space reclamation for new entries

## Schema Constraints

### Referential Integrity
- Every CacheEntry must have valid Key and Value
- Every Cache must have valid Strategy and Backend
- Statistics must reference existing Cache

### Data Consistency
- Cache entry counts must match backend storage
- Statistics must reflect actual operation results
- TTL/expiration times must be consistent across layers

### Performance Constraints
- Key lookups must be O(1) average case
- Strategy evaluations must complete within time bounds
- Backend operations must be non-blocking where possible
- Memory usage must respect configured limits