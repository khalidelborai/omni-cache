# 🚀 OmniCache Enterprise Demo

**Complete end-to-end demonstration of OmniCache Enterprise features with Docker Compose stack**

This comprehensive demo showcases all enterprise features of OmniCache including ARC strategy, hierarchical caching, ML prefetching, security features, real-time analytics, and event-driven invalidation.

## 📋 Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB+ RAM (for full monitoring stack)
- Ports available: 8000, 3000, 9090, 6379, 5601, 9200, 16686

### 🎯 One-Command Demo

```bash
# Clone and start the full demo
git clone https://github.com/khalidelborai/omni-cache
cd omni-cache
docker-compose up -d

# Wait ~2 minutes for all services to start
# Then visit: http://localhost:8000/dashboard
```

### 🌐 Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| **Demo App** | http://localhost:8000 | FastAPI app with live dashboard |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Grafana** | http://localhost:3000 | Monitoring dashboards (admin/admin) |
| **Prometheus** | http://localhost:9090 | Metrics collection |
| **Jaeger** | http://localhost:16686 | Distributed tracing |
| **Kibana** | http://localhost:5601 | Log analysis |
| **Redis** | localhost:6379 | L2 cache tier |

## 🎪 Demo Features

### 🏢 Enterprise Features Demonstrated

1. **🧠 ARC (Adaptive Replacement Cache)**
   - Self-tuning algorithm with T1/T2 lists
   - Ghost lists (B1/B2) for adaptation
   - Performance comparison with LRU

2. **🏗️ Hierarchical Multi-Tier Caching**
   - L1: Memory (hot data)
   - L2: Redis (warm data)
   - L3: Simulated cloud storage

3. **🤖 ML-Powered Prefetching**
   - Access pattern recognition
   - Predictive recommendations
   - Training simulation

4. **🔐 Zero-Trust Security**
   - AES-256-GCM encryption
   - PII detection
   - GDPR compliance simulation

5. **📊 Real-Time Analytics**
   - Prometheus metrics export
   - Grafana dashboards
   - Performance monitoring

6. **⚡ Event-Driven Invalidation**
   - Reactive cache updates
   - Dependency tracking
   - Real-time sync

### 🎮 Interactive Demo Scenarios

#### Scenario 1: Cache Performance Testing
```bash
# Generate load to see ARC vs LRU performance
curl -X POST http://localhost:8000/api/test/arc

# Monitor hit ratios in real-time
curl http://localhost:8000/api/cache/stats
```

#### Scenario 2: Security Features
```bash
# Test encryption and PII detection
curl -X POST http://localhost:8000/api/test/security

# Monitor security events in Grafana
```

#### Scenario 3: ML Prefetching
```bash
# Create access patterns for ML training
for i in {1..10}; do
  curl http://localhost:8000/api/users/user_$i
done

# Test ML predictions
curl -X POST http://localhost:8000/api/test/ml
```

#### Scenario 4: Load Testing
```bash
# Start load testing profile
docker-compose --profile load-test up -d

# Monitor performance under load in Grafana
# Visit: http://localhost:3000
```

## 📊 Monitoring & Observability

### Grafana Dashboards

The demo includes pre-configured Grafana dashboards showing:

- **Cache Performance**: Hit ratios, operations/sec, response times
- **ARC Strategy**: T1/T2 performance, adaptations, efficiency
- **ML Insights**: Prediction accuracy, prefetch success rates
- **Security Events**: Encryption operations, violations, compliance
- **System Health**: Memory usage, CPU, error rates

### Prometheus Metrics

Key metrics exported:

```prometheus
# Cache performance
omnicache_hit_ratio{cache_name="user_profiles"}
omnicache_operations_total{cache_name="product_catalog",operation="get"}
omnicache_operation_duration_seconds{cache_name="analytics_events"}

# ARC strategy
omnicache_arc_t1_hits{cache_name="user_profiles"}
omnicache_arc_adaptations{cache_name="user_profiles"}

# ML features
omnicache_ml_prediction_accuracy{cache_name="product_catalog"}
omnicache_ml_prefetch_hits_total{cache_name="product_catalog"}

# Security
omnicache_security_events_total{cache_name="secure_data",event_type="encryption"}
omnicache_security_violations_total{cache_name="secure_data"}
```

### Alerts Configuration

Pre-configured alerts for:
- Cache hit ratio < 50% (Warning) / < 20% (Critical)
- High operation latency > 100ms
- Security violations detected
- ML prediction accuracy < 70%
- Memory usage > 90%

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Demo Architecture                        │
├─────────────────────────────────────────────────────────────┤
│  FastAPI App     │  Load Generator  │  Cache Monitor       │
│  • Demo API      │  • Locust        │  • Metrics Export    │
│  • Dashboard     │  • User Sim      │  • Health Checks     │
│  • Enterprise    │  • Patterns      │  • Alerting          │
├─────────────────────────────────────────────────────────────┤
│                    OmniCache Enterprise                     │
├─────────────────────────────────────────────────────────────┤
│  ARC Strategy    │ Hierarchical     │ ML Prefetch          │
│  • T1/T2 Lists   │ • L1 Memory      │ • Pattern Learn      │
│  • B1/B2 Ghost   │ • L2 Redis       │ • Recommendations    │
│  • Auto-adapt    │ • L3 Cloud       │ • Accuracy Track     │
├─────────────────────────────────────────────────────────────┤
│  Security        │ Analytics        │ Event Invalidation   │
│  • Encryption    │ • Prometheus     │ • Dependency Graph   │
│  • PII Detect    │ • Tracing        │ • Reactive Updates   │
│  • GDPR          │ • Dashboards     │ • Event Processing   │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure                           │
├─────────────────────────────────────────────────────────────┤
│  Redis           │ Prometheus       │ Grafana              │
│  • L2 Storage    │ • Metrics        │ • Visualization      │
│  • Clustering    │ • Alerting       │ • Dashboards         │
│                  │                  │                      │
│  Jaeger          │ Elasticsearch    │ Kibana               │
│  • Tracing       │ • Logs           │ • Log Analysis       │
│  • Performance   │ • Search         │ • Debugging          │
└─────────────────────────────────────────────────────────────┘
```

## 🧪 Demo Data

The demo automatically generates:

- **100 Users**: Profiles with departments, roles, preferences
- **500 Products**: Catalog with categories, pricing, inventory
- **200+ Events**: Analytics events, page views, purchases
- **Security Data**: Encrypted sensitive information

Data is pre-loaded into appropriate cache tiers for immediate testing.

## 🎯 Performance Targets

The demo validates these enterprise performance targets:

| Metric | Target | Demo Result |
|--------|--------|-------------|
| **ARC Improvement** | >10% vs LRU | ✅ Demonstrated |
| **ML Miss Reduction** | 30-50% | ✅ Simulated |
| **Security Overhead** | <10% | ✅ Measured |
| **Throughput** | >250k ops/sec | ✅ Load tested |
| **P95 Latency** | <10ms | ✅ Monitored |

## 🔧 Configuration

### Environment Variables

```bash
# Demo app
REDIS_URL=redis://redis:6379/0
PROMETHEUS_URL=http://prometheus:9090
ENVIRONMENT=docker

# Load generator
TARGET_URL=http://omnicache-demo:8000
CONCURRENT_USERS=10
REQUESTS_PER_SECOND=50

# Cache monitor
DEMO_API_URL=http://omnicache-demo:8000
MONITOR_INTERVAL=30
```

### Resource Limits

```yaml
# Adjust in docker-compose.yml for different environments
services:
  redis:
    deploy:
      resources:
        limits:
          memory: 256M

  omnicache-demo:
    deploy:
      resources:
        limits:
          memory: 512M
```

## 🚦 Health Checks

All services include health checks:

```bash
# Check overall system health
curl http://localhost:8000/api/health

# Check individual service health
docker-compose ps
```

## 🎮 Interactive Testing

### CLI Testing

```bash
# Install OmniCache CLI
pip install -e .

# Test enterprise features
omnicache enterprise demo_cache --strategy arc --enable-all --max-size 1000

# Monitor performance
omnicache arc stats demo_cache --watch

# View analytics
omnicache analytics report demo_cache --period 1h
```

### API Testing

```bash
# Test user profiles (ARC cache)
curl http://localhost:8000/api/users/user_1

# Test products (ML-enabled cache)
curl http://localhost:8000/api/products/prod_1

# Test category search
curl http://localhost:8000/api/products/category/Electronics

# Test analytics
curl http://localhost:8000/api/analytics/events/recent?hours=24
```

### Performance Testing

```bash
# Run benchmark suite
docker-compose --profile load-test up -d

# View live performance in Grafana
# http://localhost:3000/d/omnicache-dashboard
```

## 🎓 Learning Scenarios

### 1. Cache Strategy Comparison

**Objective**: Understand ARC vs LRU performance differences

**Steps**:
1. Visit dashboard: http://localhost:8000/dashboard
2. Click "Test ARC Strategy" multiple times
3. Observe hit rate improvements in Grafana
4. Compare response times

### 2. ML Pattern Recognition

**Objective**: See ML prefetching in action

**Steps**:
1. Create access patterns: `curl http://localhost:8000/api/users/user_{1..10}`
2. Test ML predictions: `curl -X POST http://localhost:8000/api/test/ml`
3. Monitor accuracy in Grafana dashboard
4. Observe prefetch hit rates

### 3. Security Compliance

**Objective**: Validate security features

**Steps**:
1. Test encryption: `curl -X POST http://localhost:8000/api/test/security`
2. Monitor security events in Grafana
3. Check GDPR compliance indicators
4. Review audit logs in Kibana

### 4. Scale Under Load

**Objective**: Observe performance under pressure

**Steps**:
1. Start load testing: `docker-compose --profile load-test up -d`
2. Monitor cache hit ratios in real-time
3. Watch ARC adaptations in Grafana
4. Observe system resource usage

## 🛠️ Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check logs
docker-compose logs omnicache-demo

# Restart specific service
docker-compose restart omnicache-demo
```

**Grafana dashboards not loading:**
```bash
# Check Prometheus connectivity
curl http://localhost:9090/api/v1/query?query=up

# Restart Grafana
docker-compose restart grafana
```

**Performance issues:**
```bash
# Check resource usage
docker stats

# Reduce concurrent users
export CONCURRENT_USERS=5
docker-compose --profile load-test up -d
```

### Performance Tuning

```yaml
# Optimize for lower-resource environments
services:
  redis:
    command: redis-server --maxmemory 128mb

  omnicache-demo:
    environment:
      - CACHE_SIZE_LIMIT=1000
```

## 📚 Additional Resources

- **Enterprise Documentation**: `/docs/enterprise/`
- **API Reference**: http://localhost:8000/docs
- **Performance Guide**: `/docs/performance.md`
- **Security Guide**: `/docs/security.md`
- **ML Configuration**: `/docs/ml-configuration.md`

## 🤝 Contributing

Want to enhance the demo?

1. Fork the repository
2. Create feature branch: `git checkout -b feature/demo-enhancement`
3. Add your improvements
4. Test with: `docker-compose up --build`
5. Submit pull request

## 📄 License

This demo is part of the OmniCache project and follows the same licensing terms.

---

**🎉 Ready to explore enterprise caching? Start with: `docker-compose up -d`**