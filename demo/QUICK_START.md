# 🚀 OmniCache Enterprise Demo - Quick Start

**Get the full enterprise demo running in 3 minutes**

## ⚡ 30-Second Start

```bash
git clone https://github.com/khalidelborai/omni-cache
cd omni-cache
docker-compose up -d
```

Wait 2 minutes → Visit **http://localhost:8000/dashboard**

## 🎯 Essential URLs

| What | Where | Purpose |
|------|-------|---------|
| **🎪 Live Demo** | http://localhost:8000 | Interactive enterprise features |
| **📊 Monitoring** | http://localhost:3000 | Grafana dashboards (admin/admin) |
| **🔍 API Docs** | http://localhost:8000/docs | Try all endpoints |
| **📈 Metrics** | http://localhost:9090 | Raw Prometheus data |

## 🎮 Try These First

### 1. Test ARC Strategy (30 seconds)
```bash
curl -X POST http://localhost:8000/api/test/arc
```
Watch performance improvements in the dashboard!

### 2. See All Cache Stats (10 seconds)
```bash
curl http://localhost:8000/api/cache/stats | jq
```

### 3. Load Some Users (20 seconds)
```bash
curl http://localhost:8000/api/users/random?count=10 | jq
```

### 4. Test Security Features (15 seconds)
```bash
curl -X POST http://localhost:8000/api/test/security
```

## 🏃‍♂️ Load Testing (Optional)

Want to see performance under pressure?

```bash
docker-compose --profile load-test up -d
```

Then watch the fireworks in Grafana: http://localhost:3000

## 🎯 What You'll See

- **Real-time cache hit ratios** improving with ARC strategy
- **ML prefetching** learning access patterns
- **Security features** encrypting sensitive data
- **Performance metrics** in beautiful Grafana dashboards
- **Enterprise-grade** monitoring and alerting

## 🛑 Stop Everything

```bash
docker-compose down -v
```

## 🆘 Need Help?

- **Logs**: `docker-compose logs omnicache-demo`
- **Status**: `docker-compose ps`
- **Full Guide**: See [README.md](README.md)

---

**That's it! You're now running a complete enterprise caching solution. 🎉**