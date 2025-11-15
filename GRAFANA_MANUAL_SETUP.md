# 🔧 Grafana Manual Setup Guide

## Masalah yang Diselesaikan
Error: `Datasource ${DS_INFLUXDB} was not found`

## ✅ Solusi yang Sudah Diterapkan

### 1. **Fixed Datasource UID**
Dashboard JSON sudah diupdate untuk menggunakan UID fixed: `influxdb-datasource`

### 2. **Environment Variables**
Docker Compose sudah dikonfigurasi untuk pass InfluxDB credentials ke Grafana.

### 3. **Auto-Provisioning**
- ✅ Datasource otomatis terbuat saat Grafana start
- ✅ Dashboard otomatis ter-load
- ✅ Token InfluxDB otomatis dikonfigurasi

---

## 🎯 Cara Verifikasi Setup

### 1. Cek Datasource
```bash
curl -s "http://localhost:3030/api/datasources" | python3 -m json.tool
```

Harus ada datasource dengan:
- `name`: "InfluxDB"
- `uid`: "influxdb-datasource"
- `isDefault`: true
- `jsonData.defaultBucket`: "vehicle_counting"
- `jsonData.organization`: "pengmas"

### 2. Cek Dashboard
```bash
curl -s "http://localhost:3030/api/search?query=vehicle" | python3 -m json.tool
```

Harus ada dashboard:
- `uid`: "vehicle-traffic-dashboard"
- `title`: "Vehicle Traffic Dashboard"

### 3. Test di Browser
```
http://localhost:5000/analytics
```

Dashboard harus muncul tanpa error datasource.

---

## 🔨 Jika Masih Ada Masalah (Manual Setup)

### Opsi 1: Login ke Grafana dan Setup Manual

1. **Buka Grafana**
   ```
   http://localhost:3030
   ```

2. **Login**
   - Username: `admin`
   - Password: `admin123`

3. **Add Data Source**
   - Go to: ⚙️ Configuration → Data Sources
   - Click "Add data source"
   - Select "InfluxDB"
   
4. **Configure InfluxDB**
   ```
   Query Language: Flux
   URL: http://influxdb:8086
   Organization: pengmas
   Token: tokenrahasia
   Default Bucket: vehicle_counting
   ```
   
   Click "Save & Test" ✅

5. **Import Dashboard**
   - Go to: ➕ Create → Import
   - Upload file: `grafana-provisioning/dashboards/vehicle-traffic.json`
   - Select datasource: "InfluxDB"
   - Click "Import"

### Opsi 2: Rebuild dari Scratch

1. **Hapus Grafana Data**
   ```bash
   docker-compose down
   sudo rm -rf grafana-data/*
   docker-compose up -d
   ```

2. **Tunggu Provisioning**
   ```bash
   docker logs grafana -f
   ```
   
   Tunggu sampai muncul:
   ```
   logger=provisioning.datasources msg="inserting datasource"
   logger=provisioning.dashboard msg="finished to provision dashboards"
   ```

---

## 📊 Query Testing (Manual)

Jika dashboard sudah muncul tapi panel kosong, test query manual:

### 1. Buka Dashboard
```
http://localhost:3030/d/vehicle-traffic-dashboard
```

### 2. Edit Panel
- Klik title panel → Edit
- Cek datasource: harus "InfluxDB"
- Test query di Query Inspector

### 3. Sample Query (untuk testing)
```flux
from(bucket: "vehicle_counting")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] == "vehicle_counts")
  |> filter(fn: (r) => r["_field"] == "count")
```

---

## 🐛 Troubleshooting

### Error: "Failed to connect to datasource"
```bash
# Cek InfluxDB berjalan
docker-compose ps influxdb

# Test koneksi dari Grafana ke InfluxDB
docker exec grafana curl -I http://influxdb:8086/health
```

### Error: "No data"
```bash
# Cek apakah ada data di InfluxDB
docker exec influxdb influx query '
from(bucket:"vehicle_counting")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "vehicle_counts")
  |> limit(n: 5)
'
```

Jika tidak ada data:
1. ✅ Pastikan `src/main.py` running
2. ✅ Pastikan InfluxDB logging enabled
3. ✅ Cek logs: `docker logs webserver` atau container yang run detection

### Error: "Dashboard not found"
```bash
# Restart Grafana untuk reload provisioning
docker-compose restart grafana

# Atau reimport manual dashboard
```

---

## ✅ Current Status

**Semuanya sudah di-setup otomatis!**

- ✅ Datasource: `influxdb-datasource` 
- ✅ Dashboard: `vehicle-traffic-dashboard`
- ✅ Anonymous access: Enabled
- ✅ Embedding: Enabled
- ✅ Auto-refresh: 5 seconds

**Langsung akses:**
```
http://localhost:5000/analytics
```

---

## 📝 Files Modified

1. `docker-compose.yml` - Added InfluxDB env vars to Grafana
2. `grafana-provisioning/datasources/influxdb.yml` - Fixed UID
3. `grafana-provisioning/dashboards/vehicle-traffic.json` - Updated datasource references
4. `webserver/app.py` - Added /analytics route
5. `webserver/templates/analytics.html` - New template with embedded dashboard

---

Selamat! Dashboard seharusnya sudah berfungsi dengan baik! 🎉
