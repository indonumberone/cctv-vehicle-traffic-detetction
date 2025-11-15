# Grafana Analytics Setup - Panduan Lengkap

## 🎉 Selamat! Setup Berhasil!

Grafana dashboard telah berhasil di-embed ke webserver Flask Anda dengan anonymous access enabled.

## 📍 URL Akses

- **Web Application (dengan Analytics)**: http://localhost:5000/analytics
- **Grafana Direct**: http://localhost:3030
- **InfluxDB**: http://localhost:8086

## 🎯 Fitur yang Telah Disetup

### 1. **Anonymous Access Grafana** ✅

- User **TIDAK PERLU LOGIN** untuk melihat dashboard
- Auto-provisioning datasource InfluxDB
- Auto-provisioning dashboard "Vehicle Traffic Dashboard"

### 2. **Dashboard Components** 📊

#### **Time Series Chart**

- Grafik line chart untuk melihat trend kendaraan over time
- Multiple series (car, motorcycle, bus, truck)
- **Sumbu X**: Waktu (realtime)
- **Sumbu Y**: Jumlah kendaraan

#### **Gauge Panels** (4 buah)

- Total Cars
- Total Motorcycles
- Total Bus
- Total Truck

#### **Bar Chart**

- Crossing events per hour
- Stacked visualization

### 3. **Auto-Refresh** 🔄

- Dashboard refresh otomatis setiap **5 detik**
- Live data dari InfluxDB

## 🚀 Cara Menggunakan

### 1. Pastikan semua container berjalan:

```bash
docker-compose ps
```

Output harus menunjukkan:

```
NAME        STATUS
grafana     Up
influxdb    Up
webserver   Up
```

### 2. Akses Analytics Dashboard

Buka browser dan navigasi ke:

```
http://localhost:5000/analytics
```

### 3. Menu Navigasi

- **Dashboard**: Home page
- **CCTV**: Live stream video
- **Analytics**: Grafana embedded dashboard (NEW! ⭐)

## 📊 Query Data dari InfluxDB

Dashboard menggunakan **Flux Query Language**. Contoh queries:

### Total Kendaraan per Kelas

```flux
from(bucket: "vehicle_counting")
  |> range(start: -6h)
  |> filter(fn: (r) => r._measurement == "vehicle_counts")
  |> filter(fn: (r) => r._field == "count")
  |> filter(fn: (r) => r.class == "car")
  |> last()
```

### Crossing Events per Jam

```flux
from(bucket: "vehicle_counting")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "crossing_events")
  |> filter(fn: (r) => r._field == "event")
  |> aggregateWindow(every: 1h, fn: sum)
```

## 🔧 Konfigurasi

### Docker Compose - Grafana Settings

```yaml
environment:
  - GF_AUTH_ANONYMOUS_ENABLED=true # Enable anonymous access
  - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer # Read-only access
  - GF_SECURITY_ALLOW_EMBEDDING=true # Allow iframe embed
  - GF_SECURITY_COOKIE_SAMESITE=none # Allow cross-origin
```

### Provisioning Files

```
grafana-provisioning/
├── datasources/
│   └── influxdb.yml          # Auto-setup InfluxDB datasource
└── dashboards/
    ├── dashboard.yml         # Dashboard provider config
    └── vehicle-traffic.json  # Dashboard definition
```

## 🎨 Customization

### Mengubah Time Range Default

Edit `vehicle-traffic.json`:

```json
"time": {
  "from": "now-6h",  // Ubah sesuai kebutuhan (now-1h, now-24h, dll)
  "to": "now"
}
```

### Mengubah Refresh Interval

Di URL iframe atau edit dashboard:

```
?refresh=5s   // 5 detik
?refresh=10s  // 10 detik
?refresh=1m   // 1 menit
```

### Menambah Panel Baru

1. Login ke Grafana direct: http://localhost:3030
   - Username: `admin`
   - Password: `admin123`
2. Edit dashboard "Vehicle Traffic Dashboard"
3. Tambah panel dengan query Flux
4. Save dashboard

## 🐛 Troubleshooting

### Dashboard tidak muncul

```bash
# Restart Grafana
docker-compose restart grafana

# Check logs
docker logs grafana --tail 50
```

### Data tidak muncul di grafik

```bash
# Cek data di InfluxDB
docker exec -it influxdb influx query 'from(bucket:"vehicle_counting") |> range(start: -1h)'

# Atau cek apakah main.py sedang running dan logging data
```

### "No data" di panel

Pastikan:

1. ✅ Main detection script (`src/main.py`) sedang running
2. ✅ InfluxDB menerima data (cek logs)
3. ✅ Time range di dashboard sesuai (contoh: last 6 hours)

## 📖 Dokumentasi Teknis

### Iframe Embed URL Parameters

```
http://localhost:3030/d/vehicle-traffic-dashboard/vehicle-traffic-dashboard
  ?orgId=1              # Organization ID
  &refresh=5s           # Auto refresh
  &kiosk=tv             # Kiosk mode (hide controls)
```

### Kiosk Modes

- `&kiosk=tv`: Hide semua controls (cleanest)
- `&kiosk`: Hide menu bar only
- No kiosk: Full Grafana UI

## 🎯 Next Steps (Optional Enhancement)

### 1. Tambah Alerting

- Setup alert rules di Grafana
- Notifikasi via email/webhook jika traffic anomali

### 2. Advanced Analytics

- Heatmap untuk peak hours
- Forecasting dengan trendlines
- Comparative analysis (today vs yesterday)

### 3. Export & Reporting

- PDF export dari dashboard
- Scheduled reports via email
- API untuk external integration

## 📚 Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [InfluxDB Flux Query](https://docs.influxdata.com/flux/)
- [Flask Documentation](https://flask.palletsprojects.com/)

---

## ✅ Quick Verification Checklist

- [ ] Container grafana running
- [ ] Container influxdb running
- [ ] Container webserver running
- [ ] http://localhost:5000 accessible
- [ ] http://localhost:5000/analytics menampilkan dashboard
- [ ] Dashboard auto-refresh setiap 5 detik
- [ ] Data muncul di grafik (jika ada data)

**Status Implementasi**: ✅ **COMPLETE**

Enjoy your analytics dashboard! 🎉📊
