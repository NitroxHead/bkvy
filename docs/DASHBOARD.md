# bkvy Dashboard Documentation

## Overview

The bkvy dashboard provides a browser-based interface for monitoring and analyzing LLM routing statistics, transaction logs, and system performance in real-time.

## Features

### 📊 Visualizations
- **Overview Cards**: Total requests, success rate, total cost, average response time
- **Time-Series Charts**: Requests over time with success/failure breakdown
- **Distribution Charts**:
  - Provider usage (doughnut chart)
  - Intelligence levels (bar chart)
  - Routing methods (pie chart)
- **Transaction Table**: Recent 100 transactions with full details
- **Error Analysis**: Error type breakdown and recent failures

### 🔄 Interactive Features
- **Time Range Selection**: 1h, 6h, 24h, 3 days, 7 days
- **Auto-refresh**: 15s, 30s, 1m, 5m intervals
- **Manual Refresh**: On-demand data updates
- **Responsive Design**: Works on desktop and mobile devices

## Setup

### 1. Enable Required Services

The dashboard requires transaction logging to be enabled:

```bash
export DASHBOARD_ENABLED=true
export TRANSACTION_LOGGING=true
```

### 2. Configure IP Access Control

**Option A: Local Access Only (Default)**
```bash
export DASHBOARD_ALLOWED_IPS="127.0.0.1"
```

**Option B: Internal Network**
```bash
export DASHBOARD_ALLOWED_IPS="10.0.0.0/24,127.0.0.1"
```

**Option C: Specific IPs**
```bash
export DASHBOARD_ALLOWED_IPS="127.0.0.1,192.168.1.100,192.168.1.200"
```

**Option D: Wildcard Patterns**
```bash
export DASHBOARD_ALLOWED_IPS="10.0.0.*,192.168.1.*"
```

**Option E: Allow All (Not Recommended)**
```bash
export DASHBOARD_ALLOWED_IPS="all"
```

### 3. Start the Server

```bash
python main.py
```

### 4. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:10006/dashboard
```

## IP Access Control

### Supported Formats

| Format | Example | Description |
|--------|---------|-------------|
| Specific IP | `127.0.0.1` | Single IP address |
| CIDR Range | `10.0.0.0/24` | IP range in CIDR notation |
| Wildcard | `10.0.0.*` | Wildcard pattern matching |
| Multiple | `127.0.0.1,10.0.0.*` | Comma-separated list |
| Allow All | `all` or `*` | Unrestricted access |

### Behind Reverse Proxy

If running behind a reverse proxy (nginx, Apache, etc.), the dashboard checks the `X-Forwarded-For` header to determine the client IP.

**Example nginx configuration:**
```nginx
location /dashboard {
    proxy_pass http://localhost:10006;
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_set_header Host $host;
}
```

## API Endpoints

### GET /dashboard
Serves the main dashboard HTML page.

**Access**: IP-restricted
**Returns**: HTML page

### GET /dashboard/data
Returns dashboard data as JSON.

**Access**: IP-restricted
**Parameters**:
- `hours` (optional): Time range in hours (1-720, default: 24)

**Example**:
```bash
curl http://localhost:10006/dashboard/data?hours=48
```

**Response**:
```json
{
  "overview": {
    "total_requests": 1247,
    "successful_requests": 1198,
    "success_rate": 96.07,
    "total_cost": 12.456789,
    "avg_response_time_ms": 1250
  },
  "time_series": [...],
  "distributions": {...},
  "recent_transactions": [...],
  "errors": {...}
}
```

### GET /dashboard/health
Returns dashboard health status.

**Access**: IP-restricted
**Returns**: Dashboard and log file status

## Testing

### Generate Sample Data

A test script is provided to generate sample transaction data:

```bash
python3 test_dashboard_sample_data.py
```

This creates 500 sample transactions over the last 48 hours in `logs/transactions.csv`.

## Security Best Practices

### Development
```bash
# Local access only
export DASHBOARD_ENABLED=true
export DASHBOARD_ALLOWED_IPS="127.0.0.1"
```

### Production
1. **Use HTTPS**: Run behind reverse proxy with SSL/TLS
2. **Restrict IPs**: Whitelist only known IP ranges
3. **Firewall Rules**: Add additional network-level restrictions
4. **VPN Access**: Consider requiring VPN for dashboard access
5. **Monitor Access**: Check logs for unauthorized access attempts

**Example production configuration:**
```bash
export DASHBOARD_ENABLED=true
export DASHBOARD_ALLOWED_IPS="10.0.0.0/24"  # Internal network only

# Behind nginx with SSL
# Only accessible via HTTPS on internal network
```

## Troubleshooting

### Dashboard shows "No data"
**Cause**: Transaction logging is disabled or no transactions exist
**Solution**:
```bash
export TRANSACTION_LOGGING=true
# Restart the server and make some LLM requests
```

### "Access Denied" error
**Cause**: Your IP is not in the whitelist
**Solution**: Check your IP and update `DASHBOARD_ALLOWED_IPS`
```bash
# Check your IP
curl ifconfig.me

# Update whitelist
export DASHBOARD_ALLOWED_IPS="127.0.0.1,YOUR_IP_HERE"
```

### Dashboard not loading
**Cause**: Dashboard is disabled
**Solution**:
```bash
export DASHBOARD_ENABLED=true
# Restart the server
```

### Charts not updating
**Cause**: Auto-refresh is disabled or time range is too narrow
**Solution**:
- Enable auto-refresh in the dashboard UI
- Increase time range to capture more data
- Click "Refresh Now" button manually

## Architecture

```
┌─────────────────┐
│   Browser       │
│  (Dashboard UI) │
└────────┬────────┘
         │ HTTP GET /dashboard
         │ HTTP GET /dashboard/data
         ▼
┌─────────────────┐
│ IP Whitelist    │
│   Middleware    │
└────────┬────────┘
         │ IP Check Pass
         ▼
┌─────────────────┐
│   Dashboard     │
│   Processor     │
└────────┬────────┘
         │ Read CSV
         ▼
┌─────────────────┐
│ Transaction Log │
│ (transactions.  │
│      csv)       │
└─────────────────┘
```

## Performance

- **Data Processing**: CSV parsing optimized for large files
- **Memory Usage**: Loads only data within time range
- **Response Time**: Typically < 100ms for 24-hour range
- **File Size**: ~200 bytes per transaction record

**Estimated data sizes:**
- 1,000 transactions: ~200 KB
- 10,000 transactions: ~2 MB
- 100,000 transactions: ~20 MB

## Future Enhancements

Planned features:
- [ ] Export data to CSV/JSON
- [ ] Custom date range picker
- [ ] Provider/model filtering
- [ ] Cost trend predictions
- [ ] Alert configuration
- [ ] Webhook notifications
- [ ] Multi-user authentication

## Support

For issues or questions:
- GitHub Issues: https://github.com/NitroxHead/bkvy/issues
- Documentation: https://github.com/NitroxHead/bkvy/tree/main/docs
