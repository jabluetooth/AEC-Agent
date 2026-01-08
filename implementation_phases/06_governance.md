# Phase 6: Governance & Economic Optimization

**Objective:** Minimize operational costs, ensure observability, and handle heavy workloads with proper governance.

## 1. Hybrid Batch Processing
*   **Scenario:** User asks, "Export PDF sheets for these 500 drawing files."
*   **Don't:** Run this on the local RDP session. It locks the UI for hours.
*   **Do:** Route this specific tool call to **Autodesk Platform Services (APS)** or a headless **AcCoreConsole** worker pool.
    *   Upload the file to OSS Bucket.
    *   Trigger a Design Automation WorkItem.
    *   Notify user when the download link is ready.

## 2. Licensing Optimization
*   **Audit:** Analyze `Journal` files to see usage frequency.
*   **Strategy:**
    *   **Power Users (>3 days/week):** Annual Subscription.
    *   **Light Users (<1 day/week):** Switch to **Autodesk Flex Tokens**.
    *   *Savings:* This typically reduces software spend by 20-30% in large firms.

## 3. Centralized Logging

### Logging Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Chainlit   │────>│   Fluent    │────>│ CloudWatch  │
│  Frontend   │     │    Bit      │     │    Logs     │
└─────────────┘     └─────────────┘     └─────────────┘
       │                  ^                    │
       v                  │                    v
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│     MCP     │────>│   Local     │     │  Log        │
│ Middleware  │     │   Agent     │     │  Insights   │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       v
┌─────────────┐
│  Sidecars   │
│ (CAD/Revit) │
└─────────────┘
```

### Log Format Standard (JSON)
```json
{
    "timestamp": "2024-01-15T10:30:00.123Z",
    "level": "INFO",
    "component": "mcp-middleware",
    "session_id": "abc-123",
    "user": "jsmith",
    "request_id": "req-456",
    "message": "Tool invocation completed",
    "tool": "draw_wall",
    "duration_ms": 1250,
    "success": true
}
```

### Python Logging Configuration
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
            "session_id": getattr(record, 'session_id', None),
            "request_id": getattr(record, 'request_id', None),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

# Configure logger
handler = logging.FileHandler("logs/mcp_middleware.jsonl")
handler.setFormatter(JSONFormatter())
logging.getLogger("aec_agent").addHandler(handler)
```

### Log Retention Policy
| Log Type | Retention | Storage |
|----------|-----------|---------|
| Application logs | 30 days | CloudWatch |
| Security/Audit logs | 1 year | S3 Glacier |
| Performance metrics | 90 days | CloudWatch |
| Error traces | 90 days | CloudWatch |

## 4. Monitoring & Metrics

### Key Performance Indicators (KPIs)

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Tool success rate | > 99% | < 95% |
| Sidecar response time (p95) | < 2s | > 5s |
| SQLite cache hit rate | > 90% | < 70% |
| Active user sessions | - | > 55 (capacity) |
| Memory per session | < 500MB | > 800MB |

### CloudWatch Dashboard Widgets

```yaml
# cloudwatch-dashboard.yml
widgets:
  - type: metric
    title: "Tool Invocation Success Rate"
    metrics:
      - [ "AECAgent", "ToolSuccess", { "stat": "Average", "period": 300 } ]
      - [ "AECAgent", "ToolFailure", { "stat": "Sum", "period": 300 } ]

  - type: metric
    title: "Sidecar Latency (p95)"
    metrics:
      - [ "AECAgent", "SidecarLatency", { "stat": "p95", "period": 60 } ]

  - type: metric
    title: "Active Sessions"
    metrics:
      - [ "AECAgent", "ActiveSessions", { "stat": "Maximum", "period": 60 } ]

  - type: log
    title: "Error Log Stream"
    logGroupName: "/aec-agent/errors"
    query: "fields @timestamp, @message | filter level = 'ERROR' | sort @timestamp desc | limit 50"
```

### Custom Metrics Collection (Python)
```python
import boto3
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

def publish_metric(name: str, value: float, unit: str = "Count"):
    cloudwatch.put_metric_data(
        Namespace='AECAgent',
        MetricData=[{
            'MetricName': name,
            'Value': value,
            'Unit': unit,
            'Timestamp': datetime.utcnow(),
            'Dimensions': [
                {'Name': 'Environment', 'Value': 'production'},
            ]
        }]
    )

# Usage
publish_metric("ToolSuccess", 1)
publish_metric("SidecarLatency", 1250, "Milliseconds")
```

## 5. Alerting Rules

### Critical Alerts (PagerDuty/SNS)
```yaml
alerts:
  - name: "Sidecar Unreachable"
    condition: "HealthCheck.Failed > 3 in 5 minutes"
    severity: critical
    action: page_on_call

  - name: "High Error Rate"
    condition: "ToolFailure / ToolTotal > 0.10 for 10 minutes"
    severity: critical
    action: page_on_call

  - name: "Instance Memory Critical"
    condition: "MemoryUsedPercent > 90"
    severity: critical
    action: page_on_call
```

### Warning Alerts (Email/Slack)
```yaml
alerts:
  - name: "Elevated Latency"
    condition: "SidecarLatency.p95 > 3000ms for 15 minutes"
    severity: warning
    action: notify_slack

  - name: "Session Capacity Warning"
    condition: "ActiveSessions > 45"
    severity: warning
    action: notify_email

  - name: "Cache Miss Rate High"
    condition: "CacheMissRate > 0.30 for 30 minutes"
    severity: warning
    action: notify_slack
```

### CloudWatch Alarm Configuration
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_alarm(
    AlarmName='AEC-HighErrorRate',
    MetricName='ToolFailure',
    Namespace='AECAgent',
    Statistic='Sum',
    Period=300,
    EvaluationPeriods=2,
    Threshold=10,
    ComparisonOperator='GreaterThanThreshold',
    AlarmActions=['arn:aws:sns:us-east-1:123456789:aec-alerts'],
    AlarmDescription='High tool failure rate detected'
)
```

## 6. Incident Response

### Severity Levels
| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| SEV-1 | System down | 15 min | All sidecars unreachable |
| SEV-2 | Major degradation | 30 min | >50% error rate |
| SEV-3 | Minor issue | 4 hours | Single user affected |
| SEV-4 | Low priority | 24 hours | UI cosmetic issue |

### Runbook: Sidecar Unreachable

1. **Verify scope:** Check if issue affects single user or all users
2. **Check process:** `Get-Process | Where-Object {$_.MainWindowTitle -like "*AutoCAD*"}`
3. **Check port binding:** `netstat -ano | findstr ":{PORT}"`
4. **Review logs:** Check `%LOCALAPPDATA%\AECAgent\logs\sidecar.log`
5. **Restart sidecar:** Kill and restart the CAD application
6. **Escalate:** If unresolved after 15 min, escalate to Tier 2

### Runbook: High Memory Usage

1. **Identify session:** Check CloudWatch for affected instance
2. **Check for memory leaks:** Review session duration vs memory growth
3. **Clear cache:** Delete SQLite cache files older than 24h
4. **Force garbage collection:** Trigger CAD application GC
5. **User notification:** Warn user before forced session restart
6. **Restart session:** If >95% memory, force logout and restart

## 7. Audit Trail

### Security Events to Log
| Event | Log Level | Retention |
|-------|-----------|-----------|
| User login | INFO | 1 year |
| Tool invocation | INFO | 90 days |
| Token validation failure | WARN | 1 year |
| Unauthorized access attempt | ERROR | 1 year |
| Configuration change | INFO | 1 year |

### Audit Log Format
```json
{
    "event_type": "TOOL_INVOCATION",
    "timestamp": "2024-01-15T10:30:00Z",
    "user": "jsmith",
    "session_id": "abc-123",
    "client_ip": "10.0.1.50",
    "tool": "draw_wall",
    "parameters": {"start": [0,0], "end": [10,0]},
    "result": "success",
    "duration_ms": 1250
}
```

### Compliance Considerations
*   **Data Residency:** Ensure logs stay in approved regions
*   **PII Handling:** Mask or exclude personal data from logs
*   **Access Control:** Restrict log access to authorized personnel
*   **Retention:** Implement automated log lifecycle policies
