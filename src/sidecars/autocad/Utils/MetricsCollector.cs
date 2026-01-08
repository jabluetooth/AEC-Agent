using System;
using System.Collections.Concurrent;
using System.Linq;

namespace AECAgent.AutoCAD.Utils
{
    public static class MetricsCollector
    {
        private static readonly ConcurrentDictionary<string, CommandMetrics> _metrics = new ConcurrentDictionary<string, CommandMetrics>(StringComparer.OrdinalIgnoreCase);
        private static DateTime _startTime = DateTime.UtcNow;
        private static long _totalRequests;
        private static long _failedRequests;

        public static void RecordExecution(string command, TimeSpan duration, bool success)
        {
            System.Threading.Interlocked.Increment(ref _totalRequests);
            if (!success) System.Threading.Interlocked.Increment(ref _failedRequests);
            _metrics.GetOrAdd(command, _ => new CommandMetrics()).Record(duration, success);
        }

        public static object GetMetrics()
        {
            var uptime = DateTime.UtcNow - _startTime;
            return new
            {
                uptime_seconds = uptime.TotalSeconds,
                total_requests = _totalRequests,
                failed_requests = _failedRequests,
                success_rate = _totalRequests > 0 ? (double)(_totalRequests - _failedRequests) / _totalRequests : 1.0,
                commands = _metrics.ToDictionary(k => k.Key, v => new { total = v.Value.TotalCalls, success = v.Value.SuccessCalls, avg_ms = v.Value.AverageDurationMs })
            };
        }
    }

    internal class CommandMetrics
    {
        private readonly object _lock = new object();
        private long _totalCalls, _successCalls;
        private double _totalDurationMs;

        public long TotalCalls => _totalCalls;
        public long SuccessCalls => _successCalls;
        public double AverageDurationMs { get { lock (_lock) { return _totalCalls > 0 ? _totalDurationMs / _totalCalls : 0; } } }

        public void Record(TimeSpan duration, bool success)
        {
            lock (_lock)
            {
                _totalCalls++;
                if (success) _successCalls++;
                _totalDurationMs += duration.TotalMilliseconds;
            }
        }
    }
}
