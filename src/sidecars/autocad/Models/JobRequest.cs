using System;
using System.Threading;
using Newtonsoft.Json;

namespace AECAgent.AutoCAD.Models
{
    public class JobRequest
    {
        private readonly ManualResetEventSlim _completionEvent;
        private object _result;
        private string _error;
        private bool _isCompleted;

        [JsonProperty("command")]
        public string Command { get; set; }

        [JsonProperty("parameters")]
        public object Parameters { get; set; }

        [JsonProperty("request_id")]
        public string RequestId { get; set; }

        [JsonIgnore]
        public object Result => _result;

        [JsonIgnore]
        public string Error => _error;

        [JsonIgnore]
        public bool IsCompleted => _isCompleted;

        [JsonIgnore]
        public ManualResetEventSlim CompletionEvent => _completionEvent;

        [JsonIgnore]
        public DateTime ReceivedAt { get; set; }

        public JobRequest()
        {
            _completionEvent = new ManualResetEventSlim(false);
            RequestId = Guid.NewGuid().ToString("N").Substring(0, 8);
            ReceivedAt = DateTime.UtcNow;
        }

        public void SetResult(object result)
        {
            _result = result;
            _isCompleted = true;
            _completionEvent.Set();
        }

        public void SetError(string error)
        {
            _error = error;
            _isCompleted = true;
            _completionEvent.Set();
        }

        public bool WaitForCompletion(TimeSpan timeout)
        {
            return _completionEvent.Wait(timeout);
        }

        public void Dispose()
        {
            _completionEvent?.Dispose();
        }
    }
}
