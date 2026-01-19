using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using AECAgent.AutoCAD.Models;
using AECAgent.AutoCAD.Utils;
using Newtonsoft.Json;

namespace AECAgent.AutoCAD.HttpServer
{
    public class RequestHandler
    {
        private readonly string _sessionToken;
        private readonly ConcurrentQueue<JobRequest> _jobQueue;
        private static readonly TimeSpan JobTimeout = TimeSpan.FromSeconds(120);

        public RequestHandler(string sessionToken, ConcurrentQueue<JobRequest> jobQueue)
        {
            _sessionToken = sessionToken;
            _jobQueue = jobQueue;
        }

        public async Task HandleAsync(HttpListenerContext context)
        {
            var request = context.Request;
            var response = context.Response;

            try
            {
                AddCorsHeaders(response);

                if (request.HttpMethod == "OPTIONS")
                {
                    response.StatusCode = 200;
                    response.Close();
                    return;
                }

                if (request.Url.AbsolutePath == "/health")
                {
                    await SendJsonAsync(response, 200, new { status = "healthy", timestamp = DateTime.UtcNow });
                    return;
                }

                if (request.Url.AbsolutePath == "/metrics")
                {
                    if (!ValidateToken(request, response)) return;
                    await SendJsonAsync(response, 200, MetricsCollector.GetMetrics());
                    return;
                }

                if (!ValidateToken(request, response)) return;

                if (request.HttpMethod != "POST")
                {
                    await SendErrorAsync(response, 405, "Method Not Allowed", "Use POST");
                    return;
                }

                string body;
                using (var reader = new StreamReader(request.InputStream, request.ContentEncoding))
                    body = await reader.ReadToEndAsync();

                if (string.IsNullOrWhiteSpace(body))
                {
                    await SendErrorAsync(response, 400, "Bad Request", "Empty body");
                    return;
                }

                JobRequest job;
                try { job = JsonConvert.DeserializeObject<JobRequest>(body); }
                catch (JsonException ex)
                {
                    await SendErrorAsync(response, 400, "Bad Request", $"Invalid JSON: {ex.Message}");
                    return;
                }

                if (string.IsNullOrWhiteSpace(job.Command))
                {
                    await SendErrorAsync(response, 400, "Bad Request", "Missing 'command'");
                    return;
                }

                _jobQueue.Enqueue(job);

                bool completed = await Task.Run(() => job.WaitForCompletion(JobTimeout));

                if (!completed)
                {
                    await SendErrorAsync(response, 504, "Gateway Timeout", "CAD operation timed out");
                    return;
                }

                if (job.Error != null)
                {
                    // Use 400 for client/validation errors, 500 for internal errors
                    int statusCode = IsClientError(job.Error) ? 400 : 500;
                    await SendErrorAsync(response, statusCode, "CAD Error", job.Error);
                }
                else
                    await SendJsonAsync(response, 200, CommandResponse.Ok(job.Result));

                job.Dispose();
            }
            catch (Exception ex)
            {
                Logger.Error($"Request error: {ex.Message}", ex);
                await SendErrorAsync(response, 500, "Internal Error", ex.Message);
            }
        }

        private bool ValidateToken(HttpListenerRequest request, HttpListenerResponse response)
        {
            string auth = request.Headers["Authorization"];
            if (string.IsNullOrEmpty(auth))
            {
                SendError(response, 401, "Unauthorized", "Missing Authorization");
                return false;
            }

            string token = auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) ? auth.Substring(7) : auth;
            if (token != _sessionToken)
            {
                SendError(response, 401, "Unauthorized", "Invalid token");
                return false;
            }
            return true;
        }

        private void AddCorsHeaders(HttpListenerResponse response)
        {
            response.AddHeader("Access-Control-Allow-Origin", "*");
            response.AddHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            response.AddHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
        }

        private async Task SendJsonAsync(HttpListenerResponse response, int status, object data)
        {
            response.StatusCode = status;
            response.ContentType = "application/json";
            string json = JsonConvert.SerializeObject(data, new JsonSerializerSettings { NullValueHandling = NullValueHandling.Ignore });
            byte[] buffer = Encoding.UTF8.GetBytes(json);
            response.ContentLength64 = buffer.Length;
            await response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
            response.Close();
        }

        private async Task SendErrorAsync(HttpListenerResponse response, int status, string message, string details)
        {
            await SendJsonAsync(response, status, CommandResponse.Fail(status, message, details));
        }

        private void SendError(HttpListenerResponse response, int status, string message, string details)
        {
            response.StatusCode = status;
            response.ContentType = "application/json";
            string json = JsonConvert.SerializeObject(CommandResponse.Fail(status, message, details));
            byte[] buffer = Encoding.UTF8.GetBytes(json);
            response.ContentLength64 = buffer.Length;
            response.OutputStream.Write(buffer, 0, buffer.Length);
            response.Close();
        }

        /// <summary>
        /// Determines if an error message indicates a client error (400) vs server error (500).
        /// Client errors are validation failures, not found, already exists, etc.
        /// </summary>
        private static bool IsClientError(string error)
        {
            if (string.IsNullOrEmpty(error)) return false;

            string lower = error.ToLowerInvariant();
            return lower.Contains("already exists") ||
                   lower.Contains("not found") ||
                   lower.Contains("required") ||
                   lower.Contains("invalid") ||
                   lower.Contains("cannot") ||
                   lower.Contains("must be") ||
                   lower.StartsWith("error:"); // ArgumentException messages
        }
    }
}
