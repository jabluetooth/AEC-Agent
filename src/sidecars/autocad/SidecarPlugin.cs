using System;
using System.Collections.Concurrent;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Autodesk.AutoCAD.ApplicationServices;
using Autodesk.AutoCAD.Runtime;
using AECAgent.AutoCAD.HttpServer;
using AECAgent.AutoCAD.Models;
using AECAgent.AutoCAD.Commands;
using AECAgent.AutoCAD.Utils;

[assembly: ExtensionApplication(typeof(AECAgent.AutoCAD.SidecarPlugin))]

namespace AECAgent.AutoCAD
{
    public class SidecarPlugin : IExtensionApplication
    {
        private static SidecarListener _listener;
        private static ConcurrentQueue<JobRequest> _jobQueue;
        private static string _sessionToken;
        private static int _listenerPort;
        private static int _mcpServerPort;
        private static bool _isInitialized;
        private static bool _enablePostgresSync;
        private static CommandRouter _commandRouter;
        private static HttpClient _httpClient;

        public static ConcurrentQueue<JobRequest> JobQueue => _jobQueue;
        public static string SessionToken => _sessionToken;
        public static CommandRouter CommandRouter => _commandRouter;

        public void Initialize()
        {
            try
            {
                _sessionToken = Environment.GetEnvironmentVariable("SESSION_TOKEN");
                string portStr = Environment.GetEnvironmentVariable("MCP_LISTENER_PORT");
                string mcpServerPortStr = Environment.GetEnvironmentVariable("MCP_SERVER_PORT") ?? "54321";
                string enableSyncStr = Environment.GetEnvironmentVariable("ENABLE_POSTGRES_SYNC") ?? "true";

                if (string.IsNullOrEmpty(_sessionToken) || string.IsNullOrEmpty(portStr))
                {
                    Logger.Error("SESSION_TOKEN or MCP_LISTENER_PORT not set. Plugin disabled.");
                    WriteToCommandLine("[AEC Agent] Plugin disabled - missing environment variables");
                    return;
                }

                if (!int.TryParse(portStr, out _listenerPort) || _listenerPort < 1024 || _listenerPort > 65535)
                {
                    Logger.Error($"Invalid MCP_LISTENER_PORT: {portStr}");
                    return;
                }

                // MCP server configuration for PostgreSQL sync
                int.TryParse(mcpServerPortStr, out _mcpServerPort);
                if (_mcpServerPort == 0) _mcpServerPort = 54321;
                _enablePostgresSync = enableSyncStr.ToLowerInvariant() == "true";

                _jobQueue = new ConcurrentQueue<JobRequest>();
                _commandRouter = new CommandRouter();
                _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };

                Application.Idle += OnIdle;

                // Document events for PostgreSQL sync
                Application.DocumentManager.DocumentCreated += OnDocumentCreated;
                Application.DocumentManager.DocumentBecameCurrent += OnDocumentBecameCurrent;
                Application.DocumentManager.DocumentToBeDestroyed += OnDocumentClosing;

                _listener = new SidecarListener(_listenerPort, _sessionToken, _jobQueue);
                Task.Run(() => _listener.StartAsync());

                _isInitialized = true;

                Logger.Info($"AEC Agent Sidecar initialized on port {_listenerPort}");
                WriteToCommandLine($"[AEC Agent] Sidecar started on port {_listenerPort}");
            }
            catch (System.Exception ex)
            {
                Logger.Error($"Failed to initialize sidecar: {ex.Message}", ex);
                WriteToCommandLine($"[AEC Agent ERROR] {ex.Message}");
            }
        }

        public void Terminate()
        {
            try
            {
                _isInitialized = false;
                Application.Idle -= OnIdle;

                // Unregister document events
                Application.DocumentManager.DocumentCreated -= OnDocumentCreated;
                Application.DocumentManager.DocumentBecameCurrent -= OnDocumentBecameCurrent;
                Application.DocumentManager.DocumentToBeDestroyed -= OnDocumentClosing;

                _listener?.Stop();
                _listener = null;
                _httpClient?.Dispose();
                _httpClient = null;
                while (_jobQueue?.TryDequeue(out _) == true) { }
                Logger.Info("AEC Agent Sidecar terminated");
            }
            catch (System.Exception ex)
            {
                Logger.Error($"Error during termination: {ex.Message}", ex);
            }
        }

        private static void OnDocumentCreated(object sender, DocumentCollectionEventArgs e)
        {
            if (!_isInitialized || e.Document == null) return;

            var doc = e.Document;
            Logger.Info($"Document created/opened: {doc.Name}");
            NotifyPostgresSync(doc.Name, "autocad", false);
        }

        private static void OnDocumentBecameCurrent(object sender, DocumentCollectionEventArgs e)
        {
            if (!_isInitialized || e.Document == null) return;

            var doc = e.Document;
            Logger.Info($"Document became current: {doc.Name}");
            // Only notify on document switch (for caching purposes)
            // NotifyPostgresSync(doc.Name, "autocad", false);
        }

        private static void OnDocumentClosing(object sender, DocumentCollectionEventArgs e)
        {
            if (!_isInitialized || e.Document == null) return;

            Logger.Info($"Document closing: {e.Document.Name}");
        }

        private static void NotifyPostgresSync(string filePath, string source, bool force)
        {
            if (!_enablePostgresSync || _httpClient == null) return;

            Task.Run(async () =>
            {
                try
                {
                    var payload = $"{{\"source\":\"{source}\",\"file_path\":\"{filePath.Replace("\\", "\\\\")}\",\"force_sync\":{force.ToString().ToLowerInvariant()}}}";
                    var content = new StringContent(payload, Encoding.UTF8, "application/json");

                    var response = await _httpClient.PostAsync(
                        $"http://localhost:{_mcpServerPort}/tools/notify_file_opened",
                        content
                    );

                    if (response.IsSuccessStatusCode)
                    {
                        Logger.Info("PostgreSQL sync notification sent successfully");
                    }
                    else
                    {
                        Logger.Warn($"PostgreSQL sync notification failed: {response.StatusCode}");
                    }
                }
                catch (System.Exception ex)
                {
                    // Don't fail on notification errors - PostgreSQL sync is optional
                    Logger.Debug($"PostgreSQL sync notification skipped: {ex.Message}");
                }
            });
        }

        private static void OnIdle(object sender, EventArgs e)
        {
            if (!_isInitialized || _jobQueue == null)
                return;

            if (!_jobQueue.TryDequeue(out JobRequest job))
                return;

            var startTime = DateTime.UtcNow;

            try
            {
                Document doc = Application.DocumentManager.MdiActiveDocument;
                if (doc == null)
                {
                    job.SetError("No active document");
                    return;
                }

                // Async commands use SendStringToExecute and run outside a Transaction.
                // They only need a DocumentLock, not a Transaction.
                if (_commandRouter.IsAsyncCommand(job.Command))
                {
                    using (var docLock = doc.LockDocument())
                    {
                        try
                        {
                            var result = _commandRouter.Execute(job.Command, job.Parameters, doc, null);
                            job.SetResult(result);
                            MetricsCollector.RecordExecution(job.Command, DateTime.UtcNow - startTime, true);
                        }
                        catch (System.Exception ex)
                        {
                            job.SetError($"Error: {ex.Message}");
                            MetricsCollector.RecordExecution(job.Command, DateTime.UtcNow - startTime, false);
                        }
                    }
                }
                else
                {
                    // Standard transactional commands
                    using (var docLock = doc.LockDocument())
                    {
                        using (var tr = doc.TransactionManager.StartTransaction())
                        {
                            try
                            {
                                var result = _commandRouter.Execute(job.Command, job.Parameters, doc, tr);
                                tr.Commit();
                                job.SetResult(result);
                                MetricsCollector.RecordExecution(job.Command, DateTime.UtcNow - startTime, true);
                            }
                            catch (Autodesk.AutoCAD.Runtime.Exception acEx)
                            {
                                tr.Abort();
                                job.SetError($"AutoCAD error {acEx.ErrorStatus}: {acEx.Message}");
                                MetricsCollector.RecordExecution(job.Command, DateTime.UtcNow - startTime, false);
                            }
                            catch (System.Exception ex)
                            {
                                tr.Abort();
                                job.SetError($"Error: {ex.Message}");
                                MetricsCollector.RecordExecution(job.Command, DateTime.UtcNow - startTime, false);
                            }
                        }
                    }
                }
            }
            catch (System.Exception ex)
            {
                job.SetError($"Document lock error: {ex.Message}");
            }
        }

        private static void WriteToCommandLine(string message)
        {
            try
            {
                Application.DocumentManager.MdiActiveDocument?.Editor.WriteMessage($"\n{message}");
            }
            catch { }
        }
    }
}
