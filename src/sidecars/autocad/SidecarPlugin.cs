using System;
using System.Collections.Concurrent;
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
        private static bool _isInitialized;
        private static CommandRouter _commandRouter;

        public static ConcurrentQueue<JobRequest> JobQueue => _jobQueue;
        public static string SessionToken => _sessionToken;
        public static CommandRouter CommandRouter => _commandRouter;

        public void Initialize()
        {
            try
            {
                _sessionToken = Environment.GetEnvironmentVariable("SESSION_TOKEN");
                string portStr = Environment.GetEnvironmentVariable("MCP_LISTENER_PORT");

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

                _jobQueue = new ConcurrentQueue<JobRequest>();
                _commandRouter = new CommandRouter();

                Application.Idle += OnIdle;

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
                _listener?.Stop();
                _listener = null;
                while (_jobQueue?.TryDequeue(out _) == true) { }
                Logger.Info("AEC Agent Sidecar terminated");
            }
            catch (System.Exception ex)
            {
                Logger.Error($"Error during termination: {ex.Message}", ex);
            }
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
