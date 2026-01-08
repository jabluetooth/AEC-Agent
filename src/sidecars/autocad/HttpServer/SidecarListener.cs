using System;
using System.Collections.Concurrent;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using AECAgent.AutoCAD.Models;
using AECAgent.AutoCAD.Utils;

namespace AECAgent.AutoCAD.HttpServer
{
    public class SidecarListener
    {
        private readonly int _port;
        private readonly string _sessionToken;
        private readonly ConcurrentQueue<JobRequest> _jobQueue;
        private readonly RequestHandler _requestHandler;
        private HttpListener _listener;
        private CancellationTokenSource _cts;
        private bool _isRunning;

        public bool IsRunning => _isRunning;

        public SidecarListener(int port, string sessionToken, ConcurrentQueue<JobRequest> jobQueue)
        {
            _port = port;
            _sessionToken = sessionToken;
            _jobQueue = jobQueue;
            _requestHandler = new RequestHandler(sessionToken, jobQueue);
        }

        public async Task StartAsync()
        {
            if (_isRunning) return;

            try
            {
                _cts = new CancellationTokenSource();
                _listener = new HttpListener();
                _listener.Prefixes.Add($"http://127.0.0.1:{_port}/");
                _listener.Start();
                _isRunning = true;

                Logger.Info($"HTTP listener started on http://127.0.0.1:{_port}/");

                while (_isRunning && !_cts.Token.IsCancellationRequested)
                {
                    try
                    {
                        var context = await _listener.GetContextAsync();
                        _ = Task.Run(() => HandleRequestSafe(context), _cts.Token);
                    }
                    catch (HttpListenerException) { break; }
                    catch (ObjectDisposedException) { break; }
                    catch (Exception ex)
                    {
                        Logger.Error($"Listener error: {ex.Message}", ex);
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.Error($"Failed to start HTTP listener: {ex.Message}", ex);
                _isRunning = false;
                throw;
            }
        }

        private async Task HandleRequestSafe(HttpListenerContext context)
        {
            try
            {
                await _requestHandler.HandleAsync(context);
            }
            catch (Exception ex)
            {
                Logger.Error($"Request error: {ex.Message}", ex);
                try { context.Response.StatusCode = 500; context.Response.Close(); } catch { }
            }
        }

        public void Stop()
        {
            if (!_isRunning) return;
            _isRunning = false;

            try
            {
                _cts?.Cancel();
                _listener?.Stop();
                _listener?.Close();
                _listener = null;
                Logger.Info("HTTP listener stopped");
            }
            catch (Exception ex)
            {
                Logger.Error($"Error stopping listener: {ex.Message}", ex);
            }
            finally
            {
                _cts?.Dispose();
                _cts = null;
            }
        }
    }
}
