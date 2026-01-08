using System;
using System.Diagnostics;
using System.IO;

namespace AECAgent.AutoCAD.Utils
{
    public static class Logger
    {
        private static readonly object _lock = new object();
        private static readonly string LogDirectory;
        private static readonly string LogFilePath;
        private static bool _initialized;

        static Logger()
        {
            try
            {
                LogDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "AECAgent", "logs");
                if (!Directory.Exists(LogDirectory)) Directory.CreateDirectory(LogDirectory);
                LogFilePath = Path.Combine(LogDirectory, $"autocad-sidecar-{DateTime.Now:yyyy-MM-dd}.log");
                _initialized = true;
            }
            catch { _initialized = false; }
        }

        public static void Info(string message) => Log("INFO", message);
        public static void Warn(string message) => Log("WARN", message);
        public static void Error(string message, Exception ex = null) => Log("ERROR", ex != null ? $"{message}\n{ex}" : message);

        [Conditional("DEBUG")]
        public static void Debug(string message) => Log("DEBUG", message);

        private static void Log(string level, string message)
        {
            if (!_initialized) return;
            try
            {
                string entry = $"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] [{level}] {message}";
                System.Diagnostics.Debug.WriteLine(entry);
                lock (_lock) { File.AppendAllText(LogFilePath, entry + Environment.NewLine); }
            }
            catch { }
        }

        public static string GetLogFilePath() => LogFilePath;
    }
}
