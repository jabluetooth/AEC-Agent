using System;

namespace AECAgent.AutoCAD.Utils
{
    public static class SecurityValidator
    {
        private static DateTime _tokenExpiry;
        private static string _cachedToken;
        private static readonly TimeSpan TokenCacheDuration = TimeSpan.FromMinutes(5);

        public static bool ValidateToken(string token)
        {
            if (string.IsNullOrEmpty(token)) return false;
            if (DateTime.UtcNow > _tokenExpiry || _cachedToken == null)
            {
                _cachedToken = Environment.GetEnvironmentVariable("SESSION_TOKEN");
                _tokenExpiry = DateTime.UtcNow.Add(TokenCacheDuration);
            }
            return token == _cachedToken;
        }

        public static bool ValidatePort(int port) => port >= 20000 && port <= 30000;

        public static string SanitizeLayerName(string name)
        {
            if (string.IsNullOrEmpty(name)) return name;
            return name.Replace("<", "").Replace(">", "").Replace("/", "").Replace("\\", "")
                .Replace("\"", "").Replace(":", "").Replace(";", "").Replace("?", "")
                .Replace("*", "").Replace("|", "").Replace("=", "").Replace("`", "").Trim();
        }

        public static bool IsValidHandle(string handle)
        {
            if (string.IsNullOrEmpty(handle)) return false;
            try { Convert.ToInt64(handle, 16); return true; }
            catch { return false; }
        }
    }
}
