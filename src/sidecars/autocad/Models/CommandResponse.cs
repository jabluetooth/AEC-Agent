using Newtonsoft.Json;

namespace AECAgent.AutoCAD.Models
{
    public class CommandResponse
    {
        [JsonProperty("success")]
        public bool Success { get; set; }

        [JsonProperty("data")]
        public object Data { get; set; }

        [JsonProperty("error")]
        public ErrorInfo Error { get; set; }

        public static CommandResponse Ok(object data) => new CommandResponse { Success = true, Data = data };

        public static CommandResponse Fail(int code, string message, string details = null) =>
            new CommandResponse
            {
                Success = false,
                Error = new ErrorInfo { Code = code, Message = message, Details = details }
            };
    }

    public class ErrorInfo
    {
        [JsonProperty("code")]
        public int Code { get; set; }

        [JsonProperty("message")]
        public string Message { get; set; }

        [JsonProperty("details")]
        public string Details { get; set; }
    }

    public class EntityCreatedResult
    {
        [JsonProperty("handle")]
        public string Handle { get; set; }

        [JsonProperty("object_id")]
        public string ObjectId { get; set; }

        [JsonProperty("type")]
        public string Type { get; set; }

        [JsonProperty("layer")]
        public string Layer { get; set; }

        [JsonProperty("created")]
        public bool Created { get; set; }
    }

    public class LayerCreatedResult
    {
        [JsonProperty("name")]
        public string Name { get; set; }

        [JsonProperty("handle")]
        public string Handle { get; set; }

        [JsonProperty("color")]
        public int Color { get; set; }

        [JsonProperty("created")]
        public bool Created { get; set; }
    }

    public class LayerInfo
    {
        [JsonProperty("name")]
        public string Name { get; set; }

        [JsonProperty("is_frozen")]
        public bool IsFrozen { get; set; }

        [JsonProperty("is_locked")]
        public bool IsLocked { get; set; }

        [JsonProperty("is_off")]
        public bool IsOff { get; set; }

        [JsonProperty("color")]
        public int Color { get; set; }

        [JsonProperty("lineweight")]
        public string Lineweight { get; set; }

        [JsonProperty("linetype")]
        public string Linetype { get; set; }
    }

    public class EntityInfo
    {
        [JsonProperty("handle")]
        public string Handle { get; set; }

        [JsonProperty("object_id")]
        public string ObjectId { get; set; }

        [JsonProperty("type")]
        public string Type { get; set; }

        [JsonProperty("layer")]
        public string Layer { get; set; }

        [JsonProperty("color")]
        public int Color { get; set; }

        [JsonProperty("bounds")]
        public BoundsInfo Bounds { get; set; }
    }

    public class BoundsInfo
    {
        [JsonProperty("min_x")]
        public double MinX { get; set; }

        [JsonProperty("min_y")]
        public double MinY { get; set; }

        [JsonProperty("min_z")]
        public double MinZ { get; set; }

        [JsonProperty("max_x")]
        public double MaxX { get; set; }

        [JsonProperty("max_y")]
        public double MaxY { get; set; }

        [JsonProperty("max_z")]
        public double MaxZ { get; set; }
    }

    public class DrawingInfo
    {
        [JsonProperty("file_name")]
        public string FileName { get; set; }

        [JsonProperty("file_path")]
        public string FilePath { get; set; }

        [JsonProperty("is_modified")]
        public bool IsModified { get; set; }

        [JsonProperty("units")]
        public string Units { get; set; }

        [JsonProperty("dwg_version")]
        public string DwgVersion { get; set; }

        [JsonProperty("extents_min")]
        public PointInfo ExtentsMin { get; set; }

        [JsonProperty("extents_max")]
        public PointInfo ExtentsMax { get; set; }

        [JsonProperty("entity_count")]
        public int EntityCount { get; set; }

        [JsonProperty("layer_count")]
        public int LayerCount { get; set; }
    }

    public class PointInfo
    {
        [JsonProperty("x")]
        public double X { get; set; }

        [JsonProperty("y")]
        public double Y { get; set; }

        [JsonProperty("z")]
        public double Z { get; set; }
    }
}
