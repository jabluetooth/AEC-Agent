using Newtonsoft.Json;

namespace AECAgent.AutoCAD.Models
{
    public class DrawPolylineParams
    {
        [JsonProperty("points")]
        public double[][] Points { get; set; }

        [JsonProperty("closed")]
        public bool Closed { get; set; }

        [JsonProperty("layer")]
        public string Layer { get; set; }

        [JsonProperty("color")]
        public int? Color { get; set; }
    }

    public class DrawLineParams
    {
        [JsonProperty("start")]
        public double[] Start { get; set; }

        [JsonProperty("end")]
        public double[] End { get; set; }

        [JsonProperty("layer")]
        public string Layer { get; set; }
    }

    public class DrawCircleParams
    {
        [JsonProperty("center")]
        public double[] Center { get; set; }

        [JsonProperty("radius")]
        public double Radius { get; set; }

        [JsonProperty("layer")]
        public string Layer { get; set; }
    }

    public class DrawRectangleParams
    {
        [JsonProperty("corner1")]
        public double[] Corner1 { get; set; }

        [JsonProperty("corner2")]
        public double[] Corner2 { get; set; }

        [JsonProperty("layer")]
        public string Layer { get; set; }
    }

    public class DrawTextParams
    {
        [JsonProperty("text")]
        public string Text { get; set; }

        [JsonProperty("position")]
        public double[] Position { get; set; }

        [JsonProperty("height")]
        public double Height { get; set; }

        [JsonProperty("rotation")]
        public double Rotation { get; set; }

        [JsonProperty("layer")]
        public string Layer { get; set; }
    }

    public class CreateLayerParams
    {
        [JsonProperty("name")]
        public string Name { get; set; }

        [JsonProperty("color")]
        public int? Color { get; set; }

        [JsonProperty("linetype")]
        public string Linetype { get; set; }

        [JsonProperty("lineweight")]
        public int? Lineweight { get; set; }
    }

    public class ModifyLayerParams
    {
        [JsonProperty("name")]
        public string Name { get; set; }

        [JsonProperty("color")]
        public int? Color { get; set; }

        [JsonProperty("is_off")]
        public bool? IsOff { get; set; }

        [JsonProperty("is_frozen")]
        public bool? IsFrozen { get; set; }

        [JsonProperty("is_locked")]
        public bool? IsLocked { get; set; }
    }

    public class GetEntitiesParams
    {
        [JsonProperty("layer")]
        public string Layer { get; set; }

        [JsonProperty("entity_type")]
        public string EntityType { get; set; }

        [JsonProperty("limit")]
        public int? Limit { get; set; }

        [JsonProperty("include_geometry")]
        public bool IncludeGeometry { get; set; }
    }

    public class ModifyEntityParams
    {
        [JsonProperty("handle")]
        public string Handle { get; set; }

        [JsonProperty("layer")]
        public string Layer { get; set; }

        [JsonProperty("color")]
        public int? Color { get; set; }

        [JsonProperty("move")]
        public double[] Move { get; set; }

        [JsonProperty("scale")]
        public double? Scale { get; set; }

        [JsonProperty("rotate")]
        public double? Rotate { get; set; }
    }

    public class DeleteEntityParams
    {
        [JsonProperty("handle")]
        public string Handle { get; set; }
    }

    public class DeleteLayerParams
    {
        [JsonProperty("name")]
        public string Name { get; set; }
    }
}
