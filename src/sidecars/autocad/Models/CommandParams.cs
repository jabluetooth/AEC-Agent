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

    // =========================================================================
    // Raster Design Parameters
    // =========================================================================

    public class ImportPdfParams
    {
        [JsonProperty("file_path")]
        public string FilePath { get; set; }

        [JsonProperty("page")]
        public int Page { get; set; } = 1;

        [JsonProperty("insertion_point")]
        public double[] InsertionPoint { get; set; }

        [JsonProperty("scale")]
        public double Scale { get; set; } = 1.0;

        [JsonProperty("rotation")]
        public double Rotation { get; set; }

        [JsonProperty("target_layer")]
        public string TargetLayer { get; set; }
    }

    public class AttachImageParams
    {
        [JsonProperty("file_path")]
        public string FilePath { get; set; }

        [JsonProperty("image_name")]
        public string ImageName { get; set; }

        [JsonProperty("insertion_point")]
        public double[] InsertionPoint { get; set; }

        [JsonProperty("scale")]
        public double Scale { get; set; } = 1.0;

        [JsonProperty("target_layer")]
        public string TargetLayer { get; set; }
    }

    public class RasterCleanupParams
    {
        [JsonProperty("operation")]
        public string Operation { get; set; } = "despeckle";

        [JsonProperty("blob_size")]
        public int BlobSize { get; set; } = 3;

        [JsonProperty("threshold_value")]
        public int ThresholdValue { get; set; } = 128;

        [JsonProperty("brightness")]
        public int Brightness { get; set; }

        [JsonProperty("contrast")]
        public int Contrast { get; set; }
    }

    public class RasterVectorizeParams
    {
        [JsonProperty("method")]
        public string Method { get; set; } = "auto";

        [JsonProperty("target_layer")]
        public string TargetLayer { get; set; }

        [JsonProperty("detect_polygons")]
        public bool DetectPolygons { get; set; } = true;

        [JsonProperty("detect_arcs")]
        public bool DetectArcs { get; set; } = true;

        [JsonProperty("gap_tolerance")]
        public double GapTolerance { get; set; } = 0.5;
    }

    public class RasterOcrParams
    {
        [JsonProperty("target_layer")]
        public string TargetLayer { get; set; }

        [JsonProperty("text_height")]
        public double TextHeight { get; set; }

        [JsonProperty("language")]
        public string Language { get; set; } = "eng";
    }

    public class GetEntityCountParams
    {
        [JsonProperty("layer")]
        public string Layer { get; set; }
    }

    public class ExtractAllEntitiesParams
    {
        [JsonProperty("offset")]
        public int Offset { get; set; }

        [JsonProperty("limit")]
        public int Limit { get; set; } = 500;

        [JsonProperty("layer_filter")]
        public string LayerFilter { get; set; }

        [JsonProperty("include_geometry")]
        public bool IncludeGeometry { get; set; } = true;

        [JsonProperty("include_xdata")]
        public bool IncludeXdata { get; set; }
    }

    public class FadeImageParams
    {
        [JsonProperty("fade_percent")]
        public int FadePercent { get; set; } = 70;
    }

    // =========================================================================
    // Raster Design — Image Processing, Primitives, VTools, Followers
    // =========================================================================

    /// <summary>
    /// Parameters for ibfilter (image processing filters for bitonal images).
    /// Filters: smooth, thin, thicken, separate, skeletonize.
    /// </summary>
    public class ProcessImageParams
    {
        [JsonProperty("filter_type")]
        public string FilterType { get; set; } = "skeletonize";
    }

    /// <summary>
    /// Parameters for REM primitive creation (isline, isarc, iscircle, issmart).
    /// Creates overlay primitives from detected raster entities.
    /// </summary>
    public class CreatePrimitiveParams
    {
        [JsonProperty("primitive_type")]
        public string PrimitiveType { get; set; } = "smart";

        [JsonProperty("point")]
        public double[] Point { get; set; }
    }

    /// <summary>
    /// Parameters for VTools vectorization (vline, vpline, varc, vcircle, vrect).
    /// Converts raster entities to native AutoCAD vector entities.
    /// </summary>
    public class VToolParams
    {
        [JsonProperty("tool")]
        public string Tool { get; set; } = "vpline";

        [JsonProperty("method")]
        public string Method { get; set; } = "1p";

        [JsonProperty("points")]
        public double[][] Points { get; set; }

        [JsonProperty("target_layer")]
        public string TargetLayer { get; set; }
    }

    /// <summary>
    /// Parameters for VTools followers (vfpline, vfcontour, vf3dpoly).
    /// Semi-automatic tracing of raster lines and contours.
    /// </summary>
    public class FollowerParams
    {
        [JsonProperty("follower_type")]
        public string FollowerType { get; set; } = "polyline";

        [JsonProperty("start_point")]
        public double[] StartPoint { get; set; }

        [JsonProperty("target_layer")]
        public string TargetLayer { get; set; }
    }

    /// <summary>
    /// Parameters for raster text recognition (irectext).
    /// </summary>
    public class RecognizeTextParams
    {
        [JsonProperty("target_layer")]
        public string TargetLayer { get; set; }
    }

    /// <summary>
    /// Parameters for selecting raster entities in a region (isebrcon, isebrsmart).
    /// </summary>
    public class SelectRasterEntitiesParams
    {
        [JsonProperty("method")]
        public string Method { get; set; } = "smart";

        [JsonProperty("corner1")]
        public double[] Corner1 { get; set; }

        [JsonProperty("corner2")]
        public double[] Corner2 { get; set; }
    }
}
