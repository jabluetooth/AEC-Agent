using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.AutoCAD.ApplicationServices;
using Autodesk.AutoCAD.DatabaseServices;
using Autodesk.AutoCAD.Geometry;
using AECAgent.AutoCAD.Models;
using Newtonsoft.Json;

namespace AECAgent.AutoCAD.Commands
{
    /// <summary>
    /// Extraction commands for metadata pipeline.
    /// Extracts entity geometry and properties for PostgreSQL storage.
    /// </summary>
    public class ExtractionCommands
    {
        /// <summary>
        /// Extract entities in batches with full geometry information.
        /// </summary>
        public object ExtractBatch(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<ExtractBatchParams>(parameters);
            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForRead);

            int offset = param.Offset ?? 0;
            int limit = param.Limit ?? 1000;
            bool includeGeometry = param.IncludeGeometry ?? true;
            bool includeXData = param.IncludeXData ?? true;

            var entities = new List<ExtractedEntity>();
            int index = 0;
            int total = 0;

            // Count total and filter
            foreach (ObjectId objId in btr)
            {
                if (objId.IsErased) continue;
                total++;
            }

            foreach (ObjectId objId in btr)
            {
                if (objId.IsErased) continue;

                // Skip until offset
                if (index < offset)
                {
                    index++;
                    continue;
                }

                // Stop at limit
                if (entities.Count >= limit) break;

                DBObject obj = tr.GetObject(objId, OpenMode.ForRead);
                if (!(obj is Entity entity)) continue;

                // Apply layer filter if specified
                if (!string.IsNullOrEmpty(param.LayerFilter) &&
                    !entity.Layer.Equals(param.LayerFilter, StringComparison.OrdinalIgnoreCase))
                {
                    index++;
                    continue;
                }

                var extracted = new ExtractedEntity
                {
                    Handle = entity.Handle.ToString(),
                    ObjectId = objId.ToString(),
                    EntityType = entity.GetType().Name,
                    Layer = entity.Layer,
                    Color = entity.ColorIndex,
                    Linetype = entity.Linetype,
                    Bounds = GetBounds(entity)
                };

                if (includeGeometry)
                {
                    extracted.Geometry = ExtractGeometry(entity, tr);
                }

                if (includeXData)
                {
                    extracted.XData = ExtractXData(entity);
                }

                // Extract additional properties
                extracted.Properties = ExtractProperties(entity, tr);

                entities.Add(extracted);
                index++;
            }

            return new
            {
                entities,
                count = entities.Count,
                total,
                offset,
                has_more = (offset + entities.Count) < total
            };
        }

        /// <summary>
        /// Extract geometry for a specific entity type.
        /// </summary>
        private GeometryInfo ExtractGeometry(Entity entity, Transaction tr)
        {
            var info = new GeometryInfo();

            try
            {
                switch (entity)
                {
                    case Line line:
                        info.Type = "LINE";
                        info.Points = new List<double[]>
                        {
                            new[] { line.StartPoint.X, line.StartPoint.Y, line.StartPoint.Z },
                            new[] { line.EndPoint.X, line.EndPoint.Y, line.EndPoint.Z }
                        };
                        break;

                    case Circle circle:
                        info.Type = "CIRCLE";
                        info.Center = new[] { circle.Center.X, circle.Center.Y, circle.Center.Z };
                        info.Radius = circle.Radius;
                        break;

                    case Arc arc:
                        info.Type = "ARC";
                        info.Center = new[] { arc.Center.X, arc.Center.Y, arc.Center.Z };
                        info.Radius = arc.Radius;
                        info.StartAngle = arc.StartAngle * 180 / Math.PI;
                        info.EndAngle = arc.EndAngle * 180 / Math.PI;
                        break;

                    case Polyline pline:
                        info.Type = "POLYLINE";
                        info.Points = new List<double[]>();
                        for (int i = 0; i < pline.NumberOfVertices; i++)
                        {
                            Point3d pt = pline.GetPoint3dAt(i);
                            info.Points.Add(new[] { pt.X, pt.Y, pt.Z });
                        }
                        info.Closed = pline.Closed;
                        break;

                    case Polyline2d pline2d:
                        info.Type = "POLYLINE2D";
                        info.Points = new List<double[]>();
                        foreach (ObjectId vtxId in pline2d)
                        {
                            Vertex2d vtx = (Vertex2d)tr.GetObject(vtxId, OpenMode.ForRead);
                            info.Points.Add(new[] { vtx.Position.X, vtx.Position.Y, vtx.Position.Z });
                        }
                        info.Closed = pline2d.Closed;
                        break;

                    case Polyline3d pline3d:
                        info.Type = "POLYLINE3D";
                        info.Points = new List<double[]>();
                        foreach (ObjectId vtxId in pline3d)
                        {
                            PolylineVertex3d vtx = (PolylineVertex3d)tr.GetObject(vtxId, OpenMode.ForRead);
                            info.Points.Add(new[] { vtx.Position.X, vtx.Position.Y, vtx.Position.Z });
                        }
                        info.Closed = pline3d.Closed;
                        break;

                    case DBPoint point:
                        info.Type = "POINT";
                        info.Points = new List<double[]>
                        {
                            new[] { point.Position.X, point.Position.Y, point.Position.Z }
                        };
                        break;

                    case Ellipse ellipse:
                        info.Type = "ELLIPSE";
                        info.Center = new[] { ellipse.Center.X, ellipse.Center.Y, ellipse.Center.Z };
                        info.Radius = ellipse.MajorRadius;
                        info.MinorRadius = ellipse.MinorRadius;
                        break;

                    case Spline spline:
                        info.Type = "SPLINE";
                        info.Points = new List<double[]>();
                        for (int i = 0; i < spline.NumControlPoints; i++)
                        {
                            Point3d pt = spline.GetControlPointAt(i);
                            info.Points.Add(new[] { pt.X, pt.Y, pt.Z });
                        }
                        info.Closed = spline.Closed;
                        break;

                    case BlockReference blockRef:
                        info.Type = "INSERT";
                        info.Position = new[] { blockRef.Position.X, blockRef.Position.Y, blockRef.Position.Z };
                        info.Rotation = blockRef.Rotation * 180 / Math.PI;
                        info.Scale = new[] { blockRef.ScaleFactors.X, blockRef.ScaleFactors.Y, blockRef.ScaleFactors.Z };
                        // Get block name
                        BlockTableRecord blockDef = (BlockTableRecord)tr.GetObject(blockRef.BlockTableRecord, OpenMode.ForRead);
                        info.BlockName = blockDef.Name;
                        break;

                    case DBText text:
                        info.Type = "TEXT";
                        info.Position = new[] { text.Position.X, text.Position.Y, text.Position.Z };
                        info.TextContent = text.TextString;
                        info.Height = text.Height;
                        info.Rotation = text.Rotation * 180 / Math.PI;
                        break;

                    case MText mtext:
                        info.Type = "MTEXT";
                        info.Position = new[] { mtext.Location.X, mtext.Location.Y, mtext.Location.Z };
                        info.TextContent = mtext.Contents;
                        info.Height = mtext.TextHeight;
                        break;

                    case Hatch hatch:
                        info.Type = "HATCH";
                        info.PatternName = hatch.PatternName;
                        // Get first loop boundary if exists
                        if (hatch.NumberOfLoops > 0)
                        {
                            info.Loops = new List<List<double[]>>();
                            for (int i = 0; i < hatch.NumberOfLoops; i++)
                            {
                                var loop = new List<double[]>();
                                HatchLoop hatchLoop = hatch.GetLoopAt(i);
                                if (hatchLoop.IsPolyline)
                                {
                                    foreach (BulgeVertex bv in hatchLoop.Polyline)
                                    {
                                        loop.Add(new[] { bv.Vertex.X, bv.Vertex.Y, 0.0 });
                                    }
                                }
                                info.Loops.Add(loop);
                            }
                        }
                        break;

                    case Dimension dim:
                        info.Type = "DIMENSION";
                        info.Position = new[] { dim.TextPosition.X, dim.TextPosition.Y, dim.TextPosition.Z };
                        info.TextContent = dim.DimensionText;
                        break;

                    default:
                        // For unknown types, just store the type name
                        info.Type = entity.GetType().Name.ToUpper();
                        break;
                }
            }
            catch (Exception ex)
            {
                info.Error = ex.Message;
            }

            return info;
        }

        /// <summary>
        /// Extract XData (Extended Data) from an entity.
        /// </summary>
        private Dictionary<string, object> ExtractXData(Entity entity)
        {
            var xdata = new Dictionary<string, object>();

            try
            {
                ResultBuffer rb = entity.XData;
                if (rb == null) return xdata;

                string currentApp = null;
                var currentValues = new List<object>();

                foreach (TypedValue tv in rb)
                {
                    // 1001 = Application name
                    if (tv.TypeCode == 1001)
                    {
                        // Save previous app's data
                        if (currentApp != null && currentValues.Count > 0)
                        {
                            xdata[currentApp] = currentValues.ToArray();
                        }
                        currentApp = tv.Value.ToString();
                        currentValues = new List<object>();
                    }
                    else if (currentApp != null)
                    {
                        currentValues.Add(tv.Value);
                    }
                }

                // Save last app's data
                if (currentApp != null && currentValues.Count > 0)
                {
                    xdata[currentApp] = currentValues.ToArray();
                }
            }
            catch
            {
                // Ignore XData extraction errors
            }

            return xdata;
        }

        /// <summary>
        /// Extract additional properties based on entity type.
        /// </summary>
        private Dictionary<string, object> ExtractProperties(Entity entity, Transaction tr)
        {
            var props = new Dictionary<string, object>();

            try
            {
                switch (entity)
                {
                    case Line line:
                        props["length"] = line.Length;
                        props["delta_x"] = line.EndPoint.X - line.StartPoint.X;
                        props["delta_y"] = line.EndPoint.Y - line.StartPoint.Y;
                        break;

                    case Circle circle:
                        props["area"] = circle.Area;
                        props["circumference"] = 2 * Math.PI * circle.Radius;
                        break;

                    case Arc arc:
                        props["length"] = arc.Length;
                        props["area"] = arc.Area;
                        break;

                    case Polyline pline:
                        props["length"] = pline.Length;
                        props["area"] = pline.Closed ? pline.Area : 0;
                        props["vertex_count"] = pline.NumberOfVertices;
                        break;

                    case BlockReference blockRef:
                        BlockTableRecord blockDef = (BlockTableRecord)tr.GetObject(blockRef.BlockTableRecord, OpenMode.ForRead);
                        props["block_name"] = blockDef.Name;
                        props["is_dynamic"] = blockRef.IsDynamicBlock;

                        // Extract block attributes
                        if (blockRef.AttributeCollection.Count > 0)
                        {
                            var attrs = new Dictionary<string, string>();
                            foreach (ObjectId attId in blockRef.AttributeCollection)
                            {
                                AttributeReference att = (AttributeReference)tr.GetObject(attId, OpenMode.ForRead);
                                attrs[att.Tag] = att.TextString;
                            }
                            props["attributes"] = attrs;
                        }
                        break;

                    case DBText text:
                        props["text_string"] = text.TextString;
                        props["style"] = text.TextStyleName;
                        break;

                    case MText mtext:
                        props["text_string"] = mtext.Text;
                        props["width"] = mtext.Width;
                        break;

                    case Hatch hatch:
                        props["pattern_name"] = hatch.PatternName;
                        props["pattern_scale"] = hatch.PatternScale;
                        props["pattern_angle"] = hatch.PatternAngle * 180 / Math.PI;
                        break;
                }

                // Common properties
                props["visible"] = entity.Visible;
            }
            catch
            {
                // Ignore property extraction errors
            }

            return props;
        }

        private BoundsInfo GetBounds(Entity entity)
        {
            try
            {
                var ext = entity.GeometricExtents;
                return new BoundsInfo
                {
                    MinX = ext.MinPoint.X,
                    MinY = ext.MinPoint.Y,
                    MinZ = ext.MinPoint.Z,
                    MaxX = ext.MaxPoint.X,
                    MaxY = ext.MaxPoint.Y,
                    MaxZ = ext.MaxPoint.Z
                };
            }
            catch
            {
                return null;
            }
        }

        private T Deserialize<T>(object p) where T : new() =>
            p == null ? new T() : JsonConvert.DeserializeObject<T>(JsonConvert.SerializeObject(p));
    }

    #region Parameter and Result Models

    public class ExtractBatchParams
    {
        public int? Offset { get; set; }
        public int? Limit { get; set; }
        public bool? IncludeGeometry { get; set; }
        public bool? IncludeXData { get; set; }
        public string LayerFilter { get; set; }
    }

    public class ExtractedEntity
    {
        public string Handle { get; set; }
        public string ObjectId { get; set; }
        public string EntityType { get; set; }
        public string Layer { get; set; }
        public int Color { get; set; }
        public string Linetype { get; set; }
        public GeometryInfo Geometry { get; set; }
        public BoundsInfo Bounds { get; set; }
        public Dictionary<string, object> Properties { get; set; }
        public Dictionary<string, object> XData { get; set; }
    }

    public class GeometryInfo
    {
        public string Type { get; set; }
        public List<double[]> Points { get; set; }
        public double[] Center { get; set; }
        public double Radius { get; set; }
        public double MinorRadius { get; set; }
        public double StartAngle { get; set; }
        public double EndAngle { get; set; }
        public bool Closed { get; set; }
        public double[] Position { get; set; }
        public double Rotation { get; set; }
        public double[] Scale { get; set; }
        public string BlockName { get; set; }
        public string TextContent { get; set; }
        public double Height { get; set; }
        public string PatternName { get; set; }
        public List<List<double[]>> Loops { get; set; }
        public string Error { get; set; }
    }

    #endregion
}
