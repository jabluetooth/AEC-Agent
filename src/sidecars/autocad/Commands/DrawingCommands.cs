using System;
using Autodesk.AutoCAD.ApplicationServices;
using Autodesk.AutoCAD.DatabaseServices;
using Autodesk.AutoCAD.Geometry;
using AECAgent.AutoCAD.Models;
using Newtonsoft.Json;

namespace AECAgent.AutoCAD.Commands
{
    public class DrawingCommands
    {
        public object DrawPolyline(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<DrawPolylineParams>(parameters);
            if (param.Points == null || param.Points.Length < 2)
                throw new ArgumentException("At least 2 points required");

            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);

            using (Polyline pline = new Polyline())
            {
                for (int i = 0; i < param.Points.Length; i++)
                {
                    double bulge = 0;
                    if (param.Bulges != null && i < param.Bulges.Length)
                        bulge = param.Bulges[i];
                    pline.AddVertexAt(i, new Point2d(param.Points[i][0], param.Points[i].Length > 1 ? param.Points[i][1] : 0), bulge, 0, 0);
                }

                pline.Closed = param.Closed;
                if (!string.IsNullOrEmpty(param.Layer)) SetLayer(pline, param.Layer, db, tr);
                if (param.Color.HasValue) pline.ColorIndex = param.Color.Value;

                ObjectId id = btr.AppendEntity(pline);
                tr.AddNewlyCreatedDBObject(pline, true);

                return new EntityCreatedResult { Handle = pline.Handle.ToString(), ObjectId = id.ToString(), Type = "Polyline", Layer = pline.Layer, Created = true };
            }
        }

        public object DrawLine(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<DrawLineParams>(parameters);
            if (param.Start == null || param.Start.Length < 2) 
                throw new ArgumentException($"Start point required. Received: {JsonConvert.SerializeObject(parameters)}");
            if (param.End == null || param.End.Length < 2) 
                throw new ArgumentException($"End point required. Received: {JsonConvert.SerializeObject(parameters)}");

            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);

            using (Line line = new Line(ToPoint3d(param.Start), ToPoint3d(param.End)))
            {
                if (!string.IsNullOrEmpty(param.Layer)) SetLayer(line, param.Layer, db, tr);
                ObjectId id = btr.AppendEntity(line);
                tr.AddNewlyCreatedDBObject(line, true);
                return new EntityCreatedResult { Handle = line.Handle.ToString(), ObjectId = id.ToString(), Type = "Line", Layer = line.Layer, Created = true };
            }
        }

        public object DrawCircle(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<DrawCircleParams>(parameters);
            if (param.Center == null || param.Center.Length < 2) throw new ArgumentException("Center required");
            if (param.Radius <= 0) throw new ArgumentException("Radius must be > 0");

            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);

            using (Circle circle = new Circle(ToPoint3d(param.Center), Vector3d.ZAxis, param.Radius))
            {
                if (!string.IsNullOrEmpty(param.Layer)) SetLayer(circle, param.Layer, db, tr);
                ObjectId id = btr.AppendEntity(circle);
                tr.AddNewlyCreatedDBObject(circle, true);
                return new EntityCreatedResult { Handle = circle.Handle.ToString(), ObjectId = id.ToString(), Type = "Circle", Layer = circle.Layer, Created = true };
            }
        }

        public object DrawArc(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<DrawArcParams>(parameters);
            if (param.Center == null || param.Center.Length < 2) throw new ArgumentException("Center required");
            if (param.Radius <= 0) throw new ArgumentException("Radius must be > 0");

            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);

            // AutoCAD Arc constructor expects angles in radians
            double startRad = param.StartAngle * (Math.PI / 180);
            double endRad = param.EndAngle * (Math.PI / 180);

            using (Arc arc = new Arc(ToPoint3d(param.Center), param.Radius, startRad, endRad))
            {
                if (!string.IsNullOrEmpty(param.Layer)) SetLayer(arc, param.Layer, db, tr);
                ObjectId id = btr.AppendEntity(arc);
                tr.AddNewlyCreatedDBObject(arc, true);
                return new EntityCreatedResult { Handle = arc.Handle.ToString(), ObjectId = id.ToString(), Type = "Arc", Layer = arc.Layer, Created = true };
            }
        }

        public object DrawEllipse(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<DrawEllipseParams>(parameters);
            if (param.Center == null || param.Center.Length < 2) throw new ArgumentException("Center required");
            if (param.MajorAxisEndpoint == null || param.MajorAxisEndpoint.Length < 2) throw new ArgumentException("MajorAxisEndpoint required");
            if (param.AxisRatio <= 0 || param.AxisRatio > 1) throw new ArgumentException("AxisRatio must be > 0 and <= 1");

            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);

            Point3d center = ToPoint3d(param.Center);
            Vector3d majorAxis = new Vector3d(
                param.MajorAxisEndpoint[0],
                param.MajorAxisEndpoint.Length > 1 ? param.MajorAxisEndpoint[1] : 0,
                param.MajorAxisEndpoint.Length > 2 ? param.MajorAxisEndpoint[2] : 0
            );

            // Convert angles: degrees to radians
            // For a full ellipse: start=0, end=2*PI
            double startRad = param.StartAngle * (Math.PI / 180);
            double endRad = param.EndAngle * (Math.PI / 180);
            // AutoCAD Ellipse uses 0..2PI for full, partial for arcs
            if (Math.Abs(param.EndAngle - param.StartAngle) >= 359.9)
            {
                startRad = 0;
                endRad = 2 * Math.PI;
            }

            using (Ellipse ellipse = new Ellipse(center, Vector3d.ZAxis, majorAxis, param.AxisRatio, startRad, endRad))
            {
                if (!string.IsNullOrEmpty(param.Layer)) SetLayer(ellipse, param.Layer, db, tr);
                ObjectId id = btr.AppendEntity(ellipse);
                tr.AddNewlyCreatedDBObject(ellipse, true);
                return new EntityCreatedResult { Handle = ellipse.Handle.ToString(), ObjectId = id.ToString(), Type = "Ellipse", Layer = ellipse.Layer, Created = true };
            }
        }

        public object DrawSpline(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<DrawSplineParams>(parameters);
            if (param.FitPoints == null || param.FitPoints.Length < 2)
                throw new ArgumentException("At least 2 fit points required");

            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);

            // Build Point3dCollection for fit points
            Point3dCollection fitPts = new Point3dCollection();
            foreach (var pt in param.FitPoints)
                fitPts.Add(ToPoint3d(pt));

            // For closed splines, duplicate the first point at the end to
            // create a loop.  Spline.Closed is read-only in the AutoCAD API.
            if (param.Closed && fitPts.Count >= 2)
            {
                Point3d first = fitPts[0];
                Point3d last = fitPts[fitPts.Count - 1];
                if (first.DistanceTo(last) > 1e-6)
                    fitPts.Add(first);
            }

            // Create spline through fit points (order 4 = cubic, tolerance 1e-6)
            using (Spline spline = new Spline(fitPts, 4, 1e-6))
            {
                if (!string.IsNullOrEmpty(param.Layer)) SetLayer(spline, param.Layer, db, tr);
                ObjectId id = btr.AppendEntity(spline);
                tr.AddNewlyCreatedDBObject(spline, true);
                return new EntityCreatedResult { Handle = spline.Handle.ToString(), ObjectId = id.ToString(), Type = "Spline", Layer = spline.Layer, Created = true };
            }
        }

        public object DrawRectangle(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<DrawRectangleParams>(parameters);
            if (param.Corner1 == null || param.Corner1.Length < 2) throw new ArgumentException("Corner1 required");
            if (param.Corner2 == null || param.Corner2.Length < 2) throw new ArgumentException("Corner2 required");

            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);

            double x1 = param.Corner1[0], y1 = param.Corner1[1];
            double x2 = param.Corner2[0], y2 = param.Corner2[1];

            using (Polyline rect = new Polyline())
            {
                rect.AddVertexAt(0, new Point2d(x1, y1), 0, 0, 0);
                rect.AddVertexAt(1, new Point2d(x2, y1), 0, 0, 0);
                rect.AddVertexAt(2, new Point2d(x2, y2), 0, 0, 0);
                rect.AddVertexAt(3, new Point2d(x1, y2), 0, 0, 0);
                rect.Closed = true;

                if (!string.IsNullOrEmpty(param.Layer)) SetLayer(rect, param.Layer, db, tr);
                ObjectId id = btr.AppendEntity(rect);
                tr.AddNewlyCreatedDBObject(rect, true);
                return new EntityCreatedResult { Handle = rect.Handle.ToString(), ObjectId = id.ToString(), Type = "Rectangle", Layer = rect.Layer, Created = true };
            }
        }

        public object DrawText(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<DrawTextParams>(parameters);
            if (string.IsNullOrEmpty(param.Text)) throw new ArgumentException("Text required");
            if (param.Position == null || param.Position.Length < 2) throw new ArgumentException("Position required");
            if (param.Height <= 0) param.Height = 2.5;

            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);

            using (DBText text = new DBText())
            {
                text.Position = ToPoint3d(param.Position);
                text.TextString = param.Text;
                text.Height = param.Height;
                text.Rotation = param.Rotation * (Math.PI / 180);

                if (!string.IsNullOrEmpty(param.Layer)) SetLayer(text, param.Layer, db, tr);
                ObjectId id = btr.AppendEntity(text);
                tr.AddNewlyCreatedDBObject(text, true);
                return new EntityCreatedResult { Handle = text.Handle.ToString(), ObjectId = id.ToString(), Type = "DBText", Layer = text.Layer, Created = true };
            }
        }

        public object ModifyEntity(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<ModifyEntityParams>(parameters);
            if (string.IsNullOrEmpty(param.Handle)) throw new ArgumentException("Handle required");

            Database db = doc.Database;
            Handle handle = new Handle(Convert.ToInt64(param.Handle, 16));
            if (!db.TryGetObjectId(handle, out ObjectId objId) || objId.IsNull)
                throw new ArgumentException($"Entity '{param.Handle}' not found");

            Entity entity = (Entity)tr.GetObject(objId, OpenMode.ForWrite);

            if (!string.IsNullOrEmpty(param.Layer)) SetLayer(entity, param.Layer, db, tr);
            if (param.Color.HasValue) entity.ColorIndex = param.Color.Value;

            if (param.Move != null && param.Move.Length >= 2)
                entity.TransformBy(Matrix3d.Displacement(new Vector3d(param.Move[0], param.Move[1], param.Move.Length > 2 ? param.Move[2] : 0)));

            if (param.Scale.HasValue && param.Scale.Value != 1.0)
                entity.TransformBy(Matrix3d.Scaling(param.Scale.Value, GetCenter(entity)));

            if (param.Rotate.HasValue && param.Rotate.Value != 0)
                entity.TransformBy(Matrix3d.Rotation(param.Rotate.Value * (Math.PI / 180), Vector3d.ZAxis, GetCenter(entity)));

            return new { handle = entity.Handle.ToString(), type = entity.GetType().Name, layer = entity.Layer, modified = true };
        }

        public object DeleteEntity(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<DeleteEntityParams>(parameters);
            if (string.IsNullOrEmpty(param.Handle)) throw new ArgumentException("Handle required");

            Database db = doc.Database;
            Handle handle = new Handle(Convert.ToInt64(param.Handle, 16));
            if (!db.TryGetObjectId(handle, out ObjectId objId) || objId.IsNull)
                throw new ArgumentException($"Entity '{param.Handle}' not found");

            Entity entity = (Entity)tr.GetObject(objId, OpenMode.ForWrite);
            string type = entity.GetType().Name;
            entity.Erase();

            return new { handle = param.Handle, type, deleted = true };
        }

        private T Deserialize<T>(object p) where T : new() => p == null ? new T() : JsonConvert.DeserializeObject<T>(JsonConvert.SerializeObject(p));
        private Point3d ToPoint3d(double[] a) => new Point3d(a.Length > 0 ? a[0] : 0, a.Length > 1 ? a[1] : 0, a.Length > 2 ? a[2] : 0);
        private void SetLayer(Entity e, string layer, Database db, Transaction tr)
        {
            LayerTable lt = (LayerTable)tr.GetObject(db.LayerTableId, OpenMode.ForRead);
            if (!lt.Has(layer)) throw new ArgumentException($"Layer '{layer}' not found");
            e.Layer = layer;
        }
        private Point3d GetCenter(Entity e)
        {
            try { var ext = e.GeometricExtents; return new Point3d((ext.MinPoint.X + ext.MaxPoint.X) / 2, (ext.MinPoint.Y + ext.MaxPoint.Y) / 2, (ext.MinPoint.Z + ext.MaxPoint.Z) / 2); }
            catch { return Point3d.Origin; }
        }
    }
}
