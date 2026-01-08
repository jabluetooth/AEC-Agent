using System;
using System.Collections.Generic;
using Autodesk.AutoCAD.ApplicationServices;
using Autodesk.AutoCAD.DatabaseServices;
using AECAgent.AutoCAD.Models;
using Newtonsoft.Json;

namespace AECAgent.AutoCAD.Commands
{
    public class QueryCommands
    {
        public object GetEntities(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<GetEntitiesParams>(parameters);
            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForRead);

            var entities = new List<EntityInfo>();
            int limit = param.Limit ?? 1000;

            foreach (ObjectId objId in btr)
            {
                if (entities.Count >= limit) break;
                DBObject obj = tr.GetObject(objId, OpenMode.ForRead);
                if (!(obj is Entity entity)) continue;

                if (!string.IsNullOrEmpty(param.Layer) && !entity.Layer.Equals(param.Layer, StringComparison.OrdinalIgnoreCase)) continue;
                if (!string.IsNullOrEmpty(param.EntityType) && !entity.GetType().Name.ToLower().Contains(param.EntityType.ToLower())) continue;

                entities.Add(new EntityInfo
                {
                    Handle = entity.Handle.ToString(),
                    ObjectId = objId.ToString(),
                    Type = entity.GetType().Name,
                    Layer = entity.Layer,
                    Color = entity.ColorIndex,
                    Bounds = GetBounds(entity)
                });
            }

            return new { entities, count = entities.Count, limited = entities.Count >= limit };
        }

        public object GetEntity(object parameters, Document doc, Transaction tr)
        {
            string handle = GetString(parameters, "handle");
            if (string.IsNullOrEmpty(handle)) throw new ArgumentException("Handle required");

            Database db = doc.Database;
            Handle h = new Handle(Convert.ToInt64(handle, 16));
            if (!db.TryGetObjectId(h, out ObjectId objId) || objId.IsNull)
                throw new ArgumentException($"Entity '{handle}' not found");

            Entity entity = (Entity)tr.GetObject(objId, OpenMode.ForRead);
            return new
            {
                handle = entity.Handle.ToString(),
                object_id = objId.ToString(),
                type = entity.GetType().Name,
                layer = entity.Layer,
                color = entity.ColorIndex,
                linetype = entity.Linetype,
                lineweight = entity.LineWeight.ToString(),
                bounds = GetBounds(entity)
            };
        }

        public object GetDrawingInfo(object parameters, Document doc, Transaction tr)
        {
            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForRead);

            int entityCount = 0;
            foreach (ObjectId id in btr) entityCount++;

            LayerTable lt = (LayerTable)tr.GetObject(db.LayerTableId, OpenMode.ForRead);
            int layerCount = 0;
            foreach (ObjectId id in lt) layerCount++;

            return new DrawingInfo
            {
                FileName = System.IO.Path.GetFileName(doc.Name),
                FilePath = doc.Name,
                Units = db.Insunits.ToString(),
                DwgVersion = db.OriginalFileVersion.ToString(),
                ExtentsMin = new PointInfo { X = db.Extmin.X, Y = db.Extmin.Y, Z = db.Extmin.Z },
                ExtentsMax = new PointInfo { X = db.Extmax.X, Y = db.Extmax.Y, Z = db.Extmax.Z },
                EntityCount = entityCount,
                LayerCount = layerCount
            };
        }

        public object AuditLayers(object parameters, Document doc, Transaction tr)
        {
            Database db = doc.Database;
            LayerTable lt = (LayerTable)tr.GetObject(db.LayerTableId, OpenMode.ForRead);
            var layers = new List<LayerInfo>();

            foreach (ObjectId layerId in lt)
            {
                LayerTableRecord layer = (LayerTableRecord)tr.GetObject(layerId, OpenMode.ForRead);
                layers.Add(new LayerInfo
                {
                    Name = layer.Name,
                    IsFrozen = layer.IsFrozen,
                    IsLocked = layer.IsLocked,
                    IsOff = layer.IsOff,
                    Color = layer.Color.ColorIndex,
                    Lineweight = layer.LineWeight.ToString()
                });
            }

            return new { layers, count = layers.Count };
        }

        private T Deserialize<T>(object p) where T : new() => p == null ? new T() : JsonConvert.DeserializeObject<T>(JsonConvert.SerializeObject(p));
        private string GetString(object p, string key)
        {
            if (p == null) return null;
            var dict = JsonConvert.DeserializeObject<Dictionary<string, object>>(JsonConvert.SerializeObject(p));
            return dict != null && dict.TryGetValue(key, out object v) ? v?.ToString() : null;
        }
        private BoundsInfo GetBounds(Entity e)
        {
            try
            {
                var ext = e.GeometricExtents;
                return new BoundsInfo { MinX = ext.MinPoint.X, MinY = ext.MinPoint.Y, MinZ = ext.MinPoint.Z, MaxX = ext.MaxPoint.X, MaxY = ext.MaxPoint.Y, MaxZ = ext.MaxPoint.Z };
            }
            catch { return null; }
        }
    }
}
