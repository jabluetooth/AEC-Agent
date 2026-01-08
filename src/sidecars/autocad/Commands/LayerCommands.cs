using System;
using System.Collections.Generic;
using Autodesk.AutoCAD.ApplicationServices;
using Autodesk.AutoCAD.Colors;
using Autodesk.AutoCAD.DatabaseServices;
using AECAgent.AutoCAD.Models;
using Newtonsoft.Json;

namespace AECAgent.AutoCAD.Commands
{
    public class LayerCommands
    {
        public object CreateLayer(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<CreateLayerParams>(parameters);
            if (string.IsNullOrEmpty(param.Name)) throw new ArgumentException("Layer name required");

            Database db = doc.Database;
            LayerTable lt = (LayerTable)tr.GetObject(db.LayerTableId, OpenMode.ForWrite);

            if (lt.Has(param.Name)) throw new ArgumentException($"Layer '{param.Name}' already exists");

            using (LayerTableRecord ltr = new LayerTableRecord())
            {
                ltr.Name = param.Name;
                if (param.Color.HasValue) ltr.Color = Color.FromColorIndex(ColorMethod.ByAci, (short)param.Color.Value);
                if (param.Lineweight.HasValue) ltr.LineWeight = (LineWeight)param.Lineweight.Value;

                ObjectId id = lt.Add(ltr);
                tr.AddNewlyCreatedDBObject(ltr, true);

                return new LayerCreatedResult { Name = ltr.Name, Handle = ltr.Handle.ToString(), Color = ltr.Color.ColorIndex, Created = true };
            }
        }

        public object ModifyLayer(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<ModifyLayerParams>(parameters);
            if (string.IsNullOrEmpty(param.Name)) throw new ArgumentException("Layer name required");

            Database db = doc.Database;
            LayerTable lt = (LayerTable)tr.GetObject(db.LayerTableId, OpenMode.ForRead);
            if (!lt.Has(param.Name)) throw new ArgumentException($"Layer '{param.Name}' not found");

            LayerTableRecord ltr = (LayerTableRecord)tr.GetObject(lt[param.Name], OpenMode.ForWrite);

            if (param.Color.HasValue) ltr.Color = Color.FromColorIndex(ColorMethod.ByAci, (short)param.Color.Value);
            if (param.IsOff.HasValue) ltr.IsOff = param.IsOff.Value;
            if (param.IsFrozen.HasValue)
            {
                if (param.IsFrozen.Value && db.Clayer == ltr.ObjectId) throw new ArgumentException("Cannot freeze current layer");
                ltr.IsFrozen = param.IsFrozen.Value;
            }
            if (param.IsLocked.HasValue) ltr.IsLocked = param.IsLocked.Value;

            return new { name = ltr.Name, handle = ltr.Handle.ToString(), color = ltr.Color.ColorIndex, is_off = ltr.IsOff, is_frozen = ltr.IsFrozen, is_locked = ltr.IsLocked, modified = true };
        }

        public object DeleteLayer(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<DeleteLayerParams>(parameters);
            if (string.IsNullOrEmpty(param.Name)) throw new ArgumentException("Layer name required");
            if (param.Name == "0") throw new ArgumentException("Cannot delete layer '0'");

            Database db = doc.Database;
            LayerTable lt = (LayerTable)tr.GetObject(db.LayerTableId, OpenMode.ForRead);
            if (!lt.Has(param.Name)) throw new ArgumentException($"Layer '{param.Name}' not found");

            ObjectId layerId = lt[param.Name];
            if (layerId == db.Clayer) throw new ArgumentException("Cannot delete current layer");

            LayerTableRecord ltr = (LayerTableRecord)tr.GetObject(layerId, OpenMode.ForWrite);
            ltr.Erase();

            return new { name = param.Name, deleted = true };
        }

        public object SetCurrentLayer(object parameters, Document doc, Transaction tr)
        {
            string name = GetString(parameters, "name");
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("Layer name required");

            Database db = doc.Database;
            LayerTable lt = (LayerTable)tr.GetObject(db.LayerTableId, OpenMode.ForRead);
            if (!lt.Has(name)) throw new ArgumentException($"Layer '{name}' not found");

            ObjectId layerId = lt[name];
            LayerTableRecord ltr = (LayerTableRecord)tr.GetObject(layerId, OpenMode.ForRead);
            if (ltr.IsFrozen) throw new ArgumentException("Cannot set frozen layer as current");

            db.Clayer = layerId;
            return new { name, set_as_current = true };
        }

        private T Deserialize<T>(object p) where T : new() => p == null ? new T() : JsonConvert.DeserializeObject<T>(JsonConvert.SerializeObject(p));
        private string GetString(object p, string key)
        {
            if (p == null) return null;
            var dict = JsonConvert.DeserializeObject<Dictionary<string, object>>(JsonConvert.SerializeObject(p));
            return dict != null && dict.TryGetValue(key, out object v) ? v?.ToString() : null;
        }
    }
}
