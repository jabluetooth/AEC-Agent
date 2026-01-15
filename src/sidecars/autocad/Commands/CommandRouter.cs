using System;
using System.Collections.Generic;
using Autodesk.AutoCAD.ApplicationServices;
using Autodesk.AutoCAD.DatabaseServices;
using AECAgent.AutoCAD.Utils;

namespace AECAgent.AutoCAD.Commands
{
    public class CommandRouter
    {
        private readonly DrawingCommands _drawingCommands;
        private readonly QueryCommands _queryCommands;
        private readonly LayerCommands _layerCommands;
        private readonly Dictionary<string, Func<object, Document, Transaction, object>> _handlers;

        public CommandRouter()
        {
            _drawingCommands = new DrawingCommands();
            _queryCommands = new QueryCommands();
            _layerCommands = new LayerCommands();

            _handlers = new Dictionary<string, Func<object, Document, Transaction, object>>(StringComparer.OrdinalIgnoreCase)
            {
                { "draw_polyline", _drawingCommands.DrawPolyline },
                { "draw_line", _drawingCommands.DrawLine },
                { "draw_circle", _drawingCommands.DrawCircle },
                { "draw_rectangle", _drawingCommands.DrawRectangle },
                { "draw_text", _drawingCommands.DrawText },
                { "modify_entity", _drawingCommands.ModifyEntity },
                { "delete_entity", _drawingCommands.DeleteEntity },
                { "get_entities", _queryCommands.GetEntities },
                { "get_entity", _queryCommands.GetEntity },
                { "get_drawing_info", _queryCommands.GetDrawingInfo },
                { "audit_layers", _queryCommands.AuditLayers },
                { "create_layer", _layerCommands.CreateLayer },
                { "modify_layer", _layerCommands.ModifyLayer },
                { "delete_layer", _layerCommands.DeleteLayer },
                { "set_current_layer", _layerCommands.SetCurrentLayer },
                { "zoom_extents", (p, d, t) => _queryCommands.ZoomExtents(p, d, t) }
            };
        }

        public object Execute(string command, object parameters, Document doc, Transaction tr)
        {
            if (string.IsNullOrWhiteSpace(command))
                throw new ArgumentException("Command name is required");

            if (!_handlers.TryGetValue(command, out var handler))
            {
                Logger.Warn($"Unknown command: {command}");
                throw new ArgumentException($"Unknown command: {command}");
            }

            return handler(parameters, doc, tr);
        }

        public IEnumerable<string> GetAvailableCommands() => _handlers.Keys;
    }
}
