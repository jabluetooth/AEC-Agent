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
        private readonly RasterDesignCommands _rasterCommands;
        private readonly Dictionary<string, Func<object, Document, Transaction, object>> _handlers;

        /// <summary>
        /// Commands that use SendStringToExecute instead of Transaction.
        /// These are executed outside the transaction in OnIdle (tr passed as null).
        /// </summary>
        private readonly HashSet<string> _asyncCommands;

        public CommandRouter()
        {
            _drawingCommands = new DrawingCommands();
            _queryCommands = new QueryCommands();
            _layerCommands = new LayerCommands();
            _rasterCommands = new RasterDesignCommands();

            _handlers = new Dictionary<string, Func<object, Document, Transaction, object>>(StringComparer.OrdinalIgnoreCase)
            {
                // Drawing commands
                { "draw_polyline", _drawingCommands.DrawPolyline },
                { "draw_line", _drawingCommands.DrawLine },
                { "draw_circle", _drawingCommands.DrawCircle },
                { "draw_arc", _drawingCommands.DrawArc },
                { "draw_ellipse", _drawingCommands.DrawEllipse },
                { "draw_spline", _drawingCommands.DrawSpline },
                { "draw_rectangle", _drawingCommands.DrawRectangle },
                { "draw_text", _drawingCommands.DrawText },
                { "modify_entity", _drawingCommands.ModifyEntity },
                { "delete_entity", _drawingCommands.DeleteEntity },
                // Query commands
                { "get_entities", _queryCommands.GetEntities },
                { "get_entity", _queryCommands.GetEntity },
                { "get_drawing_info", _queryCommands.GetDrawingInfo },
                { "audit_layers", _queryCommands.AuditLayers },
                // Layer commands
                { "create_layer", _layerCommands.CreateLayer },
                { "modify_layer", _layerCommands.ModifyLayer },
                { "delete_layer", _layerCommands.DeleteLayer },
                { "set_current_layer", _layerCommands.SetCurrentLayer },
                { "zoom_extents", (p, d, t) => _queryCommands.ZoomExtents(p, d, t) },
                // Raster Design commands (async - use SendStringToExecute)
                { "raster_import_pdf", _rasterCommands.ImportPdf },
                { "raster_cleanup", _rasterCommands.Cleanup },
                { "raster_vectorize", _rasterCommands.Vectorize },
                { "raster_ocr", _rasterCommands.OcrExtract },
                // Raster Design — VTools, Followers, Primitives, Processing (async)
                { "raster_process_image", _rasterCommands.ProcessImage },
                { "raster_create_primitive", _rasterCommands.CreatePrimitive },
                { "raster_select_entities", _rasterCommands.SelectRasterEntities },
                { "raster_follower", _rasterCommands.Follower },
                { "raster_recognize_text", _rasterCommands.RecognizeText },
                // Raster Design commands (sync - use Transaction)
                { "raster_attach_image", _rasterCommands.AttachImage },
                { "raster_get_status", _rasterCommands.GetRasterStatus },
                { "raster_get_entity_count", _rasterCommands.GetEntityCount },
                { "extract_all_entities", _rasterCommands.ExtractAllEntities },
                { "raster_fade_image", _rasterCommands.FadeImage }
            };

            // Async commands use SendStringToExecute and do not need a Transaction.
            // The OnIdle handler should execute these with DocumentLock but no Transaction.
            _asyncCommands = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "raster_import_pdf",
                "raster_cleanup",
                "raster_vectorize",
                "raster_ocr",
                "raster_process_image",
                "raster_create_primitive",
                "raster_select_entities",
                "raster_follower",
                "raster_recognize_text"
            };
        }

        /// <summary>
        /// Returns true if the command uses SendStringToExecute and should be
        /// executed outside a Transaction (only DocumentLock required).
        /// </summary>
        public bool IsAsyncCommand(string command)
        {
            return !string.IsNullOrWhiteSpace(command) && _asyncCommands.Contains(command);
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
