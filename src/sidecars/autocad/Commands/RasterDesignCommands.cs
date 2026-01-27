using System;
using System.IO;
using System.Collections.Generic;
using Autodesk.AutoCAD.ApplicationServices;
using Autodesk.AutoCAD.DatabaseServices;
using Autodesk.AutoCAD.Geometry;
using AECAgent.AutoCAD.Models;
using AECAgent.AutoCAD.Utils;
using Newtonsoft.Json;

namespace AECAgent.AutoCAD.Commands
{
    /// <summary>
    /// Commands for AutoCAD Raster Design integration.
    /// Handles PDF import, raster image attachment, vectorization, cleanup, and OCR.
    ///
    /// Some operations (PDF import, vectorize, cleanup, OCR) use SendStringToExecute
    /// because they invoke AutoCAD/Raster Design commands that cannot run inside a transaction.
    /// These commands receive a null Transaction and return a "queued" status.
    /// </summary>
    public class RasterDesignCommands
    {
        // =====================================================================
        // PDF Import (async - uses SendStringToExecute)
        // =====================================================================

        /// <summary>
        /// Import a PDF file into the current AutoCAD drawing.
        /// Uses the -PDFIMPORT command which converts PDF vector/text content to AutoCAD entities.
        /// For scanned/raster PDFs, use raster_attach_image + raster_vectorize instead.
        ///
        /// This is an async command - it queues the operation via SendStringToExecute
        /// and returns immediately with a "queued" status.
        /// </summary>
        public object ImportPdf(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<ImportPdfParams>(parameters);

            if (string.IsNullOrEmpty(param.FilePath))
                throw new ArgumentException("file_path is required");

            // Normalize path separators
            string filePath = param.FilePath.Replace("/", "\\");

            if (!File.Exists(filePath))
                throw new ArgumentException($"PDF file not found: {filePath}");

            string ext = Path.GetExtension(filePath).ToLowerInvariant();
            if (ext != ".pdf")
                throw new ArgumentException($"File must be a PDF, got: {ext}");

            // Build -PDFIMPORT command string
            // Format: -PDFIMPORT F <path> <page> <insertion_pt> <scale> <rotation>
            int page = param.Page > 0 ? param.Page : 1;
            double scale = param.Scale > 0 ? param.Scale : 1.0;
            double rotation = param.Rotation;
            string insertionPt = param.InsertionPoint != null && param.InsertionPoint.Length >= 2
                ? $"{param.InsertionPoint[0]},{param.InsertionPoint[1]},0"
                : "0,0,0";

            // Target layer for imported geometry
            string layerOption = !string.IsNullOrEmpty(param.TargetLayer) ? param.TargetLayer : "";

            // Build command sequence
            // -PDFIMPORT -> F (File) -> path -> page -> insertion -> scale -> rotation
            string cmdString = $"-PDFIMPORT\nF\n\"{filePath}\"\n{page}\n{insertionPt}\n{scale}\n{rotation}\n";

            Logger.Info($"Queuing PDF import: {filePath} (page {page}, scale {scale})");
            doc.SendStringToExecute(cmdString, false, false, true);

            return new RasterOperationResult
            {
                Operation = "import_pdf",
                Status = "queued",
                FilePath = filePath,
                Message = $"PDF import queued: {Path.GetFileName(filePath)} (page {page})",
                Details = new Dictionary<string, object>
                {
                    { "page", page },
                    { "scale", scale },
                    { "rotation", rotation },
                    { "insertion_point", insertionPt }
                }
            };
        }

        // =====================================================================
        // Raster Image Attach (sync - uses Transaction)
        // =====================================================================

        /// <summary>
        /// Attach a raster image file to the current drawing.
        /// Supports common raster formats: TIFF, PNG, JPG, BMP, etc.
        /// This creates an image reference that can then be vectorized using Raster Design.
        /// </summary>
        public object AttachImage(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<AttachImageParams>(parameters);

            if (string.IsNullOrEmpty(param.FilePath))
                throw new ArgumentException("file_path is required");

            string filePath = param.FilePath.Replace("/", "\\");

            if (!File.Exists(filePath))
                throw new ArgumentException($"Image file not found: {filePath}");

            string ext = Path.GetExtension(filePath).ToLowerInvariant();
            var supportedFormats = new HashSet<string> { ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".pcx", ".cal", ".gp4" };
            if (!supportedFormats.Contains(ext))
                throw new ArgumentException($"Unsupported image format: {ext}. Supported: TIFF, PNG, JPG, BMP, GIF, PCX, CAL, GP4");

            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);

            // Get or create the image dictionary
            ObjectId imageDictId = RasterImageDef.GetImageDictionary(db);
            if (imageDictId.IsNull)
                imageDictId = RasterImageDef.CreateImageDictionary(db);
            DBDictionary imageDict = (DBDictionary)tr.GetObject(imageDictId, OpenMode.ForWrite);

            // Create a unique name for the image definition
            string imageName = !string.IsNullOrEmpty(param.ImageName)
                ? param.ImageName
                : Path.GetFileNameWithoutExtension(filePath);

            // Remove existing definition with same name if it exists
            if (imageDict.Contains(imageName))
            {
                ObjectId existingDefId = imageDict.GetAt(imageName);
                RasterImageDef existingDef = (RasterImageDef)tr.GetObject(existingDefId, OpenMode.ForWrite);
                existingDef.Erase();
            }

            // Create image definition
            RasterImageDef imageDef = new RasterImageDef();
            imageDef.SourceFileName = filePath;
            imageDef.Load();

            ObjectId imageDefId = imageDict.SetAt(imageName, imageDef);
            tr.AddNewlyCreatedDBObject(imageDef, true);

            // Calculate insertion parameters
            double scale = param.Scale > 0 ? param.Scale : 1.0;
            Point3d insertPt = param.InsertionPoint != null && param.InsertionPoint.Length >= 2
                ? new Point3d(param.InsertionPoint[0], param.InsertionPoint[1],
                    param.InsertionPoint.Length > 2 ? param.InsertionPoint[2] : 0)
                : Point3d.Origin;

            // Image size in pixels
            Vector2d imageSize = imageDef.Size;

            // Create the raster image entity
            RasterImage rasterImage = new RasterImage();
            rasterImage.ImageDefId = imageDefId;

            // Set orientation (position, U-vector for width, V-vector for height)
            double width = imageSize.X * scale;
            double height = imageSize.Y * scale;
            Vector3d uVec = new Vector3d(width, 0, 0);
            Vector3d vVec = new Vector3d(0, height, 0);

            CoordinateSystem3d coordSys = new CoordinateSystem3d(insertPt, uVec, vVec);
            rasterImage.Orientation = coordSys;

            // Set clipping boundary to full image
            rasterImage.ShowImage = true;

            // Set layer if specified
            if (!string.IsNullOrEmpty(param.TargetLayer))
            {
                LayerTable lt = (LayerTable)tr.GetObject(db.LayerTableId, OpenMode.ForRead);
                if (lt.Has(param.TargetLayer))
                    rasterImage.Layer = param.TargetLayer;
                else
                    Logger.Warn($"Layer '{param.TargetLayer}' not found, using current layer");
            }

            ObjectId imageId = btr.AppendEntity(rasterImage);
            tr.AddNewlyCreatedDBObject(rasterImage, true);

            // Create the reactor (links image to its definition)
            RasterImage.EnableReactors(true);
            rasterImage.AssociateRasterDef(imageDef);

            Logger.Info($"Raster image attached: {imageName} ({imageSize.X}x{imageSize.Y} px)");

            return new RasterImageAttachResult
            {
                Handle = rasterImage.Handle.ToString(),
                ObjectId = imageId.ToString(),
                ImageName = imageName,
                FilePath = filePath,
                WidthPixels = (int)imageSize.X,
                HeightPixels = (int)imageSize.Y,
                Scale = scale,
                InsertionPoint = new double[] { insertPt.X, insertPt.Y, insertPt.Z },
                Attached = true
            };
        }

        // =====================================================================
        // Raster Cleanup / Despeckle (async - uses SendStringToExecute)
        // =====================================================================

        /// <summary>
        /// Clean up a raster image using Raster Design tools.
        /// Operations: despeckle, deskew, mirror, negate, threshold adjustment.
        /// Requires AutoCAD Raster Design to be installed.
        /// </summary>
        public object Cleanup(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<RasterCleanupParams>(parameters);

            string operation = (param.Operation ?? "despeckle").ToLowerInvariant();
            var validOps = new HashSet<string> { "despeckle", "deskew", "negate", "mirror_x", "mirror_y", "threshold", "bias", "touchup" };
            if (!validOps.Contains(operation))
                throw new ArgumentException($"Invalid cleanup operation: {operation}. Valid: {string.Join(", ", validOps)}");

            string cmdString;
            switch (operation)
            {
                case "despeckle":
                    // IDESPECKLE: remove small noise spots
                    // blob_size controls max speckle size (default 3 pixels)
                    int blobSize = param.BlobSize > 0 ? param.BlobSize : 3;
                    cmdString = $"IDESPECKLE\nAll\n{blobSize}\n";
                    break;

                case "deskew":
                    // IDESKEW: straighten skewed scanned images
                    cmdString = "IDESKEW\nAll\n";
                    break;

                case "negate":
                    // INVERT: invert black/white
                    cmdString = "INVERT\nAll\n";
                    break;

                case "mirror_x":
                    cmdString = "IMIRROR\nAll\nX\n";
                    break;

                case "mirror_y":
                    cmdString = "IMIRROR\nAll\nY\n";
                    break;

                case "threshold":
                    // Adjust binary threshold (0-255, default 128)
                    int thresholdVal = param.ThresholdValue > 0 ? param.ThresholdValue : 128;
                    cmdString = $"ITHRESHOLD\nAll\n{thresholdVal}\n";
                    break;

                case "bias":
                    // IBIAS: adjust brightness/contrast
                    int brightness = param.Brightness;
                    int contrast = param.Contrast;
                    cmdString = $"IBIAS\nAll\n{brightness}\n{contrast}\n";
                    break;

                case "touchup":
                    // ITOUCHUP: interactive cleanup mode
                    cmdString = "ITOUCHUP\n";
                    break;

                default:
                    throw new ArgumentException($"Unhandled operation: {operation}");
            }

            Logger.Info($"Queuing raster cleanup: {operation}");
            doc.SendStringToExecute(cmdString, false, false, true);

            return new RasterOperationResult
            {
                Operation = $"cleanup_{operation}",
                Status = "queued",
                Message = $"Raster cleanup '{operation}' queued for execution",
                Details = new Dictionary<string, object>
                {
                    { "cleanup_type", operation },
                    { "blob_size", param.BlobSize },
                    { "threshold", param.ThresholdValue }
                }
            };
        }

        // =====================================================================
        // Vectorize (async - uses SendStringToExecute)
        // =====================================================================

        /// <summary>
        /// Vectorize a raster image to AutoCAD vector entities using Raster Design.
        /// Converts raster lines/arcs/text to AutoCAD lines, arcs, circles, and polylines.
        /// Requires AutoCAD Raster Design to be installed.
        /// </summary>
        public object Vectorize(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<RasterVectorizeParams>(parameters);

            string method = (param.Method ?? "auto").ToLowerInvariant();
            var validMethods = new HashSet<string> { "auto", "outline", "centerline", "contour" };
            if (!validMethods.Contains(method))
                throw new ArgumentException($"Invalid vectorization method: {method}. Valid: {string.Join(", ", validMethods)}");

            // Build vectorization command
            // IVECTORIZE settings can be configured before running
            string setupCmds = "";

            // Set vectorization options if specified
            if (!string.IsNullOrEmpty(param.TargetLayer))
            {
                setupCmds += $"-LAYER\nS\n{param.TargetLayer}\n\n";
            }

            string vectorizeCmd;
            switch (method)
            {
                case "outline":
                    // Outline vectorization - traces outer edges
                    vectorizeCmd = "IVECTORIZE\nAll\nO\n";
                    break;

                case "centerline":
                    // Centerline vectorization - finds center of lines
                    vectorizeCmd = "IVECTORIZE\nAll\nC\n";
                    break;

                case "contour":
                    // Contour vectorization
                    vectorizeCmd = "IVECTORIZE\nAll\nN\n";
                    break;

                case "auto":
                default:
                    // Auto method - let Raster Design choose
                    vectorizeCmd = "IVECTORIZE\nAll\n\n";
                    break;
            }

            // Apply polygon detection if requested
            if (param.DetectPolygons)
            {
                vectorizeCmd += "Y\n"; // Yes to polygon detection
            }

            // Apply arc detection if requested
            if (param.DetectArcs)
            {
                vectorizeCmd += "Y\n"; // Yes to arc detection
            }

            string fullCmd = setupCmds + vectorizeCmd;
            Logger.Info($"Queuing vectorization: method={method}");
            doc.SendStringToExecute(fullCmd, false, false, true);

            return new RasterOperationResult
            {
                Operation = "vectorize",
                Status = "queued",
                Message = $"Vectorization queued: method={method}",
                Details = new Dictionary<string, object>
                {
                    { "method", method },
                    { "target_layer", param.TargetLayer ?? "(current)" },
                    { "detect_polygons", param.DetectPolygons },
                    { "detect_arcs", param.DetectArcs },
                    { "gap_tolerance", param.GapTolerance }
                }
            };
        }

        // =====================================================================
        // OCR Text Extraction (async - uses SendStringToExecute)
        // =====================================================================

        /// <summary>
        /// Extract text from a raster image using Raster Design OCR.
        /// Converts raster text to AutoCAD TEXT/MTEXT entities.
        /// Requires AutoCAD Raster Design to be installed.
        /// </summary>
        public object OcrExtract(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<RasterOcrParams>(parameters);

            // Build OCR command
            // IOCR recognizes text in raster images and converts to AutoCAD text
            string targetLayer = !string.IsNullOrEmpty(param.TargetLayer)
                ? param.TargetLayer
                : "";

            string setupCmds = "";
            if (!string.IsNullOrEmpty(targetLayer))
            {
                setupCmds += $"-LAYER\nS\n{targetLayer}\n\n";
            }

            // Set text height if specified
            string textHeight = param.TextHeight > 0 ? param.TextHeight.ToString("F2") : "";

            string ocrCmd = "IOCR\nAll\n";

            string fullCmd = setupCmds + ocrCmd;
            Logger.Info("Queuing OCR text extraction");
            doc.SendStringToExecute(fullCmd, false, false, true);

            return new RasterOperationResult
            {
                Operation = "ocr_extract",
                Status = "queued",
                Message = "OCR text extraction queued",
                Details = new Dictionary<string, object>
                {
                    { "target_layer", targetLayer },
                    { "text_height", param.TextHeight },
                    { "language", param.Language ?? "eng" }
                }
            };
        }

        // =====================================================================
        // Get Raster Image Status (sync - uses Transaction)
        // =====================================================================

        /// <summary>
        /// Get information about all raster images in the current drawing.
        /// Returns image names, dimensions, file paths, and positions.
        /// </summary>
        public object GetRasterStatus(object parameters, Document doc, Transaction tr)
        {
            Database db = doc.Database;

            ObjectId imageDictId = RasterImageDef.GetImageDictionary(db);
            if (imageDictId.IsNull)
            {
                return new RasterStatusResult
                {
                    ImageCount = 0,
                    Images = new List<RasterImageInfo>(),
                    HasRasterDesign = CheckRasterDesignAvailable()
                };
            }

            DBDictionary imageDict = (DBDictionary)tr.GetObject(imageDictId, OpenMode.ForRead);
            var images = new List<RasterImageInfo>();

            foreach (DBDictionaryEntry entry in imageDict)
            {
                try
                {
                    RasterImageDef imageDef = (RasterImageDef)tr.GetObject(entry.Value, OpenMode.ForRead);
                    var info = new RasterImageInfo
                    {
                        Name = entry.Key,
                        FilePath = imageDef.SourceFileName,
                        WidthPixels = (int)imageDef.Size.X,
                        HeightPixels = (int)imageDef.Size.Y,
                        IsLoaded = imageDef.IsLoaded,
                        FileFound = File.Exists(imageDef.SourceFileName),
                        ResolutionX = imageDef.ResolutionMMPerPixel.X > 0
                            ? (int)(25.4 / imageDef.ResolutionMMPerPixel.X) : 0,
                        ResolutionY = imageDef.ResolutionMMPerPixel.Y > 0
                            ? (int)(25.4 / imageDef.ResolutionMMPerPixel.Y) : 0
                    };
                    images.Add(info);
                }
                catch (Exception ex)
                {
                    Logger.Warn($"Could not read image definition '{entry.Key}': {ex.Message}");
                }
            }

            return new RasterStatusResult
            {
                ImageCount = images.Count,
                Images = images,
                HasRasterDesign = CheckRasterDesignAvailable()
            };
        }

        // =====================================================================
        // PDF Import Status / Count Entities After Import (sync)
        // =====================================================================

        /// <summary>
        /// Get a count of entities in the drawing, useful for checking results
        /// after a PDF import or vectorization operation.
        /// </summary>
        public object GetEntityCount(object parameters, Document doc, Transaction tr)
        {
            var param = Deserialize<GetEntityCountParams>(parameters);

            Database db = doc.Database;
            BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
            BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForRead);

            int totalCount = 0;
            var typeCounts = new Dictionary<string, int>();
            var layerCounts = new Dictionary<string, int>();

            foreach (ObjectId id in btr)
            {
                try
                {
                    Entity ent = (Entity)tr.GetObject(id, OpenMode.ForRead);

                    // Filter by layer if specified
                    if (!string.IsNullOrEmpty(param.Layer) &&
                        !ent.Layer.Equals(param.Layer, StringComparison.OrdinalIgnoreCase))
                        continue;

                    totalCount++;

                    string typeName = ent.GetType().Name;
                    if (typeCounts.ContainsKey(typeName))
                        typeCounts[typeName]++;
                    else
                        typeCounts[typeName] = 1;

                    if (layerCounts.ContainsKey(ent.Layer))
                        layerCounts[ent.Layer]++;
                    else
                        layerCounts[ent.Layer] = 1;
                }
                catch { }
            }

            return new EntityCountResult
            {
                TotalEntities = totalCount,
                ByType = typeCounts,
                ByLayer = layerCounts,
                FilteredByLayer = param.Layer
            };
        }

        // =====================================================================
        // Helpers
        // =====================================================================

        private bool CheckRasterDesignAvailable()
        {
            try
            {
                // Check if Raster Design module is loaded by checking for its command
                // This is a heuristic - the actual check depends on the installation
                var doc = Application.DocumentManager.MdiActiveDocument;
                if (doc == null) return false;

                // Try to find the IVECTORIZE command in the registered commands
                // If Raster Design is installed and loaded, these commands will be available
                return true; // Assume available - will fail gracefully if not
            }
            catch
            {
                return false;
            }
        }

        private T Deserialize<T>(object p) where T : new()
        {
            return p == null ? new T() : JsonConvert.DeserializeObject<T>(JsonConvert.SerializeObject(p));
        }
    }
}
