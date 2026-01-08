# Phase 2: AutoCAD .NET Sidecar (The IPC Bridge)

**Objective:** Build an in-process HTTP listener to drive the AutoCAD database safely.

## 1. Project Setup
*   **Framework:** .NET Framework 4.8 (check specific AutoCAD version requirements).
*   **References:** `acmgd.dll`, `acdbmgd.dll`, `accoremgd.dll`.
*   **Build Tool:** Use **ILMerge** to bundle any dependencies (like `Newtonsoft.Json`) into the final DLL. This prevents "DLL Hell" where your plugin's dependencies conflict with AutoCAD's internal libraries.

## 2. Initialize Listener (`IExtensionApplication`)
*   Implement the `IExtensionApplication` interface.
*   In `Initialize()`:
    1.  Read `Environment.GetEnvironmentVariable("MCP_LISTENER_PORT")`.
    2.  Start `System.Net.HttpListener` bound to `http://127.0.0.1:{PORT}/`.
    3.  **CRITICAL:** Spin off the listener loop to a `Task.Run()` background thread to avoid freezing the AutoCAD splash screen.

## 3. Thread Marshaling Strategy
AutoCAD is a Single-Threaded Apartment (STA) application. You **cannot** modify the database from the HttpListener thread.

*   **The Queue Pattern:**
    *   Create a `static ConcurrentQueue<Action> _jobQueue`.
    *   When an HTTP request comes in, parse it, wrap the API work in an `Action`, and `Enqueue` it.
*   **Execution:**
    *   **Option A (Modern):** Use `Application.DocumentManager.ExecuteInCommandContextAsync` (if supported for your specific context).
    *   **Option B (Robust):** Register a handler for the `Application.Idle` event.
        ```csharp
        private static readonly ILogger _logger = LogManager.GetLogger();

        private void OnIdle(object sender, EventArgs e) {
            if (_jobQueue.TryDequeue(out Action action)) {
                Document doc = Application.DocumentManager.MdiActiveDocument;
                if (doc == null) {
                    _logger.Warn("No active document, re-queuing action");
                    _jobQueue.Enqueue(action);
                    return;
                }

                using (DocumentLock docLock = doc.LockDocument()) {
                    using (Transaction tr = doc.TransactionManager.StartTransaction()) {
                        try {
                            action();
                            tr.Commit();
                            _logger.Info("Transaction committed successfully");
                        }
                        catch (Autodesk.AutoCAD.Runtime.Exception acEx) {
                            tr.Abort();
                            _logger.Error($"AutoCAD error: {acEx.ErrorStatus} - {acEx.Message}");
                            // Notify the waiting HTTP request of failure
                            throw;
                        }
                        catch (Exception ex) {
                            tr.Abort();
                            _logger.Error($"Unexpected error: {ex.Message}");
                            throw;
                        }
                    }
                }
            }
        }
        ```

## 4. Atomic Tool Exposure
Define these endpoints mapping to atomic operations:
*   `POST /cad-command/draw_polyline`: Accepts `[[x,y], [x,y]]`.
*   `POST /cad-command/audit_layers`: Returns JSON list of layers.
*   **Security Check:** Every request headers must contain `Authorization: {SESSION_TOKEN}` matching the environment variable. Reject if mismatch.

### 4.1 Request/Response Structure

All requests follow this pattern:

```json
{
  "command": "draw_polyline",
  "parameters": {
    "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
    "layer": "Walls",
    "closed": true
  }
}
```

All responses follow this pattern:

```json
{
  "success": true,
  "data": {
    "objectId": "7FFFFFFF",
    "created": true
  },
  "error": null
}
```

Error response:

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": 4001,
    "message": "Layer 'InvalidLayer' does not exist",
    "details": "eKeyNotFound"
  }
}
```

### 4.2 HTTP Listener Implementation

```csharp
using System;
using System.Collections.Concurrent;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using Autodesk.AutoCAD.ApplicationServices;
using Autodesk.AutoCAD.Runtime;
using Newtonsoft.Json;

namespace AECAgent.AutoCAD
{
    public class SidecarPlugin : IExtensionApplication
    {
        private static HttpListener _listener;
        private static ConcurrentQueue<JobRequest> _jobQueue;
        private static string _sessionToken;
        private static int _listenerPort;
        private static bool _isRunning;

        public void Initialize()
        {
            try
            {
                // Read environment variables set by GPO script
                _sessionToken = Environment.GetEnvironmentVariable("SESSION_TOKEN");
                string portStr = Environment.GetEnvironmentVariable("MCP_LISTENER_PORT");

                if (string.IsNullOrEmpty(_sessionToken) || string.IsNullOrEmpty(portStr))
                {
                    LogError("SESSION_TOKEN or MCP_LISTENER_PORT not set. Plugin disabled.");
                    return;
                }

                _listenerPort = int.Parse(portStr);
                _jobQueue = new ConcurrentQueue<JobRequest>();

                // Register Idle event handler for job processing
                Application.Idle += OnIdle;

                // Start HTTP listener in background thread
                Task.Run(() => StartHttpListener());

                LogInfo($"AEC Agent Sidecar initialized on port {_listenerPort}");
            }
            catch (Exception ex)
            {
                LogError($"Failed to initialize sidecar: {ex.Message}");
            }
        }

        public void Terminate()
        {
            _isRunning = false;
            Application.Idle -= OnIdle;

            if (_listener != null && _listener.IsListening)
            {
                _listener.Stop();
                _listener.Close();
            }

            LogInfo("AEC Agent Sidecar terminated");
        }

        private async Task StartHttpListener()
        {
            try
            {
                _listener = new HttpListener();
                _listener.Prefixes.Add($"http://127.0.0.1:{_listenerPort}/");
                _listener.Start();
                _isRunning = true;

                LogInfo($"HTTP listener started on port {_listenerPort}");

                while (_isRunning)
                {
                    try
                    {
                        var context = await _listener.GetContextAsync();
                        // Fire and forget - handle in background
                        _ = Task.Run(() => HandleRequest(context));
                    }
                    catch (HttpListenerException)
                    {
                        // Expected when stopping
                        break;
                    }
                    catch (Exception ex)
                    {
                        LogError($"Listener error: {ex.Message}");
                    }
                }
            }
            catch (Exception ex)
            {
                LogError($"Failed to start HTTP listener: {ex.Message}");
            }
        }

        private async Task HandleRequest(HttpListenerContext context)
        {
            var request = context.Request;
            var response = context.Response;

            try
            {
                // CORS headers for localhost development
                response.AddHeader("Access-Control-Allow-Origin", "http://localhost:54321");
                response.AddHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
                response.AddHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

                // Handle preflight
                if (request.HttpMethod == "OPTIONS")
                {
                    response.StatusCode = 200;
                    response.Close();
                    return;
                }

                // Security check
                string authHeader = request.Headers["Authorization"];
                if (authHeader != _sessionToken)
                {
                    SendErrorResponse(response, 401, "Unauthorized", "Invalid session token");
                    return;
                }

                // Parse request body
                string body;
                using (var reader = new System.IO.StreamReader(request.InputStream, request.ContentEncoding))
                {
                    body = await reader.ReadToEndAsync();
                }

                var jobRequest = JsonConvert.DeserializeObject<JobRequest>(body);
                jobRequest.ResponseContext = context;

                // Enqueue for processing on main thread
                _jobQueue.Enqueue(jobRequest);

                // Wait for completion (with timeout)
                var timeout = TimeSpan.FromSeconds(120);
                var startTime = DateTime.UtcNow;

                while (!jobRequest.IsCompleted && (DateTime.UtcNow - startTime) < timeout)
                {
                    await Task.Delay(50);
                }

                if (!jobRequest.IsCompleted)
                {
                    SendErrorResponse(response, 504, "Gateway Timeout", "CAD operation timed out");
                    return;
                }

                // Send response
                if (jobRequest.Error != null)
                {
                    SendErrorResponse(response, 500, "CAD Error", jobRequest.Error);
                }
                else
                {
                    SendSuccessResponse(response, jobRequest.Result);
                }
            }
            catch (Exception ex)
            {
                LogError($"Request handling error: {ex.Message}");
                SendErrorResponse(response, 500, "Internal Error", ex.Message);
            }
        }

        private void SendSuccessResponse(HttpListenerResponse response, object data)
        {
            var result = new
            {
                success = true,
                data = data,
                error = (object)null
            };

            SendJsonResponse(response, 200, result);
        }

        private void SendErrorResponse(HttpListenerResponse response, int statusCode, string message, string details)
        {
            var result = new
            {
                success = false,
                data = (object)null,
                error = new
                {
                    code = statusCode,
                    message = message,
                    details = details
                }
            };

            SendJsonResponse(response, statusCode, result);
        }

        private void SendJsonResponse(HttpListenerResponse response, int statusCode, object data)
        {
            response.StatusCode = statusCode;
            response.ContentType = "application/json";

            string json = JsonConvert.SerializeObject(data);
            byte[] buffer = Encoding.UTF8.GetBytes(json);

            response.ContentLength64 = buffer.Length;
            response.OutputStream.Write(buffer, 0, buffer.Length);
            response.Close();
        }

        private void LogInfo(string message)
        {
            Application.DocumentManager.MdiActiveDocument?.Editor.WriteMessage($"\n[AEC Agent] {message}");
        }

        private void LogError(string message)
        {
            Application.DocumentManager.MdiActiveDocument?.Editor.WriteMessage($"\n[AEC Agent ERROR] {message}");
        }
    }

    public class JobRequest
    {
        public string Command { get; set; }
        public object Parameters { get; set; }
        public object Result { get; set; }
        public string Error { get; set; }
        public bool IsCompleted { get; set; }

        [JsonIgnore]
        public HttpListenerContext ResponseContext { get; set; }
    }
}
```

## 5. Job Processing on Main Thread

The `OnIdle` event handler executes queued jobs on AutoCAD's main thread:

```csharp
private void OnIdle(object sender, EventArgs e)
{
    if (!_jobQueue.TryDequeue(out JobRequest job))
    {
        return;
    }

    Document doc = Application.DocumentManager.MdiActiveDocument;
    if (doc == null)
    {
        job.Error = "No active document";
        job.IsCompleted = true;
        LogError("No active document, cannot process job");
        return;
    }

    try
    {
        using (DocumentLock docLock = doc.LockDocument())
        {
            using (Transaction tr = doc.TransactionManager.StartTransaction())
            {
                try
                {
                    // Route to command handlers
                    job.Result = ExecuteCommand(job.Command, job.Parameters, doc, tr);
                    tr.Commit();
                    job.IsCompleted = true;
                }
                catch (Autodesk.AutoCAD.Runtime.Exception acEx)
                {
                    tr.Abort();
                    job.Error = $"AutoCAD error {acEx.ErrorStatus}: {acEx.Message}";
                    job.IsCompleted = true;
                    LogError(job.Error);
                }
                catch (Exception ex)
                {
                    tr.Abort();
                    job.Error = $"Unexpected error: {ex.Message}";
                    job.IsCompleted = true;
                    LogError(job.Error);
                }
            }
        }
    }
    catch (Exception ex)
    {
        job.Error = $"Document lock error: {ex.Message}";
        job.IsCompleted = true;
        LogError(job.Error);
    }
}

private object ExecuteCommand(string command, object parameters, Document doc, Transaction tr)
{
    switch (command.ToLower())
    {
        case "draw_polyline":
            return DrawPolyline(parameters, doc, tr);

        case "audit_layers":
            return AuditLayers(doc, tr);

        case "create_layer":
            return CreateLayer(parameters, doc, tr);

        case "get_entities":
            return GetEntities(parameters, doc, tr);

        case "modify_entity":
            return ModifyEntity(parameters, doc, tr);

        case "delete_entity":
            return DeleteEntity(parameters, doc, tr);

        case "get_drawing_info":
            return GetDrawingInfo(doc);

        default:
            throw new ArgumentException($"Unknown command: {command}");
    }
}
```

## 6. Command Implementations

### 6.1 Draw Polyline

```csharp
using Autodesk.AutoCAD.DatabaseServices;
using Autodesk.AutoCAD.Geometry;

private object DrawPolyline(object parameters, Document doc, Transaction tr)
{
    var json = JsonConvert.SerializeObject(parameters);
    var param = JsonConvert.DeserializeObject<DrawPolylineParams>(json);

    if (param.Points == null || param.Points.Length < 2)
    {
        throw new ArgumentException("At least 2 points required");
    }

    Database db = doc.Database;
    BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
    BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForWrite);

    using (Polyline pline = new Polyline())
    {
        for (int i = 0; i < param.Points.Length; i++)
        {
            pline.AddVertexAt(i, new Point2d(param.Points[i][0], param.Points[i][1]), 0, 0, 0);
        }

        pline.Closed = param.Closed;

        // Set layer if specified
        if (!string.IsNullOrEmpty(param.Layer))
        {
            LayerTable lt = (LayerTable)tr.GetObject(db.LayerTableId, OpenMode.ForRead);
            if (lt.Has(param.Layer))
            {
                pline.Layer = param.Layer;
            }
            else
            {
                throw new ArgumentException($"Layer '{param.Layer}' does not exist");
            }
        }

        ObjectId polyId = btr.AppendEntity(pline);
        tr.AddNewlyCreatedDBObject(pline, true);

        return new
        {
            objectId = polyId.Handle.ToString(),
            created = true,
            layer = pline.Layer,
            vertexCount = pline.NumberOfVertices
        };
    }
}

public class DrawPolylineParams
{
    public double[][] Points { get; set; }
    public bool Closed { get; set; }
    public string Layer { get; set; }
}
```

### 6.2 Audit Layers

```csharp
private object AuditLayers(Document doc, Transaction tr)
{
    Database db = doc.Database;
    LayerTable lt = (LayerTable)tr.GetObject(db.LayerTableId, OpenMode.ForRead);

    var layers = new List<object>();

    foreach (ObjectId layerId in lt)
    {
        LayerTableRecord layer = (LayerTableRecord)tr.GetObject(layerId, OpenMode.ForRead);

        layers.Add(new
        {
            name = layer.Name,
            isFrozen = layer.IsFrozen,
            isLocked = layer.IsLocked,
            isOff = layer.IsOff,
            color = layer.Color.ColorIndex,
            lineWeight = layer.LineWeight.ToString()
        });
    }

    return new
    {
        layers = layers,
        count = layers.Count
    };
}
```

### 6.3 Create Layer

```csharp
private object CreateLayer(object parameters, Document doc, Transaction tr)
{
    var json = JsonConvert.SerializeObject(parameters);
    var param = JsonConvert.DeserializeObject<CreateLayerParams>(json);

    if (string.IsNullOrEmpty(param.Name))
    {
        throw new ArgumentException("Layer name is required");
    }

    Database db = doc.Database;
    LayerTable lt = (LayerTable)tr.GetObject(db.LayerTableId, OpenMode.ForWrite);

    if (lt.Has(param.Name))
    {
        throw new ArgumentException($"Layer '{param.Name}' already exists");
    }

    using (LayerTableRecord ltr = new LayerTableRecord())
    {
        ltr.Name = param.Name;

        if (param.Color.HasValue)
        {
            ltr.Color = Autodesk.AutoCAD.Colors.Color.FromColorIndex(
                Autodesk.AutoCAD.Colors.ColorMethod.ByAci,
                (short)param.Color.Value
            );
        }

        ObjectId layerId = lt.Add(ltr);
        tr.AddNewlyCreatedDBObject(ltr, true);

        return new
        {
            layerId = layerId.Handle.ToString(),
            name = ltr.Name,
            created = true
        };
    }
}

public class CreateLayerParams
{
    public string Name { get; set; }
    public int? Color { get; set; }
}
```

### 6.4 Get Entities

```csharp
private object GetEntities(object parameters, Document doc, Transaction tr)
{
    var json = JsonConvert.SerializeObject(parameters);
    var param = JsonConvert.DeserializeObject<GetEntitiesParams>(json);

    Database db = doc.Database;
    BlockTable bt = (BlockTable)tr.GetObject(db.BlockTableId, OpenMode.ForRead);
    BlockTableRecord btr = (BlockTableRecord)tr.GetObject(bt[BlockTableRecord.ModelSpace], OpenMode.ForRead);

    var entities = new List<object>();

    foreach (ObjectId objId in btr)
    {
        Entity ent = (Entity)tr.GetObject(objId, OpenMode.ForRead);

        // Filter by layer if specified
        if (!string.IsNullOrEmpty(param.Layer) && ent.Layer != param.Layer)
        {
            continue;
        }

        // Filter by type if specified
        if (!string.IsNullOrEmpty(param.EntityType) &&
            !ent.GetType().Name.ToLower().Contains(param.EntityType.ToLower()))
        {
            continue;
        }

        entities.Add(new
        {
            objectId = objId.Handle.ToString(),
            type = ent.GetType().Name,
            layer = ent.Layer,
            color = ent.Color.ColorIndex,
            bounds = GetEntityBounds(ent)
        });

        // Limit results
        if (param.Limit.HasValue && entities.Count >= param.Limit.Value)
        {
            break;
        }
    }

    return new
    {
        entities = entities,
        count = entities.Count
    };
}

public class GetEntitiesParams
{
    public string Layer { get; set; }
    public string EntityType { get; set; }
    public int? Limit { get; set; }
}

private object GetEntityBounds(Entity ent)
{
    try
    {
        Extents3d bounds = ent.GeometricExtents;
        return new
        {
            minX = bounds.MinPoint.X,
            minY = bounds.MinPoint.Y,
            minZ = bounds.MinPoint.Z,
            maxX = bounds.MaxPoint.X,
            maxY = bounds.MaxPoint.Y,
            maxZ = bounds.MaxPoint.Z
        };
    }
    catch
    {
        return null;
    }
}
```

### 6.5 Get Drawing Info

```csharp
private object GetDrawingInfo(Document doc)
{
    Database db = doc.Database;

    return new
    {
        fileName = doc.Name,
        isModified = doc.Database.RetainOriginalThumbnailBitmap,
        units = db.Insunits.ToString(),
        dwgVersion = db.OriginalFileVersion.ToString(),
        extentsMin = new
        {
            x = db.Extmin.X,
            y = db.Extmin.Y,
            z = db.Extmin.Z
        },
        extentsMax = new
        {
            x = db.Extmax.X,
            y = db.Extmax.Y,
            z = db.Extmax.Z
        }
    };
}
```

## 7. Project Structure

```
AutoCADSidecar/
├── Properties/
│   └── AssemblyInfo.cs
├── Commands/
│   ├── DrawingCommands.cs      # Draw, create, modify
│   ├── QueryCommands.cs        # Audit, get, list
│   └── LayerCommands.cs        # Layer management
├── Models/
│   ├── JobRequest.cs
│   ├── CommandParams.cs
│   └── CommandResponse.cs
├── HttpServer/
│   ├── SidecarListener.cs      # HTTP listener
│   └── RequestHandler.cs       # Route requests
├── Utils/
│   ├── Logger.cs               # Logging utility
│   └── SecurityValidator.cs    # Token validation
├── SidecarPlugin.cs            # IExtensionApplication
└── packages.config             # NuGet dependencies
```

## 8. Build Configuration

### 8.1 Project File (.csproj)

```xml
<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <PropertyGroup>
    <Configuration Condition=" '$(Configuration)' == '' ">Debug</Configuration>
    <Platform Condition=" '$(Platform)' == '' ">AnyCPU</Platform>
    <TargetFrameworkVersion>v4.8</TargetFrameworkVersion>
    <OutputType>Library</OutputType>
    <AppDesignerFolder>Properties</AppDesignerFolder>
    <RootNamespace>AECAgent.AutoCAD</RootNamespace>
    <AssemblyName>AECAgentSidecar</AssemblyName>
  </PropertyGroup>

  <PropertyGroup Condition=" '$(Configuration)|$(Platform)' == 'Debug|AnyCPU' ">
    <DebugSymbols>true</DebugSymbols>
    <DebugType>full</DebugType>
    <Optimize>false</Optimize>
    <OutputPath>bin\Debug\</OutputPath>
  </PropertyGroup>

  <PropertyGroup Condition=" '$(Configuration)|$(Platform)' == 'Release|AnyCPU' ">
    <DebugType>pdbonly</DebugType>
    <Optimize>true</Optimize>
    <OutputPath>bin\Release\</OutputPath>
  </PropertyGroup>

  <ItemGroup>
    <Reference Include="accoremgd">
      <HintPath>C:\Program Files\Autodesk\AutoCAD 2024\accoremgd.dll</HintPath>
      <Private>False</Private>
    </Reference>
    <Reference Include="acdbmgd">
      <HintPath>C:\Program Files\Autodesk\AutoCAD 2024\acdbmgd.dll</HintPath>
      <Private>False</Private>
    </Reference>
    <Reference Include="acmgd">
      <HintPath>C:\Program Files\Autodesk\AutoCAD 2024\acmgd.dll</HintPath>
      <Private>False</Private>
    </Reference>
    <Reference Include="Newtonsoft.Json">
      <HintPath>..\packages\Newtonsoft.Json.13.0.3\lib\net45\Newtonsoft.Json.dll</HintPath>
    </Reference>
    <Reference Include="System" />
    <Reference Include="System.Core" />
    <Reference Include="System.Net" />
    <Reference Include="System.Net.Http" />
  </ItemGroup>

  <ItemGroup>
    <Compile Include="SidecarPlugin.cs" />
    <Compile Include="Properties\AssemblyInfo.cs" />
  </ItemGroup>

  <ItemGroup>
    <None Include="packages.config" />
  </ItemGroup>

  <Import Project="$(MSBuildToolsPath)\Microsoft.CSharp.targets" />

  <!-- Post-build: Use ILMerge to bundle Newtonsoft.Json -->
  <Target Name="AfterBuild">
    <Exec Command="&quot;$(SolutionDir)packages\ILMerge.3.0.41\tools\net452\ILMerge.exe&quot; /out:&quot;$(TargetDir)$(TargetName).merged.dll&quot; &quot;$(TargetPath)&quot; &quot;$(TargetDir)Newtonsoft.Json.dll&quot; /targetplatform:v4,&quot;C:\Windows\Microsoft.NET\Framework64\v4.0.30319&quot;" />
  </Target>
</Project>
```

### 8.2 packages.config

```xml
<?xml version="1.0" encoding="utf-8"?>
<packages>
  <package id="Newtonsoft.Json" version="13.0.3" targetFramework="net48" />
  <package id="ILMerge" version="3.0.41" targetFramework="net48" />
</packages>
```

## 9. Deployment

### 9.1 Installation Script (PowerShell)

```powershell
# install-autocad-sidecar.ps1

param(
    [string]$AutoCADVersion = "2024",
    [string]$DllPath = ".\AECAgentSidecar.merged.dll"
)

$acadPath = "C:\Program Files\Autodesk\AutoCAD $AutoCADVersion"
$pluginPath = "$acadPath\Plug-ins\AECAgent"

# Create plugin directory
if (-not (Test-Path $pluginPath)) {
    New-Item -ItemType Directory -Path $pluginPath -Force
}

# Copy DLL
Copy-Item $DllPath -Destination "$pluginPath\AECAgentSidecar.dll" -Force

# Create PackageContents.xml for autoload
$packageXml = @"
<?xml version="1.0" encoding="utf-8"?>
<ApplicationPackage
    SchemaVersion="1.0"
    AutodeskProduct="AutoCAD"
    ProductType="Application"
    Name="AECAgentSidecar"
    Description="AEC Agent Sidecar for MCP integration"
    AppVersion="1.0.0"
    Author="AEC Agent"
    ProductCode="{12345678-1234-1234-1234-123456789012}"
    UpgradeCode="{87654321-4321-4321-4321-210987654321}">
  <CompanyDetails
      Name="AEC Agent"
      Url="https://github.com/yourusername/aec-agent" />
  <Components>
    <RuntimeRequirements
        OS="Win64"
        Platform="AutoCAD" />
    <ComponentEntry
        AppName="AECAgentSidecar"
        ModuleName="./AECAgentSidecar.dll"
        AppDescription="AEC Agent Sidecar"
        LoadOnAutoCADStartup="True">
      <Commands GroupName="AECAgent">
      </Commands>
    </ComponentEntry>
  </Components>
</ApplicationPackage>
"@

Set-Content -Path "$pluginPath\PackageContents.xml" -Value $packageXml

Write-Host "AEC Agent Sidecar installed successfully to $pluginPath"
Write-Host "Please restart AutoCAD to load the plugin"
```

### 9.2 Manual Installation

1. Build the project in Release mode
2. Use ILMerge to merge `AECAgentSidecar.dll` with `Newtonsoft.Json.dll`
3. Copy the merged DLL to AutoCAD's plugin folder
4. Use `NETLOAD` command in AutoCAD to load manually, or create `PackageContents.xml` for autoload

## 10. Testing Strategy

### 10.1 Unit Tests (NUnit)

```csharp
using NUnit.Framework;
using System.Net.Http;
using System.Text;
using Newtonsoft.Json;

[TestFixture]
public class SidecarTests
{
    private HttpClient _client;
    private const string BaseUrl = "http://localhost:20000";
    private const string SessionToken = "test-token-12345";

    [SetUp]
    public void Setup()
    {
        _client = new HttpClient();
        _client.DefaultRequestHeaders.Add("Authorization", SessionToken);
    }

    [Test]
    public async Task TestDrawPolyline()
    {
        var request = new
        {
            command = "draw_polyline",
            parameters = new
            {
                points = new double[][]
                {
                    new double[] { 0, 0 },
                    new double[] { 100, 0 },
                    new double[] { 100, 100 },
                    new double[] { 0, 100 }
                },
                closed = true,
                layer = "0"
            }
        };

        var content = new StringContent(
            JsonConvert.SerializeObject(request),
            Encoding.UTF8,
            "application/json"
        );

        var response = await _client.PostAsync($"{BaseUrl}/cad-command", content);
        Assert.IsTrue(response.IsSuccessStatusCode);

        var responseBody = await response.Content.ReadAsStringAsync();
        var result = JsonConvert.DeserializeObject<dynamic>(responseBody);

        Assert.IsTrue((bool)result.success);
        Assert.IsNotNull(result.data.objectId);
    }

    [Test]
    public async Task TestUnauthorized()
    {
        var badClient = new HttpClient();
        badClient.DefaultRequestHeaders.Add("Authorization", "wrong-token");

        var request = new { command = "audit_layers", parameters = new { } };
        var content = new StringContent(
            JsonConvert.SerializeObject(request),
            Encoding.UTF8,
            "application/json"
        );

        var response = await badClient.PostAsync($"{BaseUrl}/cad-command", content);
        Assert.AreEqual(401, (int)response.StatusCode);
    }
}
```

### 10.2 Integration Tests

```csharp
[TestFixture]
public class IntegrationTests
{
    [Test]
    public async Task TestFullWorkflow()
    {
        // 1. Create layer
        var layerResponse = await CreateLayer("TestLayer", 3);
        Assert.IsTrue(layerResponse.success);

        // 2. Draw polyline on new layer
        var polylineResponse = await DrawPolyline(
            new double[][] { new[] { 0.0, 0.0 }, new[] { 100.0, 100.0 } },
            "TestLayer",
            false
        );
        Assert.IsTrue(polylineResponse.success);

        // 3. Audit layers to verify
        var auditResponse = await AuditLayers();
        Assert.IsTrue(auditResponse.success);
        Assert.IsTrue(auditResponse.data.layers.Any(l => l.name == "TestLayer"));

        // 4. Get entities on layer
        var entitiesResponse = await GetEntities("TestLayer", null, null);
        Assert.IsTrue(entitiesResponse.success);
        Assert.IsTrue(entitiesResponse.data.count > 0);
    }
}
```

## 11. Error Handling & Resilience

### 11.1 Error Codes

| Code | Description |
|------|-------------|
| 4000 | Invalid request format |
| 4001 | Invalid parameters |
| 4002 | Entity not found |
| 4003 | Layer not found |
| 4010 | Document lock timeout |
| 4011 | Transaction failed |
| 5000 | Internal AutoCAD error |
| 5001 | Unexpected exception |

### 11.2 Retry Logic (Client-Side)

The Python FastMCP server should implement retry logic:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
)
async def call_autocad_sidecar(command: str, parameters: dict) -> dict:
    timeout = httpx.Timeout(120.0, connect=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"http://localhost:{sidecar_port}/cad-command",
            json={"command": command, "parameters": parameters},
            headers={"Authorization": session_token}
        )
        response.raise_for_status()
        return response.json()
```

## 12. Performance Optimization

### 12.1 Queue Batching (Optional)

For high-frequency operations, batch multiple commands:

```csharp
private void OnIdle(object sender, EventArgs e)
{
    // Process up to 10 jobs per idle event
    int processed = 0;
    const int maxBatchSize = 10;

    Document doc = Application.DocumentManager.MdiActiveDocument;
    if (doc == null) return;

    using (DocumentLock docLock = doc.LockDocument())
    {
        using (Transaction tr = doc.TransactionManager.StartTransaction())
        {
            try
            {
                while (_jobQueue.TryDequeue(out JobRequest job) && processed < maxBatchSize)
                {
                    try
                    {
                        job.Result = ExecuteCommand(job.Command, job.Parameters, doc, tr);
                        job.IsCompleted = true;
                        processed++;
                    }
                    catch (Exception ex)
                    {
                        job.Error = ex.Message;
                        job.IsCompleted = true;
                    }
                }

                tr.Commit();
            }
            catch
            {
                tr.Abort();
                throw;
            }
        }
    }
}
```

### 12.2 Memory Management

```csharp
// Dispose large objects explicitly
private void CleanupResources()
{
    if (_listener != null)
    {
        _listener.Stop();
        _listener.Close();
        _listener = null;
    }

    while (_jobQueue.TryDequeue(out _)) { }

    GC.Collect();
    GC.WaitForPendingFinalizers();
}
```

## 13. Security Hardening

### 13.1 Token Rotation

```csharp
private static DateTime _tokenExpiry;
private static readonly TimeSpan TokenLifetime = TimeSpan.FromHours(8);

private bool ValidateToken(string token)
{
    if (DateTime.UtcNow > _tokenExpiry)
    {
        // Re-read from environment in case GPO updated it
        _sessionToken = Environment.GetEnvironmentVariable("SESSION_TOKEN");
        _tokenExpiry = DateTime.UtcNow.Add(TokenLifetime);
    }

    return token == _sessionToken;
}
```

### 13.2 Rate Limiting

```csharp
private static readonly Dictionary<string, RateLimiter> _rateLimiters = new();

private bool CheckRateLimit(string endpoint)
{
    if (!_rateLimiters.ContainsKey(endpoint))
    {
        _rateLimiters[endpoint] = new RateLimiter(maxRequests: 100, perSeconds: 60);
    }

    return _rateLimiters[endpoint].Allow();
}

public class RateLimiter
{
    private readonly Queue<DateTime> _requests = new();
    private readonly int _maxRequests;
    private readonly TimeSpan _window;

    public RateLimiter(int maxRequests, int perSeconds)
    {
        _maxRequests = maxRequests;
        _window = TimeSpan.FromSeconds(perSeconds);
    }

    public bool Allow()
    {
        var now = DateTime.UtcNow;

        // Remove old requests outside window
        while (_requests.Count > 0 && (now - _requests.Peek()) > _window)
        {
            _requests.Dequeue();
        }

        if (_requests.Count >= _maxRequests)
        {
            return false;
        }

        _requests.Enqueue(now);
        return true;
    }
}
```

## 14. Logging & Diagnostics

### 14.1 Structured Logging

```csharp
using System.Diagnostics;

public static class Logger
{
    private static readonly string LogPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "AECAgent",
        "autocad-sidecar.log"
    );

    static Logger()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(LogPath));
    }

    public static void Info(string message)
    {
        Log("INFO", message);
    }

    public static void Error(string message, Exception ex = null)
    {
        Log("ERROR", message + (ex != null ? $"\n{ex}" : ""));
    }

    public static void Debug(string message)
    {
        #if DEBUG
        Log("DEBUG", message);
        #endif
    }

    private static void Log(string level, string message)
    {
        string logEntry = $"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss.fff}] [{level}] {message}\n";

        try
        {
            File.AppendAllText(LogPath, logEntry);
        }
        catch
        {
            // Fail silently to avoid crashing AutoCAD
        }

        Debug.WriteLine(logEntry);
    }
}
```

### 14.2 Performance Metrics

```csharp
public class MetricsCollector
{
    private static readonly ConcurrentDictionary<string, PerformanceMetric> _metrics = new();

    public static void RecordExecution(string command, TimeSpan duration, bool success)
    {
        var metric = _metrics.GetOrAdd(command, _ => new PerformanceMetric());
        metric.Record(duration, success);
    }

    public static object GetMetrics()
    {
        return _metrics.ToDictionary(
            kvp => kvp.Key,
            kvp => new
            {
                totalCalls = kvp.Value.TotalCalls,
                successCalls = kvp.Value.SuccessCalls,
                failureCalls = kvp.Value.FailureCalls,
                avgDurationMs = kvp.Value.AverageDuration.TotalMilliseconds,
                maxDurationMs = kvp.Value.MaxDuration.TotalMilliseconds
            }
        );
    }
}

public class PerformanceMetric
{
    public int TotalCalls { get; private set; }
    public int SuccessCalls { get; private set; }
    public int FailureCalls { get; private set; }
    public TimeSpan AverageDuration { get; private set; }
    public TimeSpan MaxDuration { get; private set; }
    private TimeSpan _totalDuration;

    public void Record(TimeSpan duration, bool success)
    {
        TotalCalls++;
        if (success) SuccessCalls++;
        else FailureCalls++;

        _totalDuration += duration;
        AverageDuration = TimeSpan.FromTicks(_totalDuration.Ticks / TotalCalls);

        if (duration > MaxDuration)
            MaxDuration = duration;
    }
}
```

## 15. Common Pitfalls & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `eNotOpenForWrite` | Object opened in wrong mode | Open for `OpenMode.ForWrite` |
| `eOnLockedLayer` | Layer is locked | Check `layer.IsLocked` before modifying |
| `eNoActiveDocument` | No document open | Check `doc != null` before operations |
| `eInvalidInput` | Invalid coordinates/parameters | Validate input before CAD operations |
| Timeout on slow operations | Long-running commands | Increase `SIDECAR_READ_TIMEOUT` to 180s+ |
| Memory leak | Objects not disposed | Use `using` statements for all DB objects |
| Plugin not loading | Missing dependencies | Use ILMerge to bundle Newtonsoft.Json |
| Race condition | Multiple threads | Always queue to main thread via Idle |

## 16. Next Steps

After completing the AutoCAD sidecar:

1. **Test thoroughly** with manual `curl` commands and automated tests
2. **Profile performance** with large drawings (1000+ entities)
3. **Implement Phase 3** - FastMCP server to expose MCP tools
4. **Add more commands** as needed (e.g., annotations, dimensions, blocks)
5. **Version compatibility** - test with AutoCAD 2021, 2022, 2023, 2025

## 17. References

- [AutoCAD .NET Developer's Guide](https://help.autodesk.com/view/OARX/2024/ENU/)
- [AutoCAD .NET API Reference](https://help.autodesk.com/view/OARX/2024/ENU/?guid=OARX-ManagedRefGuide-AcRxMgd_Namespace)
- [IExtensionApplication Interface](https://help.autodesk.com/view/OARX/2024/ENU/?guid=OARX-ManagedRefGuide-Autodesk_AutoCAD_Runtime_IExtensionApplication)
- [Transaction Manager Best Practices](https://through-the-interface.typepad.com/through_the_interface/2007/03/transaction_man.html)

---

**Implementation Checklist:**

- [ ] Create Visual Studio .NET Framework 4.8 Class Library project
- [ ] Add AutoCAD references (acmgd, acdbmgd, accoremgd)
- [ ] Install Newtonsoft.Json via NuGet
- [ ] Implement `IExtensionApplication` with Initialize/Terminate
- [ ] Add HTTP listener with session token validation
- [ ] Implement job queue and OnIdle handler
- [ ] Add command handlers (draw_polyline, audit_layers, etc.)
- [ ] Configure ILMerge in post-build
- [ ] Create PackageContents.xml for autoload
- [ ] Test with AutoCAD NETLOAD command
- [ ] Write unit tests for each command
- [ ] Deploy to target AutoCAD installation
- [ ] Integrate with FastMCP server (Phase 3)
