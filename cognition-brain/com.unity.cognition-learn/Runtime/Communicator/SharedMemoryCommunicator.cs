using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Unity.CognitionLearn.Actuators;
using Unity.CognitionLearn.Sensors;
using Unity.CognitionLearn.CommunicatorObjects;
using UnityEngine;
using Unity.CognitionLearn.Communicator;

namespace Unity.CognitionLearn
{
    public class SharedMemoryCommunicator : ICommunicator
    {
        private const string HandshakeChannelId = "handshake";
        private const string MainChannelId = "main_comm";
        private const int HandshakeSize = 4096;
        private const int CommSize = 1_048_576; // 1MB

        private CommunicationChannel _handshakeChannel;
        private CommunicationChannel _mainChannel;

        private Dictionary<string, List<(AgentInfo, List<ISensor>)>> _agentData = new Dictionary<string, List<(AgentInfo, List<ISensor>)>>();
        private Dictionary<string, Dictionary<int, ActionBuffers>> _lastActionsReceived = new Dictionary<string, Dictionary<int, ActionBuffers>>();
        private Dictionary<string, ActionSpec> _brainActionSpecs = new Dictionary<string, ActionSpec>();

        // Metadata for display
        private string _displayLabels = "";
        private GUIStyle _labelStyle;

        public event QuitCommandHandler QuitCommandReceived = delegate { };
        public event ResetCommandHandler ResetCommandReceived = delegate { };
 
        // Async Initialization State
        private Task _connectTask;
        private volatile bool _isConnected = false;
        private UnityRLInitParameters _receivedInitParams; // Store params received from handshake

        public SharedMemoryCommunicator()
        {
            // Start the connection process asynchronously to avoid blocking the Main Thread
            // CRITICAL: Only attempt to connect if we are NOT in the Editor or if we are in Play Mode.
            // This prevents TimeoutExceptions during Unity Build processes.
#if UNITY_EDITOR
            if (UnityEditor.EditorApplication.isPlaying)
            {
                _connectTask = Task.Run(ConnectAsync);
            }
#else
            _connectTask = Task.Run(ConnectAsync);
#endif
        }

        private async Task ConnectAsync()
        {
            try
            {
                // Retrieve the base port from Environment Variable (set by Orchestrator)
                string port = Environment.GetEnvironmentVariable("UNITY_BASE_PORT");
                string customPath = Environment.GetEnvironmentVariable("UNITY_SHARED_MEMORY_PATH");
                
                // Fallback to command line arguments if env var is missing
                if (string.IsNullOrEmpty(port))
                {
                    var args = Environment.GetCommandLineArgs();
                    for(int i=0; i<args.Length; i++) {
                        if(args[i] == "--base-port" && i+1 < args.Length) {
                            port = args[i+1];
                        }
                        if(args[i].StartsWith("--shared-memory-path=") && i < args.Length) {
                            customPath = args[i].Split('=')[1];
                        }
                    }
                }

                // ... (rest of the logic)
                
                string handshakeId, mainId;

                if (!string.IsNullOrEmpty(port))
                {
                    Debug.Log($"[SharedMemoryCommunicator] Initializing with Port: {port}");
                    handshakeId = $"{port}_handshake";
                    mainId = port; 
                }
                else
                {
                    Debug.LogWarning("[SharedMemoryCommunicator] No port specified. Using Legacy IDs.");
                    handshakeId = HandshakeChannelId; // "handshake"
                    mainId = MainChannelId;           // "main_comm"
                }

                Debug.Log($"[SharedMemoryCommunicator] Opening Channels (Background)... Handshake: {handshakeId}, Main: {mainId}. Path: {customPath ?? "Default Temp"}");

                // heavy I/O operations (File creation and checking) happen here, off the main thread.
                _handshakeChannel = new CommunicationChannel(handshakeId, HandshakeSize, customPath);
                _mainChannel = new CommunicationChannel(mainId, CommSize, customPath);

                Debug.Log($"[SharedMemoryCommunicator] Channels Opened. Starting Handshake...");
                
                await PerformHandshakeAsync();

                _isConnected = true;
                Debug.Log($"[SharedMemoryCommunicator] Async Connection Established! Ready to train.");
            }
            catch (Exception e)
            {
                Debug.LogError($"[SharedMemoryCommunicator] Async Connection Failed: {e}");
                // We don't rethrow here because it would crash the background thread silently. 
                // The main thread will just stay disconnected.
            }
        }

        private async Task PerformHandshakeAsync()
        {
             // Standard Handshake logic 
             // We can't access initParameters passed to Initialize here directly unless we stored them, 
             // but Initialize might not have been called yet.
             // Actually, usually Initialize provides `name`, `version` etc.
             // For the async case, we might have to wait for Initialize to provide this info, 
             // OR simpler: We use default values for the Initial handshake request if Initialize hasn't run.
             // But wait, Initialize IS called by Academy very early. 
             // Let's assume for this "Launch Fix", we can send generic INIT.

            var sb = new StringBuilder();
            sb.AppendLine("INIT");
            sb.AppendLine($"name:GenericAsyncAgent"); // We use a placeholder since we are async
            sb.AppendLine($"communicationVersion:1.5.0"); 
            sb.AppendLine($"packageVersion:2.0.0");

            byte[] requestData = Encoding.UTF8.GetBytes(sb.ToString());
            
            var sw = System.Diagnostics.Stopwatch.StartNew();
            byte[] responseData = await _handshakeChannel.RequestAsync(requestData);
            sw.Stop();
            Debug.Log($"[SharedMemoryCommunicator] Handshake took {sw.ElapsedMilliseconds} ms");

            string response = Encoding.UTF8.GetString(responseData);

            // Parse response...
            var responseParams = new Dictionary<string, string>();
            foreach (var line in response.Split('\n'))
            {
                if (string.IsNullOrEmpty(line)) continue;
                var parts = line.Split(':');
                if (parts.Length == 2)
                {
                    responseParams[parts[0]] = parts[1];
                }
            }
            
            // Store the seed and caps.
            // Note: We can't update the ALREADY returned InitParams in Initialize.
            // But we can store them to use later if needed.
            // The Agent has already initialized with Seed 0.
             var capabilities = new UnityRLCapabilities(
                baseRlCapabilities: bool.Parse(responseParams.GetValueOrDefault("baseRlCapabilities", "true")),
                concatenatedPngObservations: bool.Parse(responseParams.GetValueOrDefault("concatenatedPngObservations", "true")),
                compressedChannelMapping: bool.Parse(responseParams.GetValueOrDefault("compressedChannelMapping", "true")),
                hybridActions: bool.Parse(responseParams.GetValueOrDefault("hybridActions", "true")),
                trainingAnalytics: bool.Parse(responseParams.GetValueOrDefault("trainingAnalytics", "true")),
                variableLengthObservation: bool.Parse(responseParams.GetValueOrDefault("variableLengthObservation", "true")),
                multiAgentGroups: bool.Parse(responseParams.GetValueOrDefault("multiAgentGroups", "true"))
            );
            
            _receivedInitParams = new UnityRLInitParameters
            {
                seed = int.Parse(responseParams.GetValueOrDefault("seed", "0")),
                pythonCommunicationVersion = responseParams.GetValueOrDefault("communicationVersion", "unknown"),
                pythonLibraryVersion = "unknown",
                numAreas = 1,
                TrainerCapabilities = capabilities
            };
            
            Debug.Log($"[SharedMemoryCommunicator] Handshake Complete. Seed: {_receivedInitParams.seed}");
        }

        public static ICommunicator Create()
        {
            Debug.Log("*********** ICommunicator Create() called - SharedMemoryCommunicator ***********");
            return new SharedMemoryCommunicator();
        }

        public bool Initialize(CommunicatorInitParameters initParameters, out UnityRLInitParameters initParametersOut)
        {
            Debug.Log("SharedMemoryCommunicator.Initialize called (Async Mode). Returning default parameters immediately.");

            // Return immediately with default parameters to prevent blocking
            initParametersOut = new UnityRLInitParameters
            {
                seed = 0, // Default seed, actual seed will be ignored for first reset
                pythonCommunicationVersion = "async_pending",
                pythonLibraryVersion = "uncertain",
                numAreas = 1,
                TrainerCapabilities = new UnityRLCapabilities(true, true, true, true, true, true, true)
            };

            return true;
        }

        public void SubscribeBrain(string name, ActionSpec actionSpec)
        {
            if (!_agentData.ContainsKey(name))
            {
                _agentData[name] = new List<(AgentInfo, List<ISensor>)>();
                _lastActionsReceived[name] = new Dictionary<int, ActionBuffers>();
            }
            
            // Log for debugging dynamic initialization
            if (actionSpec.NumContinuousActions > 0 || actionSpec.NumDiscreteActions > 0)
            {
                Debug.Log($"[SharedMemoryCommunicator] Subscribing Brain: {name} with Actions: Continuous={actionSpec.NumContinuousActions}, Discrete={actionSpec.NumDiscreteActions}");
            }
            
            _brainActionSpecs[name] = actionSpec;
        }

        public void PutObservations(string brainKey, AgentInfo info, List<ISensor> sensors)
        {
            if (!_brainActionSpecs.ContainsKey(brainKey))
            {
                // Only create default if it really doesn't exist yet
                SubscribeBrain(brainKey, new ActionSpec());
            }
            _agentData[brainKey].Add((info, new List<ISensor>(sensors)));
        }

        public void DecideBatch()
        {
            // ASYNC CHECK: If we are not connected yet, simply do nothing.
            // This prevents using the pipes before they are ready.
            if (!_isConnected) 
            {
                // We could log something periodically, but avoiding spam is better.
                // Discard data to prevent memory leak?
                // Actually, clearing agent data is important so it doesn't build up.
                foreach (var list in _agentData.Values)
                {
                    list.Clear(); 
                }
                return; 
            }

            if (_agentData.All(kvp => kvp.Value.Count == 0))
            {
                return;
            }

            var payload = FormatStepData(_agentData);
            byte[] requestData = Encoding.UTF8.GetBytes(payload);
            byte[] responseData = _mainChannel.Request(requestData);
            string response = Encoding.UTF8.GetString(responseData);

            if (response.StartsWith("QUIT"))
            {
                QuitCommandReceived.Invoke();
                return;
            }
            if (response.StartsWith("RESET"))
            {
                ResetCommandReceived.Invoke();
            }

            // Extract LABELS if present
            int labelsIdx = response.IndexOf("\nLABELS\n");
            if (labelsIdx != -1)
            {
                _displayLabels = response.Substring(labelsIdx + 8).Trim();
                // We handle the drawing via a dynamic hook to OnGUI
                EnsureGuiHandler();
            }

            var parsedActions = ParseActionData(response);

            foreach (var behaviorActions in parsedActions)
            {
                var behaviorName = behaviorActions.Key;
                if (_agentData.ContainsKey(behaviorName))
                {
                    var agentsForBehavior = _agentData[behaviorName];
                    for (int i = 0; i < Math.Min(agentsForBehavior.Count, behaviorActions.Value.Count); i++)
                    {
                        var agentId = agentsForBehavior[i].Item1.episodeId;
                        _lastActionsReceived[behaviorName][agentId] = behaviorActions.Value[i];
                    }
                }
            }

            foreach (var list in _agentData.Values)
            {
                list.Clear();
            }
        }

        private string FormatStepData(Dictionary<string, List<(AgentInfo, List<ISensor>)>> data)
        {
            var sb = new StringBuilder();
            sb.AppendLine("STEP");
            foreach (var pair in data)
            {
                if (pair.Value.Count == 0) continue;
                
                sb.AppendLine($"BEHAVIOR:{pair.Key}");
                foreach (var (info, sensors) in pair.Value)
                {
                    sb.AppendLine("AGENT");
                    sb.AppendLine($"id:{info.episodeId}");
                    sb.AppendLine($"reward:{info.reward.ToString(CultureInfo.InvariantCulture)}");
                    sb.AppendLine($"done:{info.done.ToString().ToLower()}");
                    sb.AppendLine($"maxStepReached:{info.maxStepReached.ToString().ToLower()}");
                    
                    // Include Action Shapes for dynamic initialization on the Rust side
                    if (_brainActionSpecs.TryGetValue(pair.Key, out var actionSpec))
                    {
                        var actionShapes = new List<int>();
                        if (actionSpec.NumContinuousActions > 0) actionShapes.Add(actionSpec.NumContinuousActions);
                        if (actionSpec.NumDiscreteActions > 0) actionShapes.AddRange(actionSpec.BranchSizes);
                        
                        if (actionShapes.Count > 0)
                        {
                            sb.AppendLine($"action_shapes:{string.Join(";", actionShapes)}");
                        }
                    }
                    
                // Generic sensor handling matching RpcCommunicator / ML-Agents standard
                // We want to capture ALL vector (1D) observations, including RayPerceptionSensor
                var allObs = new List<float>();
                var shapes = new List<int>();
                var writer = new ObservationWriter(); // Helper to write sensor data
                
                foreach (var sensor in sensors)
                {
                    var spec = sensor.GetObservationSpec();
                    // We only care about 1D sensors (vector observations) for this part
                    if (spec.Shape.Length == 1)
                    {
                        var obsSize = spec.Shape[0];
                        var buffer = new float[obsSize];
                        writer.SetTarget(buffer, spec.Shape, 0);
                        sensor.Write(writer);
                        
                        allObs.AddRange(buffer);
                        shapes.Add(obsSize);
                    }
                }

                if (allObs.Count > 0)
                {
                    var obsString = string.Join(";", allObs.Select(o => o.ToString(CultureInfo.InvariantCulture)));
                    var shapeString = string.Join(";", shapes);
                    
                    sb.AppendLine($"sensor_shapes:{shapeString}");
                    sb.AppendLine($"observations:{obsString}");
                }
                else
                {
                    sb.AppendLine("observations:");
                }
                }
            }
            return sb.ToString();
        }

        private Dictionary<string, List<ActionBuffers>> ParseActionData(string response)
        {
            var allActions = new Dictionary<string, List<ActionBuffers>>();
            if (!response.StartsWith("ACTIONS\n")) return allActions;

            string currentBehavior = null;
            var tempContinuousValues = new Dictionary<string, float[]>();
            var tempDiscreteValues = new Dictionary<string, int[]>();

            foreach (var line in response.Split('\n').Skip(1))
            {
                if (string.IsNullOrEmpty(line)) continue;

                if (line.StartsWith("BEHAVIOR:"))
                {
                    currentBehavior = line.Substring(9);
                    allActions[currentBehavior] = new List<ActionBuffers>();
                    tempContinuousValues[currentBehavior] = null;
                    tempDiscreteValues[currentBehavior] = null;
                }
                else if (line.StartsWith("continuous:") && currentBehavior != null)
                {
                    var values = line.Substring(11)
                        .Split(';')
                        .Where(s => !string.IsNullOrEmpty(s))
                        .Select(s => float.Parse(s, CultureInfo.InvariantCulture))
                        .ToArray();

                    if (_brainActionSpecs.ContainsKey(currentBehavior))
                    {
                        var expectedSpec = _brainActionSpecs[currentBehavior];
                        var expectedContinuousActions = expectedSpec.NumContinuousActions;

                        if (values.Length != expectedContinuousActions && expectedContinuousActions > 0)
                        {
                            // Resize the array to match expected size
                            var adjustedValues = new float[expectedContinuousActions];
                            for (int i = 0; i < expectedContinuousActions; i++)
                            {
                                if (i < values.Length)
                                {
                                    adjustedValues[i] = values[i];
                                }
                                else
                                {
                                    adjustedValues[i] = 0f; // Fill with default value
                                }
                            }
                            tempContinuousValues[currentBehavior] = adjustedValues;
                        }
                        else
                        {
                            tempContinuousValues[currentBehavior] = values;
                        }

                        // Ensure discrete values array exists if expected
                        if (tempDiscreteValues[currentBehavior] == null && expectedSpec.NumDiscreteActions > 0)
                        {
                            tempDiscreteValues[currentBehavior] = new int[expectedSpec.NumDiscreteActions];
                            // Initialize with default values (typically 0)
                            for (int i = 0; i < expectedSpec.NumDiscreteActions; i++)
                            {
                                tempDiscreteValues[currentBehavior][i] = 0;
                            }
                        }
                    }
                    else
                    {
                        tempContinuousValues[currentBehavior] = values;
                    }
                }
                else if (line.StartsWith("discrete:") && currentBehavior != null)
                {
                    var values = line.Substring(9)
                        .Split(';')
                        .Where(s => !string.IsNullOrEmpty(s))
                        .Select(s => int.Parse(s, CultureInfo.InvariantCulture))
                        .ToArray();

                    if (_brainActionSpecs.ContainsKey(currentBehavior))
                    {
                        var expectedSpec = _brainActionSpecs[currentBehavior];
                        var expectedDiscreteActions = expectedSpec.NumDiscreteActions;

                        if (values.Length != expectedDiscreteActions && expectedDiscreteActions > 0)
                        {
                            // Resize the array to match expected size
                            var adjustedValues = new int[expectedDiscreteActions];
                            for (int i = 0; i < expectedDiscreteActions; i++)
                            {
                                if (i < values.Length)
                                {
                                    adjustedValues[i] = values[i];
                                }
                                else
                                {
                                    adjustedValues[i] = 0; // Fill with default value for discrete actions
                                }
                            }
                            tempDiscreteValues[currentBehavior] = adjustedValues;
                        }
                        else
                        {
                            tempDiscreteValues[currentBehavior] = values;
                        }

                        // Ensure continuous values array exists if expected
                        if (tempContinuousValues[currentBehavior] == null && expectedSpec.NumContinuousActions > 0)
                        {
                            tempContinuousValues[currentBehavior] = new float[expectedSpec.NumContinuousActions];
                            // Initialize with default values (typically 0)
                            for (int i = 0; i < expectedSpec.NumContinuousActions; i++)
                            {
                                tempContinuousValues[currentBehavior][i] = 0f;
                            }
                        }
                    }
                    else
                    {
                        tempDiscreteValues[currentBehavior] = values;
                    }
                }
            }

            // Construct the final ActionBuffers
            foreach (var behaviorName in allActions.Keys)
            {
                var continuousValues = tempContinuousValues.ContainsKey(behaviorName) ?
                    tempContinuousValues[behaviorName] : Array.Empty<float>();
                var discreteValues = tempDiscreteValues.ContainsKey(behaviorName) ?
                    tempDiscreteValues[behaviorName] : Array.Empty<int>();

                allActions[behaviorName].Add(new ActionBuffers(continuousValues, discreteValues));
            }

            return allActions;
        }

        public ActionBuffers GetActions(string key, int agentId)
        {
            if (_lastActionsReceived.TryGetValue(key, out var agentActions) &&
                agentActions.TryGetValue(agentId, out var action))
            {
                return action;
            }
            return ActionBuffers.Empty;
        }

        public void Dispose()
        {
            _handshakeChannel?.Dispose();
            _mainChannel?.Dispose();
            if (_guiHandler != null) { UnityEngine.Object.Destroy(_guiHandler); }
        }

        private GameObject _guiHandler;
        private void EnsureGuiHandler()
        {
            if (_guiHandler != null) return;
            _guiHandler = new GameObject("CognitionLearn_GUI");
            UnityEngine.Object.DontDestroyOnLoad(_guiHandler);
            var component = _guiHandler.AddComponent<GuiHelper>();
            component.OnGuiAction = DrawLabels;
        }

        private void DrawLabels()
        {
            if (string.IsNullOrEmpty(_displayLabels)) return;
            
            if (_labelStyle == null)
            {
                _labelStyle = new GUIStyle();
                _labelStyle.fontSize = 18;
                _labelStyle.normal.textColor = Color.yellow;
                _labelStyle.fontStyle = FontStyle.Bold;
            }

            GUI.Box(new Rect(10, 10, 250, 200), "Best Champion Status", GUI.skin.window);
            GUI.Label(new Rect(20, 35, 250, 75), _displayLabels, _labelStyle);
        }

        private class GuiHelper : MonoBehaviour
        {
            public Action OnGuiAction;
            void OnGUI() { OnGuiAction?.Invoke(); }
        }
    }
}

