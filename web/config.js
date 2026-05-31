window.CATCAT_CONFIG = {
  // World model (public — used as base for team training)
  modelUrl:   'https://huggingface.co/davemost/catcat-model/resolve/main/model.onnx',

  // Team model (private — what team members actually load)
  teamModelUrl: 'https://huggingface.co/davemost/catcat-team-model/resolve/main/model.onnx',

  backendUrl: 'https://davemost-catcat-backend.hf.space',

  // Shared team key — all team members use the same value here
  teamKey: 'REPLACE_WITH_YOUR_TEAM_KEY',

  confThreshold: 0.05,
  iouThreshold:  0.45,
  inputSize:     640,
};
