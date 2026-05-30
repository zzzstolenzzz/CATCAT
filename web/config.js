// Edit these two values after setting up your Hugging Face account.
// Model repo: huggingface.co/YOUR_HF_USERNAME/catcat-model
// Space repo: huggingface.co/YOUR_HF_USERNAME/catcat-backend
window.CATCAT_CONFIG = {
  modelUrl: 'https://huggingface.co/davemost/catcat-model/resolve/main/model.onnx',
  backendUrl: 'https://davemost-catcat-backend.hf.space',
  confThreshold: 0.01,
  iouThreshold: 0.45,
  inputSize: 640,
};
