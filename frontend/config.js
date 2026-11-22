// Backends, tried in order: the Hugging Face Space first, the Vercel function as fallback.
// Both run the same trained ONNX package detector. A backend the user saves under "Backend connection"
// in this browser takes precedence.
window.PVS_CONFIG = {
  apiUrls: ["https://rohitpro1-pack-station-api.hf.space", "https://pack-station-api.vercel.app"],
  examplesBase: "https://raw.githubusercontent.com/Rohitpro1/product-vision-station/main/frontend/",
};
