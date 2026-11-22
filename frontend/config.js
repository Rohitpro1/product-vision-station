// Backends, tried in order: the Hugging Face Space first, the Vercel function as fallback.
// Both serve the same model release (model-assets/release.json). A backend the user saves under "Backend connection"
// in this browser takes precedence.
window.PVS_CONFIG = {
  apiUrls: ["https://rohitpro1-pack-station-api.hf.space", "https://pack-station-api.vercel.app"],
  // Brands the hosted model can name (release v2-retail-brands-keep5).
  displayNames: {
    "cadbury:dairy-milk": "Cadbury Dairy Milk",
    "cadbury:gems": "Cadbury Gems",
    "nestle:kitkat": "Nestlé KitKat",
    "mars:bounty": "Bounty",
    "ferrero:kinder": "Kinder",
  },
  examplesBase: "https://raw.githubusercontent.com/Rohitpro1/product-vision-station/main/frontend/",
};
