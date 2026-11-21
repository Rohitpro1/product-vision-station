// Pack Station frontend: renders FrameResult (schema 1.0) from saved examples or a connected backend.
const $ = (s) => document.querySelector(s);
const canvas = $("#canvas");
const ctx = canvas.getContext("2d");
const COLORS = { sku: "#0F7A73", family: "#A86A12", unknown: "#6B7785", conflict: "#B83A4B", ambiguous: "#5A4FA0" };
const GOLD = "#B8901F";
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch { /* storage unavailable */ } },
};
const cfg = window.PVS_CONFIG || {};
let api = { url: store.get("pvs.api") || cfg.apiUrl || "", token: store.get("pvs.token") || "" };

// ------------------------------------------------------------------ backend connection
async function checkBackend() {
  const dot = $("#connDot"), text = $("#connText"), note = $("#dropNote");
  if (!api.url) {
    dot.className = "dot"; text.textContent = "Backend not connected";
    note.textContent = "Connect a backend to analyse your own photos."; return false;
  }
  try {
    const r = await fetch(new URL("/readyz", api.url), { headers: authHeaders() });
    const j = await r.json();
    if (r.ok && j.ready) {
      dot.className = "dot ok"; text.textContent = `Connected · ${j.profile}`;
      note.textContent = "Photos are sent to the connected backend for analysis."; return true;
    }
    dot.className = "dot bad"; text.textContent = "Backend models unavailable";
    note.textContent = j.reason ? `Backend reports: ${j.reason}` : "The backend is up but its models are not ready.";
  } catch {
    dot.className = "dot bad"; text.textContent = "Backend unreachable";
    note.textContent = "Can't reach the backend URL. Check the address and that it allows this site (CORS).";
  }
  return false;
}
function authHeaders() { return api.token ? { Authorization: `Bearer ${api.token}` } : {}; }

$("#apiUrl").value = api.url; $("#apiToken").value = api.token;
$("#connForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  api = { url: $("#apiUrl").value.trim(), token: $("#apiToken").value };
  store.set("pvs.api", api.url); store.set("pvs.token", api.token);
  const ok = await checkBackend();
  $("#connHint").textContent = ok ? "Connected. Choose a photo to analyse it." : $("#dropNote").textContent;
});
$("#disconnect").addEventListener("click", () => {
  api = { url: "", token: "" }; store.set("pvs.api", ""); store.set("pvs.token", "");
  $("#apiUrl").value = ""; $("#apiToken").value = ""; checkBackend();
});

// ------------------------------------------------------------------ drawing
function drawResult(img, res) {
  canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
  const s = Math.max(1, canvas.width / 960);
  for (const inst of res.instances || []) {
    const lvl = inst.identity.level, col = COLORS[lvl] || COLORS.unknown;
    const [x0, y0, x1, y1] = inst.bbox_xyxy_px;
    ctx.lineWidth = 3 * s; ctx.strokeStyle = col; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    const label = `${shortTrack(inst)}  ${lvl}${inst.identity.sku ? " · " + tail(inst.identity.sku) : inst.identity.family ? " · " + tail(inst.identity.family) : ""}`;
    ctx.font = `600 ${15 * s}px Archivo, sans-serif`;
    const w = ctx.measureText(label).width + 12 * s;
    ctx.fillStyle = col; ctx.fillRect(x0, Math.max(0, y0 - 24 * s), w, 22 * s);
    ctx.fillStyle = "#fff"; ctx.fillText(label, x0 + 6 * s, Math.max(16 * s, y0 - 8 * s));
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, L = 0.45 * Math.max(x1 - x0, y1 - y0);
    if (inst.pose.axis.available) {
      const a = rad(inst.pose.axis.value_deg);
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 3 * s; ctx.setLineDash([10 * s, 6 * s]);
      line(cx - L * Math.cos(a), cy - L * Math.sin(a), cx + L * Math.cos(a), cy + L * Math.sin(a)); ctx.setLineDash([]);
    }
    if (inst.pose.heading.available) arrow(cx, cy, rad(inst.pose.heading.value_deg), 0.8 * L, s);
  }
}
function line(a, b, c, d) { ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.stroke(); }
function arrow(cx, cy, h, L, s) {
  const x0 = cx - L * Math.cos(h), y0 = cy - L * Math.sin(h), x1 = cx + L * Math.cos(h), y1 = cy + L * Math.sin(h);
  ctx.strokeStyle = GOLD; ctx.lineWidth = 5 * s; line(x0, y0, x1, y1);
  const k = 18 * s; ctx.fillStyle = GOLD; ctx.beginPath(); ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - k * Math.cos(h - 0.45), y1 - k * Math.sin(h - 0.45));
  ctx.lineTo(x1 - k * Math.cos(h + 0.45), y1 - k * Math.sin(h + 0.45)); ctx.closePath(); ctx.fill();
}
const rad = (d) => (d * Math.PI) / 180;
const tail = (id) => (id || "").split(":").pop();
const shortTrack = (inst) => (inst.track_id || inst.instance_id).split(":").pop();

// dial: image-plane angles, clockwise from +x (right), y down
function dial(inst) {
  const ax = inst.pose.axis, hd = inst.pose.heading;
  const ticks = Array.from({ length: 12 }, (_, i) => {
    const a = rad(i * 30), r1 = i % 3 ? 33 : 30;
    return `<line x1="${42 + r1 * Math.cos(a)}" y1="${42 + r1 * Math.sin(a)}" x2="${42 + 37 * Math.cos(a)}" y2="${42 + 37 * Math.sin(a)}" stroke="#9AA7B3" stroke-width="1.5"/>`;
  }).join("");
  const axisLine = ax.available
    ? `<line class="needle" x1="10" y1="42" x2="74" y2="42" stroke="#17212B" stroke-width="2" stroke-dasharray="5 4" style="transform-origin:42px 42px;transform:rotate(${ax.value_deg}deg)"/>` : "";
  const head = hd.available
    ? `<g class="needle" style="transform-origin:42px 42px;transform:rotate(${hd.value_deg}deg)"><line x1="18" y1="42" x2="66" y2="42" stroke="${GOLD}" stroke-width="4"/><path d="M74 42 L62 35 L62 49 Z" fill="${GOLD}"/></g>` : "";
  return `<svg class="dial" viewBox="0 0 84 84" role="img" aria-label="${dialLabel(inst)}"><circle cx="42" cy="42" r="38" fill="#fff" stroke="#C9D3DB" stroke-width="2"/>${ticks}${axisLine}${head}<circle cx="42" cy="42" r="3" fill="#17212B"/></svg>`;
}
function dialLabel(inst) {
  const a = inst.pose.axis, h = inst.pose.heading;
  return `Axis ${a.available ? a.value_deg.toFixed(0) + " degrees" : "not available"}, heading ${h.available ? h.value_deg.toFixed(0) + " degrees" : "not available"}`;
}
const REASONS = {
  ok: "", endpoint_hidden: "ends not visible", no_mask: "no outline", near_round: "too round for an axis",
  too_small: "too small", axis_unstable: "outline too irregular", below_threshold: "not enough evidence",
  not_in_catalogue: "not in the catalogue", tied_candidates: "variants look identical here",
  conflicting_code: "barcode disagrees with artwork", variant_cue_missing: "variant mark not visible",
  evidence_expired: "evidence too old", unstable_temporal: "direction unstable", held: "held from a moment ago",
  clustered_matches: "too few artwork matches", high_reprojection_error: "artwork match too loose",
};
const why = (r) => REASONS[r] ?? r.replaceAll("_", " ");

function renderList(res) {
  const ol = $("#list");
  const inst = res.instances || [];
  if (!inst.length) {
    ol.innerHTML = `<li class="empty">${res.status === "no_packages" ? "No packs in this photo." :
      res.status === "poor_quality" ? "The photo is too dark, blurred or glary to judge. Try better light." :
      res.status === "model_failed" ? "The backend's models are unavailable, so nothing was analysed." :
      "Nothing to show."}</li>`;
    return;
  }
  ol.innerHTML = inst.map((i) => {
    const id = i.identity, lvl = id.level;
    const name = id.sku ? tail(id.sku) : id.family ? tail(id.family) : "Unrecognised pack";
    const ax = i.pose.axis.available ? `axis ${i.pose.axis.value_deg.toFixed(1)}°` : "no axis";
    const hd = i.pose.heading.available ? `heading ${i.pose.heading.value_deg.toFixed(1)}°` : "no arrow";
    const notes = [why(id.reason), i.pose.heading.available ? "" : why(i.pose.heading.reason)].filter(Boolean);
    const face = i.pose.face !== "unknown" ? ` · ${i.pose.face}` : "";
    return `<li class="inst">${dial(i)}<div><h3><span class="chip ${lvl}">${lvl}</span> ${escapeHtml(name)}</h3>
      <p class="angles">${ax} · ${hd}${face}</p>${notes.length ? `<p class="why">${escapeHtml(notes.join("; "))}</p>` : ""}</div></li>`;
  }).join("");
}
function renderMeta(res, provenance) {
  $("#frameMeta").innerHTML = `<strong>${escapeHtml(provenance)}</strong> — ${res.instances.length} pack(s), status ${res.status}, profile ${res.profile}, rules ${res.fusion_rules_version}, thresholds ${res.thresholds_version}`;
}
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function show(imgSrc, res, provenance) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => { drawResult(img, res); $("#drop").classList.add("hidden"); renderList(res); renderMeta(res, provenance); };
  img.src = imgSrc;
}

// ------------------------------------------------------------------ examples
const EX = (p) => (cfg.examplesBase || "") + p;
async function loadExamples() {
  const list = await fetch(EX("examples/index.json")).then((r) => r.json());
  const grid = $("#examples");
  grid.innerHTML = list.map((e) => `<button class="ex" data-id="${e.id}" aria-pressed="false"><img src="${EX(e.image)}" alt="" loading="lazy" crossorigin="anonymous"><span>${escapeHtml(e.title)}</span></button>`).join("");
  grid.addEventListener("click", async (ev) => {
    const b = ev.target.closest(".ex"); if (!b) return;
    grid.querySelectorAll(".ex").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    const e = list.find((x) => x.id === b.dataset.id);
    const res = await fetch(EX(e.result)).then((r) => r.json());
    show(EX(e.image), res, `Example (synthetic): ${e.title}`);
  });
}

// ------------------------------------------------------------------ uploads
async function analyse(file) {
  if (!api.url) {
    $("#connText").textContent = "Connect a backend first"; $("#conn").open = true; $("#apiUrl").focus(); return;
  }
  const fd = new FormData(); fd.append("file", file);
  $("#frameMeta").textContent = "Analysing…";
  try {
    const r = await fetch(new URL("/v1/detect", api.url), { method: "POST", body: fd, headers: authHeaders() });
    const res = await r.json();
    if (!r.ok && res.error) { $("#frameMeta").textContent = `Not analysed: ${res.detail || res.error}`; return; }
    show(URL.createObjectURL(file), res, `Your photo: ${file.name}`);
  } catch {
    $("#frameMeta").textContent = "Couldn't reach the backend. Check the connection settings.";
  }
}
$("#file").addEventListener("change", (e) => e.target.files[0] && analyse(e.target.files[0]));
const drop = $("#drop");
["dragenter", "dragover"].forEach((t) => $("#stage").addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => $("#stage").addEventListener(t, () => drop.classList.remove("over")));
$("#stage").addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) analyse(f); });

checkBackend();
loadExamples();
