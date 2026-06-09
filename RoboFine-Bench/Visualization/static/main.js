// VLM4Robotics Benchmark Viewer - Frontend Logic

let allSamples = [];
let currentSample = null;
let currentCaptionModel = "";
let currentJudgeCombo = null; // {caption_model, judge_tag}
let currentTab = "basic";
let videoElements = [];
let isPlaying = false;
let animFrameId = null;
let isVideoPinned = false;

// Data caches
let captionModels = [];
let atomicCombos = [];
let vqaCache = {};     // sample_id -> data
let atomicCache = {};  // "cap/judge/sid" -> data

// -- Init ------------------------------------------------------------------

async function init() {
  const [samples, models, captionMdls, atomicMdls] = await Promise.all([
    fetch("/api/samples").then(r => r.json()),
    fetch("/api/models").then(r => r.json()),
    fetch("/api/caption_models").then(r => r.json()),
    fetch("/api/atomic_models").then(r => r.json()),
  ]);

  allSamples = samples;
  captionModels = captionMdls;
  atomicCombos = atomicMdls;
  renderSampleList(samples);

  // Populate caption model selector
  const capSel = document.getElementById("caption-model-select");
  capSel.innerHTML = '<option value="">-- Select Caption Model --</option>';
  for (const m of captionMdls) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    capSel.appendChild(opt);
  }
  capSel.addEventListener("change", onCaptionModelChange);

  // Search
  document.getElementById("search-input").addEventListener("input", onSearch);

  // Stats
  document.getElementById("total-samples").textContent = samples.length;
}

// -- Sample list -----------------------------------------------------------

function renderSampleList(samples) {
  const list = document.getElementById("sample-list");
  list.innerHTML = "";
  for (const s of samples) {
    const div = document.createElement("div");
    div.className = "sample-item";
    div.dataset.sid = s.sample_id;
    const desc = s.description || s.dataset || "";
    div.innerHTML = `<div class="sid">${s.sample_id}</div><div class="instr">${esc(desc)}</div>`;
    div.addEventListener("click", () => selectSample(s.sample_id));
    list.appendChild(div);
  }
}

function onSearch(e) {
  const q = e.target.value.toLowerCase();
  const filtered = allSamples.filter(
    s => s.sample_id.toLowerCase().includes(q)
      || (s.description || "").toLowerCase().includes(q)
      || (s.dataset || "").toLowerCase().includes(q)
  );
  renderSampleList(filtered);
}

// -- Select sample ---------------------------------------------------------

async function selectSample(sid) {
  document.querySelectorAll(".sample-item").forEach(el => {
    el.classList.toggle("active", el.dataset.sid === sid);
  });

  const data = await fetch(`/api/samples/${sid}`).then(r => r.json());
  currentSample = data;
  renderDetail(data);
}

// -- Render detail (video + metadata + tabs) --------------------------------

function renderDetail(d) {
  const panel = document.getElementById("right-panel");
  panel.innerHTML = "";

  // Video section
  panel.appendChild(buildVideoSection(d));

  if (isVideoPinned) {
    panel.classList.add("with-pinned-video");
  } else {
    panel.classList.remove("with-pinned-video");
  }

  setTimeout(setupVideoControls, 0);

  // Tab bar
  panel.appendChild(buildTabBar());

  // Tab panels container
  const tabContent = el("div");
  tabContent.id = "tab-content";

  // Tab: Basic Inform
  const tabBasic = el("div", "tab-panel");
  tabBasic.id = "tab-basic";
  tabBasic.style.display = currentTab === "basic" ? "block" : "none";
  tabBasic.appendChild(buildMetadataSection(d));

  // QA Pairs in Basic Inform tab
  const qaSec = sec("QA Pairs");
  qaSec.id = "qa-section";
  if (d.qas && d.qas.length) {
    for (const qa of d.qas) {
      qaSec.appendChild(renderQACard(qa));
    }
  } else {
    qaSec.appendChild(emptyMsg(d.qa_status === "missing" ? "No QA data for this sample" : "QA generation failed"));
  }
  tabBasic.appendChild(qaSec);

  tabContent.appendChild(tabBasic);

  // Tab: VQA
  const tabVqa = el("div", "tab-panel");
  tabVqa.id = "tab-vqa";
  tabVqa.style.display = currentTab === "vqa" ? "block" : "none";
  tabVqa.appendChild(emptyMsg("Loading VQA data..."));
  tabContent.appendChild(tabVqa);

  // Tab: Caption & Atomic Eval (merged)
  const tabCaption = el("div", "tab-panel");
  tabCaption.id = "tab-caption";
  tabCaption.style.display = currentTab === "caption" ? "block" : "none";
  renderCaptionTab(tabCaption, d);
  tabContent.appendChild(tabCaption);

  panel.appendChild(tabContent);

  // Update top bar selector visibility
  updateSelectorVisibility();

  // Load data for active tab
  if (currentTab === "vqa") loadVQAData(d.sample_id);
  if (currentTab === "caption" && currentCaptionModel) loadCaptionData(d.sample_id);
}

// -- Build video section ---------------------------------------------------

function buildVideoSection(d) {
  const videoContainer = el("div");
  videoContainer.id = "video-container";
  if (isVideoPinned) videoContainer.classList.add("pinned");

  // Header
  const videoHeader = el("div", "video-section-header");
  const videoTitle = el("div", "section-title");
  videoTitle.textContent = "Videos";
  videoTitle.style.border = "none";
  videoTitle.style.margin = "0";
  videoTitle.style.padding = "0";
  const pinBtn = el("button");
  pinBtn.id = "pin-btn";
  pinBtn.className = isVideoPinned ? "pinned" : "";
  pinBtn.innerHTML = isVideoPinned ? "📌 Pinned" : "📌 Pin";
  pinBtn.addEventListener("click", toggleVideoPin);
  videoHeader.appendChild(videoTitle);
  videoHeader.appendChild(pinBtn);
  videoContainer.appendChild(videoHeader);

  // Video area
  const videoArea = el("div", "video-area");
  videoArea.id = "video-area";
  videoElements = [];

  for (const view of d.views) {
    const url = d.video_urls[view];
    if (!url) continue;
    const wrap = el("div", "video-wrapper");
    wrap.innerHTML = `<div class="view-label">${view}</div>`;
    const vid = document.createElement("video");
    vid.src = url;
    vid.preload = "metadata";
    vid.muted = true;
    vid.playsInline = true;
    wrap.appendChild(vid);
    videoArea.appendChild(wrap);
    videoElements.push(vid);
  }
  videoContainer.appendChild(videoArea);

  // Controls
  const controls = el("div");
  controls.id = "video-controls";
  controls.innerHTML = `
    <button id="play-btn">Play</button>
    <input type="range" id="progress-bar" min="0" max="1000" value="0">
    <span id="time-display">0:00 / 0:00</span>
  `;
  videoContainer.appendChild(controls);

  return videoContainer;
}

// -- Build metadata section ------------------------------------------------

function buildMetadataSection(d) {
  const metaSec = sec("Metadata");

  const grid = el("div", "meta-grid");
  const fields = [
    ["Dataset", d.dataset],
    ["Robot", d.robot_type],
    ["Duration", d.duration_sec ? d.duration_sec.toFixed(1) + "s" : "N/A"],
    ["Views", (d.views || []).join(", ") || "N/A"],
  ];
  for (const [k, v] of fields) {
    const item = el("div", "meta-item");
    item.innerHTML = `<div class="mk">${k}</div><div class="mv">${v || "N/A"}</div>`;
    grid.appendChild(item);
  }
  metaSec.appendChild(grid);

  // GT and FineGrainedSteps side by side
  if ((d.human_review && d.human_review.length) || (d.gt_fine_steps && d.gt_fine_steps.length)) {
    const columns = el("div", "caption-columns");
    columns.style.marginTop = "16px";

    const gtCol = el("div", "caption-col");
    const gtTitle = el("div", "section-title");
    gtTitle.textContent = "GT (Human Review)";
    gtCol.appendChild(gtTitle);
    if (d.human_review && d.human_review.length) {
      gtCol.appendChild(makeOL(d.human_review));
    } else {
      gtCol.appendChild(emptyMsg("N/A"));
    }
    columns.appendChild(gtCol);

    const fgsCol = el("div", "caption-col");
    const fgsTitle = el("div", "section-title");
    fgsTitle.textContent = "FineGrained Steps";
    fgsCol.appendChild(fgsTitle);
    if (d.gt_fine_steps && d.gt_fine_steps.length) {
      fgsCol.appendChild(makeOL(d.gt_fine_steps));
    } else {
      fgsCol.appendChild(emptyMsg("N/A"));
    }
    columns.appendChild(fgsCol);

    metaSec.appendChild(columns);
  }

  return metaSec;
}

// -- Tab system ------------------------------------------------------------

function buildTabBar() {
  const bar = el("div", "tab-bar");
  bar.id = "tab-bar";
  const tabs = [
    ["basic", "Basic Inform"],
    ["vqa", "VQA"],
    ["caption", "Caption & Atomic Eval"],
  ];
  for (const [id, label] of tabs) {
    const btn = el("button", `tab-btn ${id === currentTab ? "active" : ""}`);
    btn.textContent = label;
    btn.dataset.tab = id;
    btn.addEventListener("click", () => switchTab(id));
    bar.appendChild(btn);
  }
  return bar;
}

function switchTab(tabId) {
  currentTab = tabId;
  document.querySelectorAll(".tab-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === tabId)
  );
  document.querySelectorAll(".tab-panel").forEach(p =>
    p.style.display = p.id === `tab-${tabId}` ? "block" : "none"
  );

  updateSelectorVisibility();

  if (currentSample) {
    if (tabId === "vqa") loadVQAData(currentSample.sample_id);
    if (tabId === "caption" && currentCaptionModel) loadCaptionData(currentSample.sample_id);
  }
}

function updateSelectorVisibility() {
  const capSel = document.getElementById("caption-model-select");
  const statAcc = document.getElementById("stat-accuracy");
  const capBreak = document.getElementById("cap-breakdown");

  var isCap = (currentTab === "caption");
  capSel.style.display = isCap ? "" : "none";
  statAcc.style.display = isCap ? "" : "none";
  capBreak.style.display = isCap ? "" : "none";
}

// -- Tab: Caption ----------------------------------------------------------

function renderCaptionTab(container, d) {
  container.innerHTML = "";

  const columns = el("div", "caption-columns");

  // Left column: GT
  const leftCol = el("div", "caption-col");
  const gtTitle = el("div", "section-title");
  gtTitle.textContent = "Human Review (GT)";
  leftCol.appendChild(gtTitle);
  if (d.human_review && d.human_review.length) {
    leftCol.appendChild(makeOL(d.human_review));
  } else {
    leftCol.appendChild(emptyMsg("No human review available"));
  }
  columns.appendChild(leftCol);

  // Right column: Model Caption
  const rightCol = el("div", "caption-col");
  const capTitle = el("div", "section-title");
  capTitle.textContent = "Model Caption";
  rightCol.appendChild(capTitle);
  rightCol.id = "caption-model-content";
  if (!currentCaptionModel) {
    rightCol.appendChild(emptyMsg("Select a caption model from the top bar"));
  }
  columns.appendChild(rightCol);

  container.appendChild(columns);

  // Atomic Eval section (below caption columns)
  var atomicSec = el("div", "caption-atomic-section");
  atomicSec.id = "atomic-in-caption";
  container.appendChild(atomicSec);
}

async function onCaptionModelChange(e) {
  currentCaptionModel = e.target.value;
  if (currentCaptionModel) {
    // Auto-derive DirectAlign combo from caption model key: "model [mode]"
    var m = currentCaptionModel.match(/^(.+?) \[(easy|hard)\]$/);
    if (m) {
      currentJudgeCombo = { caption_model: m[1], judge_tag: "direct_" + m[2] };
    } else {
      currentJudgeCombo = null;
    }
    loadStats(currentCaptionModel);
    if (currentSample) await loadCaptionData(currentSample.sample_id);
  } else {
    currentJudgeCombo = null;
    clearStats();
    const content = document.getElementById("caption-model-content");
    if (content) {
      content.innerHTML = "";
      content.appendChild(emptyMsg("Select a caption model from the top bar"));
    }
  }
}

async function loadCaptionData(sid) {
  if (!currentCaptionModel) return;

  const [capRes, judgeRes] = await Promise.all([
    fetch(`/api/caption/${currentCaptionModel}/${sid}`).then(r => r.json()),
    fetch(`/api/judge/${currentCaptionModel}/${sid}`).then(r => r.json()).catch(() => ({error: true})),
  ]);

  const content = document.getElementById("caption-model-content");
  if (content) {
    while (content.children.length > 1) content.removeChild(content.lastChild);

    if (capRes.error) {
      content.appendChild(emptyMsg("No caption data for this model"));
    } else {
      // Metadata badges
      const metaRow = el("div", "caption-meta-row");
      const badges = [];
      if (capRes.call_success != null) {
        badges.push(`<span class="caption-badge ${capRes.call_success ? "badge-ok" : "badge-fail"}">${capRes.call_success ? "Success" : "Failed"}</span>`);
      }
      if (capRes.num_views != null) badges.push(`<span class="caption-badge">Views: ${capRes.num_views}</span>`);
      if (capRes.num_frames != null) badges.push(`<span class="caption-badge">Frames: ${capRes.num_frames}</span>`);
      if (capRes.elapsed_sec != null) badges.push(`<span class="caption-badge">${capRes.elapsed_sec.toFixed(1)}s</span>`);
      if (capRes.token_usage) {
        const tu = capRes.token_usage;
        badges.push(`<span class="caption-badge">Tokens: ${tu.total_tokens || "?"}</span>`);
      }
      metaRow.innerHTML = badges.join("");
      content.appendChild(metaRow);

      if (capRes.refinedInstruction && !currentCaptionModel.includes("[hard]")) {
        const ri = el("div", "caption-instruction");
        ri.innerHTML = `<strong>Instruction:</strong> ${esc(capRes.refinedInstruction)}`;
        content.appendChild(ri);
      }
      if (capRes.fineGrainedSteps && capRes.fineGrainedSteps.length) {
        const label = el("div");
        label.innerHTML = '<strong style="color:var(--text-secondary)">Caption Steps:</strong>';
        content.appendChild(label);
        content.appendChild(makeOL(capRes.fineGrainedSteps));
      } else {
        content.appendChild(emptyMsg("No caption steps returned"));
      }
    }
  }

  // Judge overlay on QA (in Basic Inform tab)
  let judgeLookup = {};
  if (!judgeRes.error && judgeRes.qa_results) {
    for (const qr of judgeRes.qa_results) {
      judgeLookup[qr.question_id] = qr;
    }
  }

  const qaSec = document.getElementById("qa-section");
  if (qaSec && Object.keys(judgeLookup).length > 0) {
    while (qaSec.children.length > 1) qaSec.removeChild(qaSec.lastChild);
    if (currentSample.qas && currentSample.qas.length) {
      for (const qa of currentSample.qas) {
        qaSec.appendChild(renderQACard(qa, judgeLookup[qa.question_id]));
      }
    }
  }

  // Also load Atomic Eval data into the same tab
  if (currentJudgeCombo) {
    loadAtomicData(sid);
  } else {
    var atomicPanel = document.getElementById("atomic-in-caption");
    if (atomicPanel) {
      atomicPanel.innerHTML = "";
      atomicPanel.appendChild(emptyMsg("Select an Atomic Eval combo from the top bar to view alignment results"));
    }
  }
}

// -- Tab: VQA --------------------------------------------------------------

async function loadVQAData(sid) {
  const panel = document.getElementById("tab-vqa");
  if (!panel) return;

  if (vqaCache[sid]) {
    renderVQATab(panel, vqaCache[sid]);
    return;
  }

  panel.innerHTML = "";
  panel.appendChild(emptyMsg("Loading VQA data..."));

  const data = await fetch(`/api/vqa/${sid}`).then(r => r.json());
  vqaCache[sid] = data;
  renderVQATab(panel, data);
}

function renderVQATab(container, data) {
  container.innerHTML = "";

  const models = data.vqa_models || [];
  const byCap = data.by_capability || {};
  const caps = Object.keys(byCap).sort();

  if (caps.length === 0) {
    container.appendChild(emptyMsg("No VQA data for this sample"));
    return;
  }

  // Short model names for headers
  const shortName = (m) => {
    return m.replace(/_vqa_result$/, "")
            .replace(/^doubao_doubao-/, "")
            .replace(/^openai_/, "")
            .replace(/^vertex_ai_/, "")
            .replace(/-preview/, "")
            .replace(/_round/, " R");
  };

  const wrapper = el("div", "vqa-container");
  const table = el("table", "vqa-table");

  // Header
  const thead = document.createElement("thead");
  const hrow = document.createElement("tr");
  hrow.innerHTML = `<th class="vqa-th-q">Question</th><th>GT</th>`;
  for (const m of models) {
    const th = document.createElement("th");
    th.className = "vqa-th-model";
    th.textContent = shortName(m);
    th.title = m;
    hrow.appendChild(th);
  }
  thead.appendChild(hrow);
  table.appendChild(thead);

  // Body grouped by capability
  const tbody = document.createElement("tbody");
  for (const cap of caps) {
    // Capability group header
    const grow = document.createElement("tr");
    grow.className = "vqa-cap-group";
    grow.innerHTML = `<td colspan="${2 + models.length}">${cap}</td>`;
    tbody.appendChild(grow);

    for (const q of byCap[cap]) {
      const row = document.createElement("tr");

      // Question cell (includes options if available)
      const qCell = document.createElement("td");
      qCell.className = "vqa-td-q";
      let qHtml = `<div class="vqa-q-text">${esc(q.question)}</div>`;
      if (q.options && q.options.length > 0) {
        const labels = "ABCDEFGH";
        qHtml += '<div class="vqa-options">';
        for (let i = 0; i < q.options.length; i++) {
          const letter = labels[i] || String(i + 1);
          const gtLower = (q.gt_answer || "").toLowerCase();
          const optLower = q.options[i].toLowerCase();
          const isGt = gtLower === optLower
            || gtLower === `${letter.toLowerCase()}. ${optLower}`
            || gtLower === letter.toLowerCase();
          qHtml += `<span class="vqa-opt ${isGt ? "vqa-opt-gt" : ""}">${letter}. ${esc(q.options[i])}</span>`;
        }
        qHtml += '</div>';
      }
      qCell.innerHTML = qHtml;
      row.appendChild(qCell);

      // GT answer
      const gtCell = document.createElement("td");
      gtCell.className = "vqa-td-gt";
      gtCell.textContent = q.gt_answer;
      row.appendChild(gtCell);

      // Model answers
      for (const m of models) {
        const td = document.createElement("td");
        const mdata = q.models[m];
        if (mdata) {
          td.textContent = mdata.answer;
          td.className = mdata.correct ? "vqa-correct" : "vqa-incorrect";
          td.title = mdata.correct ? "Correct" : "Incorrect";
        } else {
          td.textContent = "-";
          td.className = "vqa-missing";
        }
        row.appendChild(td);
      }

      tbody.appendChild(row);
    }
  }
  table.appendChild(tbody);
  wrapper.appendChild(table);
  container.appendChild(wrapper);
}

// -- Tab: AtomicEval -------------------------------------------------------

async function onJudgeComboChange(e) {
  var val = e.target.value;
  if (!val) {
    currentJudgeCombo = null;
    var panel = document.getElementById("atomic-in-caption");
    if (panel) {
      panel.innerHTML = "";
      panel.appendChild(emptyMsg("Select an Atomic Eval combo to view alignment results"));
    }
    return;
  }
  var parts = val.split("||");
  currentJudgeCombo = { caption_model: parts[0], judge_tag: parts[1] };
  if (currentSample) loadAtomicData(currentSample.sample_id);
}

async function loadAtomicData(sid) {
  var panel = document.getElementById("atomic-in-caption");
  if (!panel) return;

  if (!currentJudgeCombo) {
    panel.innerHTML = "";
    panel.appendChild(emptyMsg("Select an Atomic Eval combo from the top bar to view alignment results"));
    return;
  }

  var caption_model = currentJudgeCombo.caption_model;
  var judge_tag = currentJudgeCombo.judge_tag;
  var cacheKey = caption_model + "/" + judge_tag + "/" + sid;

  if (atomicCache[cacheKey]) {
    renderAtomicTab(panel, atomicCache[cacheKey]);
    return;
  }

  panel.innerHTML = "";
  panel.appendChild(emptyMsg("Loading AtomicEval data..."));

  var data = await fetch("/api/atomic/" + caption_model + "/" + judge_tag + "/" + sid).then(function(r) { return r.json(); });
  if (data.error) {
    panel.innerHTML = "";
    panel.appendChild(emptyMsg("No AtomicEval data: " + data.error));
    return;
  }
  atomicCache[cacheKey] = data;
  renderAtomicTab(panel, data);
}

function renderAtomicTab(container, data) {
  container.innerHTML = "";
  const scores = data.scores;
  const alignment = data.alignment;
  const isDA = alignment && alignment.type === "direct_align";

  // Header
  const header = el("div", "atomic-header");
  if (isDA) {
    header.innerHTML = '<span class="atomic-model-tag">' + esc(data.caption_model) + '</span> <span class="da-badge">Direct Alignment</span>';
  } else {
    header.innerHTML = '<span class="atomic-model-tag">' + esc(data.caption_model) + '</span> judged by <span class="atomic-judge-tag">' + esc(data.judge_tag) + '</span>';
  }
  container.appendChild(header);

  // Overall scores
  if (scores) {
    const overallDiv = el("div", "atomic-overall");
    const cs = scores.caption_score, con = scores.consistency_score;
    const cov = scores.weighted_coverage, ah = scores.weighted_anti_hallucination;
    overallDiv.innerHTML =
      '<span class="atomic-score-pill">Caption Score: <b>' + fmtScore(cs) + '</b></span>' +
      '<span class="atomic-score-pill">Consistency: <b>' + fmtScore(con) + '</b></span>' +
      '<span class="atomic-score-pill">Coverage: <b>' + fmtScore(cov) + '</b></span>' +
      '<span class="atomic-score-pill">Anti-Halluc.: <b>' + fmtScore(ah) + '</b></span>';
    container.appendChild(overallDiv);
  }

  // Verdict distribution bar (DirectAlign only)
  if (isDA && scores && scores.capability_scores) {
    container.appendChild(buildVerdictBar(scores.capability_scores));
  }

  // Summary table
  if (scores && scores.capability_scores) {
    container.appendChild(buildAtomicSummaryTable(scores.capability_scores));
  }

  // Per-capability expandable sections
  if (isDA && alignment.by_capability) {
    renderDirectAlignDetails(container, alignment, scores);
  } else if (alignment && alignment.capability_results) {
    const capSec = el("div", "atomic-capabilities");
    const capTitle = el("div", "section-title");
    capTitle.textContent = "Per-Capability Alignment Details";
    capSec.appendChild(capTitle);
    for (const capResult of alignment.capability_results) {
      capSec.appendChild(buildAtomicCapability(capResult, scores));
    }
    container.appendChild(capSec);
  }
}

// -- DirectAlign rendering helpers ------------------------------------------

var DA_CAP_ORDER = [
  "action_sequence", "active_actor", "target_object",
  "initial_configuration", "final_configuration",
  "contact_and_approach", "trajectory_and_orientation",
  "object_interaction", "failure_and_recovery", "body_motion"
];

function buildVerdictBar(capScores) {
  var tM = 0, tP = 0, tX = 0, tO = 0, tH = 0;
  for (var i = 0; i < capScores.length; i++) {
    var cs = capScores[i];
    tM += cs.M || 0; tP += cs.P || 0; tX += cs.X || 0;
    tO += cs.O || 0; tH += cs.H || 0;
  }
  var total = tM + tP + tX + tO + tH;
  if (total === 0) return el("div");
  var pM = (tM / total * 100).toFixed(1);
  var pP = (tP / total * 100).toFixed(1);
  var pX = (tX / total * 100).toFixed(1);
  var pO = (tO / total * 100).toFixed(1);
  var pH = (tH / total * 100).toFixed(1);
  var w = el("div", "verdict-bar-wrapper");
  var barHtml = '<div class="verdict-bar">';
  if (tM > 0) barHtml += '<div class="vb-seg vb-match" style="width:' + pM + '%" title="Match: ' + tM + ' (' + pM + '%)"></div>';
  if (tP > 0) barHtml += '<div class="vb-seg vb-partial" style="width:' + pP + '%" title="Partial: ' + tP + ' (' + pP + '%)"></div>';
  if (tX > 0) barHtml += '<div class="vb-seg vb-contradiction" style="width:' + pX + '%" title="Contradiction: ' + tX + ' (' + pX + '%)"></div>';
  if (tO > 0) barHtml += '<div class="vb-seg vb-omission" style="width:' + pO + '%" title="Omission: ' + tO + ' (' + pO + '%)"></div>';
  if (tH > 0) barHtml += '<div class="vb-seg vb-hallucination" style="width:' + pH + '%" title="Hallucination: ' + tH + ' (' + pH + '%)"></div>';
  barHtml += '</div>';
  barHtml += '<div class="verdict-bar-legend">' +
    '<span class="vb-legend-item"><span class="vb-dot vb-match"></span>Match ' + tM + '</span>' +
    '<span class="vb-legend-item"><span class="vb-dot vb-partial"></span>Partial ' + tP + '</span>' +
    '<span class="vb-legend-item"><span class="vb-dot vb-contradiction"></span>Contradiction ' + tX + '</span>' +
    '<span class="vb-legend-item"><span class="vb-dot vb-omission"></span>Omission ' + tO + '</span>' +
    '<span class="vb-legend-item"><span class="vb-dot vb-hallucination"></span>Hallucination ' + tH + '</span>' +
    '</div>';
  w.innerHTML = barHtml;
  return w;
}

function renderDirectAlignDetails(container, alignment, scores) {
  var capSec = el("div", "atomic-capabilities");
  var capTitle = el("div", "section-title");
  capTitle.textContent = "Per-Capability Alignment Details";
  capSec.appendChild(capTitle);

  for (var i = 0; i < DA_CAP_ORDER.length; i++) {
    var cap = DA_CAP_ORDER[i];
    var evals = alignment.by_capability[cap] || [];
    var cs = null;
    if (scores && scores.capability_scores) {
      for (var j = 0; j < scores.capability_scores.length; j++) {
        if (scores.capability_scores[j].capability === cap) { cs = scores.capability_scores[j]; break; }
      }
    }
    capSec.appendChild(buildDACapSection(cap, evals, cs));
  }

  var halluc = alignment.hallucinated_actions || [];
  if (halluc.length > 0) {
    capSec.appendChild(buildDAHallucSection(halluc));
  }
  container.appendChild(capSec);
}

function buildDACapSection(cap, evals, cs) {
  var section = el("div", "atomic-capability");
  var nM = 0, nP = 0, nX = 0, nO = 0;
  for (var i = 0; i < evals.length; i++) {
    var v = evals[i].label || evals[i].verdict || "";
    if (v === "match") nM++;
    else if (v === "partial") nP++;
    else if (v === "contradiction") nX++;
    else if (v === "omission") nO++;
  }
  var isEmpty = evals.length === 0;

  var header = el("div", "atomic-cap-header");
  var badgeHtml = isEmpty ? "(empty)" : "";
  if (!isEmpty) {
    var parts = [];
    if (nM > 0) parts.push('<span class="da-cnt da-cnt-match">M:' + nM + '</span>');
    if (nP > 0) parts.push('<span class="da-cnt da-cnt-partial">P:' + nP + '</span>');
    if (nX > 0) parts.push('<span class="da-cnt da-cnt-contradiction">X:' + nX + '</span>');
    if (nO > 0) parts.push('<span class="da-cnt da-cnt-omission">O:' + nO + '</span>');
    badgeHtml = parts.join(" ");
  }
  header.innerHTML = '<span class="atomic-cap-title">' + cap + '</span>' +
    '<span class="atomic-cap-counts">' + badgeHtml + '</span>' +
    '<span class="atomic-cap-arrow">\u25B6</span>';
  header.addEventListener("click", function() {
    var body = section.querySelector(".atomic-cap-body");
    var arrow = header.querySelector(".atomic-cap-arrow");
    if (body.style.display === "none" || !body.style.display) {
      body.style.display = "block"; arrow.textContent = "\u25BC";
    } else {
      body.style.display = "none"; arrow.textContent = "\u25B6";
    }
  });
  section.appendChild(header);

  var body = el("div", "atomic-cap-body");
  body.style.display = "none";
  if (isEmpty) {
    body.appendChild(emptyMsg("No GT facts for this capability"));
  } else {
    for (var i = 0; i < evals.length; i++) {
      body.appendChild(buildDAFactCard(evals[i]));
    }
  }
  section.appendChild(body);
  return section;
}

function buildDAFactCard(ev) {
  var verdict = ev.label || ev.verdict || "unknown";
  var card = el("div", "da-fact-card da-v-" + verdict);

  // Header row: verdict badge + fact_id + slot
  var hdr = el("div", "da-fact-header");
  var vlabel = el("span", "da-vlabel da-vlabel-" + verdict);
  vlabel.textContent = verdict.toUpperCase();
  hdr.appendChild(vlabel);
  if (ev.fact_id) {
    var fid = el("span", "da-fid");
    fid.textContent = ev.fact_id;
    hdr.appendChild(fid);
  }
  if (ev.slot) {
    var stag = el("span", "da-slot");
    stag.textContent = ev.slot;
    hdr.appendChild(stag);
  }
  card.appendChild(hdr);

  // GT value line
  if (ev.gt_value) {
    var gtDiv = el("div", "da-gt-val");
    gtDiv.innerHTML = '<span class="da-lbl">GT:</span> ' + esc(ev.gt_value);
    card.appendChild(gtDiv);
  }

  // Full fact text
  if (ev.fact_text) {
    var ftDiv = el("div", "da-fact-text");
    ftDiv.textContent = ev.fact_text;
    card.appendChild(ftDiv);
  }

  // Caption evidence
  var evidence = ev.caption_evidence || ev.matched_caption_segment;
  if (evidence) {
    var evDiv = el("div", "da-evidence");
    evDiv.innerHTML = '<span class="da-lbl">Evidence:</span> \u201C' + esc(evidence) + '\u201D';
    card.appendChild(evDiv);
  }

  // Note
  var note = ev.note || ev.reason;
  if (note) {
    var nDiv = el("div", "da-note");
    nDiv.textContent = note;
    card.appendChild(nDiv);
  }
  return card;
}

function buildDAHallucSection(halluc) {
  var section = el("div", "atomic-capability da-halluc-section");
  var header = el("div", "atomic-cap-header");
  header.innerHTML = '<span class="atomic-cap-title" style="color:var(--red)">Hallucinated Actions</span>' +
    '<span class="da-cnt da-cnt-halluc">H:' + halluc.length + '</span>' +
    '<span class="atomic-cap-arrow">\u25B6</span>';
  header.addEventListener("click", function() {
    var body = section.querySelector(".atomic-cap-body");
    var arrow = header.querySelector(".atomic-cap-arrow");
    if (body.style.display === "none" || !body.style.display) {
      body.style.display = "block"; arrow.textContent = "\u25BC";
    } else {
      body.style.display = "none"; arrow.textContent = "\u25B6";
    }
  });
  section.appendChild(header);

  var body = el("div", "atomic-cap-body");
  body.style.display = "none";
  for (var i = 0; i < halluc.length; i++) {
    var h = halluc[i];
    var card = el("div", "da-fact-card da-v-hallucination");
    var desc = h.description || h.action || "N/A";
    var step = h.caption_step || h.caption_segment || "";
    var note = h.note || h.reason || "";
    var html = '<div class="da-fact-header"><span class="da-vlabel da-vlabel-hallucination">HALLUCINATION</span>';
    if (step) html += '<span class="da-fid">' + esc(step) + '</span>';
    html += '</div>';
    html += '<div class="da-halluc-desc">' + esc(desc) + '</div>';
    if (note) html += '<div class="da-note">' + esc(note) + '</div>';
    card.innerHTML = html;
    body.appendChild(card);
  }
  section.appendChild(body);
  return section;
}

function buildAtomicSummaryTable(capScores) {
  const wrapper = el("div", "atomic-summary-wrapper");
  const table = el("table", "atomic-summary-table");

  const thead = document.createElement("thead");
  thead.innerHTML = `<tr>
    <th>Capability</th><th>G</th><th>C</th><th>M</th><th>P</th><th>X</th><th>O</th><th>H</th>
    <th>Consist.</th><th>Coverage</th><th>Anti-H.</th>
  </tr>`;
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const cs of capScores) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="atomic-cap-name">${cs.capability}</td>
      <td>${cs.G ?? "-"}</td><td>${cs.C ?? "-"}</td>
      <td>${cs.M ?? "-"}</td><td>${cs.P ?? "-"}</td><td>${cs.X ?? "-"}</td>
      <td class="${(cs.O > 0) ? "score-warn" : ""}">${cs.O ?? "-"}</td>
      <td class="${(cs.H > 0) ? "score-bad" : ""}">${cs.H ?? "-"}</td>
      <td class="${scoreClass(cs.consistency)}">${fmtScore(cs.consistency)}</td>
      <td class="${scoreClass(cs.coverage)}">${fmtScore(cs.coverage)}</td>
      <td class="${scoreClass(cs.anti_hallucination)}">${fmtScore(cs.anti_hallucination)}</td>
    `;
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  wrapper.appendChild(table);
  return wrapper;
}

function buildAtomicCapability(capResult, scores) {
  const cap = capResult.capability;
  const gtFacts = capResult.gt_atomic_facts || [];
  const capFacts = capResult.caption_atomic_facts || [];
  const alignments = capResult.alignments || [];
  const omissions = capResult.omissions || [];
  const hallucinations = capResult.hallucinations || [];

  const isEmpty = gtFacts.length === 0 && capFacts.length === 0;

  const section = el("div", "atomic-capability");

  // Header (clickable to expand)
  const header = el("div", "atomic-cap-header");
  const countInfo = isEmpty
    ? "(empty)"
    : `GT:${gtFacts.length} Cap:${capFacts.length} M:${alignments.length} O:${omissions.length} H:${hallucinations.length}`;
  header.innerHTML = `
    <span class="atomic-cap-title">${cap}</span>
    <span class="atomic-cap-counts">${countInfo}</span>
    <span class="atomic-cap-arrow">▶</span>
  `;
  header.addEventListener("click", () => {
    const body = section.querySelector(".atomic-cap-body");
    const arrow = header.querySelector(".atomic-cap-arrow");
    if (body.style.display === "none" || !body.style.display) {
      body.style.display = "block";
      arrow.textContent = "▼";
    } else {
      body.style.display = "none";
      arrow.textContent = "▶";
    }
  });
  section.appendChild(header);

  // Body (collapsed by default)
  const body = el("div", "atomic-cap-body");
  body.style.display = "none";

  if (isEmpty) {
    body.appendChild(emptyMsg("No atomic facts for this capability"));
  } else {
    // Build lookup for alignment status
    const gtStatus = {};   // gt_fact_id -> {label, caption_fact_id, note}
    const capStatus = {};  // cap_fact_id -> {label, gt_fact_id, note}

    for (const a of alignments) {
      gtStatus[a.gt_fact_id] = { label: a.label || "match", caption_fact_id: a.caption_fact_id, note: a.note };
      capStatus[a.caption_fact_id] = { label: a.label || "match", gt_fact_id: a.gt_fact_id, note: a.note };
    }
    for (const o of omissions) {
      gtStatus[o.gt_fact_id] = { label: "omission", note: o.note };
    }
    for (const h of hallucinations) {
      capStatus[h.caption_fact_id] = { label: "hallucination", note: h.reason || h.note };
    }

    const columns = el("div", "fact-columns");

    // GT facts column
    const gtCol = el("div", "fact-col");
    const gtHeader = el("div", "fact-col-header");
    gtHeader.textContent = `GT Facts (${gtFacts.length})`;
    gtCol.appendChild(gtHeader);
    for (const fact of gtFacts) {
      const item = el("div", "fact-item");
      const status = gtStatus[fact.fact_id];
      if (status) {
        item.classList.add(`fact-${status.label}`);
        item.innerHTML = `
          <div class="fact-text">${esc(fact.fact_text)}</div>
          <div class="fact-meta">${status.label}${status.note ? ": " + esc(status.note) : ""}</div>
        `;
      } else {
        item.classList.add("fact-unmatched");
        item.innerHTML = `<div class="fact-text">${esc(fact.fact_text)}</div>`;
      }
      gtCol.appendChild(item);
    }
    columns.appendChild(gtCol);

    // Caption facts column
    const capCol = el("div", "fact-col");
    const capHeader = el("div", "fact-col-header");
    capHeader.textContent = `Caption Facts (${capFacts.length})`;
    capCol.appendChild(capHeader);
    for (const fact of capFacts) {
      const item = el("div", "fact-item");
      const status = capStatus[fact.fact_id];
      if (status) {
        item.classList.add(`fact-${status.label}`);
        item.innerHTML = `
          <div class="fact-text">${esc(fact.fact_text)}</div>
          <div class="fact-meta">${status.label}${status.note ? ": " + esc(status.note) : ""}</div>
        `;
      } else {
        item.classList.add("fact-unmatched");
        item.innerHTML = `<div class="fact-text">${esc(fact.fact_text)}</div>`;
      }
      capCol.appendChild(item);
    }
    columns.appendChild(capCol);

    body.appendChild(columns);
  }

  section.appendChild(body);
  return section;
}

// -- Video controls --------------------------------------------------------

function setupVideoControls() {
  const btn = document.getElementById("play-btn");
  const bar = document.getElementById("progress-bar");
  const timeDisp = document.getElementById("time-display");
  if (!btn || !bar) return;

  isPlaying = false;

  btn.addEventListener("click", () => {
    if (isPlaying) {
      pauseAll();
      btn.textContent = "Play";
    } else {
      playAll();
      btn.textContent = "Pause";
    }
  });

  bar.addEventListener("input", () => {
    const frac = bar.value / 1000;
    for (const v of videoElements) {
      if (v.duration) v.currentTime = frac * v.duration;
    }
  });

  function updateProgress() {
    const v = videoElements[0];
    if (v && v.duration) {
      const frac = v.currentTime / v.duration;
      bar.value = Math.round(frac * 1000);
      timeDisp.textContent = `${fmtTime(v.currentTime)} / ${fmtTime(v.duration)}`;
    }
    animFrameId = requestAnimationFrame(updateProgress);
  }
  if (animFrameId) cancelAnimationFrame(animFrameId);
  animFrameId = requestAnimationFrame(updateProgress);

  if (videoElements[0]) {
    videoElements[0].addEventListener("ended", () => {
      isPlaying = false;
      btn.textContent = "Play";
    });
  }
}

function playAll() {
  isPlaying = true;
  for (const v of videoElements) v.play();
}

function pauseAll() {
  isPlaying = false;
  for (const v of videoElements) v.pause();
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

// -- Pin/Unpin video -------------------------------------------------------

function toggleVideoPin() {
  isVideoPinned = !isVideoPinned;
  const container = document.getElementById("video-container");
  const panel = document.getElementById("right-panel");
  const btn = document.getElementById("pin-btn");

  if (isVideoPinned) {
    container.classList.add("pinned");
    panel.classList.add("with-pinned-video");
    btn.classList.add("pinned");
    btn.innerHTML = "📌 Pinned";
  } else {
    container.classList.remove("pinned");
    panel.classList.remove("with-pinned-video");
    btn.classList.remove("pinned");
    btn.innerHTML = "📌 Pin";
  }
}

// -- Stats (for Caption tab) -----------------------------------------------

async function loadStats(model) {
  const stats = await fetch(`/api/stats/${model}`).then(r => r.json());
  if (stats.error) {
    clearStats();
    return;
  }

  document.getElementById("model-accuracy").textContent =
    `${(stats.accuracy * 100).toFixed(1)}% (${stats.total_correct}/${stats.total_questions})`;

  const breakdown = document.getElementById("cap-breakdown");
  breakdown.innerHTML = "";
  for (const cap of stats.capability_breakdown) {
    const pill = el("span", "cap-pill");
    pill.innerHTML = `${cap.capability}<span class="cap-acc">${(cap.accuracy * 100).toFixed(0)}%</span>`;
    breakdown.appendChild(pill);
  }
}

function clearStats() {
  document.getElementById("model-accuracy").textContent = "N/A";
  document.getElementById("cap-breakdown").innerHTML = "";
}

// -- QA card ---------------------------------------------------------------

function renderQACard(qa, judge) {
  const card = el("div", "qa-card");
  if (judge) {
    card.classList.add(judge.correct ? "correct" : "incorrect");
  }

  if (judge) {
    const badge = el("div", "judge-badge");
    badge.textContent = judge.correct ? "\u2713" : "\u2717";
    badge.style.color = judge.correct ? "var(--green)" : "var(--red)";
    card.appendChild(badge);
  }

  const tags = el("div", "qa-tags");
  const modeClass = qa.mode === "gt_only" ? "tag-mode-gt" : qa.mode === "conflict" ? "tag-mode-conflict" : "tag-mode-other";
  tags.innerHTML = `<span class="tag tag-cap">${qa.capability || ""}</span>`;
  if (qa.mode) {
    tags.innerHTML += `<span class="tag ${modeClass}">${qa.mode}</span>`;
  }
  card.appendChild(tags);

  const q = el("div", "qa-question");
  q.textContent = qa.question;
  card.appendChild(q);

  const fields = [
    ["Answer", qa.answer],
    ["Reference", qa.reference_text],
    ["Difference", qa.difference_summary],
  ];
  for (const [label, val] of fields) {
    if (!val) continue;
    const f = el("div", "qa-field");
    f.innerHTML = `<span class="fl">${label}:</span> ${esc(val)}`;
    card.appendChild(f);
  }

  if (judge && judge.judge_explanation) {
    const exp = el("div", "judge-explanation");
    exp.textContent = `Judge: ${judge.judge_explanation}`;
    card.appendChild(exp);
  }

  return card;
}

// -- Helpers ---------------------------------------------------------------

function sec(title) {
  const s = el("div", "section");
  const t = el("div", "section-title");
  t.textContent = title;
  s.appendChild(t);
  return s;
}

function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

function makeOL(items) {
  const ol = document.createElement("ol");
  ol.className = "step-list";
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    ol.appendChild(li);
  }
  return ol;
}

function emptyMsg(text) {
  const d = el("div");
  d.style.cssText = "color:var(--text-secondary);font-style:italic;padding:8px 0;";
  d.textContent = text;
  return d;
}

function esc(text) {
  if (!text) return "";
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}

function fmtScore(v) {
  if (v == null || v === undefined) return "-";
  return (v * 100).toFixed(1) + "%";
}

function scoreClass(v) {
  if (v == null || v === undefined) return "score-null";
  if (v >= 0.7) return "score-high";
  if (v >= 0.4) return "score-mid";
  return "score-low";
}

// -- Boot ------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", init);
