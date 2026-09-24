'use strict';
/* Real-World Ctrl+F — browser studio. Talks only to the local server. */

const $ = s => document.querySelector(s);
const ui = {
  objects: $('#objects'), count: $('#count'), summary: $('#summary'),
  cameraDot: $('#camera-dot'), cameraStatus: $('#camera-status'), cameraToggle: $('#camera-toggle'),
  media: $('#media'), feed: $('#feed'), still: $('#still'), overlay: $('#overlay'), selection: $('#selection'),
  placeholder: $('#placeholder'), placeholderTitle: $('#placeholder-title'), placeholderText: $('#placeholder-text'),
  placeholderStart: $('#placeholder-start'), banner: $('#banner'), liveDot: $('#live-dot'), stageMode: $('#stage-mode'),
  sceneState: $('#scene-state'), fps: $('#fps'), stageHint: $('#stage-hint'),
  detail: $('#detail'), detailEmpty: $('#detail-empty'), detailKind: $('#detail-kind'), detailName: $('#detail-name'),
  detailState: $('#detail-state'), detailText: $('#detail-text'), reasons: $('#detail-reasons'),
  snapshotWrap: $('#snapshot-wrap'), snapshot: $('#snapshot'), snapshotCaption: $('#snapshot-caption'),
  meta: $('#detail-meta'), point: $('#point'), addView: $('#add-view'), forget: $('#forget'),
  search: $('#search'), candidates: $('#candidates'), toasts: $('#toasts'),
};

const STATE_LABEL = {visible: 'IN VIEW NOW', last_seen: 'LAST SEEN', unreliable: 'UNCERTAIN LOCATION', not_observed: 'NOT SEEN YET'};
const ACTIVE_CAMERA = new Set(['running', 'starting', 'reconnecting']);
let state = null;
let mode = 'live';            // live | draw | view | aim
let still = null;             // {token}
let anchor = null, rect = null;
let streaming = false, streamRetryAt = 0;
let pollTimer = null, failures = 0;
let selectedTag = null;
const savedCorners = new Set();
const ignoredTags = new Set();   // spare tags the user chose not to be reminded about (this session)

// ------------------------------------------------------------------ helpers
async function api(path, method = 'GET', body) {
  const options = {method, cache: 'no-store', headers: {}};
  if (method !== 'GET') options.headers['X-CtrlF'] = '1';
  if (body !== undefined) { options.headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(body); }
  const res = await fetch(path, options);
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = payload.detail;
    const message = typeof detail === 'string' ? detail
      : Array.isArray(detail) && detail[0]?.msg ? `Please check the input: ${detail[0].msg}` : `Request failed (${res.status})`;
    throw new Error(message);
  }
  return payload;
}

function toast(text, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = text;
  ui.toasts.append(el);
  while (ui.toasts.children.length > 3) ui.toasts.firstElementChild.remove();
  setTimeout(() => { el.classList.add('leaving'); setTimeout(() => el.remove(), 220); }, kind === 'error' ? 7000 : 4500);
}

function ago(seconds) {
  if (seconds == null) return '';
  if (seconds < 10) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return `${Math.floor(seconds / 86400)} d ago`;
}
const clock = t => new Date(t * 1000).toLocaleString([], {hour: '2-digit', minute: '2-digit', second: '2-digit', day: 'numeric', month: 'short'});
const pct = v => `${Math.round(v * 100)}%`;
const setText = (el, text) => { if (el.textContent !== text) el.textContent = text; };
const selectedItem = () => state?.items.find(i => i.id === state.selected) || null;

function svgIcon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.append(use);
  return svg;
}

function confirmDialog(title, text, okLabel = 'Confirm', danger = false) {
  const dialog = $('#confirm-dialog');
  setText($('#confirm-title'), title);
  setText($('#confirm-text'), text);
  const ok = $('#confirm-ok');
  ok.textContent = okLabel;
  ok.classList.toggle('danger-fill', danger);
  dialog.returnValue = '';
  dialog.showModal();
  return new Promise(resolve => dialog.addEventListener('close', () => resolve(dialog.returnValue === 'ok'), {once: true}));
}

function anyDialogOpen() { return !!document.querySelector('dialog[open]'); }

// ------------------------------------------------------------------ polling
async function refresh() {
  clearTimeout(pollTimer);
  try {
    render(await api('/api/status'));
    failures = 0;
  } catch (err) {
    failures += 1;
    if (failures === 2) toast('Lost contact with the studio. Is server.py still running?', 'error');
    setText(ui.cameraStatus, 'Studio not reachable');
    ui.cameraDot.className = 'dot bad';
  }
  const delay = document.hidden ? 4000 : failures ? Math.min(8000, 1000 * failures) : 700;
  pollTimer = setTimeout(refresh, delay);
}

// ------------------------------------------------------------------ render
function render(s) {
  state = s;
  renderCamera(s.camera, s.scene);
  renderLibrary(s.items);
  renderDetail(selectedItem());
  renderOverlay(selectedItem());
  renderBanner();
  renderSpotlight(s.spotlight);
  if ($('#tag-dialog').open) renderTagChoices();
}

function renderCamera(camera, scene) {
  const active = ACTIVE_CAMERA.has(camera.state);
  const running = camera.state === 'running';
  if (camera.width && camera.height) ui.media.style.setProperty('--ar', (camera.width / camera.height).toFixed(4));
  const label = {
    running: `Watching · ${camera.source}`, starting: 'Starting camera…', reconnecting: 'Reconnecting to camera…',
    error: 'Camera problem', stopped: 'Camera is off',
  }[camera.state] || camera.state;
  setText(ui.cameraStatus, label);
  ui.cameraDot.className = `dot ${running ? 'on' : camera.state === 'error' ? 'bad' : ''}`;
  ui.liveDot.className = `live-dot ${running ? 'on' : ''}`;
  setText(ui.cameraToggle, active ? 'Stop camera' : 'Start camera');
  setText(ui.fps, running && camera.fps ? `${camera.width}×${camera.height} · ${camera.fps} FPS` : '');

  // Stream lifecycle: attach while running, detach when stopped or hidden.
  const wantStream = running && !document.hidden;
  if (wantStream && !streaming && Date.now() >= streamRetryAt) {
    streaming = true;
    ui.feed.src = `/stream?t=${Date.now()}`;
  } else if (!wantStream && streaming) {
    streaming = false;
    ui.feed.removeAttribute('src');
  }

  const drawingStill = mode === 'draw' || mode === 'view';
  ui.placeholder.hidden = running || drawingStill;
  if (!running) {
    const problem = camera.state === 'error' || camera.state === 'reconnecting';
    setText(ui.placeholderTitle, problem ? 'The camera is not available' : camera.state === 'starting' ? 'Starting camera…' : 'Camera is off');
    setText(ui.placeholderText, camera.error || (camera.state === 'starting'
      ? 'Waiting for the first picture. If this takes long, another app may be using the camera.'
      : 'Start the camera to begin watching your desk. Remembered locations stay available while it is off.'));
    ui.placeholderStart.hidden = camera.state === 'starting';
    setText(ui.placeholderStart, problem ? 'Try again' : 'Start camera');
  }
  let sceneText = '', warn = false;
  if (running) {
    if (scene.state === 'stable') sceneText = 'CAMERA POSITION ✓ STEADY';
    else if (scene.state === 'moved') { sceneText = 'CAMERA MOVED · OLDER SPOTS MARKED UNCERTAIN'; warn = true; }
    else if (scene.state === 'uncertain') { sceneText = 'CAN’T VERIFY CAMERA POSITION'; warn = true; }
    else sceneText = 'CHECKING CAMERA POSITION…';
  }
  setText(ui.sceneState, sceneText);
  ui.sceneState.classList.toggle('warn', warn);
  setText(ui.stageMode, mode === 'draw' || mode === 'view' ? 'FROZEN PICTURE' : mode === 'aim' ? 'AIM TEST' : 'LIVE CAMERA');
}

function subtitle(item) {
  if (item.problem) return 'Needs re-enrolling';
  if (item.state === 'visible') return 'In view now';
  if (item.state === 'last_seen') return `Last seen ${ago(item.age_seconds)}`;
  if (item.state === 'unreliable') return `Uncertain · ${ago(item.age_seconds)}`;
  return item.kind === 'tag' ? `Not seen yet · tag #${item.id}` : 'Not seen yet';
}

function renderLibrary(items) {
  setText(ui.count, String(items.length));
  const counts = {visible: 0, last_seen: 0, unreliable: 0, not_observed: 0};
  items.forEach(i => { counts[i.state] += 1; });
  const summary = [['visible', 'in view'], ['last_seen', 'last seen'], ['unreliable', 'uncertain'], ['not_observed', 'not seen']]
    .filter(([k]) => counts[k]).map(([k, label]) => `${counts[k]} ${label}`);
  const summaryKey = summary.join('|');
  if (ui.summary.dataset.key !== summaryKey) {
    ui.summary.dataset.key = summaryKey;
    ui.summary.replaceChildren(...summary.map(t => Object.assign(document.createElement('span'), {textContent: t})));
  }
  const existing = new Map([...ui.objects.children].map(el => [el.dataset.id, el]));
  items.forEach((item, index) => {
    const key = String(item.id);
    let el = existing.get(key);
    existing.delete(key);
    if (!el) {
      el = document.createElement('button');
      el.type = 'button';
      el.className = 'object';
      el.dataset.id = key;
      el.setAttribute('role', 'listitem');
      const icon = document.createElement('span');
      icon.className = 'obj-icon';
      icon.append(svgIcon(item.kind === 'tag' ? 'tag' : 'shape'));
      const text = document.createElement('span');
      text.className = 'obj-text';
      text.append(document.createElement('b'), document.createElement('small'));
      const dot = document.createElement('span');
      dot.className = 'sdot';
      el.append(icon, text, dot);
    }
    setText(el.querySelector('b'), item.name);
    setText(el.querySelector('small'), subtitle(item));
    el.querySelector('.sdot').className = `sdot ${item.problem ? 'problem' : item.state}`;
    el.classList.toggle('active', state.selected === item.id);
    el.setAttribute('aria-current', state.selected === item.id ? 'true' : 'false');
    el.setAttribute('aria-label', `${item.name}: ${subtitle(item)}`);
    if (ui.objects.children[index] !== el) ui.objects.insertBefore(el, ui.objects.children[index] || null);
  });
  existing.forEach(el => el.remove());
}

function describe(item) {
  if (item.problem) return item.problem;
  switch (item.state) {
    case 'visible':
      return 'In view right now — the highlighted box on the camera picture shows where it is.';
    case 'last_seen':
      return `Not in view right now. It was last seen at the marked spot ${ago(item.age_seconds)}. It may have been moved since then.`;
    case 'unreliable':
      return `Not in view right now. The marked spot (${ago(item.age_seconds)}) is only a hint, because:`;
    default:
      return item.kind === 'tag'
        ? `Not seen yet. Hold its printed tag (#${item.id}) where the camera can see it.`
        : 'Not seen yet. Put it where the camera can see the same side you enrolled.';
  }
}

function renderDetail(item) {
  ui.detail.hidden = !item;
  ui.detailEmpty.hidden = !!item;
  if (!item) return;
  setText(ui.detailKind, item.kind === 'tag' ? `PRINTED TAG #${item.id}` : `DRAWN OBJECT · ${item.views} VIEW${item.views === 1 ? '' : 'S'}`);
  setText(ui.detailName, item.name);
  ui.detailState.className = `state-badge ${item.state}`;
  setText(ui.detailState, STATE_LABEL[item.state]);
  setText(ui.detailText, describe(item));
  const lines = [
    ...(item.problem ? [] : item.reasons.map(r => ['', r.text])),
    ...item.notes.map(n => ['note', n.text]),
  ];
  const key = JSON.stringify(lines);
  if (ui.reasons.dataset.key !== key) {
    ui.reasons.dataset.key = key;
    ui.reasons.replaceChildren(...lines.map(([cls, text]) => Object.assign(document.createElement('li'), {className: cls, textContent: text})));
  }
  ui.snapshotWrap.hidden = !item.snapshot;
  if (item.snapshot && ui.snapshot.getAttribute('src') !== item.snapshot) ui.snapshot.src = item.snapshot;
  setText(ui.snapshotCaption, item.state === 'visible' ? 'Most recent clear view' : 'Last clear view before it disappeared');
  const meta = [];
  if (item.last_seen) meta.push(['Last seen', item.state === 'visible' ? 'now' : clock(item.last_seen)]);
  if (item.first_seen && item.state !== 'not_observed') meta.push(['Sighting began', clock(item.first_seen)]);
  const pos = item.live || (item.x != null ? item : null);
  if (pos) meta.push(['Picture position', `${pct(pos.x)} across, ${pct(pos.y)} down`]);
  const conf = item.live ? item.live.confidence : item.confidence;
  if (item.kind === 'visual' && conf != null) meta.push(['Match strength', conf >= .8 ? 'Strong' : conf >= .6 ? 'Moderate' : 'Weak']);
  if (item.aliases.length) meta.push(['Also called', item.aliases.join(', ')]);
  const metaKey = JSON.stringify(meta);
  if (ui.meta.dataset.key !== metaKey) {
    ui.meta.dataset.key = metaKey;
    ui.meta.replaceChildren(...meta.flatMap(([k, v]) => [
      Object.assign(document.createElement('dt'), {textContent: k}),
      Object.assign(document.createElement('dd'), {textContent: v})]));
  }
  ui.point.disabled = !state.spotlight.connected || item.state === 'not_observed';
  ui.point.title = state.spotlight.connected ? '' : 'Connect the optional spotlight first';
  ui.addView.hidden = item.kind !== 'visual' || !!item.problem;
  setText(ui.forget.querySelector('span'), item.builtin ? 'Clear history' : 'Remove');
}

function renderOverlay(item) {
  const showMemory = item && mode !== 'draw' && mode !== 'view' && !item.live && item.x != null;
  const key = showMemory ? `${item.id}|${item.state}|${item.x}|${item.y}|${item.age_seconds}` : '';
  if (ui.overlay.dataset.key === key) return;
  ui.overlay.dataset.key = key;
  if (!showMemory) { ui.overlay.replaceChildren(); return; }
  const nodes = [];
  if (item.box) {
    const [x0, y0, x1, y1] = item.box;
    const box = document.createElement('div');
    box.className = 'box-hint';
    Object.assign(box.style, {left: pct(x0), top: pct(y0), width: pct(x1 - x0), height: pct(y1 - y0)});
    nodes.push(box);
  }
  const reticle = document.createElement('div');
  reticle.className = `reticle ${item.state === 'unreliable' ? 'unreliable' : ''} ${item.y > .78 ? 'flip' : ''}`;
  reticle.style.left = pct(item.x);
  reticle.style.top = pct(item.y);
  const ring = Object.assign(document.createElement('span'), {className: 'ring'});
  const core = Object.assign(document.createElement('span'), {className: 'core'});
  const tagline = Object.assign(document.createElement('span'), {
    className: 'tagline',
    textContent: item.state === 'unreliable' ? `${item.name}: uncertain · ${ago(item.age_seconds)}` : `${item.name} · last seen ${ago(item.age_seconds)}`,
  });
  if (item.x < .15) tagline.style.transform = 'translateX(-15%)';
  if (item.x > .85) tagline.style.transform = 'translateX(-85%)';
  reticle.append(ring, core, tagline);
  nodes.push(reticle);
  ui.overlay.replaceChildren(...nodes);
}

function renderBanner() {
  let html = null;
  if (mode === 'draw' || mode === 'view') {
    const target = mode === 'view' ? ` another view of ${selectedItem()?.name || 'the object'}` : ' the object';
    html = {cls: 'info', text: `Picture frozen. Drag a tight box around${target} — as little background as possible.`,
      buttons: [['Retake picture', 'retake'], ['Cancel', 'cancel', true]]};
  } else if (mode === 'aim') {
    html = {cls: 'info', text: 'Aim test: click anywhere on the camera picture and the spotlight points there.',
      buttons: [['Done', 'aim-off', true]]};
  } else {
    const tags = (state?.unregistered_tags || []).filter(t => !ignoredTags.has(t.marker_id));
    if (tags.length && !anyDialogOpen()) {
      const ids = tags.map(t => `#${t.marker_id}`).join(', ');
      html = {cls: 'tag', text: `Printed tag ${ids} is in view but not registered yet.`,
        buttons: [['Ignore', 'ignore-tags'], ['Name it', 'name-tag', true]]};
    }
  }
  const key = html ? JSON.stringify(html) : '';
  if (ui.banner.dataset.key === key) return;
  ui.banner.dataset.key = key;
  ui.banner.hidden = !html;
  if (!html) return;
  const text = Object.assign(document.createElement('span'), {textContent: html.text});
  const actions = Object.assign(document.createElement('span'), {className: 'banner-actions'});
  html.buttons.forEach(([label, action, solid]) => {
    const b = Object.assign(document.createElement('button'), {type: 'button', textContent: label, className: solid ? 'solid' : ''});
    b.dataset.action = action;
    actions.append(b);
  });
  ui.banner.className = `banner ${html.cls}`;
  ui.banner.replaceChildren(text, actions);
}

// ------------------------------------------------------------------ library actions
ui.objects.addEventListener('click', async event => {
  const row = event.target.closest('[data-id]');
  if (!row) return;
  try { await api(`/api/select/${row.dataset.id}`, 'POST'); await refresh(); }
  catch (err) { toast(err.message, 'error'); }
});

$('#detail-close').addEventListener('click', async () => {
  try { await api('/api/select', 'DELETE'); await refresh(); } catch (err) { toast(err.message, 'error'); }
});

$('#search-form').addEventListener('submit', async event => {
  event.preventDefault();
  const query = ui.search.value.trim();
  ui.candidates.hidden = true;
  if (!query) return;
  try {
    const result = await api('/api/find', 'POST', {query});
    if (result.ambiguous) { showCandidates(result.candidates); return; }
    await refresh();
    const item = selectedItem();
    if (!item) return;
    const messages = {
      visible: `Found it — ${item.name} is in view now.`,
      last_seen: `${item.name} isn’t in view. Showing where it was last seen (${ago(item.age_seconds)}).`,
      unreliable: `${item.name} isn’t in view, and its last position is uncertain. See why on the right.`,
      not_observed: `${item.name} is registered but hasn’t been seen yet.`,
    };
    toast(messages[item.state], item.state === 'unreliable' ? 'warn' : '');
  } catch (err) { toast(err.message, 'error'); }
});

function showCandidates(candidates) {
  const label = Object.assign(document.createElement('span'), {textContent: 'Which one did you mean?'});
  const buttons = candidates.map(c => {
    const b = Object.assign(document.createElement('button'), {type: 'button', textContent: c.name});
    b.addEventListener('click', async () => {
      ui.candidates.hidden = true;
      try { await api(`/api/select/${c.id}`, 'POST'); await refresh(); } catch (err) { toast(err.message, 'error'); }
    });
    return b;
  });
  ui.candidates.replaceChildren(label, ...buttons);
  ui.candidates.hidden = false;
}

ui.forget.addEventListener('click', async () => {
  const item = selectedItem();
  if (!item) return;
  const ok = item.builtin
    ? await confirmDialog(`Clear ${item.name}'s history?`, 'Its remembered location and last-seen picture are deleted. The printed-tag object stays in the library.', 'Clear history', true)
    : await confirmDialog(`Remove ${item.name}?`, `This deletes its ${item.kind === 'visual' ? 'visual template, ' : ''}remembered location and last-seen picture from this computer.`, 'Remove', true);
  if (!ok) return;
  try {
    const result = await api(`/api/object/${item.id}`, 'DELETE');
    toast(result.removed ? `${item.name} removed.` : `${item.name}'s history cleared.`);
    await refresh();
  } catch (err) { toast(err.message, 'error'); }
});

$('#clear-memory').addEventListener('click', async () => {
  const ok = await confirmDialog('Clear all remembered locations?', 'Every last-seen position and picture is deleted. Your object library is kept.', 'Clear everything', true);
  if (!ok) return;
  try { await api('/api/memory', 'DELETE'); toast('All remembered locations cleared.'); await refresh(); }
  catch (err) { toast(err.message, 'error'); }
});

// ------------------------------------------------------------------ camera
async function toggleCamera(forceStart = false) {
  const active = ACTIVE_CAMERA.has(state?.camera?.state);
  try {
    if (active && !forceStart) {
      if (mode !== 'live') exitMode();
      await api('/api/camera/stop', 'POST');
    } else {
      await api('/api/camera/start', 'POST', {});
    }
    await refresh();
  } catch (err) { toast(err.message, 'error'); }
}
ui.cameraToggle.addEventListener('click', () => toggleCamera());
ui.placeholderStart.addEventListener('click', () => toggleCamera(true));
ui.feed.addEventListener('error', () => {
  if (!streaming) return;
  streaming = false;
  streamRetryAt = Date.now() + 2000;
});
document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); else if (state) renderCamera(state.camera, state.scene); });

// ------------------------------------------------------------------ enrolment
$('#add').addEventListener('click', () => {
  const dialog = $('#add-dialog');
  dialog.returnValue = '';
  dialog.showModal();
});
$('#add-dialog').addEventListener('close', () => {
  const choice = $('#add-dialog').returnValue;
  if (choice === 'draw') startDrawing('draw');
  if (choice === 'tag') openTagDialog();
});
ui.addView.addEventListener('click', () => startDrawing('view'));

async function startDrawing(kind) {
  if (state?.camera?.state !== 'running') { toast('Start the camera first.', 'error'); return; }
  try {
    const result = await api('/api/enroll/still', 'POST');
    still = {token: result.token};
    ui.media.style.setProperty('--ar', (result.width / result.height).toFixed(4));
    ui.still.src = `/api/enroll/still/${result.token}.jpg`;
    ui.still.hidden = false;
    ui.feed.hidden = true;
    mode = kind;
    rect = null;
    ui.selection.hidden = true;
    ui.media.classList.add('drawing');
    render(state);
    ui.media.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'center'});
  } catch (err) { toast(err.message, 'error'); }
}

function exitMode() {
  mode = 'live';
  still = null; anchor = null; rect = null;
  $('#aim-mode').checked = false;
  ui.still.hidden = true;
  ui.still.removeAttribute('src');
  ui.feed.hidden = false;
  ui.selection.hidden = true;
  ui.media.classList.remove('drawing', 'aiming');
  if (state) render(state);
}

ui.banner.addEventListener('click', event => {
  const action = event.target.closest('button')?.dataset.action;
  if (action === 'cancel' || action === 'aim-off') exitMode();
  if (action === 'retake') { const kind = mode; exitMode(); startDrawing(kind); }
  if (action === 'name-tag') openTagDialog();
  if (action === 'ignore-tags') { (state?.unregistered_tags || []).forEach(t => ignoredTags.add(t.marker_id)); renderBanner(); }
});

function pointFromEvent(event) {
  const b = ui.media.getBoundingClientRect();   // .media has exactly the picture's aspect ratio
  return {x: Math.max(0, Math.min(1, (event.clientX - b.left) / b.width)),
          y: Math.max(0, Math.min(1, (event.clientY - b.top) / b.height))};
}
function drawSelection(r) {
  Object.assign(ui.selection.style, {left: pct(r.x), top: pct(r.y), width: pct(r.w), height: pct(r.h)});
  ui.selection.hidden = false;
}
ui.media.addEventListener('pointerdown', event => {
  if (mode === 'aim') { aimAt(pointFromEvent(event)); return; }
  if (mode !== 'draw' && mode !== 'view') return;
  if (event.button !== undefined && event.button !== 0) return;
  anchor = pointFromEvent(event);
  rect = null;
  ui.media.setPointerCapture(event.pointerId);
  event.preventDefault();
});
ui.media.addEventListener('pointermove', event => {
  if (!anchor) return;
  const p = pointFromEvent(event);
  rect = {x: Math.min(p.x, anchor.x), y: Math.min(p.y, anchor.y), w: Math.abs(p.x - anchor.x), h: Math.abs(p.y - anchor.y)};
  drawSelection(rect);
});
ui.media.addEventListener('pointerup', async () => {
  if (!anchor) return;
  anchor = null;
  const minW = 40 / (ui.still.naturalWidth || 960), minH = 40 / (ui.still.naturalHeight || 540);
  if (!rect || rect.w < minW || rect.h < minH) {
    ui.selection.hidden = true;
    toast('That box is too small. Drag a larger box around the object.', 'error');
    return;
  }
  if (mode === 'view') { await saveView(); return; }
  $('#enroll-preview').src = cropPreview(rect);
  $('#enroll-error').hidden = true;
  $('#enroll-dialog').showModal();
  $('#object-name').focus();
});
ui.media.addEventListener('pointercancel', () => { anchor = null; });

function cropPreview(r) {
  const img = ui.still, w = img.naturalWidth, h = img.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(r.w * w));
  canvas.height = Math.max(1, Math.round(r.h * h));
  canvas.getContext('2d').drawImage(img, r.x * w, r.y * h, r.w * w, r.h * h, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', .85);
}

$('#enroll-cancel').addEventListener('click', () => { $('#enroll-dialog').close(); ui.selection.hidden = true; });
$('#enroll-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!rect || !still) return;
  const save = $('#enroll-save');
  save.disabled = true;
  const body = {token: still.token, rect, name: $('#object-name').value.trim(),
    aliases: $('#object-aliases').value.split(',').map(x => x.trim()).filter(Boolean)};
  try {
    const result = await api('/api/enroll', 'POST', body);
    $('#enroll-dialog').close();
    $('#object-name').value = ''; $('#object-aliases').value = '';
    exitMode();
    await refresh();
    toast(result.quality === 'good'
      ? `${result.name} saved — plenty of surface detail (${result.features} features). Keep it in view to test it.`
      : `${result.name} saved, but it has little surface detail (${result.features} features), so recognition may be unreliable. A printed tag is more dependable.`,
      result.quality === 'good' ? '' : 'warn');
  } catch (err) {
    const error = $('#enroll-error');
    error.textContent = err.message;
    error.hidden = false;
  } finally { save.disabled = false; }
});

async function saveView() {
  const item = selectedItem();
  if (!item || !still) return;
  try {
    const result = await api(`/api/object/${item.id}/views`, 'POST', {token: still.token, rect});
    exitMode();
    await refresh();
    toast(`Added view ${result.views} for ${item.name} (${result.features} features).`);
  } catch (err) { toast(err.message, 'error'); ui.selection.hidden = true; }
}

// ------------------------------------------------------------------ tags
function openTagDialog() {
  selectedTag = null;
  $('#tag-error').hidden = true;
  renderTagChoices();
  $('#tag-dialog').showModal();
}
function renderTagChoices() {
  const tags = state?.unregistered_tags || [];
  if (selectedTag == null && tags.length) selectedTag = tags[0].marker_id;
  const key = tags.map(t => t.marker_id).join(',') + `|${selectedTag}`;
  const list = $('#tag-list');
  if (list.dataset.key === key) return;
  list.dataset.key = key;
  setText($('#tag-help'), tags.length
    ? 'Choose the tag you want to name:'
    : 'Hold an unused printed tag in front of the camera (run make_markers.py for spare tags). It will appear here.');
  list.replaceChildren(...tags.map(t => {
    const b = Object.assign(document.createElement('button'), {type: 'button', textContent: `Tag #${t.marker_id}`});
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', String(t.marker_id === selectedTag));
    b.addEventListener('click', () => { selectedTag = t.marker_id; renderTagChoices(); });
    return b;
  }));
}
$('#tag-cancel').addEventListener('click', () => $('#tag-dialog').close());
$('#tag-form').addEventListener('submit', async event => {
  event.preventDefault();
  const error = $('#tag-error');
  if (selectedTag == null) { error.textContent = 'No unregistered tag is in view yet.'; error.hidden = false; return; }
  try {
    const result = await api('/api/tags', 'POST', {marker_id: selectedTag, name: $('#tag-name').value.trim(),
      aliases: $('#tag-aliases').value.split(',').map(x => x.trim()).filter(Boolean)});
    $('#tag-dialog').close();
    $('#tag-name').value = ''; $('#tag-aliases').value = '';
    toast(`Tag #${result.id} is now “${result.name}”. Attach it to the object.`);
    await refresh();
  } catch (err) { error.textContent = err.message; error.hidden = false; }
});

// ------------------------------------------------------------------ spotlight
function renderSpotlight(spot) {
  const chip = $('#spot-chip');
  chip.className = `chip ${spot.connected ? (spot.calibrated ? 'on' : 'warn') : ''}`;
  setText(chip, spot.connected ? (spot.calibrated ? 'Connected' : 'Needs calibration') : 'Not connected');
  $('#spot-disconnected').hidden = spot.connected;
  $('#spot-connected').hidden = !spot.connected;
  $('#spot-cal-warning').hidden = spot.calibrated;
  if (spot.connected) {
    setText($('#spot-info'), `${spot.port} · ${spot.confirmed ? `firmware ${spot.firmware}` : 'no replies from the board (v1 firmware?)'}` +
      (spot.error && !spot.confirmed ? '' : ''));
    setText($('#spot-light'), spot.light ? 'Light off' : 'Light on');
    setText($('#spot-angles'), spot.pan == null ? '–' : `${spot.pan}°\n${spot.tilt}°`);
  }
  document.querySelectorAll('[data-corner]').forEach(b => b.classList.toggle('saved', savedCorners.has(b.dataset.corner)));
}

async function loadPorts() {
  try {
    const {ports} = await api('/api/spotlight/ports');
    const select = $('#spot-port');
    const options = ports.length
      ? ports.map(p => Object.assign(document.createElement('option'), {value: p.device, textContent: `${p.device} ${p.description ? '· ' + p.description : ''}`}))
      : [Object.assign(document.createElement('option'), {value: '', textContent: 'No serial ports found'})];
    select.replaceChildren(...options);
  } catch (err) { toast(err.message, 'error'); }
}
$('#spot-panel').addEventListener('toggle', event => { if (event.target.open && !state?.spotlight?.connected) loadPorts(); });
$('#spot-refresh').addEventListener('click', loadPorts);
$('#spot-connect').addEventListener('click', async () => {
  const port = $('#spot-port-manual').value.trim() || $('#spot-port').value;
  if (!port) { toast('Choose or type a serial port first.', 'error'); return; }
  const button = $('#spot-connect');
  button.disabled = true;
  setText(button, 'Connecting…');
  try {
    const status = await api('/api/spotlight/connect', 'POST', {port});
    toast(status.confirmed ? `Spotlight connected on ${port}.` : status.error || `Connected on ${port}.`, status.confirmed ? '' : 'warn');
    await refresh();
  } catch (err) { toast(err.message, 'error'); }
  finally { button.disabled = false; setText(button, 'Connect'); }
});
$('#spot-disconnect').addEventListener('click', async () => {
  try { await api('/api/spotlight/disconnect', 'POST'); await refresh(); } catch (err) { toast(err.message, 'error'); }
});
$('#spot-light').addEventListener('click', async () => {
  try { await api('/api/light', 'POST', {enabled: !state.spotlight.light}); await refresh(); } catch (err) { toast(err.message, 'error'); }
});
$('#spot-home').addEventListener('click', async () => {
  try { await api('/api/spotlight/home', 'POST'); await refresh(); } catch (err) { toast(err.message, 'error'); }
});
document.querySelector('.jog').addEventListener('click', async event => {
  const jog = event.target.closest('[data-jog]')?.dataset.jog;
  if (!jog) return;
  const [axis, dir] = jog.split(':');
  const step = Number($('#spot-step').value) * Number(dir);
  const spot = state.spotlight;
  const pan = (spot.pan ?? 90) + (axis === 'pan' ? step : 0);
  const tilt = (spot.tilt ?? 90) + (axis === 'tilt' ? step : 0);
  try {
    await api('/api/spotlight/move', 'POST', {pan: Math.max(0, Math.min(180, pan)), tilt: Math.max(0, Math.min(180, tilt))});
    await refresh();
  } catch (err) { toast(err.message, 'error'); }
});
document.querySelector('.corners').addEventListener('click', async event => {
  const corner = event.target.closest('[data-corner]')?.dataset.corner;
  if (!corner) return;
  try {
    await api('/api/spotlight/corner', 'POST', {corner});
    savedCorners.add(corner);
    toast(`Saved this pose as the ${event.target.textContent.replace('Save ', '')} corner.`);
    await refresh();
  } catch (err) { toast(err.message, 'error'); }
});
$('#aim-mode').addEventListener('change', event => {
  if (event.target.checked) {
    if (mode === 'draw' || mode === 'view') exitMode();
    mode = 'aim';
    ui.media.classList.add('aiming');
  } else if (mode === 'aim') exitMode();
  if (state) render(state);
});
async function aimAt(p) {
  try {
    const r = await api('/api/spotlight/aim', 'POST', p);
    toast(`Aiming at ${pct(p.x)} across, ${pct(p.y)} down → pan ${r.pan}°, tilt ${r.tilt}°.`);
    await refresh();
  } catch (err) { toast(err.message, 'error'); }
}
ui.point.addEventListener('click', async () => {
  const item = selectedItem();
  if (!item) return;
  if (item.state === 'unreliable' &&
      !await confirmDialog('Point at an uncertain spot?', `${item.name}'s last known position may be wrong (see the reasons listed). Point there anyway?`, 'Point anyway')) return;
  try {
    const r = await api(`/api/point/${item.id}`, 'POST');
    toast(`Spotlight pointing at ${item.name} (pan ${r.pan}°, tilt ${r.tilt}°).` +
      (r.calibrated ? '' : ' It is not calibrated yet, so it may miss.'), r.calibrated ? '' : 'warn');
    await refresh();
  } catch (err) { toast(err.message, 'error'); }
});

// ------------------------------------------------------------------ keyboard + voice
window.addEventListener('keydown', event => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || '');
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'f' && !anyDialogOpen()) {
    event.preventDefault();            // Ctrl+F searches the real world here
    ui.search.focus();
    ui.search.select();
  } else if (event.key === '/' && !typing && !anyDialogOpen()) {
    event.preventDefault();
    ui.search.focus();
  } else if (event.key === 'Escape' && mode !== 'live' && !anyDialogOpen()) {
    exitMode();
  }
});

const SpeechAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
const voice = $('#voice');
if (!SpeechAPI) {
  voice.disabled = true;
  voice.title = 'Voice search is not available in this browser';
} else {
  let listening = null;
  voice.title = 'Optional: uses your browser’s speech service, which may send audio to its provider';
  voice.addEventListener('click', () => {
    if (listening) { listening.abort(); return; }
    const recognition = new SpeechAPI();
    recognition.lang = navigator.language || 'en-US';
    recognition.interimResults = false;
    recognition.onresult = e => { ui.search.value = e.results[0][0].transcript; $('#search-form').requestSubmit(); };
    recognition.onerror = e => { if (e.error !== 'aborted') toast(`Voice search: ${e.error}`, 'error'); };
    recognition.onend = () => { listening = null; voice.classList.remove('listening'); };
    listening = recognition;
    voice.classList.add('listening');
    toast('Listening… (your browser’s speech service may process the audio)');
    recognition.start();
  });
}

refresh();
