
const $ = s => document.querySelector(s);
const ui = {
  objects: $('#objects'), count: $('#count'), message: $('#message'), status: $('#camera-status'),
  selectedName: $('#selected-name'), description: $('#selected-description'), meta: $('#selected-meta'),
  snapshot: $('#snapshot'), snapshotNote: $('#snapshot-note'), feed: $('#feed'), frame: $('#camera-frame'),
  selection: $('#selection'), hint: $('#draw-hint'), dialog: $('#enroll-dialog'),
};
let state = null, enrolling = false, anchor = null, rect = null;
const icon = item => item.kind === 'tag' ? '⌗' : '◈';
function announce(value, bad = false) { ui.message.textContent = value; ui.message.classList.toggle('error', bad); }
async function request(path, method = 'GET', data) {
  const res = await fetch(path, {method, headers: {'Content-Type':'application/json'},
    body: data === undefined ? undefined : JSON.stringify(data), cache: 'no-store'});
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw Error(typeof payload.detail === 'string' ? payload.detail : `Request failed (${res.status})`);
  return payload;
}
function escapeHtml(s) {return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function age(seconds) {
  if (seconds == null) return 'Not seen yet';
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}
function render(s) {
  const before = state?.selected;
  state = s;
  ui.count.textContent = s.items.length;
  ui.status.textContent = s.error || (s.ready ? 'Camera connected' : 'Starting camera…');
  $('#camera-placeholder').style.display = s.error ? 'grid' : 'none';
  ui.objects.innerHTML = s.items.map(item => `<button class="object ${s.selected === item.id ? 'active' : ''}" data-id="${item.id}">
    <span class="obj-icon">${icon(item)}</span><span><b>${escapeHtml(item.name)}</b><small>${item.visible ? '● visible now' : age(item.age_seconds)}</small></span></button>`).join('');
  const item = s.items.find(x => x.id === s.selected);
  ui.selectedName.textContent = item ? item.name : 'Nothing selected yet';
  ui.description.textContent = !item ? 'Search above, or choose an object from the left.' : item.visible
    ? 'Detected in the current camera frame.' : item.age_seconds == null
      ? 'Not observed yet. Show its marker or enrolled object to the camera.'
      : 'Last observed at the highlighted position. It may have moved since then.';
  ui.meta.innerHTML = item ? `<span class="pill ${item.visible ? '' : 'warn'}">${item.visible ? 'VISIBLE NOW' : 'LAST SEEN'}</span>
     <span class="pill">${escapeHtml(item.kind === 'tag' ? 'PRINTED MARKER' : 'VISUAL TEMPLATE')}</span>
     ${item.last_seen ? `<span class="pill">${escapeHtml(new Date(item.last_seen).toLocaleString())}</span>` : ''}` : '';
  ui.snapshot.hidden = !item?.snapshot;
  ui.snapshotNote.hidden = !!item?.snapshot;
  if (item?.snapshot) {
    const newSrc = `${item.snapshot}?t=${encodeURIComponent(item.last_seen || '')}`;
    if (ui.snapshot.getAttribute('src') !== newSrc) ui.snapshot.setAttribute('src', newSrc);
  }
  $('#point').disabled = !item?.last_seen || !s.spotlight;
  $('#light-off').disabled = !s.spotlight;
  $('#forget').disabled = !item;
  $('#last-refresh').textContent = s.ready ? 'FRAME ACTIVE' : 'CAMERA OFFLINE';
  if (s.error && before === undefined) announce(s.error, true);
}
async function refresh() {try {render(await request('/api/status'));} catch(err) {ui.status.textContent = err.message;}}
setInterval(refresh, 950); refresh();
ui.objects.addEventListener('click', async event => {
  const button = event.target.closest('[data-id]'); if (!button) return;
  try {await request('/api/select/' + button.dataset.id, 'POST');await refresh();} catch(err){announce(err.message,true);}
});
$('#search-form').addEventListener('submit', async event => {
  event.preventDefault();const q = $('#search').value.trim(); if (!q) return;
  try {await request('/api/find','POST',{query:q});await refresh();announce('Found a registered object — see the current or last-seen position.');}
  catch(err){announce(err.message,true);}
});
$('#add').addEventListener('click', () => {
  enrolling = true;rect = null;anchor = null;ui.frame.classList.add('is-enrolling');
  ui.hint.hidden = false;ui.selection.hidden = true;
  announce('Draw a rectangle around the object in the live feed. Escape to cancel.');
  ui.frame.scrollIntoView({behavior:'smooth',block:'center'});
});
function coords(event) {
  const bounds = ui.feed.getBoundingClientRect();
  return {x:Math.max(0,Math.min(1,(event.clientX-bounds.left)/bounds.width)),
          y:Math.max(0,Math.min(1,(event.clientY-bounds.top)/bounds.height))};
}
function draw(r) {
  const image = ui.feed.getBoundingClientRect(), outer=ui.frame.getBoundingClientRect();
  const style = ui.selection.style;
  style.left = `${image.left-outer.left+r.x*image.width}px`;
  style.top = `${image.top-outer.top+r.y*image.height}px`;
  style.width = `${r.w*image.width}px`;style.height = `${r.h*image.height}px`;
  ui.selection.hidden=false;
}
ui.frame.addEventListener('pointerdown', event => {
  if (!enrolling || !state?.ready) return;
  anchor=coords(event);ui.frame.setPointerCapture(event.pointerId); event.preventDefault();
});
ui.frame.addEventListener('pointermove', event => {
  if (!anchor) return;const p=coords(event);
  rect={x:Math.min(p.x,anchor.x),y:Math.min(p.y,anchor.y),w:Math.abs(p.x-anchor.x),h:Math.abs(p.y-anchor.y)};
  draw(rect);
});
ui.frame.addEventListener('pointerup', event => {
  if (!anchor) return;anchor=null;
  if (!rect || rect.w<.045 || rect.h<.045) return announce('Choose a larger area and try again.',true);
  ui.dialog.showModal();$('#object-name').focus();
});
function stopEnroll() {
  enrolling=false;anchor=null;rect=null;ui.frame.classList.remove('is-enrolling');
  ui.selection.hidden=true;ui.hint.hidden=true;ui.dialog.close();
}
$('#cancel-enroll').addEventListener('click',stopEnroll);
ui.dialog.addEventListener('cancel',event=>{event.preventDefault();stopEnroll();});
window.addEventListener('keydown', event=>{if(event.key==='Escape' && enrolling && !ui.dialog.open)stopEnroll();});
$('#enroll-form').addEventListener('submit',async event=>{
  event.preventDefault();if(!rect)return;
  const body={name:$('#object-name').value.trim(),aliases:$('#object-aliases').value.split(',').map(x=>x.trim()).filter(Boolean),rect};
  try{await request('/api/enroll','POST',body);stopEnroll();await refresh();announce('Object enrolled. Keep it in view so the tracker can learn its position.');
      $('#object-name').value='';$('#object-aliases').value='';}
  catch(err){announce(err.message,true);}
});
$('#point').addEventListener('click',async()=>{
  try{const result=await request('/api/point/'+state.selected,'POST');announce(`Spotlight sent to pan ${result.pan}°, tilt ${result.tilt}°. Calibrate against the actual desk.`);}
  catch(err){announce(err.message,true);}
});
$('#light-off').addEventListener('click',async()=>{
  try{await request('/api/light','POST',{enabled:false});announce('Spotlight off.');}catch(err){announce(err.message,true);}
});
$('#forget').addEventListener('click',async()=>{
  const item=state.items.find(x=>x.id===state.selected);if(!item)return;
  if(!confirm(`Forget ${item.name}? This deletes its local observation history${item.kind==='enrolled'?' and visual template':''}.`))return;
  try{await request('/api/object/'+item.id,'DELETE');await refresh();announce('Object and its saved data removed.');}catch(err){announce(err.message,true);}
});

// Explicit opt-in voice search. Chromium may send audio to a vendor speech service.
const SpeechAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SpeechAPI) { $('#voice').disabled = true; $('#voice').title = 'Voice search is unavailable in this browser'; }
else $('#voice').addEventListener('click', () => {
  const recognition = new SpeechAPI(); recognition.lang = 'en-US'; recognition.interimResults = false;
  recognition.onresult = e => { $('#search').value = e.results[0][0].transcript; $('#search-form').requestSubmit(); };
  recognition.onerror = e => announce('Voice search: ' + e.error, true);
  announce('Listening… Audio may be processed by your browser’s speech service.');
  recognition.start();
});
