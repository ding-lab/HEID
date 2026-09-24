#!/usr/bin/env python
from __future__ import annotations

import base64
import json
import os
import shutil
from pathlib import Path

PANEL = """
<div id="objside">
    <div id="objctl">
      <label class="showbar"><input id="objShow" type="checkbox"><span>Show 3D bodies</span></label>
      <div id="objHover" hidden style="font:12px/1.5 system-ui,sans-serif;white-space:pre-line;padding:2px 4px;color:var(--ink)"></div>
      <div class="seg segjoin" id="objWhat">
        <button type="button" id="objBod" aria-pressed="false">Solid bodies</button>
        <button type="button" id="objPts" aria-pressed="false">Cell points</button>
      </div>
      <label>point size<input id="objPt" type="range" min="5" max="30" step="5" value="10"></label>
      <div class="showbar pillrow" id="objNerveRow"><input id="objNerve" type="checkbox" checked><span>Nerve bundles</span>
        <label class="kind" style="--kc:#d9a400"><input id="objNerveLine" type="checkbox" checked><span>Line</span></label></div>
      <label class="showbar"><input id="objTls" type="checkbox"><span>TLS bodies</span></label>
      <label class="showbar"><input id="objDuct" type="checkbox" checked><span>Duct lumens</span></label>
      <label class="showbar"><input id="objGland" type="checkbox" checked><span>Tumor glands</span></label>
    </div>
    <div id="objcls"></div>
    <div id="objden">
      <label class="showbar"><input id="objDen3d" type="checkbox"><span>Denoised</span></label>
    </div>
    <div id="objctl2">
      <label class="showbar"><input id="objFocusNearby" type="checkbox" checked><span>Focus nearby</span></label>
      <label title="Conservative 3D bounds proximity, in physical micrometres">nearby <output id="objNearValue">100 µm</output><input id="objNear" type="range" min="0" max="500" step="25" value="100"></label>
      <button id="objClearFocus" type="button" disabled>Clear focus</button>
      <label>body opacity<input id="objTa" type="range" min="15" max="100" value="100"></label>
      <label><input id="objCut" type="range" min="0" max="100" value="100"> section cut</label>
      <div class="objbtns">
        <button id="objReload" type="button" title="fetch the rebuilt geometry without touching the view">reload geometry</button>
        <button id="objReset" type="button">reset view</button>
      </div>
    </div>
    <table id="objtab"><thead><tr><th>body</th><th>sections</th><th>details</th><th>at section</th></tr></thead>
      <tbody></tbody></table>
    <details id="objdet"><summary title="about this view">&#9432;</summary>
      <div id="objnote"></div></details>
</div>
"""

CSS = """
#objside{display:flex;flex-direction:column;gap:6px;margin-top:6px}
.objbtns{display:flex;gap:6px}
.objbtns button{flex:1 1 0}
/* the card stays put for every sample; with no meshes it simply reads as
   empty rather than vanishing and moving everything below it */
#objside[data-empty="1"]{opacity:.45}
#objside[data-empty="1"]::after{content:"No solid bodies fitted for this sample yet.";font-size:11.5px;color:var(--muted,#777)}
#objside[hidden]{display:none}
#objctl,#objctl2{display:flex;flex-direction:column;gap:6px;margin-bottom:10px}
#objcls{display:flex;flex-direction:column;gap:2px;margin:6px 0 10px}
#objden{display:flex;flex-direction:column;gap:4px;margin:0 0 10px}
#objden label{display:flex;align-items:center;gap:6px;font-size:11.5px;cursor:pointer}
#objcls label{display:flex;align-items:center;gap:6px;font-size:11.5px;cursor:pointer}
#objcls .sw{width:11px;height:11px;border-radius:2px;flex:none}
#objcls .n{color:var(--muted,#888);margin-left:auto;font-variant-numeric:tabular-nums}
#objctl label{display:flex;align-items:center;gap:7px}
#objctl input[type=range]{flex:1}
#objtab{border-collapse:collapse;width:100%;font-size:11.5px;font-variant-numeric:tabular-nums}
#objtab th,#objtab td{padding:3px 5px;border-bottom:1px solid var(--line,#eee);text-align:left}
#objtab tbody tr{cursor:pointer}
#objtab tbody tr[aria-selected="true"]{background:var(--accent);color:var(--paper)}
#objtab tr[hidden]{display:none}
#objtab button[data-object-group]{text-align:left;border:0;background:transparent;font-weight:600;color:inherit;cursor:pointer}
#objtab tr[role="button"]:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
#objtab td.flag{color:#b3261e;font-weight:600}
#objdet{margin-top:12px;font-size:11.5px}
#objdet summary{cursor:pointer;color:var(--muted,#777);padding:2px 0;
  list-style:none;font-size:15px;width:20px}
#objdet summary::-webkit-details-marker{display:none}
#objdet summary:hover{color:var(--accent,#0b6bcb)}
#objnote{margin-top:8px;font-size:11.5px;line-height:1.5;color:var(--muted,#555)}
#objnote b{color:var(--fg,#222)} #objnote .hard{color:#b3261e;font-weight:600}
"""

JS = r"""
// The payload is tens of megabytes of geometry. It is fetched the first time the
// reader opens this display, not on page load: a page that downloads it eagerly
// makes every reader who never opens the bodies pay for them.
window.__SOLID_INIT__ = function(){
  if(window.__VIEWER_DISPOSING__) return false;
  var V = window.__VIEWER__;
  if(!V || !V.setOverlay3D) return;
  var D = window.__SOLID__;
  if(!D || !D.meshes || !D.meshes.length) return;

  var gl = V.gl();
  var CLS_DEFAULT = {};                 // no class body on by default
  // `only` keeps the removed switch's PRESSED behaviour: TLS whose link is not
  // above chance are filtered out rather than greyed.
  // the page never filters TLS bodies on its own: what the table calls a 3-D TLS is shown
  var S = {cls:{}, tls:false, nerve:true, nline:true, duct:true, gland:true, denClo3:false, bod:false, pts:false, ptPx:1.0, only:false,
           tumAlpha:1.0, cut:1.0, sel:null, on:false, focusNearby:true, nearUm:100};
  // Figure mode (the page's Output panel). Every field is read ONLY while FIG.on is
  // true, and nothing here writes into S: switching the mode off restores the display
  // exactly, with no state to put back.
  var FIG = {on:false, hide:Object.create(null), nerveA:1, glandA:1, nerve:true, gland:true,
             sheet:false, sheetIds:null, multi:false};
  function bodiesOn(){return FIG.on || S.on;}
  function nerveOn(){return FIG.on ? FIG.nerve : S.nerve;}
  function glandOn(){return FIG.on ? FIG.gland : S.gland;}
  var MULTI = new Set();
  function selectionIds(){ return FIG.on && FIG.multi ? Array.from(MULTI) : (S.sel ? [S.sel] : []); }
  function isSelected(id){ return FIG.on && FIG.multi ? MULTI.has(id) : S.sel === id; } // frozen selection for the figure capture
  var HARD = [];    // ids the last frame drew solid (blend off, depth tested)
  function figHidden(m){ return FIG.on && !!FIG.hide[objectId(m)]; }
  // In figure mode the nerve and gland sliders ARE the body's opacity: they replace
  // the product of the panel's body-opacity slider and the focus fade, so 100% is
  // genuinely opaque whatever else is switched on. Other structures keep the normal rule.
  function bodyAlpha(b){
    var normal = S.tumAlpha * b.fade;
    if(!FIG.on) return normal;
    var st = b.meta.structure;
    if(st === "nerve") return FIG.nerveA;
    if(st === "tumor gland") return FIG.glandA;
    return normal;
  }
  // in figure mode a nerve or gland is drawn at the slider's own opacity everywhere,
  // not only where it crosses the section
  function figSolid(m){
    return FIG.on && (m.structure === "nerve" || m.structure === "tumor gland");
  }
  function figStructOn(m){
    if(FIG.sheet && (!FIG.sheetIds || !FIG.sheetIds[objectId(m)])) return false;
    if(!FIG.on) return true;
    if(m.structure === "nerve") return FIG.nerve;
    if(m.structure === "tumor gland") return FIG.gland;
    return S.on && !FIG.sheet;        // the sheet carries nerve and gland and nothing else
  }
  // is this body one of the ones the sheet will show
  function sheetBody(b){
    var st = b.meta.structure;
    if(st !== "nerve" && st !== "tumor gland") return false;
    if(!wantMesh(b.meta) || figHidden(b.meta) || !figStructOn(b.meta)) return false;
    return st === "nerve" ? !!nerveOn() : (!!glandOn() && V.S.basis === "G_withdrawn");
  }
  function above(m){ return m.above_chance === true; }
  function need(){ V.need(); }
  var CLS_COL = {};
  function setCol(k, hex){
    var h = hex.replace("#","");
    CLS_COL[k] = [parseInt(h.slice(0,2),16)/255, parseInt(h.slice(2,4),16)/255,
                  parseInt(h.slice(4,6),16)/255];
  }
  (D.class_colours ? Object.keys(D.class_colours) : []).forEach(function(k){
    var v = null;
    try{ v = localStorage.getItem("cellColour." + k); }catch(e){}
    setCol(k, v || D.class_colours[k]); });
  // the section stack's colour picker reaches the bodies through this hook
  window.__SOLID_SETCOL__ = function(k, hex){ setCol(k, hex); need(); };

  // ---- geometry into the stack's frame. Written once, here, and nowhere else.
  var COMMON = "\n\
uniform mat4 uMVP; uniform vec2 uHalfMM; uniform float uScale;\n\
uniform float uZmid; uniform float uExag;\n\
vec4 place(vec3 p){\n\
  vec2 mm = vec2(p.x/1000.0 - uHalfMM.x, -(p.y/1000.0 - uHalfMM.y)) * uScale;\n\
  float z = ((p.z - uZmid)/1000.0) * uExag;\n\
  return uMVP * vec4(mm, z, 1.0); }\n";

  var VS = "#version 300 es\nprecision highp float;\n\
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aNrm;\n\
layout(location=2) in float aFit;\n" + COMMON + "\
out vec3 vN; out float vFit; out float vZ; out float vD;\n\
void main(){ vN = aNrm; vFit = aFit; vZ = aPos.z;\n\
  gl_Position = place(aPos); vD = gl_Position.z; }";
  var FS = "#version 300 es\nprecision highp float;\n\
in vec3 vN; in float vFit; in float vZ; in float vD;\n\
uniform vec3 uCol; uniform float uAlpha; uniform float uCut; uniform float uOIT;\n\
uniform float uSectionOn; uniform float uSectionZ; uniform float uSectionBand; uniform float uFocused;\n\
uniform float uSectionDim; uniform float uHeadlight; uniform float uFitTint; uniform mat3 uNormalView;\n\
out vec4 o;\n\
void main(){ if(vZ > uCut) discard;\n\
  vec3 N = normalize(vN);\n\
  if(!gl_FrontFacing) N = -N;\n\
  vec3 L1 = normalize(vec3(0.4,0.7,0.6)), L2 = normalize(vec3(-0.6,-0.3,0.4));\n\
  float d = max(dot(N,L1),0.0)*0.78 + max(dot(N,L2),0.0)*0.22;\n\
  // Orthographic headlight: the eye is along view-space +Z. Treat both sides\n\
  // alike because the mirrored mesh placement reverses triangle winding.\n\
  if(uHeadlight > 0.5) d = abs(normalize(uNormalView * vN).z);\n\
  vec3 c = uCol * (uHeadlight > 0.5 ? (0.32 + 0.82 * d) : (0.34 + 1.35 * d));\n\
  if(vFit > 0.5 && uFitTint > 0.5) c = mix(c, vec3(0.40,0.14,0.12), 0.42);\n\
  // the OUTWARD face is what gives a body its shape, so it is read first: a\n\
  // little brighter and a little more present than the inner wall behind it.\n\
  // Not opaque -- the point of the slider is to see through, and hiding the\n\
  // inside to emphasise the outside trades one problem for the other.\n\
  float face = gl_FrontFacing ? 1.0 : 0.62;\n\
  c *= mix(0.86, 1.12, face);\n\
  float aa = uAlpha * face;\n\
  if(uSectionOn>0.5){\n\
    float crossing=1.0-smoothstep(uSectionBand*0.35,uSectionBand,abs(vZ-uSectionZ));\n\
    aa *= mix(uSectionDim,1.0,crossing); c=mix(c,uCol*1.2,crossing);\n\
  }\n\
  if(uFocused>0.5) aa=1.0;\n\
  if(uOIT > 1.5) o = vec4(aa, aa, aa, aa);\n\
  else if(uOIT > 0.5) o = vec4(c * aa, aa);\n\
  else o = vec4(c, aa); }";

  var PVS = "#version 300 es\nprecision highp float;\n\
layout(location=0) in vec3 aPos; layout(location=1) in float aCls;\n" + COMMON + "\
uniform float uPtPx; uniform float uStride; uniform float uOn[16]; uniform vec3 uPal[16];\n\
out vec3 vC; out float vKeep; out float vZ;\n\
void main(){ int c = int(aCls + 0.5);\n\
  vC = uPal[c]; vKeep = uOn[c]; vZ = aPos.z;\n\
  gl_Position = place(aPos);\n\
  if(uStride > 1.5 && mod(float(gl_VertexID), uStride) > 0.5) vKeep = 0.0;\n\
  gl_PointSize = (vKeep > 0.5) ? uPtPx : 0.0; }";
  var PFS = "#version 300 es\nprecision highp float;\n\
in vec3 vC; in float vKeep; in float vZ; uniform float uCut;\n\
out vec4 o;\n\
void main(){ if(vKeep < 0.5 || vZ > uCut) discard;\n\
  vec2 d = gl_PointCoord - 0.5; if(dot(d,d) > 0.25) discard;\n\
  o = vec4(vC, 1.0); }";

  var MP = V.mkProg(VS, FS), PP = V.mkProg(PVS, PFS);

  // ---- order-independent transparency for the translucent stack of bodies.
  // Sorting whole bodies can never be right once cords run THROUGH tumours:
  // whichever body sorts nearer depth-kills everything inside it. Weighted
  // average per PIXEL has no order at all, so no class can bury another.
  var oitOK = !!gl.getExtension("EXT_color_buffer_float");
  var OITF = null, oitVao = gl.createVertexArray();
  var OIT_FS = "#version 300 es\nprecision highp float;\n\
uniform sampler2D uAccum; uniform sampler2D uReveal;\n\
out vec4 o;\n\
void main(){ ivec2 xy = ivec2(gl_FragCoord.xy);\n\
  vec4 a = texelFetch(uAccum, xy, 0);\n\
  float r = texelFetch(uReveal, xy, 0).r;\n\
  float cov = clamp(1.0 - r, 0.0, 1.0);\n\
  vec3 c = a.rgb / max(a.a, 1e-4);\n\
  o = vec4(c * cov, cov); }";
  var OIT_VS = "#version 300 es\n\
void main(){ vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n\
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }";
  var OP = oitOK ? V.mkProg(OIT_VS, OIT_FS) : null;
  function oitEnsure(w, h, section){
    if(OITF && OITF.w === w && OITF.h === h && OITF.section===section) return OITF;
    if(OITF){ gl.deleteFramebuffer(OITF.fa); gl.deleteFramebuffer(OITF.fb);
              gl.deleteTexture(OITF.ta); gl.deleteTexture(OITF.tb);if(OITF.depth)gl.deleteRenderbuffer(OITF.depth); }
    function tex(fmt){ var t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, w, h);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST); return t; }
    var ta = tex(gl.RGBA16F), tb = tex(gl.R16F);
    var depth=null;if(section){depth=gl.createRenderbuffer();gl.bindRenderbuffer(gl.RENDERBUFFER,depth);
      gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT24,w,h);}
    function fbo(t){ var f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      if(depth)gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,depth);
      if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Body transparency framebuffer incomplete');
      return f; }
    OITF = {w: w, h: h, section,depth,ta: ta, tb: tb, fa: fbo(ta), fb: fbo(tb)};
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return OITF;
  }

  // ---- the cut cap. Drawing the far wall is NOT a cap: at a mid-height cut it
  // shows the inside of the lower half, which reads as a hollow bowl. A real cap
  // fills the cut plane wherever a ray has entered the solid and not yet left,
  // which is what the stencil counts.
  var CAP_VS = "#version 300 es\n\
precision highp float;\n\
layout(location=0) in vec2 aP;\n\
void main(){ gl_Position = vec4(aP, 0.0, 1.0); }";
  var CAP_FS = "#version 300 es\n\
precision highp float;\n\
uniform vec3 uCol; uniform float uA;\n\
out vec4 o;\n\
void main(){ o = vec4(uCol, uA); }";
  var CAPP = V.mkProg(CAP_VS, CAP_FS);
  var capProg = CAPP.p;
  var capCol = gl.getUniformLocation(capProg, "uCol");
  var capA   = gl.getUniformLocation(capProg, "uA");
  var capVao = gl.createVertexArray(); gl.bindVertexArray(capVao);
  var capBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, capBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
  gl.bindVertexArray(null);

  function drawCap(b, col, alpha){
    // Entries and exits counted in ONE pass, with separate stencil ops for the two
    // facings and no culling. If the winding is mirrored the two roles swap and the
    // count changes sign -- and the test below is "not zero", which does not care.
    // That is the point: the cap cannot be broken by a winding question.
    gl.enable(gl.STENCIL_TEST);
    gl.clear(gl.STENCIL_BUFFER_BIT);
    gl.colorMask(false,false,false,false); gl.depthMask(false);
    gl.disable(gl.CULL_FACE);
    gl.stencilFunc(gl.ALWAYS, 0, 0xff);
    gl.stencilOpSeparate(gl.FRONT, gl.KEEP, gl.KEEP, gl.INCR_WRAP);
    gl.stencilOpSeparate(gl.BACK,  gl.KEEP, gl.KEEP, gl.DECR_WRAP);
    gl.bindVertexArray(b.vao);
    gl.drawElements(gl.TRIANGLES, b.n, gl.UNSIGNED_INT, 0);
    // fill the cut plane wherever the count is non-zero: inside the solid
    gl.colorMask(true,true,true,true); gl.depthMask(true);
    gl.disable(gl.DEPTH_TEST);
    gl.stencilFunc(gl.NOTEQUAL, 0, 0xff);
    gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
    gl.useProgram(capProg);
    // the cut face follows the body's own opacity: a translucent body must not
    // acquire an opaque lid, or the slider stops meaning anything at the cut
    gl.uniform3fv(capCol, new Float32Array([col[0]*0.62, col[1]*0.62, col[2]*0.62]));
    gl.uniform1f(capA, alpha);
    if(alpha >= 0.99) gl.disable(gl.BLEND); else gl.enable(gl.BLEND);
    gl.bindVertexArray(capVao); gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.STENCIL_TEST);
    gl.useProgram(MP.p);
  }

  function b64(s){ var bin = atob(s), a = new Uint8Array(bin.length);
    for(var i=0;i<bin.length;i++) a[i]=bin.charCodeAt(i); return a.buffer; }
  function fetchBuf(m){
    return m.b64 ? Promise.resolve(b64(m.b64))
                 : fetch(m.url, {priority: "high"}).then(function(r){
                     if(!r.ok) throw new Error("fetch " + m.url + ": " + r.status);
                     return r.arrayBuffer(); });
  }
  function lodNote(txt){
    var el = document.getElementById("objLoadNote");
    if(!el){
      var host = document.getElementById("objctl");
      if(!host) return;
      el = document.createElement("div"); el.id = "objLoadNote";
      el.style.cssText = "font-size:11px;opacity:.75;margin:2px 0 4px";
      host.insertBefore(el, host.firstChild.nextSibling);
    }
    el.textContent = txt || ""; el.style.display = txt ? "block" : "none";
  }

  var bodies = [], zlo = 1e9, zhi = -1e9;
  // hover / click identification of TLS bodies
  var HOV = -1, PIN = -1, LASTF = null;
  function isTlsBody(b){
    var st = b.meta.structure;
    if(figHidden(b.meta) || !figStructOn(b.meta)) return false;
    if(st === "3d cluster") return S.tls;
    if(st === "nerve") return nerveOn();
    if(st === "duct lumen") return S.duct;
    if(st === "tumor gland") return glandOn() && V.S.basis === "G_withdrawn";
    if(st === "cell class") return false;
    return S.tls && (!S.only || above(b.meta));   // legacy linked TLS objects
  }
  var lineVao = null, lineBuf = null;
  // Bodies are fetched ON DEMAND: a reader who never switches a class on never
  // pays for its geometry, and the ones being drawn arrive first. Embedding all
  // of it in the payload made the page wait on ~200 MB before drawing anything.
  var meshState = new Array(D.meshes.length), inflight = 0, MAXQ = 6;
  function wantMesh(m){
    var st = m.structure;
    if(figHidden(m) || !figStructOn(m)) return false;
    if(st === "cell class") return S.bod && !!S.cls[m.cell_class];
    if(st === "nerve") return nerveOn();
    if(st === "duct lumen") return S.duct;
    if(st === "tumor gland") return glandOn() && V.S.basis === "G_withdrawn";
    if(st === "3d cluster") return S.tls;
    return S.tls && (!S.only || above(m));
  }
  function buildBody(m, buf){
    var dv = new DataView(buf);
    var nv = dv.getUint32(4,true), nf = dv.getUint32(8,true), o = 16;
    var pos = new Float32Array(buf, o, nv*3); o += nv*12;
    var nrm = new Float32Array(buf, o, nv*3); o += nv*12;
    var fit = new Uint8Array(buf, o, nv);     o += nv;
    while(o % 4) o++;
    var idx = new Uint32Array(buf.slice(o, o + nf*12));
    for(var i=2;i<nv*3;i+=3){ if(pos[i]<zlo) zlo=pos[i]; if(pos[i]>zhi) zhi=pos[i]; }
    var vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    function ab(loc, arr, n){ var b=gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER,b); gl.bufferData(gl.ARRAY_BUFFER,arr,gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,n,gl.FLOAT,false,0,0); }
    ab(0,pos,3); ab(1,nrm,3); ab(2,new Float32Array(fit),1);
    var ib=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,idx,gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    var cx=0, cy=0, cz=0,lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    for(var i2=0;i2<nv*3;i2+=3){ cx+=pos[i2]; cy+=pos[i2+1]; cz+=pos[i2+2];
      for(var a=0;a<3;a++){lo[a]=Math.min(lo[a],pos[i2+a]);hi[a]=Math.max(hi[a],pos[i2+a]);} }
    var body={meta:m, vao:vao, n:nf*3, c:[cx/nv, cy/nv, cz/nv],bounds:{lo,hi},fade:1};
    bodies.push(body);
    if(S.sel===objectId(m)) V.focusBody(S.sel,body.c);
    retargetFocus();
  }
  // a rebuilt payload: bodies whose file did not change (same url, same stamp) are
  // kept as they are; only new or changed ones are fetched again
  function swapPayload(nd){
    var old = {};
    bodies.forEach(function(b){ if(b.meta && b.meta.url) old[b.meta.url] = b; });
    var oldCloud = D.cloud && D.cloud.url, oldDen = D.cloud_denoised && D.cloud_denoised.url,
        oldDen3 = D.cloud_denoised3d && D.cloud_denoised3d.url;
    D = nd;
    bodies = []; meshState = new Array(D.meshes.length); inflight = 0;
    D.meshes.forEach(function(m, i){
      var b = old[m.url];
      if(b){ b.meta = m; bodies.push(b); meshState[i] = 2; }
    });
    if(!D.cloud || D.cloud.url !== oldCloud) cloud = null;
    if(!D.cloud_denoised || D.cloud_denoised.url !== oldDen) cloudDenoised = null;
    if(!D.cloud_denoised3d || D.cloud_denoised3d.url !== oldDen3) cloudDenoised3 = null;
    pumpMeshes();
  }
  // the bodies fetched before anything else and that the section stack waits for:
  // the TLS and nerve objects. The duct lumens are 29 bodies / 31 MB on HT891Z1,
  // and holding the sections back for them made the page open on an empty
  // canvas for as long as that download took; they load with the class bodies
  // when the 3-D panel is on.
  function small(m){ return m.structure !== "cell class" && m.structure !== "duct lumen"; }
  function pumpMeshes(){
    if(window.__VIEWER_DISPOSING__) return;
    var wanted = 0, done = 0;
    for(var i = 0; i < D.meshes.length; i++){
      var m = D.meshes[i];
      if(!bodiesOn() && !small(m)) continue;
      if(bodiesOn() && !wantMesh(m)) continue;
      if(bodiesOn()) wanted++;
      if(meshState[i] === 2){ done++; continue; }
      if(meshState[i] === 1 || inflight >= MAXQ) continue;
      meshState[i] = 1; inflight++;
      (function(mm, ii){
        fetchBuf(mm).then(function(buf){ if(window.__VIEWER_DISPOSING__)return; buildBody(mm, buf); meshState[ii] = 2; })
          .catch(function(){ meshState[ii] = 0; })
          .then(function(){ inflight--; need(); pumpMeshes(); });
      })(m, i);
    }
    lodNote(done < wanted ? "loading bodies " + done + " / " + wanted : "");
    var q = window.__LOADQ__;
    var nS = 0, dS = 0, nSmall = 0, dSmall = 0;
    for(var j = 0; j < D.meshes.length; j++){
      var mj = D.meshes[j];
      if(small(mj) && (!bodiesOn() || wantMesh(mj))){ nSmall++; if(meshState[j] === 2) dSmall++; }
      if(!(bodiesOn() ? wantMesh(mj) : small(mj))) continue;
      nS++; if(meshState[j] === 2) dS++;
    }
    if(q) q.set("bodies", "3-D bodies", dS, nS);
    // the section stack holds its finer ladder back until these are in
    // This runs after the payload is initialized. Only requested small bodies
    // may hold the ladder: hidden TLS/glands are never fetched with S.on=true,
    // so waiting for those forced every sample switch through the 30 s timeout.
    window.__SOLID_SMALL_DONE__ = dSmall >= nSmall;
  }
  window.__SOLID_PUMP__ = pumpMeshes;

  function parseCloud(cb, dc){
    var cd2 = new DataView(cb);
    var nc = cd2.getUint8(7), co = 8, cnt = [], tot = 0;
    for(var i=0;i<nc;i++){ var q = cd2.getUint32(co,true); co += 4; cnt.push(q); tot += q; }
    var raw = new Uint16Array(cb, co, tot*3); co += tot*6; while(co % 4) co++;
    var zt = new Float32Array(cb, co, dc.n_planes);
    var cp = new Float32Array(tot*3), ck = new Float32Array(tot), k = 0;
    for(var c=0;c<nc;c++) for(var j=0;j<cnt[c];j++,k++){
      cp[k*3]   = raw[k*3]   * dc.um_px;
      cp[k*3+1] = raw[k*3+1] * dc.um_px;
      cp[k*3+2] = zt[raw[k*3+2]];
      ck[k] = c; }
    var pv = gl.createVertexArray(); gl.bindVertexArray(pv);
    var pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb);
    gl.bufferData(gl.ARRAY_BUFFER, cp, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
    var kb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, kb);
    gl.bufferData(gl.ARRAY_BUFFER, ck, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,1,gl.FLOAT,false,0,0);
    gl.bindVertexArray(null);
    return {vao: pv, n: tot, classes: dc.classes};
  }
  var cloudDenoised = null, cloudDenoised3 = null, cloud = null;
  function ensureCloud(which){
    if(window.__VIEWER_DISPOSING__) return;
    var meta = which === "raw" ? D.cloud
             : which === "den" ? D.cloud_denoised : D.cloud_denoised3d;
    var have = which === "raw" ? cloud
             : which === "den" ? cloudDenoised : cloudDenoised3;
    if(have || !meta || (!meta.b64 && !meta.url) || meta.__busy) return;
    meta.__busy = 1;
    if(window.__LOADQ__) window.__LOADQ__.note("cloud", "point cloud loading");
    fetchBuf(meta).then(function(buf){
      if(window.__VIEWER_DISPOSING__) return;
      if(window.__LOADQ__) window.__LOADQ__.note("cloud", "");
      var pc = which === "raw" ? parseRawCloud(buf) : parseCloud(buf, meta);
      if(which === "raw") cloud = pc;
      else if(which === "den") cloudDenoised = pc;
      else cloudDenoised3 = pc;
      need();
    });
  }
  function parseRawCloud(cb){ return parseCloud(cb, D.cloud); }

  function common(prog, U, MVP, hw, hh, sc){
    gl.useProgram(prog.p);
    gl.uniformMatrix4fv(U.uMVP, false, MVP);
    gl.uniform2f(U.uHalfMM, hw, hh);
    gl.uniform1f(U.uScale, sc);
    gl.uniform1f(U.uZmid, V.zMidUm());
    gl.uniform1f(U.uExag, V.S.exag);
    gl.uniform1f(U.uCut, zlo + (zhi - zlo) * S.cut + 1);
  }

  // Display context uses physical mesh bounds, independent of the exaggerated
  // view. This conservative broad-phase distance is not a contact measurement.
  function boundsGap(a,b){
    return Math.hypot(...a.lo.map((x,k)=>Math.max(0,x-b.hi[k],b.lo[k]-a.hi[k])));
  }
  function retargetFocus(){
    var ids=selectionIds(),chosen=bodies.filter(b=>ids.includes(objectId(b.meta))),now=performance.now();
    updateFocusFade(now);
    bodies.forEach(b=>{
      b.nearGap=chosen.length?Math.min(...chosen.map(c=>boundsGap(b.bounds,c.bounds))):null;
      var target=chosen.length && S.focusNearby && b.nearGap>S.nearUm ? .06 : 1;
      if(b.fadeTo!==target){
        b.fadeFrom=b.fade==null?1:b.fade;b.fadeTo=target;b.fadeStart=now;
      }
    });
    need();
  }
  function updateFocusFade(now){
    var reduced=matchMedia('(prefers-reduced-motion: reduce)').matches,moving=false;
    bodies.forEach(b=>{if(b.fadeTo==null)return;
      var u=reduced?1:Math.min(1,(now-b.fadeStart)/320),a=u*u*(3-2*u);
      b.fade=b.fadeFrom+(b.fadeTo-b.fadeFrom)*a;
      if(u<1 && Math.abs(b.fadeFrom-b.fadeTo)>1e-5)moving=true;});
    if(moving)need();
  }
  function clearFocus(){
    MULTI.clear();S.sel=null;PIN=-1;HOV=-1;hideTip();V.focusBody(null);retargetFocus();table();
    if(window.__FIG_ON_SELECT__) window.__FIG_ON_SELECT__(null);
    var button=document.getElementById('objClearFocus');if(button){button.disabled=true;button.textContent='Clear focus';}
  }
  window.__SOLID_CLEAR_FOCUS__=clearFocus;
  window.__SOLID_FOCUS_STATE__=()=>({selected:S.sel,nearby:S.focusNearby,radiusUm:S.nearUm,
    bodies:bodies.filter(b=>wantMesh(b.meta)).map(b=>({id:objectId(b.meta),centerUm:b.c,
      bounds:b.bounds,gapUm:b.nearGap,opacity:b.fade,target:b.fadeTo}))});

  // ---- picking: a quarter-scale id pass; the readback names the body
  var KVS = "#version 300 es\nprecision highp float;\n\
layout(location=0) in vec3 aPos;\n" + COMMON + "\
out float vZ;\nvoid main(){ vZ = aPos.z; gl_Position = place(aPos); }";
  var KFS = "#version 300 es\nprecision highp float;\n\
in float vZ; uniform float uCut; uniform vec3 uPick; out vec4 o;\n\
void main(){ if(vZ > uCut) discard; o = vec4(uPick, 1.0); }";
  var KP = V.mkProg(KVS, KFS);

  // One exact screen ray, only on a press: integer colour stores both object ID
  // and surface depth. No full-screen readback or CPU copy of the mesh is needed.
  var surfacePick=null;
  window.__SOLID_HIT_AT__=function(clientX,clientY){
    if(!bodiesOn() || !bodies.length || gl.isContextLost())return null;
    var frame=V.cameraFrame(),r=gl.canvas.getBoundingClientRect();
    if(!r.width||!r.height)return null;
    var nx=2*(clientX-r.left)/r.width-1,ny=1-2*(clientY-r.top)/r.height;
    if(Math.abs(nx)>1||Math.abs(ny)>1)return null;
    var M=frame.MVP,pickM=new Float32Array(M),w=gl.drawingBufferWidth,h=gl.drawingBufferHeight;
    for(var c=0;c<4;c++){pickM[c*4]=w*(M[c*4]-nx*M[c*4+3]);pickM[c*4+1]=h*(M[c*4+1]-ny*M[c*4+3]);}
    var caps=[gl.BLEND,gl.CULL_FACE,gl.DEPTH_TEST,gl.SCISSOR_TEST,gl.STENCIL_TEST,gl.DITHER,
      gl.RASTERIZER_DISCARD,gl.POLYGON_OFFSET_FILL,gl.SAMPLE_ALPHA_TO_COVERAGE];
    var old={caps:caps.map(k=>gl.isEnabled(k)),draw:gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING),
      read:gl.getParameter(gl.READ_FRAMEBUFFER_BINDING),viewport:gl.getParameter(gl.VIEWPORT),
      program:gl.getParameter(gl.CURRENT_PROGRAM),vao:gl.getParameter(gl.VERTEX_ARRAY_BINDING),
      texture:gl.getParameter(gl.TEXTURE_BINDING_2D),rbo:gl.getParameter(gl.RENDERBUFFER_BINDING),
      depthFunc:gl.getParameter(gl.DEPTH_FUNC),depthMask:gl.getParameter(gl.DEPTH_WRITEMASK),
      depthRange:gl.getParameter(gl.DEPTH_RANGE),colorMask:gl.getParameter(gl.COLOR_WRITEMASK)};
    try{
      if(!surfacePick){
        var prog=V.mkProg(KVS,"#version 300 es\nprecision highp float;precision highp int;\nin float vZ;uniform float uCut;uniform uint uId;layout(location=0) out uvec4 o;\nvoid main(){if(vZ>uCut)discard;o=uvec4(uId,floatBitsToUint(gl_FragCoord.z),0u,1u);}");
        var t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texStorage2D(gl.TEXTURE_2D,1,gl.RGBA32UI,1,1);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
        var d=gl.createRenderbuffer();gl.bindRenderbuffer(gl.RENDERBUFFER,d);gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT24,1,1);
        var f=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,f);
        gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,t,0);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,d);
        if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('Surface pick framebuffer incomplete');
        surfacePick={prog,t,d,f};
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER,surfacePick.f);gl.viewport(0,0,1,1);
      caps.forEach(k=>gl.disable(k));gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LESS);gl.depthMask(true);
      gl.depthRange(0,1);gl.colorMask(true,true,true,true);
      common(surfacePick.prog,surfacePick.prog.u,pickM,frame.hw,frame.hh,frame.sc);
      var out=new Uint32Array(4),focused=!!S.sel&&S.focusNearby&&!FIG.on;
      for(var pass=0;pass<(focused?2:1);pass++){
        gl.clearBufferuiv(gl.COLOR,0,new Uint32Array(4));gl.clearBufferfv(gl.DEPTH,0,new Float32Array([1]));
        bodies.forEach((b,i)=>{
          if(!wantMesh(b.meta) || (focused && ((b.fadeTo<.5)!==(pass===1))))return;
          gl.uniform1ui(surfacePick.prog.u.uId,i+1);gl.bindVertexArray(b.vao);gl.drawElements(gl.TRIANGLES,b.n,gl.UNSIGNED_INT,0);
        });
        gl.readPixels(0,0,1,1,gl.RGBA_INTEGER,gl.UNSIGNED_INT,out);
        if(out[0])break;
      }
      if(!out[0])return null;
      var i=out[0]-1,ndcZ=2*new Float32Array(out.buffer)[1]-1,ori=V.viewM();
      var zScale=M[2]*ori[2]+M[6]*ori[6]+M[10]*ori[10],depth=(ndcZ-M[14])/zScale;
      var xy=V.clientToView(clientX,clientY);
      return {source:'body',objectId:objectId(bodies[i].meta),index:i,depth,
        point:V.pointFromView(xy[0],xy[1],depth)};
    }finally{
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,old.draw);gl.bindFramebuffer(gl.READ_FRAMEBUFFER,old.read);
      gl.viewport(...old.viewport);gl.useProgram(old.program);gl.bindVertexArray(old.vao);
      gl.bindTexture(gl.TEXTURE_2D,old.texture);gl.bindRenderbuffer(gl.RENDERBUFFER,old.rbo);
      caps.forEach((k,i)=>old.caps[i]?gl.enable(k):gl.disable(k));
      gl.depthFunc(old.depthFunc);gl.depthMask(old.depthMask);gl.depthRange(...old.depthRange);gl.colorMask(...old.colorMask);
    }
  };
  var pickFbo = null, pickW = 0, pickH = 0;
  function hoverable(b){return isTlsBody(b)&&(figSolid(b.meta)||!S.sel||!S.focusNearby||b.fadeTo>=.5);}
  function pickAt(px, py){
    if(!LASTF) return -1;
    var w = Math.max(1, gl.drawingBufferWidth >> 1),
        h = Math.max(1, gl.drawingBufferHeight >> 1);
    if(!pickFbo || w !== pickW || h !== pickH){
      if(pickFbo){ gl.deleteFramebuffer(pickFbo.f); gl.deleteTexture(pickFbo.t);
                   gl.deleteRenderbuffer(pickFbo.d); }
      var t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      var d = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, d);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
      var f = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, d);
      pickFbo = {f: f, t: t, d: d}; pickW = w; pickH = h;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, pickFbo.f);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS); gl.depthMask(true);
    gl.disable(gl.BLEND); gl.disable(gl.CULL_FACE);
    if(V.sectionContext())V.writeSectionDepth(LASTF.MVP);
    common(KP, KP.u, LASTF.MVP, LASTF.hw, LASTF.hh, LASTF.sc);
    for(var i = 0; i < bodies.length; i++){
      var b = bodies[i];
      if(!hoverable(b)) continue;
      gl.uniform3f(KP.u.uPick, ((i + 1) & 255) / 255, ((i + 1) >> 8) / 255, 0);
      gl.bindVertexArray(b.vao);
      gl.drawElements(gl.TRIANGLES, b.n, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);
    // the whole id buffer comes back ONCE per view; every hover after that is a
    // CPU array lookup, with no GPU sync at all
    var buf = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    PICKC = {key: pickKey(), depthSource:V.sectionDepthSource(), w: w, h: h, buf: buf};
    return pickLookup(px, py);
  }
  var PICKC = null;
  function pickKey(){
    if(!LASTF) return "";
    var m = LASTF.MVP, k = "";
    for(var i = 0; i < 16; i++) k += m[i].toFixed(5) + ",";
    return k + LASTF.hw + "," + LASTF.hh + "," + LASTF.sc + "," + bodies.length + ","
           + (S.tls ? 1 : 0) + (nerveOn() ? 1 : 0) + (S.duct ? 1 : 0) + (glandOn() ? 1 : 0) + V.S.basis + (S.only ? 1 : 0)
           + ','+[S.cut,S.sel,S.nearUm,S.focusNearby,V.sectionContext(),V.secList().join('|')].join(',')
           + ','+(FIG.on ? [1,FIG.nerve?1:0,FIG.gland?1:0,FIG.nerveA,FIG.glandA,
                            Object.keys(FIG.hide).filter(k=>FIG.hide[k]).sort().join('|'),selectionIds().join('|')].join(',') : '0')
           + "," + gl.drawingBufferWidth + "x" + gl.drawingBufferHeight;
  }
  function pickLookup(px, py){
    var w = PICKC.w, h = PICKC.h;
    var px4 = Math.min(w - 1, Math.max(0, px >> 1)),
        py4 = Math.min(h - 1, Math.max(0, py >> 1));
    var o = ((h - 1 - py4) * w + px4) * 4;
    return (PICKC.buf[o] | (PICKC.buf[o + 1] << 8)) - 1;
  }
  // CPU fallback: when the id buffer has nothing under the pointer (thin body,
  // a GPU that will not read the half-size buffer back), the body whose
  // projected centroid is nearest within FALLBACK_PX still answers
  var FALLBACK_PX = 24;
  function pickNearest(px, py){
    if(!LASTF || V.sectionContext()) return -1;
    var best = -1, bd = FALLBACK_PX * FALLBACK_PX, M = LASTF.MVP;
    for(var i = 0; i < bodies.length; i++){
      var b = bodies[i];
      if(!hoverable(b) || !b.c) continue;
      var x = b.c[0]/1000 - LASTF.hw, y = -(b.c[1]/1000 - LASTF.hh);
      x *= LASTF.sc; y *= LASTF.sc;
      var z = (b.c[2] - V.zMidUm())/1000 * V.S.exag;
      var cx = M[0]*x + M[4]*y + M[8]*z + M[12], cy = M[1]*x + M[5]*y + M[9]*z + M[13],
          cw = M[3]*x + M[7]*y + M[11]*z + M[15];
      if(cw <= 0) continue;
      var sx = (cx/cw*0.5 + 0.5) * gl.drawingBufferWidth, sy = (0.5 - cy/cw*0.5) * gl.drawingBufferHeight;
      var d2 = (sx - px) * (sx - px) + (sy - py) * (sy - py);
      if(d2 < bd){ bd = d2; best = i; }
    }
    return best;
  }
  function pickCached(px, py){
    var i = -1;
    try { i = (PICKC && PICKC.key === pickKey() && PICKC.depthSource===V.sectionDepthSource()) ? pickLookup(px, py) : pickAt(px, py); }
    catch(e){ i = -1; }
    return i >= 0 ? i : pickNearest(px, py);
  }
  var tip = document.createElement("div");
  tip.style.cssText = "position:fixed;z-index:60;pointer-events:none;display:none;" +
    "background:rgba(18,20,26,.94);color:#fff;font:12px/1.5 system-ui,sans-serif;" +
    "padding:7px 10px;border-radius:5px;max-width:280px;box-shadow:0 2px 10px rgba(0,0,0,.4)";
  document.body.appendChild(tip);
  function fmtVol(mm3){
    var um3 = mm3 * 1e9;
    return Math.round(um3).toLocaleString("en-US") + " \u00b5m\u00b3 (" +
           mm3.toFixed(5) + " mm\u00b3)";
  }
  function tipHtml(b){
    // one id everywhere: the panel table's object_id first, then the mesh name;
    // an internal "tls_" prefix never reaches the reader
    var m = b.meta, oid = String(m.object_id || m.name || "TLS").replace(/^tls_/, "");
    var L = ["<b>" + oid + "</b>"];
    if(m.structure === "tumor gland"){
      L.push(m.gland_id + " · " + m.nerve_id + " vicinity");
      L.push(m.display_mode === "profiles" ? "H&E profile outlines · spatial associations" : "H&E contour candidate · fitted 3D surface");
      if(m.display_mode === "profiles") L.push(m.display_thickness_um + " µm display thickness; lines link sections");
      if(m.n_profiles!=null)L.push(m.n_profiles + " observed profiles · " + (m.n_branch_junctions||0) + " branch junctions");
      if(m.contact_analysis){
        var c=m.contact_analysis;
        L.push("3D: " + c.relation);
        if(!c.model_overlap_detected && m.fitted_surface && c.sampled_surface_gap_um != null)
          L.push("sampled surface gap " + c.sampled_surface_gap_um.toFixed(1) + " µm");
        L.push("2D: " + c.measured_overlap_sections + " sections overlap nerve raster");
        if(c.measured_min_gap_um != null) L.push("minimum 2D raster gap " + c.measured_min_gap_um.toFixed(1) + " µm");
        L.push(c.nerve_support === "observed-contour-constrained SDF" ?
          "Nerve surface matches section contours; intermediate geometry is inferred" :
          "nerve raster and fitted body use different spatial supports");
      }else if(m.n_sections_touching_nerve != null) L.push(m.n_sections_touching_nerve + " sections overlap the source nerve region");
      if(m.cell_class === "Unclassified_HE_gland"){
        L.push("Visually outlined H&E candidate; malignancy unclassified");
        L.push("Cross-section connections and intermediate geometry are inferred");
        if(m.boundary_uncertain)L.push("Some observed contour boundaries are uncertain");
        if(m.roi_clipped)L.push("Structure extends beyond the annotated field");
        if(m.inferred_missing_section_ids && m.inferred_missing_section_ids.length)
          L.push("No observed contour at " + m.inferred_missing_section_ids.join(", "));
      }
      if(m.interpolation_conflict_ids && m.interpolation_conflict_ids.length)
        L.push("track overlap needs review: " + m.interpolation_conflict_ids.join(", "));
      L.push("largest section " + m.area_max_section + ": " + m.area_max_um2 + " µm²");
    }
    if(m.kind) L.push(m.kind + (m.rim_cells != null ? " \u00b7 lining " + m.rim_cells + " cells, tumor "
                      + Math.round((m.rim_tumor_frac || 0) * 100) + "%" : ""));
    if(m.structure === "nerve" && m.geometry_method === "observed-contour-constrained SDF"){
      L.push("Surface constrained by the existing section contours");
      if(m.n_solids > 1) L.push(m.n_solids + " separate components under this nerve ID");
      L.push("Between-section geometry and end caps are inferred");
    }
    if(m.n_sections != null && m.z_min_um != null)
      L.push(m.n_sections + " sections, z " + m.z_min_um + "\u2013" + m.z_max_um + " \u00b5m");
    if(m.in_tumor != null) L.push(m.in_tumor ? ("in tumor (" + (m.tumor_sections || 0) + " sections touch)") : "outside tumor");
    if(m.volume_mm3 != null) L.push("volume " + fmtVol(m.volume_mm3));
    if(m.volume_um3 != null && m.volume_mm3 == null) L.push((m.volume_is_display ? "display volume " : m.volume_is_fitted ? "fitted volume " : "volume ") + fmtVol(m.volume_um3 / 1e9));
    if(m.max_diameter_um != null)
      L.push("thickest " + m.max_diameter_um + " \u00b5m across, at z "
             + m.max_diameter_z_um + " \u00b5m (yellow chord)");
    if(m.z_um) L.push("z " + m.z_um[0] + "\u2013" + m.z_um[1] + " \u00b5m" +
                      (m.n_z_slabs ? " \u00b7 " + m.n_z_slabs + " slabs" : ""));
    if(m.n_cells_inside != null)
      L.push(m.n_cells_inside.toLocaleString("en-US") + " cells inside");
    if(m.frac_across_gap) L.push("\u26a0 " + Math.round(m.frac_across_gap * 100) +
                                 "% of surface fitted across a \u226525 \u00b5m gap");
    return L.join("<br>");
  }
  var hoverBox = document.createElement("div");
  hoverBox.style.cssText = "position:fixed;left:12px;top:44px;z-index:61;pointer-events:none;display:none;" +
    "background:rgba(18,20,26,.94)!important;color:#fff!important;font:12px/1.5 system-ui,sans-serif!important;" +
    "padding:7px 10px;border-radius:5px;max-width:320px;white-space:pre-line";
  document.body.appendChild(hoverBox);
  function tipLines(b){
    var d = document.createElement("div"); d.innerHTML = tipHtml(b);
    return Array.prototype.map.call(d.childNodes, function(n){ return n.nodeType === 3 ? n.textContent : (n.tagName === "BR" ? "\n" : n.textContent); }).join("").split("\n");
  }
  function showTip(b, x, y){
    // plain text lines (no HTML) into the floating tip and the fixed corner box
    var lines = tipLines(b);
    tip.textContent = ""; hoverBox.textContent = "";
    lines.forEach(function(t, i){
      var e1 = document.createElement("div"), e2 = document.createElement("div");
      e1.textContent = t; e2.textContent = t;
      if(i === 0){ e1.style.fontWeight = "700"; e2.style.fontWeight = "700"; }
      e1.style.cssText += ";color:#fff!important;font-size:12px!important;line-height:1.5!important;min-height:18px";
      e2.style.cssText += ";color:#fff!important;font-size:12px!important;line-height:1.5!important;min-height:18px";
      tip.appendChild(e1); hoverBox.appendChild(e2);
    });
    hoverBox.style.display = "none";
    var ph = document.getElementById("objHover"); if(ph){ ph.textContent = lines.join("  \u00b7  "); ph.hidden = false; }
    tip.style.left = Math.min(window.innerWidth - 290, x + 14) + "px";
    tip.style.top = (y + 14) + "px";
    tip.style.display = "block";
  }
  function hideTip(){ tip.style.display = "none"; hoverBox.style.display = "none";
    var ph = document.getElementById("objHover"); if(ph){ ph.textContent = ""; ph.hidden = true; } }
  // headless-probe access: pick at buffer coords, and each TLS body's screen spot
  window.__SOLID_PICK__ = function(px, py){ return pickAt(px, py); };
  // the same lines the 3-D tip shows, for one object by its id: the section view asks
  // for them when the reader hovers that object's 2-D region, so one object reads the
  // same way whichever view names it
  // the body under a point, in buffer pixels, with the same lines the tip shows: the
  // section view calls it when the sections are hidden and the reader is pointing at
  // a body rather than at a plane
  window.__SOLID_TIP_AT__ = function(px, py){
    if(!bodiesOn()) return null;
    var i = pickCached(px, py);
    if(i < 0 || !bodies[i]) return null;
    return tipLines(bodies[i]);
  };
  window.__SOLID_TIP__ = function(oid){
    if(!D || !D.meshes) return null;
    var want = String(oid).replace(/^tls_/, "");
    for(var i = 0; i < D.meshes.length; i++){
      var m = D.meshes[i];
      var nm = String(m.object_id || m.name || "").replace(/^tls_/, "");
      if(nm === want) return tipLines({meta: m});
    }
    return null;
  };
  window.__SOLID_TLS_SCREEN__ = function(){
    if(!LASTF) return null;
    var out = [];
    for(var i = 0; i < bodies.length; i++){
      var b = bodies[i];
      if(!isTlsBody(b)) continue;
      var x = b.c[0]/1000 - LASTF.hw, y = -(b.c[1]/1000 - LASTF.hh);
      x *= LASTF.sc; y *= LASTF.sc;
      var z = (b.c[2] - V.zMidUm())/1000 * V.S.exag;
      var M = LASTF.MVP;
      var cx = M[0]*x + M[4]*y + M[8]*z + M[12],
          cy = M[1]*x + M[5]*y + M[9]*z + M[13],
          cw = M[3]*x + M[7]*y + M[11]*z + M[15];
      out.push({i: i, name: String(b.meta.object_id || b.meta.name || "").replace(/^tls_/, ""),
                sx: (cx/cw*0.5 + 0.5) * gl.drawingBufferWidth,
                sy: (0.5 - cy/cw*0.5) * gl.drawingBufferHeight});
    }
    return out;
  };
  window.__SOLID_LABELS__ = function(){
    if(!bodiesOn() || !LASTF) return [];
    return window.__SOLID_TLS_SCREEN__().flatMap(p=>{
      const b=bodies[p.i],m=b.meta;
      if(b.c[2]>zlo+(zhi-zlo)*S.cut+1) return [];
      const group=m.structure==="nerve"?"Nerve":m.structure==="tumor gland"?"Tumor glands":m.structure==="duct lumen"?"Duct":"TLS";
      if(b.fade<.15)return [];
      return [{...p,id:objectId(m),group,text:m.gland_id||objectId(m),colour:m.colour||(group==="Nerve"?"#2166f2":"#29b8f2")}];
    });
  };
  // The pick is a full render of every body into a buffer plus a readPixels,
  // which stalls the GPU: it runs only once the pointer has PAUSED, never while
  // it is moving across the canvas
  var cvEl = gl.canvas, pickTick = 0;
  cvEl.addEventListener("mousemove", function(e){
    if(e.buttons)return;
    if(!bodiesOn() || !(S.tls || nerveOn() || S.duct || glandOn()) || PIN >= 0){ if(PIN < 0) hideTip(); return; }
    var now = performance.now();
    if(now - pickTick < 30) return;
    pickTick = now;
    var r = cvEl.getBoundingClientRect();
    var i = pickCached((e.clientX - r.left) * gl.drawingBufferWidth / r.width,
                       (e.clientY - r.top) * gl.drawingBufferHeight / r.height);
    if(i !== HOV){ HOV = i; need(); }
    if(i >= 0) showTip(bodies[i], e.clientX, e.clientY); else hideTip();
  });
  // The shared pointer recognizer calls this only for a bare short click, never
  // after a drag. Shared surface picking respects the section's tissue depth.
  window.__SOLID_CLICK_AT__=function(x,y){
    if(!bodiesOn())return false;
    var hit=V.surfaceUnderCursor(x,y,false);
    if(!hit){if(S.sel)clearFocus();return false;}
    if(hit.source!=='body')return false;
    selectObject(bodies[hit.index].meta);return true;
  };

  V.setOverlay3D(function(MVP, hw, hh, sc){
    window.__SOLID_ACTIVE__ = !!bodiesOn();
    LASTF = {MVP: MVP, hw: hw, hh: hh, sc: sc};
    if(!bodiesOn()){ if(S.sel)clearFocus();if(PIN >= 0 || HOV >= 0){ PIN = -1; HOV = -1; hideTip(); } return; }
    if(!FIG.sheet) pruneSelection();
    updateFocusFade(performance.now());
    updateSectionRelations();
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS);
    // NO FACE CULLING, on purpose. Which side of a triangle counts as "front"
    // depends on the winding, and this path mirrors y (the stack's canvas runs y
    // downwards) on top of whatever the stack's own matrices do -- every guess at
    // the right cull flag has produced a different wrong picture: the near shell
    // dropped, or the inside shown. For a CLOSED solid the depth test alone
    // already leaves exactly the nearest surface; culling was only ever an
    // optimisation. The shader flips the normal for back-facing fragments, so the
    // lighting is right whichever way a triangle happens to be wound.
    gl.disable(gl.CULL_FACE);
    // geometry and clouds are pulled in as the switches ask for them
    pumpMeshes();
    if(S.pts) ensureCloud("raw");
    if(S.pts && S.denClo3) ensureCloud("den3");
    var section=V.sectionContext(),opaque = !section && S.tumAlpha >= 0.99 && !bodies.some(b=>wantMesh(b.meta)&&b.fade<.999);
    if(opaque){ gl.disable(gl.BLEND); } else {
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); }
    // depth is WRITTEN even when translucent: the far-to-near sort feeds correct
    // blending where the order is right, and the depth test overrules it wherever
    // two bodies interpenetrate -- a surface behind another NEVER paints over it,
    // whatever class either of them is. Uniform occlusion beats perfect
    // see-through layering here.
    gl.depthMask(true);
    common(MP, MP.u, MVP, hw, hh, sc);
    gl.uniform1f(MP.u.uHeadlight, FIG.on || FIG.sheet ? 1.0 : 0.0);
    // Inverse transpose of mesh placement followed by the camera rotation.
    // Normals need inverse XY scale, the Y mirror and inverse depth stretch;
    // projection, panning and zoom do not change the direction of the headlight.
    var R = V.viewM(), xy = 1 / sc, iz = 1 / V.S.exag;
    gl.uniformMatrix3fv(MP.u.uNormalView, false, new Float32Array([
      R[0]*xy, R[1]*xy, R[2]*xy,
      -R[4]*xy, -R[5]*xy, -R[6]*xy,
      R[8]*iz, R[9]*iz, R[10]*iz]));
    gl.uniform1f(MP.u.uOIT, 0.0);
    gl.uniform1f(MP.u.uSectionOn,section?1:0);
    gl.uniform1f(MP.u.uSectionZ,V.Z[V.secList()[0] ?? V.S.zi]);
    // A narrow display band highlights surface crossings. fwidth-like sizing in
    // physical units is bounded so a distant view cannot imply a thick section.
    gl.uniform1f(MP.u.uSectionBand,Math.min(2,Math.max(.5,V.umPerDevicePx())));
    // how much a body dims where it is AWAY from the shown section. 0.14 is the
    // ordinary rule, which is why a body reads as a faint shell beside a section;
    // figure mode lifts it for nerve and gland so the slider alone decides.
    gl.uniform1f(MP.u.uSectionDim, 0.14);
    // translucent bodies must be blended FAR to NEAR, or whichever body happens
    // to sit later in the list paints over everything in front of it. The key is
    // the body centroid through the same placement the vertex shader uses.
    function drawBodies(list){
      list.forEach(function(b){
      var isCls = b.meta.structure === "cell class";
      if(figHidden(b.meta) || !figStructOn(b.meta)) return;
      if(isCls){ if(!S.bod || !S.cls[b.meta.cell_class]) return; }
      else if(b.meta.structure === "nerve"){
        if(!nerveOn()) return;
      }
      else if(b.meta.structure === "tumor gland"){
        if(!glandOn() || V.S.basis !== "G_withdrawn") return;
      }
      else if(b.meta.structure === "duct lumen"){
        if(!S.duct) return;
      }
      else if(b.meta.structure === "3d cluster"){
        // defined in space, so there is no link p-value to filter on: the
        // "above chance" gate belongs to the linked TLS and does not apply here
        if(!S.tls) return;
      }
      else { if(!S.tls) return;
             if(S.only && !above(b.meta)) return; }
      var col = isCls ? (CLS_COL[b.meta.cell_class] || [0.72,0.72,0.75])
             : b.meta.structure === "nerve" ? [0.13,0.40,0.95]   // nerve = blue
             : b.meta.structure === "duct lumen" ? [0.96,0.62,0.12] // duct lumen = amber
             : b.meta.structure === "tumor gland" ? b.meta.colour.match(/[a-f0-9]{2}/gi).map(x => parseInt(x,16)/255)
             : (above(b.meta) ? [0.16,0.72,0.95] : [0.55,0.58,0.64]);
      var bi = bodies.indexOf(b);
      if(!FIG.sheet && isSelected(objectId(b.meta))) col = col.map(v=>Math.min(1,v*1.3+.16));
      else if(!FIG.sheet && bi >= 0 && (bi === HOV || bi === PIN))
        col = [Math.min(1, col[0]*1.45+0.08), Math.min(1, col[1]*1.45+0.08),
               Math.min(1, col[2]*1.45+0.08)];
      gl.uniform3fv(MP.u.uCol, new Float32Array(col));
      // Nerve gap connections use the same material as the observed spans.
      gl.uniform1f(MP.u.uFitTint, b.meta.structure === "nerve" ? 0.0 : 1.0);
      var focused=S.sel===objectId(b.meta) && !figSolid(b.meta);
      gl.uniform1f(MP.u.uFocused,focused?1:0);
      gl.uniform1f(MP.u.uSectionDim, figSolid(b.meta) ? 1.0 : 0.14);
      gl.uniform1f(MP.u.uAlpha,focused?1:bodyAlpha(b));
      gl.bindVertexArray(b.vao);
      gl.drawElements(gl.TRIANGLES, b.n, gl.UNSIGNED_INT, 0);
    });
    }
    var solidFocus=bodies.find(b=>S.sel===objectId(b.meta)&&wantMesh(b.meta)&&!figSolid(b.meta));
    var contextBodies=solidFocus?bodies.filter(b=>b!==solidFocus):bodies;
    // A figure-mode body at the top of its slider is SOLID, not merely less faint.
    // The weighted-transparency path below averages every surface it is given, so a
    // body sent through it keeps showing its own far wall whatever alpha it carries.
    // Those bodies are therefore drawn here, depth-tested with blending off, and
    // taken out of the translucent passes.
    var figHard = FIG.on ? contextBodies.filter(function(b){
      return wantMesh(b.meta) && figSolid(b.meta) && !figHidden(b.meta)
             && figStructOn(b.meta) && bodyAlpha(b) >= 0.999; }) : [];
    HARD = figHard.map(function(b){ return objectId(b.meta); });
    if(figHard.length){
      gl.disable(gl.BLEND);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LESS);gl.depthMask(true);
      if(section)V.writeSectionDepth(MVP);
      common(MP,MP.u,MVP,hw,hh,sc);gl.uniform1f(MP.u.uOIT,0.0);drawBodies(figHard);
      contextBodies=contextBodies.filter(function(b){ return figHard.indexOf(b)<0; });
      if(opaque){ gl.disable(gl.BLEND); }
      else { gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); }
    }
    if(solidFocus){
      // The selected body is opaque and writes depth. Keeping it out of the
      // weighted transparency passes prevents far surfaces tinting it through
      // the foreground body. Actual tissue occlusion still applies.
      gl.disable(gl.BLEND);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LESS);gl.depthMask(true);
      if(section)V.writeSectionDepth(MVP);
      common(MP,MP.u,MVP,hw,hh,sc);drawBodies([solidFocus]);
    }
    var drawn = contextBodies;
    if(!opaque && oitOK){
      // OIT: two geometry passes into float buffers, then one composite.
      var W2 = gl.drawingBufferWidth, H2 = gl.drawingBufferHeight;
      var F = oitEnsure(W2, H2,section||!!solidFocus||!!figHard.length);
      gl.disable(gl.DEPTH_TEST); gl.depthMask(false);
      gl.enable(gl.BLEND);
      gl.bindFramebuffer(gl.FRAMEBUFFER, F.fa);
      gl.viewport(0, 0, W2, H2);
      if(section||solidFocus||figHard.length){
        gl.depthMask(true);gl.clearDepth(1);gl.clear(gl.DEPTH_BUFFER_BIT);
        if(section)V.writeSectionDepth(MVP);
        common(MP,MP.u,MVP,hw,hh,sc);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);
        if(solidFocus||figHard.length){
          gl.disable(gl.BLEND);gl.colorMask(false,false,false,false);
          if(solidFocus)drawBodies([solidFocus]);
          if(figHard.length)drawBodies(figHard);
          gl.colorMask(true,true,true,true);gl.enable(gl.BLEND);
        }
        gl.depthMask(false);
      }
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.uniform1f(MP.u.uOIT, 1.0);
      drawBodies(contextBodies);
      gl.bindFramebuffer(gl.FRAMEBUFFER, F.fb);
      gl.clearColor(1, 1, 1, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.blendFunc(gl.ZERO, gl.ONE_MINUS_SRC_COLOR);
      gl.uniform1f(MP.u.uOIT, 2.0);
      drawBodies(contextBodies);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, W2, H2);
      gl.disable(gl.DEPTH_TEST);gl.depthMask(false);
      gl.useProgram(OP.p);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, F.ta);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, F.tb);
      gl.uniform1i(OP.u.uAccum, 0); gl.uniform1i(OP.u.uReveal, 1);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(oitVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
      gl.activeTexture(gl.TEXTURE0);
      gl.useProgram(MP.p);
      gl.uniform1f(MP.u.uOIT, 0.0);
      gl.enable(gl.DEPTH_TEST); gl.depthMask(true);
      drawn = null;                     // handled
    } else if(!opaque){
      if(section){V.writeSectionDepth(MVP);common(MP,MP.u,MVP,hw,hh,sc);}
      // no float-buffer support, or a cut is active (the stencil caps need the
      // direct path): far-to-near by centroid, depth write on
      var zmid = V.zMidUm(), exag = V.S.exag;
      gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
      drawn = contextBodies.slice().sort(function(a, b2){
        function key(bb){
          var mx = (bb.c[0]/1000 - hw) * sc, my = -(bb.c[1]/1000 - hh) * sc;
          var mz = ((bb.c[2] - zmid)/1000) * exag;
          var w = MVP[3]*mx + MVP[7]*my + MVP[11]*mz + MVP[15];
          var cz = MVP[2]*mx + MVP[6]*my + MVP[10]*mz + MVP[14];
          return cz / (w || 1);
        }
        return key(b2) - key(a);   // far first
      });
    }
    if(drawn) drawBodies(drawn);
    // OIT depth belongs to its offscreen buffers. Points below use the default
    // framebuffer, which needs the same tissue depth to honour that occlusion.
    if(section)V.writeSectionDepth(MVP);
    // POINTS LAST, with blending OFF and depth writes ON. A point is then written
    // as exactly its class colour: nothing in front of it is mixed in, so turning
    // the block cannot change how bright a cell looks. Drawing them BEFORE the
    // bodies puts a translucent body over every point, and the amount of body in
    // front changes with the camera -- that is the flicker.
    // Denoised replaces ONLY the Schwann channel: the raw cloud still draws
    // every other class, with its Schwann points suppressed, and the in-region
    // tau-gated Schwann set draws on top of it.
    // the 3-D definition wins when both are on; each replaces only the
    // classes its cloud carries, everything else stays raw
    var denCloud = (S.denClo3 && cloudDenoised3) ? cloudDenoised3 : null;
    var denOn = !!denCloud;
    var drawCloudOnce = (function(cloud, mode){
      var on = new Float32Array(16), pal = new Float32Array(48);
      cloud.classes.forEach(function(nm, i){ if(i >= 16) return;
        // selection always rules; "suppress" hides the raw copy of any class
        // the denoised cloud carries, and the denoised cloud itself still obeys
        // the same class selection
        on[i] = (mode === "suppress" && denCloud
                 && denCloud.classes.indexOf(nm) >= 0) ? 0
              : (S.cls[nm] ? 1 : 0);
        var c = CLS_COL[nm] || [0.72,0.72,0.75];
        pal[i*3]=c[0]; pal[i*3+1]=c[1]; pal[i*3+2]=c[2]; });
      common(PP, PP.u, MVP, hw, hh, sc);
      // NO DEPTH WRITES for points. With writes on, two points of different
      // classes that land on the same pixel are resolved by depth -- so turning
      // the block swaps which class is in front and the cloud changes colour,
      // most visibly when the points are large and overlap a lot. Without writes
      // they no longer occlude each other and the winner is fixed buffer order,
      // which the camera cannot change. Opaque bodies still hide points behind
      // them, because those bodies DID write depth.
      gl.disable(gl.BLEND); gl.depthMask(false); gl.disable(gl.CULL_FACE);
      // NOT scaled by the device-pixel ratio. The page halves that ratio while
      // the pointer is down, to keep a drag smooth; a point size tied to it then
      // halves too, the cloud covers a quarter of the pixels, and it reads as the
      // cloud going dim the moment you start dragging and bright when you let go.
      // A cell is a fixed size on screen, drag or no drag.
      // A cell keeps its FOOTPRINT ON THE TISSUE: the point grows with the zoom
      // (zoomed in, every cell is bigger on screen, as on the sections), and never
      // shrinks below the chosen size when zoomed out.
      var zoomF = Math.max(1, (V.S && V.S.zoom) || 1);
      gl.uniform1f(PP.u.uPtPx, S.ptPx * zoomF * (V.DPR ? V.DPR() : Math.min(2, window.devicePixelRatio || 1)));
      // every point, every frame: subsampling during a drag made the cloud dim
      gl.uniform1f(PP.u.uStride, 1.0);
      gl.uniform1fv(PP.u.uOn, on); gl.uniform3fv(PP.u.uPal, pal);
      gl.bindVertexArray(cloud.vao);
      gl.drawArrays(gl.POINTS, 0, cloud.n);
      });
    if(cloud && S.pts && !FIG.sheet) drawCloudOnce(cloud, denOn ? "suppress" : "");
    if(denOn && S.pts && !FIG.sheet) drawCloudOnce(denCloud, "");

    // the thickest-spot chord of the hovered / pinned nerve, on top of all
    var db = (PIN >= 0 ? bodies[PIN] : (HOV >= 0 ? bodies[HOV] : null));
    if(db && db.meta.max_diameter_line && !section && !FIG.sheet){
      if(!lineVao){
        lineVao = gl.createVertexArray(); gl.bindVertexArray(lineVao);
        lineBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
        gl.bufferData(gl.ARRAY_BUFFER, 24, gl.DYNAMIC_DRAW);
        gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
        gl.bindVertexArray(null);
      }
      var L2 = db.meta.max_diameter_line;
      gl.bindBuffer(gl.ARRAY_BUFFER, lineBuf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(
        [L2[0][0],L2[0][1],L2[0][2], L2[1][0],L2[1][1],L2[1][2]]), gl.DYNAMIC_DRAW);
      common(KP, KP.u, MVP, hw, hh, sc);
      gl.uniform3f(KP.u.uPick, 1.0, 0.85, 0.1);
      gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.disable(gl.BLEND);
      gl.bindVertexArray(lineVao);
      gl.drawArrays(gl.LINES, 0, 2);
      gl.enable(gl.DEPTH_TEST); gl.depthMask(true);
    }
    // the fitted centreline of every nerve body (the curve the cells were
    // regressed on), yellow, on top of the bodies
    if(!FIG.on && nerveOn() && S.nline && !section && !FIG.sheet){
      var any = false;
      bodies.forEach(function(b){
        var cl = b.meta.centreline_um;
        if(!cl || b.meta.structure !== "nerve" || cl.length < 2 || b.fade<.5) return;
        if(figHidden(b.meta) || !figStructOn(b.meta)) return;
        if(!any){
          common(KP, KP.u, MVP, hw, hh, sc);
          gl.uniform3f(KP.u.uPick, 1.0, 0.85, 0.1);
          gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.disable(gl.BLEND);
          any = true;
        }
        if(!b.clVao){
          var arr = new Float32Array(cl.length * 3);
          for(var i = 0; i < cl.length; i++){ arr[3*i] = cl[i][0]; arr[3*i+1] = cl[i][1]; arr[3*i+2] = cl[i][2]; }
          b.clVao = gl.createVertexArray(); gl.bindVertexArray(b.clVao);
          var bf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, bf);
          gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
          gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
          gl.bindVertexArray(null); b.clN = cl.length;
        }
        gl.bindVertexArray(b.clVao); gl.drawArrays(gl.LINE_STRIP, 0, b.clN);
      });
      if(any){ gl.enable(gl.DEPTH_TEST); gl.depthMask(true); }
    }
    // Each gland retains its measured section profiles. Association lines are
    // separate geometry and never fill the space between those profiles.
    if(glandOn() && V.S.basis === "G_withdrawn" && !section && !FIG.sheet){
      common(KP, KP.u, MVP, hw, hh, sc);
      gl.enable(gl.DEPTH_TEST); gl.depthMask(false); gl.disable(gl.BLEND);
      bodies.forEach(function(b){
        var segments=b.meta.association_segments_um;
        var cl = segments ? segments.flat() : b.meta.association_line_um;
        if(b.meta.structure !== "tumor gland" || !cl || cl.length < 2 || b.fade<.5) return;
        if(figHidden(b.meta) || !figStructOn(b.meta)) return;
        if(!b.glandLineVao){
          var arr = new Float32Array(cl.flat());
          b.glandLineVao = gl.createVertexArray(); gl.bindVertexArray(b.glandLineVao);
          var bf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,bf);
          gl.bufferData(gl.ARRAY_BUFFER,arr,gl.STATIC_DRAW);
          gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
          b.glandLineN = cl.length;
        }
        var colour = b.meta.colour.match(/[a-f0-9]{2}/gi).map(x => parseInt(x,16)/255);
        gl.uniform3f(KP.u.uPick,colour[0],colour[1],colour[2]);
        gl.bindVertexArray(b.glandLineVao); gl.drawArrays(segments?gl.LINES:gl.LINE_STRIP,0,b.glandLineN);
      });
      gl.depthMask(true);
    }
    gl.bindVertexArray(null);
    gl.disable(gl.CULL_FACE); gl.enable(gl.BLEND); gl.depthMask(true);
  });

  // ---- panel: the solid page's own controls, moved across unchanged. The only
  // row dropped is depth -- that is the stack's slider now, one for both.
  function ptTag(c){
    var r = (D.cloud && D.cloud.per_class || []).filter(function(x){
      return x.cell_class === c; })[0];
    if(!r || !r.n_cells) return "";
    return r.kept_fraction >= 0.999
      ? (r.n_cells/1000).toFixed(0) + "k all"
      : (r.n_points/1000).toFixed(0) + "k of " + (r.n_cells/1000).toFixed(0) + "k";
  }

  (function(){
    var host = document.getElementById("objcls"); if(!host) return;
    // the same order the page's own class list uses, so a reader does not have to
    // re-find a class when moving between the two panels
    var UI = V.CELLUI || [];
    var names = D.meshes.filter(function(m){ return m.structure === "cell class"; })
      .sort(function(a, b){
        var ia = UI.indexOf(a.cell_class), ib = UI.indexOf(b.cell_class);
        return (ia < 0 ? 999 : ia) - (ib < 0 ? 999 : ib); })
      .map(function(m){ return m.cell_class; });
    names.forEach(function(c){ S.cls[c] = !!CLS_DEFAULT[c]; });
    function hexOf(c){
      if(V.CELLINFO && V.CELLINFO[c]) return V.CELLINFO[c].colour;
      var k = CLS_COL[c] || [0.7,0.7,0.7];
      return "#" + k.map(function(v){ return ("0" + Math.round(v*255).toString(16)).slice(-2); }).join("");
    }
    function cellsOf(c){
      var r = (D.cloud && D.cloud.per_class || []).filter(function(x){ return x.cell_class === c; })[0];
      return r && r.n_cells ? r.n_cells : null;
    }
    // the SAME row builder and the SAME colour store as the section list: the
    // swatch is the picker, and a colour set here recolours the sections too
    V.buildClassList(host, {
      names: names,
      colour: hexOf,
      setColour: function(c, h){ V.setCellColour(c, h); },
      count: cellsOf,
      isOn: function(c){ return !!S.cls[c]; },
      toggle: function(c){ S.cls[c] = !S.cls[c]; },
      title: function(c){ return c + "  " + hexOf(c) + (ptTag(c) ? "  points: " + ptTag(c) : ""); },
      id: function(c){ return "orow-" + c; }});
  })();

  var GROUP_OPEN = {TLS:false, Nerve:false, "Tumor glands":false, Duct:false};
  var relationKey='';
  function updateSectionRelations(force){
    var on=V.sectionContext(),zi=V.S.sel,key=[on,zi,bodies.length].join('|');
    if(!force && key===relationKey)return;relationKey=key;
    var byId=new Map(bodies.map(b=>[objectId(b.meta),b]));
    document.querySelectorAll('#objtab tr[data-o]').forEach(row=>{
      var td=row.lastElementChild,b=byId.get(row.dataset.o);
      td.textContent='';td.title='';if(!on||!b)return;
      var z=V.Z[zi],lo=b.bounds.lo[2],hi=b.bounds.hi[2],gap=lo>z?lo-z:hi<z?hi-z:0;
      td.textContent=gap===0?'spans z':(gap>0?'+':'−')+Math.abs(gap).toFixed(0)+' µm';
      td.title=gap===0?'Its z range includes the current section':gap>0?'Toward later sections':'Toward earlier sections';
    });
  }
  function objectId(m){ return String(m.object_id || m.name); }
  function selectionChanged(){
    var b=bodies.find(b=>objectId(b.meta)===S.sel);
    PIN=b?bodies.indexOf(b):-1;HOV=-1;hideTip();
    if(b)V.focusBody(S.sel,b.c);else V.focusBody(null);
    retargetFocus();table();
    var ids=selectionIds(),button=document.getElementById('objClearFocus');
    if(button){button.disabled=!ids.length;button.textContent=ids.length>1?'Clear '+ids.length+' selected':'Clear focus'+(S.sel?' · '+S.sel:'');}
    if(window.__FIG_ON_SELECT__)window.__FIG_ON_SELECT__(S.sel);
  }
  function pruneSelection(){
    if(FIG.on && FIG.multi){
      var changed=false;
      MULTI.forEach(id=>{var m=D.meshes.find(m=>objectId(m)===id);if(!m||!wantMesh(m)){MULTI.delete(id);changed=true;}});
      if(changed){if(!MULTI.has(S.sel))S.sel=Array.from(MULTI).pop()||null;selectionChanged();}
    }else{
      var m=D.meshes.find(m=>objectId(m)===S.sel);if(m&&!wantMesh(m))clearFocus();
    }
  }
  function selectObject(m){
    var id = objectId(m);
    if(FIG.on && FIG.multi){
      if(MULTI.has(id)){MULTI.delete(id);if(S.sel===id)S.sel=Array.from(MULTI).pop()||null;selectionChanged();return;}
      MULTI.add(id);
    }else if(S.sel === id){ clearFocus();return; }
    {
      S.sel = id; PIN = -1; HOV = -1;
      if(m.structure === "tumor gland" && V.S.basis !== "G_withdrawn") V.setBasis("G_withdrawn");
      var toggle = m.structure === "nerve" ? "objNerve" : m.structure === "tumor gland" ? "objGland" : m.structure === "duct lumen" ? "objDuct" : "objTls";
      for(var key of (FIG.on ? [] : ["objShow", toggle])){
        var box = document.getElementById(key);
        if(box && !box.checked){ box.checked = true; box.dispatchEvent(new Event("change",{bubbles:true})); }
      }
      if(!FIG.on && m.structure === "tumor gland") showGlandProfiles();
      pumpMeshes();
      var b=bodies.find(b=>objectId(b.meta)===id);
      if(b){PIN=bodies.indexOf(b);V.focusBody(id,b.c);}
      retargetFocus();
      var clear=document.getElementById('objClearFocus');if(clear){clear.disabled=false;clear.textContent='Clear focus · '+id;}
      var box = gl.canvas.getBoundingClientRect(); showTip({meta:m},box.left+16,box.top+16);
    }
    table(); need();
    if(window.__FIG_ON_SELECT__) window.__FIG_ON_SELECT__(S.sel);
  }
  // The page's figure mode drives these; the module keeps its own copy so nothing
  // in S is touched and switching the mode off needs no restore.
  window.__SOLID_FIG__ = function(patch){
    var wasMulti=FIG.on && FIG.multi;
    if(patch && typeof patch === "object"){
      if("hide" in patch) FIG.hide = patch.hide || Object.create(null);
      if("sheetIds" in patch) FIG.sheetIds = patch.sheetIds;
      ["on","nerve","gland","sheet","multi"].forEach(k => { if(k in patch) FIG[k] = !!patch[k]; });
      ["nerveA","glandA"].forEach(k => { if(k in patch) FIG[k] = Math.max(0, Math.min(1, +patch[k])); });
    }
    if(wasMulti !== (FIG.on && FIG.multi)){
      MULTI.clear();if(FIG.on && FIG.multi && S.sel)MULTI.add(S.sel);
      selectionChanged();
    }
    window.__SOLID_ACTIVE__=bodiesOn();
    if(!FIG.sheet)pruneSelection();
    pumpMeshes();need();
  };
  // every nerve and gland body, for the figure mode's per-object list
  window.__SOLID_OBJECTS__ = function(){
    return (D.meshes || []).filter(m => m.structure === "nerve" || m.structure === "tumor gland")
      .map(m => ({id: objectId(m), group: m.structure === "nerve" ? "Nerve" : "Tumor glands",
                  label: m.gland_id || objectId(m), nerve_id: m.nerve_id || ""}))
      .sort((x, y) => x.group.localeCompare(y.group)
                   || x.label.localeCompare(y.label, undefined, {numeric: true}));
  };
  // the focus rule's own fade for one body, for checks
  window.__SOLID_FADE__ = function(id){
    var b = (bodies || []).find(function(x){ return objectId(x.meta) === id; });
    return b ? b.fade : null;
  };
  // the opacity the renderer hands the shader for one body, for checks
  window.__SOLID_ALPHA__ = function(id){
    var b = (bodies || []).find(function(x){ return objectId(x.meta) === id; });
    return b ? bodyAlpha(b) : null;
  };
  // the ids the renderer is actually drawing, under every gate including figure mode
  window.__SOLID_DRAWN__ = () => (bodies || []).filter(b => wantMesh(b.meta)).map(b => objectId(b.meta));
  // the bodies the last frame drew through the blend-off, depth-tested pass: those
  // are the ones that came out genuinely solid rather than weighted-averaged
  window.__SOLID_HARD__ = () => HARD.slice();
  // what the figure sheet is about to show: which bodies, the world box they occupy
  // (micrometres, the volume canvas frame) and the colours the key must match
  window.__SOLID_SHEET_INFO__ = function(){
    var ids=selectionIds(), selected = bodies.filter(b => ids.includes(objectId(b.meta)) && sheetBody(b));
    if(!bodiesOn() || !selected.length) return null;
    // Use the same physical, unexaggerated bounds distance as Focus nearby.
    // Other nerves are never exported, even when they lie in this neighbourhood.
    var shown = bodies.filter(b => sheetBody(b) && (selected.includes(b) ||
      (b.meta.structure === "tumor gland" && selected.some(c=>boundsGap(c.bounds,b.bounds) <= S.nearUm))));
    var lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    shown.forEach(function(b){
      for(var a = 0; a < 3; a++){
        lo[a] = Math.min(lo[a], b.bounds.lo[a]); hi[a] = Math.max(hi[a], b.bounds.hi[a]); }
    });
    var hex = function(c){ return "#" + c.map(function(v){
      return ("0" + Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16)).slice(-2); }).join(""); };
    var glandCols = [];
    shown.forEach(function(b){
      if(b.meta.structure !== "tumor gland" || !b.meta.colour) return;
      var c = b.meta.colour.toLowerCase();
      if(glandCols.indexOf(c) < 0) glandCols.push(c);
    });
    return {lo: lo, hi: hi, selected: selected.length===1?objectId(selected[0].meta):selected.length+" selected structures", selection: selected.map(b=>objectId(b.meta)), nearby_um: S.nearUm,
            ids: shown.map(b => objectId(b.meta)),
            nerves: shown.filter(function(b){ return b.meta.structure === "nerve"; })
                         .map(function(b){ return objectId(b.meta); }),
            glands: shown.filter(function(b){ return b.meta.structure === "tumor gland"; })
                         .map(function(b){ return objectId(b.meta); }),
            nerve_colour: hex([0.13, 0.40, 0.95]), gland_colours: glandCols};
  };
  window.__SOLID_SELECTED__ = () => S.sel;
  window.__SOLID_SELECTION__ = () => selectionIds();
  window.__SOLID_SELECT__ = id => {
    var m = D.meshes.find(m => objectId(m) === id); if(m) selectObject(m); return !!m;
  };
  function table(){
    var tb=document.querySelector("#objtab tbody"); if(!tb) return; tb.innerHTML="";
    var groups = {TLS:[], Nerve:[], "Tumor glands":[], Duct:[]};
    D.meshes.forEach(m => {
      if(m.structure === "nerve") groups.Nerve.push(m);
      else if(m.structure === "tumor gland") groups["Tumor glands"].push(m);
      else if(m.structure === "3d cluster" || m.structure === "TLS object") groups.TLS.push(m);
      else if(m.structure === "duct lumen") groups.Duct.push(m);
    });
    groups.Nerve.sort((a,b) => {
      var ak = a.nerve_id || objectId(a), bk = b.nerve_id || objectId(b);
      return ak.localeCompare(bk,undefined,{numeric:true}) || (a.structure === "nerve" ? -1 : b.structure === "nerve" ? 1 : objectId(a).localeCompare(objectId(b),undefined,{numeric:true}));
    });
    Object.keys(groups).forEach(group => {
      var members = groups[group]; if(!members.length && (group === "Duct" || group === "Tumor glands")) return;
      var header=document.createElement("tr"), cell=document.createElement("td"), button=document.createElement("button");
      cell.colSpan=4; button.type="button"; button.dataset.objectGroup=group;
      button.setAttribute("aria-expanded",String(GROUP_OPEN[group]));
      button.textContent=(GROUP_OPEN[group] ? "▾ " : "▸ ")+group+" ("+members.length+")";
      button.addEventListener("click",() => {GROUP_OPEN[group]=!GROUP_OPEN[group];table();});
      cell.appendChild(button);
      header.appendChild(cell);tb.appendChild(header);
      members.forEach(m => {
        var id=objectId(m), tr=document.createElement("tr"); tr.dataset.o=id;tr.hidden=!GROUP_OPEN[group];
        tr.setAttribute("aria-selected",String(isSelected(id)));tr.tabIndex=0;tr.setAttribute("role","button");
        tr.setAttribute("aria-label","Highlight "+id);
        var values=[m.gland_id || id, m.n_sections || m.n_z_slabs || m.n_measured_sections || "",
          m.nerve_id || (m.volume_mm3!=null ? fmtVol(m.volume_mm3) : m.kind || ""), ""];
        values.forEach(value => {var td=document.createElement("td");td.textContent=value;tr.appendChild(td);});
        tr.title=tipLines({meta:m}).join("\n");
        tr.addEventListener("click",() => selectObject(m));
        tr.addEventListener("keydown",e => {if(e.key === "Enter" || e.key === " "){e.preventDefault();selectObject(m);}});
        tb.appendChild(tr);
      });
    });
    updateSectionRelations(true);
  }

  function bind(id, fn){ var el = document.getElementById(id); if(!el) return;
    el.addEventListener(el.type === "checkbox" ? "change" : "input",
      function(){ fn(el); V.need(); }); }
  // same look as the Modality pair, but these two are NOT alternatives: a reader
  // can have the surfaces and the measured cells on at once, so each button is its
  // own toggle rather than one selection among the group.
  function toggle(id, set){
    var b = document.getElementById(id); if(!b) return;
    b.addEventListener("click", function(){
      var on = b.getAttribute("aria-pressed") !== "true";
      b.setAttribute("aria-pressed", on ? "true" : "false");
      set(on); V.need(); });
  }
  toggle("objBod", function(v){ S.bod = v; });
  toggle("objPts", function(v){ S.pts = v; });
  bind("objPt",  function(e){ S.ptPx = (parseInt(e.value,10)||10)/10; });
  bind('objFocusNearby',function(e){S.focusNearby=e.checked;retargetFocus();});
  bind('objNear',function(e){S.nearUm=Number(e.value);document.getElementById('objNearValue').textContent=S.nearUm+' µm';retargetFocus();});
  document.getElementById('objClearFocus')?.addEventListener('click',clearFocus);
  bind("objTls", function(e){ S.tls = e.checked; });
  bind("objNerve", function(e){ S.nerve = e.checked; });
  bind("objNerveLine", function(e){ S.nline = e.checked; });
  (function(){ var k = document.getElementById("objNerveLine"); if(k) k.addEventListener("click", function(e){ e.stopPropagation(); }); })();
  // a row carrying sub-option pills is a <div> (a <label> cannot hold another
  // <label>), so give it the click behaviour the plain rows get for free: anywhere
  // on the row toggles the row's own switch, while the pills keep their own clicks
  Array.prototype.forEach.call(document.querySelectorAll(".showbar.pillrow"), function(rowEl){
    var box = rowEl.querySelector("input[type=checkbox]");
    if(!box || box.parentNode !== rowEl) return;
    // bound once per row: the 2-D card's handler runs over the same rows; two handlers
    // on one row flip the box twice, which is no change at all
    if(rowEl.dataset.rowclick) return;
    rowEl.dataset.rowclick = "1";
    rowEl.addEventListener("click", function(e){
      if(e.target === box || (e.target.closest && e.target.closest(".kind"))) return;
      if(box.matches(":disabled")) return;
      box.checked = !box.checked;
      box.dispatchEvent(new Event("change", {bubbles: true}));
    });
  });
  bind("objDuct", function(e){ S.duct = e.checked; });
  function showGlandProfiles(){
    if(V.setGlandKind && D.meshes.some(m=>m.structure==="tumor gland")){
      V.setCellsOverlay(true);V.setGlandReg(true);V.setGlandKind("3d",true);
    }
  }
  bind("objGland", function(e){ S.gland = e.checked;if(e.checked) showGlandProfiles(); });
  bind("objDen3d", function(e){ S.denClo3 = e.checked; });
  bind("objTa",  function(e){ S.tumAlpha = (parseInt(e.value,10)||100)/100; });
  bind("objCut", function(e){
    S.cut = (parseInt(e.value,10)||100)/100;
    // With the sections showing, the cut is only readable against the section it
    // lands on: fifty planes in front of the cut hide it. So dragging the cut puts
    // the stack into single-layer mode and moves the selection to the measured
    // section nearest the cut plane -- the picture then answers "what does the
    // body look like HERE", which is the question the cut is asking.
    if(!V.S.baseOn) return;
    var zc = zlo + (zhi - zlo) * S.cut, best = 0, bd = Infinity;
    for(var i = 0; i < V.N; i++){
      var d = Math.abs(V.Z[i] - zc);
      if(d < bd){ bd = d; best = i; }
    }
    if(V.S.sel !== best) V.setSelected(best);
    if(!V.S.solo) V.setOnly(true);
  });
  var rb = document.getElementById("objReset");
  if(rb) rb.addEventListener("click", function(){ V.resetView(); });

  // Fetch the rebuilt geometry and swap it in place. The CAMERA, the class
  // switches, the opacity and the cut are state of this session, not of the
  // payload, so they are left exactly as they are -- a page reload would throw
  // all of them away, which is the whole reason this button exists.
  var rl = document.getElementById("objReload");
  if(rl) rl.addEventListener("click", function(){
    if(rl.disabled) return;
    var was = rl.textContent;
    rl.disabled = true; rl.textContent = "loading\u2026";
    var s = document.createElement("script");
    // a fresh URL each time: the browser would otherwise hand back the copy it
    // already has and the button would appear to do nothing
    s.src = (window.__DATA_BASE__ || "") + "objects_payload.js?t=" + Date.now();
    s.onload = function(){
      // whatever happens inside, the button must come back: an uncaught throw here
      // would leave it stuck on "loading..." with no message
      try{
        swapPayload(window.__SOLID__);
        // classes that are new since the last load start off, so a rebuild cannot
        // silently switch something on; the ones already chosen keep their state
        D.meshes.filter(function(m){ return m.structure === "cell class"; })
          .forEach(function(m){
            if(!(m.cell_class in S.cls)) S.cls[m.cell_class] = false; });
        table();
        rl.title = "fetch the rebuilt geometry without touching the view";
        V.need();
      }catch(e){
        rl.title = "reload failed: " + (e && e.message ? e.message : e);
        console.error("reload geometry", e);
      }finally{
        rl.disabled = false; rl.textContent = was; s.remove();
      }
    };
    s.onerror = function(){
      rl.disabled = false; rl.textContent = was;
      rl.title = "the rebuilt payload did not load"; s.remove(); };
    document.head.appendChild(s);
  });

  var note=document.getElementById("objnote");
  if(note) note.innerHTML =
    'Every cell type is a <b>pan-cancer Cell model '+
    'prediction</b> on H&amp;E. This cohort has <b>no spatial ground truth</b>, so no '+
    'tumour boundary and no TLS here has ever been scored.'+
    '<br><br><b>Fitted bodies.</b> Each fitted body contains closed surfaces '+
    'fitted through the measured sections: every section contributes its own signed '+
    'distance field, those fields are interpolated along z, and one isosurface is taken. '+
    '<span class="hard">The surface between two measured sections is fitted geometry, '+
    'not a measurement.</span>'+
    (D.meshes.some(m=>m.structure==="tumor gland") ? '<br><br><b>Individual gland tracks.</b> '+
    'IDs and colours match the observed 2D outlines. '+
    (D.meshes.some(m=>m.structure==="tumor gland" && m.display_mode==="branched") ?
    'Orange 3d-gland IDs identify fitted branch families near the nerve; green 2d-gland IDs identify other observed profiles. '+
    'Each section keeps its separate observed outlines and shows their parent 3D ID when assigned. '+
    'Lines follow the reviewed geometric branch graph. Hover reports model proximity. '+
    'Model overlap does not establish biological contact. ' :
    D.meshes.some(m=>m.structure==="tumor gland" && m.fitted_surface) ?
    'Multi-section tracks have separate fitted surfaces; single-section tracks remain display slabs. '+
    'Hover shows model proximity to the nerve and any detected track intersections needing review. '+
    'The nerve raster and fitted nerve body use different spatial supports; model overlap does not establish tissue contact. ' :
    'Each observed outline is a separate 2 um display slab. Lines show geometric associations across sections. ')+
    'These assistant-delineated profiles and geometric associations await pathology review.' : '')+
    '<br><br>Nerve gap connections retain the body colour. On other bodies, '+
    '<span class="hard">dark-red patches</span> are where the fit spans a '+
    'gap of <b>25 um or more</b> ('+D.n_gaps+' such gaps; '+
    Math.round(D.tumour_frac_fitted*100)+'% of the tumour surface). '+
    'Nothing was measured there and the shape inside is unknown.'+
    '<br><br><b>Depth is magnified</b> by the badge amount; set it to 1 for the true '+
    '1:13 proportion. '+D.n_measured+' of '+D.n_slots+' 5 um depth slots carry a measured '+
    'section, so '+Math.round(100-100*D.n_measured/D.n_slots)+'% of the depth was never cut.'+
    '<br><br>TLS bodies in colour are <b>above chance</b>; grey ones are '+
    '<b>present but not above chance</b>. &#9888; marks bodies whose links cross the wide '+
    'gaps. Only objects spanning two or more sections can have a surface at all, so the '+
    D.n_single+' single-section TLS have no body here.'+
    '<br><br>Surface smoothing '+D.smoothing+' voxels; basis <b>G_withdrawn</b>.';
  table();

  // LOADED, NOT SHOWN. The geometry is fetched in the background so the switch is
  // instant when it is pressed; initialising must therefore leave the display OFF,
  // or preloading turns into "the bodies appear by themselves".
  window.__SOLID_ON__ = function(on){ S.on = !!on;window.__SOLID_ACTIVE__=bodiesOn();V.syncOnly();if(!FIG.on && bodiesOn() && glandOn())showGlandProfiles(); V.need(); pumpMeshes(); };
  // The controls may already be set -- the carry-over replays them the moment the
  // page is up, before this payload arrived and before the listeners above
  // existed -- so the state is read FROM the controls now, exactly as they stand.
  // A ticked box then shows its bodies at once, the way a section layer does.
  function syncFromControls(){
    var g = function(id){ return document.getElementById(id); };
    var e;
    if((e = g("objTls"))) S.tls = e.checked;
    if((e = g("objNerve"))) S.nerve = e.checked;
    if((e = g("objNerveLine"))) S.nline = e.checked;
    if((e = g("objGland"))){ S.gland = e.checked;
      e.closest("label").hidden = !(D.meshes || []).some(m => m.structure === "tumor gland"); }
    if((e = g("objDuct"))) S.duct = e.checked;
    // a sample without duct lumens does not show the row
    if((e = g("objDuct")) && e.closest("label"))
      e.closest("label").hidden = !(D.meshes || []).some(function(m){ return m.structure === "duct lumen"; });
    if((e = g("objDen3d"))) S.denClo3 = e.checked;
    if((e = g("objBod"))) S.bod = e.getAttribute("aria-pressed") === "true";
    if((e = g("objPts"))) S.pts = e.getAttribute("aria-pressed") === "true";
    if((e = g("objPt"))) S.ptPx = (parseInt(e.value,10)||10)/10;
    if((e = g("objTa"))) S.tumAlpha = (parseInt(e.value,10)||100)/100;
    if((e = g('objFocusNearby')))S.focusNearby=e.checked;
    if((e = g('objNear'))){S.nearUm=Number(e.value);document.getElementById('objNearValue').textContent=S.nearUm+' µm';}
    if((e = g("objCut"))) S.cut = (parseInt(e.value,10)||100)/100;
    document.querySelectorAll("#objcls .crow[id^='orow-']").forEach(function(b){
      S.cls[b.id.slice(5)] = b.getAttribute("aria-pressed") === "true"; });
    if((e = g("objShow"))) S.on = !!e.checked;
    window.__SOLID_ACTIVE__=bodiesOn();
    V.syncOnly();
    retargetFocus();
  }
  syncFromControls();
  window.__SOLID_SYNC__ = syncFromControls;
  pumpMeshes();                    // the small structures start arriving now
  V.need();
  return true;
};

// Loaded in the background as soon as the page is up, drawn only when the switch is
// pressed. Waiting for the click means the reader stares at a dead button while tens
// of megabytes arrive; fetching it early costs nothing they can see, because it is
// the LAST thing queued and the sections are already on screen by then.
(function(){
  var cb = document.getElementById("objShow");
  if(!cb) return;
  var ready = false, want = false;
  cb.disabled = true;
  cb.addEventListener("change", function(){
    want = cb.checked;
    if(ready) window.__SOLID_ON__(want);
  });
  function start(){
    if(window.__VIEWER_DISPOSING__) return;
    var sc = document.createElement("script");
    sc.src = (window.__DATA_BASE__ || "") + "objects_payload.js?v=" + (window.__SOLID_STAMP__ || 0);
    if(window.__LOADQ__) window.__LOADQ__.note("payload", "3-D payload loading");
    sc.onload = function(){
      if(window.__LOADQ__) window.__LOADQ__.note("payload", "");
      ready = !!window.__SOLID_INIT__();
      cb.disabled = !ready;
      if(ready && want) window.__SOLID_ON__(true);
      if(ready) window.dispatchEvent(new Event("viewer-solid-ready"));
    };
    sc.onerror = function(){
      window.__SOLID_NONE__ = true;
      if(window.__LOADQ__) window.__LOADQ__.note("payload", "");
      cb.title = "no solid bodies have been fitted for this sample yet";
      var side = document.getElementById("objside");
      if(side) side.setAttribute("data-empty", "1"); };
    document.head.appendChild(sc);
  }
  // straight away: the window "load" event waits for every plane and texture the
  // page has already started fetching, so hanging the payload on it puts the 3-D
  // panel minutes behind the sections
  setTimeout(start, 0);
})();
"""


def payload(root: Path | str, sample: str = "HT891Z1",
            assets: Path | None = None, url_prefix: str = "") -> dict:
    root = Path(root)

    def _stash(f: Path) -> str:
        sub = assets / f.parent.parent.name if f.parent.name in ("", ".") else               assets / f.parent.name if f.parent.name.startswith("S1") else               assets / f.parent.name
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / f.name
        st = f.stat()
        if not (dst.exists() and dst.stat().st_size == st.st_size
                and int(dst.stat().st_mtime) == int(st.st_mtime)):
            dst.unlink(missing_ok=True)
            try:
                os.link(f, dst)
            except OSError:
                shutil.copy2(f, dst)
        return (f"{url_prefix}{assets.name}/{sub.name}/{f.name}?v={int(st.st_mtime)}")

    import os as _os
    _obj3d = _os.environ.get("HTAN3D_OBJ3D", "")
    d = root / "objects_3d/outputs/S11_meshes"
    if _obj3d:
        d = Path(_obj3d) / "S11_meshes"
    elif sample != "HT891Z1":
        d = d.parent / f"S11_meshes_{sample}"
    j = d / "meshes.json"
    if not j.exists():
        return {}
    rep = json.loads(j.read_text())
    out, faces = [], 0
    for m in rep["meshes"]:


        p = d / m["file"]
        if not p.exists():
            continue
        e = dict(m)
        if assets is not None:
            e["url"] = _stash(p)
        else:
            e["b64"] = base64.b64encode(p.read_bytes()).decode("ascii")
        e.pop("file", None)
        out.append(e)
        faces += m["n_faces"]


    for base, key, tag in (("S13_structures_3d", "structures", "3d cluster"),
                           ("S14_nerve_3d", "nerves", "nerve"),
                           ("S15_duct_3d", "lumens", "duct lumen"),
                           ("S16_tumor_glands", "glands", "tumor gland")):
        sd = root / "objects_3d/outputs" / base
        if _obj3d:
            sd = Path(_obj3d) / base
        elif sample != "HT891Z1":
            sd = sd.parent / f"{base}_{sample}"
        if base == "S16_tumor_glands" and _os.environ.get("HTAN3D_GLANDS"):
            sd = Path(_os.environ["HTAN3D_GLANDS"])
        if not sd.is_dir():
            continue
        for j in sorted(sd.glob("*.json")):
            rep = json.loads(j.read_text())
            for m in rep.get(key, []):
                if "file" not in m:
                    continue
                f = sd / m["file"]
                if not f.exists():
                    continue
                e = dict(m)
                if assets is not None:
                    e["url"] = _stash(f)
                else:
                    e["b64"] = base64.b64encode(f.read_bytes()).decode("ascii")
                e.pop("file", None)
                e["structure"] = tag
                out.append(e)
                faces += m["n_faces"]

    cloud = {}


    cd = (Path(_obj3d) / "S12_cloud") if _obj3d else root / "objects_3d/outputs/S12_cloud"
    if not _obj3d and sample != "HT891Z1":
        cd = cd.parent / f"S12_cloud_{sample}"
    if (cd / "cloud.json").exists() and (cd / "cloud.bin").exists():
        cj = json.loads((cd / "cloud.json").read_text())
        cloud = {"classes": cj["classes"], "n_planes": cj["n_planes"],
                 "um_px": 8.0, "per_class": cj["per_class"]}
        if assets is not None:
            cloud["url"] = _stash(cd / "cloud.bin")
        else:
            cloud["b64"] = base64.b64encode(
                (cd / "cloud.bin").read_bytes()).decode("ascii")
    def _cl(stem):
        if (cd / f"{stem}.json").exists() and (cd / f"{stem}.bin").exists():
            nj = json.loads((cd / f"{stem}.json").read_text())
            r2 = {"classes": nj["classes"], "n_planes": nj["n_planes"],
                  "um_px": nj["um_px"]}
            if assets is not None:
                r2["url"] = _stash(cd / f"{stem}.bin")
            else:
                r2["b64"] = base64.b64encode(
                    (cd / f"{stem}.bin").read_bytes()).decode("ascii")
            return r2
        return None
    cloud2 = _cl("nerve_cloud")
    cloud3 = _cl("denoised_cloud")
    cols = {}
    cmj = root / "reconstruction/volume_8um/cells_metadata.json"
    if sample != "HT891Z1":
        alt = root / f"reconstruction/volume_8um_{sample[-3:]}/cells_metadata.json"
        if alt.exists():
            cmj = alt
    if cmj.exists():
        cols = {c["name"]: c["colour"] for c in json.loads(cmj.read_text())["classes"]}
    return {"meshes": out, "cloud": cloud, "cloud_denoised": cloud2,
            "cloud_denoised3d": cloud3,
            "class_colours": cols,
            "n_faces_total": faces,
            "claim": "MODEL PREDICTION on H&E (pan-cancer Cell head, single fold). "
                     "No spatial ground truth in this cohort."}
