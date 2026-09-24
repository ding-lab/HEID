#!/usr/bin/env python

CELL_PX = 36
CELL_H = 40

CSS = """
:root{
  --ink:#12161A; --paper:#F2F5F6; --rule:#C8D2D6; --muted:#5C696F;
  --accent:#2F6F7D; --warn:#9E4726; --panel:#E8EDEF; --canvasbg:#0A0D10;
  --m-he:#9AA7AD; --m-codex:#C3CCD1; --m-xenium:#737F85; --cell-ink:#12161A;
  --oncanvas:#C6D0D4; --oncanvas-bg:rgba(0,0,0,.45);
  /* the corner sits on a dark slab whatever the page theme is, so its warn
     colour is fixed rather than following --warn (which is a dark brown in
     the light theme and would read at 2.9:1 there) */
  --warn-on-dark:#E2814F;
}
@media (prefers-color-scheme: dark){
  :root{
    --ink:#E4EAEC; --paper:#0F1418; --rule:#2C363B; --muted:#93A2A8;
    --accent:#5FB3C4; --warn:#E2814F; --panel:#161C21; --canvasbg:#05070A;
    --m-he:#4A565C; --m-codex:#5E6C73; --m-xenium:#38434A; --cell-ink:#E4EAEC;
  }
}
:root[data-theme="dark"]{
  --ink:#E4EAEC; --paper:#0F1418; --rule:#2C363B; --muted:#93A2A8;
  --accent:#5FB3C4; --warn:#E2814F; --panel:#161C21; --canvasbg:#05070A;
  --m-he:#4A565C; --m-codex:#5E6C73; --m-xenium:#38434A; --cell-ink:#E4EAEC;
}
:root[data-theme="light"]{
  --ink:#12161A; --paper:#F2F5F6; --rule:#C8D2D6; --muted:#5C696F;
  --accent:#2F6F7D; --warn:#9E4726; --panel:#E8EDEF; --canvasbg:#0A0D10;
  --m-he:#9AA7AD; --m-codex:#C3CCD1; --m-xenium:#737F85; --cell-ink:#12161A;
}
:root[data-bg="light"]{ --oncanvas:#23292D; --oncanvas-bg:rgba(255,255,255,.62); }
:root[data-bg="mid"]{ --oncanvas:#0E1215; --oncanvas-bg:rgba(255,255,255,.5); }
*{box-sizing:border-box}
/* A viewer is a tool, not a document: the page itself never scrolls. Only the
   control flyout and the layer strip scroll, inside themselves. 100dvh rather than
   100vh so a mobile URL bar cannot push the bottom row out of reach. */
html,body{height:100%;margin:0;padding:0;overflow:hidden;overscroll-behavior:none;
  background:var(--paper);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:13px}
#app{height:100dvh;display:grid;grid-template-rows:auto 1fr auto;overflow:hidden}
.mono{font-family:ui-monospace,"SF Mono",Consolas,monospace;
  font-variant-numeric:tabular-nums}
#topbar{height:34px;display:flex;align-items:center;gap:8px;padding:0 8px;
  background:var(--panel);border-bottom:1px solid var(--rule);overflow:hidden;
  white-space:nowrap;font-family:ui-monospace,Consolas,monospace;font-size:12px}
.shotrow{display:flex;gap:6px;margin-top:2px}
.shotcard{flex:1 1 0;min-height:30px;font-size:12px;letter-spacing:.04em;
  font-family:ui-monospace,Consolas,monospace}
#topbar button{min-height:26px;height:26px;padding:0 8px;font-size:13px;line-height:1}
#topenc{color:var(--muted);margin-left:auto;padding-left:10px}
#progress{color:var(--muted);min-width:0;overflow:hidden;text-overflow:ellipsis}
#labelToggle{display:flex;flex:0 0 auto;align-items:center;gap:4px;white-space:nowrap;color:var(--accent);cursor:pointer}
#topbar>#infoBtn,#topbar>#resetBtn{flex:0 0 auto}
#topbar>#modeTag{min-width:0;overflow:hidden;text-overflow:ellipsis;flex:0 1 auto}
#banner{font-family:ui-monospace,"SF Mono",Consolas,monospace;
  font-variant-numeric:tabular-nums;font-size:12px;line-height:1.5;white-space:pre-wrap}
#banner .w{color:var(--warn);font-weight:600}
#infoPanel{position:fixed;left:10px;top:42px;width:min(780px,calc(100vw - 26px));
  max-height:calc(100dvh - 86px);overflow:auto;overscroll-behavior:contain;
  background:var(--paper);border:1px solid var(--rule);padding:12px 14px;z-index:60;
  display:none;box-shadow:0 8px 24px rgba(0,0,0,.28)}
#errs{position:absolute;left:10px;right:10px;bottom:10px;color:#fff;
  background:var(--warn);font-family:ui-monospace,Consolas,monospace;font-size:12px;
  white-space:pre-wrap;padding:4px 8px;border-radius:2px;display:none;z-index:8;
  max-height:32%;overflow:auto}
#wrap{display:grid;grid-template-columns:1fr 44px;min-height:0;position:relative;
  overflow:hidden}
#left{display:flex;flex-direction:column;min-width:0;min-height:0}
#rail{border-left:1px solid var(--rule);background:var(--panel);display:flex;
  flex-direction:column;gap:2px;padding:4px 0;overflow:hidden}
.railbtn{width:40px;height:38px;margin:0 auto;border:1px solid transparent;
  background:transparent;color:var(--ink);font-size:15px;line-height:1;cursor:pointer;
  border-radius:2px;display:flex;align-items:center;justify-content:center;padding:0}
.railbtn:hover{background:var(--paper);border-color:var(--rule)}
.railbtn[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:#fff}
#bottom{background:var(--panel);overflow:hidden}
/* Two layouts. Windowed keeps the ordinary one, where the strip
   holds its own row. Full screen goes immersive: the canvas takes the whole screen
   and the strip floats, retracted to a 3 px hint line until the pointer goes for it. */
#app[data-fs="1"]{grid-template-rows:auto 1fr}
/* the bar stays above the canvas in full screen, and stays clickable: the stage
   is positioned there, so without a stacking context of its own the bar ends up
   underneath it and the sample control cannot be reached at all */
#app[data-fs="1"] #topbar{position:relative;z-index:40}
#app[data-fs="1"] #bottom{position:absolute;left:0;right:0;bottom:0;z-index:15;
  border-top:3px solid var(--rule);
  transform:translateY(calc(100% - 3px));
  transition:transform 180ms cubic-bezier(.22,.61,.36,1)}
#app[data-fs="1"][data-strip="open"] #bottom{transform:translateY(0)}
/* 10(c): the strip must not cover the corner readouts, so they step up by exactly the
   height of the strip, measured when it opens and written into --striph. */
#app[data-fs="1"][data-strip="open"] #corner{bottom:calc(var(--striph,0px) + 8px)}
#app[data-fs="1"] #stage{position:relative}
@media (prefers-reduced-motion: reduce){#app[data-fs="1"] #bottom{transition:none}}
#stage{flex:1;position:relative;background:var(--canvasbg);min-height:0}
#gl{width:100%;height:100%;display:block}
/* A click must not draw a focus ring, but a keyboard user must see where focus is.
   :focus-visible is exactly that distinction; outline:none alone would take the ring
   away from the keyboard too. The ring is drawn INSIDE the element so no ancestor
   with overflow:hidden can clip it. */
/* :focus-visible is a browser HEURISTIC, and this page defeats it: it moves focus to
   the canvas from script on pointerdown (so the keyboard still works after a click),
   which Chrome treats as a keyboard focus and rings. So the ring is driven by a class
   this page sets itself, and no :focus-visible rule is left to fire behind it. */
#gl:focus,#selector:focus,.cell:focus{outline:none}
#gl.kbfocus,#selector.kbfocus,.cell.kbfocus{outline:2px solid var(--accent);
  outline-offset:-2px}
#warnbadges{display:flex;flex-direction:column;gap:4px;margin-bottom:8px}
/* the corner lines sit on the picture, so the background is dark and the warning is
   carried by the text colour and a 2 px bar, not by a slab of orange */
.wbadge{color:var(--warn-on-dark);background:rgba(18,22,26,.82);font-weight:700;
  border-left:2px solid var(--warn-on-dark);
  font-family:ui-monospace,Consolas,monospace;font-size:12px;padding:4px 8px;
  border-radius:2px}
#chips{position:absolute;right:10px;top:8px;display:flex;flex-direction:column;
  gap:4px;align-items:flex-end;pointer-events:none}
/* the slab is a fixed dark colour, so the text colour must be fixed light as well.
   A theme token would flip with the theme and give dark-on-dark, which is
   unreadable. Same reason --warn-on-dark exists. */
.chip{background:rgba(18,22,26,.82);color:#EAF0F2;
  font-family:ui-monospace,Consolas,monospace;font-size:11.5px;padding:4px 8px;
  border-radius:2px}
#corner{position:absolute;left:10px;bottom:8px;color:#EAF0F2;
  background:rgba(18,22,26,.82);font-family:ui-monospace,Consolas,monospace;
  font-variant-numeric:tabular-nums;font-size:12px;padding:4px 8px;
  border-radius:2px;white-space:pre;pointer-events:none}
/* the only-this-layer switch sits directly above the corner readout, so what it acts
   on is the layer named one line below it. It is in the DOM only while something is
   selected. */
/* the switch lives in the Display group, not on the picture: the canvas carries
   only the readout line and the warning badges */
#onlyBtn{font-family:ui-monospace,Consolas,monospace;font-size:12px;
  padding:3px 8px;border-radius:2px;cursor:pointer;
  /* it sits in the panel, so both colours are theme tokens and flip together;
     a fixed dark slab with a theme-token text colour would be unreadable. */
  background:var(--paper);color:var(--ink);border:1px solid var(--rule)}
#onlyBtn[aria-pressed=true]{background:var(--accent);border-color:var(--accent);
  color:#fff}
#planetag{position:absolute;color:var(--oncanvas);background:var(--oncanvas-bg);
  font-family:ui-monospace,Consolas,monospace;font-size:11.5px;padding:1px 5px;
  border-radius:2px;pointer-events:none;display:none;border:1px solid var(--accent)}
#overlay{position:absolute;inset:0;pointer-events:none}
.glab{position:absolute;color:var(--oncanvas);background:var(--oncanvas-bg);
  font-family:ui-monospace,Consolas,monospace;font-size:10px;padding:0 3px}
.gsel{position:absolute;border:2px solid var(--accent)}
.ggap{position:absolute;background:var(--warn)}
#recutInfo{position:absolute;right:10px;bottom:8px;color:var(--oncanvas);
  background:var(--oncanvas-bg);font-family:ui-monospace,Consolas,monospace;
  font-size:11px;line-height:1.45;padding:4px 8px;border-radius:2px;white-space:pre;
  display:none}
#recutInfo .w{color:var(--warn);font-weight:700}
#hover{position:absolute;color:var(--oncanvas);background:var(--oncanvas-bg);
  font-family:ui-monospace,Consolas,monospace;font-size:11px;padding:1px 5px;
  border-radius:2px;pointer-events:none;display:none}
#scalebar{position:absolute;right:12px;bottom:8px;z-index:6;pointer-events:none;
  display:flex;flex-direction:column;align-items:flex-end;gap:3px;
  background:rgba(18,22,26,.82);padding:5px 8px;border-radius:2px}
#sbline{height:0;border-bottom:2px solid #EAF0F2}
#sblab{font-size:11px;font-family:ui-monospace,Consolas,monospace;color:#EAF0F2;
  font-variant-numeric:tabular-nums}
#rulerbox{border-top:1px solid var(--rule);background:var(--panel);padding:0 12px;
  position:relative;cursor:default}
/* the ruler is as wide as the whole strip and scrolls WITH it (its scroll is slaved
   to the strip's), so the far end is never cut off; the top padding lives inside the
   scroll box so the z labels above the ticks are not clipped */
#rulerscroll{overflow:hidden;width:100%;padding-top:14px}
#ruler{position:relative;height:34px;border-bottom:1px solid var(--rule)}
#ruler *{pointer-events:none}
.rtick{position:absolute;top:6px;width:1px;height:10px;background:var(--muted);opacity:.9}
.rtick2{position:absolute;top:22px;width:1px;height:8px;background:transparent;
  border-left:1px dashed var(--muted);opacity:.75}
.rlink{position:absolute;top:16px;height:6px;border-left:1px solid var(--rule)}
.rgap{position:absolute;top:5px;height:11px;background:var(--warn);opacity:.95}
.rgaplab{position:absolute;top:19px;font-size:11px;color:var(--warn);
  font-family:ui-monospace,Consolas,monospace;transform:translateX(-50%);white-space:nowrap}
.rzlab{position:absolute;top:-13px;font-size:11px;color:var(--muted);
  font-family:ui-monospace,Consolas,monospace;transform:translateX(-50%)}
#rcur{position:absolute;top:2px;width:2px;height:18px;background:var(--accent)}
#rhint{position:absolute;top:-13px;font-size:11px;color:var(--ink);
  font-family:ui-monospace,Consolas,monospace;transform:translateX(-50%);display:none;
  background:var(--paper);padding:0 3px;border:1px solid var(--rule);border-radius:2px}
#guidebox{height:16px;position:relative;background:var(--panel);padding:0 12px}
#guide{position:absolute;left:12px;top:0;width:calc(100% - 24px);height:16px;overflow:visible}
#stripbox{background:var(--panel);padding:0 12px 6px}
/* The strip scrolls - 50 cells are wider than any window - but its scrollbar is not
   drawn: a horizontal bar across the bottom of the page reads as "the page is
   scrolling", which is not the intended reading. The scrolling itself,
   the drag-scrub and the keyboard navigation are untouched.
   A scrollbar is also a signal that there is more to see, so that job moves to a
   16 px fade at each end, dropped on whichever end is already at the stop. It takes
   no space and cannot be clicked, and it is not an arrow button. */
#selector{overflow-x:auto;overflow-y:hidden;border:1px solid var(--rule);
  background:var(--paper);scrollbar-width:none;-ms-overflow-style:none;
  --mL:0px;--mR:0px;
  -webkit-mask-image:linear-gradient(to right,transparent 0,#000 var(--mL),
    #000 calc(100% - var(--mR)),transparent 100%);
  mask-image:linear-gradient(to right,transparent 0,#000 var(--mL),
    #000 calc(100% - var(--mR)),transparent 100%)}
#selector::-webkit-scrollbar{display:none;width:0;height:0}
#cells{display:flex;flex-direction:row;width:max-content;height:__CELLH__px}
.cell{flex:0 0 __CELLW__px;flex-shrink:0;width:__CELLW__px;min-width:__CELLW__px;
  height:__CELLH__px;border-right:1px solid var(--rule);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  cursor:pointer;gap:1px;
  font-family:ui-monospace,Consolas,monospace;font-size:11px;color:var(--cell-ink);
  user-select:none;position:relative}
.cell .cmod{font-size:7px;line-height:1;letter-spacing:.02em;opacity:.72;
  text-transform:uppercase}
.cell[data-mod="he"]{background:var(--m-he)}
.cell[data-mod="codex"]{background:var(--m-codex)}
.cell[data-mod="xenium"]{background:var(--m-xenium)}
.cell:hover{filter:brightness(1.12)}
.cell.sel{outline:2px solid var(--accent);outline-offset:-2px;font-weight:700;
  background:var(--accent);color:#fff;box-shadow:inset 0 0 0 1px var(--paper)}
.cell.sel::before{content:"";position:absolute;left:2px;top:2px;width:4px;height:4px;
  border-radius:50%;background:var(--paper)}
.cell.cur{outline:2px dashed var(--accent);outline-offset:-2px}
.cell.hid{opacity:.35;text-decoration:line-through}
.cell.err::after{content:"\\00d7";position:absolute;right:2px;top:0;color:var(--warn);
  font-weight:700;font-size:14px}
.gapmark{flex:0 0 16px;flex-shrink:0;width:16px;height:__CELLH__px;position:relative;
  border-right:2px solid var(--warn);background:transparent}
.gapmark span{position:absolute;left:50%;top:50%;
  transform:translate(-50%,-50%) rotate(-90deg);
  font-size:12px;color:var(--warn);font-family:ui-monospace,Consolas,monospace;
  white-space:nowrap}
#navrow{display:flex;align-items:center;gap:4px;padding:0;flex-wrap:wrap}
#navrow .cur{font-family:ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums;
  font-size:13px}
#panel{position:absolute;right:44px;top:0;bottom:0;width:302px;padding:10px 12px;
  overflow-y:auto;overscroll-behavior:contain;background:var(--paper);
  border-left:1px solid var(--rule);display:none;z-index:12;
  box-shadow:-6px 0 14px rgba(0,0,0,.20)}
#panel .grp{display:none}
#panel .grp.open{display:block}
.grp{margin-bottom:10px;border-bottom:1px solid var(--rule);padding-bottom:9px}
.grp:last-child{border-bottom:none}
.grp>label,.grp>.hd{display:block;font-size:12px;color:var(--muted);margin-bottom:4px;
  text-transform:uppercase;letter-spacing:.04em}
details>summary{font-size:12px;color:var(--muted);text-transform:uppercase;
  letter-spacing:.04em;cursor:pointer;min-height:32px;display:flex;align-items:center}
.seg{display:flex;flex-wrap:wrap;gap:4px}
button,select{font:inherit;color:var(--ink)}
button{background:transparent;border:1px solid var(--rule);border-radius:2px;
  padding:0 10px;min-height:32px;cursor:pointer;font-size:12px}
button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:#fff}
button:disabled{opacity:.45;cursor:not-allowed}
button:focus,select:focus,input:focus,summary:focus,[tabindex]:focus{outline:none}
button.kbfocus,select.kbfocus,input.kbfocus,summary.kbfocus,[tabindex].kbfocus{
  outline:2px solid var(--accent);outline-offset:-2px}
input[type=range]{width:100%;height:32px;accent-color:var(--accent)}
input[type=number]{width:78px;min-height:32px;font:inherit;
  font-family:ui-monospace,Consolas,monospace}
.note{font-size:12px;color:var(--muted);margin-top:4px;line-height:1.45}
.val{font-family:ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums;
  color:var(--ink)}
#limits{font-size:11px;color:var(--muted);font-family:ui-monospace,Consolas,monospace;
  padding:6px 12px;border-top:1px solid var(--rule);white-space:pre-wrap}
#keyhelp{position:fixed;inset:8% 18%;background:var(--paper);border:1px solid var(--rule);
  padding:14px 18px;overflow:auto;z-index:50;display:none;
  font-family:ui-monospace,Consolas,monospace;font-size:12px;white-space:pre-wrap}
/* Display is three groups, and which controls are mutually exclusive has to be
   readable without trying them. A .sub is a group: separated from its neighbours by a
   rule and 10 px, while the controls INSIDE one sit 4 px apart. Same-spacing-
   everywhere is exactly what made the flat column unreadable. */
.sub{margin-top:10px;padding-top:9px;border-top:1px solid var(--rule)}
.sub:first-of-type{margin-top:0;padding-top:0;border-top:none}
.sublab{display:block;font-size:11px;color:var(--muted);margin-bottom:4px;
  text-transform:uppercase;letter-spacing:.04em}
/* A pick-one group is ONE control cut into halves or thirds, not two or three
   buttons standing next to each other: no gaps inside, shared edges, and only the
   chosen segment carries the filled background that button[aria-pressed] already
   gives it. */
.segjoin{display:flex;gap:0;width:100%}
.segjoin button{flex:1 1 0;min-width:0;border-radius:0;margin-left:-1px;padding:0 6px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.segjoin button:first-child{margin-left:0;border-radius:2px 0 0 2px}
.segjoin button:last-child{border-radius:0 2px 2px 0}
.segjoin button[aria-pressed=true]{position:relative;z-index:1}
/* The overlay switch must NOT look like a segment, because it is not part of any
   pick-one group: full width, left aligned, and a box that fills when it is on. */
.ovbtn{display:flex;align-items:center;gap:8px;width:100%;text-align:left}
.ovbtn::before{content:"";flex:0 0 12px;width:12px;height:12px;border-radius:2px;
  border:1px solid currentColor}
.ovbtn[aria-pressed=true]::before{background:currentColor}
/* a switch that belongs to the switch above it: indented, and it goes with it */
.ovsub{margin-top:3px;width:calc(100% - 18px);margin-left:18px}
/* The cell-prediction layer's class list. A row is a switch, a colour swatch and a
   count, and the swatch is the ONLY place a class colour appears in the interface,
   so what a colour on the picture means is never in doubt. The rows are ordinary
   segment-style buttons rather than a new control idiom. The list belongs to the
   overlay switch above it, so with the overlay off it is dimmed AND disabled. */
#cellsBox{margin-top:6px}
#cellsBox[data-on="0"]{display:none}
#figBody{min-width:0;margin:0;padding:0;border:0}
#figBody:disabled{opacity:.55}
#figObjs .figpick{border:0;background:transparent;color:inherit;font:inherit;text-align:left;flex:1;min-width:0;padding:3px 4px;cursor:pointer}
#figObjs .figpick:hover{background:rgba(47,111,125,.12)}
#figObjs .figpick:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.figselection{justify-content:space-between;margin:5px 0;font-size:11px}
#figPanel{margin-top:10px;border-top:1px solid var(--rule);padding-top:8px}
#figPanel select{width:100%;margin-bottom:4px}
#figSheet{width:100%;display:flex;align-items:center;gap:12px;text-align:left;
  padding:12px;margin:2px 0 5px;border:1px solid var(--accent);border-radius:5px;
  color:var(--accent);background:color-mix(in srgb,var(--accent) 8%,var(--paper));}
#figSheet:not(:disabled):hover{background:color-mix(in srgb,var(--accent) 16%,var(--paper))}
#figSheet:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
#figSheet:disabled{opacity:.5;cursor:default}
#figSheet svg{flex:0 0 26px;width:26px;height:32px}
#figSheet strong{display:block;font-size:14px;line-height:1.4;font-weight:650}
#figSheet small{display:block;font-size:11px;line-height:1.5;color:var(--muted);font-weight:400}
#figSheetNote{font-size:11px;line-height:1.5;color:var(--muted);margin-bottom:12px;overflow-wrap:anywhere}
.figbtn{font-size:11px;padding:1px 8px}
#figObjs{max-height:230px;overflow:auto}
#figObjs .fgrp{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin:6px 0 2px}
#figObjs .figobj{display:flex;align-items:center;gap:6px;font-size:11.5px;padding:1px 2px;cursor:pointer}
#figObjs .figobj:hover{background:color-mix(in srgb, var(--accent) 14%, transparent)}
#figObjs .figobj.figsel,#figObjs .figobj.figsel:hover{background:var(--accent);color:var(--paper);font-weight:600;border-radius:3px}
#xgtBody[data-on="0"]{display:none}
/* A switch that decides whether a whole layer is drawn at all -- not one more
   line item in a column of checkboxes. Both the sections and the solid bodies use
   it, so the two read as the same kind of control. */
.showbar{display:flex;align-items:center;gap:8px;cursor:pointer;
  margin:2px 0 8px;padding:7px 10px;border-radius:3px;
  border:1px solid var(--accent);
  background:color-mix(in srgb, var(--accent) 10%, transparent);
  font-weight:700;font-size:12px;letter-spacing:.02em;color:var(--accent)}
.showbar:hover{background:color-mix(in srgb, var(--accent) 18%, transparent)}
.hdrow{display:flex;align-items:baseline;gap:10px;flex-wrap:nowrap}
.hdrow .hd{margin:0}
.hdrow .sublab{margin:0;white-space:nowrap}
#secRow .showbar{margin:2px 0 8px}
/* the TLS row: the layer switch, then two pill toggles in the figure colours --
   filled when on, outlined when off */
.pillrow{gap:8px}
.pillrow>span{flex:1 1 auto}
.pillrow .kind{display:inline-flex;align-items:center;margin:0;padding:2px 10px;border-radius:999px;
  border:1.5px solid var(--kc);color:var(--kc);background:transparent;font-size:11.5px;font-weight:700;
  letter-spacing:.03em;cursor:pointer;transition:background .12s,color .12s;user-select:none}
.pillrow .kind:has(input:checked){background:var(--kc);color:#fff}
.pillrow .kind input{position:absolute;opacity:0;width:0;height:0;margin:0}
.pillrow .kind:hover{filter:brightness(1.1)}
.showbar:has(> input:checked){background:var(--accent);color:#fff}
.pillrow:not(:has(> input:checked)) .kind{opacity:.45;pointer-events:none}
.showbar input{accent-color:currentColor;width:13px;height:13px;margin:0}
#heSwapBox[data-empty="1"]{opacity:.45;pointer-events:none}
#heSwapBox[data-empty="1"]::after{content:"Only one modality on this sample.";font-size:11.5px;color:var(--muted,#777)}
#heSwapBox[data-on="0"] #heSwapSeg{
  opacity:.35;pointer-events:none}
#cellList,#xgtList{display:flex;flex-direction:column;gap:2px;margin-top:4px}
#xgtBox[hidden]{display:none}
#xgtList .crow .cname{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.crow{display:flex;align-items:center;gap:6px;width:100%;min-height:26px;
  padding:0 4px;text-align:left;font-size:12px}
.crow[aria-pressed=true]{background:var(--panel);border-color:var(--rule);
  color:var(--ink)}
.csw{flex:0 0 12px;width:12px;height:12px;border-radius:2px;border:1px solid var(--rule)}
input.cpick{-webkit-appearance:none;appearance:none;padding:0;background:none;cursor:pointer}
input.cpick::-webkit-color-swatch-wrapper{padding:0}
input.cpick::-webkit-color-swatch{border:none;border-radius:2px}
input.cpick::-moz-color-swatch{border:none;border-radius:2px}
.crow[aria-pressed=false] .csw{opacity:.25}
.cname{flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ccount{font-family:ui-monospace,Consolas,monospace;font-size:11px;color:var(--muted)}
.predclaim{font-size:11px;color:var(--warn);font-weight:700;line-height:1.35;
  letter-spacing:.02em}
@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}
""".replace("__CELLW__", str(CELL_PX)).replace("__CELLH__", str(CELL_H))


BROWSER_CSS = """
#samplebar{display:flex;align-items:center;gap:6px;flex:0 0 auto;min-width:0}
#sampleLab{color:var(--muted)}
/* A list, not a <select>. Twenty-seven rows each carrying four facts do not fit a
   one-line option, and the reader needs to find one by name among them, so the
   control is a button that opens a filtered listbox. */
#sampleBtn{max-width:min(42vw,380px);min-height:26px;height:26px;font-size:12px;
  font-family:ui-monospace,Consolas,monospace;background:var(--paper);
  border:1px solid var(--rule);border-radius:2px;padding:0 8px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:left}
#sampleBtn[aria-expanded=true]{border-color:var(--accent);color:var(--accent)}
#sampleBtn::after{content:" \\25BE"}
#samplePop{position:fixed;left:8px;top:38px;width:min(560px,calc(100vw - 20px));
  max-height:min(70dvh,560px);display:none;flex-direction:column;z-index:70;
  background:var(--paper);border:1px solid var(--rule);border-radius:2px;
  box-shadow:0 10px 28px rgba(0,0,0,.32);padding:8px}
#samplePop[data-open="1"]{display:flex}
#sampleFilter{min-height:30px;font:inherit;font-family:ui-monospace,Consolas,monospace;
  font-size:12px;background:var(--paper);color:var(--ink);
  border:1px solid var(--rule);border-radius:2px;padding:0 6px;width:100%}
#sampleList{overflow-y:auto;overscroll-behavior:contain;margin-top:6px;min-height:0;
  display:flex;flex-direction:column;gap:1px}
#sampleFoot{margin-top:6px;border-top:1px solid var(--rule);padding-top:5px}
.srow{display:flex;align-items:center;gap:8px;width:100%;text-align:left;
  min-height:30px;padding:0 6px;font-size:12px;border:1px solid transparent;
  background:transparent;color:var(--ink);cursor:pointer;border-radius:2px}
.srow:hover{background:var(--panel);border-color:var(--rule)}
.srow[aria-selected=true]{background:var(--accent);border-color:var(--accent);color:#fff}
.srow.cursor{border-color:var(--accent)}
.srow .sid{flex:0 0 84px;font-family:ui-monospace,Consolas,monospace;font-weight:700}
.srow .sn{flex:1 1 auto;color:var(--muted);font-family:ui-monospace,Consolas,monospace;
  font-size:11.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.srow[aria-selected=true] .sn{color:#EAF0F2}
.srow .skind{flex:0 0 auto;font-size:11px;font-family:ui-monospace,Consolas,monospace;
  padding:1px 6px;border-radius:2px;border:1px solid currentColor}
.srow .skind[data-mode="volume"]{color:var(--accent)}
.srow .skind[data-mode="section"]{color:var(--warn)}
.srow[aria-selected=true] .skind{color:#fff}
/* What kind of thing is on screen. It is a fact about the data, not a setting, so
   it is a tag and not a button, and it is coloured by which of the two it is. */
#modeTag{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;padding:2px 7px;
  border-radius:2px;white-space:nowrap;border:1px solid var(--rule);color:var(--ink)}
#modeTag[data-mode="volume"]{border-color:var(--accent);color:var(--accent)}
#modeTag[data-mode="section"]{border-color:var(--warn);color:var(--warn);font-weight:700}
#browser{display:none;min-height:0;min-width:0;overflow:hidden}
#app[data-view="section"] #wrap,#app[data-view="section"] #bottom{display:none}
#app[data-view="section"] #browser{display:grid;
  grid-template-columns:minmax(0,1fr) 300px;grid-template-rows:minmax(0,1fr) auto;
  min-height:0;overflow:hidden}
/* white, because that is what the glass is: a tile with no tissue on it is not
   written at all, so the background has to be the same colour the blank tiles
   would have been or the section would sit in a box of a different shade */
/* Every cell is placed explicitly. Auto-placement runs items with a definite ROW
   before fully automatic ones, so the side panel -- which spans both rows -- was
   taking column 1 and leaving the picture the 300 px meant for the panel. The
   canvas came out 300 px wide and the whole slide was drawn at 33 um/px. */
#bstage{grid-column:1;grid-row:1;
  position:relative;background:#fff;min-width:0;min-height:0;overflow:hidden}
#bgl{width:100%;height:100%;display:block;touch-action:none;cursor:grab}
#bgl.drag{cursor:grabbing}
#bgl:focus{outline:none}
#bgl.kbfocus{outline:2px solid var(--accent);outline-offset:-2px}
#bnotice{position:absolute;left:10px;top:8px;max-width:calc(100% - 24px);
  color:#fff;background:rgba(158,71,38,.94);font-family:ui-monospace,Consolas,monospace;
  font-size:11.5px;font-weight:700;padding:4px 8px;border-radius:2px;
  pointer-events:none;line-height:1.4}
#bcorner{position:absolute;left:10px;bottom:8px;color:#EAF0F2;
  background:rgba(18,22,26,.82);font-family:ui-monospace,Consolas,monospace;
  font-variant-numeric:tabular-nums;font-size:12px;padding:4px 8px;border-radius:2px;
  white-space:pre;pointer-events:none}
#bchips{position:absolute;right:10px;top:8px;display:flex;flex-direction:column;gap:4px;
  align-items:flex-end;pointer-events:none}
#bscale{position:absolute;right:10px;bottom:8px;color:#EAF0F2;
  background:rgba(18,22,26,.82);font-family:ui-monospace,Consolas,monospace;
  font-size:11.5px;padding:4px 8px;border-radius:2px;pointer-events:none;
  display:flex;align-items:center;gap:6px}
#bscale i{display:block;height:3px;background:#EAF0F2}
#bside{grid-column:2;grid-row:1 / span 2;
  border-left:1px solid var(--rule);background:var(--paper);
  padding:8px 10px;overflow-y:auto;overscroll-behavior:contain;min-height:0}
#bbar{grid-column:1;grid-row:2;
  border-top:1px solid var(--rule);background:var(--panel);padding:6px 10px;
  display:flex;align-items:center;gap:8px;flex-wrap:wrap;min-width:0}
#bbar .cur{font-family:ui-monospace,Consolas,monospace;font-variant-numeric:tabular-nums;
  font-size:12px}
#bbar button{min-height:28px;height:28px;font-size:12px}
/* The section list is grouped by TISSUE BLOCK, and the grouping is load-bearing:
   sections from two blocks are two different stacks of tissue, so they get two
   headed lists and never one running sequence. */
.bblock{margin-bottom:8px}
.bblock>.bh{display:flex;align-items:baseline;gap:6px;font-size:11px;color:var(--muted);
  text-transform:uppercase;letter-spacing:.04em;margin:0 0 3px;
  border-bottom:1px solid var(--rule);padding-bottom:2px}
.bblock>.bh b{color:var(--ink);letter-spacing:0;text-transform:none;
  font-family:ui-monospace,Consolas,monospace;font-size:12px}
.bsecs{display:flex;flex-wrap:wrap;gap:3px}
.bsec{font-family:ui-monospace,Consolas,monospace;font-size:11px;min-height:24px;
  padding:0 6px;border:1px solid var(--rule);border-radius:2px;background:transparent;
  color:var(--ink);cursor:pointer;position:relative}
.bsec[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:#fff}
/* a section where the region operator found nerve regions carries a dot, so the
   ones worth opening are visible without clicking through 63 of them */
.bsec.nerve::after{content:"";position:absolute;right:2px;top:2px;width:4px;height:4px;
  border-radius:50%;background:var(--warn)}
.bsec[aria-pressed=true].nerve::after{background:#fff}
/* which assays this section carries, as initials under the section number. A
   section with two is the interesting case, so it is also outlined. */
.bsec{padding-bottom:9px}
.bsec .modtag{position:absolute;left:0;right:0;bottom:1px;font-style:normal;
  font-size:8px;letter-spacing:.5px;color:var(--muted);text-align:center;
  font-family:ui-monospace,Consolas,monospace}
.bsec[aria-pressed=true] .modtag{color:#fff}
.bsec.multi{border-color:var(--accent);border-width:2px}
#bclasses{display:flex;flex-direction:column;gap:2px;margin-top:4px}
#binfo{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;color:var(--ink);
  line-height:1.5;white-space:pre-wrap;margin-top:4px}
#bwarn{font-size:11px;color:var(--warn);font-weight:700;line-height:1.35;margin-top:6px}
/* 37 markers, each assignable to one of three composite slots. A list with the
   slots ON each row answers "where do I put CD8" in one click; a dropdown per slot
   would answer "what goes in red" instead, which is not the question. */
#bchanList{max-height:min(46dvh,340px);overflow-y:auto;overscroll-behavior:contain;
  border:1px solid var(--rule);border-radius:2px;margin-top:4px}
.chanrow{display:flex;align-items:center;gap:4px;padding:1px 4px;min-height:24px;
  border-bottom:1px solid var(--rule)}
.chanrow:last-child{border-bottom:none}
.channame{flex:1 1 auto;font-size:11.5px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap}
.chanwin{flex:0 0 auto;font-family:ui-monospace,Consolas,monospace;font-size:10px;
  color:var(--muted)}
.chanslot{flex:0 0 20px;width:20px;min-height:20px;height:20px;padding:0;font-size:10px;
  font-family:ui-monospace,Consolas,monospace;border-radius:2px;line-height:1}
.chanslot.s0[aria-pressed=true]{background:#c22;border-color:#c22;color:#fff}
.chanslot.s1[aria-pressed=true]{background:#1a1;border-color:#1a1;color:#fff}
.chanslot.s2[aria-pressed=true]{background:#25c;border-color:#25c;color:#fff}
"""
CSS += BROWSER_CSS

VS_QUAD = """#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;
uniform mat4 uMVP;
uniform vec2 uHalf;
uniform float uZ;
// A tile is a sub-rectangle of the plane it belongs to: the same quad, restricted to
// a piece of the section in mm and to the matching piece of the texture. uUseRect 0
// draws the whole plane, which is what every layer except the selected one does.
uniform int  uUseRect;
uniform vec4 uRect;    // x0, y0, x1, y1 in mm
out vec2 vUV;
void main(){
  float fx = aPos.x*0.5+0.5, fy = aPos.y*0.5+0.5;
  // v runs opposite to y either way: a texture's first row is the top of the section
  vUV = vec2(fx, 1.0 - fy);
  gl_Position = (uUseRect == 1)
    ? uMVP * vec4(mix(uRect.x, uRect.z, fx), mix(uRect.y, uRect.w, fy), uZ, 1.0)
    : uMVP * vec4(aPos.x*uHalf.x, aPos.y*uHalf.y, uZ, 1.0);
}
"""

VS_RECUT = """#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;
uniform mat4 uMVP;
uniform vec4 uRect;    // x0, z0, x1, z1 in mm (panel space)
uniform vec2 uUV;      // fixed uv coordinate along the cut, and its axis (0 = row, 1 = col)
out vec2 vUV;
void main(){
  float fx = aPos.x*0.5+0.5;
  vec2 p = vec2(mix(uRect.x, uRect.z, fx), mix(uRect.y, uRect.w, aPos.y*0.5+0.5));
  vUV = (uUV.y < 0.5) ? vec2(fx, uUV.x) : vec2(uUV.x, fx);
  gl_Position = uMVP * vec4(p, 0.0, 1.0);
}
"""

FS_QUAD = """#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTex;
uniform float uOpacity;
uniform int   uIsRGB;   // 1 = this texture already carries the modality's own colour
uniform vec3  uTint;    // multiplies single-channel data; (1,1,1) is plain grey
out vec4 frag;
void main(){
  vec4 t = texture(uTex, vUV);
  if(t.a < 0.004) discard;
  // un-premultiply, then map. RGB planes carry three channels of real stain colour;
  // single-channel planes carry one, and the tint is what makes DAPI blue.
  vec3 col = (uIsRGB == 1) ? t.rgb / max(t.a, 1.0/255.0)
                           : vec3(t.r / max(t.a, 1.0/255.0));
  if(uIsRGB == 0) col *= uTint;
  float a = t.a * uOpacity;
  frag = vec4(col * a, a);
}
"""

VS_FULL = """#version 300 es
precision highp float;
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main(){ vUV = aPos*0.5+0.5; gl_Position = vec4(aPos, 0.0, 1.0); }
"""

FS_RESOLVE = """#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uAcc;
out vec4 frag;
void main(){
  vec4 s = texture(uAcc, vUV);
  if(s.a < 1.0/255.0) discard;
  frag = vec4(s.rgb, s.a);
}
"""

JS = r"""
"use strict";
const META = window.__META__, SRC = window.__SOURCE__;
const N = META.planes.length;
const Z = META.planes.map(p => p.z_um);
const MODS = META.planes.map(p => p.modality);
const ZMIN = Math.min(...Z), ZMAX = Math.max(...Z), ZSPAN = (ZMAX - ZMIN) || 1;
const DEPTH_MM = ZSPAN / 1000;
const BASES = Object.keys(META.bases);
// The rungs this viewer offers, whether or not a given sample has built them all.
// A sample that is missing one gets a disabled button, not a shorter control: the
// panel must not change shape when the reader switches sample.
const RES_ALL = ["8", "4"];
const RESES = SRC.mode === "fetch" ? Object.keys(SRC.res) : [String(META.in_plane_um_per_px)];
const RES_SHOWN = SRC.mode === "fetch"
  ? RES_ALL.filter(r => RES_ALL.indexOf(r) >= 0) : RESES;
const GAPS = META.gaps_ge_25um;
const GAP_AFTER = {}; for(const g of GAPS) GAP_AFTER[g.after_index] = g.dz_um;
const GAP_IDX = GAPS.map(g => g.after_index);
const RM = !!(window.matchMedia
  && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

// The opacities and the spread are NOT interface controls. They are
// named constants, tuned by editing this block, so the panel does not grow a knob for
// every number in the renderer.
const SEL = {
  OPACITY_SELECTED : 1.00,   // the selected plane
  OPACITY_OTHER    : 0.12,   // every other plane once something is selected: a fixed
                             // value, deliberately not a function of distance
  SCALE_SELECTED   : 1.00,   // the selected plane is NOT grown: every plane stays where
  GAP_UM           : 0,      // the 3-D bodies are, so the section and the bodies keep
  SPREAD_UM        : 0,      // lining up. Selection is shown by opacity alone.
  SPREAD_MAX_UM    : 0,
  ANIM_MS          : 320,    // expand / collapse duration
  STAGGER_MS       : 8,      // per-plane start offset, near planes first
  UI_MS            : 180,    // interface elements move faster than the content
  OPACITY_BASE     : 0.10    // with nothing selected: ALL 50 planes at this value
};
const BASE_OPACITY = SEL.OPACITY_BASE;
// The third base style. It is the same grey stack drawn at a fraction of its
// weight, so planes deeper in the block show through the ones in front of them, and
// with the cell overlay on it is a backdrop the class colours can be read against.
// Grey and native do NOT go through this multiply at all, which is what keeps those
// two displays bit-identical to what they were before the overlay existed.
const TRANSLUCENT_W = 0.35;
function baseAlpha(a){ return (S.colour === "translucent") ? a * TRANSLUCENT_W : a; }
// The overlay's resting weight, and the Opacity slider does NOT reach it. The slider
// is the BASE's control; letting it scale the overlay too meant one control doing two
// jobs, and dimming the sections to look at them also dimmed the thing being looked
// for. Same value the slider ships at, so the default picture is unchanged.
const CELL_OPACITY = 0.10;
// The Cell-size slider means "how visible one predicted cell is". Zoomed in that is
// the drawn radius; in the whole-block view a cell is below one pixel and the radius
// cannot express it, so there the same control raises the layer's weight instead.
// One control, one job, two views -- not two jobs.
function cellW(){ return Math.min(0.90, CELL_OPACITY * S.cellPt); }

// opening view: what key 6 (bottom) gives, el -1.55, lifted to -1.35 so the
// block is read from just below rather than dead flat underneath it.
const OPEN_AZ = 0, OPEN_EL = -1.35;
const S = {tlsReg: false, tls3d: true, tls2d: true, nerveReg: false, nlineReg: true, ductReg: false, glandReg: false, gland3d: true, gland2d: true, tumorReg: false, den3d: false, 
  showLabels: true,
  basis: BASES[0], res: SRC.mode === "fetch" ? SRC.default_res : RESES[0],
  exag: 1.0,
  // opening view: the one key 6 gives (bottom), lifted a little so the
  // block is read from just below rather than dead flat underneath
  az: OPEN_AZ, el: OPEN_EL, zoom: 1.0, cx: 0, cy: 0, pivot: [0,0,0], tool: "select",
  pickMsg: "",
  zi: Math.floor(N/2), solo: false, hidden: [],
  // BASE STYLE: how the sections themselves are drawn. One of three, and exactly one
  // at a time. Independent of the cell overlay below, which can sit on any of them.
  colour: "native",
  // whether the sections themselves are drawn. Off leaves the cell overlay
  // and the solid bodies in place, which is the point: it is how a reader
  // looks at what was called WITHOUT the tissue in front of it.
  baseOn: true,
  heSwap: !!(SRC.mode === "fetch" && SRC.res && Object.keys(SRC.res).some(r => SRC.res[r].swap)
             && ((META.he_swap && META.he_swap.indices)
                 ? META.he_swap.indices.map(i => META.planes[i]) : META.planes)
                .some(p => p && p.modality !== "he")),
                          // the page opens on the arm that shows H&E: the swap arm where the
                          // twice-imaged sections carry their CODEX/Xenium scan in the base arm
                          // (891), the base arm where those sections are H&E in the base and
                          // the swap arm holds the other scan (HT206B1)
  // The cell overlay is a switch, not a fourth base style. It is off by default,
  // so the page opens with no overlay.
  cellsOverlay: false,
  // Does the overlay follow the layer selection, or stay on every section?
  // ON by default. Selecting a layer drops every other plane to 0.12, which is an
  // eightfold contrast and reads as "only this one is highlighted" -- and how a class
  // runs THROUGH the block is the question a stack of fifty sections exists to answer.
  // Default false: the overlay follows the layer selection instead of sitting
  // on every plane at once.
  cellsAll: false,
  // which predicted cell classes the overlay draws. Eleven at once is coloured noise,
  // so the overlay opens with NO class on and the reader switches on the ones the
  // question is about, one at a time.
  cellsOn: [],
  // how big the overlay draws each cell. A rare class is a handful of dots
  // on a 50-plane block and reads as nothing at true scale, so the size is
  // the reader's control. 1.0 = the true nucleus radius.
  cellPt: 1.0,
  playing: false,
  // selection: null until the user picks a plane. zi is "which plane the controls act
  // on", sel is "which plane the user chose", and only sel drives the expansion.
  sel: null, transient: "",
  group: null
};
// FIGURE MODE (Output panel). A separate display option: every field below is read
// ONLY while FIG.on is true and nothing here writes into S, so switching it off puts
// the page back exactly as it was with no state to restore. It adds three things the
// ordinary display does not have: a second section shown beside the first, nerve and
// gland kept separately in 2-D and in 3-D, and a per-object list.
const FIG = {on: false, sections: true, nerveFill: false, secA: null, secB: null,
             nerve: true, n2: true, n3: true, gland: true, g2: true, g3: true,
             nerveA: 1, glandA: 1, hide: Object.create(null), multi: false,
             // set only while the figure sheet is being captured: the frame then holds
             // the nerve and gland bodies and nothing else, on a clear background
             sheet: false};
// the planes that count as "the section" - one normally, up to two in figure mode.
// In figure mode the set is the panel's own, so clearing the selection (a click on the
// background) leaves the picture where the reader put it.
function secList(){
  if(FIG.on){
    const a = FIG.secA === null ? S.sel : FIG.secA;
    if(a === null) return [];
    if(FIG.secB === null || FIG.secB === a) return [a];
    return [a, FIG.secB].sort((x, y) => x - y);
  }
  return S.sel === null ? [] : [S.sel];
}
function sectionsOn(){ return !FIG.sheet && (FIG.on ? FIG.sections : S.baseOn); }
function regionOn(source, ordinary){
  if(FIG.sheet) return false;
  if(FIG.on && (source===NERVE || source===GLANDR)) return figOn2d(source);
  return S.cellsOverlay && ordinary;
}
function figFixed(){ return FIG.on && secList().length > 0; }
function figOn2d(source){          // is this 2-D region layer drawn, figure mode or not
  if(FIG.sheet) return false;
  if(!FIG.on) return true;
  if(source === NERVE) return FIG.nerve && FIG.n2;
  if(source === GLANDR) return FIG.gland && FIG.g2;
  return true;
}
function figObjOn(id){ return !(FIG.on && FIG.hide[id]); }
// fixed display mapping: identity. There are no window / gamma / palette controls.
const WIN_LO = 0, WIN_HI = 255, GAMMA = 1.0, BG = "dark", FILTER = "smooth";
function resInfo(){
  return SRC.mode === "fetch" ? SRC.res[S.res]
    : {canvas_mm: META.canvas_mm, canvas_px: META.canvas_px,
       in_plane_um_per_px: META.in_plane_um_per_px, encoding: META.encoding};
}
const RAD0 = 6.0142;                       // spec 1.5: hypot(7.352,6.328)*0.62 mm

// Two ways to show the data and no third: plain grey, or each modality in its own
// colour. Nothing here invents a colour that the microscope did not produce.
//   H&E     -- real RGB, re-extracted for this page (a separate set of textures)
//   CODEX   -- DAPI, one channel, tinted
//   Xenium  -- DAPI morphology, one channel, tinted
// DAPI_TINT is DAPI's own emission colour, not "blue": the DAPI-DNA band (peak
// 461 nm, FWHM 59 nm) against the CIE 1931 2-degree observer gives chromaticity
// PLACEHOLDER_XY, which is outside sRGB; mixed with the least white that brings it
// inside and scaled so its largest channel is 1, it is #0081FF. Pure #0000FF would
// be a different colour AND would throw away the green channel, which carries most
// of the luminance a reader needs to see nuclei.
const DAPI_TINT = [0.0, 0.506, 1.0];
const WHITE = [1, 1, 1];
function tintFor(i){
  if(S.colour !== "native" || i < 0) return WHITE;
  // the tint follows what this arm PUTS on screen for this plane, not what the
  // section is: an H&E picture gets its own RGB and no tint, a DAPI picture gets
  // DAPI's emission colour. Reading the section's identity instead left the
  // CODEX/Xenium scans plain grey in the arm that shows them.
  return (modsNow()[i] === "he") ? WHITE : DAPI_TINT;
}
// Two regimes, deliberately not mixed:
//   nothing selected -> EVERY plane at the same opacity. This page exists to show 50
//     sections stacked into a volume, so the default has to draw 50 of them. An
//     earlier version faded them by distance from the current plane, which at a
//     575 um span left 4 planes above one 8-bit step and the rest invisible: the
//     viewer opened looking like a single slice.
//   a plane selected  -> the others take one FIXED opacity and move out of the way
//     along z, because the question has changed from "how deep is this" to "look at
//     this one".
function easeOut(t){ return 1 - Math.pow(1 - t, 3); }
// Three transitions, one mechanism. A transition interpolates from
// a SNAPSHOT of the state as it looked when the transition began - not from "collapsed"
// - so a new one started mid-flight continues from where the picture actually is
// instead of jumping or queueing. That is what makes holding the arrow key smooth.
const ANIM = {from: null, to: null, t0: 0, dur: SEL.ANIM_MS};
// A layout is the resting state for a given (selection, only-this-layer) pair. Showing
// one layer on its own is a third opacity regime rather than a filter on the draw list,
// so leaving it plays as a fade back in - the same mechanism as every other transition.
// alc is the CELL OVERLAY's alpha, carried beside the base's so the two never share a
// number. The only term the Opacity slider touches is the resting level, so that is
// the only one that differs: the overlay rests at its own constant. The selection
// levels (1.00 for the chosen plane, 0.12 for the rest, 0 for hidden) are semantics,
// not brightness, and are shared. Scaling the overlay by a ratio instead would have
// been wrong at both ends -- at a selection it would have divided 1.00 by the slider.
function layoutFor(sel, only){
  const off = new Float64Array(N), al = new Float64Array(N), alc = new Float64Array(N);
  const solo = !!only && sel !== null;
  // figure mode with a second section: the pair sits flat at its real z, everything
  // else is off. Same regime as "one layer on its own", widened to the chosen pair.
  const figSet = figFixed() ? secList() : null;
  for(let i=0;i<N;i++){
    if(figSet){ off[i] = 0;
      al[i] = figSet.indexOf(i) >= 0 ? SEL.OPACITY_SELECTED : 0;
      alc[i] = al[i]; continue; }
    if(sel === null){ off[i] = 0; al[i] = SEL.OPACITY_BASE; alc[i] = cellW();
                      continue; }
    // "stack up to here": every plane from the first one up to the chosen one is
    // drawn at its real z with the resting opacity, and everything above it is off.
    // Stepping the selection therefore builds the block up one sheet at a time.
    // No spread and no growth here - the point is the pile, not one card lifted out.
    if(solo){ off[i] = 0;
      al[i] = (i === sel) ? SEL.OPACITY_SELECTED
            : (i <  sel) ? SEL.OPACITY_OTHER : 0;
      // "stack up to here" still means up to here: a plane that is not in the pile
      // has no cells either. Within the pile the scope decides.
      alc[i] = S.cellsAll ? (i > sel ? 0 : cellW()) : al[i];
      continue; }
    const r = i - sel;
    off[i] = r === 0 ? 0
      : Math.sign(r) * (SEL.GAP_UM
          + Math.min(Math.abs(r) * SEL.SPREAD_UM, SEL.SPREAD_MAX_UM));
    al[i] = (r === 0) ? SEL.OPACITY_SELECTED : (solo ? 0 : SEL.OPACITY_OTHER);
    alc[i] = S.cellsAll ? cellW() : al[i];
  }
  return {off: off, al: al, alc: alc,
          scale: (sel === null || solo || figSet) ? 1 : SEL.SCALE_SELECTED,
          sel: sel, only: solo || !!figSet};
}
function animU(){
  if(!ANIM.to) return 1;
  if(RM) return 1;                              // reduced motion: straight to the end
  return Math.max(0, Math.min(1, (nowMs() - ANIM.t0) / ANIM.dur));
}
function animating(){ return !!ANIM.to && animU() < 1; }
function curLayout(){
  const to = ANIM.to || layoutFor(S.sel, S.solo);
  const from = ANIM.from || to;
  // Once the transition is over, hand back the target itself rather than an
  // interpolation that lands on it: 0.12 + (0.10-0.12)*1 is 0.09999999999999998, and a
  // resting state that is a hair off its own constant is a trap for anything that
  // compares against it.
  if(animU() >= 1) return {u: 1, from: to, to: to};
  return {u: easeOut(animU()), from: from, to: to};
}
function spreadUm(i){
  if(FIG.on || window.__SOLID_ACTIVE__)return 0;
  const L = curLayout();
  return L.from.off[i] + (L.to.off[i] - L.from.off[i]) * L.u;
}
function maxSpreadUm(){
  let m = 0; for(let i=0;i<N;i++) m = Math.max(m, Math.abs(spreadUm(i)));
  return m;
}
function selScale(){
  if(FIG.on || window.__SOLID_ACTIVE__)return 1;
  const L = curLayout();
  return L.from.scale + (L.to.scale - L.from.scale) * L.u;
}
function alphaFor(i){
  const L = curLayout();
  return L.from.al[i] + (L.to.al[i] - L.from.al[i]) * L.u;
}
// the overlay's own alpha, on its own track through the same transition
function cellAlphaFor(i){
  const L = curLayout();
  return L.from.alc[i] + (L.to.alc[i] - L.from.alc[i]) * L.u;
}
function beginTransition(sel, only, dur){
  const L = curLayout();
  // snapshot where every plane is RIGHT NOW, mid-flight included
  const snap = {off: new Float64Array(N), al: new Float64Array(N),
                alc: new Float64Array(N),
                scale: L.from.scale + (L.to.scale - L.from.scale) * L.u, sel: S.sel};
  for(let i=0;i<N;i++){
    snap.off[i] = L.from.off[i] + (L.to.off[i] - L.from.off[i]) * L.u;
    snap.al[i] = L.from.al[i] + (L.to.al[i] - L.from.al[i]) * L.u;
    snap.alc[i] = L.from.alc[i] + (L.to.alc[i] - L.from.alc[i]) * L.u;
  }
  ANIM.from = snap; ANIM.to = layoutFor(sel, only); ANIM.t0 = nowMs(); ANIM.dur = dur;
}
function setSelected(i){
  if(i !== null && (!Number.isInteger(i) || i < 0 || i >= N)) return;
  // Selecting and clearing are transitions and are animated. Moving the selection
  // from one plane to another is NOT: flipping between neighbouring sections is how
  // this stack gets compared, and any tween puts a smear between the two pictures
  // being compared. It has to land instantly so consecutive slices can be flicked
  // back and forth and read as one continuous motion.
  const dur = (S.sel !== null && i !== null) ? 0 : SEL.ANIM_MS;
  // clearing the selection also leaves the only-this-layer state: it is a statement
  // about a particular layer, so it cannot outlive the selection it refers to
  const only = (i === null) ? false : S.solo;
  beginTransition(i, only, dur);
  S.sel = i; S.solo = only;
  if(i !== null) setZ(i); else setZ(S.zi);   // repaint the strip either way
  syncOnly();
  need();
}
// "Show only this layer". One switch, no slider and no menu, and it only
// exists while a layer is selected.
function setOnly(v){
  if(S.sel === null){ if(!v) return; setSelected(S.zi); }
  const on = !!v;
  if(on === S.solo){ syncOnly(); return; }
  beginTransition(S.sel, on, SEL.ANIM_MS);
  S.solo = on;
  syncOnly(); need();
}
const BGCOL = {dark:[0.039,0.051,0.063], light:[0.949,0.961,0.965], mid:[0.431,0.478,0.502]};
// set only while a saved image is being rendered; null the rest of the time
let SHOT_BG = null;
// what the PNG / JPG / PDF buttons produce. scale multiplies the on-screen pixel
// grid, alpha asks for a cut-out background (PNG only -- JPG and PDF cannot carry one)
const SHOT = {scale: 2, alpha: false};
const SHOT_MAX_PX = 8192;
let EXPORT_JOB = null;

// ------------------------------------------------------------------ GL setup
const cv = document.getElementById("gl");
// stencil: the solid bodies cap their cut face by counting entries and exits into
// the stencil buffer. Without one the cut shows the far wall, which reads as a
// hollow bowl rather than a cut solid.
const gl = cv.getContext ? cv.getContext("webgl2",
  {alpha:false, antialias:false, premultipliedAlpha:true, stencil:true}) : null;
const errEl = document.getElementById("errs"), progEl = document.getElementById("progress");
const LOADQ = (function(){
  const items = new Map();          // key -> {label, got, tot}
  const ORDER = ["planes", "colour", "cells", "tiles", "payload", "bodies", "cloud"];
  function render(){
    const parts = [];
    ORDER.concat([...items.keys()].filter(k => ORDER.indexOf(k) < 0)).forEach(k => {
      const it = items.get(k);
      if(!it) return;
      if(it.tot != null && it.got >= it.tot && !it.note) return;
      parts.push(it.note ? it.note
                 : it.label + " " + it.got + (it.tot != null ? " / " + it.tot : ""));
    });
    progEl.textContent = parts.join("  \u00b7  ");
  }
  return {
    set(key, label, got, tot){ items.set(key, {label, got, tot}); render(); },
    note(key, note){ if(note) items.set(key, {note}); else items.delete(key); render(); },
    render
  };
})();
window.__LOADQ__ = LOADQ;
if(!gl){ errEl.textContent = "This browser did not give a WebGL2 context, so nothing "
  + "can be drawn. Everything else on the page still works."; }
function mkProg(vs, fs){
  const p = gl.createProgram();
  for(const [t, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]){
    const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
    if(!gl.getShaderParameter(s, gl.COMPILE_STATUS))
      throw new Error("shader compile (" + (t === gl.VERTEX_SHADER ? "VS" : "FS")
        + "): " + (gl.getShaderInfoLog(s) || "no log; context lost="
        + gl.isContextLost()) + " src head: " + src.slice(0, 120));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if(!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error("program link: " + (gl.getProgramInfoLog(p) || "no log; context lost="
      + gl.isContextLost()));
  const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for(let i=0;i<n;i++){ const nm = gl.getActiveUniform(p, i).name.replace(/\[0\]$/, "");
    u[nm] = gl.getUniformLocation(p, nm); }
  return {p: p, u: u};
}
// One program and one vertex array: fifty textured quads and nothing else. The flat
// colour program existed only to stroke a rectangle around the selected plane.
// A second program, for one point per cell. It is not a variant of the quad
// program -- the quad draws a texture, this draws geometry -- and it exists only
// because a magnified density texture is the wrong picture of a cell.
const VS_CELLPT = `#version 300 es
precision highp float;
layout(location=0) in vec2 aXY;      // canvas pixels
layout(location=1) in float aR;      // radius, canvas pixels
layout(location=2) in float aCls;
layout(location=3) in float aK2;    // per-section denoise keep (1 = survives)
layout(location=4) in float aK3;    // 3-D denoise keep
uniform mat4 uMVP;
uniform vec2 uHalf;                  // canvas half extent, mm
uniform vec2 uCanvasPx;
uniform float uZ;
uniform float uPxPerMM;
uniform float uMinPx; uniform float uMaxPx;
uniform float uD2[16]; uniform float uD3[16];
uniform float uPtScale;
uniform vec3 uPalette[16];
uniform float uOn[16];               // 1 = this class is switched on
out vec3 vCol;
out float vKeep;
void main(){
  int c = int(aCls + 0.5);
  vCol = uPalette[c];
  vKeep = uOn[c];
  if(uD2[c] > 0.5 && aK2 < 0.5) vKeep = 0.0;
  if(uD3[c] > 0.5 && aK3 < 0.5) vKeep = 0.0;
  vec2 mm = (aXY / uCanvasPx * 2.0 - 1.0) * vec2(uHalf.x, -uHalf.y);
  gl_Position = uMVP * vec4(mm, uZ, 1.0);
  float mmPerPx = (2.0 * uHalf.x) / uCanvasPx.x;
  // physical size, but clamped on SCREEN: zooming in must not turn cells into
  // blobs that bury the tissue. The slider still scales the capped size.
  gl_PointSize = (vKeep > 0.5)
    ? uPtScale * clamp(2.0 * aR * mmPerPx * uPxPerMM, uMinPx, uMaxPx) : 0.0;
}
`;
const FS_CELLPT = `#version 300 es
precision highp float;
in vec3 vCol;
in float vKeep;
uniform float uOpacity;
out vec4 frag;
void main(){
  if(vKeep < 0.5) discard;
  vec2 d = gl_PointCoord - 0.5;
  float r = length(d);
  if(r > 0.5) discard;
  // one pixel of feather, so a cell is a disc and not a staircase
  float a = uOpacity * (1.0 - smoothstep(0.42, 0.5, r));
  if(a <= 0.0) discard;
  frag = vec4(vCol * a, a);
}
`;
let PQ, PC, quadVAO, SMOOTH, NEAREST;
if(gl){
  PQ = mkProg(window.__VS_QUAD__, window.__FS_QUAD__);
  PC = mkProg(VS_CELLPT, FS_CELLPT);
  quadVAO = gl.createVertexArray(); gl.bindVertexArray(quadVAO);
  const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
  gl.bindVertexArray(null);
  // one sampler object changes filtering for all 50 textures at once (spec 3.1.6)
  SMOOTH = gl.createSampler(); NEAREST = gl.createSampler();
  for(const [s, f] of [[SMOOTH, gl.LINEAR], [NEAREST, gl.NEAREST]]){
    gl.samplerParameteri(s, gl.TEXTURE_MIN_FILTER, f);
    gl.samplerParameteri(s, gl.TEXTURE_MAG_FILTER, f);
    gl.samplerParameteri(s, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.samplerParameteri(s, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }
  document.getElementById("limits").textContent =
    "device limits read at runtime  MAX_TEXTURE_SIZE=" + gl.getParameter(gl.MAX_TEXTURE_SIZE)
    + "  MAX_3D_TEXTURE_SIZE=" + gl.getParameter(gl.MAX_3D_TEXTURE_SIZE)
    + "  MAX_TEXTURE_IMAGE_UNITS=" + gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)
    + "  MAX_RENDERBUFFER_SIZE=" + gl.getParameter(gl.MAX_RENDERBUFFER_SIZE)
    + "\n2D textures only, one bound per draw call; no 3D texture and no texture array, "
    + "so neither of those two limits constrains this page.";
}

// ------------------------------------------------------------------ textures
const TEX = {}, LOADED = {}, FAILED = {}, LRU = [];
// A coarse copy of each plane's ALPHA channel, kept on the CPU so a click can be
// answered without reading pixels back off the GPU. Built from the same Image the
// texture is uploaded from, so it cannot describe a different picture.
const AMASK = {}, AMW = 256, PICK_ALPHA = 128;
let amCv = null, amCtx = null;
function amCtx2d(){
  if(amCtx !== null) return amCtx;
  amCtx = false;
  try {
    amCv = document.createElement("canvas");
    if(amCv && amCv.getContext) amCtx = amCv.getContext("2d", {willReadFrequently: true})
                                       || false;
  } catch(_){ amCtx = false; }
  return amCtx;
}
function makeMask(im){
  const c = amCtx2d(); if(!c) return null;
  const w = im.width || 0, h = im.height || 0; if(!w || !h) return null;
  const sc = Math.min(1, AMW / Math.max(w, h));
  const mw = Math.max(1, Math.round(w*sc)), mh = Math.max(1, Math.round(h*sc));
  try {
    amCv.width = mw; amCv.height = mh;
    c.clearRect(0, 0, mw, mh);
    c.drawImage(im, 0, 0, mw, mh);
    const d = c.getImageData(0, 0, mw, mh).data;
    const a = new Uint8Array(mw*mh);
    for(let i=0;i<a.length;i++) a[i] = d[i*4+3];
    return {w: mw, h: mh, a: a};
  } catch(_){ return null; }
}
function alphaAt(k, i, u, v){                 // -1 = no mask for this plane
  const set = AMASK[k], m = set && set[i];
  if(!m) return -1;
  if(u < 0 || u > 1 || v < 0 || v > 1) return 0;
  const x = Math.min(m.w-1, Math.max(0, Math.floor(u*m.w)));
  const y = Math.min(m.h-1, Math.max(0, Math.floor(v*m.h)));
  return m.a[y*m.w + x];
}
function urlsFor(b, r){
  if(SRC.mode === "embedded")
    return JSON.parse(document.getElementById("imgs-" + b).textContent);
  const R = armOf(r);
  return R.files[b].map(f => R.dir + "/" + f + "?v=" + (R.v || 0));
}
function reportProgress(k, r){
  // The counter is a "not in the default state" line. Once every
  // plane is in, it disappears rather than sitting there saying nothing is wrong.
  const n = LOADED[k] || 0, f = (FAILED[k] || []).length;
  const key = (String(r) === String(S.res)) ? "planes" : "planes_" + r;
  const label = (String(r) === String(S.res)) ? "sections " + r + " \u00b5m" : r + " \u00b5m ladder (prefetch)";
  if(n >= N && f) LOADQ.note(key, f + " of " + N + " planes failed to load at " + r + " \u00b5m");
  else LOADQ.set(key, label, n, N);
}
function evict(keep){
  while(LRU.length > 2){
    const k = LRU.shift();
    if(k === keep) { LRU.push(k); continue; }
    const arr = TEX[k]; if(!arr) continue;
    for(const t of arr) if(t) gl.deleteTexture(t);
    delete TEX[k]; delete LOADED[k]; delete FAILED[k]; delete AMASK[k];
    const rs = RGBTEX[k];                       // the colour set goes with its grey set
    if(rs){ for(const i in rs) if(rs[i]) gl.deleteTexture(rs[i]);
      delete RGBTEX[k]; delete RGBLOADED[k]; delete RGBFAILED[k]; }
  }
}
// Whole-section previews stay at 1024 px; the visible ROI uses full-detail tiles.
const STACK_TEX_CAP = 1024;
const PLANE_QUEUE = []; let PLANE_ACTIVE = 0, PLANE_PEAK = 0;
function queuePlane(i, run){
  if(window.__VIEWER_DISPOSING__) return;
  PLANE_QUEUE.push({i,run}); pumpPlaneQueue();
}
function pumpPlaneQueue(){
  if(window.__VIEWER_DISPOSING__) return;
  while(PLANE_ACTIVE < 4 && PLANE_QUEUE.length){
    PLANE_QUEUE.sort((a,b)=>Math.abs(a.i-S.zi)-Math.abs(b.i-S.zi));
    const task=PLANE_QUEUE.shift(); PLANE_ACTIVE++; PLANE_PEAK=Math.max(PLANE_PEAK,PLANE_ACTIVE);
    let released=false;
    task.run(()=>{if(released)return;released=true;PLANE_ACTIVE--;pumpPlaneQueue();});
  }
}
window.__STACK_LOAD_QA__=()=>({active:PLANE_ACTIVE,queued:PLANE_QUEUE.length,peak:PLANE_PEAK,textureCap:STACK_TEX_CAP});
// VRAM guard: a stack texture never uploads wider than the preview cap. The far view is
// screen-bounded and the zoomed view reads the tile pyramid, so nothing needs a
// 3000-px plane resident on the GPU -- seventy of them per basis is gigabytes
// and killed the context on the merged sample.
function texUpload(im, cap){
  cap = cap || STACK_TEX_CAP;
  const ds = Math.max(1, Math.ceil(Math.max(im.width, im.height) / cap));
  if(ds === 1){
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, im);
    return;
  }
  const c = document.createElement("canvas");
  c.width = Math.ceil(im.width / ds); c.height = Math.ceil(im.height / ds);
  c.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
}

// EVERY section image request (grey ladder and native H&E alike) waits for the
// 3-D: until the solid payload's small bodies are in -- or the sample has no
// 3-D, or 30 s have passed -- the request is queued and released afterwards
const PAGE_T0 = performance.now();
const SOLID_WAIT = [];
function solidReady(){ return !!(window.__SOLID_SMALL_DONE__ || window.__SOLID_NONE__) || performance.now() - PAGE_T0 > 30000; }
function waitSolid(fn){
  // true = go ahead now (the caller continues itself); false = queued, the
  // caller returns and is re-invoked once the 3-D is in
  if(solidReady()) return true;
  SOLID_WAIT.push(fn);
  if(!waitSolid._t) waitSolid._t = setInterval(() => {
    if(!solidReady()) return;
    clearInterval(waitSolid._t); waitSolid._t = null;
    const q = SOLID_WAIT.splice(0); q.forEach(f => f());
  }, 250);
  return false;
}
function loadSet(b, r, done){
  // the coarsest grey set is the frame every draw needs (the 3-D overlay
  // renders on top of it): it loads at once; anything finer waits for the 3-D
  const coarsest = String(Math.max(...RESES.map(parseFloat)));
  if(String(r) !== coarsest && !waitSolid(() => loadSet(b, r, done))) return;
  const k = setKey(b, r);
  const li = LRU.indexOf(k); if(li >= 0) LRU.splice(li, 1);
  LRU.push(k);
  if(TEX[k]){ reportProgress(k, r); markErrors(k); evict(k); done && done(); return; }
  if(!gl) return;
  evict(k);
  const uris = urlsFor(b, r);
  TEX[k] = new Array(N).fill(null); LOADED[k] = 0; FAILED[k] = [];
  AMASK[k] = new Array(N).fill(null);
  reportProgress(k, r);
  // fetch order = distance from the plane being LOOKED AT, not file order: the
  // section on screen sharpens first instead of waiting out the whole stack
  const ordIdx = uris.map((u, i2) => i2)
    .sort((a2, b2) => Math.abs(a2 - S.zi) - Math.abs(b2 - S.zi));
  ordIdx.forEach(i => queuePlane(i, release => { const uri = uris[i];
    if(!TEX[k]){release();return;}
    let im = new Image();
    im.fetchPriority = (String(r) === String(S.res)) ? "high" : "low";
    im.onload = () => {
      if(window.__VIEWER_DISPOSING__){im.onload=null;im=null;release();return;}
      // decode AND downscale off the main thread: a 37-megapixel webp decoded
      // synchronously freezes the page for hundreds of milliseconds, and a
      // whole set of them arriving together froze it for half a minute
      const ds = Math.max(1, Math.ceil(Math.max(im.width, im.height) / STACK_TEX_CAP));
      createImageBitmap(im, {resizeWidth: Math.ceil(im.width / ds),
                             resizeHeight: Math.ceil(im.height / ds),
                             resizeQuality: "high"}).then(bmp => {
      if(window.__VIEWER_DISPOSING__){bmp.close();return;}
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      if(TEX[k]) TEX[k][i] = t; else gl.deleteTexture(t);
      if(AMASK[k]) AMASK[k][i] = makeMask(bmp);     // alpha copy for picking
      bmp.close();
      LOADED[k]++; reportProgress(k, r);
      im.onload = null; im = null;
      if(LOADED[k] === N){ markErrors(k); done && done(); prefetchFiner(); }
      need();
      }).catch(()=>{LOADED[k]++;(FAILED[k]||[]).push({i,url:uri});reportProgress(k,r);markErrors(k);if(LOADED[k]===N&&done)done();}).finally(release);
    };
    im.onerror = () => {
      LOADED[k]++; (FAILED[k] || []).push({i: i, url: uri});
      reportProgress(k, r); markErrors(k);
      if(LOADED[k] === N && done) done();
      release();
    };
    im.src = uri;
  }));
}
// the finer resolution downloads ITSELF once the page is idle: the reader who
// zooms two minutes in finds it already there instead of watching a bar
let PREFETCHED = false;
function prefetchFiner(){
  if(PREFETCHED || SRC.mode !== "fetch") return;
  if(!sectionsOn()) return;                 // sections hidden: the reader is looking at the 3-D, no ladder
  const finer = RESES.find(r2 => r2 !== S.res);
  if(!finer) return;
  // only when it is cheap: a finer ladder is fetched when the reader asks for it
  // (resolution control), not in the background, once it is bigger than this --
  // the merged S22 ladder is 300 MB per basis and starved everything else
  const PREFETCH_MAX_BYTES = 64e6;
  const bpb = (SRC.res[finer] && SRC.res[finer].bytes_per_basis || {})[S.basis];
  if(bpb && bpb > PREFETCH_MAX_BYTES){ PREFETCHED = true; return; }
  PREFETCHED = true;
  // the 3-D bodies come first: the finer ladder starts only once the solid
  // payload and its small bodies (TLS, nerves) are in, or the sample has none,
  // or 30 s have passed -- the reader asked to see the 3-D before the sharpening
  const t0 = performance.now();
  const go = () => { loadSet(S.basis, finer, null);
                     if(S.colour === "native") loadRGBSet(S.basis, finer, null); };
  const tick = () => {
    if(window.__SOLID_SMALL_DONE__ || window.__SOLID_NONE__ || performance.now() - t0 > 30000) go();
    else setTimeout(tick, 500);
  };
  setTimeout(tick, 1000);
}
function markErrors(k){
  const f = FAILED[k] || [];
  for(const c of document.querySelectorAll(".cell")) c.classList.remove("err");
  if(!f.length){ errEl.textContent = ""; return; }
  const lines = [];
  for(const e of f){
    const cell = document.getElementById("cell-" + e.i);
    if(cell) cell.classList.add("err");
    lines.push("plane " + e.i + " (" + META.planes[e.i].section_id.split("-").pop()
               + ", z=" + Z[e.i] + " um) did not load.  requested: " + e.url);
  }
  errEl.textContent = lines.join("\n");
  errEl.style.display = lines.length ? "block" : "none";
}

// ------------------------------------------------- H&E native colour
// The volume the page normally draws is single channel: for H&E that channel is
// haematoxylin optical density after CLAHE, which has no colour at all. H&E colour
// is therefore a SECOND set of textures -- 25 RGB planes built on the same canvas,
// with alpha verified pixel-identical to their grey twins -- fetched the first time
// native colour is asked for, and only at the resolutions where they exist.
const RGBTEX = {}, RGBLOADED = {}, RGBFAILED = {};
function rgbUrlsFor(b, r){
  if(SRC.mode !== "fetch") return null;
  const R = armOf(r);
  if(!R || !R.he_rgb || !R.he_rgb[b]) return null;
  const m = {};
  for(const k in R.he_rgb[b]) m[+k] = R.dir + "/" + R.he_rgb[b][k] + "?v=" + (R.v || 0);
  return m;
}
function loadRGBSet(b, r, done){
  // the colour set of the resolution being LOOKED AT is not gated, exactly like its
  // grey set: sections are drawn as they arrive, so gating colour behind the 3-D
  // bodies left the stack sitting there in grey. A finer set still waits.
  const coarsest = String(Math.max(...RESES.map(parseFloat)));
  if(String(r) !== coarsest && !waitSolid(() => loadRGBSet(b, r, done))) return;
  const k = setKey(b, r);
  if(RGBTEX[k]){ done && done(); return; }
  const urls = rgbUrlsFor(b, r);
  if(!urls || !gl){ done && done(); return; }
  const idx = Object.keys(urls).map(Number);
  RGBTEX[k] = {}; RGBLOADED[k] = 0; RGBFAILED[k] = 0;
  const finish = () => { if(RGBLOADED[k] + RGBFAILED[k] === idx.length){
    reportRGB(); done && done(); } };
  idx.sort((a2, b2) => Math.abs(a2 - S.zi) - Math.abs(b2 - S.zi));
  for(const i of idx) queuePlane(i, release => {
    if(!RGBTEX[k]){release();return;}
    let im = new Image();
    im.fetchPriority = (String(r) === String(S.res)) ? "high" : "low";
    im.onload = () => {
      if(window.__VIEWER_DISPOSING__){im.onload=null;im=null;release();return;}
      const ds = Math.max(1, Math.ceil(Math.max(im.width, im.height) / STACK_TEX_CAP));
      createImageBitmap(im, {resizeWidth: Math.ceil(im.width / ds),
                             resizeHeight: Math.ceil(im.height / ds),
                             resizeQuality: "high"}).then(bmp => {
      if(window.__VIEWER_DISPOSING__){bmp.close();return;}
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      bmp.close();
      if(RGBTEX[k]) RGBTEX[k][i] = t; else gl.deleteTexture(t);
      RGBLOADED[k]++; im.onload = null; im = null; reportRGB(); need(); finish();
      }).catch(()=>{RGBFAILED[k]++;reportRGB();need();finish();}).finally(release);
    };
    im.onerror = () => { RGBFAILED[k]++; reportRGB(); need(); finish(); release(); };
    im.src = urls[i];
  });
}
function heRGBTotal(){ const u = rgbUrlsFor(S.basis, S.res);
  return u ? Object.keys(u).length : 0; }
function heRGBLoaded(){ return RGBLOADED[setKey(S.basis, S.res)] || 0; }

// ------------------------------------------- H&E on the multi-modal planes
// Sixteen of the fifty sections were scanned twice: CODEX first, then re-stained and
// re-scanned in H&E, at zero z separation. This switch shows those sixteen as their
// H&E instead of their CODEX, which makes the stack 41 H&E planes and 9 Xenium ones.
// It is OFF by default and it replaces nothing: the CODEX planes stay exactly as they
// are, and the swapped set is a third group of textures fetched only when asked for.
// Two files per plane, because a swapped plane has to behave like a real H&E plane in
// both display modes - grey for the grey ramp, rgb for native colour.
// IT IS A WHOLE SECOND RECONSTRUCTION, NOT SIXTEEN REPLACEMENT TEXTURES.
// Re-solving those sections' registration edges against their H&E neighbours moved
// 25 of the 49 edges of the tree the poses propagate along, so most of the fifty
// planes sit somewhere new. Everything the page draws therefore comes from a
// different directory in this arm: planes, colour planes and tiles. That is why the
// switch is an ARM selector and not a per-plane texture override -- an override
// would have put sixteen re-registered planes on top of thirty-four that had not
// moved with them.
const SWAPSET = (META.he_swap && META.he_swap.indices) ? new Set(META.he_swap.indices)
                                                       : new Set();
// modality per plane is a property of the arm: in the swapped one those sixteen are
// H&E, which is what decides tint, which texture ladder is used, and the readouts
// what each swapped plane IS in the other arm: read from the data (he_swap.modalities,
// index -> modality), "he" where the data does not say. A plane whose swapped
// modality has no colour goes down the grey path with that modality's tint.
const SWAPMODS = (META.he_swap && META.he_swap.modalities) || {};
const MODS_SWAP = MODS.map((m, i) => SWAPSET.has(i) ? (SWAPMODS[i] || SWAPMODS[String(i)] || "he") : m);
// What a section IS, whichever arm is on screen: a section imaged twice is named by
// its non-H&E scan, because that is the thing that distinguishes it. Read from both
// arms rather than from the base alone, so it stays right whichever arm the
// reconstruction was built on.
const MOD_TRUE = MODS.map((m, i) => m !== "he" ? m
                          : (SWAPSET.has(i) ? (MODS_SWAP[i] || "he") : "he"));
// Which arm carries H&E for the twice-imaged sections. Judged on those sections alone:
// the base arm when they are H&E in it (HT206B1: its other CODEX-only planes do not
// count), the swapped arm when the base carries their CODEX/Xenium scans (891).
const BASE_IS_HE = (SWAPSET.size ? [...SWAPSET].map(i => MODS[i]) : MODS).every(m => m === "he");
const ARM_LABEL = {he: "H&E", codex: "CODEX", xenium: "Xenium", visium: "Visium"};
function armKey(){ return S.heSwap ? "swap" : "base"; }
function armOf(r){
  const R = SRC.res ? SRC.res[r] : null;
  if(!R) return null;
  return (S.heSwap && R.swap) ? R.swap : R;
}
function swapAvailable(){
  if(SRC.mode !== "fetch" || !SWAPSET.size) return false;
  for(const r in SRC.res) if(SRC.res[r].swap) return true;
  return false;
}
function swapHere(){ return !!(SRC.mode === "fetch" && S.heSwap
                               && SRC.res[S.res] && SRC.res[S.res].swap); }
function modsNow(){ return (S.heSwap && swapHere()) ? MODS_SWAP : MODS; }
// more than one modality in the stack: only then does a chip say which it is
// (the single-modality samples would repeat "he" seventy times)
const MODMIX = new Set(MOD_TRUE).size > 1;
function swapped(i){ return swapHere() && SWAPSET.has(i); }
// One cache key per (arm, basis, resolution). Without the arm in it, flipping the
// switch would hand back the other arm's textures under the same name and the page
// would show the old poses while claiming the new ones.
function setKey(b, r){ return armKey() + "|" + b + "@" + r; }
// What the reader is looking at right now. In the swapped arm those sixteen planes
// ARE H&E sections, so the readouts must stop announcing CODEX at them.
function modLabel(i){
  // What is on screen, and -- when this arm shows the H&E of a section that was
  // also imaged some other way -- what that section was imaged as. A section that
  // only ever had an H&E says "he", never "he (was he)".
  const now = modsNow()[i];
  return (now === "he" && MOD_TRUE[i] !== "he") ? ("he (was " + MOD_TRUE[i] + ")") : now;
}
function reportRGB(){
  if(S.colour !== "native") return;
  const tot = heRGBTotal(), got = heRGBLoaded();
  if(!tot) LOADQ.note("colour", "no H&E colour at " + S.res + " \u00b5m/px");
  else LOADQ.set("colour", "H&E colour", got, tot);
}

// ------------------------------------------------ cell predictions
// A THIRD display next to grey and native colour, and the only one whose pixels are
// not a measurement: every one of them is where the pan-cancer Cell classifier put
// cells of ONE class on the H&E. This cohort has no spatial ground truth, so this
// layer says MODEL PREDICTION wherever it appears and never says "detected".
//
// WHY A DENSITY RASTER AND NOT A DOT PER CELL
//   A dot per cell is one more plane. With fifty planes at 10% opacity a single
//   pixel contributes about half a percent of the final colour and is invisible --
//   the same way an outline drawn around the selected layer was invisible. What
//   survives fifty-fold blending is AREA carrying a smooth alpha, so each class is
//   baked as its own density map on the canvas the planes already sit on and drawn
//   as one more textured quad.
// WHY ONE TEXTURE PER CLASS AND NOT ONE PICTURE
//   Colour has to mean class and nothing else. A single picture of all eleven would
//   need a colour for "tumour and fibroblast in the same pixel", which is a colour
//   that means a mixture. Separate textures let a class be switched OFF instead.
// WHY IT IS THE SAME SHADER
//   The texture is white with the density in its alpha, so the tint path that makes
//   DAPI blue makes a class exactly its own colour, everywhere, at every density.
const CELLS = SRC.cells || null;
const CELLINFO = {};
if(CELLS) for(const c of CELLS.classes) CELLINFO[c.name] = c;
// a reader's colour choices survive reload; one name = one colour on every layer
function savedColour(nm){ try{ return localStorage.getItem("cellColour." + nm); }
                          catch(e){ return null; } }
function storeColour(nm, hex){ try{ localStorage.setItem("cellColour." + nm, hex); }
                               catch(e){} }
for(const nm in CELLINFO){ CELLINFO[nm].colour0 = CELLINFO[nm].colour;
  const v = savedColour(nm); if(v) CELLINFO[nm].colour = v; }
// ONE colour per class name on every layer and every panel: the section list, the
// 3-D bodies and the point cloud all read CELLINFO, and every picker showing that
// class is refreshed, whichever picker was used
function setCellColour(c, h){
  if(CELLINFO[c]) CELLINFO[c].colour = h;
  storeColour(c, h);
  if(window.__SOLID_SETCOL__) window.__SOLID_SETCOL__(c, h);
  document.querySelectorAll('input.cpick[data-cls="' + c + '"]').forEach(ip => { if(ip.value !== h) ip.value = h; });
  need();
}
function resetColours(){
  try{
    const dead = [];
    for(let i = 0; i < localStorage.length; i++){
      const k = localStorage.key(i);
      if(k && k.indexOf("cellColour.") === 0) dead.push(k);
    }
    dead.forEach(k => localStorage.removeItem(k));
  }catch(e){}
  for(const nm in CELLINFO){
    if(CELLINFO[nm].colour0) CELLINFO[nm].colour = CELLINFO[nm].colour0;
    if(window.__SOLID_SETCOL__) window.__SOLID_SETCOL__(nm, CELLINFO[nm].colour);
  }
  if(typeof XGT !== "undefined" && XGT && XGT.colours0)
    XGT.colours = XGT.colours0.slice();
  document.querySelectorAll("#cellList input.cpick, #objcls input.cpick").forEach(ip => {
    const c = ip.dataset.cls; if(CELLINFO[c]) ip.value = CELLINFO[c].colour; });
  document.querySelectorAll("#xgtList input.cpick").forEach(ip => {
    const c = ip.dataset.cls, k = XGT ? XGT.classes.indexOf(c) : -1;
    if(k >= 0) ip.value = XGT.colours[k]; });
  need();
}
// draw order: most cells first, so the RARE classes land on top. Schwann is 0.06%
// of some sections; drawn under tumour it would never be seen.
const CELLDRAW = CELLS ? CELLS.classes.slice()
  .sort((a, b) => b.n_cells - a.n_cells).map(c => c.name) : [];
// The order the LISTS use, which is not the order the layer draws in: drawing goes
// most-numerous-first so the rare classes land on top, while a reader wants the
// classes this cohort is about at the top and the catch-all at the bottom.
const CELLUI_HEAD = ["Tumor", "NK_T", "B_cell", "Schwann"];
const CELLUI = CELLDRAW.slice().sort((a, b) => rankUI(a) - rankUI(b));
function rankUI(c){
  const i = CELLUI_HEAD.indexOf(c);
  if(i >= 0) return i;
  if(c === "Others") return 900;
  return 100 + CELLDRAW.indexOf(c);
}
const CELLTEX = {}, CELLSTART = {}, CELLDONE = {}, CELLFAILED = {};
function hex3(h){
  return [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255,
          parseInt(h.slice(5, 7), 16) / 255];
}
function cellsBuilt(){ return !!(SRC.mode === "fetch" && CELLS); }
// The layer was placed on the PUBLISHED reconstruction's poses. The re-registered
// arm may share them or not, and the page already knows which: same_geometry is the
// same test it uses before it lets that arm borrow the published tile pyramid. If
// they do not share a geometry the cells would sit at the other arm's positions, so
// the layer refuses rather than drawing something that merely looks plausible.
function cellsHere(){
  if(!cellsBuilt()) return false;
  if(!(S.heSwap && swapHere())) return true;
  const R = SRC.res[S.res];
  return !!(R && R.swap && R.swap.same_geometry);
}
function cellSetKey(b){ return "cells|" + b; }
function cellFile(b, i, cls){
  return CELLS.dir + "/" + b + "/cells/" + CELLS.stem[i] + "." + cls + ".webp?v=" + (CELLS.v || 0);
}
// `src` is the only thing that differs between the layers that draw cells as
// densities: which texture set, which files, which class order, which palette,
// which classes are on, and which point source takes over when magnified. The
// loader and the plane drawer take one of these and are otherwise identical.
function cellTexSource(){
  return {built: cellsBuilt, key: cellSetKey, file: cellFile, classes: CELLDRAW,
          isOn: nm => S.cellsOn.indexOf(nm) >= 0,
          colour: nm => (CELLINFO[nm] || {}).colour || "#FFFFFF",
          avail: b => (CELLS && CELLS.avail[b]) || {},
          marks: markSource};
}
// what this basis has to fetch: every switched-on class on the planes that carry it,
// and nothing else. A class that is off costs nothing.
function cellWanted(b, src){
  src = src || cellTexSource();
  const out = [];
  const av = src.avail(b);
  for(const cls of src.classes){
    if(!src.isOn(cls)) continue;
    for(const i of (av[cls] || [])) out.push([cls, i]);
  }
  return out;
}
function loadCellTextures(b, done, src){
  if(window.__VIEWER_DISPOSING__) return;
  src = src || cellTexSource();
  if(!src.built()){ done && done(); return; }
  const k = src.key(b);
  const set = CELLTEX[k] || (CELLTEX[k] = {});
  if(CELLSTART[k] === undefined){ CELLSTART[k] = 0; CELLDONE[k] = 0; CELLFAILED[k] = 0; }
  const todo = cellWanted(b, src).filter(([cls, i]) => !(cls + "|" + i in set));
  if(!todo.length || !gl){ done && done(); return; }
  CELLSTART[k] += todo.length;
  for(const [cls, i] of todo){
    set[cls + "|" + i] = null;                 // in flight; keeps it from being asked twice
    let im = new Image();
    im.onload = () => {
      if(window.__VIEWER_DISPOSING__){im.onload=null;im=null;return;}
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
      // 2048, not 1024: an overlay may be published at a finer pitch than the
      // volume canvas (the duct outline is), and a cap below its own size is a
      // smooth downscale -- a thin line comes back blurred and broken
      texUpload(im, 2048);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      if(CELLTEX[k]) CELLTEX[k][cls + "|" + i] = t; else gl.deleteTexture(t);
      CELLDONE[k]++; im.onload = null; im = null; reportCells(); need();
      if(CELLDONE[k] + CELLFAILED[k] >= CELLSTART[k]) done && done();
    };
    im.onerror = () => { CELLFAILED[k]++; reportCells(); need();
      if(CELLDONE[k] + CELLFAILED[k] >= CELLSTART[k]) done && done(); };
    im.src = src.file(b, i, cls);
  }
}
function cellsTotal(){ return CELLSTART[cellSetKey(S.basis)] || 0; }
function cellsLoaded(){ return CELLDONE[cellSetKey(S.basis)] || 0; }
function reportCells(){
  if(!S.cellsOverlay) return;
  const tot = cellsTotal(), got = cellsLoaded();
  LOADQ.set("cells", "cell textures", got, tot);
}
function cellsOnList(){ return CELLDRAW.filter(c => S.cellsOn.indexOf(c) >= 0); }

// ------------------------------------------------ one point per cell
// At 8 um/px a nucleus is under one pixel, so the density textures are the only
// honest picture at that scale. Magnified, they are the wrong picture: the reader is
// looking at cells and being shown a blur. The fine level therefore draws the CELLS,
// from their own coordinates -- 6 bytes each, positions at the 0.5 um pitch they were
// measured at, so it is exact at any magnification instead of quantised to a level.
//
// It is not an image pyramid, and deliberately: per class, per plane, per basis, a
// 0.5 um/px ladder is 902 pyramids over 186-megapixel planes. The vector form is
// 74 MB for the whole set and one 0.9 MB fetch for the plane being looked at.
//
// WHICH SHAPE A CELL IS is a property of the DATA, not of this code path: the file
// declares kind = "disc" (a radius from the nucleus area) and the renderer is chosen
// by that name. InstanSeg's outlines were not kept by the detection step; when they
// are, they arrive as kind = "polygon" with its own entry in CELLSHAPE and nothing
// here changes.
const CELLPT = SRC.cell_points || null;
const CELL_LADDER = [8, 4, 2, 1, 0.5];
const CELL_MARK_AT = 2.0;      // the rung at which cells stop being a density field
const CELL_MIN_PX = 1.5;       // a cell never shrinks below this on screen
const CELL_MAX_PX = 14.0;      // ... and never balloons past this when zoomed in
let cellLevel = null;
const CELLBUF = new Map();     // basis@index -> {vao, n} | null while in flight
function cellPointsBuilt(){ return !!(SRC.mode === "fetch" && CELLPT && CELLS); }
function cellMarksLevel(){
  cellLevel = stepLevel(CELL_LADDER, cellLevel, umPerDevicePx());
  return cellLevel;
}
// Per-cell drawing is for the plane the reader SELECTED, exactly as the image tile
// pyramid is. Fifty planes of individual cells is 5 million points and 37 MB of
// fetches for a view in which none of them is a pixel wide; the all-sections switch
// governs the density layer, which costs nothing extra because it is the same 41
// quads either way.
function marksZoomed(){
  // In single-layer mode the selected plane IS the picture being read: draw the
  // cells themselves at any zoom, never their density blur, on that plane.
  return !!(S.sel !== null && !animating()
            && (S.solo || cellMarksLevel() <= CELL_MARK_AT));
}
function cellMarksOn(){
  return !!(cellPointsBuilt() && S.cellsOverlay && cellsHere() && marksZoomed());
}
function cellPtKey(b, i){ return b + "@" + i; }
function cellPtUrl(b, i){
  const f = CELLPT.files[b] && CELLPT.files[b][i];
  return f ? (CELLPT.dir + "/" + f + "?v=" + (CELLPT.v || 0)) : null;
}
function decodeCells(buf){
  const dv = new DataView(buf);
  let magic = ""; for(let k = 0; k < 8; k++) magic += String.fromCharCode(dv.getUint8(k));
  if(magic !== "CELLPT01") throw new Error("cell points: bad magic " + magic);
  const n = dv.getUint32(8, true);
  // header: magic 0..7, n 8..11, canvas_w 12..13, canvas_h 14..15, sub_px 16..17
  const cw = dv.getUint16(12, true), ch = dv.getUint16(14, true);
  const sub = dv.getUint16(16, true);
  // the canvas the positions were written on has to be the canvas they are drawn on,
  // and the divisor has to be the one the writer used. Reading either from the wrong
  // offset does not fail: it silently rescales every cell, and the layer collapses
  // into a corner while the page goes on reporting that it drew them all.
  if(cw !== CELLS.canvas_px.width || ch !== CELLS.canvas_px.height || !sub)
    throw new Error(`cell points: ${cw}x${ch}/${sub} against canvas `
                    + `${CELLS.canvas_px.width}x${CELLS.canvas_px.height}`);
  const xy = new Uint16Array(buf, 32, 2 * n);
  const r = new Uint8Array(buf, 32 + 4 * n, n);
  const cls = new Uint8Array(buf, 32 + 5 * n, n);
  // one interleaved float buffer: x, y (canvas px), r (canvas px), class index
  const out = new Float32Array(4 * n);
  for(let k = 0; k < n; k++){
    out[4*k]     = xy[2*k] / sub;
    out[4*k + 1] = xy[2*k + 1] / sub;
    out[4*k + 2] = r[k] / sub;
    out[4*k + 3] = cls[k];
  }
  return {n: n, data: out};
}
// ---- the Xenium annotation layer. Same point format and same shader as the
// predicted cells, but its OWN buffers, its own class list and its own switch:
// it is ground truth in a different taxonomy, and merging the two would read a
// class the model cannot emit as a class the model got wrong.
const XGT = SRC.xenium_gt || null;
// ---- the DENOISED nerve layer: Nerve operator regions per plane, only regions belonging
// to a cord that spans >= 3 measured sections. One pseudo-class, Schwann blue.
const NERVE = SRC.nerve_regions || null;
const NERVEFILE = {};
if(NERVE) for(const r of NERVE.planes) NERVEFILE[r.index] = r.file;
function nerveBuilt(){ return !!NERVE; }
function nerveSource(){
  return {built: nerveBuilt, key: b => "nerve|" + b,
          file: (b, i) => NERVE.dir + "/" + b + "/nerve/" + NERVEFILE[i]
                          + "?v=" + (NERVE.v || 0),
          classes: ["Nerve regions"],
          isOn: () => regionOn(NERVE,S.nerveReg),
          colour: () => (NERVE && NERVE.colour) || "#00B0F0",
          avail: () => ({"Nerve regions": NERVE ? NERVE.planes.map(r => r.index) : []}),
          marks: () => null};
}
// Two SUB-OPTIONS of the Cell prediction card, independent of each other:
//   Nerve regions - the Nerve region operator's layer drawn on top of the celltype display
//   Denoised      - every class swaps to the set the 3-D denoise kept: a cell is
//                   drawn only where it belongs to a body that holds together
//                   across sections. Schwann there is the 3-D nerve's own cells.
// the TLS layer: two rings per plane, 3-D TLS (red) and 2-D-only TLS (green),
// each with its own switch on the same row, drawn together when both are on
const TLSR = SRC.tls_regions || null;
const TLSFILE3 = {}, TLSFILE2 = {};
if(TLSR) for(const r of TLSR.planes){ TLSFILE3[r.index] = r.file3d; TLSFILE2[r.index] = r.file2d; }
function tlsSource(kind){
  const files = kind === "3d" ? TLSFILE3 : TLSFILE2;
  return {built: () => !!TLSR, key: b => "tls" + kind + "|" + b,
          file: (b, i) => TLSR.dir + "/" + b + "/tls/" + files[i] + "?v=" + (TLSR.v || 0),
          classes: ["TLS " + kind],
          isOn: () => S.tlsReg && (kind === "3d" ? S.tls3d : S.tls2d),
          colour: () => (TLSR && (kind === "3d" ? TLSR.colour3d : TLSR.colour2d)) || "#1f5fd6",
          avail: () => ({["TLS " + kind]: TLSR ? TLSR.planes.map(r => r.index) : []}),
          marks: () => null};
}
function setTlsReg(v){
  v = !!v; S.tlsReg = v;
  const t = document.getElementById("tlsToggle"); if(t) t.checked = v;
  for(const id of ["tls3dToggle", "tls2dToggle"]){ const k = document.getElementById(id); if(k) k.disabled = !(TLSR && v); }
  if(v){ if(S.tls3d) loadCellTextures(S.basis, need, tlsSource("3d"));
         if(S.tls2d) loadCellTextures(S.basis, need, tlsSource("2d")); }
  need();
}
function setTlsKind(kind, v){
  v = !!v; if(kind === "3d") S.tls3d = v; else S.tls2d = v;
  const t = document.getElementById(kind === "3d" ? "tls3dToggle" : "tls2dToggle"); if(t) t.checked = v;
  if(v && S.tlsReg) loadCellTextures(S.basis, need, tlsSource(kind));
  need();
}
// the nerve guide lines: the fitted centreline of every nerve where it passes the plane
const NLINE = SRC.nerve_lines || null;
const NLINEFILE = {};
if(NLINE) for(const r of NLINE.planes){ NLINEFILE[r.index] = r.file; }
function nlineSource(){
  return {built: () => !!NLINE, key: b => "nline|" + b,
          file: (b, i) => NLINE.dir + "/" + b + "/nerveline/" + NLINEFILE[i] + "?v=" + (NLINE.v || 0),
          classes: ["Nerve lines"],
          isOn: () => S.nlineReg,
          colour: () => (NLINE && NLINE.colour) || "#ffd400",
          avail: () => ({"Nerve lines": NLINE ? NLINE.planes.map(r => r.index) : []}),
          marks: () => null};
}
function setNlineReg(v){
  v = !!v; S.nlineReg = v;
  const t = document.getElementById("nlineToggle"); if(t) t.checked = v;
  if(v) loadCellTextures(S.basis, need, nlineSource());
  need();
}
// the 2-D tumour territory: one ring per plane (build_tumor_planes.py)
const TUMR = SRC.tumor_regions || null;
const TUMFILE = {};
if(TUMR) for(const r of TUMR.planes){ TUMFILE[r.index] = r.file; }
function tumorSource(){
  return {built: () => !!TUMR, key: b => "tumor|" + b,
          file: (b, i) => TUMR.dir + "/" + b + "/tumor/" + TUMFILE[i] + "?v=" + (TUMR.v || 0),
          classes: ["Tumor boundary"],
          isOn: () => S.tumorReg,
          colour: () => (TUMR && TUMR.colour) || "#e0245e",
          avail: () => ({"Tumor boundary": TUMR ? TUMR.planes.map(r => r.index) : []}),
          marks: () => null};
}
function setTumorReg(v){
  v = !!v; S.tumorReg = v;
  const t = document.getElementById("tumorToggle"); if(t) t.checked = v;
  if(v) loadCellTextures(S.basis, need, tumorSource());
  need();
}
// the duct lumen layer: one ring per plane, the 3-D lumens' member cavities
const DUCTR = SRC.duct_regions || null;
const DUCTFILE = {};
if(DUCTR) for(const r of DUCTR.planes){ DUCTFILE[r.index] = r.file; }
function ductSource(){
  return {built: () => !!DUCTR, key: b => "duct|" + b,
          file: (b, i) => DUCTR.dir + "/" + b + "/duct/" + DUCTFILE[i] + "?v=" + (DUCTR.v || 0),
          classes: ["Duct lumens"],
          isOn: () => S.ductReg,
          colour: () => (DUCTR && DUCTR.colour) || "#f59e0b",
          avail: () => ({"Duct lumens": DUCTR ? DUCTR.planes.map(r => r.index) : []}),
          marks: () => null};
}
function setDuctReg(v){
  v = !!v; S.ductReg = v;
  const t = document.getElementById("ductToggle"); if(t) t.checked = v;
  if(v) loadCellTextures(S.basis, need, ductSource());
  need();
}
// Individually numbered C8 glands, in the same registered frame as nerve regions.
const GLANDR = SRC.gland_regions || null;
const GLANDFILE = {};
if(GLANDR) for(const p of GLANDR.planes) GLANDFILE[p.index] = p.files;
function glandKindOn(g){
  const kind=typeof g==="string" ? GLANDR?.kinds?.[g] : g.kind;
  return FIG.on ? FIG.gland && FIG.g2 : (kind==="2d" ? S.gland2d : S.gland3d);
}
function glandSource(){
  return {built: () => !!GLANDR, key: b => "gland|" + b,
    file: (b,i,c) => GLANDR.dir + "/" + b + "/gland/" + GLANDFILE[i][c] + "?v=" + GLANDR.v,
    classes: GLANDR ? GLANDR.classes : [], isOn: c => regionOn(GLANDR,S.glandReg) && glandKindOn(c),
    colour: c => GLANDR.colours[c],
    rect: (i,c) => GLANDR.raster_mode === "roi" ? GLANDR.planes.find(p=>p.index===i)?.raster_bounds_um?.[c] : null,
    avail: b => {
      const out = {};
      if(GLANDR && GLANDR.bases.indexOf(b)>=0) for(const c of GLANDR.classes)
        out[c] = GLANDR.planes.filter(p => p.files[c]).map(p => p.index);
      return out;
    }, marks: () => null};
}
function setGlandReg(v){
  S.glandReg = !!v;
  const e = document.getElementById("glandToggle"); if(e) e.checked = S.glandReg;
  for(const id of ["gland3dToggle","gland2dToggle"]){const p=document.getElementById(id);if(p)p.disabled=!(GLANDR && S.glandReg);}
  if(v) loadCellTextures(S.basis, need, glandSource());
  need();
}
function setGlandKind(kind,v){
  if(kind==="3d")S.gland3d=!!v;else S.gland2d=!!v;
  const e=document.getElementById(kind==="3d"?"gland3dToggle":"gland2dToggle");if(e)e.checked=!!v;
  if(v && S.glandReg)loadCellTextures(S.basis,need,glandSource());need();
}
function setNerveReg(v){
  v = !!v;
  S.nerveReg = v;
  const t = document.getElementById("nerveToggle");
  if(t) t.checked = v;
  if(v) loadCellTextures(S.basis, need, nerveSource());
  need();
}
// the 3-D denoise, PROJECTED back onto each section: the layer product
// <cls>_denoised3d, behind the one Denoised switch
const DEN3 = SRC.cells_denoised3d || null;
const DEN3FILE = {};
if(DEN3 && DEN3.files){
  for(const c in DEN3.files){ DEN3FILE[c] = {};
    for(const r of DEN3.files[c]) DEN3FILE[c][r.index] = r.file; }
}
function den3Source(){
  const cls = DEN3 ? (DEN3.classes || []) : [];
  return {built: () => !!DEN3, key: b => "cden3|" + b,
          file: (b, i, c) => DEN3.dir + "/" + b + "/cells/" + DEN3FILE[c][i]
                             + "?v=" + (DEN3.v || 0),
          classes: cls,
          isOn: c => S.den3d && S.cellsOn.indexOf(c) >= 0,
          colour: c => (CELLINFO[c] || {}).colour || "#FFFFFF",
          avail: () => {
            const out = {};
            for(const c of cls) out[c] = Object.keys(DEN3FILE[c] || {}).map(Number);
            return out;
          },
          marks: () => null};
}
function setDen3d(v){
  v = !!v;
  S.den3d = v;
  const t = document.getElementById("den3dToggle");
  if(t) t.checked = v;
  if(v && DEN3) loadCellTextures(S.basis, need, den3Source());
  need();
}
const XGTBUF = new Map();
const XS = {on: false, cls: [], pt: 1.0, tls: false};
function xgtUrl(b, i){
  const f = XGT && XGT.files[b] && XGT.files[b][String(i)];
  return f ? (XGT.dir + "/" + f + "?v=" + (XGT.v || 0)) : null;
}
function xgtHere(i){ return !!xgtUrl(S.basis, i); }
function loadXgt(b, i){
  const k = b + "@" + i;
  if(XGTBUF.has(k)) return XGTBUF.get(k);
  const url = xgtUrl(b, i);
  if(!url || !gl){ XGTBUF.set(k, false); return false; }
  XGTBUF.set(k, null);
  fetch(url).then(r => { if(!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
    .then(buf => {
      const d = decodeCells(buf);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const vb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, d.data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 16, 8);
      gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 16, 12);
      gl.bindVertexArray(null);
      XGTBUF.set(k, {vao: vao, vb: vb, n: d.n});
      need();
    }).catch(() => { XGTBUF.set(k, false); });
  return null;
}
// most-numerous first, so the rare classes land on top -- the same rule the
// model layer's draw order follows.
const XGTDRAW = XGT ? XGT.classes.slice()
  .sort((a, b) => ((XGT.counts || {})[b] || 0) - ((XGT.counts || {})[a] || 0)) : [];
function xgtBuilt(){ return !!(SRC.mode === "fetch" && XGT && XGT.stem); }
function xgtTexSource(){
  return {built: xgtBuilt, key: b => "xgt|" + b,
          file: (b, i, cls) => XGT.dir + "/" + b + "/cells/"
                               + XGT.stem[i] + "." + cls.replace(/\//g, "_") + ".webp?v=" + (XGT.v || 0),
          classes: XGTDRAW,
          isOn: nm => XS.cls.indexOf(nm) >= 0,
          colour: nm => XGT.colours[XGT.classes.indexOf(nm)] || "#FFFFFF",
          avail: b => (XGT.avail || {})[b] || {},
          marks: xgtSource};
}
// TLS regions called on the Xenium annotation itself (TLS Define geometry,
// run on the annotated B/T/Plasma cells) -- a sub-layer of the annotation card
const XTLS = (XGT && XGT.tls) || null;
const XTLSFILE = {};
if(XTLS) for(const r of XTLS.planes) XTLSFILE[r.index] = r.file;
function xtlsSource(){
  return {built: () => !!XTLS, key: b => "xtls|" + b,
          file: (b, i) => XGT.dir + "/" + b + "/tls/" + XTLSFILE[i]
                          + "?v=" + (XTLS.v || 0),
          classes: ["TLS (Xenium)"],
          isOn: () => XS.on && XS.tls,
          colour: () => (XTLS && XTLS.colour) || "#FF2D95",
          avail: () => ({"TLS (Xenium)": XTLS ? XTLS.planes.map(r => r.index) : []}),
          marks: () => null};
}
function xgtSource(){
  return {buf: XGTBUF, key: (b, i) => b + "@" + i, load: loadXgt,
          classes: XGT ? XGT.classes : [],
          colour: nm => XGT.colours[XGT.classes.indexOf(nm)] || "#FFFFFF",
          isOn: nm => XS.cls.indexOf(nm) >= 0,
          size: () => S.cellPt,
          canvasPx: XGT ? XGT.canvas_px : null};
}

function loadCellPoints(b, i){
  const k = cellPtKey(b, i);
  if(CELLBUF.has(k)) return CELLBUF.get(k);
  const url = cellPtUrl(b, i);
  if(!url || !gl){ CELLBUF.set(k, false); return false; }
  CELLBUF.set(k, null);                       // in flight
  fetch(url).then(r => { if(!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
    .then(buf => {
      const d = decodeCells(buf);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const vb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, d.data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);      // x, y
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 16, 8);      // r
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 16, 12);     // class
      gl.bindVertexArray(null);
      const rec = {vao: vao, vb: vb, n: d.n};
      CELLBUF.set(k, rec);
      // per-cell denoise keep masks, one byte per cell in this row order; a
      // missing file leaves the generic attribute value 1 (= keep everything)
      for(const [sub, loc] of [["keep2d", 3], ["keep3d", 4]]){
        fetch(CELLPT.dir + "/" + sub + "/z" + String(i).padStart(2, "0")
              + "." + sub + ".bin?v=" + (CELLPT.v || 0))
          .then(r => { if(!r.ok) throw 0; return r.arrayBuffer(); })
          .then(kb => {
            const a = new Uint8Array(kb);
            if(a.length !== d.n) return;
            const f = new Float32Array(a);
            gl.bindVertexArray(vao);
            const b2 = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, b2);
            gl.bufferData(gl.ARRAY_BUFFER, f, gl.STATIC_DRAW);
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, 1, gl.FLOAT, false, 0, 0);
            gl.bindVertexArray(null);
            need();
          }).catch(() => {});
      }
      need();
    })
    .catch(() => { CELLBUF.set(k, false); need(); });
  return null;
}
// device pixels per millimetre, from the projection the frame is drawn with
function pxPerMM(h){ return h / (2 * radNow()); }
// one entry per value of the file's `kind`. Adding outlines adds an entry here.
const CELLSHAPE = {disc: "disc"};
// C flips the overlay. The base style underneath it does not change, which is the
// whole point of the two being separate controls.
function toggleCellLayer(){ setCellsOverlay(!S.cellsOverlay); }
// Whether the overlay follows the layer selection or holds on every section. It only
// changes ALPHAS -- the same 41 quads are drawn either way, so it costs nothing.
function setCellsAll(v){
  S.cellsAll = !!v;
  const b = document.getElementById("cellsAllToggle");
  if(b) b.setAttribute("aria-pressed", String(S.cellsAll));
  // the resting alphas are baked into the current layout, so it has to be rebuilt
  ANIM.to = layoutFor(S.sel, S.solo); ANIM.from = ANIM.to; ANIM.t0 = 0;
  updateCellsNote();
  need();
}
function toggleCellClass(cls){
  const j = S.cellsOn.indexOf(cls);
  if(j >= 0) S.cellsOn.splice(j, 1); else S.cellsOn.push(cls);
  pressCellRows();
  loadCellTextures(S.basis, need);
  updateCellsNote();
  need();
}

// ------------------------------------------------------------------ math
function mul(a,b){ const o=new Float32Array(16);
  for(let i=0;i<4;i++) for(let j=0;j<4;j++){ let s=0;
    for(let k=0;k<4;k++) s+=a[k*4+j]*b[i*4+k]; o[i*4+j]=s; } return o; }
function ortho(l,r,b,t,n,f){ return new Float32Array([
  2/(r-l),0,0,0, 0,2/(t-b),0,0, 0,0,-2/(f-n),0,
  -(r+l)/(r-l), -(t+b)/(t-b), -(f+n)/(f-n), 1]); }
function rotX(a){ const c=Math.cos(a),s=Math.sin(a);
  return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]); }
function rotY(a){ const c=Math.cos(a),s=Math.sin(a);
  return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]); }
// The camera orientation is a matrix, not a pair of angles.
//
// rotX(el) . rotY(az) with "horizontal rotation" implemented as "change az,
// freeze el" is NOT a left-right turn on screen: az turns about the WORLD
// vertical, which the elevation has already tilted away from the screen
// vertical, so the specimen tumbles instead of turning.
//
// Rotating about the SCREEN vertical is a pre-multiplication: v = V p, and spinning
// the scene about the view-space y axis is v' = rotY(t) v, i.e. V' = rotY(t) . V.
// Every point keeps its view-space y, so the specimen turns like a turntable and its
// apparent height cannot change at all.
// opening view: what key 6 (bottom) gives, el -1.55, lifted to -1.35 so the
// block is read from just below rather than dead flat underneath it.
// This matrix IS the camera; S.az / S.el are derived readouts and changing
// them alone moves nothing.
let ORI = mul(rotX(OPEN_EL), rotY(OPEN_AZ));
function viewM(){ return ORI; }
function setOri(m){ ORI = m; deriveAngles(); }
// az / el are kept only as readouts and for the standard views
function deriveAngles(){
  S.el = Math.asin(Math.max(-1, Math.min(1, ORI[6])));
  S.az = Math.atan2(ORI[2], ORI[10]);
}
// screen-space turntable: dx spins about the screen vertical, dy tips about the
// screen horizontal. Either can be passed as 0 to be discarded entirely.
function spinScreen(dx, dy){
  let m = ORI;
  if(dx) m = mul(rotY(dx), m);
  if(dy) m = mul(rotX(dy), m);
  setOri(m);
}
function viewXY(V, p){                       // view-plane coords of a world point
  return [V[0]*p[0] + V[4]*p[1] + V[8]*p[2], V[1]*p[0] + V[5]*p[1] + V[9]*p[2]];
}
function zmm(i){
  // real z, plus the expansion offset when a plane is selected. The offset is drawn,
  // announced on the canvas, and never written anywhere that claims to be a position.
  return ((Z[i] - (ZMIN+ZMAX)/2 + spreadUm(i)) / 1000) * S.exag;
}
function zmmReal(i){ return ((Z[i] - (ZMIN+ZMAX)/2) / 1000) * S.exag; }
let DPR_ACTIVE = 1;
function devH(){ return Math.max(1, Math.round(cv.clientHeight * DPR_ACTIVE)); }
function radNow(){ return RAD0 / S.zoom; }
function umPerDevicePx(){ return 2000 * radNow() / devH(); }
// Progressive refinement: pick the plane images whose own sampling is at least as
// fine as what one device pixel now covers, so magnifying stops smearing an 8 um
// texture and switches to the 4 um one instead. The comparison is against the real
// on-screen scale (umPerDevicePx), not against the zoom number, because window size
// and device pixel ratio both change what a zoom factor means.
// Hysteresis: step finer only once the current level is being magnified past 0.9 of
// its own pitch, step coarser only well after (1.7x), so a jitter around the
// boundary cannot make it load and unload a 33 MB set over and over.
function autoRes(){
  if(SRC.mode !== "fetch" || RESES.length < 2) return;
  if(!sectionsOn()) return;                 // sections hidden: zooming the 3-D never fetches a finer ladder
  // A selected tiled section refines only its visible ROI. Loading the finer
  // full stack here competed with a few tiles for hundreds of MB of bandwidth.
  const active=secList();
  if(active.length && active.every(i=>{
    const rgb=S.colour === "native" && modsNow()[i] === "he",dir=tileDirFor(i,rgb);
    if(!dir)return false;
    const idx=tileIndex(S.basis,META.planes[i].section_id,dir);
    return idx===null || !!(rgb?idx?.rgb_levels:idx?.levels)?.length;
  })) return;
  const target = umPerDevicePx();
  const cur = parseFloat(S.res);
  const nums = RESES.map(parseFloat).sort((a,b) => b - a);   // coarse -> fine
  let want = cur;
  if(target < cur * 0.9)      want = nums.filter(r => r <= Math.max(target, nums[nums.length-1])).pop() ?? cur;
  else if(target > cur * 1.7) want = nums.find(r => r <= target) ?? cur;
  if(want === cur) return;
  if(want < cur){
    const mm=resInfo().canvas_mm,extent=Math.max(mm.width,mm.height)*1000;
    const uploadedPitch=mpp=>mpp*Math.max(1,Math.ceil(extent/mpp/STACK_TEX_CAP));
    if(uploadedPitch(want)>=uploadedPitch(cur)) return;
  }
  // a FINER ladder waits for the 3-D: until the solid payload and its small
  // bodies are in (or the sample has none, or 30 s passed) the step is deferred
  if(want < cur && !(window.__SOLID_SMALL_DONE__ || window.__SOLID_NONE__ || performance.now() - PAGE_T0 > 30000)){
    clearTimeout(autoRes._t); autoRes._t = setTimeout(autoRes, 500); return;
  }
  S.res = String(want);
  if(S.colour === "native") loadRGBSet(S.basis, S.res, need);
  loadSet(S.basis, S.res, need);
  const seg = document.getElementById("resSeg");
  if(seg) for(const b of seg.children)
    b.setAttribute("aria-pressed", String(b.dataset.v === S.res));
  need();
}
function zoomMax(){
  if(!sectionsOn() && window.__SOLID_ACTIVE__) return 200;
  // The ceiling is set by the finest sampling actually available, not by the full
  // image alone: once a plane is tiled the real floor is 0.5 um/px, so stopping at
  // the 4 um image's limit would refuse zoom the data can still support.
  let mpp = resInfo().in_plane_um_per_px;
  for(const i of secList()){
    const ladder=tileLadder(i,S.colour === "native" && modsNow()[i] === "he");
    if(ladder)mpp=Math.min(mpp,...ladder.map(L=>L.mpp));
  }
  const z = 12000 * RAD0 / (devH() * mpp);
  return Math.max(2, Math.min(200, z));
}
function sepPx(){
  const j = Math.min(N-1, S.zi+1), i = Math.min(S.zi, N-2);
  const dz = Math.abs(zmm(j) - zmm(i));
  const k = Math.sqrt(Math.sin(S.az)**2 + (Math.sin(S.el)*Math.cos(S.az))**2);
  return dz * k * devH() / (2 * radNow());
}
function sepPxAt(exag){
  const old = S.exag; S.exag = exag; const v = sepPx(); S.exag = old; return v;
}

// --------------------------------------------------------------- weights
let faded = 0;                    // planes the emphasis pushed below one 8-bit step
function sectionContext(){return sectionsOn() && (FIG.on ? secList().length>0 : !!(window.__SOLID_ACTIVE__ && S.sel!==null));}
function drawList(){
  const out = [];
  faded = 0;
  if(FIG.sheet) return out;        // the sheet draws bodies only, no section images
  const sec = FIG.on || sectionContext() ? secList() : null;   // hoisted: this runs per frame
  for(let i=0;i<N;i++){
    if(!FIG.on && S.hidden.indexOf(i) >= 0) continue;
    if(sec && sec.indexOf(i) < 0)continue;
    // "only this layer" is carried by the layout (the others sit at alpha 0), not by a
    // filter here, so leaving it fades the stack back in instead of snapping it on
    const a = sec?1:alphaFor(i);
    if(a < 1/255){ faded++; continue; }
    out.push({i: i, a: a, f: a / BASE_OPACITY});
  }
  return out;
}
function forwardN(K){
  return Math.min(K, Math.ceil(Math.log(0.10) / Math.log(1 - BASE_OPACITY)));
}

// --------------------------------------------------------------- rAF
let pending = false, lowDpr = false, lowTimer = null;
// frame cost, shown in the top readout once a frame takes longer than a vsync
// tick: the reader sees WHY the page feels slow, and which switch caused it
let FRAME_MS = 0, FRAME_N = 0;
function need(){ if(pending || window.__VIEWER_DISPOSING__) return; pending = true;
  requestAnimationFrame(() => { pending = false;
    if(window.__VIEWER_DISPOSING__) return;
    const t0 = performance.now(); draw(); paintOverlay();
    const dt = performance.now() - t0;
    FRAME_MS = FRAME_N < 5 ? dt : FRAME_MS * 0.8 + dt * 0.2; FRAME_N++;
    if(FRAME_N % 5 === 0 && typeof LOADQ !== "undefined")
      LOADQ.note("frame", FRAME_MS > 40 ? "frame " + Math.round(FRAME_MS) + " ms" : ""); }); }
function interacting(){
  // the pick readout names a screen point, so a camera move retires it
  if(S.pickMsg){ S.pickMsg = ""; }
  pickAnchor = null;
  // with the 3-D bodies/cloud on, resolution stays up: a half-res drag makes
  // the points soften and the cloud read dimmer, which is worse than the lag
  if(window.__SOLID_ACTIVE__){ need(); return; }
  lowDpr = true; DPR_ACTIVE = 1.0;
  if(lowTimer) clearTimeout(lowTimer);
  lowTimer = setTimeout(() => { lowDpr = false;
    DPR_ACTIVE = Math.min(2, window.devicePixelRatio || 1); need(); }, 150);
  need();
}
DPR_ACTIVE = Math.min(2, window.devicePixelRatio || 1);

// --------------------------------------------------------------- draw
function setCanvasSize(){
  const w = Math.max(1, Math.round(cv.clientWidth * DPR_ACTIVE));
  const h = Math.max(1, Math.round(cv.clientHeight * DPR_ACTIVE));
  if(cv.width !== w || cv.height !== h){ cv.width = w; cv.height = h; }
  return [w, h];
}
function projFor(w, h){
  const rad = radNow(), asp = w / h;
  return (asp >= 1) ? ortho(S.cx - rad*asp, S.cx + rad*asp, S.cy - rad, S.cy + rad, -60, 60)
                    : ortho(S.cx - rad, S.cx + rad, S.cy - rad/asp, S.cy + rad/asp, -60, 60);
}
function setQuadUniforms(P, i, alpha, isRGB){
  gl.uniform1i(P.u.uTex, 0);
  gl.uniform1f(P.u.uOpacity, alpha);
  gl.uniform1i(P.u.uIsRGB, isRGB ? 1 : 0);
  const c = isRGB ? WHITE : tintFor(i);
  gl.uniform3f(P.u.uTint, c[0], c[1], c[2]);
}
let lastK = 0;

function bindTex(t){ gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.bindSampler(0, SMOOTH); }

// One composite rule and no selector for it: depth-sorted alpha-over,
// back to front. Max / min / mean were order-independent but told you nothing about
// whether 50 sections stack into the right shape, which is what this page is for.
let lastMVP = null;
let OVERLAY3D = null;
// Depth of the actual tissue support, not the rectangular image. The solid OIT
// buffers call this with colour writes off so structures behind H&E stay behind.
function sectionDepthSource(){
  if(!sectionContext() || S.colour==='translucent')return null;
  const k=setKey(S.basis,S.res);
  for(const i of secList()){
    if(!FIG.on && S.hidden.includes(i))continue;
    const c=texFor(TEX[k],rgbSet(),i); if(c.t)return c.t;
  }
  return null;
}
function writeSectionDepth(MVP){
  if(!sectionDepthSource())return false;
  const k=setKey(S.basis,S.res),R=resInfo();
  gl.useProgram(PQ.p);gl.bindVertexArray(quadVAO);
  gl.uniformMatrix4fv(PQ.u.uMVP,false,MVP);
  gl.uniform2f(PQ.u.uHalf,R.canvas_mm.width/2,R.canvas_mm.height/2);
  gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.depthMask(true);gl.colorMask(false,false,false,false);
  let wrote=false;
  for(const i of secList()){
    if(!FIG.on && S.hidden.includes(i))continue;
    const c=texFor(TEX[k],rgbSet(),i); if(!c.t)continue;
    bindTex(c.t);setQuadUniforms(PQ,i,1,c.rgb);
    gl.uniform1f(PQ.u.uZ,zmmReal(i));
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);wrote=true;
  }
  gl.colorMask(true,true,true,true);gl.bindVertexArray(null);return wrote;
}
function drawStack(w, h, tex, list, MVP){
  const V = viewM();
  lastMVP = MVP;
  gl.useProgram(PQ.p); gl.bindVertexArray(quadVAO);
  gl.uniformMatrix4fv(PQ.u.uMVP, false, MVP);
  const R = resInfo();
  const hw = R.canvas_mm.width/2, hh = R.canvas_mm.height/2;
  const order = list.slice();
  order.sort((a,b) => V[10]*zmm(a.i) - V[10]*zmm(b.i));
  gl.enable(gl.BLEND); gl.blendEquation(gl.FUNC_ADD);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  const rgb = rgbSet(), sc = selScale();
  // Base style and cell overlay are independent. The overlay is drawn per plane,
  // immediately after that plane's own base quad and before the next plane in front of
  // it, so the cells sit INSIDE the stack at their own z rather than as a sheet pasted
  // over the finished picture. Drawing all the bases and then all the cells would put
  // every cell in front of every section.
  // the cell layer belongs to the sections: it is drawn per plane, at that
  // plane's z. Hiding the sections hides it too, so "show sections" means
  // the whole section layer and not just its picture.
  const fast = lowDpr;   // interacting: shed fill-rate, not geometry
  const over = sectionsOn() && S.cellsOverlay && cellsHere();
  const marks = over && cellMarksOn();
  // the annotation layer is the same kind of thing and reads through the same
  // ramp: density until the reader magnifies onto the selected plane, then cells
  const xover = !!(XGT && sectionsOn() && XS.on && cellsHere());
  const nover = !!(NERVE && sectionsOn() && regionOn(NERVE,S.nerveReg) && !((S.solo || sectionContext()) && NERVE.vector_contours));
  const t3over = !!(TLSR && sectionsOn() && S.tlsReg && S.tls3d && S.cellsOverlay);
  const t2over = !!(TLSR && sectionsOn() && S.tlsReg && S.tls2d && S.cellsOverlay);
  const dover = !!(DUCTR && sectionsOn() && S.ductReg && S.cellsOverlay);
  const gover = !!(GLANDR && GLANDR.bases.indexOf(S.basis)>=0 && sectionsOn() && regionOn(GLANDR,S.glandReg) && !(S.solo || sectionContext()));
  const nlover = !!(!FIG.on && NLINE && sectionsOn() && S.nerveReg && S.nlineReg && S.cellsOverlay);
  const tmover = !!(TUMR && sectionsOn() && S.tumorReg && S.cellsOverlay);
  const xtover = !!(XTLS && sectionsOn() && XS.on && XS.tls);
  const xmarks = xover && xgtBuilt() && marksZoomed();
  if(!marks) cellMarkStats = {plane: null, level: null, n: 0, drawn: 0};
  for(const it of order){
    const c = texFor(tex, rgb, it.i);
    // the selected plane grows in place; everything else keeps its true size
    const k = (it.i === S.sel) ? sc : 1;
    if(c.t && sectionsOn()){
    bindTex(c.t);
    setQuadUniforms(PQ, it.i, baseAlpha(it.a), c.rgb);
    gl.uniform2f(PQ.u.uHalf, hw*k, hh*k);
    gl.uniform1f(PQ.u.uZ, zmm(it.i));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    // the selected plane gets its tiles laid over the full image it just drew, so a
    // tile still in flight leaves coarse pixels rather than a hole. Not during a
    // transition: swapping the texture source mid-animation would flash.
    // A colour plane is served by the colour ladder when that section has one,
    // and falls back to the full colour image when it does not.
    // a swapped plane has no tile pyramid of its own, and the CODEX one underneath it
    // is a different picture, so zooming into it stays on the full image
    if(secList().includes(it.i) && !animating() && tileDirFor(it.i,c.rgb))
      drawTiles(it.i, baseAlpha(it.a), k, hw, hh, c.rgb);
    }
    // after this plane's base and its tiles, still at this plane's z, and on the
    // overlay's OWN alpha so the base's Opacity slider cannot dim it
    const ovOk = !fast || it.i === S.sel;
    if(over && ovOk) drawCellPlane(it.i, cellAlphaFor(it.i), sc, marks);
    // ground truth after the prediction, at the same z, so the two can be
    // compared on the same plane rather than across a page reload
    if(xover && ovOk) drawCellPlane(it.i, cellAlphaFor(it.i), sc, xmarks, xgtTexSource());
    if(nover && ovOk && NERVEFILE[it.i] !== undefined)
      drawCellPlane(it.i, cellAlphaFor(it.i), sc, false, nerveSource());
    if(t3over && ovOk && TLSFILE3[it.i] !== undefined)
      drawCellPlane(it.i, cellAlphaFor(it.i), sc, false, tlsSource("3d"));
    if(t2over && ovOk && TLSFILE2[it.i] !== undefined)
      drawCellPlane(it.i, cellAlphaFor(it.i), sc, false, tlsSource("2d"));
    if(dover && ovOk && DUCTFILE[it.i] !== undefined)
      drawCellPlane(it.i, cellAlphaFor(it.i), sc, false, ductSource());
    if(gover && GLANDFILE[it.i]) drawCellPlane(it.i, cellAlphaFor(it.i), sc, false, glandSource());
    if(nlover && ovOk && NLINEFILE[it.i] !== undefined)
      drawCellPlane(it.i, cellAlphaFor(it.i), sc, false, nlineSource());
    if(tmover && ovOk && TUMFILE[it.i] !== undefined)
      drawCellPlane(it.i, cellAlphaFor(it.i), sc, false, tumorSource());
    if(xtover && ovOk && XTLSFILE[it.i] !== undefined)
      drawCellPlane(it.i, cellAlphaFor(it.i), sc, false, xtlsSource());
  }
  gl.bindVertexArray(null);
  // The solid bodies draw LAST, into this same scene: same context, same MVP, same
  // depth exaggeration. A hook, so the bodies keep their own module, but emphatically
  // NOT a second renderer -- a second camera drifts from the sections the moment the
  // reader turns the block, and a body that does not sit on its own sections is
  // worse than no body at all.
  if(OVERLAY3D) OVERLAY3D(MVP, hw, hh, sc);
  // In single-layer mode the chosen section is the reference the bodies are being
  // read against, so it goes back on top at full opacity after them. Drawn in the
  // stack's own order it sits at its z and the body's cut face lands on the same
  // plane, which leaves the two fighting for the same pixels.
  if(!FIG.on && S.solo && S.sel !== null && sectionsOn() && !sectionContext()){
    const pick = order.find(o => o.i === S.sel);
    const pc = pick && texFor(tex, rgb, pick.i);
    if(pc && pc.t){
      gl.useProgram(PQ.p); gl.bindVertexArray(quadVAO);
      gl.uniformMatrix4fv(PQ.u.uMVP, false, MVP);
      gl.disable(gl.DEPTH_TEST);
      bindTex(pc.t);
      setQuadUniforms(PQ, pick.i, 1.0, pc.rgb);
      gl.uniform2f(PQ.u.uHalf, hw*sc, hh*sc);
      gl.uniform1f(PQ.u.uZ, zmm(pick.i));
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      // the FULL image is coarse; the tiles are the sharp version of this same
      // plane and were already drawn once further back. Re-drawing the full image
      // on top without them is what made single-layer mode look blurry while the
      // stacked view stayed sharp.
      if(!animating() && tilesDir())
        drawTiles(pick.i, 1.0, sc, hw, hh, pc.rgb);
      if(over) drawCellPlane(pick.i, cellAlphaFor(pick.i), sc, marks);
      // The Xenium layer goes wherever the prediction layer goes. Leaving it out
      // of this re-draw is why single-layer mode showed nothing: the section is
      // painted back on top at full opacity and covered it.
      if(xover) drawCellPlane(pick.i, cellAlphaFor(pick.i), sc, xmarks, xgtTexSource());
      if(nover && NERVEFILE[pick.i] !== undefined)
        drawCellPlane(pick.i, cellAlphaFor(pick.i), sc, false, nerveSource());
      if(t3over && TLSFILE3[pick.i] !== undefined)
        drawCellPlane(pick.i, cellAlphaFor(pick.i), sc, false, tlsSource("3d"));
      if(t2over && TLSFILE2[pick.i] !== undefined)
        drawCellPlane(pick.i, cellAlphaFor(pick.i), sc, false, tlsSource("2d"));
      if(dover && DUCTFILE[pick.i] !== undefined)
        drawCellPlane(pick.i, cellAlphaFor(pick.i), sc, false, ductSource());
    if(gover && GLANDFILE[pick.i]) drawCellPlane(pick.i, cellAlphaFor(pick.i), sc, false, glandSource());
      if(nlover && NLINEFILE[pick.i] !== undefined)
        drawCellPlane(pick.i, cellAlphaFor(pick.i), sc, false, nlineSource());
      if(tmover && TUMFILE[pick.i] !== undefined)
        drawCellPlane(pick.i, cellAlphaFor(pick.i), sc, false, tumorSource());
      if(xtover && XTLSFILE[pick.i] !== undefined)
        drawCellPlane(pick.i, cellAlphaFor(pick.i), sc, false, xtlsSource());
      gl.enable(gl.DEPTH_TEST);
      gl.bindVertexArray(null);
    }
  }
}
// One plane of the cell layer: the switched-on classes as separate quads at the same
// z, most-numerous first so the rare ones end up on top. The nine planes with no
// prediction get a hatched copy of their own tissue mask instead -- they are not
// empty, they were never asked, and an empty plane would say the opposite.
// The quad uses the CELL layer's canvas_mm, not the current resolution's: these
// textures belong to the 8 µm canvas, and the 4 µm build rounds to a canvas 12 µm
// taller, which would slide the cells against the tissue by about a pixel.
function drawCellPlane(i, alpha, sc, marks, src){
  src = src || cellTexSource();
  const set = CELLTEX[src.key(S.basis)]; if(!set) return;
  const hw = CELLS.canvas_mm.width / 2, hh = CELLS.canvas_mm.height / 2;
  const k = (i === S.sel) ? sc : 1;
  gl.uniform2f(PQ.u.uHalf, hw * k, hh * k);
  gl.uniform1f(PQ.u.uZ, zmm(i));
  gl.uniform1i(PQ.u.uTex, 0);
  gl.uniform1i(PQ.u.uIsRGB, 0);
  const paint = (key, hex, a, cls) => {
    const t = set[key]; if(!t) return;
    bindTex(t);
    const c = hex3(hex);
    gl.uniform3f(PQ.u.uTint, c[0], c[1], c[2]);
    gl.uniform1f(PQ.u.uOpacity, a);
    const bounds = src.rect ? src.rect(i,cls) : null;
    gl.uniform1i(PQ.u.uUseRect,bounds?1:0);
    if(bounds)gl.uniform4f(PQ.u.uRect,(bounds[0]/1000-hw)*k,(hh-bounds[3]/1000)*k,(bounds[2]/1000-hw)*k,(hh-bounds[1]/1000)*k);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.uniform1i(PQ.u.uUseRect,0);
  };
  // magnified far enough, and this is the plane being looked at: draw the cells
  // themselves instead of their density. Not both -- that would be the same cells
  // twice, once sharp and once as a halo.
  if(marks && i === S.sel && drawCellMarks(i, alpha, k, src.marks())) return;
  const mainLayer = src.key === cellSetKey;
  const dset3 = (S.den3d && DEN3 && mainLayer) ? CELLTEX["cden3|" + S.basis] : null;
  for(const cls of src.classes){
    if(!src.isOn(cls)) continue;
    // Denoised is not its own display: the classes shown are exactly the ones
    // selected; a class that HAS a denoised texture renders the denoised one,
    // every other class renders raw. Adding a denoised build for another class
    // later makes it swap here with no further wiring.
    const dt = dset3 && dset3[cls + "|" + i];
    if(dt){ bindTex(dt);
      const c = hex3(src.colour(cls));
      gl.uniform3f(PQ.u.uTint, c[0], c[1], c[2]);
      gl.uniform1f(PQ.u.uOpacity, alpha);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      continue;
    }
    paint(cls + "|" + i, src.colour(cls), alpha, cls);
  }
}
// One draw call for every cell of one plane. Returns false if the points are not
// here yet, and the caller falls back to the density texture, so magnifying shows a
// blur that sharpens rather than a hole.
let cellMarkStats = {plane: null, level: null, n: 0, drawn: 0};
// `src` is the only thing that differs between the layers that draw cells as
// points: which buffers, which class order, which palette, which classes are
// switched on, how big a cell is drawn. Everything else -- the shader, the
// uniforms, the canvas, the z, the selection scale -- is identical, so it lives
// here once. A second copy of this function is how a layer ends up missing the
// cases this one already handles.
function markSource(){
  return {buf: CELLBUF, key: cellPtKey, load: loadCellPoints,
          classes: CELLPT ? CELLPT.classes : [],
          colour: nm => (CELLINFO[nm] || {}).colour || "#FFFFFF",
          isOn: nm => S.cellsOn.indexOf(nm) >= 0,
          size: () => S.cellPt,
          canvasPx: CELLS ? CELLS.canvas_px : null};
}
function buildClassList(list, src){
  if(!list) return;
  list.innerHTML = "";
  for(const cls of src.names){
    const b = document.createElement("button");
    b.className = "crow";
    if(src.id) b.id = src.id(cls);
    b.setAttribute("aria-pressed", String(src.isOn(cls)));
    let sw;
    if(src.setColour){
      // the swatch IS the picker: click it to recolour this class everywhere
      sw = document.createElement("input");
      sw.type = "color"; sw.className = "csw cpick";
      sw.value = src.colour(cls);
      sw.dataset.cls = cls;
      sw.title = "click to change this class's colour";
      const eat = e => e.stopPropagation();
      sw.onclick = eat; sw.onpointerdown = eat;
      sw.oninput = () => { src.setColour(cls, sw.value); need(); };
    }else{
      sw = document.createElement("span");
      sw.className = "csw"; sw.style.background = src.colour(cls);
    }
    const nm = document.createElement("span");
    nm.className = "cname"; nm.textContent = cls.replace(/_/g, " ");
    b.appendChild(sw); b.appendChild(nm);
    const n = src.count ? src.count(cls) : null;
    if(n != null){
      const ct = document.createElement("span");
      ct.className = "ccount";
      ct.textContent = n >= 1e6 ? (n / 1e6).toFixed(1) + "M"
                                : Math.round(n / 1e3) + "k";
      b.appendChild(ct);
    }
    if(src.title) b.title = src.title(cls);
    b.onclick = () => { src.toggle(cls);
      b.setAttribute("aria-pressed", String(src.isOn(cls))); need(); };
    if(src.enabled) b.disabled = !src.enabled();
    list.appendChild(b);
  }
}
function drawCellMarks(i, alpha, k, src){
  src = src || markSource();
  const b = src.buf.get(src.key(S.basis, i));
  if(b === undefined){ src.load(S.basis, i); return false; }
  if(!b) return false;
  const shape = CELLSHAPE[CELLPT.kind];
  if(!shape) return false;                    // a kind this page cannot draw yet
  const hw = CELLS.canvas_mm.width / 2, hh = CELLS.canvas_mm.height / 2;
  const pal = new Float32Array(48), on = new Float32Array(16);
  for(let c = 0; c < src.classes.length && c < 16; c++){
    // the palette is indexed by the FILE's class order, not the draw order
    const nm = src.classes[c];
    const v = hex3(src.colour(nm));
    pal[3*c] = v[0]; pal[3*c+1] = v[1]; pal[3*c+2] = v[2];
    on[c] = src.isOn(nm) ? 1 : 0;
  }
  gl.useProgram(PC.p);
  gl.bindVertexArray(b.vao);
  gl.uniformMatrix4fv(PC.u.uMVP, false, lastMVP);
  gl.uniform2f(PC.u.uHalf, hw * k, hh * k);
  gl.uniform2f(PC.u.uCanvasPx, src.canvasPx.width, src.canvasPx.height);
  gl.uniform1f(PC.u.uZ, zmm(i));
  gl.uniform1f(PC.u.uPxPerMM, pxPerMM(cv.height));
  gl.uniform1f(PC.u.uMinPx, CELL_MIN_PX * DPR_ACTIVE);
  // classes filtered by the denoise switch
  const d2 = new Float32Array(16), d3f = new Float32Array(16);
  for(let c = 0; c < src.classes.length && c < 16; c++){
    if(S.den3d && DEN3 && DEN3FILE[src.classes[c]]) d3f[c] = 1;
  }
  gl.uniform1fv(PC.u.uD2, d2);
  gl.uniform1fv(PC.u.uD3, d3f);
  gl.vertexAttrib1f(3, 1.0); gl.vertexAttrib1f(4, 1.0);
  gl.uniform1f(PC.u.uMaxPx, CELL_MAX_PX * DPR_ACTIVE);
  gl.uniform1f(PC.u.uPtScale, src.size());
  gl.uniform1f(PC.u.uOpacity, alpha);
  gl.uniform3fv(PC.u.uPalette, pal);
  gl.uniform1fv(PC.u.uOn, on);
  gl.drawArrays(gl.POINTS, 0, b.n);
  gl.bindVertexArray(quadVAO);
  gl.useProgram(PQ.p);
  gl.uniformMatrix4fv(PQ.u.uMVP, false, lastMVP);
  cellMarkStats = {plane: i, level: cellLevel, n: b.n,
                   drawn: cellsOnList().length};
  return true;
}
// ------------------------------------------------------------------- tiles
// Zooming past 4 um/px is served by a tile pyramid, and only for the plane the user
// SELECTED. Two reasons, and the second is the one that decides it:
//   - a 0.5 um/px plane is 186 megapixels; fifty of them cannot be resident, and
//     nobody magnifies fifty sections at once - they magnify one;
//   - the tiles carry a contrast window measured on the finest level, while the
//     full-plane volumes carry the 8 um window. Mixing the two INSIDE one stack
//     would put a section next to its neighbours on a different grey ramp, and
//     comparing neighbours is what this page is for. So: nothing selected -> every
//     plane comes from the full images, one ramp for all fifty. One selected -> that
//     plane is tiled all the way from 8 um, so no ramp change happens inside a zoom
//     either. The switch rides on an explicit user action, never on a zoom.
// The full plane image is always drawn first and the tiles land on top of it, so a
// tile that has not arrived shows the coarse pixels rather than a hole.
const TILE_PX = 512, TILE_CACHE_MAX = 256;
const TIDX = {};                 // basis@sid -> index object, or false = not there
const TTEX = new Map();          // basis@sid@mpp@c_r -> texture | null (in flight)
const TORDER = [];
const TILE_JOBS=new Map();
let TILE_WANTED=new Map(), tilePumpQueued=false;
const TILE_METRICS={started:0,completed:0,cancelled:0,cacheHits:0};
function tileKey(basis,sid,mpp,name,kind,dir){return (dir||"-")+"|"+basis+"@"+sid+"@"+kind+mpp+"@"+name;}
function tileReport(){
  let n=0,failed=0;for(const [key,rank] of TILE_WANTED) if(rank<1000000){if(TTEX.get(key)===false)failed++;else if(!TTEX.get(key))n++;}
  LOADQ.note("tiles",[n?"zoom tiles "+n+" pending":"",failed?failed+" ROI tiles unavailable":""].filter(Boolean).join(" · "));
}
function pumpTileJobs(){
  if(window.__VIEWER_DISPOSING__) return;
  tilePumpQueued=false;
  const foregroundWaiting=[...TILE_JOBS.values()].some(j=>TILE_WANTED.has(j.key) && TILE_WANTED.get(j.key)<1000000 && !j.controller);
  for(const [key,job] of TILE_JOBS){
    if(!TILE_WANTED.has(key)){
      if(job.controller){job.controller.abort();TILE_METRICS.cancelled++;}
      TILE_JOBS.delete(key);if(TTEX.get(key)===null)TTEX.delete(key);
    }else {
      job.rank=TILE_WANTED.get(key);
      if(foregroundWaiting && job.rank>=1000000 && job.controller){
        job.controller.abort();TILE_METRICS.cancelled++;
        TILE_JOBS.set(key,{key,rank:job.rank,url:job.url});
      }
    }
  }
  let active=[...TILE_JOBS.values()].filter(j=>j.controller).length;
  const currentPending=[...TILE_JOBS.values()].some(j=>j.rank<1000000);
  for(const job of [...TILE_JOBS.values()].filter(j=>!j.controller && (!j.retryAt || j.retryAt<=performance.now())).sort((a,b)=>a.rank-b.rank)){
    if(active>=4 || (job.rank>=1000000 && (currentPending || active>=1))) break;
    active++;job.retryAt=0;job.controller=new AbortController();TILE_METRICS.started++;
    fetch(job.url,{signal:job.controller.signal,priority:job.rank<1000000?"high":"low"})
      .then(r=>{if(!r.ok){const e=Error("tile HTTP "+r.status);e.status=r.status;throw e;}return r.blob();})
      .then(blob=>window.__VIEWER_DISPOSING__ ? null : createImageBitmap(blob,{premultiplyAlpha:"premultiply"}))
      .then(bmp=>{
        if(!bmp)return;
        if(window.__VIEWER_DISPOSING__ || TILE_JOBS.get(job.key)!==job){bmp.close();return;}
        const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);texUpload(bmp);bmp.close();
        TTEX.set(job.key,t);TORDER.push(job.key);TILE_METRICS.completed++;
      }).catch(e=>{
        if(TILE_JOBS.get(job.key)!==job || e.name==="AbortError")return;
        if(e.status!==404 && e.status!==410 && (job.retries||0)<2){
          job.retries=(job.retries||0)+1;const delay=500*4**(job.retries-1);
          job.retryAt=performance.now()+delay;setTimeout(queueTilePump,delay+10);
        }else TTEX.set(job.key,false);
      })
      .finally(()=>{if(TILE_JOBS.get(job.key)===job){if(job.retryAt)job.controller=null;else TILE_JOBS.delete(job.key);}tileReport();need();queueTilePump();});
  }
  tileReport();
}
function queueTilePump(){if(!window.__VIEWER_DISPOSING__ && !tilePumpQueued){tilePumpQueued=true;queueMicrotask(pumpTileJobs);}}
window.__TILE_QA__=()=>({...TILE_METRICS,stats:tileStats,jobs:[...TILE_JOBS.values()].map(j=>({key:j.key,rank:j.rank,active:!!j.controller})),cached:TORDER.length});
let tileLevel = null, tileStats = {level: null, wanted: 0, ready: 0, rgb: false, sections:[]};

// WHICH PYRAMID A PLANE'S TILES COME FROM, per plane AND per ladder.
//
// One directory for the whole page is not enough, and the reason is a trap this
// page walked into: the published pyramid holds a GREY ladder for those sixteen
// sections built from their CODEX image. In the swapped arm those sections are
// H&E, but they share a geometry with the published arm, so the grey tiles line up
// perfectly -- and drawing them puts CODEX detail on a plane the page is calling
// H&E, the moment anyone selects it and zooms. It looks right. It is not.
//
//   grey ladder, swapped plane   -> only the swapped arm's own pyramid will do.
//                                   Absent, draw no tiles: the full image is soft
//                                   when magnified, but it is the right picture.
//   colour ladder, swapped plane -> the published pyramid, where those sixteen
//                                   colour ladders were built. Safe because they
//                                   were built from the H&E on this same geometry.
//   any plane the arms agree on  -> the published pyramid either way.
function tileDirFor(i, isRGB){
  const shared = (SRC.tiles && SRC.tiles.dir) ? SRC.tiles.dir : null;
  const own = (SRC.tiles_swap && SRC.tiles_swap.dir) ? SRC.tiles_swap.dir : null;
  if(!(S.heSwap && swapHere())) return shared;
  if(!SRC.res[S.res].swap.same_geometry) return own;      // different poses: own only
  if(swapped(i)) return own;                              // grey AND colour: the arm's own pyramid carries both for its planes
  return shared;
}
// the whole-page answer, used where no particular plane is in hand
function tilesDir(){
  const shared = (SRC.tiles && SRC.tiles.dir) ? SRC.tiles.dir : null;
  if(!(S.heSwap && swapHere())) return shared;
  if(SRC.tiles_swap && SRC.tiles_swap.dir) return SRC.tiles_swap.dir;
  return SRC.res[S.res].swap.same_geometry ? shared : null;
}
// every tile URL carries the pyramid's build stamp, so a pyramid rebuilt on a new
// placement is never served from a browser's cache of the old one
function tileVer(dir){
  const T = SRC.tiles, W = SRC.tiles_swap;
  if(T && dir === T.dir) return T.v || 0;
  if(W && dir === W.dir) return W.v || 0;
  return 0;
}
// A plane with no index.json simply has no tiles: the page keeps drawing the full
// image and says nothing. Data is still being generated, so absence is normal.
function tileIndex(basis, sid, dir){
  if(window.__VIEWER_DISPOSING__) return false;
  const key = (dir || "-") + "|" + basis + "@" + sid;
  if(key in TIDX) return TIDX[key];
  TIDX[key] = null;
  if(!dir){TIDX[key]=false;return false;}
  fetch(dir + "/" + basis + "/" + sid + "/index.json?v=" + tileVer(dir))
    .then(r => r.ok ? r.json() : null)
    .then(j => {TIDX[key]=(j && j.levels && j.levels.length)?j:false;autoRes();need();})
    .catch(() => {TIDX[key]=false;autoRes();need();});
  return null;
}
// the coarsest level that still resolves one device pixel, with the same hysteresis
// as the full-image ladder so a jitter at the boundary cannot start a load storm
// The hysteresis, in one place. A level is kept until the demand leaves the band
// around it, so a slow zoom does not flap between two ladders' worth of fetches.
// The cell layer picks its level with this same function and these same two numbers
// rather than inventing a second pair.
const LEVEL_IN = 0.9, LEVEL_OUT = 1.7;
function stepLevel(ladder, cur, target){
  const finest = ladder[ladder.length - 1];
  const at = t => { const ok = ladder.filter(m => m <= t); return ok.length ? ok[0] : finest; };
  let c = (cur !== null && ladder.indexOf(cur) >= 0) ? cur : ladder[0];
  if(target < c * LEVEL_IN || target > c * LEVEL_OUT) c = at(target);
  return c;
}
function pickTileLevel(levels){
  const ladder = levels.map(L => L.mpp).sort((a, b) => b - a);
  if(EXPORT_JOB){
    const target=2000*radNow()/Math.max(1,Math.round(cv.clientHeight*EXPORT_JOB.dpr));
    return ladder.find(m=>m<=target) ?? ladder[ladder.length-1];
  }
  tileLevel = stepLevel(ladder, tileLevel, umPerDevicePx());
  return tileLevel;
}
function tileTexture(basis, sid, mpp, name, kind, dir, rank=0){
  if(window.__VIEWER_DISPOSING__) return null;
  const key = tileKey(basis,sid,mpp,name,kind,dir);
  TILE_WANTED.set(key,Math.min(rank,TILE_WANTED.get(key)??Infinity));
  if(TTEX.has(key)){
    const v = TTEX.get(key);
    if(v){const ix=TORDER.indexOf(key);if(ix>=0)TORDER.splice(ix,1);TORDER.push(key);TILE_METRICS.cacheHits++;}
    return (v === false) ? null : v;              // false = it 404ed; do not retry
  }
  TTEX.set(key,null);
  TILE_JOBS.set(key,{key,rank,url:dir+"/"+basis+"/"+sid+"/"+kind+mpp+"/"+name+".webp?v="+tileVer(dir)});
  queueTilePump();
  return null;
}
function evictTiles(live){
  let attempts=TORDER.length;
  while(TORDER.length > TILE_CACHE_MAX && attempts-->0){
    const k = TORDER.shift();
    if(live && live.has(k)){ TORDER.push(k); continue; }
    const t = TTEX.get(k);
    if(t) gl.deleteTexture(t);
    TTEX.delete(k);
  }
}
// what part of the section is on screen, as a uv box, by asking the same ray/plane
// intersection the picker uses at the four canvas corners
function visibleUV(i){
  const V = viewM(), r = cv.getBoundingClientRect();
  let u0 = 1, u1 = 0, v0 = 1, v1 = 0;
  for(const p of [[r.left, r.top], [r.right, r.top], [r.left, r.bottom], [r.right, r.bottom]]){
    const vv = clientToView(p[0], p[1]);
    const g = rayHit(V, i, vv[0], vv[1]);
    if(!g) return null;                            // edge-on: no tiles this frame
    const sc=i===S.sel?selScale():1,u=(g.u-.5)/sc+.5,v=(g.v-.5)/sc+.5;
    u0 = Math.min(u0, u); u1 = Math.max(u1, u);
    v0 = Math.min(v0, v); v1 = Math.max(v1, v);
  }
  // Undo the selected-plane scale above, then allow a small view-relative margin.
  // A fixed fraction of the whole slide fetched far outside a magnified ROI.
  const mu = (u1 - u0) * 0.03, mv = (v1 - v0) * 0.03;
  return {u0: Math.max(0, u0 - mu), u1: Math.min(1, u1 + mu),
          v0: Math.max(0, v0 - mv), v1: Math.min(1, v1 + mv)};
}
// Which ladder a plane is drawn from: the slide's own colour when the page is in
// native colour AND that section has a colour pyramid, otherwise the grey one. H&E is
// the only modality with a colour ladder - CODEX and Xenium are single channel and get
// their colour from a tint in the shader, so their grey tiles already are the picture.
function tileLadder(i, isRGB){
  const dir = tileDirFor(i, isRGB);
  if(!dir) return null;
  const idx = tileIndex(S.basis, META.planes[i].section_id, dir);
  if(!idx) return null;
  // In native colour a section with no COLOUR pyramid keeps its 8 um plane
  // untouched: the grey ladder must never stand in, or zooming visibly turns a
  // native view grey. The pyramid belongs to grey mode.
  const L = isRGB ? idx.rgb_levels : idx.levels;
  return (L && L.length) ? L : null;
}
function drawTiles(i, alpha, k, hw, hh, isRGB){
  const sid = META.planes[i].section_id;
  const ladder = tileLadder(i, isRGB);
  if(!ladder) return false;
  // the ladder may have fallen back to grey; the FILE PREFIX has to fall back with
  // it, or every request goes to rgb/ and 404s
  const idxNow = tileIndex(S.basis, META.planes[i].section_id, tileDirFor(i, isRGB));
  const useRGB = isRGB && !!(idxNow && idxNow.rgb_levels && idxNow.rgb_levels.length);
  const kind = useRGB ? "rgb/" : "";
  const mpp = pickTileLevel(ladder);
  const L = ladder.find(L => Math.abs(L.mpp - mpp) < 1e-9);
  if(!L || !L.present || !L.present.length) return false;
  const box = visibleUV(i);
  if(!box) return false;
  const present = L._set || (L._set = new Set(L.present));
  const c0 = Math.max(0, Math.floor(box.u0 * L.width / TILE_PX));
  const c1 = Math.min(L.cols - 1, Math.ceil(box.u1 * L.width / TILE_PX)-1);
  const r0 = Math.max(0, Math.floor(box.v0 * L.height / TILE_PX));
  const r1 = Math.min(L.rows - 1, Math.ceil(box.v1 * L.height / TILE_PX)-1);
  let wanted = 0, ready = 0;
  gl.uniform1i(PQ.u.uUseRect, 1);
  for(let r = r0; r <= r1; r++) for(let c = c0; c <= c1; c++){
    const name = c + "_" + r;
    if(!present.has(name)) continue;              // no tissue there; nothing to draw
    wanted++;
    const rank=(c-(c0+c1)/2)**2+(r-(r0+r1)/2)**2;
    const t = tileTexture(S.basis, sid, mpp, name, kind, tileDirFor(i, isRGB),rank);
    if(!t) continue;                              // still in flight: the floor shows
    ready++;
    const px0 = c * TILE_PX, px1 = Math.min((c + 1) * TILE_PX, L.width);
    const py0 = r * TILE_PX, py1 = Math.min((r + 1) * TILE_PX, L.height);
    const x0 = -hw * k + 2 * hw * k * (px0 / L.width);
    const x1 = -hw * k + 2 * hw * k * (px1 / L.width);
    const yTop = hh * k - 2 * hh * k * (py0 / L.height);
    const yBot = hh * k - 2 * hh * k * (py1 / L.height);
    bindTex(t);
    setQuadUniforms(PQ, i, alpha, useRGB);
    gl.uniform4f(PQ.u.uRect, x0, yBot, x1, yTop);
    gl.uniform1f(PQ.u.uZ, zmm(i));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  gl.uniform1i(PQ.u.uUseRect, 0);
  tileStats.sections.push({section:sid,index:i,level:mpp,wanted,ready,rgb:!!useRGB});
  tileStats.level=mpp;tileStats.wanted+=wanted;tileStats.ready+=ready;tileStats.rgb=!!useRGB;
  // Prefetch only the same physical ROI on the two adjacent measured sections.
  // Background requests start after the current ROI is complete, at one at a time.
  for(const j of [i-1,i+1]){
    if(j<0||j>=N||modsNow()[j]!==modsNow()[i])continue;
    const adj=tileLadder(j,isRGB),A=adj?.find(a=>a.mpp===mpp);if(!A)continue;
    const aset=A._set||(A._set=new Set(A.present));
    for(let r=Math.max(0,Math.floor(box.v0*A.height/TILE_PX));r<=Math.min(A.rows-1,Math.ceil(box.v1*A.height/TILE_PX)-1);r++)
      for(let c=Math.max(0,Math.floor(box.u0*A.width/TILE_PX));c<=Math.min(A.cols-1,Math.ceil(box.u1*A.width/TILE_PX)-1);c++){
        const name=c+"_"+r;if(aset.has(name))tileTexture(S.basis,META.planes[j].section_id,mpp,name,kind,tileDirFor(j,isRGB),1000000+(c-(c0+c1)/2)**2+(r-(r0+r1)/2)**2);
      }
  }
  return ready > 0;
}
function rgbSet(){ return RGBTEX[setKey(S.basis, S.res)] || null; }
// While a finer set is still arriving (it is queued behind the 3-D, then fetched
// plane by plane) every plane it has not delivered yet is drawn from the coarsest
// set that has it, so a zoom during loading refines the picture instead of
// blanking it. The coarsest set is what the page opens on; it is never evicted
// before the finer one is complete (evict keeps the two newest sets).
// the coarsest resolution key: RESES comes from Object.keys, which orders "4"
// before "8", so it is picked by value and not by position
const R_COARSE = RESES.slice().sort((a, b) => parseFloat(b) - parseFloat(a))[0];
function coarseFallback(i, wantRGB){
  const r0 = R_COARSE;
  if(String(r0) === String(S.res)) return null;
  const k = setKey(S.basis, r0);
  if(wantRGB){ const g = RGBTEX[k]; if(g && g[i]) return {t: g[i], rgb: true}; }
  const t = TEX[k]; return (t && t[i]) ? {t: t[i], rgb: false} : null;
}
function texFor(tex, rgb, i){
  // No special case for the swapped planes. In this arm they ARE H&E planes -- they
  // come out of the arm's own grey and colour sets like any other plane, and
  // modsNow() is what makes the colour branch apply to them. Reaching into a
  // separate per-plane swap set here would mean two ways for a plane to be H&E.
  if(S.colour === "native" && modsNow()[i] === "he" && rgb && rgb[i])
    return {t: rgb[i], rgb: true};
  if(tex && tex[i]) return {t: tex[i], rgb: false};
  return coarseFallback(i, S.colour === "native" && modsNow()[i] === "he") || {t: null, rgb: false};
}

// Nothing is drawn around the selected plane. A line traced on one sheet inside a
// stack of fifty translucent sheets is itself just another plane, and it cannot say
// which sheet it belongs to. What the selection is made of instead: the rest drop to
// OPACITY_OTHER, the rest step away along z, the chosen one grows, and the corner
// readout names it. The strip cell keeps its own outline, which is a control, not a
// mark on the picture.

// True while a single section is on screen instead of the volume. Two things hang
// off it, and both are about NOT touching the volume while it is out of view: the
// draw below stands down, which also means setCanvasSize is not called, so the
// hidden canvas is never resized to the zero its hidden layout reports; and the
// keyboard stops driving a stack nobody can see. Coming back sets it false and
// redraws from the state that was left, so HT891Z1 resumes rather than restarts.
let BROWSING = false;
function draw(){
  if(!gl || window.__VIEWER_DISPOSING__) { return; }
  // One wanted set for the whole frame: two figure sections must not cancel
  // each other's tiles or demote the other visible section to prefetch priority.
  TILE_WANTED=new Map();tileStats={level:null,wanted:0,ready:0,rgb:false,sections:[]};
  queueTilePump();
  if(BROWSING) { return; }
  const [w, h] = setCanvasSize();
  const tex = TEX[setKey(S.basis, S.res)];
  // the saved image may ask for a background of its own: a transparent PNG is made
  // by rendering the same frame on white and again on black and solving for the
  // coverage, which needs the clear colour to be steerable for those two frames only
  const bg = SHOT_BG || BGCOL[BG];
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0,0,w,h);
  gl.disable(gl.DEPTH_TEST);
  // Depth and stencil are cleared too, although the STACK needs neither: it is a
  // painter's-algorithm blend, back to front, with depth testing off. The solid
  // bodies drawn after it DO depth-test, and a depth buffer that is never cleared
  // still holds the previous frame -- fragments then lose the test against stale
  // values in a pattern that moves with the camera, which reads as a body that is
  // solid in places and hollow in others.
  gl.clearColor(bg[0], bg[1], bg[2], 1);
  gl.depthMask(true);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT | gl.STENCIL_BUFFER_BIT);
  // A section is drawn the moment its own image is decoded: on a slow link the
  // stack builds up one section at a time, which is what tells the reader the page
  // is working. The 3-D bodies draw over it as soon as they arrive; holding the
  // whole stack back until they did left the canvas black for half a minute and
  // then painted all fifty sections in one frame.
  if(!solidReady() && !draw._wait) draw._wait = setInterval(() => {
    if(!solidReady()) return;
    clearInterval(draw._wait); draw._wait = null; need();
  }, 250);
  if(!tex && !TEX[setKey(S.basis, R_COARSE)]) return;   // nothing of this basis has arrived yet
  const list = drawList(); lastK = list.length;
  const MVP = mul(projFor(w, h), viewM());
  drawStack(w, h, tex, list, MVP);
  evictTiles(TILE_WANTED);
  updateScalebar();
  syncSheetControl();
  if(animating()) need();          // keep the expansion running to its end state
}
const SB_STEPS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];
function updateScalebar(){
  const el = document.getElementById("scalebar"); if(!el) return;
  // um per CSS pixel: umPerDevicePx is per DEVICE pixel and one CSS pixel is
  // DPR device pixels, so the two multiply. Anything else makes the number
  // depend on DPR_ACTIVE, which drops to 1 during interaction -- the bar must
  // read the same mid-zoom and settled.
  const upp = umPerDevicePx() * (DPR_ACTIVE || 1);
  let um = SB_STEPS[SB_STEPS.length - 1];
  for(const c of SB_STEPS){ if(c / upp >= 60){ um = c; break; } }
  const px = um / upp;
  document.getElementById("sbline").style.width = px.toFixed(1) + "px";
  document.getElementById("sblab").textContent =
    um >= 1000 ? (um / 1000) + " mm" : um + " \u00b5m";
}

// --------------------------------------------------------------- overlays
function fmt(x, n){ return (x).toFixed(n === undefined ? 1 : n); }
// Exact measured gland contours stay sharp in the single-section view. The
// stacked view uses depth-sorted raster planes, so this SVG never floats above a stack.
const contourSvgs={};
function paintContours(source,enabled,id){
  if(!source) return;
  let glandSvg=contourSvgs[id];
  if(!glandSvg){
    glandSvg=contourSvgs[id]=document.createElementNS("http://www.w3.org/2000/svg","svg");glandSvg.id=id;
    glandSvg.style.cssText="position:absolute;pointer-events:none;z-index:2;overflow:hidden";
    document.getElementById("stage").appendChild(glandSvg);
  }
  const on=sectionsOn() && regionOn(source,enabled) && (S.solo || sectionContext() || figFixed()) && secList().length && (!source.bases || source.bases.includes(S.basis)) && !BROWSING;
  glandSvg.style.display=on?"block":"none";
  if(!on) return;
  const r=cv.getBoundingClientRect(),w=r.width,h=r.height;
  glandSvg.style.left=cv.offsetLeft+"px";glandSvg.style.top=cv.offsetTop+"px";
  glandSvg.setAttribute("width",w);glandSvg.setAttribute("height",h);glandSvg.setAttribute("viewBox",`0 0 ${w} ${h}`);
  const rad=radNow(),asp=w/h,sc=selScale(),M=viewM(),R=resInfo();
  const hw=R.canvas_mm.width/2,hh=R.canvas_mm.height/2;
  function screen(x,y,zi){
    const p=viewXY(M,[(x/1000-hw)*sc,-(y/1000-hh)*sc,zmm(zi)]);
    return [((p[0]-S.cx)/(rad*(asp>=1?asp:1))+1)*w/2,(1-(p[1]-S.cy)/(rad*(asp>=1?1:1/asp)))*h/2];
  }
  glandSvg.replaceChildren();
  for(const zi of secList()){
    const pl=source.planes.find(p=>p.index===zi);
    for(const g of (pl?pl.regions:[])){
      if(source===GLANDR && !glandKindOn(g))continue;
      if(!figObjOn(g.id))continue;
      const colour=source.colours?.[g.id] || source.colour;
      // Authored boundary segments retain local uncertainty; the closed polygon
      // remains the sole geometry for area and hit testing.
      if(g.outline_segments_um?.length){
        for(const segment of g.outline_segments_um){
          const el=document.createElementNS(glandSvg.namespaceURI,"polyline");
          el.setAttribute("points",segment.points.map(p=>screen(p[0],p[1],zi).join(",")).join(" "));
          el.setAttribute("fill","none");el.setAttribute("stroke",colour);
          el.setAttribute("stroke-width",(window.__SOLID_SELECTION__?.()||[]).includes(g.id)?"3.5":"2");
          if(segment.uncertain)el.setAttribute("stroke-dasharray","5 3");
          el.dataset.objectId=g.id;if(id==="glandContours")el.dataset.gland=g.id;glandSvg.appendChild(el);
        }
        continue;
      }
      for(const [pi,poly] of (g.polygons_um || []).entries()){
        const holes=g.polygon_holes_um?.[pi]||[],rings=[poly,...holes].map(ring=>ring.map(p=>screen(p[0],p[1],zi)));
        const el=document.createElementNS(glandSvg.namespaceURI,holes.length?"path":"polygon");
        if(holes.length){el.setAttribute("d",rings.map(r=>"M"+r.map(p=>p.join(",")).join(" L")+" Z").join(" "));el.setAttribute("fill-rule","evenodd");el.dataset.rings=JSON.stringify(rings);}
        else el.setAttribute("points",rings[0].map(p=>p.join(",")).join(" "));
        el.setAttribute("fill",FIG.on && FIG.nerveFill && source===NERVE ? colour : "none");el.setAttribute("stroke",colour);el.setAttribute("stroke-width",(window.__SOLID_SELECTION__?.()||[]).includes(g.id)?"3.5":"2");
        if(source===NERVE && g.boundary_uncertain)el.setAttribute("stroke-dasharray","5 3");
        el.dataset.objectId=g.id;if(id==="glandContours")el.dataset.gland=g.id;glandSvg.appendChild(el);
      }
    }
  }
}
let regionLabelSvg=null;
function paintRegionLabels(){
  if(!regionLabelSvg){
    regionLabelSvg=document.createElementNS("http://www.w3.org/2000/svg","svg");
    regionLabelSvg.id="regionLabels";
    regionLabelSvg.style.cssText="position:absolute;pointer-events:none;z-index:2;overflow:hidden";
    document.getElementById("stage").appendChild(regionLabelSvg);
  }
  regionLabelSvg.replaceChildren();
  if(BROWSING || !S.showLabels) return;
  const r=cv.getBoundingClientRect(),w=r.width,h=r.height;
  regionLabelSvg.style.left=cv.offsetLeft+"px";regionLabelSvg.style.top=cv.offsetTop+"px";
  regionLabelSvg.setAttribute("width",w);regionLabelSvg.setAttribute("height",h);
  const labels=[], seen=new Set();
  const sectionLabels=(S.solo || sectionContext() || figFixed()) && secList().length && sectionsOn();
  if(sectionLabels){
    const rad=radNow(),asp=w/h,sc=selScale(),M=viewM(),R=resInfo();
    const layers=[["Nerve",NERVE,S.nerveReg,"#2166f2"],["TLS",TLSR,S.tlsReg,"#1f5fd6"],
      ["Tumor glands",GLANDR,S.glandReg,"#f0b800"],["Duct",DUCTR,S.ductReg,"#f59e1f"]];
    for(const [group,source,on,colour] of layers){
      if(!source || !regionOn(source,on) || (source.bases && !source.bases.includes(S.basis))) continue;
      for(const zi of secList()){
      const pl=source.planes.find(p=>p.index===zi);
      for(const g of pl?.regions || []){
        if(group==="TLS" && !((g.kind==="3d" && S.tls3d)||(g.kind==="2d" && S.tls2d))) continue;
        if(group==="Tumor glands" && !glandKindOn(g))continue;
        if(!figObjOn(g.id))continue;
        const profileKey=group+"|"+((g.profile_label&&g.profile_id)||g.id);
        if(seen.has(profileKey)) continue;
        if(!Number.isFinite(g.cx_um)||!Number.isFinite(g.cy_um)) continue;
        const anchor=g.label_anchor_um || [g.cx_um,g.cy_um];
        const p=viewXY(M,[(anchor[0]/1000-R.canvas_mm.width/2)*sc,-(anchor[1]/1000-R.canvas_mm.height/2)*sc,zmm(zi)]);
        labels.push({id:g.id,group,dimension:"2d",text:g.profile_label||g.gland_id||g.id,colour:source.colours?.[g.id]||colour,
          x:((p[0]-S.cx)/(rad*(asp>=1?asp:1))+1)*w/2,y:(1-(p[1]-S.cy)/(rad*(asp>=1?1:1/asp)))*h/2});
        seen.add(profileKey);
      }
      }
    }
  }
  // A single-section view labels its actual profiles, including the parent 3D
  // IDs of branch sections. Mesh centres from other z planes do not belong here.
  for(const p of (!sectionLabels ? window.__SOLID_LABELS__?.() || [] : [])){
    if(!seen.has(p.group+"|"+p.id)) labels.push({...p,dimension:"3d",x:p.sx*w/cv.width,y:p.sy*h/cv.height});
  }
  const occupied=[];
  const selected=window.__SOLID_SELECTED__?.();
  labels.sort((a,b)=>(b.id===selected?1:0)-(a.id===selected?1:0));
  for(const p of labels){
    if(!Number.isFinite(p.x)||!Number.isFinite(p.y)||p.x<0||p.x>w||p.y<0||p.y>h) continue;
    const halfWidth=Math.max(22,p.text.length*4.5),anchor=[p.x,p.y];
    const boxAt=(x,y)=>[x-halfWidth-3,y-12,x+halfWidth+3,y+8];
    const collides=box=>occupied.some(b=>box[0]<b[2]&&box[2]>b[0]&&box[1]<b[3]&&box[3]>b[1]);
    let box=boxAt(p.x,p.y);
    if(p.dimension==="3d" && collides(box)) continue;
    if(p.dimension==="2d" && collides(box)){
      for(const dy of [-24,24,-48,48,-72,72]){
        const candidate=boxAt(p.x,p.y+dy);
        if(candidate[1]>=0 && candidate[3]<=h && !collides(candidate)){
          p.y+=dy;box=candidate;break;
        }
      }
    }
    occupied.push(box);
    if(p.y!==anchor[1]){
      const line=document.createElementNS(regionLabelSvg.namespaceURI,"line");
      line.setAttribute("x1",anchor[0]);line.setAttribute("y1",anchor[1]);
      line.setAttribute("x2",p.x);line.setAttribute("y2",p.y+(p.y<anchor[1]?7:-12));
      line.setAttribute("stroke",p.colour);line.setAttribute("stroke-width","1.5");
      regionLabelSvg.appendChild(line);
    }
    const t=document.createElementNS(regionLabelSvg.namespaceURI,"text");
    t.setAttribute("x",p.x);t.setAttribute("y",p.y);t.setAttribute("fill",p.colour);
    t.setAttribute("stroke","#fff");t.setAttribute("stroke-width","3");t.setAttribute("paint-order","stroke");
    t.style.cssText="font:bold 14px system-ui;text-anchor:middle";
    t.dataset.labelGroup=p.group;t.dataset.labelDimension=p.dimension;t.dataset.objectId=p.id;t.textContent=p.text;regionLabelSvg.appendChild(t);
  }
}
function paintOverlay(){
  paintContours(GLANDR,S.glandReg,"glandContours");
  if(NERVE?.vector_contours)paintContours(NERVE,S.nerveReg,"nerveContours");
  paintRegionLabels();
  const R = resInfo(), K = lastK;
  const p = META.planes[S.zi];

  // ONE line stays on screen. Slice number, real z, modality.
  // Everything else is either conditional or in the info panel.
  document.getElementById("corner").textContent = sectionReadout();

  // 11(b): five conditional lines. Each is "the current state differs from the
  // default", so on a page nobody has touched, none of them exist at all.
  const wb = document.getElementById("warnbadges"); wb.innerHTML = "";
  function badge(txt, warn){
    const d = document.createElement("div");
    d.className = warn ? "wbadge" : "chip";
    d.textContent = txt; wb.appendChild(d);
  }
  const spread = maxSpreadUm();
  if(spread > 0.5)
    badge("expanded +" + Math.round(spread) + " \u00b5m \u00b7 NOT real z", true);
  if(S.exag !== 1) badge("depth \u00d7" + fmt(S.exag,1), true);
  if(S.basis !== BASES[0]) badge(META.bases[S.basis].short, true);
  const loaded = LOADED[setKey(S.basis, S.res)] || 0;
  if(loaded < N) badge(loaded + " / " + N, false);
  const fails = (FAILED[setKey(S.basis, S.res)] || []).length;
  if(fails) badge(fails + " plane" + (fails > 1 ? "s" : "") + " failed to load", true);
  if(S.colour === "native" && heRGBTotal() && heRGBLoaded() < heRGBTotal())
    badge("colour " + heRGBLoaded() + " / " + heRGBTotal(), false);
  // the swap changes what the picture IS, so it is announced whenever it is on -- and
  // while its textures are still arriving it says how many planes have actually turned
  if(S.heSwap && swapHere())
    badge("re-registered arm \u00b7 " + SWAPSET.size + " planes as "
          + [...new Set([...SWAPSET].map(i => ARM_LABEL[MODS_SWAP[i]] || MODS_SWAP[i]))].join("/"), true);
  else if(S.heSwap)
    badge("re-registered arm not built at " + S.res + " \u00b5m/px", true);
  if(S.colour === "translucent")
    badge("base at " + Math.round(TRANSLUCENT_W * 100) + "% · not the full stack",
          true);
  // The cell layer is the one thing on this page that is not a measurement, so
  // while it is on it says so on the picture, not only in a panel nobody opened.
  if(S.cellsOverlay && cellsBuilt()){
    if(!cellsHere())
      badge("cell prediction sits on the published poses \u00b7 turn the "
            + "re-registered arm off to see it", true);
    else{
      const on = cellsOnList();
      badge(on.length ? (on.length <= 4
              ? on.map(c => c.replace(/_/g, " ")).join(" \u00b7 ")
              : on.length + " classes drawn")
            : "no cell class switched on", !on.length);
      const ct = cellsTotal(), cg = cellsLoaded();
      if(cg < ct) badge("cells " + cg + " / " + ct, false);
      if(S.cellsAll && S.sel !== null)
        badge("overlay on all " + CELLS.with.length + " predicted sections", false);
      // when the picture stops being a density field and becomes the cells, say so:
      // the two answer different questions and look different on purpose
      if(cellMarkStats.plane !== null)
        badge("per cell · " + cellMarkStats.n.toLocaleString() + " nuclei at "
              + cellMarkStats.level + " µm/px", false);
    }
  }
  // one layer on its own: say so in words rather than drawing ghosts of the other 49,
  // which cost picture and still could not be read
  if(S.solo && S.sel !== null) badge("showing 1 / " + N + " layers", true);
  if(S.pickMsg) badge(S.pickMsg, true);
  if(S.transient) badge(S.transient, false);

  infoLive(p, R, K);
  banner();
  const g = document.getElementById("guide"); if(g) drawGuide();
}
// 11(c): the numbers that used to sit on the canvas live in the info panel, still
// live, but only when someone opens it.
function infoLive(p, R, K){
  const el = document.getElementById("infolive"); if(!el) return;
  const prev = S.zi > 0 ? (Z[S.zi] - Z[S.zi-1]) : null;
  const next = S.zi < N-1 ? (Z[S.zi+1] - Z[S.zi]) : null;
  const tf = (META.tissue_frac && META.tissue_frac[S.basis])
    ? META.tissue_frac[S.basis][S.zi] : null;
  el.textContent =
    "plane " + (S.zi+1) + " of " + N + "   " + p.section_id.split("-").pop()
      + "   z=" + Math.round(p.z_um) + " \u00b5m   " + modLabel(S.zi)
      + (prev === null ? "" : "   previous -" + prev)
      + (next === null ? "" : "   next +" + next) + " \u00b5m\n"
    + "planes drawn " + K + " of " + N + " at " + Math.round(SEL.OPACITY_BASE*100)
      + "% each"
      + (S.sel === null ? "" : "   (selected " + META.planes[S.sel].section_id.split("-").pop()
         + " at " + SEL.OPACITY_SELECTED + ", the rest at " + SEL.OPACITY_OTHER + ")")
      + "\n"
    + "zoom " + fmt(S.zoom,2) + "\u00d7   1 device px = " + fmt(umPerDevicePx(),1)
      + " \u00b5m (source " + R.in_plane_um_per_px + " \u00b5m/px)   neighbour "
      + fmt(sepPx(),1) + " px apart\n"
    + "view az " + fmt(S.az*180/Math.PI,0) + "\u00b0 el " + fmt(S.el*180/Math.PI,0)
      + "\u00b0   depth \u00d7" + fmt(S.exag,1)
      + (tf === null ? "" : "   tissue area " + fmt(tf*100,1) + "%");
}
// --------------------------------------------------------------- banner
function banner(){
  const b = META.bases[S.basis], R = resInfo();
  const ar = R.canvas_mm.width / DEPTH_MM;
  const drawn = N - S.hidden.length;
  const l1 = (S.hidden.length
      ? drawn + " of " + N + " planes drawn (" + S.hidden.length + " hidden by you)"
      : N + " planes")
    + " . every one a measured section . zero interpolation along z . in-plane "
    + R.in_plane_um_per_px + " um/px (downsampled from the source)";
  const l3 = (S.exag === 1)
    ? "real proportions 1:" + ar.toFixed(0) + " (" + R.canvas_mm.width.toFixed(2) + " x "
      + R.canvas_mm.height.toFixed(2) + " mm in plane against " + DEPTH_MM.toFixed(3)
      + " mm of depth)"
    : '<span class="w">depth stretched ' + fmt(S.exag,1)
      + "x - the depth axis is NOT to scale</span>";
  const l6 = "one composite rule: depth-sorted alpha-over. It is view-dependent and "
    + "front-weighted, so turning 180 degrees shows you a different set of planes. "
    + "Selecting a plane pushes the others apart along z; that offset is drawn, "
    + "announced, and never a measurement.";
  document.getElementById("banner").innerHTML =
    l1 + "\n"
    + "tree: pre-registered minimum total |dz| . anchor " + META.anchor
      + " . current basis: " + b.label + "\n"
    + l3 + "\n" + META.encoding_line + "\n" + META.grey_line + "\n" + l6;
}

// --------------------------------------------------------------- ruler
const ruler = document.getElementById("ruler");
const rcur = document.getElementById("rcur"), rhint = document.getElementById("rhint");
function buildRuler(){
  ruler.querySelectorAll(".rtick,.rtick2,.rgap,.rgaplab,.rzlab,.rlink")
       .forEach(e=>e.remove());
  // the ruler maps z across ITS OWN width; when the chip strip does not fill the
  // row (few sections), a full-width ruler puts the last tick far right of the
  // last chip and its leader line runs off screen. Match the strip's width.
  const stripW = cellsBox ? cellsBox.scrollWidth : 0;
  if(stripW > 100) ruler.style.width = stripW + "px";
  for(const g of GAPS){
    const d = document.createElement("div"); d.className = "rgap";
    d.style.left = ((g.from_z_um - ZMIN)/ZSPAN*100) + "%";
    d.style.width = ((g.to_z_um - g.from_z_um)/ZSPAN*100) + "%";
    ruler.appendChild(d);
    const l = document.createElement("div"); l.className = "rgaplab";
    l.style.left = (((g.from_z_um + g.to_z_um)/2 - ZMIN)/ZSPAN*100) + "%";
    l.textContent = g.dz_um + "um"; ruler.appendChild(l);
  }
  for(let i=0;i<N;i++){
    const pReal = (Z[i] - ZMIN)/ZSPAN*100;
    const zd = Z[i] + spreadUm(i);          // where it is actually drawn right now
    const pDraw = Math.max(-2, Math.min(102, (zd - ZMIN)/ZSPAN*100));
    const t = document.createElement("div"); t.className = "rtick";
    t.style.left = pReal + "%"; ruler.appendChild(t);
    const t2 = document.createElement("div"); t2.className = "rtick2";
    t2.style.left = pDraw + "%"; ruler.appendChild(t2);
    const lk = document.createElement("div"); lk.className = "rlink";
    lk.style.left = Math.min(pReal,pDraw) + "%";
    lk.style.width = Math.abs(pDraw-pReal) + "%"; ruler.appendChild(lk);
  }
  for(const zz of [0,100,200,300,400,500]){
    const l = document.createElement("div"); l.className = "rzlab";
    l.style.left = ((zz - ZMIN)/ZSPAN*100) + "%"; l.textContent = zz;
    ruler.appendChild(l);
  }
}
document.getElementById("rulerbox").addEventListener("mousemove", e => {
  const r = ruler.getBoundingClientRect();
  const f = Math.max(0, Math.min(1, (e.clientX - r.left)/(r.width || 1)));
  const z = ZMIN + f*ZSPAN;
  let best = 0; for(let i=1;i<N;i++) if(Math.abs(Z[i]-z) < Math.abs(Z[best]-z)) best = i;
  rhint.style.display = "block"; rhint.style.left = (f*100) + "%";
  rhint.textContent = z.toFixed(0) + " um  nearest "
    + META.planes[best].section_id.split("-").pop();
});
document.getElementById("rulerbox").addEventListener("mouseleave",
  () => { rhint.style.display = "none"; });

// --------------------------------------------------------------- layer strip
const selector = document.getElementById("selector"), cellsBox = document.getElementById("cells");
let scrub = false;
function buildCells(){
  for(let i=0;i<N;i++){
    const p = META.planes[i];
    const c = document.createElement("div");
    c.className = "cell"; c.id = "cell-" + i; c.dataset.i = String(i);
    c.dataset.mod = MOD_TRUE[i]; c.tabIndex = -1; c.setAttribute("role","option");   // the strip names what the section IS, whichever arm is shown
    // the U number wherever it sits in the id: "HT891Z1-U66", "S22-27909-P1_U74" and
    // "HT206B1-H2L1Us1_17" (series marker s1, section 17) all label as their number
    const utok = (p.section_id.match(/U(?:s\d+_)?(\d+)[^-_]*$/) || [null, p.section_id])[1];
    const num = document.createElement("span");
    num.textContent = utok;
    c.appendChild(num);
    if(MODMIX){
      const mo = document.createElement("span");
      mo.className = "cmod"; mo.textContent = MOD_TRUE[i];
      c.appendChild(mo);
    }
    c.title = "U" + utok + "  z=" + p.z_um + " um  " + modLabel(i);
    c.setAttribute("aria-label", c.title);
    // The strip cell is a toggle -- clicking the slice number that
    // is already selected clears the selection and plays the expansion backwards
    c.addEventListener("mousedown", e => { e.preventDefault(); scrub = true;
      pointerFocus = true;
      setSelected(S.sel === i ? null : i);
      setTimeout(() => { cv.focus(); pointerFocus = false; }, 0);
    });
    c.addEventListener("mouseenter", () => { if(scrub && S.sel !== null) setSelected(i); });
    cellsBox.appendChild(c);
    if(GAP_AFTER[i] !== undefined && i < N-1){
      const g = document.createElement("div"); g.className = "gapmark";
      const s = document.createElement("span"); s.textContent = GAP_AFTER[i] + "um";
      g.appendChild(s);
      g.title = GAP_AFTER[i] + " um with no section between "
        + META.planes[i].section_id.split("-").pop() + " and "
        + META.planes[i+1].section_id.split("-").pop();
      cellsBox.appendChild(g);
    }
  }
}
window.addEventListener("mouseup", () => { scrub = false; });
const guide = document.getElementById("guide");
function drawGuide(){
  if(!guide.getBoundingClientRect) return;
  const r = ruler.getBoundingClientRect(), gb = guide.getBoundingClientRect();
  const x1 = r.left - gb.left + r.width * (Z[S.zi] - ZMIN)/ZSPAN;
  const cell = document.getElementById("cell-" + S.zi);
  let svg = "";
  if(cell){
    const cb = cell.getBoundingClientRect(), sb = selector.getBoundingClientRect();
    const x2 = cb.left - gb.left + cb.width/2;
    if(cb.right > sb.left && cb.left < sb.right)
      svg = '<line x1="'+x1+'" y1="0" x2="'+x2+'" y2="16" stroke="var(--accent)" '
          + 'stroke-width="1"/>';
  }
  guide.innerHTML = svg;
}

// --------------------------------------------------------------- selection
function setZ(i){
  if(!Number.isFinite(i)) return;
  i = Math.trunc(i);
  S.zi = Math.max(0, Math.min(N-1, i));
  for(const c of cellsBox.querySelectorAll(".cell")){
    const k = +c.dataset.i;
    // "sel" is the plane the user chose and can clear by clicking again; "cur" is
    // simply where the layer controls are pointing. They are different things.
    c.classList.toggle("sel", k === S.sel);
    c.classList.toggle("cur", k === S.zi && k !== S.sel);
    c.classList.toggle("hid", S.hidden.indexOf(k) >= 0);
    if(k === S.sel) c.title = c.getAttribute("aria-label")
      + "  - selected; click again to clear";
  }
  const cell = document.getElementById("cell-" + S.zi);
  if(cell && cell.getBoundingClientRect){
    const sb = selector.getBoundingClientRect(), cb = cell.getBoundingClientRect();
    if(cb.left < sb.left) selector.scrollLeft -= (sb.left - cb.left) + 40;
    else if(cb.right > sb.right) selector.scrollLeft += (cb.right - sb.right) + 40;
  }
  rcur.style.left = ((Z[S.zi] - ZMIN)/ZSPAN*100) + "%";
  const p = META.planes[S.zi];
  document.getElementById("curlab").textContent =
    p.section_id.split("-").pop() + " . z=" + p.z_um + " um . " + modLabel(S.zi)
    + "  (" + (S.zi+1) + " / " + N + ")";
  S.pivot = [0, 0, zmm(S.zi)];
  need();
}
function setBasis(v){
  S.basis = v;
  const box = document.getElementById("basisSeg");
  for(const c of box.children) c.setAttribute("aria-pressed", String(c.dataset.v === v));
  document.getElementById("basisNote").textContent = META.bases[v].label;
  loadSet(v, S.res, need);
  // the colour set belongs to a basis@res too, so a switch has to fetch its own one
  // or H&E would quietly fall back to grey after the switch
  if(S.colour === "native") loadRGBSet(v, S.res, need);
  // the cell layer is per basis as well: the two bases put the same section on the
  // canvas with different matrices, so they are different pictures of the same cells
  if(S.cellsOverlay) loadCellTextures(v, need);
  if(XS.on) loadCellTextures(v, need, xgtTexSource());
  if(XS.on && XS.tls && XTLS) loadCellTextures(v, need, xtlsSource());
  if(S.nerveReg) loadCellTextures(v, need, nerveSource());
  if(S.tlsReg && S.tls3d) loadCellTextures(v, need, tlsSource("3d"));
  if(S.tlsReg && S.tls2d) loadCellTextures(v, need, tlsSource("2d"));
  if(S.ductReg && DUCTR) loadCellTextures(v, need, ductSource());
  if(S.glandReg && GLANDR) loadCellTextures(v, need, glandSource());
  if(S.nerveReg && S.nlineReg && NLINE) loadCellTextures(v, need, nlineSource());
  if(S.tumorReg && TUMR) loadCellTextures(v, need, tumorSource());
  if(S.den3d && DEN3) loadCellTextures(v, need, den3Source());
  need();
}
function press(id, v){
  const box = document.getElementById(id); if(!box) return;
  for(const c of box.children) c.setAttribute("aria-pressed", String(c.dataset.v === String(v)));
}
// 11(d): a control does not carry a printed copy of its own value. While it is being
// dragged the value appears for a moment and then goes away again.
let flashT2 = null;
function flash(txt){
  S.transient = txt;
  if(flashT2) clearTimeout(flashT2);
  flashT2 = setTimeout(() => { S.transient = ""; need(); }, 1200);
  need();
}
// The BASE STYLE: three, mutually exclusive, and none of them knows about the overlay.
function setColour(v){
  S.colour = v; press("colourSeg", v);
  if(v === "native") loadRGBSet(S.basis, S.res, need);
  else reportProgressCurrent();
  // the swapped planes keep their own grey and their own colour, so changing the
  // display mode means fetching the other one of the two
  need();
}
// The OVERLAY: a switch, not a fourth base style. It composes with all three.
function setCellsOverlay(v){
  if(!cellsBuilt()) return;
  S.cellsOverlay = !!v;
  if(S.cellsOverlay && XS.on){
    XS.on = false;
    const x = document.getElementById("xgtShow");
    if(x) x.checked = false;
    const xb = document.getElementById("xgtBody");
    if(xb) xb.dataset.on = "0";
  }
  const b = document.getElementById("cellsToggle");
  if(b) b.checked = S.cellsOverlay;
  const box = document.getElementById("cellsBox");
  if(box) box.dataset.on = S.cellsOverlay ? "1" : "0";
  // the class switches are meaningless with the overlay off, so they are genuinely
  // disabled rather than merely faded: a disabled button is out of the tab order too
  for(const cls of CELLDRAW){
    const r = document.getElementById("crow-" + cls);
    if(r) r.disabled = !S.cellsOverlay;
  }
  const sb = document.getElementById("cellsAllToggle");
  if(sb) sb.disabled = !S.cellsOverlay;
  const app = document.getElementById("app");
  if(app) app.dataset.cells = S.cellsOverlay ? "1" : "0";
  if(S.cellsOverlay){ loadCellTextures(S.basis, need); updateCellsNote(); }
  else reportProgressCurrent();
  need();
}
// The swap switch: OFF is the default display; ON shows the sections that
// were scanned twice as their H&E instead of their CODEX.
function setHeSwap(v){
  S.heSwap = !!v; press("heSwapSeg", S.heSwap ? "on" : "off");
  // the arm is part of every cache key, so switching is an ordinary set load; the
  // state has to move BEFORE the load or the textures are filed under the old arm
  if(S.colour === "native") loadRGBSet(S.basis, S.res, need);
  loadSet(S.basis, S.res, need);
  if(S.cellsOverlay) loadCellTextures(S.basis, need);
  if(XS.on) loadCellTextures(S.basis, need, xgtTexSource());
  syncCellMods();
  need();
}
// the strip colour-codes by modality, and modality is a property of the arm.
// buildCells APPENDS, so this refreshes in place rather than rebuilding.
function syncCellMods(){
  const m = modsNow();
  for(let i = 0; i < N; i++){
    const c = document.getElementById("cell-" + i);
    if(!c) continue;
    c.dataset.mod = MOD_TRUE[i];
    c.title = META.planes[i].section_id.split("-").pop() + "  z="
            + META.planes[i].z_um + " um  " + modLabel(i);
    c.setAttribute("aria-label", c.title);
  }
}
const N_HE = MODS.filter(m => m === "he").length;
const N_XEN = MODS.filter(m => m === "xenium").length;
function swapNoteText(){
  if(!swapAvailable()) return "The H&E arm was not built for this volume.";
  const n = SWAPSET.size;
  return S.heSwap
    ? ("Those " + n + " sections were scanned twice - CODEX, then re-stained and "
       + "re-scanned in H&E on the SAME physical section - and this shows the H&E. "
       + "z does not change, and neither does anyone's position: the geometry is the "
       + "published reconstruction's, unchanged. No per-section registration was "
       + "fitted for these. The stack reads as " + (N_HE + n) + " H&E and " + N_XEN
       + " Xenium.")
    : ("Off: the reconstruction as published, with those " + n + " sections shown as "
       + "the CODEX that was measured on them.");
}
function reportProgressCurrent(){ reportProgress(setKey(S.basis, S.res), S.res); }
// The flyout: 44 px of icons, and one group of controls at a time on top of the canvas.
function openGroup(g){
  S.group = (S.group === g) ? null : g;
  const p = document.getElementById("panel");
  for(const el of p.querySelectorAll(".grp"))
    el.classList.toggle("open", el.dataset.g === S.group);
  p.style.display = S.group ? "block" : "none";
  for(const b of document.getElementById("rail").children)
    if(b.dataset.g) b.setAttribute("aria-pressed", String(b.dataset.g === S.group));
  need();
}
function toggleInfo(force){
  const el = document.getElementById("infoPanel");
  const on = (force === undefined) ? (el.style.display !== "block") : !!force;
  el.style.display = on ? "block" : "none";
  document.getElementById("infoBtn").setAttribute("aria-pressed", String(on));
}
function fullscreenOn(){ return !!(document.fullscreenElement
  || document.webkitFullscreenElement); }
function toggleFullscreen(){
  const el = document.getElementById("app");
  if(fullscreenOn()){
    if(document.exitFullscreen) document.exitFullscreen();
    else if(document.webkitExitFullscreen) document.webkitExitFullscreen();
  } else if(el.requestFullscreen){
    const r = el.requestFullscreen(); if(r && r.catch) r.catch(() => {});
  } else if(el.webkitRequestFullscreen) el.webkitRequestFullscreen();
}

// --------------------------------------------------------------- camera
// An object focus uses physical sample coordinates; placement follows the same
// scale/exaggeration as its mesh. It is session state, never sample data.
let bodyFocus = null, focusMotion = 0;
function bodyFocusPoint(){
  if(!bodyFocus) return null;
  const p=bodyFocus.centerUm,R=resInfo(),sc=selScale();
  return [(p[0]/1000-R.canvas_mm.width/2)*sc,-(p[1]/1000-R.canvas_mm.height/2)*sc,
    (p[2]-(ZMIN+ZMAX)/2)/1000*S.exag];
}
function useBodyPivot(){
  const p=bodyFocusPoint(); if(p && !drag?.localOrbit) S.pivot=p;
}
function focusBody(id,centerUm){
  ++focusMotion;
  bodyFocus=id ? {id,centerUm:centerUm.slice()} : null;
  if(!bodyFocus){S.pivot=[0,0,zmm(S.zi)];return;}
  // Selecting an object changes the pivot only. Keep the user's existing pan,
  // zoom and orientation, including when its mesh finishes loading later.
  useBodyPivot();
  need();
}
function zoomAt(nx, ny, factor){
  ++focusMotion;useBodyPivot();
  const zm = zoomMax();
  const before = radNow();
  if(bodyFocus){
    const p=viewXY(viewM(),bodyFocusPoint()),a=cv.width/cv.height;
    nx=(p[0]-S.cx)/(before*(a>=1?a:1));
    ny=(p[1]-S.cy)/(before*(a>=1?1:1/a));
  }
  S.zoom = Math.max(0.40, Math.min(zm, S.zoom * factor));
  const after = radNow();
  const asp = cv.width / cv.height;
  S.cx += nx * (asp >= 1 ? asp : 1) * (before - after);
  S.cy += ny * (asp >= 1 ? 1 : 1/asp) * (before - after);
  if(S.zoom >= zm - 1e-9 || S.zoom <= 0.40 + 1e-9) flashZoom();
  autoRes();
  interacting();
}
let flashT = null;
function flashZoom(){
  const el = document.getElementById("corner");
  if(RM){ el.style.color = "var(--muted)"; }
  else { el.style.opacity = "0.45"; }
  if(flashT) clearTimeout(flashT);
  flashT = setTimeout(() => { el.style.opacity = ""; el.style.color = ""; }, 400);
}
function panByPx(dxPx, dyPx){
  ++focusMotion;
  const mmPerPx = 2 * radNow() / devH();
  S.cx -= dxPx * DPR_ACTIVE * mmPerPx;
  S.cy += dyPx * DPR_ACTIVE * mmPerPx;
  interacting();
}
function rotateTo(az, el){
  ++focusMotion;useBodyPivot();
  const V0 = viewM(); const p0 = viewXY(V0, S.pivot);
  const e = Math.max(-1.55, Math.min(1.55, el));
  ORI = mul(rotX(e), rotY(az));
  S.az = az; S.el = e;                 // an absolute view: the angles ARE these
  const V1 = viewM(); const p1 = viewXY(V1, S.pivot);
  S.cx += p1[0] - p0[0]; S.cy += p1[1] - p0[1];
  interacting();
}
// dragging rotates about the axes of the SCREEN, so what the pointer does and what
// the specimen does agree at every camera attitude
function rotateByScreen(dx, dy){
  ++focusMotion;useBodyPivot();
  const V0 = viewM(); const p0 = viewXY(V0, S.pivot);
  spinScreen(dx, dy);
  const V1 = viewM(); const p1 = viewXY(V1, S.pivot);
  S.cx += p1[0] - p0[0]; S.cy += p1[1] - p0[1];
  interacting();
}
function resetView(full){
  ++focusMotion;window.__SOLID_CLEAR_FOCUS__?.();bodyFocus=null;
  S.zoom = 1; S.cx = 0; S.cy = 0;
  if(full){ setOri(mul(rotX(OPEN_EL), rotY(OPEN_AZ))); S.pivot = [0,0,0]; }
  need();
}
// SolidWorks view numbering. az/el are this page's camera angles; the camera sits
// along (-cos el sin az, sin el, cos el cos az), x/y in the section plane, z depth.
// 5 and 6 stop at el = +-1.55 rad = 88.81 deg, which is where rotateTo clamps.
const STDVIEW = {1: ["front", 0, 0], 2: ["back", Math.PI, 0],
                 3: ["left", Math.PI/2, 0], 4: ["right", -Math.PI/2, 0],
                 5: ["top", 0, 1.55], 6: ["bottom", 0, -1.55],
                 7: ["isometric", -Math.PI/4, 0.61548]};
function stdView(n){
  const v = STDVIEW[n]; if(!v) return null;
  rotateTo(v[1], v[2]); need(); return v[0];
}
function viewBounds(){                    // the whole volume, in view-plane mm
  const R = resInfo(), hw = R.canvas_mm.width/2, hh = R.canvas_mm.height/2, V = viewM();
  let x0=1e9, x1=-1e9, y0=1e9, y1=-1e9;
  for(const sx of [-1,1]) for(const sy of [-1,1]) for(const i of [0, N-1]){
    const q = viewXY(V, [sx*hw, sy*hh, zmm(i)]);
    x0=Math.min(x0,q[0]); x1=Math.max(x1,q[0]); y0=Math.min(y0,q[1]); y1=Math.max(y1,q[1]);
  }
  return [x0, y0, x1, y1];
}
function fitRect(x0, y0, x1, y1, margin){
  ++focusMotion;
  const asp = cv.width / Math.max(1, cv.height);
  const kx = asp >= 1 ? asp : 1, ky = asp >= 1 ? 1 : 1/asp;
  const rad = Math.max(Math.abs(x1-x0)/2/kx, Math.abs(y1-y0)/2/ky) * (margin || 1);
  const zm = zoomMax();
  const want = RAD0 / Math.max(rad, 1e-6);
  S.zoom = Math.max(0.40, Math.min(zm, want));
  S.cx = (x0+x1)/2; S.cy = (y0+y1)/2;
  if(want > zm + 1e-9 || want < 0.40 - 1e-9) flashZoom();
  autoRes();
  interacting();
  return S.zoom;
}
function zoomToFit(){ const b = viewBounds(); return fitRect(b[0], b[1], b[2], b[3], 1.04); }
function clientToView(clientX, clientY){
  const r = cv.getBoundingClientRect();
  const nx = ((clientX - r.left)/r.width)*2 - 1, ny = 1 - ((clientY - r.top)/r.height)*2;
  const rad = radNow(), asp = r.width/r.height;
  return [S.cx + nx*(asp>=1?rad*asp:rad), S.cy + ny*(asp>=1?rad:rad/asp)];
}

// ------------------------------------------------------- picking
// Orthographic ray through (vx,vy) in the view plane, met with plane i.
function rayHit(V, i, vx, vy){
  const det = V[0]*V[5] - V[4]*V[1];
  if(Math.abs(det) < 1e-9) return null;             // edge-on: the ray never lands
  const zw = zmm(i), r1 = vx - V[8]*zw, r2 = vy - V[9]*zw;
  const x = ( r1*V[5] - V[4]*r2) / det;
  const y = (-r1*V[1] + V[0]*r2) / det;
  const R = resInfo();
  return {x: x, y: y, u: x/R.canvas_mm.width + 0.5, v: 0.5 - y/R.canvas_mm.height,
          depth: V[2]*x + V[6]*y + V[10]*zw};       // larger = nearer the camera
}
// What a click may reach is not what the emphasis currently makes bright: a layer
// faded to below one 8-bit step is still there, and clicking through to it has to
// keep working. So picking uses the same list WITHOUT the distance falloff.
function pickCandidates(){
  if(FIG.on)return sectionsOn()?secList().map(i=>({i,a:1})):[];
  const out = [];
  for(let i=0;i<N;i++){
    if(!FIG.on && S.hidden.indexOf(i) >= 0) continue;
    // while only one layer is shown, only that layer can be clicked: the others are
    // not on screen, so clicking through to them would pick something invisible
    if(S.solo && S.sel !== null && i !== S.sel) continue;
    out.push({i: i, a: BASE_OPACITY});
  }
  return out;
}
function pickList(vx, vy){
  const V = viewM(), k = setKey(S.basis, S.res), out = [];
  let masked = false, edgeOn = false;
  for(const it of pickCandidates()){
    const g = rayHit(V, it.i, vx, vy);
    if(!g){ edgeOn = true; break; }
    // Undo the selected plane's display-only enlargement before reading tissue
    // alpha or reporting physical coordinates, just as surfaceUnderCursor does.
    const sc = it.i === S.sel ? selScale() : 1;
    const u = (g.u - .5)/sc + .5, v = (g.v - .5)/sc + .5;
    if(u < 0 || u > 1 || v < 0 || v > 1) continue;
    const av = alphaAt(k, it.i, u, v);
    if(av >= 0) masked = true;
    if(av >= 0 && av < PICK_ALPHA) continue;
    out.push({i: it.i, depth: g.depth, alpha: av, u: u, v: v});
  }
  out.sort((a, b) => b.depth - a.depth);            // front-most first
  return {edgeOn: edgeOn, hits: out, masked: masked};
}
let pickAnchor = null;
function pickAt(clientX, clientY){
  const p = clientToView(clientX, clientY);
  const P = pickList(p[0], p[1]);
  if(P.edgeOn){
    S.pickMsg = "edge-on: every layer projects to a line here, so a click cannot name one";
    need(); return null; }
  if(!P.hits.length){
    // Empty space is how you get out of a selection
    S.pickMsg = S.sel === null ? "no layer carries tissue under that point"
                              : "selection cleared - nothing under that point";
    if(S.sel !== null) setSelected(null); else need();
    return null; }
  // Stepping deeper is anchored on the POINT, not on the list of layers under it:
  // selecting a layer expands the stack, which legitimately changes that list, and
  // keying on the list would reset the step and pick the same front layer forever.
  const now = nowMs();
  let k = 0;
  if(pickAnchor && (now - pickAnchor.t) < 2500
     && Math.abs(pickAnchor.x - clientX) <= 6 && Math.abs(pickAnchor.y - clientY) <= 6)
    k = (pickAnchor.k + 1) % P.hits.length;
  pickAnchor = {x: clientX, y: clientY, k: k, t: now};
  const h = P.hits[k];
  // Picking deliberately reaches every layer, including ones the emphasis has faded
  // out -- otherwise clicking through the stack would stop after a few layers. The
  // honest half of that bargain is saying when the layer you just picked is not
  // really visible, measured on the SAME product the shader uses.
  const drawnAlpha = alphaFor(h.i);
  const faint = drawnAlpha < 2/255;
  S.pickMsg = "picked " + META.planes[h.i].section_id.split("-").pop() + " - "
    + (k+1) + " of " + P.hits.length + " layers with tissue under that point"
    + (P.masked ? "" : " (no alpha map loaded: geometry only)")
    + (faint ? " - currently drawn at alpha " + drawnAlpha.toFixed(5)
               + ", below one 8-bit step, so it is picked but not visible" : "");
  setSelected(h.i);
  return {index: h.i, order: k, hits: P.hits.map(x => x.i)};
}
function clampPanSoft(){
  const R = resInfo(), hw = R.canvas_mm.width/2, hh = R.canvas_mm.height/2;
  const V = viewM();
  let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
  for(const sx of [-1,1]) for(const sy of [-1,1]) for(const i of [0, N-1]){
    const q = viewXY(V, [sx*hw, sy*hh, zmm(i)]);
    x0=Math.min(x0,q[0]); x1=Math.max(x1,q[0]); y0=Math.min(y0,q[1]); y1=Math.max(y1,q[1]);
  }
  const rad = radNow(), asp = cv.width/cv.height;
  const vx = asp>=1 ? rad*asp : rad, vy = asp>=1 ? rad : rad/asp;
  const short = 2*Math.min(vx, vy) * 0.15;
  const ovx = Math.min(x1, S.cx+vx) - Math.max(x0, S.cx-vx);
  const ovy = Math.min(y1, S.cy+vy) - Math.max(y0, S.cy-vy);
  let tx = S.cx, ty = S.cy;
  if(ovx < short) tx = (S.cx > (x0+x1)/2) ? x1 + vx - short : x0 - vx + short;
  if(ovy < short) ty = (S.cy > (y0+y1)/2) ? y1 + vy - short : y0 - vy + short;
  if(tx !== S.cx || ty !== S.cy){
    if(RM){ S.cx = tx; S.cy = ty; need(); }
    else {
      const sx0 = S.cx, sy0 = S.cy, t0 = performance.now();
      (function step(){
        const u = Math.min(1, (performance.now()-t0)/180);
        S.cx = sx0 + (tx-sx0)*u; S.cy = sy0 + (ty-sy0)*u; need();
        if(u < 1) requestAnimationFrame(step);
      })();
    }
  }
}

// --------------------------------------------------------------- pointer
//   left drag          rotate            Ctrl + left drag     local orbit
//   left click         pick the front-most layer with tissue there; click again at
//                      the same point to go one layer deeper
//   wheel              zoom at cursor    middle drag          rotate as well
//   Shift + left drag  rotate horizontally only (the elevation is frozen at the
//                      angle the drag started from, and vertical movement is
//                      discarded rather than scaled down)
//   middle 2x click    zoom to fit
//   P                  set the rotation centre under the cursor
// A click is a press that moved less than CLICK_PX and lasted less than CLICK_MS;
// anything else is a drag, so rotating the view never selects a layer by accident.
const CLICK_PX = 4, CLICK_MS = 300;
// macOS can report Ctrl+left as button 2. That combination is local orbit;
// an unmodified right press still pans. Cmd pans on every platform.
const IS_MAC = /Mac|iPhone|iPad|iPod/i.test(
  (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform
  || navigator.userAgent || "");
function panModifier(e){ return e.metaKey; }
let drag = null;
let midClick = {t: 0, x: 0, y: 0};
let lastPointer = null;                 // for keys that act "under the cursor"
function nowMs(){ return (window.performance && performance.now)
  ? performance.now() : Date.now(); }
function pointFromView(vx,vy,vz){
  const m=viewM();
  return [m[0]*vx+m[1]*vy+m[2]*vz,m[4]*vx+m[5]*vy+m[6]*vz,m[8]*vx+m[9]*vy+m[10]*vz];
}
function surfaceUnderCursor(clientX,clientY,background){
  const [vx,vy]=clientToView(clientX,clientY),m=viewM(),k=setKey(S.basis,S.res);
  let section=null;
  if(sectionsOn())for(const it of drawList()){
    if(!FIG.on && S.solo && S.sel!==null && it.i!==S.sel)continue;
    if(!texFor(TEX[k],RGBTEX[k],it.i).t)continue;
    const g=rayHit(m,it.i,vx,vy);if(!g)continue;
    const sc=it.i===S.sel?selScale():1,u=(g.u-.5)/sc+.5,v=(g.v-.5)/sc+.5;
    if(u<0||u>1||v<0||v>1)continue;
    const alpha=alphaAt(k,it.i,u,v);if(alpha>=0&&alpha<PICK_ALPHA)continue;
    if(!section||g.depth>section.depth) section={source:"section",index:it.i,
      depth:g.depth,point:[g.x,g.y,zmm(it.i)]};
  }
  const body=window.__SOLID_HIT_AT__?.(clientX,clientY);
  // Match the tissue depth used by the combined renderer. A surface behind the
  // section cannot own a click through that section's opaque tissue.
  if(body && (!section || !sectionDepthSource() || body.depth>=section.depth-1e-6))return body;
  if(section)return section;
  if(!background)return null;
  const p=bodyFocusPoint()||S.pivot,depth=m[2]*p[0]+m[6]*p[1]+m[10]*p[2];
  return {source:"view-plane",depth,point:pointFromView(vx,vy,depth)};
}
function pivotAtClient(x,y){
  const hit=surfaceUnderCursor(x,y,true);
  S.pivot=hit.point.slice();need();return hit;
}
// The rotation centre used to hang off a left double-click. A left click now selects a
// layer, and a double click would select twice and step deeper, so this moved to a key.
function setPivotUnderCursor(){
  if(!lastPointer){ S.pickMsg = "move the pointer over the canvas first, then press P";
    need(); return null; }
  const h=surfaceUnderCursor(lastPointer.x,lastPointer.y,false);
  if(!h){S.pickMsg="no visible surface under the pointer";need();return null;}
  window.__SOLID_CLEAR_FOCUS__?.();focusBody(null);S.pivot=h.point.slice();
  S.pickMsg="rotation centre set on "+(h.objectId||META.planes[h.index].section_id);
  need();
  return {...h,pivot:S.pivot.slice()};
}
function clickSelect(e){
  if(window.__SOLID_CLICK_AT__?.(e.clientX,e.clientY))return;
  // Picking names a SECTION. With the sections hidden there is no section to
  // name, and selecting one anyway lifts it out of the stack and rescales it --
  // the bodies then appear to deform under a click that was aimed at them.
  if(!sectionsOn()) return;
  pickAt(e.clientX, e.clientY);
}
// A middle press starts autoscroll (Chrome) or paste (X11 Firefox) unless the
// default is cancelled, and the click that follows opens a tab on a link.
cv.addEventListener("mousedown", e => { if(e.button === 1) e.preventDefault(); });
cv.addEventListener("auxclick", e => { if(e.button === 1) e.preventDefault(); });
// Only on the canvas: a right press here is a pan handle, so the menu would be in the
// way. Everywhere else on the page the right button keeps the browser's menu.
cv.addEventListener("contextmenu", e => { e.preventDefault(); });
cv.addEventListener("pointerdown", e => {
  cv.focus();
  ++focusMotion;
  const mid = (e.button === 1), right = (e.button === 2);
  if(mid || right) e.preventDefault();
  const panMod = panModifier(e);
  const localOrbit=e.ctrlKey && !panMod && !spaceHeld && (e.button===0 || (IS_MAC && right));
  let mode = null;
  if(localOrbit){e.preventDefault();pivotAtClient(e.clientX,e.clientY);mode=e.shiftKey?"rotH":"rot";}
  else if(mid) mode = (panMod || e.ctrlKey) ? "pan" : "rot";
  else if(right) mode = "pan";
  else if(e.button === 0){
    if(panMod || spaceHeld) mode = "pan";
    else if(e.shiftKey) mode = "rotH";
    else mode = "rot";
  }
  if(!mode) return;
  drag = {x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY,
          az: S.az, el: S.el, mode: mode, mid: mid, localOrbit, moved: 0, t0: nowMs(),
          // a selection is a bare left press: no modifier, no other button
          left: e.button === 0 && !panMod && !e.ctrlKey && !e.shiftKey && !e.altKey && !spaceHeld};
  if(cv.setPointerCapture){ try { cv.setPointerCapture(e.pointerId); } catch(_){} }
});
cv.addEventListener("pointermove", e => {
  lastPointer = {x: e.clientX, y: e.clientY};
  if(!drag){ hoverReadout(e); return; }
  drag.moved = Math.max(drag.moved,
    Math.abs(e.clientX - drag.x0) + Math.abs(e.clientY - drag.y0));
  if(drag.mode === "pan"){ panByPx(e.clientX - drag.x, e.clientY - drag.y);
    drag.x = e.clientX; drag.y = e.clientY; }
  else if(drag.mode === "rot"){
    rotateByScreen((e.clientX - drag.x)*0.008, (e.clientY - drag.y)*0.008);
    drag.x = e.clientX; drag.y = e.clientY; }
  // Shift: the specimen turns left and right about the screen vertical and nothing
  // else. The vertical component of the drag is discarded, not scaled down.
  else if(drag.mode === "rotH"){
    rotateByScreen((e.clientX - drag.x)*0.008, 0);
    drag.x = e.clientX; drag.y = e.clientY; }
});
cv.addEventListener("pointerup", e => {
  if(drag){
    const isClick = drag.moved < CLICK_PX && (nowMs() - drag.t0) < CLICK_MS;
    // a press that stayed still and was short is a selection, whatever it was aiming
    // to do; a rotate that moved 4 px or took 300 ms never selects
    if(drag.left && isClick) clickSelect(e);
    if(drag.mode === "pan") clampPanSoft();
    if(drag.mid && drag.moved <= 4){
      const now = nowMs();
      if(now - midClick.t < 400 && Math.abs(e.clientX - midClick.x) < 6
         && Math.abs(e.clientY - midClick.y) < 6){ zoomToFit(); midClick.t = 0; }
      else midClick = {t: now, x: e.clientX, y: e.clientY};
    }
  }
  drag = null; need();
});
cv.addEventListener("pointercancel", () => { drag = null; });
cv.addEventListener("wheel", e => {
  e.preventDefault();
  let d = e.deltaY;
  if(e.deltaMode === 1) d *= 16; else if(e.deltaMode === 2) d *= 400;
  let f = Math.exp(-d * 0.0015 * (e.ctrlKey ? 2.5 : 1));
  f = Math.max(1/1.5, Math.min(1.5, f));
  const r = cv.getBoundingClientRect();
  const nx = ((e.clientX - r.left)/r.width)*2 - 1;
  const ny = 1 - ((e.clientY - r.top)/r.height)*2;
  zoomAt(nx, ny, f);
}, {passive: false});
// Same reason as the wheel: a passive listener cannot cancel the gesture, and the
// page would zoom AND scroll at once. Only the canvas is blocked; the flyout and the
// layer strip still scroll normally.
cv.addEventListener("touchmove", e => { e.preventDefault(); }, {passive: false});
function canvasUV(e){
  const r = cv.getBoundingClientRect();
  const nx = ((e.clientX - r.left)/r.width)*2 - 1, ny = 1 - ((e.clientY - r.top)/r.height)*2;
  const rad = radNow(), asp = r.width/r.height;
  const vx = S.cx + nx*(asp>=1?rad*asp:rad), vy = S.cy + ny*(asp>=1?rad:rad/asp);
  const R = resInfo();
  return [vx/R.canvas_mm.width + 0.5, 0.5 - vy/R.canvas_mm.height];
}
// the section's own number wherever it sits in the id: "HT891Z1-U101", "S22-27909-P1_U74"
// and "HT206B1-H2L1Us1_17" (series marker s1, section 17) all read as their U number
function uTok(sid){ const m = String(sid).match(/U(?:s\d+_)?(\d+)[^-_]*$/); return m ? "U" + m[1] : String(sid); }
function regionPointInside(poly, x, y){
  let inside = false;
  for(let i=0,j=poly.length-1; i<poly.length; j=i++){
    const a=poly[i],b=poly[j];
    if((a[1]>y)!==(b[1]>y) && x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]) inside=!inside;
  }
  return inside;
}
function regionLines(i, xu, yu){
  // every 2-D region layer that is switched on, named at this point: the id the rest
  // of the delivery uses, what it is, and how big its footprint is on this section
  const out = [];
  const layers = [];
  if(regionOn(NERVE,S.nerveReg) && NERVE && NERVE.planes) layers.push(["nerve", NERVE, null]);
  if(S.tlsReg && TLSR && TLSR.planes)
    layers.push(["TLS", TLSR, g => (g.kind === "3d" && S.tls3d) || (g.kind === "2d" && S.tls2d)]);
  if(S.ductReg && DUCTR && DUCTR.planes) layers.push(["duct", DUCTR, null]);
  if(regionOn(GLANDR,S.glandReg) && GLANDR && GLANDR.bases.indexOf(S.basis)>=0) layers.push(["tumor gland", GLANDR, glandKindOn]);
  for(const [name, LY, keep] of layers){
    const pl = LY.planes.find(p => p.index === i);
    if(!pl || !pl.regions) continue;
    const hit = pl.regions.filter(g => figObjOn(g.id) && (!keep || keep(g)) && g.bbox_um
      && xu >= g.bbox_um[0] && xu <= g.bbox_um[2] && yu >= g.bbox_um[1] && yu <= g.bbox_um[3]
      && (!g.polygons_um || g.polygons_um.some((poly,pi) => regionPointInside(poly, xu, yu)
        && !(g.polygon_holes_um?.[pi]||[]).some(h=>regionPointInside(h,xu,yu)))));
    if(!hit.length) continue;
    // nested bboxes: the region whose centroid is nearest wins
    hit.sort((a, b) => Math.hypot(a.cx_um - xu, a.cy_um - yu) - Math.hypot(b.cx_um - xu, b.cy_um - yu));
    const g = hit[0];
    const w = Math.round(g.bbox_um[2] - g.bbox_um[0]), h = Math.round(g.bbox_um[3] - g.bbox_um[1]);
    const what = g.kind && g.kind !== "duct" ? g.kind : name;
    // the object's own 3-D lines, exactly as the 3-D panel words them, when its body
    // is loaded; then this section's own footprint. One object, one wording.
    const body = window.__SOLID_TIP__ ? window.__SOLID_TIP__(g.id) : null;
    if(body && body.length){
      for(const t of body) if(t) out.push(t);
      out.push("this section: " + w + "x" + h + " um");
    } else {
      out.push(g.id + " . " + what + " . " + w + "x" + h + " um");
    }
    if(name==="nerve" && Number.isFinite(g.area_um2)){
      out.push("2D predicted region: "+Math.round(g.area_um2)+" µm²");
      if(g.boundary_uncertain)out.push("Uncertain boundary near tissue edge");
      if(g.candidate_status)out.push(g.candidate_status);
    }
    if(g.gland_id){
      if(g.parent_3d_id)out.push("this profile belongs to " + g.parent_3d_id);
      if(!body || !body.length) out.push(g.gland_id + " · adjacent to " + g.nerve_id);
      out.push("this section: " + g.area_um2 + " µm²" + (g.n_tumor_cells == null ? "" : " · " + g.n_tumor_cells + " tumor-predicted cells"));
      if(g.annotation_source === "observed_HE_visual_annotation"){
        out.push("Visually outlined H&E candidate; malignancy unclassified");
        if(g.boundary_uncertain)out.push("Observed contour boundary is uncertain");
        if(g.roi_clipped)out.push("Profile extends beyond the annotated field");
      }
      out.push(g.dist_nerve_um == null ? "nerve not present on this section" : "distance to nerve " + g.dist_nerve_um + " µm");
    }
  }
  return out;
}
let hoverT = 0;
function hoverReadout(e){
  const el = document.getElementById("hover");
  const R = resInfo();
  const now = nowMs();
  if(now - hoverT < 40) return;                 // the pick walks every plane: not per pixel
  hoverT = now;
  // the plane UNDER THE CURSOR, in any view: tilted, a screen point still lands on a
  // definite layer, and that is the layer whose regions the reader is pointing at
  const p = clientToView(e.clientX, e.clientY);
  const P = pickList(p[0], p[1]);
  const hit = (!P.edgeOn && P.hits.length) ? P.hits[0] : null;
  const lines = [];
  // the bodies are drawn over the same canvas: when the sections are hidden, or when a
  // body sits in front of them, what the cursor is on is the body, and it is named the
  // way the 3-D panel names it
  const dpr = DPR_ACTIVE || window.devicePixelRatio || 1;
  const rr = cv.getBoundingClientRect();
  const bodyAt = window.__SOLID_TIP_AT__
    ? window.__SOLID_TIP_AT__((e.clientX - rr.left) * dpr, (e.clientY - rr.top) * dpr) : null;
  if(bodyAt && bodyAt.length){
    for(const t of bodyAt) if(t) lines.push(t);
  }
  const i = (sectionsOn() && hit) ? hit.i : null;
  if(i !== null && META.planes[i]){
    // XY are the aligned canvas coordinates of this ray/plane intersection;
    // Z is physical section depth, without display spreading or exaggeration.
    const x = hit.u*R.canvas_mm.width*1000, y = hit.v*R.canvas_mm.height*1000;
    lines.push("plane " + uTok(META.planes[i].section_id));
    lines.push("x=" + fmt(x,1) + " µm  y=" + fmt(y,1) + " µm  z=" + fmt(Z[i],1) + " µm");
    for(const t of regionLines(i, x, y))
      lines.push(t);
  }
  if(!lines.length){ el.style.display = "none"; return; }
  el.style.whiteSpace = "pre-line";
  el.textContent = [...new Set(lines)].join("\n");
  const r = cv.getBoundingClientRect();
  el.style.left = (e.clientX - r.left + 12) + "px";
  el.style.top = (e.clientY - r.top + 12) + "px";
  el.style.display = "block";
}
cv.addEventListener("pointerleave", () => {
  document.getElementById("hover").style.display = "none"; });

// --------------------------------------------------------------- controls
function seg(id, get, vals, set){
  const box = document.getElementById(id); if(!box) return;
  vals.forEach(([v, lab]) => {
    const b = document.createElement("button");
    b.textContent = lab; b.dataset.v = String(v);
    b.setAttribute("aria-pressed", String(get() === v));
    b.onclick = () => { set(v); press(id, get()); };
    box.appendChild(b);
  });
}
// This panel exposes exactly: basis and resolution (data), colour, depth stretch,
// opacity, the layer strip with previous / next / gap jump / play,
// show-only-this-layer, and full screen. Nothing else.
seg("basisSeg", () => S.basis, BASES.map(b => [b, META.bases[b].short]), setBasis);
// Base style: pick one of three. Short labels, because the control's job is to show
// that they are alternatives; what each one IS goes in the note under it.
seg("colourSeg", () => S.colour,
    // Translucent is gone: on this page it is indistinguishable from Gray,
    // and a control whose two settings look the same is noise.
    [["grey", "Gray"], ["native", "Native"]],
    setColour);
// An extra switch next to the colour one, not a replacement for it. The
// control is built for EVERY sample so the panel does not change shape when the
// reader switches sample; where the swapped arm was never built the segment is
// simply disabled and says so. A card that disappears with the data makes every
// other card move, and the reader has to find them again.
{
  seg("heSwapSeg", () => (S.heSwap ? "on" : "off"),
      [["off", BASE_IS_HE ? "H&E" : "All modalities"],
       ["on", BASE_IS_HE ? "All modalities" : "H&E"]],
      v => { if(swapAvailable()) setHeSwap(v === "on"); });
  const box = document.getElementById("heSwapBox");
  if(box && !swapAvailable()) box.setAttribute("data-empty", "1");
}
if(swapAvailable()){
  // the segment itself is built above, for every sample; only the caveat text
  // below depends on the swapped arm existing. Building it twice put two pairs
  // of buttons in the card.
  // the one thing the info panel has to say about it, once, in the panel that
  // carries every other caveat on this page
  const ab = document.getElementById("heSwapAbout");
  if(ab){
    ab.style.display = "block";
    ab.textContent =
      "Display › Sections: " + SWAPSET.size + " of the " + N + " sections were "
      + "imaged twice - CODEX or Xenium first, then re-stained and re-scanned in H&E on "
      + "the SAME physical section, so at the same z. H&E shows the H&E of every "
      + "section; All modalities shows the other scan on those " + SWAPSET.size + " "
      + "layers. Nothing else changes: the geometry is the published reconstruction's, "
      + "every plane keeps the position it has under either setting, and NO per-section "
      + "registration was fitted for the second arm - its placement inherits what the "
      + "reconstruction already had. Nothing is interpolated along z either: all " + N
      + " are measured sections at their real z.";
  }
}else{
  const box = document.getElementById("heSwapBox");
  if(box) box.style.display = "none";
}
// The class list. One row per predicted class, ordered as it is drawn, so the
// row order and the stacking order are the same thing. The swatch is the only place
// a class colour is shown outside the picture.
function pressCellRows(){
  for(const cls of CELLDRAW){
    const r = document.getElementById("crow-" + cls);
    if(r) r.setAttribute("aria-pressed", String(S.cellsOn.indexOf(cls) >= 0));
  }
}
function updateCellsNote(){}
{
  const tt = document.getElementById("tlsToggle");
  const pillsFollow = () => { for(const id of ["tls3dToggle", "tls2dToggle"]){
    const k = document.getElementById(id); if(k) k.disabled = !(TLSR && tt && tt.checked); } };
  if(tt){ tt.disabled = !TLSR; tt.addEventListener("change", () => { setTlsReg(tt.checked); pillsFollow(); }); }
  for(const [id, kind] of [["tls3dToggle", "3d"], ["tls2dToggle", "2d"]]){
    const k = document.getElementById(id);
    if(k){ k.addEventListener("change", () => setTlsKind(kind, k.checked));
           k.addEventListener("click", e => e.stopPropagation()); }
  }
  pillsFollow();
  const tmT = document.getElementById("tumorToggle");
  if(tmT){ tmT.disabled = !TUMR; if(tmT.closest("label")) tmT.closest("label").hidden = !TUMR;
           tmT.addEventListener("change", () => setTumorReg(tmT.checked)); }
  const nlT = document.getElementById("nlineToggle");
  if(nlT){ nlT.disabled = !NLINE; if(nlT.closest("label")) nlT.closest("label").hidden = !NLINE;
           nlT.addEventListener("change", () => setNlineReg(nlT.checked));
           nlT.addEventListener("click", e => e.stopPropagation()); }
  const glandT = document.getElementById("glandToggle");
  if(glandT){ glandT.disabled = !GLANDR; glandT.closest(".showbar").hidden = !GLANDR;
    glandT.addEventListener("change", () => setGlandReg(glandT.checked)); }
  for(const [id,kind] of [["gland3dToggle","3d"],["gland2dToggle","2d"]]){
    const p=document.getElementById(id);if(p){p.disabled=!(GLANDR && glandT.checked);p.addEventListener("change",()=>setGlandKind(kind,p.checked));p.addEventListener("click",e=>e.stopPropagation());}
  }
  const ductT = document.getElementById("ductToggle");
  if(ductT){ ductT.disabled = !DUCTR; if(ductT.closest("label")) ductT.closest("label").hidden = !DUCTR;
             ductT.addEventListener("change", () => setDuctReg(ductT.checked)); }
  const nt = document.getElementById("nerveToggle");
  if(nt){
    nt.addEventListener("change", () => setNerveReg(nt.checked));
    if(!NERVE){ nt.disabled = true; nt.parentElement.style.opacity = 0.45;
      nt.parentElement.title = "no nerve-region layer built for this sample"; }
  }
  // A row that carries sub-option pills is a <div>, because a <label> cannot hold
  // another <label>; without this it is the only row where the click has to land on
  // the little box itself. Clicking anywhere on the row toggles the row's own
  // switch, exactly like the plain rows; the pills keep their own clicks.
  document.querySelectorAll(".showbar.pillrow").forEach(rowEl => {
    const box = rowEl.querySelector(":scope > input[type=checkbox]");
    if(!box) return;
    // bound once per row: the 2-D card and the 3-D panel both run this over every
    // pill row of the page, and two handlers on one row flip the box twice (no change)
    if(rowEl.dataset.rowclick) return;
    rowEl.dataset.rowclick = "1";
    rowEl.addEventListener("click", e => {
      if(e.target === box || (e.target.closest && e.target.closest(".kind"))) return;
      if(box.matches(":disabled")) return;
      box.checked = !box.checked;
      box.dispatchEvent(new Event("change", {bubbles: true}));
    });
  });
  const d3 = document.getElementById("den3dToggle");
  if(d3){
    d3.addEventListener("change", () => setDen3d(d3.checked));
    if(!DEN3){ d3.disabled = true; d3.parentElement.style.opacity = 0.45;
      d3.parentElement.title = "no denoised celltype layer built for this sample"; }
  }
}
{
  const tg = document.getElementById("cellsToggle");
  if(tg && !cellsBuilt()){
    tg.disabled = true;
    tg.title = "cell prediction layer is still being built for this sample";
  }
  if(tg && cellsBuilt()){
    tg.addEventListener("change", () => setCellsOverlay(tg.checked));
    tg.title = "Draw the predicted cell classes on top of whichever base style is "
             + "selected. It is not one of the base styles.";
  }
  const sg = document.getElementById("cellsAllToggle");
  if(sg){
    sg.setAttribute("aria-pressed", String(S.cellsAll));
    sg.disabled = !S.cellsOverlay;
    sg.onclick = () => setCellsAll(!S.cellsAll);
    sg.title = "On: the overlay stays at full strength on every section, so a class "
             + "can be followed through the depth of the block. Off: it follows the "
             + "layer selection, and only the chosen layer stands out.";
  }
  // one builder for every class list on this page. The rows differ only in where
  // the names, colours, counts and the on/off state come from; writing a second
  // builder is how one list ends up with a different row than the other.
  buildClassList(document.getElementById("cellList"), {
    names: CELLUI,
    colour: c => CELLINFO[c].colour,
    setColour: setCellColour,
    count: c => CELLINFO[c].n_cells,
    isOn: c => S.cellsOn.indexOf(c) >= 0,
    toggle: c => toggleCellClass(c),
    enabled: () => S.cellsOverlay,
    title: c => c + "  " + CELLINFO[c].colour + "  " + CELLINFO[c].n_cells
                + " predicted cells over " + CELLS.with.length + " planes",
    id: c => "crow-" + c});
  updateCellsNote();
  const ab = document.getElementById("cellsAbout");
  if(ab){
    ab.style.display = "block";
    if(cellsBuilt()) ab.textContent =
      "Display › Overlay › Cell prediction - an OVERLAY, not a base style: it draws on "
      + "top of Gray, Native or Translucent, and switching it changes nothing about the "
      + "sections underneath. "
      + "The pan-cancer Cell classifier (single fold) was run on the H&E of all "
      + CELLS.with.length + " sections and its cells were carried onto this canvas. "
      + "Placement was checked per plane against each plane's own tissue mask: worst "
      + (100 * CELLS.checks.in_mask_min).toFixed(1)
      + "% of cells on tissue (" + CELLS.checks.worst_section + "), median "
      + (100 * CELLS.checks.in_mask_median).toFixed(1) + "%, against "
      + (100 * CELLS.checks.control_far_median).toFixed(0) + "% when the same cells "
      + "are placed with another section's pose. Most planes are carried by the "
      + "reconstruction's own chain, with nothing re-registered - the same edge set, "
      + "tree, anchor, per-section matrix and crop the planes were built with. The "
      + CELLS.n_swap + " sections whose H&E was scanned after the CODEX run inherit "
      + "that placement's accuracy (4.9-38.1 µm) and are the worst planes here. The "
      + CELLS.n_xenium + " Xenium sections are the one place something WAS fitted: "
      + "their frame is the Xenium DAPI image, and the H&E reaches it through a "
      + "similarity fitted on nuclear centroids, H&E nucleus to Xenium nucleus at a "
      + "median 0.96-1.59 µm against a 4.5-5.0 µm chance rate. "
      + "The Opacity slider is the BASE's control and does not reach this "
      + "overlay, so dimming the sections to look past them cannot dim what you are "
      + "looking for."
      + " Colour is class and only class; brightness is that class's local density on "
      + "its OWN ramp, so it compares planes and never classes.";
  }
}
{
  const vs = document.getElementById("viewSeg");
  if(vs){
    for(const n of [1,2,3,4,5,6,7]){
      const b = document.createElement("button");
      b.textContent = n + " " + STDVIEW[n][0];
      b.onclick = () => stdView(n);
      vs.appendChild(b);
    }
    const f = document.createElement("button");
    f.textContent = "F fit"; f.onclick = () => zoomToFit();
    vs.appendChild(f);
  }
}
if(SRC.mode === "fetch")
  seg("resSeg", () => S.res, RES_SHOWN.map(r => [r, r + " um/px"]),
      v => { if(RESES.indexOf(v) < 0) return;      // not built for this sample
             S.res = v; if(S.colour === "native") loadRGBSet(S.basis, v, need);
 loadSet(S.basis, v, need);
             need(); });
  {
    const sg = document.getElementById("resSeg");
    if(sg) for(const b of sg.children){
      if(RESES.indexOf(b.dataset.v) < 0){
        b.disabled = true;
        b.title = b.dataset.v + " um/px was not built for this sample";
        b.style.opacity = ".42";
      }
    }
  }

function rng(id, get, set, fmtf){
  const el = document.getElementById(id); if(!el) return;
  el.value = get();
  el.oninput = () => { set(+el.value);
    const o = document.getElementById(id + "Val"); if(o) o.textContent = fmtf(+el.value);
    need(); };
  const o = document.getElementById(id + "Val"); if(o) o.textContent = fmtf(get());
}
// The base opacity is exposed as a control. The two selection values stay
// constants, because what the user adjusts is how the stack reads, not how
// selection behaves.
rng("opacitySl", () => Math.round(SEL.OPACITY_BASE*100), v => {
  SEL.OPACITY_BASE = v/100;
  // the layouts hold this value in an array, so rebuild the resting state or the
  // slider would do nothing until the next selection change
  ANIM.to = layoutFor(S.sel, S.solo); ANIM.from = ANIM.to; ANIM.t0 = 0;
  flash("opacity " + v + "%"); }, v => "");
// the overlay's own control: it scales the drawn cell, not the base and not the
// opacity. Rare classes are the reason it exists.
(function(){
  // "show" on the Modality card: off puts the base back on the arm the page opens
  // with and greys the control, so the card reads as a layer that is switched off
  // rather than one whose setting is being ignored.
  const c = document.getElementById("modShow"), box = document.getElementById("heSwapBox");
  if(!c) return;
  c.addEventListener("change", () => {
    if(box) box.setAttribute("data-on", c.checked ? "1" : "0");
    S.baseOn = c.checked;
    if(!c.checked && S.heSwap) setHeSwap(false);
    need();
  });
})();
(function(){
  // the card is present for every sample; it only carries content where a Xenium
  // run exists, and says so rather than vanishing
  const box = document.getElementById("xgtBox");
  if(!box) return;
  if(!XGT){
    box.hidden = true;                  // no Xenium run: no card
    return;
  }
  box.hidden = false;
  if(!XGT.colours0) XGT.colours0 = XGT.colours.slice();
  XGT.classes.forEach((nm, i) => {
    const v = savedColour(nm); if(v) XGT.colours[i] = v; });
  buildClassList(box.querySelector("#xgtList"), {
    names: XGT.classes,
    colour: c => XGT.colours[XGT.classes.indexOf(c)],
    setColour: (c, h) => { XGT.colours[XGT.classes.indexOf(c)] = h; storeColour(c, h);
      if(window.__SOLID_SETCOL__) window.__SOLID_SETCOL__(c, h); },
    count: c => (XGT.counts ? XGT.counts[c] : null),
    isOn: c => XS.cls.indexOf(c) >= 0,
    toggle: c => { const k = XS.cls.indexOf(c);
                   if(k >= 0) XS.cls.splice(k, 1); else XS.cls.push(c);
                   if(XS.on) loadCellTextures(S.basis, need, xgtTexSource()); },
    title: c => c + (XGT.xenium_only.indexOf(c) >= 0
                     ? "  \u2014 Xenium only" : "")});
  { const xt = box.querySelector("#xtlsToggle");
    if(!XTLS){ xt.disabled = true; xt.parentElement.style.opacity = 0.45;
               xt.parentElement.title = "no Xenium TLS layer built for this sample"; }
    else xt.addEventListener("change", e => {
      XS.tls = e.target.checked;
      if(XS.tls) loadCellTextures(S.basis, need, xtlsSource());
      need();
    }); }
  box.querySelector("#xgtShow").addEventListener("change", e => {
    XS.on = e.target.checked;
    if(XS.on && S.cellsOverlay) setCellsOverlay(false);
    if(XS.on && !XS.cls.length){            // something to see on first press
      XS.cls = XGT.classes.slice(0, 1);
      const f = box.querySelector("#xgtList .crow");
      if(f) f.setAttribute("aria-pressed", "true");
    }
    const body = box.querySelector("#xgtBody");
    if(body) body.dataset.on = XS.on ? "1" : "0";
    if(XS.on) loadCellTextures(S.basis, need, xgtTexSource());
    need();
  });
})();
rng("cellPtSl", () => Math.round(S.cellPt*100), v => {
  S.cellPt = v/100;
  ANIM.to = layoutFor(S.sel, S.solo); ANIM.from = ANIM.to; ANIM.t0 = 0;
  need(); }, v => (v/100).toFixed(1) + "x");
rng("exagSl", () => S.exag*10, v => {
  let x = v/10; if(Math.abs(x-1) < 0.3) x = 1; if(Math.abs(x-12.8) < 0.3) x = 12.8;
  S.exag = x; buildRuler(); }, v => (v/10).toFixed(1) + "x");
// "Show only this layer" is a switch that exists only while a layer is selected, and it
// sits next to the corner readout so it is obvious which layer it acts on. It is the
// visible form of a key the page already had; no new key was added for it.
function toggleSectionScope(){
  // Keep S's one-section / stack meaning in the combined display too. Clearing
  // the section preserves zi and the independent body focus for the return.
  if(sectionContext())setSelected(null);else setOnly(!S.solo);
}
function syncOnly(){
  const box = document.getElementById("onlybox"); if(!box) return;
  if(S.sel === null){                       // no selection -> the switch is not there
    if(box.firstChild) box.textContent = "";
    return;
  }
  let b = document.getElementById("onlyBtn");
  if(!b){
    b = document.createElement("button");
    b.id = "onlyBtn"; b.type = "button";
    b.addEventListener("click", () => { toggleSectionScope(); setTimeout(() => cv.focus(), 0); });
    box.appendChild(b);
  }
  const lab = META.planes[S.sel].section_id.split("-").pop();
  if(sectionContext()){
    b.textContent='show section stack (S)';b.setAttribute('aria-pressed','true');b.title='Show all sections; press S again to return to this section';return;
  }
  b.textContent = S.solo ? "show all layers (S)" : ("stack up to " + lab + " (S)");
  b.setAttribute("aria-pressed", String(S.solo));
  b.title = S.solo ? "draw all " + N + " layers again"
                   : "draw the block from the first section up to " + lab
                     + " at their real z, and hide everything above it";
}
document.getElementById("fsBtn").onclick = () => { toggleFullscreen(); backToCanvas(); };
document.getElementById("showLabels").onchange = e => {S.showLabels=e.target.checked;need();};
// arriving from another page's fullscreen: re-enter on the first gesture
if(location.hash.indexOf("fs") >= 0){
  history.replaceState(null, "", location.pathname + location.search);
  const once = e => {
    // A sample-menu click must not start a native fullscreen resize immediately
    // before navigation. Restore on a canvas gesture, not on unrelated controls.
    if(e.target !== cv) return;
    window.removeEventListener("pointerdown", once, true);
    window.removeEventListener("keydown", once, true);
    if(!fullscreenOn()) toggleFullscreen(); };
  window.addEventListener("pointerdown", once, true);
  window.addEventListener("keydown", once, true);
}

// ---- carry every control across a sample switch. The state is NOT copied field by
// field: the page snapshots what each control shows and the next page replays the
// controls through their own handlers, so every side-effect a click has still runs.
const CARRY_KEY = "viewerCarry.v1";
function carrySnapshot(){
  const snap = {inputs: {}, segs: {}, cls2d: {}, cls3d: {},
                view: {az: S.az, el: S.el, zoom: S.zoom, cx: S.cx, cy: S.cy, exag: S.exag, solo: S.solo}};
  document.querySelectorAll("input[id]").forEach(ip => {
    if(ip.type === "checkbox") snap.inputs[ip.id] = {t: "c", v: ip.checked};
    else if(ip.type === "range" || ip.type === "number") snap.inputs[ip.id] = {t: "r", v: ip.value};
  });
  document.querySelectorAll(".seg[id]").forEach(box => {
    const on = box.querySelector('button[aria-pressed="true"]');
    if(on) snap.segs[box.id] = on.dataset.v !== undefined ? {v: on.dataset.v} : {id: on.id};
  });
  document.querySelectorAll("#cellList .crow[id^='crow-']").forEach(b => {
    snap.cls2d[b.id.slice(5)] = b.getAttribute("aria-pressed") === "true"; });
  document.querySelectorAll("#objcls .crow[id^='orow-']").forEach(b => {
    snap.cls3d[b.id.slice(5)] = b.getAttribute("aria-pressed") === "true"; });
  return snap;
}
function carrySave(){ try{ localStorage.setItem(CARRY_KEY, JSON.stringify(carrySnapshot())); }catch(e){} }
function carryApply(snap){
  // Migrate the two former dimension switches into the one current control.
  snap={...snap,inputs:{...(snap.inputs||{})}};
  if(!snap.inputs.showLabels){const old=snap.view?.solo?snap.inputs.secLabels:snap.inputs.objLabels;if(old)snap.inputs.showLabels=old;}
  delete snap.inputs.secLabels;delete snap.inputs.objLabels;
  const fire = (el, ev) => el.dispatchEvent(new Event(ev, {bubbles: true}));
  const setCheck = (ip, v) => { if(ip && ip.type === "checkbox" && ip.checked !== !!v && !ip.disabled){ ip.checked = !!v; fire(ip, "change"); return true; } return false; };
  const setRange = (ip, v) => { if(ip && String(ip.value) !== String(v)){ ip.value = v; fire(ip, "input"); fire(ip, "change"); } };
  // segmented controls first (base style, basis, resolution, view), then switches
  Object.entries(snap.segs || {}).forEach(([id, w]) => {
    const box = document.getElementById(id); if(!box) return;
    const btn = w.v !== undefined ? box.querySelector('button[data-v="' + w.v + '"]') : document.getElementById(w.id);
    if(btn && btn.getAttribute("aria-pressed") !== "true") btn.click();
  });
  const later = {};                       // controls that do not exist yet (3-D panel builds lazily)
  Object.entries(snap.inputs || {}).forEach(([id, w]) => {
    const ip = document.getElementById(id);
    if(!ip || ip.disabled){ later[id] = w; return; }
    if(w.t === "c") setCheck(ip, w.v); else setRange(ip, w.v);
  });
  Object.entries(snap.cls2d || {}).forEach(([c, on]) => {
    const row = document.getElementById("crow-" + c); const ip = row && row.querySelector("input[type=checkbox]");
    setCheck(ip, on);
  });
  if(snap.view){
    const v = snap.view;
    if(typeof v.az === "number") S.az = v.az; if(typeof v.el === "number") S.el = v.el;
    if(typeof v.zoom === "number") S.zoom = v.zoom;
    if(typeof v.cx === "number") S.cx = v.cx; if(typeof v.cy === "number") S.cy = v.cy;
    if(typeof v.exag === "number"){ const sl = document.getElementById("exagSl"); if(sl) setRange(sl, Math.round(v.exag * 10)); }
    need();
  }
  // the 3-D panel: its class rows and its own controls appear once the payload is in
  let tries = 0, classesRestored = false, segmentRestored = false;
  function restoreSolid(){
    tries++;
    Object.keys(later).forEach(id => { const ip = document.getElementById(id); if(!ip || ip.disabled) return;
      const w = later[id]; if(w.t === "c") setCheck(ip, w.v); else setRange(ip, w.v); delete later[id]; });
    const objShow = snap.inputs && snap.inputs.objShow;
    const rows = document.querySelectorAll("#objcls .crow[id^='orow-']");
    if(rows.length && !classesRestored){ if(snap.cls3d) rows.forEach(b => {
      const want = !!snap.cls3d[b.id.slice(5)];
      if((b.getAttribute("aria-pressed") === "true") !== want) b.click(); }); classesRestored = true; }
    if(!segmentRestored && window.__SOLID_SYNC__){
      if(snap.segs && snap.segs.objWhat && snap.segs.objWhat.id){ const b = document.getElementById(snap.segs.objWhat.id); if(b && b.getAttribute("aria-pressed") !== "true") b.click(); }
      segmentRestored = true;
    }
    if(window.__SOLID_SYNC__){ window.__SOLID_SYNC__(); need(); }
    if((rows.length || !objShow || !objShow.v) && Object.keys(later).length === 0){ clearInterval(t); window.removeEventListener("viewer-solid-ready", restoreSolid); }
    if(tries > 60) clearInterval(t);
  }
  const t = setInterval(restoreSolid, 300);
  window.addEventListener("viewer-solid-ready", restoreSolid, {once:true});
}
try{
  const raw = localStorage.getItem(CARRY_KEY);
  if(raw){ localStorage.removeItem(CARRY_KEY); setTimeout(() => carryApply(JSON.parse(raw)), 0); }
}catch(e){ console.error("carry-over", e); }
window.addEventListener("beforeunload", carrySave);
window.__VIEWER_DISPOSE__ = function(){
  if(window.__VIEWER_DISPOSING__) return;
  window.__VIEWER_DISPOSING__ = true;
  PLANE_QUEUE.length = 0;
  for(const job of TILE_JOBS.values()) if(job.controller) job.controller.abort();
  TILE_JOBS.clear(); TILE_WANTED.clear();
  // Release the outgoing stack before the next sample allocates GPU resources.
  // The browser may otherwise retain the old document during navigation.
  const ext=gl && gl.getExtension("WEBGL_lose_context");
  if(ext) ext.loseContext();
};
window.addEventListener("pagehide", window.__VIEWER_DISPOSE__);
window.addEventListener("pageshow", e=>{if(e.persisted && window.__VIEWER_DISPOSING__) location.reload();});
function shotName(ext){
  const ids=FIG.on ? (sectionsOn()?secList():[]) : (S.sel===null?[]:[S.sel]);
  const sid=ids.length?"_"+ids.map(i=>META.planes[i].section_id).join("_"):"";
  const base = (document.title.split(" ")[0] || "view");
  return base + sid + "." + ext;
}
function sectionReadout(){
  if(FIG.on && !sectionsOn())return "3D figure view";
  return (FIG.on?secList():[S.zi]).map(i=>{
    const p=META.planes[i];
    return p.section_id.split("-").pop()+" · z="+Math.round(p.z_um)+" µm · "+modLabel(i).toUpperCase();
  }).join("  /  ");
}
function rawGrab(){
  draw();                                   // same task: the buffer is still there
  const out = document.createElement("canvas");
  out.width = cv.width; out.height = cv.height;
  out.getContext("2d").drawImage(cv, 0, 0);
  return out;
}
// The same frame rendered on black and again on white gives the coverage directly.
// On black a pixel IS the premultiplied foreground; on white it is that plus the
// uncovered share of the background, so the difference between the two is exactly
// what the background contributed. Antialiased edges come out right for free.
function matteFrom(onBlack, onWhite){
  const W = onBlack.width, H = onBlack.height;
  const gb = onBlack.getContext("2d");
  const B = gb.getImageData(0, 0, W, H);
  const E = onWhite.getContext("2d").getImageData(0, 0, W, H).data;
  const d = B.data;
  for(let i = 0; i < d.length; i += 4){
    const a = Math.max(0, Math.min(255, Math.round(
      255 - ((E[i] - d[i]) + (E[i+1] - d[i+1]) + (E[i+2] - d[i+2])) / 3)));
    if(a === 0){ d[i] = d[i+1] = d[i+2] = d[i+3] = 0; continue; }
    for(let k = 0; k < 3; k++)
      d[i+k] = Math.max(0, Math.min(255, Math.round(d[i+k] * 255 / a)));
    d[i+3] = a;
  }
  gb.putImageData(B, 0, 0);
  return onBlack;
}
function annotateView(out){
  const c = out.getContext("2d");
  // burn in the annotations, so the file carries its own scale. Device pixels
  // here, since the capture IS the backing store.
  const dpr = DPR_ACTIVE || 1;
  const upp = umPerDevicePx();
  let um = SB_STEPS[SB_STEPS.length - 1];
  for(const st of SB_STEPS){ if(st / upp >= 60 * dpr){ um = st; break; } }
  const bw = um / upp;
  const x1 = out.width - 14 * dpr, x0 = x1 - bw, yb = out.height - 20 * dpr;
  const txt = t => { c.strokeText(t.t, t.x, t.y); c.fillText(t.t, t.x, t.y); };
  c.lineJoin = "round";
  c.strokeStyle = "rgba(255,255,255,.85)"; c.lineWidth = 4 * dpr;
  c.beginPath(); c.moveTo(x0, yb); c.lineTo(x1, yb); c.stroke();
  c.strokeStyle = "#111"; c.lineWidth = 2 * dpr;
  c.beginPath(); c.moveTo(x0, yb); c.lineTo(x1, yb); c.stroke();
  c.font = "600 " + (12 * dpr) + "px system-ui, sans-serif";
  c.fillStyle = "#111"; c.strokeStyle = "rgba(255,255,255,.85)";
  c.lineWidth = 3 * dpr; c.textAlign = "center";
  txt({t: um >= 1000 ? (um / 1000) + " mm" : um + " \u00b5m",
       x: (x0 + x1) / 2, y: yb - 6 * dpr});
  c.textAlign = "left";
  const label=sectionReadout();
  if(label)txt({t:label,x:14*dpr,y:out.height-14*dpr});
  return out;
}
function grabView(){ return annotateView(rawGrab()); }
function captureDpr(){
  const base=Math.min(2,window.devicePixelRatio||1),longSide=Math.max(cv.clientWidth,cv.clientHeight,1);
  const lim=Math.min(SHOT_MAX_PX,gl?gl.getParameter(gl.MAX_RENDERBUFFER_SIZE):SHOT_MAX_PX);
  return Math.max(base,Math.min(base*Math.max(1,SHOT.scale),lim/longSide));
}
// The region SVGs sit over WebGL on screen. Paint the SAME projected paths into
// the saved canvas, at its pixel scale, so fills/contours cannot disappear on export.
function captureRegions(out){
  if(FIG.sheet)return;
  paintContours(GLANDR,S.glandReg,"glandContours");
  if(NERVE?.vector_contours)paintContours(NERVE,S.nerveReg,"nerveContours");
  paintRegionLabels();
  const ctx=out.getContext("2d");ctx.save();
  ctx.scale(out.width/cv.clientWidth,out.height/cv.clientHeight);
  ctx.lineJoin="round";
  for(const svg of Object.values(contourSvgs)){
    if(svg.style.display==="none")continue;
    for(const el of svg.querySelectorAll("polygon,polyline,path[data-rings]")){
      const rings=el.dataset.rings?JSON.parse(el.dataset.rings):[Array.from(el.points,p=>[p.x,p.y])];
      ctx.beginPath();for(const pts of rings){if(!pts.length)continue;for(let i=0;i<pts.length;i++){
        const p=pts[i];if(i)ctx.lineTo(p[0],p[1]);else ctx.moveTo(p[0],p[1]);
      }if(el.localName!=="polyline")ctx.closePath();}
      const fill=el.getAttribute("fill");if(fill!=="none"){ctx.fillStyle=fill;ctx.fill("evenodd");}
      ctx.setLineDash((el.getAttribute("stroke-dasharray")||"").split(/[ ,]+/).filter(Boolean).map(Number));
      ctx.strokeStyle=el.getAttribute("stroke");ctx.lineWidth=+el.getAttribute("stroke-width");ctx.stroke();ctx.setLineDash([]);
    }
  }
  for(const el of regionLabelSvg?.children || []){
    ctx.strokeStyle=el.getAttribute("stroke");ctx.lineWidth=+el.getAttribute("stroke-width");
    if(el.tagName==="line"){
      ctx.beginPath();ctx.moveTo(+el.getAttribute("x1"),+el.getAttribute("y1"));
      ctx.lineTo(+el.getAttribute("x2"),+el.getAttribute("y2"));ctx.stroke();
    }else if(el.tagName==="text"){
      ctx.font="bold 14px system-ui";ctx.textAlign="center";ctx.fillStyle=el.getAttribute("fill");
      const x=+el.getAttribute("x"),y=+el.getAttribute("y");
      ctx.strokeText(el.textContent,x,y);ctx.fillText(el.textContent,x,y);
    }
  }
  ctx.restore();
}
// the saved image, at the chosen pixel scale and with the chosen background. The
// canvas is re-rendered larger for the capture and put back straight after, so the
// view on screen is untouched.
function captureView(opt){
  const withNotes = !(opt && opt.annotate === false);
  const keepDpr = DPR_ACTIVE, keepLow = lowDpr;
  const want = captureDpr();
  let out;
  try{
    lowDpr = false; DPR_ACTIVE = want;
    if(SHOT.alpha){
      SHOT_BG = [0, 0, 0]; const onBlack = rawGrab();
      SHOT_BG = [1, 1, 1]; const onWhite = rawGrab();
      SHOT_BG = null;
      out = matteFrom(onBlack, onWhite);
    }else{
      out = rawGrab();
    }
    captureRegions(out);
    if(withNotes) annotateView(out);  // annotations sized for the capture, not the screen
  } finally {
    SHOT_BG = null; DPR_ACTIVE = keepDpr; lowDpr = keepLow;
    setCanvasSize(); need();
  }
  return out;
}
function exportKey(){
  // UI changes remain possible while loading; a changed picture cancels instead
  // of silently saving a different section, angle, scale or selection.
  return JSON.stringify([window.__SAMPLE_ID__,S.basis,S.heSwap,S.colour,S.zoom,S.cx,S.cy,
    S.exag,Array.from(viewM()),S.sel,S.solo,S.hidden,FIG,SHOT,
    cv.clientWidth,cv.clientHeight,window.devicePixelRatio,
    window.__SOLID_SELECTION__?.(),Array.from(document.querySelectorAll("#panel input,#panel select,#showLabels"))
      .filter(e=>e.id!=="shotCancel").map(e=>[e.id,e.type==="checkbox"?e.checked:e.value])]);
}
function exportStatus(text){
  const el=document.getElementById("shotStatus");if(el)el.textContent=text;
}
function exportControls(busy){
  for(const id of ["shotPng","shotJpg","shotPdf","figSheet"]){
    const e=document.getElementById(id);if(e)e.disabled=busy;
  }
  document.getElementById("shotCancel").hidden=!busy;
  syncSheetControl();
}
function exportReadiness(){
  const result={pending:0,ready:tileStats.ready,wanted:tileStats.wanted,sections:tileStats.sections.map(s=>({...s}))};
  if(!sectionsOn())return result;
  const tex=TEX[setKey(S.basis,S.res)],rgb=rgbSet(),active=new Set(secList());
  for(const it of drawList()){
    const i=it.i,c=texFor(tex,rgb,i),native=S.colour==="native" && modsNow()[i]==="he";
    if(!c.t || (native && !c.rgb)){result.pending++;continue;}
    if(!active.has(i) || !visibleUV(i))continue;
    const dir=tileDirFor(i,native);
    if(!dir){if(native)throw Error("This section has no published H&E zoom tiles. No image saved.");continue;}
    const idx=tileIndex(S.basis,META.planes[i].section_id,dir);
    if(idx===null){result.pending++;continue;}
    const ladder=native?idx?.rgb_levels:idx?.levels;
    if(!ladder?.length)throw Error("Zoom tiles unavailable for "+META.planes[i].section_id+". No image saved; retry when available.");
    if(!tileStats.sections.some(s=>s.index===i))result.pending++;
  }
  for(const [key,rank] of TILE_WANTED)if(rank<1000000){
    if(TTEX.get(key)===false)throw Error("A required zoom tile failed to load. No image saved; click export to retry.");
    if(!TTEX.get(key))result.pending++;
  }
  if(animating())result.pending++;
  return result;
}
async function readyExport(action,tiles=true){
  if(EXPORT_JOB)return {ok:false,busy:true};
  const job={dpr:captureDpr(),key:exportKey(),cancelled:false};EXPORT_JOB=job;
  exportControls(true);exportStatus("Preparing image…");
  // A new explicit export retries previously failed tiles/indices, without
  // disturbing successful cache entries or changing their source build stamps.
  for(const [k,v] of TTEX)if(v===false)TTEX.delete(k);
  for(const k of Object.keys(TIDX))if(TIDX[k]===false)delete TIDX[k];
  const start=performance.now();
  try{
    let state={sections:[],wanted:0,ready:0};
    while(true){
      if(job.cancelled || window.__VIEWER_DISPOSING__)throw Error("Export cancelled. No image saved.");
      if(exportKey()!==job.key)throw Error("View changed. Export cancelled; click export for the new view.");
      if(gl.isContextLost())throw Error("Graphics context lost. No image saved.");
      draw();state=tiles?exportReadiness():state;
      if(!state.pending)break;
      exportStatus("Loading export detail · "+state.ready+" / "+state.wanted+" zoom tiles…");
      if(performance.now()-start>120000)throw Error("Zoom tiles are taking too long. No image saved; click export to retry.");
      await new Promise(resolve=>setTimeout(resolve,120));
    }
    exportStatus("Saving image…");
    const value=await action();
    exportStatus("Image saved"+(state.wanted?" · "+state.wanted+" zoom tiles ready":""));
    return {ok:true,tiles:state,value};
  }catch(e){exportStatus(e.message);return {ok:false,error:e.message};}
  finally{EXPORT_JOB=null;exportControls(false);need();}
}
function saveBlob(blob, name){
  if(!blob)throw Error("The browser could not encode this image. Try a smaller saved image size.");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
}
// A single-image PDF written by hand: one page the size of the capture, the JPEG
// itself embedded as a DCTDecode XObject. No library -- the page is self-contained.
function jpegToPdf(jpegBytes, wPx, hPx){
  const enc = new TextEncoder();
  const wPt = wPx * 0.75, hPt = hPx * 0.75;   // 96 dpi -> points
  const parts = [], xref = [];
  let off = 0;
  const push = b => { parts.push(b); off += b.length; };
  const obj = t => { xref.push(off); push(enc.encode(t)); };
  push(enc.encode("%PDF-1.4\n"));
  obj("1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n");
  obj("2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n");
  obj("3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 " + wPt.toFixed(2) + " "
      + hPt.toFixed(2) + "]/Resources<</XObject<</I 5 0 R>>>>/Contents 4 0 R>>endobj\n");
  const cs = "q " + wPt.toFixed(2) + " 0 0 " + hPt.toFixed(2) + " 0 0 cm /I Do Q";
  obj("4 0 obj<</Length " + cs.length + ">>stream\n" + cs + "\nendstream endobj\n");
  xref.push(off);
  push(enc.encode("5 0 obj<</Type/XObject/Subtype/Image/Width " + wPx + "/Height "
      + hPx + "/ColorSpace/DeviceRGB/BitsPerComponent 8/Filter/DCTDecode/Length "
      + jpegBytes.length + ">>stream\n"));
  push(jpegBytes);
  push(enc.encode("\nendstream endobj\n"));
  const xs = off;
  let x = "xref\n0 6\n0000000000 65535 f \n";
  for(const o of xref) x += String(o).padStart(10, "0") + " 00000 n \n";
  x += "trailer<</Size 6/Root 1 0 R>>\nstartxref\n" + xs + "\n%%EOF";
  push(enc.encode(x));
  return new Blob(parts, {type: "application/pdf"});
}
function shoot(fmt){
  return readyExport(async()=>{
    const keepAlpha=SHOT.alpha;let out;
    try{SHOT.alpha=keepAlpha && fmt==="png";out=captureView();}
    finally{SHOT.alpha=keepAlpha;}
    if(tileStats.ready<tileStats.wanted)throw Error("Export detail changed before capture. No image saved; click export again.");
    const blob=await new Promise(resolve=>out.toBlob(resolve,fmt==="png"?"image/png":"image/jpeg",.95));
    if(!blob)throw Error("Image encoding failed. Try a smaller saved image size.");
    saveBlob(fmt==="pdf"?jpegToPdf(new Uint8Array(await blob.arrayBuffer()),out.width,out.height):blob,shotName(fmt));
    return {width:out.width,height:out.height};
  });
}
// --------------------------------------------------------------- figure sheet
// One sheet: the nerve and gland bodies at the CURRENT camera on a clear background,
// with the section planes, the section/z ruler, both scale bars, the colour key and the
// heading drawn around them as editable vectors. The bodies arrive as a transparent PNG
// layer with a camera-facing light; everything else is vector.
// A white sheet by default; the shared transparency option removes its paper layer.
const SHEET = {ink: "#111417", muted: "#5a6a72", halo: "#ffffff"};
function sheetEsc(t){
  return String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function sheetText(x, y, t, size, opt){
  const o = opt || {};
  return '<text x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" font-size="' + size.toFixed(1)
    + '" font-family="DejaVu Sans, Arial, Helvetica, sans-serif"'
    + (o.anchor ? ' text-anchor="' + o.anchor + '"' : "")
    + (o.weight ? ' font-weight="' + o.weight + '"' : "")
    + ' fill="' + (o.fill || SHEET.ink) + '"'
    + '>' + sheetEsc(t) + "</text>";
}
function sheetLine(a, b, stroke, w, op, halo){
  const d = "M " + a[0].toFixed(1) + "," + a[1].toFixed(1) + " L " + b[0].toFixed(1)
    + "," + b[1].toFixed(1);
  const one = (col, wid, o) => '<path d="' + d + '" fill="none" stroke="' + col
    + '" stroke-width="' + wid.toFixed(2) + '" stroke-linecap="round"'
    + (o == null ? "" : ' opacity="' + o + '"') + "/>";
  // a dark line vanishes on a dark page, so the pieces that must always be legible
  // carry the same white halo the text does
  return (halo ? one(SHEET.halo, w + 2.5, 0.85) : "") + one(stroke, w, op);
}
// the same placement the bodies get: micrometres on the volume canvas, z through the
// display stretch, then the live camera. Returns pixels inside the captured image.
function sheetProject(F, xUm, yUm, zUm, W, H){
  const x = (xUm / 1000 - F.hw) * F.sc;
  const y = -(yUm / 1000 - F.hh) * F.sc;
  const z = ((zUm - (ZMIN + ZMAX) / 2) / 1000) * S.exag;
  const M = F.MVP;
  const cx = M[0]*x + M[4]*y + M[8]*z + M[12];
  const cy = M[1]*x + M[5]*y + M[9]*z + M[13];
  const cw = M[3]*x + M[7]*y + M[11]*z + M[15] || 1;
  return [(cx / cw * 0.5 + 0.5) * W, (0.5 - cy / cw * 0.5) * H];
}
let SHEET_NOTICE=null;
function sheetNoticeKey(){return [window.__SOLID_SELECTION__?.().join("|"),S.zoom,S.cx,S.cy,S.exag,...viewM()].join(",");}
function syncSheetControl(){
  const button=document.getElementById("figSheet"),note=document.getElementById("figSheetNote");
  if(!button || FIG.sheet) return;
  const info=window.__SOLID_SHEET_INFO__?.();
  button.disabled=!!EXPORT_JOB || !info;
  if(SHEET_NOTICE && SHEET_NOTICE.key!==sheetNoticeKey())SHEET_NOTICE=null;
  const text=SHEET_NOTICE?.text || (info ? info.selected+" + "+info.glands.filter(id=>!info.selection.includes(id)).length+" nearby glands · "+info.nearby_um+" µm"
                  : "Select a nerve or gland in the 3D view.");
  if(note && note.textContent!==text) note.textContent=text;
}
function figureSheet(opt){
  const info=window.__SOLID_SHEET_INFO__?.();
  if(!info){syncSheetControl();return;}
  const keepAlpha=SHOT.alpha,keepSheet=FIG.sheet,F=__VIEWER__.cameraFrame();
  const camera={orientation:Array.from(viewM()),zoom:S.zoom,pan:[S.cx,S.cy],depthStretch:S.exag,
                viewportCss:[cv.clientWidth,cv.clientHeight],MVP:Array.from(F.MVP),scale:F.sc};
  const sheetIds=Object.fromEntries(info.ids.map(id=>[id,true]));
  let cap;
  try{
    FIG.sheet=true;SHOT.alpha=true;
    window.__SOLID_FIG__({sheet:true,sheetIds:sheetIds});
    cap=captureView({annotate:false});
  }finally{
    FIG.sheet=keepSheet;SHOT.alpha=keepAlpha;
    window.__SOLID_FIG__({sheet:keepSheet,sheetIds:null});need();
  }
  const pixels=cap.getContext("2d").getImageData(0,0,cap.width,cap.height).data;
  let ax=cap.width,ay=cap.height,bx=-1,by=-1;
  for(let y=0;y<cap.height;y++)for(let x=0;x<cap.width;x++)if(pixels[(y*cap.width+x)*4+3]>8){
    ax=Math.min(ax,x);ay=Math.min(ay,y);bx=Math.max(bx,x);by=Math.max(by,y);
  }
  if(bx<0){
    SHEET_NOTICE={key:sheetNoticeKey(),text:"The selected body is outside this view. Pan to it before exporting."};
    syncSheetControl();
    return;
  }
  SHEET_NOTICE=null;
  const clipped=ax<2 || ay<2 || bx>cap.width-3 || by>cap.height-3;
  const zlo=info.lo[2],zhi=info.hi[2],rows=[];
  const margin=Math.max(12,.04*Math.max(info.hi[0]-info.lo[0],info.hi[1]-info.lo[1]));
  const x0=info.lo[0]-margin,x1=info.hi[0]+margin,y0=info.lo[1]-margin,y1=info.hi[1]+margin;
  for(let i=0;i<N;i++)if(Z[i]>=zlo-2.6&&Z[i]<=zhi+2.6){
    const quad=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]].map(q=>sheetProject(F,q[0],q[1],Z[i],cap.width,cap.height));
    for(const q of quad){ax=Math.min(ax,q[0]);ay=Math.min(ay,q[1]);bx=Math.max(bx,q[0]);by=Math.max(by,q[1]);}
    rows.push({i,z:Z[i],id:String(META.planes[i].section_id).split("_").pop(),quad});
  }
  // One SVG unit is one CSS pixel at the current camera. Cropping removes empty
  // margins only; unlike fit-to-box, it cannot cancel a change of viewer zoom.
  const rasterScale=cap.width/camera.viewportCss[0],fit=1/rasterScale,padPx=24*rasterScale;
  const box={x:Math.max(0,Math.floor(ax-padPx)),y:Math.max(0,Math.floor(ay-padPx))};
  box.w=Math.min(cap.width,Math.ceil(bx+padPx))-box.x;box.h=Math.min(cap.height,Math.ceil(by+padPx))-box.y;
  const shot=document.createElement("canvas");shot.width=box.w;shot.height=box.h;
  shot.getContext("2d").drawImage(cap,-box.x,-box.y);
  const PAD=36,HEAD=126,FOOT=116,SIDE=230,GAP=32;
  const plotW=Math.max(620,Math.ceil(box.w*fit)),plotH=Math.max(460,Math.ceil(box.h*fit));
  const SW=PAD*2+plotW+GAP+SIDE,SH=HEAD+plotH+FOOT;
  const imgX=PAD+(plotW-box.w*fit)/2,imgY=HEAD+(plotH-box.h*fit)/2,rx=PAD+plotW+GAP;
  const P=(x,y,z)=>{const q=sheetProject(F,x,y,z,cap.width,cap.height);return [imgX+(q[0]-box.x)*fit,imgY+(q[1]-box.y)*fit];};
  const L=[],group=(id,label,content)=>'<g inkscape:groupmode="layer" id="'+id+'" inkscape:label="'+label+'">'+content+'</g>';
  const number=n=>Number(n.toFixed(2)).toString(),sample=window.__SAMPLE_ID__||document.title.split(" ")[0];
  const title=info.selected,titleSub=info.nerves.length&&info.glands.length?"Nerve–gland neighbourhood":"Selected 3D neighbourhood";
  let g=sheetText(PAD,29,"LOCAL 3D VIEW",11,{fill:SHEET.muted,weight:"600"});
  g+=sheetText(PAD,66,title,32,{weight:"600"});
  g+=sheetText(PAD,93,sample+"  /  "+titleSub,14,{fill:SHEET.muted});
  g+=sheetText(SW-PAD,33,"CURRENT CAMERA",11,{anchor:"end",fill:SHEET.muted,weight:"600"});
  g+=sheetText(SW-PAD,61,"Viewer zoom ×"+number(S.zoom),19,{anchor:"end",weight:"600"});
  g+=sheetText(SW-PAD,87,"Z stretch ×"+number(S.exag)+"  ·  XY undistorted",13,{anchor:"end",fill:SHEET.muted});
  g+=sheetLine([PAD,109],[SW-PAD,109],"#d9e1e5",1);
  L.push(group("heading","Selection and current camera",g));
  const clip='<defs><clipPath id="plot_clip"><rect x="'+PAD+'" y="'+HEAD+'" width="'+plotW+'" height="'+plotH+'"/></clipPath></defs>';
  // Section frames carry geometry; sparse emphasis avoids a dark cage of lines.
  const stride=Math.max(1,Math.ceil(rows.length/7));
  g='<g clip-path="url(#plot_clip)">';
  const projectedFrames=new Set();
  rows.forEach((r,j)=>{const key=r.quad.flat().map(n=>(n*fit).toFixed(1)).join(',');
    if(projectedFrames.has(key))return;projectedFrames.add(key);
    const major=j===0||j===rows.length-1||j%stride===0;
    g+='<path d="M '+r.quad.map(q=>[imgX+(q[0]-box.x)*fit,imgY+(q[1]-box.y)*fit].map(n=>n.toFixed(2)).join(',')).join(' L ')+' Z" fill="none" stroke="#6b8290" stroke-width="'+(major?.8:.45)+'" opacity="'+(major?.3:.1)+'"/>';});
  L.push(group("section_planes","Measured section positions",g+'</g>'));
  L.push(group("bodies",info.ids.length+" selected and neighbouring bodies",'<image x="'+imgX+'" y="'+imgY+'" width="'+(box.w*fit)+'" height="'+(box.h*fit)+'" xlink:href="'+shot.toDataURL("image/png")+'"/>'));
  // The true-depth ruler has its own physical Z scale. It stays readable from
  // above, where the projected section planes all coincide; no false leaders.
  g=sheetText(rx,HEAD+17,"SECTION DEPTH",11,{weight:"600",fill:SHEET.muted});
  g+=sheetText(rx,HEAD+42,rows.length+" section planes",16,{weight:"600"});
  g+=sheetText(rx,HEAD+65,"True z · µm",12,{fill:SHEET.muted});
  const rulerTop=HEAD+88,rulerBottom=HEAD+plotH-128;
  const zmin=rows.length?rows[0].z:zlo,zmax=rows.length?rows[rows.length-1].z:zhi;
  const Y=z=>rulerTop+(z-zmin)/Math.max(1,zmax-zmin)*(rulerBottom-rulerTop);
  g+=sheetLine([rx+10,rulerTop],[rx+10,rulerBottom],"#a9bac3",1);
  let labelY=-Infinity;
  rows.forEach((r,j)=>{const y=Y(r.z),last=j===rows.length-1,major=j===0||last||(y-labelY>=32&&Y(zmax)-y>=27);
    g+=sheetLine([rx+10,y],[rx+(major?24:17),y],major?"#526b78":"#b6c4cc",major?1.2:.7);
    if(major){labelY=y;g+=sheetText(rx+34,y+4,r.id,12);g+=sheetText(rx+SIDE,y+4,number(r.z),12,{anchor:"end",fill:SHEET.muted});}});
  L.push(group("section_ruler","True section depth independent of projection",g));
  const centre=info.lo.map((v,i)=>(v+info.hi[i])/2),origin=P(...centre);
  const unitAxis=i=>{const p=centre.slice();p[i]+=1;const q=P(...p);return [q[0]-origin[0],q[1]-origin[1]];};
  const axes=[0,1,2].map(unitAxis),ux=Math.hypot(...axes[0]),uy=Math.hypot(...axes[1]),xy=ux>=uy?0:1;
  const scaleItems=[];
  function bar(axis,y){
    const u=Math.hypot(...axes[axis]);if(u<.025)return sheetText(rx,y,"Z axis viewed end-on",11,{fill:SHEET.muted});
    const ladder=[.1,.2,.5,1,2,5,...SB_STEPS];
    const lengths=ladder.filter(n=>n*u<=140),um=lengths.length?lengths[lengths.length-1]:ladder[0];
    const length=um*u,label=(um>=1000?number(um/1000)+" mm":number(um)+" µm")+" ("+['X','Y','Z'][axis]+")";
    scaleItems.push({axis:['X','Y','Z'][axis],um,svgLength:length,svgUnitsPerUm:u});
    return sheetText(rx,y,label,12,{weight:"600"})+sheetLine([rx,y+12],[rx+length,y+12],SHEET.ink,2.5)
      +sheetLine([rx,y+8],[rx,y+16],SHEET.ink,1)+sheetLine([rx+length,y+8],[rx+length,y+16],SHEET.ink,1);
  }
  g=sheetText(rx,HEAD+plotH-87,"PROJECTED SCALE",10,{weight:"600",fill:SHEET.muted});
  g+=bar(xy,HEAD+plotH-61);g+=bar(2,HEAD+plotH-17);
  const fy=HEAD+plotH+30;
  g+=sheetLine([PAD,fy-12],[SW-PAD,fy-12],"#d9e1e5",1);
  const swatch=(x,y,c)=>'<circle cx="'+x+'" cy="'+y+'" r="5" fill="'+c+'"/>';
  let lx=PAD+5;
  if(info.nerves.length){g+=swatch(lx,fy+10,info.nerve_colour)+sheetText(lx+14,fy+14,(info.nerves.length===1?info.nerves[0]:info.nerves.length+" nerves"),13,{weight:"600"});lx+=140;}
  if(info.glands.length){const colours=info.gland_colours.slice(0,4);colours.forEach((c,j)=>g+=swatch(lx+j*12,fy+10,c));
    g+=sheetText(lx+colours.length*12+10,fy+14,info.glands.length+" gland"+(info.glands.length===1?"":"s"),13,{weight:"600"});}
  g+=sheetText(PAD,fy+42,"Neighbourhood ≤"+info.nearby_um+" µm · 3D bounds distance · model-derived surfaces",12,{fill:SHEET.muted});
  g+=sheetText(PAD,fy+64,clipped?"Current viewport clips the selected geometry; zoom out to include it all.":"Current angle and zoom preserved; empty margins trimmed only.",11,{fill:clipped?"#9e4726":SHEET.muted});
  // Compact projected orientation glyph; components retain foreshortening.
  const gx=SW-PAD-54,gy=fy+36,R=camera.orientation,colours=['#af5547','#537867','#4f72a0'];
  const directionAxes=[[R[0],-R[1]],[-R[4],R[5]],[R[8],-R[9]]],axisLabels=[];
  directionAxes.forEach((a,i)=>{const dx=a[0]*30,dy=a[1]*30;if(Math.hypot(dx,dy)<3)return;
    const tx=gx+dx+(dx<0?-8:8);let ty=gy+dy+4;
    while(axisLabels.some(p=>Math.abs(p[0]-tx)<18&&Math.abs(p[1]-ty)<13))ty+=13;
    axisLabels.push([tx,ty]);
    g+=sheetLine([gx,gy],[gx+dx,gy+dy],colours[i],1.8)+sheetText(tx,ty,['X','Y','Z'][i],11,{fill:colours[i],anchor:dx<0?'end':'start'});});
  L.push(group("scale_and_key","Physical scales and selection key",g));
  const metadata={sample,solidBuildStamp:window.__SOLID_STAMP__,frameworkBuildStamp:window.__FRAMEWORK_BUILD_STAMP__,selection:info.selected,selectedIds:info.selection,objectIds:info.ids,nearbyUm:info.nearby_um,neighbourhoodMetric:"unexaggerated 3D bounding-box distance",camera,
    capture:{width:cap.width,height:cap.height,rasterScale,crop:box,svgUnitsPerCssPx:1,clipped,transparent:keepAlpha},scales:scaleItems,sections:rows.map(r=>({id:r.id,z_um:r.z}))};
  const svg='<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="'+SW+'" height="'+SH+'" viewBox="0 0 '+SW+' '+SH+'">'
    +'<title>'+sheetEsc(sample+' · '+title+' · current camera')+'</title><metadata id="figure_metadata">'+sheetEsc(JSON.stringify(metadata))+'</metadata>'+clip
    +(keepAlpha?'':'<rect id="paper" width="100%" height="100%" fill="#ffffff"/>')+L.join('')+'</svg>';
  if(!(opt&&opt.save===false))saveBlob(new Blob([svg],{type:"image/svg+xml"}),sample+'_'+(info.selection.length===1?info.selected:info.selection.length+'-selected')+'_zoom'+number(S.zoom)+'_z'+number(S.exag)+'.svg');
  return svg;
}
{ const sb = document.getElementById("figSheet");
  if(sb) sb.onclick = () => readyExport(()=>{
    const svg=figureSheet();
    if(!svg)throw Error(SHEET_NOTICE?.text || "Select a visible nerve or gland before exporting.");
    return svg;
  },false); }
{ const rb = document.getElementById("colReset");
  if(rb) rb.onclick = () => resetColours(); }
document.getElementById("shotPng").onclick = () => shoot("png");
document.getElementById("shotJpg").onclick = () => shoot("jpg");
document.getElementById("shotPdf").onclick = () => shoot("pdf");
document.getElementById("shotCancel").onclick=()=>{if(EXPORT_JOB)EXPORT_JOB.cancelled=true;};
(function(){
  const sl = document.getElementById("shotScale"), lab = document.getElementById("shotScaleVal"),
        note = document.getElementById("shotSizeNote"), al = document.getElementById("shotAlpha");
  if(!sl) return;
  function show(){
    SHOT.scale = Math.max(1, parseInt(sl.value, 10) || 1);
    if(lab) lab.textContent = SHOT.scale + "x";
    if(!note) return;
    const base = Math.min(2, window.devicePixelRatio || 1);
    const longSide = Math.max(cv.clientWidth, cv.clientHeight, 1);
    const lim = Math.min(SHOT_MAX_PX, gl ? gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) : SHOT_MAX_PX);
    const f = Math.max(base, Math.min(base * SHOT.scale, lim / longSide));
    const w = Math.round(cv.clientWidth * f), h = Math.round(cv.clientHeight * f);
    note.textContent = w + " x " + h + " px"
      + (f < base * SHOT.scale - 1e-6 ? " (as large as this browser allows)" : "");
  }
  sl.oninput = show; show();
  window.addEventListener("resize", show);
  if(al) al.onchange = () => { SHOT.alpha = al.checked; };
})();
// --------------------------------------------------------------- figure mode
// A separate display option under Output. It only ever writes into FIG, so clearing
// the checkbox puts every layer back exactly where the ordinary controls left it.
(function(){
  const on = document.getElementById("figOn"), body = document.getElementById("figBody");
  if(!on || !body) return;
  const A = document.getElementById("figSecA"), B = document.getElementById("figSecB"),
        note = document.getElementById("figNote"), objs = document.getElementById("figObjs");
  function push(){
    if(window.__SOLID_FIG__) window.__SOLID_FIG__({
      on: FIG.on, nerve: FIG.nerve && FIG.n3, gland: FIG.gland && FIG.g3,
      nerveA: FIG.nerveA, glandA: FIG.glandA, hide: FIG.hide, multi: FIG.multi});
  }
  function apply(){
    body.disabled = !FIG.on;
    push();
    if(FIG.on){
      if(NERVE && FIG.nerve && FIG.n2 && !NERVE.vector_contours)loadCellTextures(S.basis,need,nerveSource());
      autoRes();
    }
    ANIM.to = layoutFor(S.sel, S.solo); ANIM.from = ANIM.to; ANIM.t0 = 0;
    need(); paintOverlay();
  }
  const opt = (v, txt) => { const o = document.createElement("option");
                            o.value = v; o.textContent = txt; return o; };
  function fillSections(){
    if(!A || !B) return;
    A.replaceChildren(); B.replaceChildren();
    B.appendChild(opt("", "second section: none"));
    META.planes.forEach((pl, i) => {
      const nm = String(pl.section_id).split("-").pop();
      A.appendChild(opt(String(i), nm)); B.appendChild(opt(String(i), nm));
    });
    A.value = String(FIG.secA !== null ? FIG.secA : (S.sel === null ? S.zi : S.sel));
    B.value = FIG.secB === null ? "" : String(FIG.secB);
  }
  function figureObjects(){
    const list = window.__SOLID_OBJECTS__ ? [...window.__SOLID_OBJECTS__()] : [];
    const ids = new Set(list.map(o=>o.id));
    for(const p of NERVE?.planes||[])for(const r of p.regions||[]){
      if(r.kind==="2d" && !ids.has(r.id)){
        ids.add(r.id);list.push({id:r.id,label:r.id+" (2D)",group:"NERVE 2D",regionOnly:true});
      }
    }
    return list;
  }
  function fillObjects(){
    if(!objs) return;
    const list = figureObjects();
    objs.replaceChildren();
    let group = "";
    for(const o of list){
      if(o.group !== group){ group = o.group;
        const h = document.createElement("div"); h.className = "fgrp";
        h.textContent = group; objs.appendChild(h); }
      const l = document.createElement("div"), c = document.createElement("input"),
            s = document.createElement("button");
      l.className="figobj";s.type="button";s.className="figpick";s.dataset.selectId=o.id;
      c.setAttribute("aria-label","Show "+o.label);
      s.title="Select or deselect "+o.label;
      if(o.regionOnly){s.disabled=true;s.title="2D region; no reconstructed 3D body";}
      s.onclick=()=>{if(!s.matches(":disabled"))window.__SOLID_SELECT__?.(o.id);};
      c.type = "checkbox"; c.checked = !FIG.hide[o.id]; c.dataset.oid = o.id;
      c.onchange = () => { if(c.checked) delete FIG.hide[o.id]; else FIG.hide[o.id] = true;
                           apply(); };
      s.textContent = o.label;
      const selected=(window.__SOLID_SELECTION__?.()||[]).includes(o.id);
      l.classList.toggle("figsel",selected);s.setAttribute("aria-pressed",String(selected));
      l.appendChild(c); l.appendChild(s); objs.appendChild(l);
    }
    if(note) note.textContent = list.length
      ? list.length + " objects · boxes show/hide; body names select"
      : "3-D bodies are loading";
  }
  function setAll(v){
    if(v) FIG.hide = Object.create(null);
    else figureObjects()
           .forEach(o => { FIG.hide[o.id] = true; });
    objs && objs.querySelectorAll("input[data-oid]").forEach(c => { c.checked = v; });
    apply();
  }
  on.onchange = () => { FIG.on = on.checked;
                        if(FIG.on){
                          if(FIG.secA === null) FIG.secA = (S.sel === null ? S.zi : S.sel);
                          fillSections(); fillObjects(); }
                        apply(); };
  // The controls stay expanded. Refresh asynchronous data without a fold/unfold
  // gesture, and replay figure settings if the payload arrived after activation.
  window.addEventListener("viewer-solid-ready", () => { fillObjects(); push(); });
  fillSections(); fillObjects();
  if(A) A.onchange = () => { const i = +A.value;
                             if(Number.isInteger(i)) FIG.secA = i;
                             apply(); };
  if(B) B.onchange = () => { FIG.secB = B.value === "" ? null : +B.value; apply(); };
  const flag = (id, key) => { const el = document.getElementById(id); if(!el) return;
    el.onchange = () => { FIG[key] = el.checked; apply(); }; };
  flag("figNerve", "nerve"); flag("figN2", "n2"); flag("figN3", "n3");
  flag("figGland", "gland"); flag("figG2", "g2"); flag("figG3", "g3");
  flag("figMulti", "multi"); flag("figSections", "sections"); flag("figNerveFill", "nerveFill");
  document.getElementById("figClearSelection").onclick=()=>window.__SOLID_CLEAR_FOCUS__?.();
  const slide = (id, key) => { const el = document.getElementById(id); if(!el) return;
    const out = document.getElementById(id + "Val");
    el.oninput = () => { FIG[key] = (+el.value) / 100;
                         if(out) out.textContent = el.value + "%"; apply(); }; };
  slide("figNA", "nerveA"); slide("figGA", "glandA");
  // clicking a body in the canvas brings its row into view and marks it, so the box
  // to clear is the one under the pointer rather than one to hunt for in 130 rows
  window.__FIG_ON_SELECT__ = id => {
    if(!objs) return;
    const ids=window.__SOLID_SELECTION__?.()||[],selected=new Set(ids);
    objs.querySelectorAll(".figobj").forEach(row=>{
      const button=row.querySelector(".figpick"),on=selected.has(button.dataset.selectId);
      row.classList.toggle("figsel",on);button.setAttribute("aria-pressed",String(on));
    });
    const count=document.getElementById("figSelectionCount"),clear=document.getElementById("figClearSelection");
    if(count)count.textContent=ids.length+" selected";
    if(clear)clear.disabled=!ids.length;
    if(FIG.on&&id){const row=Array.from(objs.querySelectorAll(".figpick")).find(b=>b.dataset.selectId===id);row?.scrollIntoView({block:"nearest"});}
    syncSheetControl();
  };
  const ab = document.getElementById("figAll"), nb = document.getElementById("figNone");
  if(ab) ab.onclick = () => setAll(true);
  if(nb) nb.onclick = () => setAll(false);
})();
document.getElementById("infoBtn").onclick = () => { toggleInfo(); backToCanvas(); };
{
  const rail = document.getElementById("rail");
  for(const b of rail.children) if(b.dataset.g){
    const g = b.dataset.g; b.onclick = () => { openGroup(g); backToCanvas(); };
  }
}
// A button keeps keyboard focus after a click, and the page only reads keys while the
// canvas (or nothing) has focus -- so clicking an icon would silently switch the
// keyboard off. Pointer clicks on the chrome hand focus back; Tab does not, so a
// keyboard user keeps it.
function backToCanvas(){
  if(document.activeElement && document.activeElement.matches
     && document.activeElement.matches("#rail button, #topbar button"))
    setTimeout(() => cv.focus(), 0);
}
// Fullscreen and any resize must RE-SIZE the canvas, not stretch it: draw() reads
// clientWidth/clientHeight every frame and sets the backing store from them.
// ------------------------------------------- the layer strip in full screen (10b)
// Retracted to a 3 px hint line, out when the pointer goes for it, and it does not
// snap shut the moment the pointer leaves, because then it cannot be reached.
const STRIP_IN_PX = 64, STRIP_OUT_PX = 72;   // 8 px of hysteresis, so it cannot flicker
const STRIP_CLOSE_MS = 800;
let stripTimer = null, stripPinned = false, pointerFocus = false;
function stripOpen(on){
  const app = document.getElementById("app");
  if(!app) return;
  // the corner readouts step up by the strip's own height, so measure it rather than
  // guessing: the ruler, the guide and the cells all contribute to it
  const b = document.getElementById("bottom");
  if(b) app.style.setProperty("--striph", Math.round(b.offsetHeight) + "px");
  if(on) app.setAttribute("data-strip", "open"); else app.removeAttribute("data-strip");
}
// with the scrollbar hidden, the fade is what says "there is more strip this way"
function stripFade(){
  const el = document.getElementById("selector"); if(!el) return;
  const max = Math.max(0, el.scrollWidth - el.clientWidth), x = el.scrollLeft;
  el.style.setProperty("--mL", (x > 1 ? 16 : 0) + "px");
  el.style.setProperty("--mR", (x < max - 1 ? 16 : 0) + "px");
}
function closeStripSoon(ms){
  if(stripTimer) clearTimeout(stripTimer);
  stripTimer = setTimeout(() => {
    if(stripPinned || scrub) return;
    const b = document.getElementById("bottom");
    if(b && b.contains(document.activeElement)) return;
    stripOpen(false);
  }, ms);
}
// In full screen the strip comes out for the POINTER and nothing else. Changing
// layers from the keyboard used to slide it out for 1.5 s, which both interrupted the
// picture and, held down, re-armed the timer on every key so it never went back.
// The corner readout is what tells a keyboard user which layer they are on.
{
  const b = document.getElementById("bottom");
  if(b){
    b.addEventListener("pointerenter", () => { stripPinned = true; stripOpen(true);
      if(stripTimer) clearTimeout(stripTimer); });
    b.addEventListener("pointerleave", () => { stripPinned = false;
      closeStripSoon(STRIP_CLOSE_MS); });
    // Only a KEYBOARD arrival pins the strip open. A click also moves focus into the
    // cell (they are focusable so the strip can be tabbed through), and that used to
    // satisfy "focus is inside" forever, so the strip never retracted again.
    b.addEventListener("focusin", () => {
      if(!pointerFocus){ stripPinned = true; stripOpen(true); }
    });
    b.addEventListener("focusout", () => { stripPinned = false;
      closeStripSoon(STRIP_CLOSE_MS); });
  }
  window.addEventListener("pointermove", e => {
    if(!fullscreenOn() || stripPinned) return;
    const fromBottom = window.innerHeight - e.clientY;
    if(fromBottom <= STRIP_IN_PX){ if(stripTimer) clearTimeout(stripTimer);
      stripOpen(true); }
    else if(fromBottom > STRIP_OUT_PX) closeStripSoon(STRIP_CLOSE_MS);
  });
}
function onFullscreenChange(){
  const b = document.getElementById("fsBtn");
  if(b) b.setAttribute("aria-pressed", String(fullscreenOn()));
  const app = document.getElementById("app");
  if(app){
    if(fullscreenOn()) app.setAttribute("data-fs", "1");
    else { app.removeAttribute("data-fs"); app.removeAttribute("data-strip"); }
  }
  if(!lowDpr) DPR_ACTIVE = Math.min(2, window.devicePixelRatio || 1);
  // force a re-measure rather than let the old backing store be stretched
  cv.width = 1; cv.height = 1;
  requestAnimationFrame(() => { setCanvasSize(); need(); });
}
for(const ev of ["fullscreenchange", "webkitfullscreenchange"])
  document.addEventListener(ev, onFullscreenChange);
document.getElementById("resetCam").onclick = () => resetView(true);
document.getElementById("prevBtn").onclick = () => stepLayer(-1);
document.getElementById("nextBtn").onclick = () => stepLayer(1);
document.getElementById("clearSelBtn").onclick = () => setSelected(null);
document.getElementById("gapBtn").onclick = () => jumpGap(1);
let timer = null;
function setPlay(on){
  S.playing = on;
  const b = document.getElementById("playBtn");
  b.setAttribute("aria-pressed", String(on)); b.textContent = on ? "pause" : "play";
  if(timer){ clearInterval(timer); timer = null; }
  if(on) timer = setInterval(() => setZ((S.zi + 1) % N), 220);
}
document.getElementById("playBtn").onclick = () => setPlay(!S.playing);
// Stepping through layers keeps the selection on the layer you land on, so the
// expansion follows you instead of staying behind on the one you clicked.
function stepLayer(d){
  const i = Math.max(0, Math.min(N-1, S.zi + d));
  if(S.sel !== null) setSelected(i); else setZ(i);
}
document.getElementById("keysBtn").onclick = () => {
  const k = document.getElementById("keyhelp");
  k.style.display = (k.style.display === "block") ? "none" : "block";
};
function jumpGap(dir){
  if(!GAP_IDX.length) return;
  const cur = S.zi;
  const arr = dir > 0 ? GAP_IDX : GAP_IDX.slice().reverse();
  for(const g of arr){ if(dir > 0 ? g > cur : g < cur){ setZ(g); return; } }
  setZ(arr[0]);
}

// --------------------------------------------------------------- keyboard
let spaceHeld = false;
window.addEventListener("keyup", e => { if(e.key === " ") spaceHeld = false; });
// Every shortcut is a bare key or a Shift combination, and NO Ctrl / Cmd / Alt
// combination is ever taken. That is what keeps Ctrl+R, Ctrl+F, Ctrl+S, Ctrl+P,
// Ctrl+O, Ctrl+L, Ctrl+W/T/N, Ctrl +/-/0, Ctrl+A/C/V/Z, Ctrl+Tab, Alt+arrows,
// F5, F11 and Ctrl+Shift+I working: the handler returns before it looks at the key.
// The page also only listens while the canvas has focus or nothing is focused, and
// preventDefault is called for the keys it actually handles and for nothing else.
function keyMine(e){
  if(BROWSING) return false;      // the volume is not on screen; its keys are not live
  if(e.ctrlKey || e.metaKey || e.altKey) return false;
  const t = e.target || {}, tag = t.tagName;
  // a focused control owns its own keys: space presses a button, arrows move a slider
  if(tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || tag === "BUTTON"
     || tag === "SUMMARY" || tag === "A") return false;
  const a = document.activeElement;
  if(a && (a.tagName === "BUTTON" || a.tagName === "SUMMARY" || a.tagName === "A"
           || a.tagName === "INPUT" || a.tagName === "SELECT")) return false;
  if(!a || a === document.body || a === cv) return true;
  return a.id === "selector" || a.id === "gl";
}
window.addEventListener("keydown", e => {
  if(!keyMine(e)) return;
  let h = true;
  const K = e.key;
  if(e.shiftKey && K.indexOf("Arrow") === 0){   // pan; bare arrows change plane
    const px = 0.10 * devH();
    panByPx(K === "ArrowLeft" ? px : K === "ArrowRight" ? -px : 0,
            K === "ArrowUp" ? px : K === "ArrowDown" ? -px : 0); }
  else if(K === "ArrowUp") stepLayer(1);
  else if(K === "ArrowDown") stepLayer(-1);
  else if(K === "ArrowLeft" || K === "ArrowRight")
    rotateByScreen((K === "ArrowLeft" ? -1 : 1) * 5 * Math.PI/180, 0);
  else if(K === "PageUp") stepLayer(10);
  else if(K === "PageDown") stepLayer(-10);
  else if(K === "Home") stepLayer(-N);
  else if(K === "End") stepLayer(N);
  else if(K === "G" || K === "g") jumpGap(e.shiftKey ? -1 : 1);
  // S is the switch in the corner, said with the keyboard: one press shows only the
  // layer in hand, the next press brings the stack back. With nothing selected yet it
  // selects the layer the controls are pointing at, so the key never does nothing.
  else if(K === "s" || K === "S") toggleSectionScope();
  else if(K === "f") zoomToFit();
  else if(K === "F") toggleFullscreen();
  else if(K === " "){ if(!spaceHeld){ spaceHeld = true; setPlay(!S.playing); } }
  else if(K === "+" || K === "=") zoomAt(0, 0, 1.25);
  else if(K === "-" || K === "_") zoomAt(0, 0, 1/1.25);
  else if(K === "0") { if(e.shiftKey) resetView(true); else resetView(false); }
  else if(K === "v" || K === "V"){ const order = ["top","side","iso"];
    const cur = (Math.abs(S.el) < 0.05 && Math.abs(S.az) < 0.05) ? 0
              : (Math.abs(S.el) > 1.4 ? 1 : 2);
    const nx = order[(cur+1)%3];
    if(nx === "top") rotateTo(0, 0);
    else if(nx === "side") rotateTo(0, Math.PI/2 - 0.02);
    else rotateTo(-0.62, 0.42); }
  else if(K === "p" || K === "P") setPivotUnderCursor();
  // D steps the BASE STYLE. It cycles all three because there are three; it has never
  // touched the overlay, which has its own key.
  else if(K === "d" || K === "D"){
    const o = ["grey", "native", "translucent"];
    setColour(o[(o.indexOf(S.colour) + 1) % o.length]);
  }
  // Its own key rather than a third stop on D, so the reader can flip between
  // the prediction and whichever picture of the tissue they were already using
  else if((K === "c" || K === "C") && cellsBuilt()) toggleCellLayer();
  else if((K === "n" || K === "N") && nerveBuilt()) setNerveReg(!S.nerveReg);
  else if(K === "i" || K === "I") toggleInfo();
  else if(K >= "1" && K <= "7") stdView(+K);
  else if(K === "8") setBasis(BASES[0]);
  else if(K === "9"){ if(BASES[1]) setBasis(BASES[1]); }
  else if(K === "Escape"){ S.pickMsg = "";
    window.__SOLID_CLEAR_FOCUS__?.();focusBody(null);
    toggleInfo(false); if(S.group) openGroup(S.group);
    setSelected(null);          // which also leaves "only this layer", if it was on
    h = false; }        // Escape is left to the browser: it is how fullscreen ends
  else if(K === "?") document.getElementById("keysBtn").click();
  else h = false;
  if(h && e.preventDefault) e.preventDefault();
});

// A resize -- including entering or leaving full screen, and a move to a screen with
// a different pixel ratio -- has to re-read the device pixel ratio, or the backing
// store keeps the old one and the picture is a stretched version of the old raster.
function refreshDpr(){
  if(!lowDpr) DPR_ACTIVE = Math.min(2, window.devicePixelRatio || 1);
  stripFade();          // a narrower window changes which end of the strip is at a stop
  need();
}
// Focus rings are decided here, not by the browser. A ring appears only when focus
// arrived from the keyboard; a pointer click never draws one, including the click that
// hands focus back to the canvas so the keyboard keeps working.
let focusByPointer = true;
window.addEventListener("pointerdown", () => { focusByPointer = true;
  const a = document.activeElement; if(a && a.classList) a.classList.remove("kbfocus");
}, true);
window.addEventListener("keydown", e => {
  if(e.key === "Tab") focusByPointer = false;
}, true);
window.addEventListener("focusin", e => {
  const t = e.target;
  if(t && t.classList) t.classList.toggle("kbfocus", !focusByPointer);
}, true);
window.addEventListener("focusout", e => {
  const t = e.target; if(t && t.classList) t.classList.remove("kbfocus");
}, true);
window.addEventListener("resize", refreshDpr);
// A media query only tracks one ratio, so it has to be re-armed on the new one after
// every change; this is what catches a move to a second screen with no window resize.
let dprMQ = null;
function onDprChange(){ refreshDpr(); watchDpr(); }
function watchDpr(){
  if(!window.matchMedia) return;
  try {
    if(dprMQ && dprMQ.removeEventListener) dprMQ.removeEventListener("change", onDprChange);
    dprMQ = window.matchMedia("(resolution: " + (window.devicePixelRatio || 1) + "dppx)");
    if(dprMQ && dprMQ.addEventListener) dprMQ.addEventListener("change", onDprChange);
  } catch(_){}
}
watchDpr();
selector.addEventListener("scroll", () => {
  const rs = document.getElementById("rulerscroll"); if(rs) rs.scrollLeft = selector.scrollLeft;
  drawGuide(); stripFade(); });

buildCells(); buildRuler(); setZ(S.zi); syncOnly(); stripFade();
document.getElementById("basisNote").textContent = META.bases[S.basis].label;
document.documentElement.setAttribute("data-bg", BG);
if(S.colour === "native") loadRGBSet(S.basis, S.res, need);
loadSet(S.basis, S.res, need);
need();

cv.addEventListener("webglcontextlost", e => {
  e.preventDefault();
  if(window.__VIEWER_DISPOSING__) return;
  const el = document.getElementById("corner");
  if(el) el.textContent = "graphics memory exhausted - reloading";
}, false);
cv.addEventListener("webglcontextrestored", () => {
  if(!window.__NO_RELOAD__) location.reload();
}, false);
window.__VIEWER__ = {S: S, N: N, Z: Z, setZ: setZ, setBasis: setBasis,
  FIG: FIG, secList: secList, grabView: grabView,
  SHOT: SHOT, captureView: captureView, figureSheet: figureSheet, exportView: shoot,
  exportState:()=>EXPORT_JOB?{dpr:EXPORT_JOB.dpr,cancelled:EXPORT_JOB.cancelled}:null,
  zoomAt: zoomAt, zmm: zmm, zmmReal: zmmReal, SEL: SEL, BASE_OPACITY: BASE_OPACITY,
  setSelected: setSelected, setOnly: setOnly, syncOnly, stepLayer: stepLayer, spreadUm: spreadUm,
  drawn: () => lastK, MODS: MODS, need: need, stripFade: stripFade,
  tiles: () => tileStats, tileIndex: tileIndex, tileLadder: tileLadder,
  TIDX: TIDX, TTEX: TTEX,
  visibleUV: visibleUV, umPerDevicePx: umPerDevicePx,
  maxSpreadUm: maxSpreadUm, selScale: selScale, alphaFor: alphaFor,
  cellAlphaFor: cellAlphaFor, CELL_OPACITY: CELL_OPACITY,
  animating: animating, animU: animU,
  layoutFor: layoutFor, beginTransition: beginTransition, ANIM: ANIM,
  setPivotUnderCursor: setPivotUnderCursor, IS_MAC: IS_MAC,
  pivotAtClient, surfaceUnderCursor, pointFromView, focusBody, bodyFocusPoint,
  focusedBody:()=>bodyFocus?.id||null,
  sectionContext,writeSectionDepth,sectionDepthSource,
  cameraFrame:()=>({MVP:mul(projFor(cv.width,cv.height),viewM()),
    hw:resInfo().canvas_mm.width/2,hh:resInfo().canvas_mm.height/2,sc:selScale()}),
  sepPx: sepPx, sepPxAt: sepPxAt, umPerDevicePx: umPerDevicePx, zoomMax: zoomMax,
  drawList: drawList, forwardN: forwardN, TEX: TEX, jumpGap: jumpGap,
  resetView: resetView, panByPx: panByPx,
  rotateTo: rotateTo, rotateByScreen: rotateByScreen, spinScreen: spinScreen,
  viewXY: viewXY, viewM: viewM, radNow: radNow, GAP_IDX: GAP_IDX,
  stdView: stdView, STDVIEW: STDVIEW, zoomToFit: zoomToFit,
  fitRect: fitRect, viewBounds: viewBounds, clientToView: clientToView,
  pickAt: pickAt, pickList: pickList, rayHit: rayHit, alphaAt: alphaAt, AMASK: AMASK,
  PICK_ALPHA: PICK_ALPHA,
  draw: draw, gl: gl, dpr: () => DPR_ACTIVE, refreshDpr: () => refreshDpr(),
  setColour: setColour, heRGBLoaded: heRGBLoaded,
  heRGBTotal: heRGBTotal, RGBTEX: RGBTEX, openGroup: openGroup,
  setHeSwap: setHeSwap, swapped: swapped, SWAPSET: SWAPSET, armKey: armKey,
  armOf: armOf, swapAvailable: swapAvailable, swapHere: swapHere,
  modsNow: modsNow, setKey: setKey,
  regionLines: regionLines, setGlandReg: setGlandReg, setGlandKind: setGlandKind, glandSource: glandSource,
  loadedNow: () => (LOADED[setKey(S.basis, S.res)] || 0),
  rgbLoadedNow: () => (RGBLOADED[setKey(S.basis, S.res)] || 0), texFor: texFor, tintFor: tintFor, tilesDir: tilesDir,
  modLabel: modLabel,
  CELLS: CELLS, CELLDRAW: CELLDRAW, CELLINFO: CELLINFO, CELLTEX: CELLTEX,
  cellsBuilt: cellsBuilt, cellsHere: cellsHere, cellsOnList: cellsOnList,
  toggleCellClass: toggleCellClass, toggleCellLayer: toggleCellLayer,
  setCellsOverlay: setCellsOverlay, TRANSLUCENT_W: TRANSLUCENT_W,
  setCellsAll: setCellsAll, CELLPT: CELLPT, CELLBUF: CELLBUF,
  cellMarkStats: () => cellMarkStats, cellMarksOn: cellMarksOn,
  cellMarksLevel: cellMarksLevel, stepLevel: stepLevel,
  cellsTotal: cellsTotal, cellsLoaded: cellsLoaded, cellFile: cellFile,
  toggleFullscreen: toggleFullscreen, toggleInfo: toggleInfo, DAPI_TINT: DAPI_TINT,
  // guarded on the TRANSITION, and it does nothing but ask for a redraw. Selecting
  // HT891Z1 at load must not run a single line that the page did not run before the
  // sample control existed, and "set the same value again" is the case that would.
  setBrowsing: v => { v = !!v; if(v === BROWSING) return; BROWSING = v;
                      if(!BROWSING) need(); },
  browsing: () => BROWSING,
  // what an overlay needs to draw into this scene, and nothing more
  gl: () => gl, mkProg: mkProg, canvas: () => cv,
  zMidUm: () => (ZMIN + ZMAX) / 2,
  resInfo: resInfo, DPR: () => DPR_ACTIVE, fast: () => lowDpr,
  setOverlay3D: fn => { OVERLAY3D = fn; need(); },
  CELLUI: CELLUI, CELLINFO: CELLINFO, buildClassList: buildClassList, setCellColour: setCellColour};
"""


BROWSER_JS = r"""
"use strict";
(function(){
const BR = window.__BROWSER__ || null;
const V = window.__VIEWER__;
const btn = document.getElementById("sampleBtn");
const pop = document.getElementById("samplePop");
const filt = document.getElementById("sampleFilter");
const list = document.getElementById("sampleList");
const foot = document.getElementById("sampleFoot");
const tag = document.getElementById("modeTag");
const bar = document.getElementById("samplebar");
if(!BR || !btn){ if(bar) bar.style.display = "none";
                 if(pop) pop.style.display = "none";
                 if(tag) tag.style.display = "none"; return; }

const app = document.getElementById("app");
const cvB = document.getElementById("bgl");
const ctx = cvB.getContext("2d");
const stage = document.getElementById("bstage");
const elNotice = document.getElementById("bnotice");
const elCorner = document.getElementById("bcorner");
const elChips = document.getElementById("bchips");
const elScale = document.getElementById("bscale");
const elSecs = document.getElementById("bsections");
const elClasses = document.getElementById("bclasses");
const elInfo = document.getElementById("binfo");
const elWarn = document.getElementById("bwarn");
const elCur = document.getElementById("bcur");

const CLS = BR.classes.map(c => c.name);
const COL = {}; BR.classes.forEach(c => { COL[c.name] = c.colour; });
// The volume opens with no class on, for the same reason: eleven classes at once is
// a colour field, not a reading. Nothing here either, so the two displays open
// saying the same thing.
const DEFAULT_ON = [];
// The disc the volume's per-cell layer draws, measured at 17.4 um across against a
// 15.7 um median from the segmentation's own areas. Same object, same size here.
const CELL_R_UM = 7.5;
const MIN_R_PX = 1.1;

const B = {sample: null, slide: null, list: [], idx: 0,
           scale: 1, cx: 0, cy: 0,          // px per micron; centre in microns
           cells: true, regions: true,
           on: CLS.filter(c => DEFAULT_ON.indexOf(c) >= 0),
           // which section, and which assay on it. "he" for all but one sample.
           section: null, layer: "he",
           // the three CODEX slots, as channel indices. DAPI in red to start; the
           // reader changes all three. Nothing here decides which markers matter.
           codex: [0, null, null],
           spots: true, inTissueOnly: false, xcells: true, xcolour: true};

const IMG = new Map();      // url -> Image | false (failed)
const CELLS = new Map();    // slide -> {x,y,cls,n} | null (in flight) | false
let pend = false, dpr = 1;

function slideInfo(s){ return BR.slides[s]; }
function sampleInfo(id){ return BR.samples.filter(s => s.id === id)[0]; }
// "(NONE)" is what the run manifest writes where a section has no block. It is a
// missing value, and every place the interface names a block says so in words.
function blockName(s){ return s.block_unknown ? "unknown" : s.block; }
function need(){ if(pend) return; pend = true;
  requestAnimationFrame(() => { pend = false; render(); }); }

// ------------------------------------------------------------------ cell binary
function decodeCells(buf, slide){
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1),
                                    dv.getUint8(2), dv.getUint8(3));
  if(magic !== "SBC1") throw new Error("cells " + slide + ": magic " + magic);
  const n = dv.getUint32(4, true), nc = dv.getUint16(8, true);
  // The header is checked rather than trusted. The volume's per-cell layer was
  // read at the wrong offset once: nothing failed, the divisor was the canvas
  // height instead of the sub-pixel step, and the whole layer drew into a corner
  // of the picture 19 px across while the page reported drawing 144220 cells.
  // A reader that cannot fail is a reader that can be silently wrong, so every
  // field that the drawing depends on is compared with what this page expects.
  if(nc !== CLS.length)
    throw new Error("cells " + slide + ": " + nc + " classes, palette has " + CLS.length);
  const q = dv.getFloat32(20, true);
  if(!(q > 0)) throw new Error("cells " + slide + ": quantisation " + q);
  const want = 32 + n * 5;
  if(buf.byteLength !== want)
    throw new Error("cells " + slide + ": " + buf.byteLength + " bytes, expected " + want);
  const x0 = dv.getFloat32(12, true), y0 = dv.getFloat32(16, true);
  const xq = new Uint16Array(buf, 32, n), yq = new Uint16Array(buf, 32 + 2 * n, n);
  const cls = new Uint8Array(buf, 32 + 4 * n, n);
  const x = new Float32Array(n), y = new Float32Array(n);
  for(let i = 0; i < n; i++){ x[i] = x0 + xq[i] * q; y[i] = y0 + yq[i] * q; }
  return {x: x, y: y, cls: cls, n: n};
}
function cellsFor(slide){
  if(CELLS.has(slide)) return CELLS.get(slide) || null;
  CELLS.set(slide, null);
  fetch(BR.dir + "/" + slideInfo(slide).cells)
    .then(r => { if(!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
    .then(b => { CELLS.set(slide, decodeCells(b, slide)); need(); })
    .catch(e => { CELLS.set(slide, false); console.error(e); need(); });
  return null;
}

// ------------------------------------------------------------------- tiles
function tileUrl(slide, lev, c, r){
  return BR.tiles_dir + "/" + slide + "/" + lev.dir + "/" + c + "_" + r + ".webp";
}
function imgFor(url){
  if(IMG.has(url)) return IMG.get(url);
  const im = new Image();
  IMG.set(url, null);
  im.onload = () => { IMG.set(url, im); need(); };
  im.onerror = () => { IMG.set(url, false); need(); };
  im.src = url;
  return null;
}
// A decoded 512 px tile is a megabyte of bitmap in the renderer whatever its file
// size, and the reader walks 27 samples. A cache measured in hundreds is a cache
// measured in hundreds of megabytes, which is how a tab dies. One screen needs a few
// dozen; this keeps about four screens' worth.
const IMG_CAP = 160, IMG_KEEP = 110;
function evictImages(){
  if(IMG.size < IMG_CAP) return;
  let drop = IMG.size - IMG_KEEP;
  for(const k of IMG.keys()){ if(drop-- <= 0) break; IMG.delete(k); }
}
function levels(){ const t = slideInfo(B.slide).tiles; return t ? t.levels : null; }
function pickLevel(umPerPx){
  const L = levels(); if(!L) return null;
  // the coarsest rung that still has at least one texel per screen pixel; when the
  // reader is past the finest rung there is nothing finer to pick, so it says so
  let best = L[0];
  for(const l of L) if(l.mpp <= umPerPx * 1.0001) best = l;
  return best;
}

// ------------------------------------------------------------------ camera
function stageSize(){ return [stage.clientWidth || 1, stage.clientHeight || 1]; }
// The camera works in microns and knows nothing about which assay is on screen; it
// only needs the current layer's extent. H&E additionally has a measured tissue box
// to open on, which the others do not, so they open on the whole thing.
function layerNow(){ const s = sectionInfo(B.section);
                     return s ? s.layers[B.layer] : null; }
function extentNow(){
  if(B.layer === "he") return slideInfo(B.slide).extent_um;
  const L = layerNow();
  return L && L.extent_um ? L.extent_um : [1, 1];
}
function finestNow(){
  if(B.layer === "he"){ const L = levels(); return L ? L[0].mpp : 1; }
  const L = layerNow();
  if(!L) return 1;
  const lv = (B.layer === "codex") ? (L.channels[0] || {}).levels : L.levels;
  return lv && lv.length ? lv[0].mpp : 1;
}
function fit(){
  const ex = extentNow();
  const t = (B.layer === "he") ? slideInfo(B.slide).tiles : null;
  const s = {extent_um: ex, tiles: t};
  const bb = (t && t.bbox_um) ? t.bbox_um : [0, 0, ex[0], ex[1]];
  const [w, h] = stageSize();
  const bw = Math.max(1, bb[2] - bb[0]), bh = Math.max(1, bb[3] - bb[1]);
  B.scale = Math.min(w / bw, h / bh) * 0.98;
  B.cx = (bb[0] + bb[2]) / 2; B.cy = (bb[1] + bb[3]) / 2;
}
function clampCam(){
  const ex = extentNow();
  const [w, h] = stageSize();
  const minS = Math.min(w / ex[0], h / ex[1]) * 0.25;
  // do not let the reader zoom more than 8x past the finest rung: past that it is
  // magnifying the encoder, and the picture stops being evidence of anything
  const maxS = 8 / finestNow();
  B.scale = Math.max(minS, Math.min(maxS, B.scale));
  B.cx = Math.max(-ex[0] * 0.25, Math.min(ex[0] * 1.25, B.cx));
  B.cy = Math.max(-ex[1] * 0.25, Math.min(ex[1] * 1.25, B.cy));
}
function toScreen(xu, yu){
  const [w, h] = stageSize();
  return [(xu - B.cx) * B.scale + w / 2, (yu - B.cy) * B.scale + h / 2];
}
function toUm(px, py){
  const [w, h] = stageSize();
  return [(px - w / 2) / B.scale + B.cx, (py - h / 2) / B.scale + B.cy];
}

// -------------------------------------------------- modality layers on a section
//
// A section can carry more than one assay. HT206B1's U1, U9 and U17 were imaged as
// H&E and run on Xenium; every other section in this cohort carries exactly one
// thing, and for those the layer machinery is invisible. What a layer has to supply
// is a tile ladder in MICRONS and, optionally, something to draw on top -- so the
// camera, the level picker and the tile loop below are shared, and only the choice
// of URL and of overlay differs.
const SECT = window.__SECTIONS__ || null;
function sectionInfo(id){ return SECT ? SECT.sections[id] : null; }
function layerOf(id, k){ const s = sectionInfo(id); return s ? s.layers[k] : null; }
const MOD_LABEL = {he: "H&E", codex: "CODEX", visium: "Visium",
                   xenium: "Xenium", cosmx: "CosMx"};
const MOD_ORDER = ["he", "codex", "xenium", "visium", "cosmx"];
// R, G, B in that order: the three slots a CODEX pick composites into
const CODEX_SLOT = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const CODEX_SLOT_NAME = ["red", "green", "blue"];
let offc = null, offx = null;
function offscreen(w, h){
  if(!offc){ offc = document.createElement("canvas"); offx = offc.getContext("2d"); }
  if(offc.width !== w || offc.height !== h){ offc.width = w; offc.height = h; }
  return [offc, offx];
}

// The tile loop, shared by every layer. `lev` is a level descriptor with mpp, dir,
// cols, rows and present; `url` builds a tile's address. Returns [drawn, pending,
// failed] so the readouts can tell "still coming" from "never coming".
function drawTiles(lev, tilePx, url, target, w, h){
  const set = lev._set || (lev._set = new Set(lev.present));
  const [x0, y0] = toUm(0, 0), [x1, y1] = toUm(w, h);
  const c0 = Math.max(0, Math.floor(x0 / lev.mpp / tilePx));
  const c1 = Math.min(lev.cols - 1, Math.floor(x1 / lev.mpp / tilePx));
  const r0 = Math.max(0, Math.floor(y0 / lev.mpp / tilePx));
  const r1 = Math.min(lev.rows - 1, Math.floor(y1 / lev.mpp / tilePx));
  const bleed = 0.5 / B.scale;
  let nt = 0, miss = 0, bad = 0;
  target.imageSmoothingEnabled = true;
  for(let r = r0; r <= r1; r++) for(let c = c0; c <= c1; c++){
    if(!set.has(c + "_" + r)) continue;
    const im = imgFor(url(c, r));
    if(im === false){ bad++; continue; }
    if(!im){ miss++; continue; }
    const ux = c * tilePx * lev.mpp, uy = r * tilePx * lev.mpp;
    const [sx, sy] = toScreen(ux - bleed, uy - bleed);
    target.drawImage(im, sx, sy, (im.width * lev.mpp + 2 * bleed) * B.scale,
                     (im.height * lev.mpp + 2 * bleed) * B.scale);
    nt++;
  }
  return [nt, miss, bad];
}
function pickFrom(levels, umPerPx){
  let best = levels[0];
  for(const l of levels) if(l.mpp <= umPerPx * 1.0001) best = l;
  return best;
}

function renderCodex(w, h){
  const L = layerOf(B.section, "codex");
  ctx.fillStyle = "#000";                    // fluorescence background is black
  ctx.fillRect(0, 0, w, h);
  let nt = 0, miss = 0, bad = 0, drawn = 0;
  const [oc, ox] = offscreen(cvB.width, cvB.height);
  for(let slot = 0; slot < 3; slot++){
    const ci = B.codex[slot];
    if(ci === null || ci === undefined) continue;
    const ch = L.channels[ci];
    if(!ch) continue;
    const lev = pickFrom(ch.levels, 1 / B.scale);
    ox.setTransform(1, 0, 0, 1, 0, 0);
    // Black, not transparent. The colouring step below is a multiply over the whole
    // offscreen, and `multiply` against a TRANSPARENT destination leaves the source
    // colour standing -- so every pixel outside the tiles came out fully saturated,
    // and three slots summed to white across the entire margin. An opaque black
    // ground makes the multiply zero there, and zero adds nothing under `lighter`.
    ox.globalCompositeOperation = "source-over";
    ox.fillStyle = "#000";
    ox.fillRect(0, 0, oc.width, oc.height);
    ox.setTransform(dpr, 0, 0, dpr, 0, 0);
    const st = drawTiles(lev, L.tile_px,
                         (c, r) => SECT.dirs.codex + "/" + L.slide + "/" + ci + "/"
                                   + lev.dir + "/" + c + "_" + r + ".webp",
                         ox, w, h);
    nt += st[0]; miss += st[1]; bad += st[2];
    if(!st[0]) continue;
    // the tile is a grey image; multiply it down to this slot's primary, then add
    // it to the picture. Three single-channel images become one RGB composite
    // without ever needing a shader.
    ox.globalCompositeOperation = "multiply";
    ox.fillStyle = ["#f00", "#0f0", "#00f"][slot];
    ox.fillRect(0, 0, w, h);
    ox.globalCompositeOperation = "source-over";
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = "lighter";
    ctx.drawImage(oc, 0, 0);
    ctx.restore();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawn++;
  }
  ctx.globalCompositeOperation = "source-over";
  return {tiles: nt, missing: miss, failed: bad, channels: drawn};
}

const VIS = {};        // slide -> spot table | null in flight | false failed
function visiumSpots(slide){
  if(VIS[slide] !== undefined) return VIS[slide] || null;
  VIS[slide] = null;
  fetch(SECT.dirs.visium + "/" + slide + "/spots.json")
    .then(r => { if(!r.ok) throw new Error(r.status); return r.json(); })
    .then(j => { VIS[slide] = j; need(); })
    .catch(e => { VIS[slide] = false; console.error(e); need(); });
  return null;
}
// A perceptually ordered ramp for continuous values and a categorical set for
// clusters. Cluster ids are labels, not magnitudes, so they never get the ramp.
const RAMP = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"];
const CATS = ["#8C8C8C", "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
              "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990", "#DCBEFF"];
function rampAt(t){
  t = Math.max(0, Math.min(1, t));
  const x = t * (RAMP.length - 1), i = Math.min(RAMP.length - 2, Math.floor(x));
  return RAMP[x - i > 0.5 ? i + 1 : i];
}
function renderVisium(w, h){
  const L = layerOf(B.section, "visium");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  const lev = pickFrom(L.levels, 1 / B.scale);
  const st = drawTiles(lev, L.tile_px,
                       (c, r) => SECT.dirs.visium + "/" + L.slide + "/" + lev.dir
                                 + "/" + c + "_" + r + ".webp",
                       ctx, w, h);
  let n = 0;
  const S = visiumSpots(L.slide);
  if(S && B.spots){
    // GEOMETRY ONLY. The spots are drawn as one uniform outline to show where the
    // capture array sat -- no fill, one colour, no value of any kind mapped to
    // colour. Nothing here can be read as a measurement, which is the point: a
    // 55 um spot pools every cell under it, and a coloured spot invites being read
    // as a property of a cell.
    const r = Math.max(1.2, (S.spot_um / 2) * B.scale);
    ctx.strokeStyle = "rgba(20,90,140,.75)";
    ctx.lineWidth = Math.max(0.6, Math.min(1.6, r * 0.22));
    ctx.beginPath();
    for(let i = 0; i < S.spots.x_um.length; i++){
      if(B.inTissueOnly && !S.spots.in_tissue[i]) continue;
      const sx = (S.spots.x_um[i] - B.cx) * B.scale + w / 2;
      const sy = (S.spots.y_um[i] - B.cy) * B.scale + h / 2;
      if(sx < -r || sx > w + r || sy < -r || sy > h + r) continue;
      ctx.moveTo(sx + r, sy); ctx.arc(sx, sy, r, 0, 6.28318530718);
      n++;
    }
    ctx.stroke();
  }
  return {tiles: st[0], missing: st[1], failed: st[2], spots: n, loaded: !!S};
}

const XEN = {};       // slide -> {x,y,cnt,area,n} | null in flight | false failed
function xeniumCells(slide){
  if(XEN[slide] !== undefined) return XEN[slide] || null;
  XEN[slide] = null;
  fetch(SECT.dirs.xenium + "/" + slide + "/cells.bin")
    .then(r => { if(!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
    .then(buf => {
      const dv = new DataView(buf);
      const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1),
                                        dv.getUint8(2), dv.getUint8(3));
      if(magic !== "XEC1") throw new Error("xenium " + slide + ": magic " + magic);
      const n = dv.getUint32(4, true), q = dv.getFloat32(12, true);
      if(!(q > 0)) throw new Error("xenium " + slide + ": quantisation " + q);
      const want = 32 + n * 8;
      if(buf.byteLength !== want)
        throw new Error("xenium " + slide + ": " + buf.byteLength + " bytes, want " + want);
      const xq = new Uint16Array(buf, 32, n), yq = new Uint16Array(buf, 32 + 2 * n, n);
      const cnt = new Uint16Array(buf, 32 + 4 * n, n);
      const area = new Uint16Array(buf, 32 + 6 * n, n);
      const x = new Float32Array(n), y = new Float32Array(n);
      for(let i = 0; i < n; i++){ x[i] = xq[i] * q; y[i] = yq[i] * q; }
      XEN[slide] = {x: x, y: y, cnt: cnt, area: area, n: n,
                    cntMax: dv.getUint32(24, true)};
      need();
    })
    .catch(e => { XEN[slide] = false; console.error(e); need(); });
  return null;
}
function renderXenium(w, h){
  const L = layerOf(B.section, "xenium");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  const lev = pickFrom(L.levels, 1 / B.scale);
  const st = drawTiles(lev, L.tile_px,
                       (c, r) => SECT.dirs.xenium + "/" + L.slide + "/" + lev.dir
                                 + "/" + c + "_" + r + ".webp",
                       ctx, w, h);
  let n = 0;
  const C = B.xcells ? xeniumCells(L.slide) : null;
  if(C){
    // These cells are MEASURED, not predicted. Colour is transcript count on a
    // perceptual ramp, scaled to this section's own 99th percentile so one bright
    // outlier cannot flatten the rest.
    let hi = 1;
    const sample = Math.max(1, Math.floor(C.n / 20000));
    const vals = [];
    for(let i = 0; i < C.n; i += sample) vals.push(C.cnt[i]);
    vals.sort((p, q2) => p - q2);
    hi = vals[Math.floor(vals.length * 0.99)] || 1;
    const r = Math.max(1.0, 4.0 * B.scale);
    // Colour is a transcript count, which -- unlike a Visium spot -- is a property
    // of THIS cell and of nothing else, so it cannot be misread as a pooled value.
    // It is still switchable: off draws the same cells in one colour, which is the
    // picture to look at when the question is where the cells are, not how busy.
    const buckets = B.xcolour ? RAMP.length : 1;
    for(let k = 0; k < buckets; k++){
      ctx.fillStyle = B.xcolour ? RAMP[k] : "#7ad151";
      ctx.beginPath();
      let inPath = 0;
      for(let i = 0; i < C.n; i++){
        if(B.xcolour){
          const t = Math.min(0.999, C.cnt[i] / hi);
          if(Math.floor(t * buckets) !== k) continue;
        }
        const sx = (C.x[i] - B.cx) * B.scale + w / 2;
        if(sx < -r || sx > w + r) continue;
        const sy = (C.y[i] - B.cy) * B.scale + h / 2;
        if(sy < -r || sy > h + r) continue;
        if(r <= 1.6) ctx.rect(sx - r, sy - r, 2 * r, 2 * r);
        else { ctx.moveTo(sx + r, sy); ctx.arc(sx, sy, r, 0, 6.28318530718); }
        n++;
        if(++inPath >= 2000){ ctx.fill(); ctx.beginPath(); inPath = 0; }
      }
      ctx.fill();
    }
  }
  return {tiles: st[0], missing: st[1], failed: st[2], cells: n,
          loaded: !!C, hi: L.n_cells};
}

// ------------------------------------------------------------------ drawing
let lastStat = {tiles: 0, missing: 0, cells: 0, level: null};
function render(){
  if(!B.slide) return;
  const [w, h] = stageSize();
  const W = Math.max(1, Math.round(w * dpr)), H = Math.max(1, Math.round(h * dpr));
  if(cvB.width !== W || cvB.height !== H){ cvB.width = W; cvB.height = H; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  clampCam();
  // Everything above is shared; below the H&E path is exactly what it was. A layer
  // that is not H&E paints its own background, because "no tile here" means blank
  // glass in brightfield and black in fluorescence, and painting white under a
  // CODEX composite would turn every gap into signal.
  if(B.layer !== "he"){
    const s = (B.layer === "codex") ? renderCodex(w, h)
            : (B.layer === "xenium") ? renderXenium(w, h)
            : renderVisium(w, h);
    lastStat = Object.assign({cells: 0, level: null, umPerPx: 1 / B.scale}, s);
    evictImages();
    paintReadouts();
    return;
  }
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  const umPerPx = 1 / B.scale;
  const lev = pickLevel(umPerPx);
  let nt = 0, miss = 0, bad = 0;
  if(lev){
    const T = slideInfo(B.slide).tiles.tile_px;
    const have = lev.present;
    const set = lev._set || (lev._set = new Set(have));
    const [x0, y0] = toUm(0, 0), [x1, y1] = toUm(w, h);
    const c0 = Math.max(0, Math.floor(x0 / lev.mpp / T));
    const c1 = Math.min(lev.cols - 1, Math.floor(x1 / lev.mpp / T));
    const r0 = Math.max(0, Math.floor(y0 / lev.mpp / T));
    const r1 = Math.min(lev.rows - 1, Math.floor(y1 / lev.mpp / T));
    // half a screen pixel of bleed, or the seams between tiles show as hairlines
    const bleed = 0.5 / B.scale;
    ctx.imageSmoothingEnabled = true;
    for(let r = r0; r <= r1; r++) for(let c = c0; c <= c1; c++){
      if(!set.has(c + "_" + r)) continue;    // blank glass: never written
      const im = imgFor(tileUrl(B.slide, lev, c, r));
      // pending and failed are counted apart. A tile that will never arrive looks
      // exactly like a tile still on the way if they share a number, and a page
      // that says "loading" forever is how a 404 on every tile went unnoticed.
      if(im === false){ bad++; continue; }
      if(!im){ miss++; continue; }
      const ux = c * T * lev.mpp, uy = r * T * lev.mpp;
      const uw = im.width * lev.mpp, uh = im.height * lev.mpp;
      const [sx, sy] = toScreen(ux - bleed, uy - bleed);
      ctx.drawImage(im, sx, sy, (uw + 2 * bleed) * B.scale,
                    (uh + 2 * bleed) * B.scale);
      nt++;
    }
  }
  let ncell = 0;
  if(B.cells){
    const C = cellsFor(B.slide);
    if(C){
      const r = Math.max(MIN_R_PX, CELL_R_UM * B.scale);
      ncell = (r <= MARK_R_PX) ? cellsAsPixels(C, w, h, r)
                               : cellsAsDiscs(C, w, h, r);
    }
  }
  if(B.regions) drawRegions(w, h);
  lastStat = {tiles: nt, missing: miss, failed: bad, cells: ncell,
              level: lev ? lev.mpp : null, umPerPx: umPerPx};
  evictImages();
  paintReadouts();
}
// Below this on-screen radius a cell is a MARK, not a shape, and it is written
// straight into the pixels. A section here carries up to 180 000 cells; asking the
// 2D rasteriser for 180 000 sub-paths in one fill is not slow, it is fatal -- the
// renderer process dies before the frame lands. Writing the squares by hand is O(n)
// with no path at all, and at one or two pixels across the two are the same picture.
const MARK_R_PX = 2.2;
function classRGB(){
  const out = [];
  for(const c of CLS){
    const h = COL[c].replace("#", "");
    out.push([parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16),
              parseInt(h.slice(4, 6), 16)]);
  }
  return out;
}
const RGB = classRGB();
function onMask(){
  const m = new Uint8Array(CLS.length);
  B.on.forEach(c => { const i = CLS.indexOf(c); if(i >= 0) m[i] = 1; });
  return m;
}
function cellsAsPixels(C, w, h, r){
  const W = cvB.width, H = cvB.height;
  const img = ctx.getImageData(0, 0, W, H), d = img.data;
  const on = onMask();
  const side = Math.max(1, Math.round(2 * r * dpr));
  const half = (side - 1) / 2;
  let n = 0;
  for(let i = 0; i < C.n; i++){
    const k = C.cls[i];
    if(!on[k]) continue;
    const sx = ((C.x[i] - B.cx) * B.scale + w / 2) * dpr;
    const sy = ((C.y[i] - B.cy) * B.scale + h / 2) * dpr;
    let x0 = Math.round(sx - half), y0 = Math.round(sy - half);
    if(x0 + side <= 0 || y0 + side <= 0 || x0 >= W || y0 >= H) continue;
    const c0 = RGB[k];
    const x1 = Math.min(W, x0 + side), y1 = Math.min(H, y0 + side);
    if(x0 < 0) x0 = 0;
    if(y0 < 0) y0 = 0;
    for(let y = y0; y < y1; y++){
      let o = (y * W + x0) * 4;
      for(let x = x0; x < x1; x++){
        d[o] = c0[0]; d[o + 1] = c0[1]; d[o + 2] = c0[2]; d[o + 3] = 255;
        o += 4;
      }
    }
    n++;
  }
  ctx.putImageData(img, 0, 0);
  return n;
}
function cellsAsDiscs(C, w, h, r){
  const on = onMask();
  const box = r + 2;
  let total = 0;
  for(let k = 0; k < CLS.length; k++){
    if(!on[k]) continue;
    ctx.fillStyle = COL[CLS[k]];
    ctx.beginPath();
    let inPath = 0;
    for(let i = 0; i < C.n; i++){
      if(C.cls[i] !== k) continue;
      const sx = (C.x[i] - B.cx) * B.scale + w / 2;
      if(sx < -box || sx > w + box) continue;
      const sy = (C.y[i] - B.cy) * B.scale + h / 2;
      if(sy < -box || sy > h + box) continue;
      ctx.moveTo(sx + r, sy); ctx.arc(sx, sy, r, 0, 6.28318530718);
      total++;
      // flushed in batches for the same reason the branch above exists: one path
      // is allowed to be big, not unbounded
      if(++inPath >= 2000){ ctx.fill(); ctx.beginPath(); inPath = 0; }
    }
    ctx.fill();
  }
  return total;
}
function drawRegions(w, h){
  const s = slideInfo(B.slide);
  if(!s.regions || !s.regions.length) return;
  ctx.lineWidth = 2;
  ctx.strokeStyle = COL["Schwann"];
  ctx.font = "11px ui-monospace,Consolas,monospace";
  for(const g of s.regions){
    // the cluster table gives an area, not an outline, so what is drawn is a ring
    // of that area at that centroid -- a marker for "a region was called here",
    // never a claim about its shape
    const rum = Math.sqrt(Math.max(g.area_mm2, 1e-6) * 1e6 / Math.PI);
    const [sx, sy] = toScreen(g.x, g.y);
    const r = Math.max(9, rum * B.scale);
    if(sx < -r - 40 || sx > w + r + 40 || sy < -r - 40 || sy > h + r + 40) continue;
    ctx.beginPath(); ctx.arc(sx, sy, r, 0, 6.28318530718); ctx.stroke();
    const lab = g.n + " Schwann - " + (g.status || "region");
    const tw = ctx.measureText(lab).width;
    ctx.fillStyle = "rgba(18,22,26,.82)";
    ctx.fillRect(sx - tw / 2 - 4, sy - r - 17, tw + 8, 14);
    ctx.fillStyle = "#EAF0F2";
    ctx.textAlign = "center"; ctx.fillText(lab, sx, sy - r - 6);
    ctx.textAlign = "left";
  }
}
function paintReadouts(){
  const s = slideInfo(B.slide);
  const C = CELLS.get(B.slide);
  const L = lastStat;
  const um = L.umPerPx;
  elCorner.textContent =
    B.slide + "\n" +
    "block " + blockName(s) + "  -  section " + (B.idx + 1) + " of " + B.list.length + "\n" +
    (L.level ? ("tiles at " + L.level.toFixed(2) + " um/px, screen "
                + um.toFixed(2) + " um/px") : "no tile pyramid built") +
    (L.missing ? ("  (" + L.missing + " loading)") : "")
    + (L.failed ? ("  (" + L.failed + " TILES FAILED)") : "") + "\n" +
    (C === false ? "cell file failed to load"
     : C ? (L.cells.toLocaleString() + " of " + C.n.toLocaleString() + " cells drawn")
         : "cells loading");
  // a bar of a round number of microns, measured on this frame's scale
  const targetPx = 120;
  const nice = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  let barUm = nice[nice.length - 1];
  for(const v of nice) if(v * B.scale <= targetPx){ barUm = v; }
  elScale.innerHTML = "";
  const bar = document.createElement("i");
  bar.style.width = Math.round(barUm * B.scale) + "px";
  elScale.appendChild(bar);
  const lab = document.createElement("span");
  lab.textContent = barUm >= 1000 ? (barUm / 1000) + " mm" : barUm + " µm";
  elScale.appendChild(lab);
}

// ------------------------------------------------------------------ panels
// How this sample's sections are grouped. Named blocks and the no-block group are
// counted separately: "2 blocks" for a sample with one named block and three
// sections whose block was never recorded would be a claim about the tissue that
// the manifest does not make.
function blocksPhrase(s){
  const named = s.n_blocks_named;
  const un = s.blocks.length - named;
  const a = named + " block" + (named === 1 ? "" : "s");
  return un ? (named ? a + " + unknown" : "block unknown") : a;
}
function facts(s){
  return s.n_slides + " section" + (s.n_slides === 1 ? "" : "s")
    + " · " + s.n_nerve + " with nerve regions"
    + " · " + blocksPhrase(s);
}
let cursor = 0, shown = [];
function buildSampleList(){
  list.innerHTML = "";
  const q = filt.value.trim().toLowerCase();
  shown = BR.samples.filter(s => !q || s.id.toLowerCase().indexOf(q) >= 0
                                 || s.blocks.some(b => b.id.toLowerCase().indexOf(q) >= 0));
  shown.forEach((s, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "srow" + (i === cursor ? " cursor" : "");
    b.setAttribute("role", "option");
    b.setAttribute("data-sample", s.id);
    b.setAttribute("aria-selected", s.id === B.sample ? "true" : "false");
    const id = document.createElement("span"); id.className = "sid"; id.textContent = s.id;
    const nm = document.createElement("span"); nm.className = "sn"; nm.textContent = facts(s);
    const kd = document.createElement("span"); kd.className = "skind";
    kd.setAttribute("data-mode", s.mode);
    kd.textContent = s.mode === "volume" ? "3D volume" : "single sections";
    b.appendChild(id); b.appendChild(nm); b.appendChild(kd);
    b.addEventListener("click", () => chooseSample(s));
    list.appendChild(b);
  });
  foot.textContent = ""; foot.hidden = true;      // controls only; no explanatory text
}
function chooseSample(s){
  if(s.page){
    carrySave();
    if(window.__NAVIGATE_SAMPLE__) window.__NAVIGATE_SAMPLE__(s.id);
    else location.href = s.page + (document.fullscreenElement ? "#fs" : "");
  } else { setSample(s.id); openPop(false); }
}
function openPop(v){
  pop.setAttribute("data-open", v ? "1" : "0");
  btn.setAttribute("aria-expanded", v ? "true" : "false");
  if(v){ cursor = Math.max(0, shown.map(s => s.id).indexOf(B.sample));
         buildSampleList(); filt.focus(); filt.select();
         const cur = list.querySelector(".cursor");
         if(cur && cur.scrollIntoView) cur.scrollIntoView({block: "nearest"}); }
}
function paintButton(){
  const s = sampleInfo(B.sample);
  // the NAME only. The counts are what the reader opens the control to see; on
  // the button they push everything else out of a 34 px bar and, in full screen,
  // out of reach.
  btn.textContent = s.id;
  btn.title = s.id + " — " + facts(s) + (s.mode === "volume"
    ? " — registered 3D volume" : " — single sections, not registered");
}
function sampleSections(id){
  if(!SECT) return null;
  return SECT.samples.filter(s => s.id === id)[0] || null;
}
// The section list, when the section index is present: one chip per SECTION rather
// than per H&E slide, so the sections that carry no H&E at all -- every CODEX,
// Visium and Xenium section of HT206B1 -- are reachable. Each chip says which
// assays that section carries.
function buildSectionsFromIndex(S){
  elSecs.innerHTML = "";
  for(const blk of S.blocks){
    const d = document.createElement("div");
    d.className = "bblock" + (blk.unknown ? " unknownblock" : "");
    const hd = document.createElement("div"); hd.className = "bh";
    const nm = document.createElement("b");
    nm.textContent = blk.unknown ? "unknown" : blk.id;
    hd.appendChild(document.createTextNode("block "));
    hd.appendChild(nm);
    hd.appendChild(document.createTextNode(" · " + blk.sections.length
      + " section" + (blk.sections.length === 1 ? "" : "s")));
    d.appendChild(hd);
    if(blk.unknown){
      const n = document.createElement("div");
      n.className = "note";
      n.textContent = "These sections have no block recorded in the run manifest. "
        + "They are kept apart rather than folded into a named block, because which "
        + "block they came from is not known.";
      d.appendChild(n);
    }
    const row = document.createElement("div"); row.className = "bsecs";
    for(const sid of blk.sections){
      const sec = sectionInfo(sid);
      const keys = MOD_ORDER.filter(k => sec.layers[k]);
      const he = sec.layers.he;
      const b = document.createElement("button");
      b.className = "bsec" + (he && he.has_nerve ? " nerve" : "")
        + (keys.length > 1 ? " multi" : "");
      b.type = "button";
      b.textContent = "U" + sec.u;
      b.setAttribute("data-section", sid);
      b.title = sid + " — " + keys.map(k => MOD_LABEL[k]).join(" + ")
        + (he && he.has_nerve ? " (nerve regions found)" : "");
      b.setAttribute("aria-pressed", sid === B.section ? "true" : "false");
      b.addEventListener("click", () => pickSection(sid));
      const tag = document.createElement("i");
      tag.className = "modtag";
      tag.textContent = keys.map(k => MOD_LABEL[k][0]).join("");
      b.appendChild(tag);
      row.appendChild(b);
    }
    d.appendChild(row);
    elSecs.appendChild(d);
  }
  const multi = S.n_multi;
  const n = document.createElement("div");
  n.className = "note";
  n.textContent = "Chips are labelled by section number; the letters name the assays "
    + "on that section (H = H&E, C = CODEX, X = Xenium, V = Visium)."
    + (multi ? (" " + multi + " of these sections carry more than one assay — the "
        + "same piece of tissue imaged more than one way.") : "")
    + (S.blocks.length > 1 ? " Sections from different blocks are different pieces "
        + "of tissue and do not form one series, so previous/next stays inside a "
        + "block." : "");
  elSecs.appendChild(n);
}
function pickSection(sid){
  const sec = sectionInfo(sid);
  if(!sec) return;
  B.section = sid;
  const blk = (sampleSections(B.sample).blocks
               .filter(b => b.sections.indexOf(sid) >= 0)[0]) || {sections: [sid]};
  B.list = blk.sections; B.idx = blk.sections.indexOf(sid);
  Array.prototype.forEach.call(elSecs.querySelectorAll(".bsec"), b =>
    b.setAttribute("aria-pressed",
                   b.getAttribute("data-section") === sid ? "true" : "false"));
  const first = MOD_ORDER.filter(k => sec.layers[k])[0];
  B.layer = first;
  if(first === "he"){ B.slide = sec.layers.he.slide; buildClasses();
                      cellsFor(B.slide); }
  if(first === "codex"){ B.codex = [sec.layers.codex.dapi_index, null, null];
                         buildCodexPick(); }
  if(first === "visium"){ visiumSpots(sec.layers.visium.slide); buildVisValue(); }
  if(first === "xenium"){ xeniumCells(sec.layers.xenium.slide); }
  paintLayers(); paintInfo(); fit(); need();
}
function buildSections(){
  const idx = sampleSections(B.sample);
  if(idx) return buildSectionsFromIndex(idx);
  const s = sampleInfo(B.sample);
  elSecs.innerHTML = "";
  for(const blk of s.blocks){
    const d = document.createElement("div");
    d.className = "bblock" + (blk.unknown ? " unknownblock" : "");
    const hd = document.createElement("div"); hd.className = "bh";
    const nb = blk.slides.filter(x => slideInfo(x).has_nerve).length;
    const nm = document.createElement("b");
    nm.textContent = blk.unknown ? "unknown" : blk.id;
    hd.appendChild(document.createTextNode("block "));
    hd.appendChild(nm);
    hd.appendChild(document.createTextNode(" · " + blk.slides.length
      + " section" + (blk.slides.length === 1 ? "" : "s") + " · " + nb + " nerve"));
    d.appendChild(hd);
    if(blk.unknown){
      // said once, next to the sections it applies to. A group headed "(NONE)" reads
      // as a block with a strange name; what is true is that the run manifest has no
      // block for these, and putting them in a neighbouring block would assert a
      // tissue relationship nobody recorded.
      const n = document.createElement("div");
      n.className = "note";
      n.textContent = "These sections have no block recorded in the run manifest. "
        + "They are kept apart rather than folded into a named block, because which "
        + "block they came from is not known.";
      d.appendChild(n);
    }
    const row = document.createElement("div"); row.className = "bsecs";
    for(const sl of blk.slides){
      const b = document.createElement("button");
      b.className = "bsec" + (slideInfo(sl).has_nerve ? " nerve" : "");
      b.type = "button";
      b.textContent = shortName(sl, B.sample);
      b.title = sl + (slideInfo(sl).has_nerve ? " (nerve regions found)" : "");
      b.setAttribute("aria-pressed", sl === B.slide ? "true" : "false");
      b.addEventListener("click", () => setSlide(sl));
      row.appendChild(b);
    }
    d.appendChild(row);
    elSecs.appendChild(d);
  }
  if(s.blocks.length > 1){
    const n = document.createElement("div");
    n.className = "note";
    n.textContent = "This sample is " + blocksPhrase(s) + ". Sections from different "
      + "blocks are different pieces of tissue and do not form one series, so they "
      + "are listed apart and previous/next does not cross between them.";
    elSecs.appendChild(n);
  }
}
function shortName(slide, sample){
  let s = slide;
  if(s.indexOf(sample) === 0) s = s.slice(sample.length).replace(/^[-_]/, "");
  const m = s.match(/[Uu]s?\d*[-_]?(\d+)$/);
  return m ? ("U" + m[1]) : (s || slide);
}
function buildClasses(){
  elClasses.innerHTML = "";
  const s = slideInfo(B.slide);
  for(const c of CLS){
    const b = document.createElement("button");
    b.className = "crow"; b.type = "button";
    b.setAttribute("aria-pressed", B.on.indexOf(c) >= 0 ? "true" : "false");
    const sw = document.createElement("span");
    sw.className = "csw"; sw.style.background = COL[c];
    const nm = document.createElement("span");
    nm.className = "cname"; nm.textContent = c;
    const ct = document.createElement("span");
    ct.className = "ccount";
    ct.textContent = (s.counts[c] || 0).toLocaleString();
    b.appendChild(sw); b.appendChild(nm); b.appendChild(ct);
    b.addEventListener("click", () => {
      const i = B.on.indexOf(c);
      if(i >= 0) B.on.splice(i, 1); else B.on.push(c);
      b.setAttribute("aria-pressed", B.on.indexOf(c) >= 0 ? "true" : "false");
      need();
    });
    elClasses.appendChild(b);
  }
}
function paintInfo(){
  const s = slideInfo(B.slide);
  const nreg = s.regions ? s.regions.length : 0;
  elInfo.textContent =
    s.extent_um.map(v => (v / 1000).toFixed(2)).join(" x ") + " mm at "
    + s.mpp.toFixed(4) + " um/px scanned\n"
    + s.n_cells.toLocaleString() + " typed cells\n"
    + (nreg ? (nreg + " nerve region" + (nreg === 1 ? "" : "s") + " called")
            : "no nerve regions called")
    + (s.tumor_fraction !== null && s.tumor_fraction !== undefined
        ? ("\ntumour " + (s.tumor_fraction * 100).toFixed(1) + "% of tissue area")
        : "")
    + (s.restained ? "\nre-stained section" : "");
  elWarn.textContent = s.tiles ? "" : "No tile pyramid was built for this section, "
    + "so only the prediction is drawn.";
  elChips.innerHTML = "";
  const chip = document.createElement("div");
  chip.className = "chip";
  chip.textContent = "single section · " + B.sample + " · block " + blockName(s);
  elChips.appendChild(chip);
  elCur.textContent = "section " + (B.idx + 1) + " of " + B.list.length
    + "  ·  " + B.slide;
}

function paintInfoLayer(){
  const sec = sectionInfo(B.section), L = layerNow();
  const ex = extentNow();
  const bits = [MOD_LABEL[B.layer] + " — " + (L.slide || ""),
                ex.map(v => (v / 1000).toFixed(2)).join(" x ") + " mm"];
  if(B.layer === "codex")
    bits.push(L.n_channels + " channels at " + L.native_mpp.toFixed(4) + " um/px scanned");
  if(B.layer === "visium")
    bits.push(L.n_in_tissue + " of " + L.n_spots + " spots in tissue",
              "a spot is 55 um across and pools every cell under it");
  if(B.layer === "xenium")
    bits.push((L.n_cells || 0).toLocaleString() + " cells measured");
  elInfo.textContent = bits.join("\n");
  elWarn.textContent = "";
  elChips.innerHTML = "";
  const chip = document.createElement("div");
  chip.className = "chip";
  chip.textContent = "single section · " + B.sample + " · block "
    + (sec.block_unknown ? "unknown" : sec.block) + " · " + MOD_LABEL[B.layer];
  elChips.appendChild(chip);
  elCur.textContent = "section " + (B.idx + 1) + " of " + B.list.length
    + "  ·  " + B.section + "  ·  " + MOD_LABEL[B.layer];
}
// ------------------------------------------------------------------ switching
function blockOf(slide){
  const s = sampleInfo(B.sample);
  for(const b of s.blocks) if(b.slides.indexOf(slide) >= 0) return b;
  return s.blocks[0];
}
function setSlide(slide){
  B.slide = slide;
  const blk = blockOf(slide);
  B.list = blk.slides; B.idx = blk.slides.indexOf(slide);
  Array.prototype.forEach.call(elSecs.querySelectorAll(".bsec"), b => {
    b.setAttribute("aria-pressed", b.title.split(" ")[0] === slide ? "true" : "false");
  });
  if(SECT){
    // find the section this H&E slide belongs to, so the layer controls follow
    const hit = Object.keys(SECT.sections).filter(k =>
      (SECT.sections[k].layers.he || {}).slide === slide)[0];
    if(hit) B.section = hit;
  }
  B.layer = "he";
  paintLayers();
  buildClasses(); paintInfo(); cellsFor(slide); fit(); need();
}
// Which assays this section carries, as a pick-one control. It is in the DOM only
// when there is more than one -- a control with a single option is noise, and 183 of
// the 184 sections have exactly one.
function paintLayers(){
  const box = document.getElementById("blayerBox");
  const seg = document.getElementById("blayers");
  const note = document.getElementById("blayerNote");
  const s = sectionInfo(B.section);
  const keys = s ? MOD_ORDER.filter(k => s.layers[k]) : [];
  document.getElementById("bcodexBox").style.display =
    (B.layer === "codex") ? "" : "none";
  document.getElementById("bvisiumBox").style.display =
    (B.layer === "visium") ? "" : "none";
  document.getElementById("bxeniumBox").style.display =
    (B.layer === "xenium") ? "" : "none";
  document.getElementById("bcellBox").style.display =
    (B.layer === "he") ? "" : "none";
  if(!s || keys.length < 2){ box.style.display = "none"; return; }
  box.style.display = "";
  seg.innerHTML = "";
  for(const k of keys){
    const b = document.createElement("button");
    b.type = "button"; b.textContent = MOD_LABEL[k] || k;
    b.setAttribute("data-layer", k);
    b.setAttribute("aria-pressed", k === B.layer ? "true" : "false");
    b.addEventListener("click", () => setLayer(k));
    seg.appendChild(b);
  }
  note.textContent = "This one piece of tissue was imaged " + keys.length
    + " ways. Switching between them is not a registration: they are separate "
    + "images of the same section, each in its own frame, and nothing here has "
    + "aligned one to another.";
}
function setLayer(k){
  const s = sectionInfo(B.section);
  if(!s || !s.layers[k]) return;
  B.layer = k;
  if(k === "he") B.slide = s.layers.he.slide;
  if(k === "codex"){
    const L = s.layers.codex;
    B.codex = [L.dapi_index, null, null];
    buildCodexPick();
  }
  if(k === "visium"){ visiumSpots(s.layers.visium.slide); buildVisValue(); }
  if(k === "xenium"){
    xeniumCells(s.layers.xenium.slide);
    document.getElementById("bxenNote").textContent =
      (s.layers.xenium.n_cells || 0).toLocaleString() + " cells";
  }
  paintLayers(); paintInfo(); fit(); need();
}
// One row per channel, three assignment buttons on each. Thirty-seven markers do not
// fit a dropdown, and the reader's question is "which slot do I put CD8 in", which
// this answers in one click. A channel can hold at most one slot, and clicking the
// slot it already holds clears it.
function buildCodexPick(){
  const L = layerOf(B.section, "codex");
  const box = document.getElementById("bcodexPick");
  box.innerHTML = "";
  if(!L) return;
  const list = document.createElement("div");
  list.id = "bchanList";
  for(const c of L.channels){
    const row = document.createElement("div");
    row.className = "chanrow";
    const nm = document.createElement("span");
    nm.className = "channame"; nm.textContent = c.name;
    nm.title = c.name + " — window " + c.window[0] + " to " + c.window[1]
      + " (p50 to p99.5 inside the tissue)";
    const win = document.createElement("span");
    win.className = "chanwin"; win.textContent = c.window[0] + "-" + c.window[1];
    row.appendChild(nm); row.appendChild(win);
    for(let slot = 0; slot < 3; slot++){
      const b = document.createElement("button");
      b.type = "button"; b.className = "chanslot s" + slot;
      b.textContent = "RGB"[slot];
      b.setAttribute("data-slot", String(slot));
      b.setAttribute("data-channel", String(c.index));
      b.setAttribute("aria-pressed", B.codex[slot] === c.index ? "true" : "false");
      b.setAttribute("aria-label", c.name + " in " + CODEX_SLOT_NAME[slot]);
      b.addEventListener("click", () => {
        if(B.codex[slot] === c.index){ B.codex[slot] = null; }
        else {
          for(let k = 0; k < 3; k++) if(B.codex[k] === c.index) B.codex[k] = null;
          B.codex[slot] = c.index;
        }
        buildCodexPick(); need();
      });
      row.appendChild(b);
    }
    list.appendChild(row);
  }
  box.appendChild(list);
  const n = B.codex.filter(x => x !== null && x !== undefined).length;
  const note = document.createElement("div");
  note.className = "note";
  note.textContent = n ? (n + " of 3 slots in use: "
    + B.codex.map((ci, s) => ci === null || ci === undefined ? null
        : CODEX_SLOT_NAME[s] + " = " + L.channels[ci].name)
      .filter(Boolean).join(", "))
    : "No channel selected, so the picture is black.";
  box.appendChild(note);
}
function buildVisValue(){
  const L = layerOf(B.section, "visium");
  if(!L) return;
  document.getElementById("bvisNote").textContent =
    L.n_in_tissue + " of " + L.n_spots + " spots in tissue";
}
function step(d){
  const i = B.idx + d;
  if(i < 0 || i >= B.list.length) return;
  setSlide(B.list[i]);
}
function setSample(id){
  const s = sampleInfo(id);
  if(!s) return;
  B.sample = id;
  paintButton();
  Array.prototype.forEach.call(list.querySelectorAll(".srow"), r =>
    r.setAttribute("aria-selected",
                   r.getAttribute("data-sample") === id ? "true" : "false"));
  if(s.mode === "volume"){
    app.setAttribute("data-view", "volume");
    tag.setAttribute("data-mode", "volume");
    tag.textContent = "registered 3D volume · " + s.n_slides + " sections" + (window.__BUILD_STAMP__ ? " · built " + window.__BUILD_STAMP__ : "");
    tag.title = "these sections were registered to one another, so the stack is a "
      + "measured geometry";
    V.setBrowsing(false);
    return;
  }
  V.setBrowsing(true);
  app.setAttribute("data-view", "section");
  tag.setAttribute("data-mode", "section");
  tag.textContent = "single sections · not registered";
  tag.title = "these sections were never registered to one another; they are "
    + "browsed one at a time";
  elNotice.textContent = "SINGLE SECTION. These " + s.n_slides + " section"
    + (s.n_slides === 1 ? " was" : "s were") + " never registered to one another, "
    + "so there is no 3D stack for this sample - one section is shown at a time."
    + (s.blocks.length > 1 ? (" They are also split across more than one group ("
        + blocksPhrase(s) + ").") : "");
  buildSections();
  setSlide(s.blocks[0].slides[0]);
}

// ------------------------------------------------------------------ input
let drag = null;
cvB.addEventListener("pointerdown", e => {
  cvB.setPointerCapture(e.pointerId);
  drag = {x: e.clientX, y: e.clientY};
  cvB.classList.add("drag");
});
cvB.addEventListener("pointermove", e => {
  if(!drag) return;
  B.cx -= (e.clientX - drag.x) / B.scale;
  B.cy -= (e.clientY - drag.y) / B.scale;
  drag = {x: e.clientX, y: e.clientY};
  need();
});
function endDrag(e){ drag = null; cvB.classList.remove("drag");
  try { cvB.releasePointerCapture(e.pointerId); } catch(_){} }
cvB.addEventListener("pointerup", endDrag);
cvB.addEventListener("pointercancel", endDrag);
cvB.addEventListener("wheel", e => {
  e.preventDefault();
  const r = cvB.getBoundingClientRect();
  const [ux, uy] = toUm(e.clientX - r.left, e.clientY - r.top);
  const f = Math.pow(1.0015, -e.deltaY);
  const before = B.scale;
  B.scale *= f; clampCam();
  const k = B.scale / before;
  // keep the point under the cursor under the cursor
  B.cx = ux + (B.cx - ux) / k; B.cy = uy + (B.cy - uy) / k;
  need();
}, {passive: false});
document.getElementById("bprev").addEventListener("click", () => step(-1));
document.getElementById("bnext").addEventListener("click", () => step(1));
document.getElementById("bfit").addEventListener("click", () => { fit(); need(); });
function ovbtn(id, get, set){
  const b = document.getElementById(id);
  b.addEventListener("click", () => { set(!get());
    b.setAttribute("aria-pressed", get() ? "true" : "false"); need(); });
}
ovbtn("bcellsToggle", () => B.cells, v => { B.cells = v; });
ovbtn("bspotsToggle", () => B.spots, v => { B.spots = v; });
ovbtn("bintissue", () => B.inTissueOnly, v => { B.inTissueOnly = v; });
ovbtn("bxcellsToggle", () => B.xcells, v => { B.xcells = v; });
ovbtn("bxcolour", () => B.xcolour, v => { B.xcolour = v; });
ovbtn("bregionsToggle", () => B.regions, v => { B.regions = v; });
btn.addEventListener("click", () => openPop(pop.getAttribute("data-open") !== "1"));
filt.addEventListener("input", () => { cursor = 0; buildSampleList(); });
pop.addEventListener("keydown", e => {
  if(e.key === "Escape"){ openPop(false); btn.focus(); e.preventDefault(); }
  else if(e.key === "ArrowDown" || e.key === "ArrowUp"){
    cursor = Math.max(0, Math.min(shown.length - 1,
                                  cursor + (e.key === "ArrowDown" ? 1 : -1)));
    buildSampleList();
    const cur = list.querySelector(".cursor");
    if(cur && cur.scrollIntoView) cur.scrollIntoView({block: "nearest"});
    e.preventDefault();
  } else if(e.key === "Enter" && shown[cursor]){
    chooseSample(shown[cursor]); btn.focus(); e.preventDefault();
  }
});
document.addEventListener("pointerdown", e => {
  if(pop.getAttribute("data-open") !== "1") return;
  if(pop.contains(e.target) || btn.contains(e.target)) return;
  openPop(false);
});
window.addEventListener("keydown", e => {
  if(app.getAttribute("data-view") !== "section") return;
  const t = e.target || {};
  if(t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "BUTTON") return;
  if(e.ctrlKey || e.metaKey || e.altKey) return;
  if(e.key === "ArrowUp" || e.key === "ArrowRight"){ step(1); e.preventDefault(); }
  else if(e.key === "ArrowDown" || e.key === "ArrowLeft"){ step(-1); e.preventDefault(); }
  else if(e.key === "f" || e.key === "F"){ fit(); need(); e.preventDefault(); }
  else if(e.key === "Home"){ setSlide(B.list[0]); e.preventDefault(); }
  else if(e.key === "End"){ setSlide(B.list[B.list.length - 1]); e.preventDefault(); }
});
function onResize(){
  dpr = Math.min(2, window.devicePixelRatio || 1);
  if(app.getAttribute("data-view") === "section") need();
}
window.addEventListener("resize", onResize);
onResize();

setSample(BR.volume_sample);
buildSampleList();

window.__BROWSER_VIEW__ = {B: B, BR: BR, CLS: CLS, COL: COL,
  setSample: setSample, setSlide: setSlide, step: step, fit: fit,
  openPop: openPop, shown: () => shown,
  SECT: SECT, sectionInfo: sectionInfo, layerOf: layerOf,
  setLayer: setLayer, layerNow: layerNow, extentNow: extentNow,
  pickSection: pickSection, buildCodexPick: buildCodexPick,
  visiumSpots: visiumSpots, VIS: VIS,
  xeniumCells: xeniumCells, XEN: XEN,
  render: () => { render(); }, need: need,
  stat: () => lastStat, cells: s => CELLS.get(s),
  cellsFor: cellsFor, pickLevel: pickLevel, toScreen: toScreen, toUm: toUm,
  images: () => IMG, canvas: cvB,
  view: () => app.getAttribute("data-view")};
})();
"""
JS += BROWSER_JS


KEYHELP = """MOUSE
  gesture            what it does                                 preventDefault
  left drag          rotate                                       no
  Shift + left drag  rotate horizontally only; the elevation is   no
                     frozen at the angle the drag started from
                     and vertical movement is discarded
  Cmd + left drag    pan (macOS)                                  no
  Ctrl + left drag   rotate about the surface under the press;   yes, on the canvas
                     also handles macOS Ctrl-click
  right drag         pan, on every platform                       yes, on the canvas
  left click         pick the front-most layer with tissue        no
                     under the cursor; the same point again
                     steps one layer deeper; empty space clears
                     the selection. A click is a press under
                     4 px and under 300 ms, so rotating never
                     selects by accident
  middle drag        rotate as well                               yes, on the canvas
  Cmd/Ctrl + middle  pan                                          yes, on the canvas
  middle 2x click    zoom to fit                                  yes, on the canvas
  body click         highlight and set rotation/zoom pivot; fade
                     distant bodies. Click again / Esc to clear
  wheel              zoom about focused body, otherwise cursor   yes, on the canvas
  right click        the canvas menu is suppressed, because the   yes, on the canvas
                     right button is a pan handle here; the menu
                     works everywhere else on the page
  touch drag         handled by the canvas, so the page cannot    yes, on the canvas
                     scroll under your finger
  layer strip        click a slice number to select it, click     no
                     the same number again to clear the selection

KEYBOARD -- bare keys and Shift only. No Ctrl, Cmd or Alt combination is taken, so
the browser keeps all of its own: Ctrl+R, F5, Ctrl+W/T/N/Q, Ctrl+F, Ctrl+S/P/O,
Ctrl +/-/0, Ctrl+L, Alt+D, Ctrl+Tab, Alt+arrows, Ctrl+Shift+I/J/C, F11, Ctrl+A/C/V/Z.
Keys are read only while the canvas has focus or nothing is focused; a slider, a
button or any field keeps its own keys.

  key                what it does                                 preventDefault
  1 2 3 4 5 6 7      front, back, left, right, top, bottom, iso   yes
  F                  zoom to fit                                  yes
  Shift+F            full screen on / off                         yes
  8 / 9              basis: G withdrawn / G not withdrawn         yes
  up / down          next / previous plane                        yes
  left / right       turn the specimen 5 degrees                  yes
  Shift + arrows     pan 10% of the viewport                      yes
  PgUp / PgDn        +-10 planes                                  yes
  Home / End         first / last plane                           yes
  G / Shift+G        next / previous gap of 25 um or more         yes
  S                  show only the selected layer / show all      yes
  space (hold)       temporary hand    space (tap) play / pause   yes
  + / -              zoom on the CANVAS CENTRE (a key has no      yes
                     cursor to anchor on)
  0 / Shift+0        reset zoom+pan / reset the whole camera      yes
  V                  quick cycle front -> top -> opening view     yes
  P                  put the rotation centre under the pointer    yes
  D                  base style: gray -> native -> translucent    yes
  C                  cell prediction overlay on / off. It does    yes
                     not change the base style; the two are
                     independent and compose
  I                  info panel on / off                          yes
  ?                  this list                                    yes
  Esc                clear the selection, show all layers, close  NO -- Escape stays
                     the panels                                    the browser's, it
                                                                  is how full screen
                                                                  ends
"""


def body(res_control: bool) -> str:
    res = ('<div class="grp" data-g="data"><span class="hd">Resolution</span>'
           '<div class="seg" id="resSeg"></div>'
           '</div>\n    ') if res_control else ""
    detail = ("4 um/px can be fetched here, which is one further step of real detail."
              if res_control else
              "This single file only carries 8 um/px, so zooming past that changes the "
              "framing but does not add detail.")
    he_rgb_note = ("H&amp;E colour is a second set of 25 RGB planes, fetched when you "
                   "ask for it." if res_control else
                   "This single file carries the grey volume only, so in native colour "
                   "the two DAPI modalities are tinted and H&amp;E stays grey.")
    return f"""<div id="app" data-view="volume">
<div id="topbar">
  <button id="infoBtn" aria-pressed="false" title="About this page (I)"
    aria-label="what this page is showing">&#9432;</button>
  <span id="samplebar"><span id="sampleLab">sample</span>
    <button id="sampleBtn" aria-haspopup="listbox" aria-expanded="false"
      aria-controls="samplePop" aria-label="choose which sample to show"></button></span>
  <span id="modeTag" data-mode="volume"></span>
  <label id="labelToggle"><input id="showLabels" type="checkbox" checked aria-label="Show labels"><span>Show labels</span></label>
  <span id="progress"></span>
</div>
<div id="wrap">
  <div id="left">
    <div id="stage">
      <canvas id="gl" tabindex="0" role="img"
        aria-label="Three-dimensional stack of 50 measured serial sections at their real z positions"></canvas>
      <div id="chips"></div><div id="corner" class="mono"></div>
      <div id="planetag"></div><div id="overlay"></div>
      <div id="scalebar"><div id="sbline"></div><span id="sblab"></span></div>
      <div id="hover"></div><div id="errs"></div>
    </div>
  </div>
  <div id="rail">
    <button class="railbtn" data-g="data" aria-pressed="false"
      title="Basis and resolution" aria-label="basis and resolution">&#9636;</button>
    <button class="railbtn" data-g="layers" aria-pressed="false"
      title="Layers" aria-label="layers">&#8801;</button>
    <button class="railbtn" data-g="view" aria-pressed="false"
      title="View" aria-label="view">&#8982;</button>
    <button class="railbtn" data-g="depth" aria-pressed="false"
      title="Depth" aria-label="depth axis">&#8597;</button>
    <button class="railbtn" data-g="display" aria-pressed="false"
      title="Display" aria-label="display">&#9680;</button>
    <button class="railbtn" id="fsBtn" aria-pressed="false" title="Full screen (Shift+F)"
      aria-label="full screen">&#9974;</button>
    <button class="railbtn" id="keysBtn" title="Keys (?)" aria-label="keys">?</button>
    <button class="railbtn" data-g="output" aria-pressed="false"
      title="Output (PNG / JPG / PDF)" aria-label="output">&#8681;</button>

  </div>
  <div id="panel">
    <div class="grp" data-g="data"><span class="hd">Basis</span>
      <div class="seg" id="basisSeg"></div>
      <div class="note" id="basisNote"></div></div>
    {res}<div class="grp" data-g="layers"><span class="hd">Layers</span>
      <div id="navrow">
        <button id="prevBtn" aria-label="previous plane">&#9664; previous</button>
        <button id="nextBtn" aria-label="next plane">next &#9654;</button>
        <button id="gapBtn" aria-label="jump to the next sampling gap">next gap (G)</button>
        <button id="playBtn" aria-pressed="false">play</button>
        <button id="clearSelBtn">clear selection (Esc)</button>
        <span class="cur" id="curlab"></span>
      </div>
</div>
    <div class="grp" data-g="view"><span class="hd">View</span>
      <div class="seg" id="viewSeg"></div>
      <button id="resetCam" style="margin-top:4px">reset camera (Shift+0)</button>
</div>
    <div class="grp" data-g="depth"><span class="hd">Depth</span>
      <label for="exagSl">depth stretch <span class="val" id="exagSlVal">1.0x</span></label>
      <input type="range" id="exagSl" min="10" max="200" step="1" value="10">
      <div class="note" id="sepGuide"></div></div>
    <div class="grp" data-g="output"><span class="hd">Output</span>
      <button id="figSheet" type="button" aria-describedby="figSheetNote" disabled
              title="Export the selected body and nearby glands at this camera angle and zoom">
        <svg viewBox="0 0 26 32" fill="none" aria-hidden="true"><path d="M4 1.5h12l6 6v23H4z M16 1.5v6h6" stroke="currentColor" stroke-width="1.3"/>
          <path d="M8 23l4-8 5 3 2-5 M8 27h11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
        <span><strong>Figure sheet</strong><small>Selected body + nearby glands · SVG</small></span>
      </button>
      <div id="figSheetNote" role="status">Select a nerve or gland in the 3D view.</div>
      <div class="shotrow">
        <button class="shotcard" id="shotPng" title="Save the current view as PNG">PNG</button>
        <button class="shotcard" id="shotJpg" title="Save the current view as JPG">JPG</button>
        <button class="shotcard" id="shotPdf" title="Save the current view as PDF">PDF</button>
      </div>
      <label for="shotScale">saved image size <span class="val" id="shotScaleVal">2x</span></label>
      <input type="range" id="shotScale" min="1" max="6" step="1" value="2"
             title="pixel grid of the saved file, as a multiple of the view on screen">
      <div class="note" id="shotSizeNote"></div>
      <div class="note" id="shotStatus" role="status" aria-live="polite"></div>
      <button id="shotCancel" type="button" hidden>Cancel export</button>
      <label class="showbar"><input id="shotAlpha" type="checkbox"><span>Transparent background (PNG / SVG)</span></label>
      <section id="figPanel" aria-labelledby="figHeading">
        <label class="showbar"><input id="figOn" type="checkbox"><span id="figHeading">Use figure mode</span></label>
        <fieldset id="figBody" aria-label="Figure mode controls" disabled>
          <div class="sub">
            <label class="showbar"><input id="figSections" type="checkbox" checked><span>Show sections</span></label>
            <span class="sublab">Sections (up to two)</span>
            <select id="figSecA" aria-label="first section"></select>
            <select id="figSecB" aria-label="second section"></select>
          </div>
          <div class="sub">
            <div class="showbar pillrow" id="figNerveRow"><input id="figNerve" type="checkbox" checked>
              <span>Nerve</span>
              <label class="kind" style="--kc:#2166f2"><input id="figN3" type="checkbox" checked><span>3D</span></label>
              <label class="kind" style="--kc:#2ca02c"><input id="figN2" type="checkbox" checked><span>2D</span></label></div>
            <label class="showbar"><input id="figNerveFill" type="checkbox"><span>Fill 2D nerve regions</span></label>
            <label for="figNA">3D nerve opacity <span class="val" id="figNAVal">100%</span></label>
            <input type="range" id="figNA" min="30" max="100" step="5" value="100">
            <div class="showbar pillrow" id="figGlandRow"><input id="figGland" type="checkbox" checked>
              <span>Glands</span>
              <label class="kind" style="--kc:#f59e0b"><input id="figG3" type="checkbox" checked><span>3D</span></label>
              <label class="kind" style="--kc:#2ca02c"><input id="figG2" type="checkbox" checked><span>2D</span></label></div>
            <label for="figGA">3D gland opacity <span class="val" id="figGAVal">100%</span></label>
            <input type="range" id="figGA" min="30" max="100" step="5" value="100">
          </div>
          <div class="sub">
            <div class="hdrow"><span class="sublab">Objects</span>
              <button id="figAll" type="button" class="figbtn">all</button>
              <button id="figNone" type="button" class="figbtn">none</button></div>
            <label class="showbar"><input id="figMulti" type="checkbox"><span>Select multiple</span></label>
            <div class="hdrow figselection"><span id="figSelectionCount" aria-live="polite">0 selected</span><button id="figClearSelection" type="button" class="figbtn" disabled>Clear selection</button></div>
            <div id="figObjs"></div>
          </div>
          <div class="note" id="figNote"></div>
        </fieldset>
      </section>
</div>
    <div class="grp" data-g="display"><div class="hdrow"><span class="hd">Display</span><span class="sublab">Sections</span></div>
      <div class="sub" id="secRow">
        <label class="showbar"><input id="modShow" type="checkbox" checked><span>Show sections</span></label>
      </div>
      <div class="sub" id="heSwapBox">
        <span class="sublab">Modality</span>
        <div class="seg segjoin" id="heSwapSeg"></div>
        </div>
      <div class="sub">
        <span class="sublab">Base</span>
        <div class="seg segjoin" id="colourSeg"></div>
</div>
      <div class="sub" id="overlayBox">
        <span class="sublab">Overlay</span>
        <label class="showbar"><input id="cellsToggle" type="checkbox">
          <span>Cell prediction</span></label>
        <div id="cellsBox" data-on="0">
          <div id="cellList"></div>
          <button id="colReset" type="button" style="margin-top:6px;font-size:11.5px"
            title="forget every custom colour and go back to the defaults">reset colours</button>
          <div class="showbar pillrow" id="nerveRow"><input id="nerveToggle" type="checkbox">
            <span>Nerve regions</span>
            <label class="kind" style="--kc:#d9a400"><input id="nlineToggle" type="checkbox" checked><span>Line</span></label></div>
          <label class="showbar"><input id="den3dToggle" type="checkbox">
            <span>Denoised</span></label>
          <div class="showbar pillrow" id="tlsRow"><input id="tlsToggle" type="checkbox">
            <span>TLS</span>
            <label class="kind" style="--kc:#1f5fd6"><input id="tls3dToggle" type="checkbox" checked><span>3D</span></label>
            <label class="kind" style="--kc:#2ca02c"><input id="tls2dToggle" type="checkbox" checked><span>2D</span></label></div>
          <div class="showbar pillrow" id="glandRow"><input id="glandToggle" type="checkbox"><span>Tumor glands</span>
            <label class="kind" style="--kc:#f59e0b"><input id="gland3dToggle" type="checkbox" checked><span>3D</span></label>
            <label class="kind" style="--kc:#2ca02c"><input id="gland2dToggle" type="checkbox" checked><span>2D</span></label></div>
          <label class="showbar"><input id="ductToggle" type="checkbox">
            <span>Duct lumens</span></label>
          <label class="showbar"><input id="tumorToggle" type="checkbox">
            <span>Tumor boundary</span></label>
          </div></div>
      <div class="sub" id="xgtBox" hidden>
        <span class="sublab">Xenium annotation</span>
        <label class="showbar"><input id="xgtShow" type="checkbox">
          <span>Show Xenium cell types</span></label>
        <div id="xgtBody" data-on="0">
          <label class="showbar"><input id="xtlsToggle" type="checkbox">
            <span>TLS regions (Xenium)</span></label>
          <div id="xgtList"></div>
        </div>
      </div>
      <div class="sub" id="cellSizeBox">
        <label for="cellPtSl">Cell size</label>
        <input type="range" id="cellPtSl" min="10" max="600" step="10" value="100"
          aria-label="how large each drawn cell is">
      </div>
      <div class="sub">
        <label for="opacitySl">Base opacity</label>
        <input type="range" id="opacitySl" min="2" max="30" step="1" value="10"
          aria-label="opacity of every base plane while nothing is selected">
        <div id="onlybox" style="margin-top:6px"></div>
</div></div>
  </div>
</div>
<div id="browser">
  <div id="bstage">
    <canvas id="bgl" tabindex="0" role="img"
      aria-label="one serial section on its own slide, with the cell prediction over it"></canvas>
    <div id="bnotice"></div><div id="bchips"></div>
    <div id="bcorner" class="mono"></div><div id="bscale"></div>
  </div>
  <div id="bside">
    <div class="grp" style="border-bottom:none;padding-bottom:0">
      <span class="hd">Sections</span>
      <div id="bsections"></div></div>
    <div class="sub" id="blayerBox" style="display:none">
      <span class="sublab">Modality on this section</span>
      <div class="seg segjoin" id="blayers"></div>
      <div class="note" id="blayerNote"></div></div>
    <div class="sub" id="bcodexBox" style="display:none">
      <span class="sublab">CODEX channels</span>
      <div id="bcodexPick"></div></div>
    <div class="sub" id="bvisiumBox" style="display:none">
      <span class="sublab">Visium spots</span>
      <button class="ovbtn" id="bspotsToggle" aria-pressed="true">Show spots</button>
      <button class="ovbtn ovsub" id="bintissue" aria-pressed="false">In tissue only</button>
      <div class="note" id="bvisNote"></div>
</div>
    <div class="sub" id="bxeniumBox" style="display:none">
      <span class="sublab">Xenium cells</span>
      <button class="ovbtn" id="bxcellsToggle" aria-pressed="true">Show cells</button>
      <button class="ovbtn ovsub" id="bxcolour" aria-pressed="true">Colour by transcript count</button>
      <div class="note" id="bxenNote"></div></div>
    <div class="sub" id="bcellBox"><span class="sublab">Cell prediction</span>
      <button class="ovbtn" id="bcellsToggle" aria-pressed="true">Cells</button>
      <button class="ovbtn" id="bregionsToggle" aria-pressed="true">Nerve regions</button>
      <div id="bclasses"></div></div>
    <div class="sub"><span class="sublab">This section</span>
      <div id="binfo"></div>
      <div id="bwarn"></div></div>
  </div>
  <div id="bbar">
    <button id="bprev" aria-label="previous section">&#9664; previous</button>
    <button id="bnext" aria-label="next section">next &#9654;</button>
    <button id="bfit">fit</button>
    <span class="cur" id="bcur"></span>
  </div>
</div>
<div id="bottom">
  <div id="rulerbox">
    <div id="rulerscroll"><div id="ruler"><div id="rcur"></div><div id="rhint"></div></div></div>
  </div>
  <div id="guidebox"><svg id="guide"></svg></div>
  <div id="stripbox">
    <div id="selector" role="listbox" aria-label="layer selector" tabindex="0">
      <div id="cells"></div>
    </div>
  </div>
</div>
<div id="infoPanel">
  <div id="warnbadges"></div>
  <div id="banner" class="mono"></div>
  <div class="note" style="margin-top:8px">The z ruler is read-only: hovering reads
    out, it does not select. Solid ticks are each section's real z; dashed ticks are
    where it is currently drawn, so selecting a plane visibly moves them apart.</div>
  <div class="note" style="margin-top:8px">Zooming into a SELECTED layer switches that
    layer to a tile pyramid, down to 0.5 &micro;m/px - the finest the prepared sections
    hold, and the native pitch of the H&amp;E scans. That layer then uses a contrast
    window measured on the finest level (about 0.7% of its tissue pixels fall outside
    it and clip); the whole-stack view always uses the 8 &micro;m window, where about
    12% would clip at 0.5 &micro;m/px. The window is what maps grey values to the
    screen, so this is why the two views are kept apart: within either one, every
    layer is on the same ramp and neighbours can be compared.</div>
  <div class="note" id="heSwapAbout" style="margin-top:8px;display:none"></div>
  <div class="note" id="cellsAbout" style="margin-top:8px;display:none"></div>
  <div id="infolive" class="mono" style="margin-top:8px;white-space:pre-wrap"></div>
  <div id="limits" class="mono"></div>
</div>
<div id="samplePop" role="listbox" aria-label="samples">
  <input id="sampleFilter" type="text" autocomplete="off" spellcheck="false"
    placeholder="type to filter, e.g. HT913 or SP0" aria-label="filter samples">
  <div id="sampleList"></div>
  <div id="sampleFoot" class="note"></div>
</div>
<div id="keyhelp" class="mono">{KEYHELP}</div>
</div>"""
