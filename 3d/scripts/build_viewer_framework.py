#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import viewer_assets as VA
import viewer_solid as VS


def render_framework(have, data_dirs=None):
    if not have:
        raise ValueError('No published sample data found')
    link = ('<button id="resetBtn" title="forget everything this browser stored for the page '
            '(saved switches, colours, cached section images) and reload it clean">reset &amp; reload</button>')
    body_html = VA.body(res_control=True).replace(
        '<span id="progress"></span>', '<span id="progress"></span>\n  ' + link, 1)
    body_html = body_html.replace(
        '<button class="railbtn" id="fsBtn"',
        '<button class="railbtn" data-g="solid" aria-pressed="false"\n'
        '      title="Solid 3D bodies" aria-label="solid three-dimensional bodies">&#10697;</button>\n'
        '    <button class="railbtn" id="fsBtn"', 1)


    body_html = body_html.replace(
        '<div class="grp" data-g="display">',
        '<div class="grp" data-g="solid"><span class="hd">Solid 3D bodies</span>'
        + VS.PANEL + '</div>\n    <div class="grp" data-g="display">', 1)


    static_js = ("window.__BUILD_STAMP__ = window.__FRAMEWORK_BUILD_STAMP__;\n"
                 f"window.__VS_QUAD__ = {json.dumps(VA.VS_QUAD)};\n"
                 f"window.__FS_QUAD__ = {json.dumps(VA.FS_QUAD)};\n"
                 f"window.__VS_RECUT__ = {json.dumps(VA.VS_RECUT)};\n"
                 f"window.__VS_FULL__ = {json.dumps(VA.VS_FULL)};\n"
                 f"window.__FS_RESOLVE__ = {json.dumps(VA.FS_RESOLVE)};")


    boot = ("window.__SAMPLES__ = " + json.dumps(have) + ";\n"
            "(function(){var q=new URLSearchParams(location.search);var reg=window.__SAMPLES__;"
            "var s=q.get('sample');if(reg.indexOf(s)<0)s=reg[0];window.__SAMPLE_ID__=s;"
            "window.__DATA_BASE__='../'+s+'/viewer/';document.title=s+' serial-section stack';"
            "document.write('<script src=\"'+window.__DATA_BASE__+'sample_data.js?v='+Date.now()+'\"><\\/script>');})();")
    boot = boot.replace("window.__DATA_BASE__='../'+s+'/viewer/';",
                        "window.__DATA_DIRS__=" + json.dumps(data_dirs or {}) + ";window.__DATA_BASE__='../'+s+'/'+(window.__DATA_DIRS__[s]||'viewer')+'/';")
    boot += "\nwindow.__SAMPLE_URL__=function(id){var u=new URL(location.href);u.searchParams.set('sample',id);u.searchParams.delete('r');u.hash=document.fullscreenElement?'fs':'';return u.href;};"
    boot += """
window.__NAVIGATE_SAMPLE__=async function(id){
  if(id===window.__SAMPLE_ID__)return;
  if(window.__SAMPLE_NAV_PENDING__)return;
  window.__SAMPLE_NAV_PENDING__=true;
  var url=window.__SAMPLE_URL__(id);
  if(document.fullscreenElement && document.exitFullscreen){
    try{
      await document.exitFullscreen();
      // Let the native window return before replacing its document. On the
      // user's Mac, navigating during this transition left the next tab hidden
      // and stopped requestAnimationFrame although script execution continued.
      await new Promise(function(resolve){
        var last=performance.now(),finished=false;
        function resized(){last=performance.now();}
        function done(){if(finished)return;finished=true;clearInterval(timer);window.removeEventListener('resize',resized);resolve();}
        window.addEventListener('resize',resized);
        var start=performance.now();
        var timer=setInterval(function(){if((performance.now()-last>350&&document.visibilityState==='visible')||performance.now()-start>2000)done();},50);
      });
    }catch(e){}
  }
  if(window.__VIEWER_DISPOSE__)window.__VIEWER_DISPOSE__();
  await new Promise(function(resolve){setTimeout(resolve,50);});
  location.href=url;
};
"""
    switch_js = ("(function(){var e=document.getElementById('xsample');if(!e)return;"
                 "e.value=window.__SAMPLE_ID__;e.addEventListener('change',function(){"
                 "try{carrySave();}catch(x){}"
                 "window.__NAVIGATE_SAMPLE__(e.value);});})();\n"


                 "(function(){var b=document.getElementById('resetBtn');if(!b)return;"
                 "b.addEventListener('click',function(){b.disabled=true;b.textContent='resetting\\u2026';"
                 "try{localStorage.clear();sessionStorage.clear();}catch(x){}"
                 "var urls=[];try{var S=window.__SOURCE__||{};var base=window.__DATA_BASE__||'';"
                 "urls.push(base+'sample_data.js',base+'objects_payload.js');"
                 "Object.keys(S.res||{}).forEach(function(r){var R=S.res[r];"
                 "var add=function(d,f){if(!d||!f)return;Object.keys(f).forEach(function(b){var v=f[b];"
                 "(Array.isArray(v)?v:Object.keys(v).map(function(k){return v[k];})).forEach(function(p){urls.push(d+'/'+p);});});};"
                 "add(R.dir,R.files);add(R.dir,R.he_rgb);if(R.swap){add(R.swap.dir,R.swap.files);add(R.swap.dir,R.swap.he_rgb);}});}catch(x){}"
                 "var done=0,i=0;function next(){if(i>=urls.length){if(++done>=6)finish();return;}"
                 "var u=urls[i++];fetch(u,{cache:'reload'}).catch(function(){}).then(next);}"
                 "function finish(){if(finish.ran)return;finish.ran=true;"
                 "var p=(window.caches&&caches.keys)?caches.keys().then(function(ks){return Promise.all(ks.map(function(k){return caches.delete(k);}));}).catch(function(){}):Promise.resolve();"
                 "p.then(function(){var u=new URL(window.__SAMPLE_URL__(window.__SAMPLE_ID__));u.searchParams.set('r',Date.now());location.replace(u.href);});}"
                 "if(!urls.length)finish();else for(var k=0;k<6;k++)next();"
                 "setTimeout(finish,60000);});})();")
    html = f"""<!doctype html>
<html lang="en" translate="no" class="notranslate"><head><meta charset="utf-8">
<meta name="google" content="notranslate">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body,body *{{-webkit-user-select:none;user-select:none}}</style>
<script>window.__FRAMEWORK_BUILD_STAMP__ = "{__import__('datetime').datetime.now(__import__('zoneinfo').ZoneInfo('America/Chicago')).strftime('%m-%d %H:%M')} CT";</script>
<link rel="icon" href="data:,">
<title>serial-section stack</title>
<style>{VA.CSS}{VS.CSS}
#xsample{{margin-left:10px;color:var(--accent);background:transparent;border:1px solid var(--accent);border-radius:4px;font-size:12px;padding:1px 4px}}
#resetBtn{{margin-left:10px;color:var(--accent);background:transparent;border:1px solid var(--accent);border-radius:4px;font-size:12px;padding:1px 6px;cursor:pointer}}
#resetBtn:disabled{{opacity:.5;cursor:default}}</style></head>
<body>
{body_html}
<script>{boot}</script>
<script>{static_js}</script>
<script>{VA.JS}</script>
<script>{VS.JS}</script>
<script>{switch_js}</script>
</body></html>
"""
    return html


def build_framework(out, data_dirs=None):
    out = Path(out)
    samples = sorted(p.parent.parent.name for p in out.parent.parent.glob('*/viewer/sample_data.js'))
    for sample, directory in (data_dirs or {}).items():
        if sample not in samples or Path(directory).name != directory:
            raise ValueError('Data override must name a registered sample and a sibling data directory')
        if not (out.parent.parent / sample / directory / 'sample_data.js').is_file():
            raise ValueError('Missing data override: ' + sample)
    html = render_framework(samples, data_dirs)

    node = shutil.which('node')
    if not node:
        raise RuntimeError('node is required for JavaScript syntax validation')
    check = r"const fs=require('fs');for(const m of fs.readFileSync(0,'utf8').matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g))new Function(m[1]);"
    subprocess.run([node, '-e', check], input=html, text=True, check=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + '.tmp')
    temporary.write_text(html)
    temporary.replace(out)
    print(f'Framework: {out}; {len(samples)} samples; sample data untouched')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'samples/viewer/index.html')
    ap.add_argument('--data-dirs', type=Path, help='Optional JSON mapping of sample IDs to candidate data directory names')
    args = ap.parse_args()
    build_framework(args.out, json.loads(args.data_dirs.read_text()) if args.data_dirs else None)
