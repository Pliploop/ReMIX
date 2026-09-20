import json, html, hashlib, sys
d=json.load(open(sys.argv[1]))
OUT=sys.argv[2]
GC={5:("#0E7A3B","EXACT MATCH"),4:("#1FA347","STRONG"),2:("#E8871E","PARTIAL"),1:("#D9662B","NEAR MISS"),0:("#8A9099","HARD NEGATIVE")}
def esc(s): return html.escape(str(s or ""))
def clip(s,n): s=str(s or "").strip(); return esc(s[:n].rsplit(" ",1)[0]+"…") if len(s)>n else esc(s)
def hue(cid): return int(hashlib.md5(cid.encode()).hexdigest()[:6],16)%360
def art(cid,big=False):
    h=hue(cid); sz="92px" if big else "58px"
    bars="".join(f'<span style="height:{20+ (i*7)%26}px"></span>' for i in range(5))
    return f'''<div class="art" style="width:{sz};height:{sz};background:
      linear-gradient(135deg,hsl({h} 70% 62%),hsl({(h+40)%360} 72% 46%))">
      <svg class="play" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
      <div class="eq">{bars}</div></div>'''
def chips(items,kind):
    return "".join(f'<span class="chip {kind}">{esc(t)}</span>' for t in items if t)
def playcard(c,big=False):
    return f'''<div class="pc">{art(c["clip_id"],big)}
      <div class="meta"><div class="ttl">{esc(c.get("title") or "Untitled")}</div>
      <div class="art-name">{esc(c.get("artist") or "Unknown artist")}</div>
      <div class="tagline">{chips(c.get("tags",[])[:4],"tag")}</div></div></div>'''
cands=""
for p in d["candidates"]:
    col,lbl=GC.get(p["grade"],("#888","?"))
    cands+=f'''<div class="cand" style="--gc:{col}">
      <div class="chead">{playcard(p["clip"])}
        <div class="badge" style="background:{col}">{lbl}<b>{p["grade"]}</b></div></div>
      <div class="ptype">{esc(p["pool_type"])}</div>
      <p class="reason">{clip(p["reason"],230)}</p>
      <div class="cons">{chips(p.get("satisfied",[])[:3],"ok")}{chips([f for f in p.get("failed",[]) if f not in set(p.get("satisfied",[]))][:3],"no")}</div>
    </div>'''
HTML=f'''<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;font-family:Inter,system-ui,sans-serif}}
body{{background:radial-gradient(1200px 600px at 15% -10%,#dfe9ff,transparent),
 radial-gradient(1000px 500px at 110% 10%,#ffe8f0,transparent),linear-gradient(160deg,#eef2f8,#e7ecf5);padding:44px;width:1480px}}
.glass{{background:rgba(255,255,255,.55);backdrop-filter:blur(16px);
 border:1px solid rgba(255,255,255,.7);border-radius:22px;
 box-shadow:0 12px 40px rgba(30,50,90,.13),inset 0 1px 0 rgba(255,255,255,.6)}}
.query{{padding:26px 30px;display:flex;align-items:center;gap:28px;margin-bottom:12px}}
.qlabel{{font-size:13px;font-weight:700;letter-spacing:.14em;color:#5566aa;text-transform:uppercase;margin-bottom:14px}}
.qsrc{{flex:0 0 auto}}
.instr{{flex:1;display:flex;align-items:center;gap:18px}}
.arrow{{font-size:34px;color:#7d8bbf}}
.pill{{background:linear-gradient(135deg,#3a57d6,#6a3ad6);color:#fff;padding:16px 24px;border-radius:16px;
 font-size:23px;font-weight:700;box-shadow:0 8px 24px rgba(70,60,200,.32);line-height:1.25}}
.pill small{{display:block;font-size:12px;font-weight:600;opacity:.8;letter-spacing:.12em;margin-bottom:4px}}
.pc{{display:flex;gap:14px;align-items:center}}
.art{{position:relative;border-radius:14px;flex:0 0 auto;box-shadow:0 6px 16px rgba(0,0,0,.18);overflow:hidden}}
.art .play{{position:absolute;left:50%;top:50%;transform:translate(-50%,-60%);width:40%;fill:rgba(255,255,255,.92);filter:drop-shadow(0 2px 3px rgba(0,0,0,.3))}}
.art .eq{{position:absolute;bottom:7px;left:0;right:0;display:flex;gap:3px;justify-content:center;align-items:flex-end;height:26px;opacity:.9}}
.art .eq span{{width:4px;background:rgba(255,255,255,.85);border-radius:2px}}
.meta .ttl{{font-weight:700;font-size:18px;color:#1a2338;line-height:1.15}}
.art-name{{font-size:14px;color:#5b6480;font-weight:500;margin-top:1px}}
.tagline{{margin-top:7px;display:flex;flex-wrap:wrap;gap:5px}}
.chip{{font-size:11px;font-weight:600;padding:3px 9px;border-radius:20px;white-space:nowrap}}
.chip.tag{{background:rgba(90,110,170,.13);color:#41507f}}
.chip.ok{{background:rgba(20,120,60,.14);color:#0e7a3b}}
.chip.ok:before{{content:"✓ "}}
.chip.no{{background:rgba(200,60,60,.13);color:#c0392b}}
.chip.no:before{{content:"✕ "}}
.row{{display:flex;gap:16px}}
.cand{{flex:1;padding:18px 18px 16px;border-top:5px solid var(--gc)}}
.chead{{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}}
.badge{{color:#fff;font-size:10px;font-weight:800;letter-spacing:.08em;padding:6px 10px;border-radius:11px;text-align:center;line-height:1.1;box-shadow:0 4px 10px rgba(0,0,0,.16);flex:0 0 auto}}
.badge b{{display:block;font-size:17px;margin-top:2px}}
.ptype{{font-size:11px;font-weight:700;letter-spacing:.1em;color:#8a90a0;margin:12px 0 8px}}
.reason{{font-size:12.5px;line-height:1.5;color:#39415a;margin-bottom:11px}}
.cons{{display:flex;flex-wrap:wrap;gap:5px}}
.title{{font-size:15px;font-weight:700;color:#5566aa;letter-spacing:.14em;text-transform:uppercase;margin:22px 4px 12px}}
</style></head><body>
<div class="glass query">
  <div class="qsrc"><div class="qlabel">Seed track</div>{playcard(d["source"],big=True)}</div>
  <div class="instr"><div class="arrow">→</div>
    <div class="pill"><small>COMPOSED EDIT INSTRUCTION</small>{esc(d["instruction"])}</div>
  <div class="arrow">→</div></div>
</div>
<div class="title">Graded relevance pool · LLM-verified over the heuristic score</div>
<div class="row">{cands}</div>
</body></html>'''
open(OUT,"w").write(HTML)
print("wrote", OUT)
