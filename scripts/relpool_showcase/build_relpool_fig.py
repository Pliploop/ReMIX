import json, html, hashlib, sys
d=json.load(open(sys.argv[1])); OUT=sys.argv[2]
CHAIN="#1FA347"; INSTRUCT="#FB8B24"; SEEDC="#2E6FD6"
# v2 grade -> (color, label). Bright saturated ramp shared with relevance_pool_analysis.
GC={6:("#0A7D3E","EXACT MATCH"),5:("#22C55E","STRONG"),4:("#A7E32C","GOOD"),
    3:("#F7B500","PARTIAL"),2:("#FB6A0A","SOFT FAIL"),1:("#E4231B","HARD FAIL"),
    0:("#CBD0D6","HARD NEGATIVE")}
def esc(s): return html.escape(str(s or ""))
def clip(s,n):
    s=str(s or "").strip()
    return esc(s[:n].rsplit(" ",1)[0]+"…") if len(s)>n else esc(s)
def rgba(hexc,a):
    h=hexc.lstrip("#"); return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"
def wave(cid,accent):
    hh=hashlib.md5(cid.encode()).hexdigest()
    bars="".join(f'<span style="height:{6+int(hh[i%32],16)*1.5:.0f}px"></span>' for i in range(38))
    return f'<div class="wave" style="--wc:{accent}">{bars}</div>'
def tags(ts):
    return "".join(f'<span class="tag">{esc(t)}</span>' for t in (ts or [])[:4] if t)
def cons(sat,fail):
    return ("".join(f'<span class="c ok">{esc(t)}</span>' for t in (sat or [])[:3])
          + "".join(f'<span class="c no">{esc(t)}</span>' for t in (fail or [])[:3]))
def trackcard(c,accent,badge,cls="src"):
    return f'''<div class="card {cls}" style="--ac:{accent}">
      <div class="chead"><span class="badge" style="background:{accent}">{badge}</span>
        <div class="tt"><div class="ttl">{esc(c.get("title") or "Untitled")}</div>
        <div class="art">{esc(c.get("artist") or "Unknown artist")}</div></div></div>
      <div class="tags">{tags(c.get("tags"))}</div>
      {wave(c["clip_id"],accent)}</div>'''
def candcard(p):
    col,lbl=GC.get(p["grade"],("#888","?")); c=p["clip"]
    return f'''<div class="card cand" style="--ac:{col}">
      <div class="chead"><div class="tt"><div class="ttl">{esc(c.get("title") or "Untitled")}</div>
        <div class="art">{esc(c.get("artist") or "Unknown artist")}</div></div>
        <span class="gbadge" style="background:{col}">{lbl}<b>{p["grade"]}</b></span></div>
      <div class="tags">{tags(c.get("tags"))}</div>
      {wave(c["clip_id"],col)}
      <div class="ptype" style="color:{col}">{esc(", ".join((p.get("failure_modes") or [])[:3]).replace("_"," ") or "—")}</div>
      <p class="reason">{clip(p["reason"],210)}</p>
      <div class="conswrap">{cons(p.get("satisfied"),p.get("failed"))}</div></div>'''
def example(e):
    cands=[c for c in e["candidates"]]
    sel=cands[:3]+[c for c in cands if c["grade"]==0][:1]
    seen=set(); sel=[c for c in sel if not (id(c) in seen or seen.add(id(c)))][:4]
    row="".join(candcard(p) for p in sel)
    return f'''<section class="ex">
      <div class="qrow">
        {trackcard(e["source"],SEEDC,"SEED")}
        <div class="arrow">→</div>
        <div class="instr"><div class="ilabel">
          <svg viewBox="0 0 24 24" fill="none" stroke="{INSTRUCT}" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M8 10h8M8 14h5M21 12a8 8 0 0 1-8 8H7l-4 3v-6.5A8 8 0 0 1 11 4h2a8 8 0 0 1 8 8z"/></svg>
          COMPOSED EDIT INSTRUCTION</div>
          <div class="itext">“{esc(e["instruction"])}”</div></div>
      </div>
      <div class="poollabel">Graded relevance pool <span>· LLM-verified over the heuristic score</span></div>
      <div class="crow">{row}</div>
    </section>'''
body="".join(example(e) for e in d["examples"])
HTML=f'''<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;font-family:Inter,'Helvetica Neue',Arial,system-ui,sans-serif}}
body{{background:#fff;padding:40px;width:1360px;color:#171717}}
.ex{{margin-bottom:30px;padding-bottom:26px;border-bottom:1px solid #ececec}}
.ex:last-child{{border-bottom:none;margin-bottom:0}}
.qrow{{display:flex;align-items:center;gap:18px;margin-bottom:18px}}
.card{{border-radius:16px;border:1px solid var(--ac);border-color:{rgba('#000000',0)};
 border:1px solid;padding:16px}}
.card{{border:1px solid;border-color:color-mix(in srgb,var(--ac) 35%,transparent);
 background:color-mix(in srgb,var(--ac) 5%,#fff)}}
.src{{flex:0 0 340px}}
.cand{{flex:1;min-width:0}}
.chead{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px}}
.badge{{color:#fff;font-size:11px;font-weight:600;letter-spacing:.06em;padding:4px 11px;border-radius:999px;flex:0 0 auto}}
.tt{{min-width:0}}
.ttl{{font-size:15px;font-weight:700;color:#171717;line-height:1.2}}
.art{{font-size:12.5px;color:#666;margin-top:2px}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin-top:11px}}
.tag{{font-size:11px;font-weight:500;color:#374151;background:#fff;border:1px solid #e5e7eb;padding:3px 9px;border-radius:999px}}
.wave{{display:flex;align-items:center;gap:2.5px;height:34px;margin-top:13px}}
.wave span{{flex:1;background:var(--wc);opacity:.5;border-radius:2px;min-height:3px}}
.arrow{{font-size:26px;color:#c9c9c9;flex:0 0 auto}}
.instr{{flex:1;border-radius:16px;padding:16px 20px;
 border:1px solid color-mix(in srgb,{INSTRUCT} 45%,transparent);
 background:color-mix(in srgb,{INSTRUCT} 7%,#fff)}}
.ilabel{{display:flex;align-items:center;gap:8px;font-size:11px;font-weight:700;letter-spacing:.1em;color:{INSTRUCT};text-transform:uppercase}}
.ilabel svg{{width:16px;height:16px}}
.itext{{font-size:22px;font-weight:700;color:#171717;margin-top:8px;line-height:1.3}}
.poollabel{{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#374151;margin:4px 2px 12px}}
.poollabel span{{color:#9aa0a6;font-weight:600}}
.crow{{display:flex;gap:14px;align-items:stretch}}
.gbadge{{color:#fff;font-size:9.5px;font-weight:700;letter-spacing:.05em;padding:5px 9px;border-radius:9px;text-align:center;line-height:1.15;flex:0 0 auto}}
.gbadge b{{display:block;font-size:15px;margin-top:1px}}
.ptype{{font-size:10.5px;font-weight:700;letter-spacing:.08em;margin:11px 0 7px}}
.reason{{font-size:12px;line-height:1.5;color:#404040;margin-bottom:10px}}
.conswrap{{display:flex;flex-wrap:wrap;gap:5px}}
.c{{font-size:10.5px;font-weight:500;padding:3px 8px;border-radius:999px}}
.c.ok{{background:{rgba(CHAIN,.13)};color:{CHAIN}}}
.c.ok:before{{content:"✓ "}}
.c.no{{background:{rgba('#E23B34',.12)};color:#E23B34}}
.c.no:before{{content:"✗ "}}
</style></head><body>{body}</body></html>'''
open(OUT,"w").write(HTML); print("wrote",OUT)
