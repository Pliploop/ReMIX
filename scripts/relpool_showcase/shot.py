from playwright.sync_api import sync_playwright
import sys
src,out=sys.argv[1],sys.argv[2]
with sync_playwright() as p:
    b=p.chromium.launch()
    pg=b.new_page(viewport={"width":1480,"height":900},device_scale_factor=3)
    pg.goto("file://"+src)
    try: pg.wait_for_load_state("networkidle",timeout=8000)
    except: pass
    pg.wait_for_timeout(1200)
    el=pg.query_selector("body")
    el.screenshot(path=out)
    b.close()
print("shot ->",out)
