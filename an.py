from PIL import Image
im=Image.open('/data/tool/browser_snapshots/mobile390-full.png').convert('RGB')
w,h=im.size
px=im.load()
bands=[]
cur=None
for y in range(h):
    nw=0
    for x in range(0,392,3):
        r,g,b=px[x,y]
        if r<245 or g<245 or b<245: nw+=1
    empty=(nw<=1)
    if (not empty) and cur is None: cur=y
    elif empty and cur is not None:
        if y-cur>3: bands.append((cur,y,y-cur))
        cur=None
if cur is not None: bands.append((cur,h,h-cur))
print('SIZE',w,h,'NBANDS',len(bands))
for b in bands: print(b)
