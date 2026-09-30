from PIL import Image
import os
src='/data/tool/browser_snapshots/'
H=379
W=390
TOTAL=10531
canvas=Image.new('RGB',(W,TOTAL),(255,255,255))
maxscroll=TOTAL-H
for k in range(28):
    p=src+'mc'+str(k).zfill(2)+'.png'
    im=Image.open(p).convert('RGB')
    im=im.crop((0,0,W,H))
    y=k*H
    if y>maxscroll:
        y=maxscroll
    canvas.paste(im,(0,y))
os.makedirs('/workspace/shots',exist_ok=True)
canvas.save('/workspace/shots/mobile390-stitched.png')
print('stitched', canvas.size)
px=canvas.load()
empty=[]
for y in range(TOTAL):
    nw=0
    for x in range(0,W,4):
        r,g,b=px[x,y]
        if r<245 or g<245 or b<245:
            nw+=1
    if nw<=1:
        empty.append(y)
print('empty rows', len(empty))
print('first empty', empty[:15])
print('last empty', empty[-15:])
