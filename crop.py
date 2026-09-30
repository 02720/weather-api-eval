from PIL import Image
im=Image.open('/data/tool/browser_snapshots/mobile390-full.png')
import os; os.makedirs('/workspace/shots',exist_ok=True)
crops={
 's1-hero':(0,0,392,780),
 's2-board-top':(0,730,392,1750),
 's2-board-card':(0,1100,392,1980),
 's4-heat':(0,7430,392,8790),
 's5-decay':(0,5940,392,6800),
 's5-wx':(0,6700,392,7540),
}
for k,v in crops.items():
    c=im.crop(v); c=c.resize((c.width*2,c.height*2), Image.LANCZOS); c.save('/workspace/shots/'+k+'.png')
    print(k, v, c.size)
print('done')
