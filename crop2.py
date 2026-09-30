from PIL import Image
im=Image.open('/workspace/shots/mobile390-stitched.png')
crops={'hero2':(0,0,390,760),'board1':(0,1000,390,1520),'board2':(0,1520,390,2040),'heat2':(0,7440,390,8800),'decay2':(0,5930,390,6800),'wx2':(0,6700,390,7540)}
for k,v in crops.items():
    c=im.crop(v)
    c=c.resize((c.width*2,c.height*2), Image.LANCZOS)
    c.save('/workspace/shots/'+k+'.png')
    print(k,v,c.size)
