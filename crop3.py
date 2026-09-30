from PIL import Image
im=Image.open('/workspace/shots/mobile390-stitched.png')
crops={'heatLabels':(24,7768,230,8090,3),'heatBottom':(24,8380,390,8560,3),'heatFull':(0,7430,390,8790,2)}
for k,v in crops.items():
    c=im.crop(v[:4])
    c=c.resize((c.width*v[4],c.height*v[4]), Image.LANCZOS)
    c.save('/workspace/shots/'+k+'.png')
    print(k,c.size)
