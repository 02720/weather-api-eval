from PIL import Image
im=Image.open('/workspace/shots/mobile390-stitched.png')
c=im.crop((0,735,390,1120))
c=c.resize((c.width*2,c.height*2), Image.LANCZOS)
c.save('/workspace/shots/board0.png')
print('ok')
