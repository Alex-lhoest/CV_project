from PIL import Image
import glob


# adjust width and height to your needs
width = 512
height = 512

# use one of these filter options to resize the image
#im2 = im1.resize((width, height), Image.NEAREST)      # use nearest neighbour
#im3 = im1.resize((width, height), Image.BILINEAR)     # linear interpolation in a 2x2 environment
#im4 = im1.resize((width, height), Image.BICUBIC)      # cubic spline interpolation in a 4x4 environment
#im5 = im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
ext = ".tif"


#for filename in glob.glob('../Dataset/Training_input/*'):
#    inp = Image.open(filename)
#    im = inp.resize((width, height), Image.ANTIALIAS)
#    im.save(filename + ext)
    
for filename in glob.glob('../../data/buildings/resized_image/Train/Mask/output/*'):
    inp = Image.open(filename)
    im = inp.resize((width, height), Image.ANTIALIAS)
    im.save(filename + ext)
    inp.close()
    

#for filename in glob.glob('../Dataset/Test_input/*'):
#    inp = Image.open(filename)
#    im = inp.resize((width, height), Image.ANTIALIAS)
#    im.save(filename + ext)
#    
#for filename in glob.glob('../Dataset/Test_output/*'):
#    inp = Image.open(filename)
#    im = inp.resize((width, height), Image.ANTIALIAS)
#    im.save(filename + ext)
    

    