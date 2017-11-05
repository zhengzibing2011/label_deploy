import base64
try:
    import io
except ImportError:
    import io as io

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw
import scipy.misc
import skimage.color

#import user-defined module,make "label_name_to_val" become global variable.
import global_variable 

#define labelcolormap depend on CamVid class-11
labelcolormap = np.array([[128,128,128],[128,0,0],[192,192,128],[128,64,128],[0,0,192],[128,128,0],[192,128,128],[64,64,128],[64,0,128],[64,64,0],[0,128,192],[0,0,0]],dtype=np.float32) / 255

def img_b64_to_array(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr

 
def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def draw_label(label, img, label_names, colormap=None):
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

#modify colormap
    if colormap is None:
        reordered_label_value = sorted(global_variable.label_name_to_val.values())
        list_colormap = []
        for i in reordered_label_value:
            list_colormap.append(labelcolormap[i])
        colormap = np.array(list_colormap)

#modify bg_label from 0 to 11,bg_color to the last element of colormap
    label_viz = skimage.color.label2rgb(
        label, colors=colormap, bg_label=11, bg_color=labelcolormap[-1])
    plt.imshow(label_viz)
    plt.axis('off')

    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(label_names): 
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append(label_name)
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)

    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

    out = np.array(PIL.Image.open(f))[:, :, :3]
    out = scipy.misc.imresize(out, img.shape[:2])
    return out


def labelme_shapes_to_label(img_shape, shapes):
#define the collection of all label_name and its corresponding index
    all_labelname_index = {'b_Building':1, 'c_Column_Pole':2, 'd_Road':3, 'e_Sidewalk':4, 'f_Tree':5, 'g_SignSymbol':6, 'h_Fence':7, 'i_Car':8, 'j_Pedestrain':9, 'k_Bicyclist':10, 'l_Sky':11}

    global_variable.label_name_to_val = {'a_Void':0}

#change dtype to np.uint8
    lbl = np.zeros(img_shape[:2], dtype=np.uint8)
    for shape in sorted(shapes, key=lambda x: x['label']):
        polygons = shape['points']
        label_name = shape['label']

        if label_name in all_labelname_index:
            label_value = all_labelname_index[label_name]
            global_variable.label_name_to_val[label_name] = label_value
#modify
        else:
            label_value = 0    
            global_variable.label_name_to_val['a_Void'] = label_value
        mask = polygons_to_mask(img_shape[:2], polygons)
        lbl[mask] = label_value
    
#exchange 0 and 11 in lbl
#12 is a intermidate value
    lbl[lbl==0] = 12 
    lbl[lbl==11] = 0
    lbl[lbl==12] = 11

    M = len(global_variable.label_name_to_val)
    lbl_names = [None] * M

# sort the dict global_variable.label_name_to_val as the value, from small to large, make dict become tuple list, such as[tupel1,tuple2,...]

    reordered_label_names = sorted(global_variable.label_name_to_val.items(), key=lambda v: v[0])
    for i in range(M):
        key = reordered_label_names[i][0]
        lbl_names[i] = key 
# exchange the first and the last element
    lbl_names[0],lbl_names[M-1] = lbl_names[M-1],lbl_names[0]      
    return lbl, lbl_names 
