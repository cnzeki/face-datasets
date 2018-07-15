import matplotlib.pyplot as plt
import random
import os

def getPlotColor(i):
    cnames = [
    '#ED1F24',
    '#B8529F',
    '#3952A3',
    '#69BC45', 
    '#FF8400',
    '#A74E44', 
    '#7642CC', 
    '#000000', 
    '#00FF00',
    '#FF0000']
    return cnames[i % 10]
    
def draw_chart(log_name, path_to_png, maps, precision, threshold):
    line_width = 1.0 # the line width
    # plot 
    figure_1 = plt.figure(log_name,figsize=(12, 6))
    ax= plt.subplot(1,1,1)
    ax.grid(True,color="gray",linestyle="-." )
    max_size = 0
    for name in maps:
        y = maps[name]
        max_size = max(max_size, len(y))
    idx = 0    
    for name in maps:
        y = maps[name]
        #print(y)
        n = len(y)
        if n < max_size * 0.2:
            continue
            
        x = [i for i in range(0,n)]
        ave = float(sum(y))/n
        label = '%.1f %s' % (ave, name)
        # ema 
        c = getPlotColor(idx)
        plt.plot(x , y, color=c, linewidth=line_width,label = label)
        idx += 1
    # threshold line
    label = 'threshold:%.4f' % (threshold)
    plt.plot([0, max_size], [threshold,threshold], color='green', linewidth=line_width,label = label)    
    plt.title('%.4f -- %s' % (precision, log_name))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='lower left')
    png_path = os.path.join(path_to_png, '%.4f--%s.png'%(precision, log_name))
    plt.savefig(png_path)