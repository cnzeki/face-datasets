import matplotlib.pyplot as plt
import random
import itertools
import json
import sys

def main(result_json, distractors):
    # load json  
    with open(result_json, 'r') as f:
        data = json.load(f)
    cmc = data['cmc']
    roc = data['roc']

    train_color = [random.random(), random.random(), random.random()]  # the color of line
    # cmc
    cmc_title = 'CMC Distractors-%d' % distractors
    figure_1 = plt.figure(1)
    plt.plot(cmc[0], cmc[1], color=train_color, linewidth=2)
    plt.title(cmc_title)
    plt.xlabel('Rank')
    plt.ylabel('Prob')
    plt.savefig(cmc_title+'.png')
    # roc
    roc_title = 'ROC Distractors-%d' % distractors
    figure_1 = plt.figure(2)
    plt.plot(roc[0], roc[1], color=train_color, linewidth=2)
    plt.title(roc_title)
    plt.xlabel('FAR')
    plt.ylabel('TAR')
    plt.savefig(roc_title+'.png')
    plt.show()
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('result_json')
        exit()
    result_json = sys.argv[1]
    segs = result_json.split('_')
    distractors = int(segs[len(segs)-2])
    main(result_json, distractors)
    