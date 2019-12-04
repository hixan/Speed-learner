from matplotlib import pyplot as plt, animation  # type: ignore
import numpy as np  # type: ignore
import json

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('batch')
plt.ylabel('loss (absolute)')


def anim(_):
    ws = [
        (10, 'orange'),
        (50, 'red')
    ]
    with open('losses.txt', 'r') as f:
        try:
            data = json.load(f)
        except:
            data = []
    ax.clear()
    for d in data[1:]:
        #ax.plot(d, color='blue')
        for w, c in ws:
            ax.plot(
                [None] * int(w/2) +
                [np.mean(d[x: x+w]) for x in range(len(d)-w)],
                color=c
            )

anim8 = animation.FuncAnimation(fig, anim, interval=1000)
plt.show()
