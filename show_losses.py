from matplotlib import pyplot as plt, animation  # type: ignore
import numpy as np  # type: ignore
import json

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('batch')
plt.ylabel('loss (absolute)')


def anim(_):
    # (window, quantile, colour)
    ws = [
        #(30, 0.25, 'red'),
        #(30, 0.75, 'red'),
        #(30, 0.5, 'blue'),
        (60, 0.3, 'green'),
        (60, 0.5, 'yellow'),
        (60, 0.7, 'green'),

    ]
    with open('losses.txt', 'r') as f:
        try:
            data = json.load(f)
        except:
            data = []
    ax.clear()
    for d in data:
        #ax.plot(d, color='blue')
        for w, q, c in ws:
            ax.plot(
                [None] * (w//2) +
                [np.quantile(d[x: x+w], q) for x in range(len(d)-w)],
                color=c
            )

anim8 = animation.FuncAnimation(fig, anim, interval=10000)
plt.show()
