import matplotlib.pyplot as plt
import pickle

fig, axes = plt.subplots(ncols=3, sharey="row")
axes[0].plot([0,1],[0,1])
axes[1].plot([0,1],[1,2])
axes[2].plot([0,1],[2,3])

pickle.dump(fig, file('fig1.pkl', 'wb'))  

plt.close("all")

fig2 = pickle.load(file('fig1.pkl','rb'))
ax_master = fig2.axes[0]
for ax in fig2.axes:
    if ax is not ax_master:
        ax_master.get_shared_y_axes().join(ax_master, ax)

plt.show()