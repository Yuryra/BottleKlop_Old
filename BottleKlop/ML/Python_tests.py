
import numpy as np
np.random.seed(19680801)
import matplotlib.pyplot as plt

#print('matplotlib: {}'.format(matplotlib.__version__))


volume = np.random.rayleigh(27, size=40)
#amount = np.random.poisson(10, size=40)
#ranking = np.random.normal(size=40)
#price = np.random.uniform(1, 10, size=40)

#fig, ax = plt.subplots()

## Because the price is much too small when being provided as size for ``s``,
## we normalize it to some useful point sizes, s=0.3*(price*3)**2
#scatter = ax.scatter(volume, amount, c=ranking, s=0.3*(price*3)**2,
#                     vmin=-3, vmax=3, cmap="Spectral")

## Produce a legend for the ranking (colors). Even though there are 40 different
## rankings, we only want to show 5 of them in the legend.
#legend1 = ax.legend(*scatter.legend_elements(num=5),
#                    loc="upper left", title="Ranking")
#ax.add_artist(legend1)

## Produce a legend for the price (sizes). Because we want to show the prices
## in dollars, we use the *func* argument to supply the inverse of the function
## used to calculate the sizes from above. The *fmt* ensures to show the price
## in dollars. Note how we target at 5 elements here, but obtain only 4 in the
## created legend due to the automatic round prices that are chosen for us.
#kw = dict(prop="sizes", num=5, color=scatter.cmap(0.7), fmt="$ {x:.2f}",
#          func=lambda s: np.sqrt(s/.3)/3)
#legend2 = ax.legend(*scatter.legend_elements(**kw),
#                    loc="lower right", title="Price")

#plt.show()





#fig, ax = plt.subplots()
#for color in ['tab:blue', 'tab:orange', 'tab:green']:
#    n = 750
#    x, y = np.random.rand(2, n)
#    scale = 200.0 * np.random.rand(n)
#    ax.scatter(x, y, c=color, s=scale, label=color,
#               alpha=0.3, edgecolors='none')

#ax.legend()
#ax.grid(True)

#plt.show()

N = 45
x, y = np.random.rand(2, N)
c = np.random.randint(1, 5, size=N)
s = np.random.randint(10, 220, size=N)

#https://matplotlib.org/users/colors.html
z = np.random.randint(0, 7, size=N) #np.array([1,0,1,0,1])
colors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])#['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])
plt.scatter(x,y, c=colors[z])
plt.show()

cc = np.array(c.shape, np.str)
for i in range(len(c)):
  cc[i] = 'g'
c = cc


fig, ax = plt.subplots()

scatter = ax.scatter(x, y, c=c, s=s)

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes", bbox_to_anchor=(1,0))
ax.add_artist(legend1)

#plt.legend(loc='upper left', bbox_to_anchor=(1,1))

## produce a legend with a cross section of sizes from the scatter
#handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
#legend2 = ax.legend(handles, labels, loc="upper right", title="Sizes")

plt.show()