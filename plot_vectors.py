import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster')
sns.set_style('ticks')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

names = g.vs['name']
pivot_w, context_w = model.layers[0].get_weights()
np.hstack((pivot_w, context_w)).shape
vectors = np.hstack((pivot_w, context_w))
pca_m = PCA(n_components=2)
XX = pca_m.fit_transform(vectors)
vclust = g.community_optimal_modularity()
colors = dict(zip(set(vclust.membership), ['r', 'g', 'b', 'k']))
v_colors = [colors[k] for k in vclust.membership]
plt.scatter(XX[:,0], XX[:,1], c=v_colors, s=5*degrees)
for n, x, y in zip(names, XX[:,0], XX[:,1]):
      plt.annotate(n, xy=(x,y), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
plt.savefig("Karate.pdf")

