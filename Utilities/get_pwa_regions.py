import numpy as np
import os
import pwa_cluster as pwac

n=6
homedir = os.path.expanduser("~")    
affine=False

data = np.load(homedir + '/barc/workspace/src/barc/src/Utilities/cluster_labels.npz')
clustering = pwac.ClusterPWA.from_labels(data['zs'], data['ys'], 
                                   data['labels'], z_cutoff=n)
clustering.determine_polytopic_regions(verbose=True)

np.savez('cluster_labels', labels=clustering.cluster_labels,
                                           region_fns=clustering.region_fns,
                                           thetas=clustering.thetas,
                                           zs=data['zs'], ys=data['ys'])