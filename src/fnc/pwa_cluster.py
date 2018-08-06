import numpy as np
import scipy.linalg as la
import rls
import cvxpy as cvx



class ClusterPWA:
    """stores clustered points and associate affine models

    Attributes:
        TODO
        TODO make non-affine
    """

    def __init__(self, zs, ys, num_clusters, centroids, thetas,
                 cluster_labels, cov_c, z_cutoff=None, affine=True):
        """object initialization

        Args:
            zs, ys: datapoints with which to estimate PWA function from z->y
            z_cutoff: dimensions of z to ignore for clustering
            initialization: can be one of following
                1. an integer describing the number of clusters
                2. a tuple containing the centroids and affine functions
                3. a list of cluster labels for each data point 
        """
        # TODO assertions about types and shapes
        # Initializing data
        self.ys = ys; self.dimy = ys[0].size
        self.zs = zs; self.dimz = zs[0].size
        if z_cutoff == None:
            self.z_cutoff = self.dimz
        else:
            assert z_cutoff <= self.dimz, ("Cannot ignore z dimensions, \
                                            %d > %d").format(z_cutoff, self.dimz) 
            self.z_cutoff = z_cutoff
        self.affine = affine
        self.Nd = zs.shape[0]
        self.cov_e = np.eye(self.dimy) # TODO: change error model?

        # Initializing clusters and models
        self.cluster_labels = cluster_labels
        self.centroids = centroids
        self.thetas = thetas
        self.Nc = num_clusters
        self.cov_c = cov_c
        self.region_fns = None


        self.update_thetas = True

    @classmethod
    def from_num_clusters(cls, zs, ys, num_clusters, initialization=None, z_cutoff=None, affine=True):
        dimy = ys[0].size; dimz = zs[0].size
        z_lim = dimz if z_cutoff is None else z_cutoff
        # centroids are initialized to be randomly spread over the range of the data
        if initialization is None:
            centroids = np.random.uniform(size=np.hstack([num_clusters, z_lim]))
            offset = np.amin(zs, axis=0)
            spread = np.amax(zs, axis=0) - offset
            for i in range(z_lim):
                centroids[:,i] = spread[i]*centroids[:,i] + offset[i]
        else:
            centroids = initialization
        # covariances are initialized as identity
        cov_c = [np.eye(z_lim) for i in range(centroids.shape[0])]
        # labels are initialized to zero
        cluster_labels = np.zeros(zs.shape[0])
        # models are initialized to zero
        dim0 = dimz+1 if affine else dimz
        thetas = np.zeros( np.hstack([num_clusters, dim0, dimy]))
        return cls(zs, ys, num_clusters, centroids, thetas, 
                   cluster_labels, cov_c, z_cutoff, affine=affine)

    @classmethod
    def from_centroids_models(cls, zs, ys, centroids, thetas, z_cutoff=None, affine=True):
        z_lim = zs[0].size if z_cutoff is None else z_cutoff
        cov_c = [np.eye(z_lim) for i in range(centroids.shape[0])]
        return cls(zs, ys, len(centroids), centroids, thetas, 
                   np.zeros(zs.shape[0]), cov_c, z_cutoff, affine=affine)

    @classmethod
    def from_labels(cls, zs, ys, cluster_labels, z_cutoff=None, affine=True):
        centroids, thetas, cov_c = ClusterPWA.get_model_from_labels(zs, ys, 
                                                     cluster_labels, z_cutoff,affine=affine)
        return cls(zs, ys, np.unique(cluster_labels).size, centroids, thetas, 
                   cluster_labels, cov_c, z_cutoff, affine=affine)

    def add_data(self, new_zs, new_ys):
        # TODO assertions about data size
        self.zs = np.vstack([self.zs, new_zs])
        self.ys = np.vstack([self.ys, new_ys])
        self.cluster_labels = np.hstack([self.cluster_labels, 
                                          np.zeros(new_zs.shape[0])])
        self.Nd = self.zs.shape[0]

    def add_data_update(self, new_zs, new_ys, verbose=False, full_update=True):
        if verbose: print('adding new data')
        Nd_old = self.Nd
        self.add_data(new_zs, new_ys)
        self.update_clusters(verbose=verbose, data_start=Nd_old)
        if full_update: self.fit_clusters(verbose=verbose)

    def fit_clusters(self, data_start=0, verbose=False):
        """iteratively fits points to clusters and affine models

        Args:
            verbose: flag for printing centroid movement at each iteration
        """
        c_error = 100
        while c_error > 1e-6:
            c_error = self.update_clusters(verbose=verbose, data_start=data_start)
            if verbose: print('centroid movement', c_error)
        if verbose: print("done")

    def determine_polytopic_regions(self, verbose=False):
        ws = self.get_polytopic_regions(verbose)
        if ws[0] is not None:
            self.region_fns = np.array(ws)
        
        for i in range(self.Nd):
            dot_pdt = [w.T.dot(np.hstack([self.zs[i,0:self.z_cutoff], [1]])) for w in self.region_fns]
            self.cluster_labels[i] = np.argmax(dot_pdt)
        if self.update_thetas:
            self.centroids, self.thetas, self.cov_c = ClusterPWA.get_model_from_labels(self.zs, 
                                             self.ys, self.cluster_labels, self.z_cutoff, affine=self.affine)
        else:
            self.centroids, _, self.cov_c = ClusterPWA.get_model_from_labels(self.zs, self.ys, 
                                             self.cluster_labels, self.z_cutoff, affine=self.affine)
        
    def get_region_matrices(self):
        return getRegionMatrices(self.region_fns)

    def get_prediction_errors(self, new_zs=None, new_ys=None):
        estimation_errors = []
        if new_zs is None:
            # compute errors on the training data
            for i in range(self.Nd):
                idx = int(self.cluster_labels[i])
                if self.affine:
                    yhat = self.thetas[idx].T.dot(np.hstack([self.zs[i], 1]))
                else:
                    yhat = self.thetas[idx].T.dot(self.zs[i])
                estimation_errors.append(yhat-self.ys[i])
        else:
            # compute errors on the test data new_zs, new_ys
            for i in range(new_zs.shape[0]):
                yhat = self.get_prediction(new_zs[i])
                estimation_errors.append(yhat-new_ys[i])
        return np.array(estimation_errors)

    def update_clusters(self, data_start=0, verbose=False):
        """updates cluster assignment, centroids, and affine models

        Returns:
            c_error: the centroid movement during the update
        """
        # Assigning each value point to best-fit cluster
        if verbose:
            print("assigning datapoints to clusters")
        for i in range(data_start, self.Nd):
            quality_of_clusters = self.cluster_quality(self.zs[i], self.ys[i])
            cluster = np.argmin(quality_of_clusters)
            self.cluster_labels[i] = cluster
            if verbose and int(self.Nd/8) == 0:
                print('processed datapoint', i)
        # Storing the old centroid values
        centroids_old = np.copy(self.centroids)
        # updating model based on new clusters
        if verbose: print("updating models")
        if self.update_thetas:
            self.centroids, self.thetas, self.cov_c = ClusterPWA.get_model_from_labels(self.zs, self.ys, 
                                             self.cluster_labels, self.z_cutoff, affine=self.affine)
        else:
            self.centroids, _, self.cov_c = ClusterPWA.get_model_from_labels(self.zs, self.ys, 
                                             self.cluster_labels, self.z_cutoff, affine=self.affine)        
        try:
            c_error = np.linalg.norm(self.centroids-centroids_old, ord='fro')
        except ValueError as e:
            # TODO: deal with this better
            print(e)
            self.Nc = len(self.centroids)
            c_error = 1
        return c_error

    def cluster_quality(self, z, y, no_y = False):
        """evaluates the quality of the fit of (z, y) to each current cluster

        Args:
            z, y: datapoint
        Returns:
            an array of model quality for each cluster
        """
        #scaling_c = [la.pinv(la.sqrtm(self.cov_c[i])) for i in range(self.Nc)] TODO
        scaling_c = [np.eye(self.z_cutoff) for i in range(self.Nc)]
        scaling_e = la.inv(la.sqrtm(self.cov_e))
        
        # is distz the WRONG measure of locality for PWA?
        def distz(idx): 
            return np.linalg.norm(scaling_c[idx].dot(z[0:self.z_cutoff]-self.centroids[idx]),2)
        if self.affine:
            disty = lambda idx: np.linalg.norm(scaling_e.dot(y-self.thetas[idx].T.dot(np.hstack([z, 1]))),2)
        else:
            disty = lambda idx: np.linalg.norm(scaling_e.dot(y-self.thetas[idx].T.dot(z)),2)

        
        zdists = [distz(i) for i in range(self.Nc)]
        if no_y: 
            return np.array(zdists)
        ydists = [disty(i) for i in range(self.Nc)]
        return np.array(zdists) + np.array(ydists)

    @staticmethod
    def get_model_from_labels(zs, ys, labels, z_cutoff=None, affine=True):
        """ 
        Uses the cluster labels and data to return centroids and models and spatial covariances

        Returns
            centroid, affine model, and spatial covariance for each cluster
        """
        dimy = ys[0].size; dimz = zs[0].size
        if z_cutoff is None:
            z_cutoff = dimz
        Nc = np.unique(labels).size
        Nd = zs.shape[0]
        
        dim0 = dimz+1 if affine else dimz
        thetas = np.zeros( np.hstack([Nc, dim0, dimy]))
        centroids = np.random.uniform(size=np.hstack([Nc, z_cutoff]))
        cov_c = [np.eye(z_cutoff) for i in range(Nc)]

        # for each cluster
        for i in range(Nc):
            # gather points within the cluster
            points = [zs[j] for j in range(Nd) if labels[j] == i]
            points_cutoff = [zs[j][0:z_cutoff] for j in range(Nd) if labels[j] == i]
            points_y = [ys[j] for j in range(Nd) if labels[j] == i]
            if len(points) == 0:
                # if empty, place a random point
                # TODO more logic
                ind = int(np.round(Nd*np.random.rand()))
                labels[ind] == i
                points = [zs[ind]]
                points_cutoff = [zs[ind][0:z_cutoff]]
                points_y = [ys[ind]]
            else:
                # compute covariance
                cov_c[i] = np.cov(points_cutoff, rowvar=False)
                if len(cov_c[i].shape) != 2:
                    cov_c[i] = np.array([[cov_c[i]]])
            # compute centroids and affine fit
            centroids[i] = np.mean(np.array(points_cutoff), axis=0)
            thetas[i] = affine_fit(points, points_y, affine=affine) 
            assert len(cov_c[i].shape) == 2, cov_c[i].shape
        return centroids, thetas, cov_c

    def get_updated_thetas(self):
        """
        Uses recursive least squares to update theta based on the data points in the cluster
        """
        dim0 = dimz+1 if self.affine else dimz
        thetas = np.zeros( np.hstack([self.Nc, self.dim0, self.dimy]))
        
        for i in range(self.Nc):
            est = rls.Estimator(self.thetas[i], np.eye(self.dim0))

            points = [self.zs[j] for j in range(self.Nd) if self.cluster_labels[j] == i]
            points_y = [self.ys[j] for j in range(self.Nd) if self.cluster_labels[j] == i]
            for point, y in zip(points, points_y):
                if self.affine:
                    est.update(np.hstack([point,[1]]), y)
                else:
                    est.update(point, y)
            thetas[i] = est.theta
        return thetas

    def get_polytopic_regions(self, verbose=False):
        # TODO: smart filtering of points in clusters to use fewer
        # or iterative method
        prob, ws = cvx_cluster_problem(self.zs[:,0:self.z_cutoff], self.cluster_labels)
        # TODO check solver settings, max iter, tol, etc
        prob.solve(verbose=verbose,solver=cvx.SCS)
        if prob.status != 'optimal': print("WARNING: nonoptimal polytope regions:", prob.status)
        return [w.value for w in ws]

    def get_prediction(self, z):
        idx = self.get_region(z)
        if self.affine:
            yhat = self.thetas[idx].T.dot(np.hstack([z, 1]))
        else:
            yhat = self.thetas[idx].T.dot(z)
        return yhat

    def get_region(self, z):
        if self.region_fns is not None:
            # use region functions to assign model
            dot_pdt = [w.T.dot(np.hstack([z[0:self.z_cutoff], [1]])) for w in self.region_fns]
            idx = np.argmax(dot_pdt)
        else:
            # use clustering to assign model
            quality_of_clusters = self.cluster_quality(z, None, no_y=True)
            idx = np.argmin(quality_of_clusters)
        return idx

def affine_fit(x,y,affine=True):
        # TODO use best least squares (scipy?)
        if affine:
            ls_res = np.linalg.lstsq(np.hstack([x, np.ones([len(x),1])]), y)
        else:
            ls_res = np.linalg.lstsq(x, y)
        return ls_res[0]

def cvx_cluster_problem(zs, labels):
    s = np.unique(labels).size

    Ms = []
    ms = []
    ws = []
    for i,label in enumerate(np.sort(np.unique(labels))):
        selected_z = zs[np.where(labels == label)]
        num_selected = selected_z.shape[0]
        M = np.hstack([selected_z,np.ones([num_selected,1])])
        Ms.append(M); ms.append(num_selected)
        ws.append(cvx.Variable(zs[0].size + 1,1))
        
    cost = 0
    constr = []
    for i in range(s):
        for j in range(s):
            if i == j: continue;
            expr = Ms[i] * (ws[j] - ws[i]) + np.squeeze(np.ones([ms[i],1]))
            cost = cost + np.ones(ms[i]) * ( cvx.pos(expr) ) / ms[i]
            
    return cvx.Problem(cvx.Minimize(cost)), ws

def getRegionMatrices(region_fns):
    F_region = []; b_region = []
    Nr = len(region_fns)
    dim = region_fns[0].size
    for i in range(Nr):
        F = np.zeros([Nr-1, dim-1])
        b = np.zeros(Nr-1)
        for j in range(Nr):
            if j < i:
                F[j,:] = (region_fns[j,:-1] - region_fns[i,:-1]).T
                b[j] = region_fns[i,-1] - region_fns[j,-1]
            if j > i:
                F[j-1,:] = (region_fns[j,:-1] - region_fns[i,:-1]).T
                b[j-1] = region_fns[i,-1] - region_fns[j,-1]
        F_region.append(F); b_region.append(b)
    return F_region, b_region

def get_PWA_models(thetas, n, p):
    # if thetas include no d term, jut discard that result
    As = []; Bs = []; ds = [];
    for theta in thetas:
        assert theta.shape[0] == n+p+1 or theta.shape[0] == n+p
        assert theta.shape[1] == n
        As.append(theta[:n, :].copy().T)
        Bs.append(theta[n:(n+p), :].copy().T)
        ds.append(theta[-1,:].copy().T)
    return As, Bs, ds

def check_equivalence(region_fns, F_region, b_region, x):
    dot_pdt = [w.T.dot(np.hstack([x, [1]])) for w in region_fns]
    region_label = np.argmax(dot_pdt)
    
    matrix_label = []
    for i in range(len(F_region)):
        if np.all(F_region[i].dot(x) <= b_region[i]):
            matrix_label.append(i)
    print(region_label, matrix_label)

def select_nc_cross_validation(nc_list, zs, ys, verbose=False,
                               with_polytopic_regions=False, z_cutoff=None,
                               portion_test=0.25, affine=True):
    # TODO test this function
    # NOTE the data stored in clustering will be a different order than
    # the data given to this function.
    
    z_lim = dimz if z_cutoff is None else z_cutoff
    # splitting into test/train
    perm = np.random.permutation(len(zs))
    n_test = int(portion_test * len(zs))
    ind_test = perm[:n_test]
    ind_train = perm[n_test:]
    zs_train = zs[ind_train]; ys_train = ys[ind_train]
    zs_test = zs[ind_test]; ys_test = ys[ind_test]
    # fitting each cluster value
    clustering_list = []; errors = []
    for nc in nc_list: # TODO parallel?
        if verbose: print("===================== Fitting model with Nc=", nc, "====================")
        clustering = ClusterPWA.from_num_clusters(zs_train, ys_train, nc, z_cutoff=z_cutoff, affine=affine)
        clustering.fit_clusters(verbose=verbose)
        if with_polytopic_regions: # takes longer
            clustering.determine_polytopic_regions()
        train_errors = np.abs(clustering.get_prediction_errors())
        test_errors = np.abs(clustering.get_prediction_errors(new_zs=zs_test, new_ys=ys_test))
        print(train_errors.shape, test_errors.shape)
        if verbose: print('avg train error:', np.linalg.norm(train_errors, ord='fro') / int((1-portion_test) * len(zs)))
        if verbose: print('avg test error:', np.linalg.norm(test_errors, ord='fro') / n_test)
        # TODO: best error metric?
        metric = np.linalg.norm(test_errors, ord='fro')
        errors.append(metric)
        clustering_list.append(clustering)
    idx_best = np.argmin(errors)
    clustering_list[idx_best].add_data_update(zs_test, ys_test, verbose=verbose)
    if with_polytopic_regions:
        clustering_list[idx_best].determine_polytopic_regions()
    return clustering_list[idx_best], np.append(ind_train, ind_test)
    

def print_PWA_models(models):
    As, Bs, ds = models
    for A,B,d in zip(As, Bs, ds):
        spacer = np.nan * np.ones([A.shape[0], 1])
        stacked = np.hstack([A, spacer, B, spacer, d[:,np.newaxis]])
        print(np.array_str(stacked, precision=2, suppress_small=True))