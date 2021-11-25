# ========================================  Library SPACE: ==================================================:


#App builder
import streamlit as st 
from PIL import Image 

#Visualization
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import time 

#Mathematics
import pandas as pd 
import numpy as np

#Timeseries dataframe
from TimeSeriesDataSet import *
#animation
from IPython.display import Video

# ========================================  Needed Functions SPACE: ==================================================:
# COST FUNCTIONS
#Euclidian Distance
import numpy
def sig(z):
    return 1/(1+np.e**-(z))


def compute_grad(X, y, w):
    """
    Compute gradient of cross entropy function with sigmoidal probabilities
    Args: 
        X (numpy.ndarray): examples. Individuals in rows, features in columns
        y (numpy.ndarray): labels. Vector corresponding to rows in X
        w (numpy.ndarray): weight vector
    Returns: 
        numpy.ndarray 
    """
    m = X.shape[0]
    Z = w.dot(X.T)
    A = sig(Z)
    cost = (-1/ m) * (X.T * (A - y)).sum(axis=1) 
    return cost

def euclidian_distance(x, y):
    from scipy.spatial import distance
    return distance.euclidean(x, y)


#Research Paper distance
#Linear regression
#pick some random value to start with
theta_0 = np.random.random()
theta_1 = np.random.random()


def hypothesis (theta_0,theta_1,X):
    return theta_1*X + theta_0


def cost_function (X,y,theta_0,theta_1):
    m = len(X)
    summation = 0.0
    for i in range (m):
        summation += ((theta_1 * X[i] + theta_0) - y[i])**2
    return summation /(2*m)


def gradient_descent(X,y,theta_0,theta_1,learning_rate):
    t0_deriv = 0
    t1_deriv = 0
    m = len(X)
    
    for i in range (m):
        t0_deriv += (theta_1 * X[i] + theta_0) - y[i]
        t1_deriv += ((theta_1 * X[i] + theta_0) - y[i])* X[i]
    theta_0 -= (1/m) * learning_rate * t0_deriv
    theta_1 -= (1/m) * learning_rate * t1_deriv
    
    return theta_0,theta_1


def training (X, y, theta_0, theta_1, learning_rate, iters):
    cost_history = [0]
    t0_history = [0]
    t1_history = [0]
    
    for i in range(iters):
        theta_0,theta_1 = gradient_descent(X, y, theta_0, theta_1, learning_rate)
        t0_history.append(theta_0)
        t1_history.append(theta_1)
        cost = cost_function(X, y, theta_0, theta_1)
        cost_history.append(cost)
        if i%10 == 0:
            print("iter={}, theta_0={}, theta_1={}, cost= {}".format(i, theta_0, theta_1, cost))
    return t0_history, t1_history, cost_history



# Geodesic Distance: pip install geopy
def geodesic_distance(x, y):
    from geopy.distance import geodesic
    # Print the distance calculated in km
    return geodesic(x, y)

    
# Visualization 
def scatter_plotting(dataset):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as seabornInstance
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    
    #Split dataset
    X = dataset['low'].values.reshape(-1,1).astype('float32')
    y = dataset['high'].values.reshape(-1,1).astype('float32')
    
    #Plotting
    fig,(ax1) = plt.subplots(1, figsize = (12,6))
    ax1.scatter (X, y, s = 8)
    plt.title ('Min vs Max Stock Rate')
    plt.xlabel('low')
    plt.ylabel('high')
    return plt.show()

#plot of Apple Price
def price_plotting(data, stock):
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial import distance_matrix
    from scipy import stats
    import seaborn as sns
    
    company = data.close[data['Ticker'].str.contains(stock)]
    Time = data.copy()
    Time['year'], Time['month'], Time['day'] = data.index.year, data.index.month, data.index.day
    time_pandas = Time.iloc[:,[-1]]
    # changing data from pandas data series to numpy array
    COMPANY = np.array(company)
    time = np.array(time_pandas)
  
    return COMPANY, time, company
    


# Dataset splitting
def splitter(Dataset):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as seabornInstance
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    
    #Split dataset
    close = Dataset.close
    high = Dataset.high
    X = close.values.reshape(-1,1).astype('float32')
    y = high.values.reshape(-1,1).astype('float32')
    
    #Split 80% of the data into the training set while 20% of the data go into the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


#Model Evaluator

#1- Mean Absolute Error

def mae(predictions, targets):
    # Retrieving number of samples in dataset
    samples_num = len(predictions)
    
    # Summing absolute differences between predicted and expected values
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += np.abs(prediction - target)
        
    # Calculating mean
    mae_error = (1.0 / samples_num) * accumulated_error
    
    return mae_error

# 2-Mean Square error
def mse(predictions, targets):
    # Retrieving number of samples in dataset
    samples_num = len(predictions)
    
    # Summing square differences between predicted and expected values
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += (prediction - target)**2
        
    # Calculating mean and dividing by 2
    mae_error = (1.0 / (2*samples_num)) * accumulated_error
    
    return mae_error


# converting prices to returns

def financial_returns(x):
    return np.diff(x) / x[ : -1]


#CDE_LOSS
def cde_loss(cdes, z_grid, true_z):
    n_obs, _ = cdes.shape
    term1 = np.mean(np.trapz(cdes ** 2, z_grid))
    nns = [np.argmin(np.abs(z_grid - true_z[ii])) for ii in range(n_obs)]
    term2 = np.mean(cdes[range(n_obs), nns])
    return term1 - 2 * term2

# CKDE
def nnkcde_function(dataset):
    #required library
    import numpy as np
    import  nnkcde
    #from nnkcde.test_loss_estimation import cde_loss
    from matplotlib import pyplot as plt
    from scipy.integrate import simps
    
    #trained data
    x_train = np.array(splitter(dataset)[0])
    x_test = np.array(splitter(dataset)[1])
    z_train = np.array(splitter(dataset)[2])
    z_test = np.array(splitter(dataset)[3])
    
    #creating CDE model
    k = 50
    model = nnkcde.NNKCDE(k=k)

    #linspace
    model.fit(x_train, z_train)

    st.warning('''
    For prediction, we need to specify: the CDE support, i.e. the grid over which we want the CDE to be predicted. Here we use the training data to inform the redshift minimum and maximum, and generate n_grid linearly separated values between the two.
    the bandwith of the KDE with bandwidth
    ''')
    n_grid = 1000
    bandwidth = 0.01
    z_grid = np.linspace(z_train.min(), z_train.max(), n_grid)
    cde_test = model.predict(x_test, z_grid, bandwidth=bandwidth)
    den_integral = simps(cde_test[0, :], x=z_grid)
    
    return cde_test, z_grid,den_integral, z_test


# Barycenter computation
def baycenter_computation(dataset):
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    #required libraries 
    import numpy as np
    import pandas as pd
    
    #Scipy: scientific computing
    import scipy
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial import distance_matrix
    from scipy import stats
    
    #Visualization
    import plotly.graph_objs as go
    import matplotlib.pyplot as pl
    import seaborn as sns
    
    #optimal transport
    import ot
    
    # necessary for 3d plot even if not used
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.collections import PolyCollection
    from distutils.core import setup
    
    #%% parameters
    n = 120  # nb bins
    
    # bin positions
    x = np.arange(n, dtype=np.float64)

    #definining the datapoints:
    a = dataset['close'].to_numpy()
    a= a[:n]
    b = dataset["high"].to_numpy()
    b = b[:n]
    
    # wasserstein
    reg = 1e-3
    
    # creating matrix A containing all distributions
    A = np.vstack((a, b)).T
    n_distributions = A.shape[1]
    
    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()

    
    #Computer OT_Matrix
    # a,b are 1D histograms (sum to 1 and positive)
    # M is the ground cost matrix
    T=ot.emd(a,b,M) # exact linear program
    T_reg=ot.sinkhorn(a,b,M,reg) # entropic regularized OT
    
    #Compute Wasserstein barycenter
    # A is a n*d matrix containing d  1D histograms
    # M is the ground cost matrix
    ba=ot.barycenter(A,M,reg) # reg is regularization parameter
    
    # creating matrix A containing all distributions
    A = np.vstack((a, b)).T
    n_distributions = A.shape[1]

    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()
    
    ##############################################################################
    # Barycenter computation
    # ----------------------

    #%% barycenter computation

    alpha = 0.2  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    # l2bary
    bary_l2 = A.dot(weights)

    # wasserstein
    reg = 1e-3
    bary_wass = ot.bregman.barycenter(A, M, reg, weights)

    pl.figure(2)
    pl.clf()
    pl.subplot(2, 1, 1)
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')

    pl.subplot(2, 1, 2)
    pl.plot(x, bary_l2, 'r', label='l2')
    pl.plot(x, bary_wass, 'g', label='Wasserstein')
    pl.legend()
    pl.title('Barycenters')
    pl.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    pl.show()
    st.pyplot()

    
    return 200

# Barycentric interpolation
def baycenter_interpolation(dataset):
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    #required libraries 
    import numpy as np
    import pandas as pd
    
    #Scipy: scientific computing
    import scipy
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial import distance_matrix
    from scipy import stats
    
    #Visualization
    import plotly.graph_objs as go
    import matplotlib.pyplot as pl
    import seaborn as sns
    
    #optimal transport
    import ot
    
    # necessary for 3d plot even if not used
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.collections import PolyCollection
    from distutils.core import setup
    
    #%% parameters
    n = 120  # nb bins
    
    # bin positions
    x = np.arange(n, dtype=np.float64)

    #definining the datapoints:
    a = dataset['close'].to_numpy()
    a= a[:n]
    b = dataset["high"].to_numpy()
    b = b[:n]
    
    # wasserstein
    reg = 1e-3
    
    # creating matrix A containing all distributions
    A = np.vstack((a, b)).T
    n_distributions = A.shape[1]
    
    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()

    
    #Computer OT_Matrix
    # a,b are 1D histograms (sum to 1 and positive)
    # M is the ground cost matrix
    T=ot.emd(a,b,M) # exact linear program
    T_reg=ot.sinkhorn(a,b,M,reg) # entropic regularized OT
    
    #Compute Wasserstein barycenter
    # A is a n*d matrix containing d  1D histograms
    # M is the ground cost matrix
    ba=ot.barycenter(A,M,reg) # reg is regularization parameter
    
    # creating matrix A containing all distributions
    A = np.vstack((a, b)).T
    n_distributions = A.shape[1]

    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()
    ##############################################################################
    # Barycentric interpolation
    # -------------------------

    #%% barycenter interpolation

    n_alpha = 11
    alpha_list = np.linspace(0, 1, n_alpha)


    B_l2 = np.zeros((n, n_alpha))

    B_wass = np.copy(B_l2)

    for i in range(0, n_alpha):
        alpha = alpha_list[i]
        weights = np.array([1 - alpha, alpha])
        B_l2[:, i] = A.dot(weights)
        B_wass[:, i] = ot.bregman.barycenter(A, M, reg, weights)

    #%% plot interpolation

    pl.figure(3)

    cmap = pl.cm.get_cmap('viridis')
    verts = []
    zs = alpha_list
    for i, z in enumerate(zs):
        ys = B_l2[:, i]
        verts.append(list(zip(x, ys)))

    ax = pl.gcf().gca(projection='3d')

    poly = PolyCollection(verts, facecolors=[cmap(a) for a in alpha_list])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')
    ax.set_xlabel('x')
    ax.set_xlim3d(0, n)
    ax.set_ylabel('$\\alpha$')
    ax.set_ylim3d(0, 1)
    ax.set_zlabel('')
    ax.set_zlim3d(0, B_l2.max() * 1.01)
    pl.title('Barycenter interpolation with l2')
    pl.tight_layout()
    pl.show()
    st.pyplot()
    
    pl.figure(4)
    cmap = pl.cm.get_cmap('viridis')
    verts = []
    zs = alpha_list
    for i, z in enumerate(zs):
        ys = B_wass[:, i]
        verts.append(list(zip(x, ys)))

    ax = pl.gcf().gca(projection='3d')

    poly = PolyCollection(verts, facecolors=[cmap(a) for a in alpha_list])
    poly.set_alpha(0.7)
    ax.add_collection3d(poly, zs=zs, zdir='y')
    ax.set_xlabel('x')
    ax.set_xlim3d(0, n)
    ax.set_ylabel('$\\alpha$')
    ax.set_ylim3d(0, 1)
    ax.set_zlabel('')
    ax.set_zlim3d(0, B_l2.max() * 1.01)
    pl.title('Barycenter interpolation with Wasserstein')
    pl.tight_layout()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    pl.tight_layout()
    pl.show()
    st.pyplot()

    
    return  201


# Sinkhorn matrix
def optimal_transport_sinkhorn(dataset):
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    #required libraries 
    import numpy as np
    import pandas as pd
    
    #Scipy: scientific computing
    import scipy
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial import distance_matrix
    from scipy import stats
    
    #Visualization
    import plotly.graph_objs as go
    import matplotlib.pyplot as pl
    import seaborn as sns
    
    #optimal transport
    import ot
    
    # necessary for 3d plot even if not used
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.collections import PolyCollection
    from distutils.core import setup
    
    #%% parameters
    n = 120  # nb bins
    
    # bin positions
    x = np.arange(n, dtype=np.float64)

    #definining the datapoints:
    a = dataset['close'].to_numpy()
    a= a[:n]
    b = dataset["high"].to_numpy()
    b = b[:n]
    
    # wasserstein
    reg = 1e-3
    
    # creating matrix A containing all distributions
    A = np.vstack((a, b)).T
    n_distributions = A.shape[1]
    
    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()
    
    # A is a n*d matrix containing d  1D histograms
    # M is the ground cost matrix
    ba=ot.barycenter(A,M,reg) # reg is regularization parameter
    st.warning(ba)
   
    return ba
    

def use_case(dataset):
    st.info('''
    
    *1D Wasserstein barycenter comparison between exact LP and entropic regularization*
    
    This example illustrates the computation of regularized Wasserstein Barycenter
    and exact LP barycenters using standard LP solver.
    ''')
    import numpy as np
    import matplotlib.pylab as pl
    import ot
    # necessary for 3d plot even if not used
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.collections import PolyCollection  # noqa
    
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    #%% parameters
    problems = []
    n = 100  # nb bins

    # bin positions
    x = np.arange(n, dtype=np.float64)

    #definining the datapoints:
    ##############################################################################
    # Gaussian Data
    # -------------

    # Gaussian distributions
    a1 = dataset['close'].to_numpy()
    a1= a1[:n]
    a2 = dataset["high"].to_numpy()
    a2 = a2[:n]


    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T
    n_distributions = A.shape[1]

    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()


    #%% plot the distributions

    pl.figure(1, figsize=(6.4, 3))
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')
    pl.tight_layout()

    #%% barycenter computation

    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    # l2bary
    bary_l2 = A.dot(weights)

    # wasserstein
    reg = 1e-3
    ot.tic()
    bary_wass = ot.bregman.barycenter(A, M, reg, weights)
    ot.toc()


    ot.tic()
    bary_wass2 = ot.lp.barycenter(A, M, weights, solver='interior-point', verbose=True)
    ot.toc()

    pl.figure(2)
    pl.clf()
    pl.subplot(2, 1, 1)
    for i in range(n_distributions):
        pl.plot(x, A[:, i])
    pl.title('Distributions')

    pl.subplot(2, 1, 2)
    pl.plot(x, bary_l2, 'r', label='l2')
    pl.plot(x, bary_wass, 'g', label='Reg Wasserstein')
    pl.plot(x, bary_wass2, 'b', label='LP Wasserstein')
    pl.legend()
    pl.title('Barycenters')
    pl.tight_layout()

    problems.append([A, [bary_l2, bary_wass, bary_wass2]])


    ##############################################################################
    # Final figure
    # ------------
    #

    #%% plot

    nbm = len(problems)
    nbm2 = (nbm // 2)


    pl.figure(2, (20, 6))
    pl.clf()

    for i in range(nbm):

        A = problems[i][0]
        bary_l2 = problems[i][1][0]
        bary_wass = problems[i][1][1]
        bary_wass2 = problems[i][1][2]

        pl.subplot(2, nbm, 1 + i)
        for j in range(n_distributions):
            pl.plot(x, A[:, j])
        if i == nbm2:
            pl.title('Distributions')
        pl.xticks(())
        pl.yticks(())

        pl.subplot(2, nbm, 1 + i + nbm)
        

        pl.plot(x, bary_l2, 'r', label='L2 (Euclidean)')
        pl.plot(x, bary_wass, 'g', label='Reg Wasserstein')
        pl.plot(x, bary_wass2, 'b', label='LP Wasserstein')
        if i == nbm - 1:
            pl.legend()
        if i == nbm2:
            pl.title('Barycenters')

        pl.xticks(())
        pl.yticks(())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        pl.tight_layout()
        pl.show()
        st.pyplot()
        
    return 200


#Scott's factor
#implementing neff: Effective number of datapoints.
def neff(self):
        try:
            return self._neff
        except AttributeError:
            self._neff = 1/sum(self.weights**2)
            return self._neff


def scotts_factor(self):
    #scipy.stats.gaussian_kde.scotts_factor
    """Compute Scott's factor.
    Returns
    -------
    s : float
    Scott's factor.
    """
    return power(self.neff, -1./(self.d+4))



#Silverman's factor
def silverman_factor(self):
    """Compute Silverman's factor.
    Returns
    -------
    s : float
    Silverman's factor.
    """
    return power(self.neff*(self.d+2.0)/4.0, -1./(self.d+4))
    

#Default method to calculate bandwidth, can be overwritten by subclass
covariance_factor = scotts_factor
def set_bandwidth(self, bw_method=None):
        """Compute the estimator bandwidth with given method.
        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.
        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `kde.factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `kde.covariance_factor` method is kept.
        """
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, str):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()
        
#Cost Matrix
#Cost Matrix

def cost_matrix(dataset):
    #required optimal transport package
    import ot
    import numpy as np  # always need it
    import pylab as pl  # do the plots
    import streamlit as st
    from scipy.spatial import distance

    #learn about OT distance 
    #st.warning(help(ot.dist))
    
    #%% parameters
    n = 5 # nb points
    
    #definining the datapoints: The dataset with which `gaussian_kde` was initialized
    hlassa = dataset[['high', 'low', 'close']]
    hlassa.reset_index(drop=True, inplace=True)
    hlassa.index = hlassa.high
    a = hlassa.high.values
    #a = a[:n]
    a_list = list(a)
    
    hlassa.index = hlassa.low
    b = hlassa.low.values
    #b= b[:n]
    b_list = list(b)
    
    hlassa.index = hlassa.close
    cl = hlassa.close.values
    #cl= cl[:n]
    cl_list = list(cl)

    Imap = dataset[['high', 'low', 'close']].values
    #Cost
    #C = ot.dist(a, b)
    C= distance.euclidean(a, b)
    pl.figure(1, (7, 6))
    pl.clf()
    pl.imshow(Imap, interpolation='bilinear')  # plot the map
    pl.scatter(a[:], b[:], s=cl, c='r', ec='k', label='high')
    pl.scatter(b[:], a[:], s=cl, c='b', ec='k', label='low')
    pl.legend()
    pl.title('Cost matrix')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    pl.show()
    st.pyplot()
    
    return C
    
    
#Time optimization
#Stockastic Gradient Descent
def stochastic_gradient_descent(dataset):
    #required librairie
    import time
    import numpy as np
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import streamlit as st
    
    st.info(
        """ Computes Ordinary Least SquaresLinear Regression with Stochastic Gradient Descent as the optimization algorithm.
        :param feature_array: array with all feature vectors used to train the model
        :param target_array: array with all target vectors used to train the model
        :param to_predict: feature vector that is not contained in the training set. Used to make a new prediction
        :param learn_rate_type: algorithm used to set the learning rate at each iteration.
        :return: Predicted cooking time for the vector to_predict and the R-squared of the model.
    """)
    
    #parameters
    feature_array = dataset[['high']]
    hlassa = dataset[['high', 'low', 'close']]
    hlassa.reset_index(drop=True, inplace=True)
    hlassa.index = hlassa.high
    mulla = hlassa.high.values
    target_array = hlassa.high.values
    
    to_predict = mulla.reshape(-1, 1)
    learn_rate_type = "invscaling"
    #print(target_array)
    
    start_time = time.time()
    linear_regression_pipeline = make_pipeline(StandardScaler(), SGDRegressor(learning_rate=learn_rate_type))
    
    linear_regression_pipeline.fit(feature_array.values, target_array)
    stop_time = time.time()
    st.markdown("Total runtime: %.6fs" % (stop_time - start_time))
    st.markdown("Algorithm used to set the learning rate: "+ learn_rate_type)
    st.markdown("Model Coeffiecients: " + str(linear_regression_pipeline[1].coef_))
    st.markdown("Number of iterations: " + str(linear_regression_pipeline[1].n_iter_))    # Make a prediction for a feature vector not in the training set
    prediction = np.round(linear_regression_pipeline.predict(to_predict), 0)[1]
    st.markdown("Predicted highest stock to buy: ***$" + str(prediction) + "*** USD")    
    r_squared = np.round(linear_regression_pipeline.score(feature_array, target_array).reshape(-1, 1)[0][0], 2)
    st.markdown("R-squared: " + str(r_squared))
    st.markdown("----------------------------")
    
#Animated plot of cost function

#Animated plot of cost function

# Simple linear regression model:
class LinearRegression(object):
    def __init__(self,w=1,b=1, lr=0.01): 
        self.lr=lr
        self.w=np.array([[w]])
        self.b=np.array([b])

    def cost(self,x,y):     
        pred = x@self.w+self.b  # predicted y-values
        e=y-pred             # error term
        return np.mean(e**2)  # mean squared error

    def fit(self, x,y):
        pred = x@self.w+self.b
        e=y-pred
        dJ_dw=(np.mean(e*(-2*x), axis=0)) # partial derivate of J with respect to w
        dJ_db=(np.mean(e*(-2),axis=0)) # partial derivate of J with respect to b
        self.w = (self.w.T-self.lr*dJ_dw).T  # update w
        self.b = self.b - self.lr*dJ_db    # update b

    def predict(self, x):
        return (x @ self.w.T + self.b)  # return predicted values

    def params(self):
        return (self.w,self.b)   # return parameters
    
    
def animated_plot_LR(dataset):
    # Import libraries
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import celluloid
    from celluloid import Camera
    import pandas_alive
    import matplotlib.animation as manimation
    import streamlit as st
    manimation.writers.list()
    
    # Introduce training data
    x_train = dataset[['high', 'low', 'close']].tail(30)
    
    x_train.reset_index(drop=True, inplace=True)
    x_train.index = x_train.high.tolist()
    mulla = x_train.high.values.reshape(-1 , 1)
    
    x_train = mulla[:30]
    arr = np.array([range(len(mulla[:30]))])
    y_train = arr.reshape(-1, 1)
    
    
    # Introduce lists where data points are being stored: 
    w_list=[]   # list contains weights
    b_list=[]   # list contains biases
    c_list=[]   # list contains costs 
    ys_list=[]  # store arrays of predicted y-values for xs ( -> plot regression line!) 
    cl_list = [] # list contains predicted y-values for x_train ( -> plot connecting lines!) 

    xs= np.array([    # set x-values for regression line plot               
                [0],
                 [3000]
                 ])

    # Train model: 
    model=LinearRegression(w=3,b=-1,lr=0.001) # set initial parameters and learning rate 

    for i in range(60000):      # set number of epochs
        w_list.append(model.params()[0])    # append weights (=slopes) to list
        b_list.append(model.params()[1])    # append biases (=y-intercepts) to list
        c_list.append(model.cost(y_train,x_train))  # append costs to list
        ys_list.append(model.predict(xs).T)     # append pairs of predicted y-values for xs 
        cl_list.append(model.predict(x_train).T) # append predicted y-values for x_train to list
        model.fit(y_train, x_train) # fit model


    #make prediction
    # print parameters and costs after all epochs
    st.info("weight: " + str( model.params()[0]) )  
    st.info("y-intercept: " + str( model.params()[1]) )
    st.info("costs: "+ str(model.cost(y_train, x_train)))
    
    # Define which epochs/data points to plot
    a=np.arange(0,50,1).tolist()
    b=np.arange(50,100,5).tolist()
    c=np.arange(100,12000,200).tolist()
    p = a+b+c # points we want to plot

    # Turn lists into arrays
    w= np.array(w_list).flatten()
    b= np.array(b_list).flatten()
    c= np.array(c_list).flatten()
    ys = np.array(ys_list) 
    p=np.array(p)

    
    # Create first animation: 
    fig = plt.figure(figsize=(10,10)) # create figure
    labelsize_ = 14
    camera = Camera(fig)  # create camera
    
    
    for i in p:
        
        #W chart
        ax1=fig.add_subplot(3, 2, 2)  
        ax1.plot(w[0:i], color='blue', linestyle="dashed", alpha=0.5)
        ax1.set_title("w", fontsize=17)
        ax1.tick_params(axis='both', which='major', labelsize=labelsize_)
        
        #B chart
        ax2=fig.add_subplot(3, 2, 4, sharex=ax1) # right plots share x-axis. 
        ax2.plot(b[0:i], color='red', linestyle="dashed", alpha=0.5)
        ax2.set_title("b", fontsize=17)
        ax2.tick_params(axis='both', which='major', labelsize=labelsize_)
        
        #Cost chart
        ax3=fig.add_subplot(3, 2, 6, sharex=ax1) 
        ax3.plot(c[0:i],color='black',linestyle="dashed")
        ax3.set_title("costs", fontsize=17)
        ax3.tick_params(axis='both', which='major', labelsize=labelsize_)
        ax3.set_xlabel("epochs", fontsize=14, labelpad=10)
        
        
        #Linear regression
        ax0=fig.add_subplot(1, 2, 1) # plot fit
        leg=ax0.plot(xs.T.flatten(),ys[i].flatten(),color='r', label=str(i))  # set legend; flatten arrays to get plots!
        ax0.scatter(x_train, y_train, color='b',marker='o', s=44)
        ax0.legend(leg,[f'epochs: {i}'], loc='upper right', fontsize=15)
        ax0.set_title(dataset.Ticker[0]+" Stock Linear fit", fontsize=25)
        ax0.tick_params(axis='both', which='major', labelsize=labelsize_)
        ax0.set_xlabel(dataset.Ticker[0]+" Stock Price", fontsize=25, labelpad=10)
        ax0.set_ylabel(dataset.Ticker[0]+" nth stock", fontsize=25, labelpad=10)
        ax0.tick_params(axis='both', which='major', labelsize=labelsize_) 
        ax0.set_ylim([0, 31])
        ax0.set_xlim([0.75*min(mulla), 1.25*min(mulla)])

        plt.tight_layout()
        camera.snap() # take snapshot after each frame/iteration

    anim = camera.animate(blit=False, interval=20)
    #delete video if exists
    import os
    filePath = 'img';
    if os.path.exists('im2ages/'+dataset.Ticker[0]+'.mp4'):
        os.remove('images/'+dataset.Ticker[0]+'.mp4')
    else:
        st.warning('Saving the animation..')
    anim.save('images/'+dataset.Ticker[0]+'.mp4')
    return 200
        
    
# 3D Animation
def cost_3d(x,y,w,b):  # predicts costs for every pair of w and b. 
        pred = x@w.T+b                       
        e=y-pred
        return np.mean(e**2)
    
def three_dimension_animation(dataset):
    # Import libraries
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import celluloid
    from celluloid import Camera
    import pandas_alive
    import matplotlib.animation as manimation
    manimation.writers.list()
    
    # Introduce training data
    x_train = dataset[['high', 'low', 'close']].tail(30)
    
    x_train.reset_index(drop=True, inplace=True)
    x_train.index = x_train.high.tolist()
    mulla = x_train.high.values.reshape(-1 , 1)
    
    x_train = mulla[:30]
    arr = np.array([range(len(mulla[:30]))])
    y_train = arr.reshape(-1, 1)
    
    
    # Introduce lists where data points are being stored: 
    w_list=[]   # list contains weights
    b_list=[]   # list contains biases
    c_list=[]   # list contains costs 
    ys_list=[]  # store arrays of predicted y-values for xs ( -> plot regression line!) 
    cl_list = [] # list contains predicted y-values for x_train ( -> plot connecting lines!) 

    xs= np.array([    # set x-values for regression line plot               
                [0],
                 [3000]
                 ])

    # Train model: 
    model=LinearRegression(w=3,b=-1,lr=0.001) # set initial parameters and learning rate 
    
    # Train model: 
    model=LinearRegression(w=3,b=-1,lr=0.001) # set initial parameters and learning rate 

    for i in range(60000):      # set number of epochs
        w_list.append(model.params()[0])    # append weights (=slopes) to list
        b_list.append(model.params()[1])    # append biases (=y-intercepts) to list
        c_list.append(model.cost(y_train,x_train))  # append costs to list
        ys_list.append(model.predict(xs).T)     # append pairs of predicted y-values for xs 
        cl_list.append(model.predict(x_train).T) # append predicted y-values for x_train to list
        model.fit(y_train, x_train) # fit model


    #make prediction
    # print parameters and costs after all epochs
    print("weight: " + str( model.params()[0]) )  
    print("y-intercept: " + str( model.params()[1]) )
    print("costs: "+ str(model.cost(y_train, x_train)))
    
    # Define which epochs/data points to plot
    a=np.arange(0,50,1).tolist()
    b=np.arange(50,100,5).tolist()
    c=np.arange(100,12000,200).tolist()
    p = a+b+c # points we want to plot

    # Turn lists into arrays
    w= np.array(w_list).flatten()
    b= np.array(b_list).flatten()
    c= np.array(c_list).flatten()
    ys = np.array(ys_list) 
    p=np.array(p)
    
    
    
    ws = np.linspace(-5, 5.0, 10) # set range of values for w and b for surface plot
    bs = np.linspace(-5, 5, 10)
    M, B = np.meshgrid(ws, bs) # create meshgrid

    zs = np.array([cost_3d(x_train,y_train,       # determine costs for each pair of w and b 
            np.array([[wp]]), np.array([[bp]]))  # cost_3d() only accepts wp and bp as matrices. 
                   for wp, bp in zip(np.ravel(M), np.ravel(B))])
    Z = zs.reshape(M.shape) # get z-values for surface plot in shape of M.
    
    # Third Animation
    fig = plt.figure(figsize=(10,10))  
    ax1=fig.add_subplot(121)
    ax1.set_title(dataset.Ticker[0]+" Linear fit", fontsize=30 )
    ax2 = fig.add_subplot(122, projection='3d') # projection='3d'
    ax2.set_title("cost function", fontsize=30)
    ax2.view_init(elev=20., azim=145)           # set view
    camera = Camera(fig)

    for i in p:       
        leg=ax1.plot(xs.T.flatten(),ys[i].flatten(), color='r', label=str(i))  
        ax1.vlines(x_train.T, ymin=y_train.T, ymax=cl_list[i], linestyle="dashed",
                   color='r',alpha=0.3)
        ax1.scatter(x_train, y_train, color='b',marker='x', s=44)
        ax1.legend(leg,[f'epochs: {i}'], loc='upper right', fontsize=15) 
        ax1.set_xlabel(dataset.Ticker[0]+" Stock Price", fontsize=25, labelpad=10)
        ax1.set_ylabel(dataset.Ticker[0]+" nth Stock", fontsize=25, labelpad=10)
        ax1.tick_params(axis='both', which='major', labelsize=15) 
        ax1.set_ylim([0, 31])
        ax1.set_xlim([0.75*min(mulla), 1.25*min(mulla)])

        ax2.plot_surface(M, B, Z, rstride=1, cstride=1, color='b',
                         alpha=0.35) # create surface plot
        ax2.scatter(w[i],b[i],c[i],marker='o', s=12**2, color='orange' )
        ax2.set_xlabel("w", fontsize=25, labelpad=10)
        ax2.set_ylabel("b", fontsize=25, labelpad=10)
        ax2.set_zlabel("costs", fontsize=25,
        labelpad=-35) # negative value for labelpad places z-label left of z-axis.
        ax2.tick_params(axis='both', which='major', labelsize=15) 
        ax2.plot(w[0:i],b[0:i],c[0:i], linestyle="dashed",linewidth=2,color="grey") # (dashed) lineplot

        plt.tight_layout()
        camera.snap()

    animation = camera.animate(interval = 5,repeat = False, repeat_delay = 500)
    #animation.save('SimpleLinReg_3.gif', writer = 'imagemagick')
    anim = camera.animate(blit=False, interval=31)
    #delete video if exists
    import os
    filePath = 'img';
    if os.path.exists('images/'+dataset.Ticker[0]+'2.mp4'):
        os.remove('images/'+dataset.Ticker[0]+'2.mp4')
    else:
        print('Saving the animation..')
    anim.save('images/'+dataset.Ticker[0]+'2.mp4')
    
    return 200


#Training Loss
#Training Loss function 

def getPrice(x, df):
    df = df.loc[(df['Index']  == x['Index'])  & (df['Date'] == x['Date'])]
    if len(df['close'])>0:
        return df['close'].values[0]
    return 0   

#Data processing

def process_time_series(dataset):
    #required librairies 
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    import time
    import numpy as np
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras
    from tensorflow.keras import layers
    
    #Label encoding
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    a = le.fit(dataset['Ticker'])
    le.classes_
    
    #Encode the string to digits
    dataset['Ticker'] = le.transform(dataset['Ticker'])
    
    #Preprocessing
    dataset['Date'] = dataset.index #get date from index
    dataset = dataset.astype({'Date':'datetime64[ns]'}) #format date
    dataset['Year'] = dataset['Date'].dt.year #extract year
    dataset= dataset.reset_index(drop=True) # reset index
    dataset['Index'] = dataset.Ticker
    

        
    return dataset


def final_dataset_pro(donnee):
    #check data
    dataset = process_time_series(donnee)
    
    #Look for last day of an year for an Index
    df_last_date = dataset.groupby(['Ticker', 'Year']).agg({'Date':['max']})

    #reduce the column hierarachy to one.
    df_last_date.columns = df_last_date.columns.get_level_values(0)

    df_last_date.reset_index(inplace = True)


    return dataset


def concise_data(dataset):    
    #Rebuild preprocessed
    fetched = final_dataset_pro(dataset)

    #Look for last day of an year for an Index
    df_last_date = fetched.groupby(['Ticker', 'Year']).agg({'Date':['max']})

    #Look for the price in the main df dataframe for last date of a year for an Index
    #fetched['Price'] = df_last_date.apply(lambda x: getPrice(x, fetched), axis = 1)
    datacom = fetched[['Index', 'open', 'high', 'low', 'close', 'adjClose',
                              'volume', 'Year']]
    datacom['Price'] = datacom.close
    return datacom


def training_loss(data):
    #required librairies 
    import sys
    import warnings

    #if not sys.warnoptions:
    #    warnings.simplefilter("ignore")
    import time
    import numpy as np
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras
    from tensorflow.keras import layers
    import streamlit as st
    
    #processed data
    
    dataset = concise_data(data)
    
    # Create training and validation splits
    df_train = dataset.sample(frac=0.7, random_state=0)
    df_valid = dataset.drop(df_train.index)
    #st.write(df_train.head(2))
    
    # Split features and target
    X_train = df_train.drop('Price', axis=1)
    X_valid = df_valid.drop('Price', axis=1)

    y_train = df_train['Price']
    y_valid = df_valid['Price']
    
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=[8]),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1),
    ])
    
    #After defining the model, we compile in the optimizer and loss function.
    model.compile(
        optimizer='adam',
        loss='mae',
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=256,
        epochs=20,
    )
    
    return history 

def training_log_catching(dataset):
    import wandb
    import streamlit as st
    
    #deactivate login
    import logging
    logger = logging.getLogger("wandb")
    logger.setLevel(logging.WARNING)


    #init wandb
    WANDB_API_KEY='b7efaac6cd672df5b932e26a7b9271852962958c'
    WANDB_NAME="DataDrivenAppUsingOptimalTransportAlgorithms"
    WANDB_NOTES="Smaller learning rate, more regularization."
    WANDB_ENTITY='rmastour'

    api = wandb.Api()


    wandb.init()
    # define our custom x axis metric
    wandb.define_metric("custom_step")
    # define which metrics will be plotted against it
    wandb.define_metric("validation_loss", step_metric="custom_step")



    for i in range(len(dataset)):
        log_dict = {
            "train_loss": dataset.loss[i],
            "validation_loss": dataset.val_loss[i]
        }

    st.write(wandb.log(log_dict))
    return 200   



#testing regression model
def regression_model(dataset):
    #required librairies
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as seabornInstance
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    
    #defining the coordinates 
    X = dataset['low'].values.reshape(-1,1).astype('float32')
    y = dataset['high'].values.reshape(-1,1).astype('float32')
    
    #visualization
    #fig,(ax1) = plt.subplots(1, figsize = (12,6))
    #ax1.scatter (X, y, s = 8)
    #plt.title ('Min vs Max Stock Rate')
    #plt.xlabel('low')
    #plt.ylabel('high')
    #plt.show()

        
    #splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    
    #Creating the prediction model:
    h = LinearRegression()
    h.fit(X_train,y_train)
    #print("The regression model line y-intercept h.intercept_ = : ", h.intercept_) # to retrieve theta_0
    #print("The regression model line y-intercept h.coef_ = : ", h.coef_) # to retrieve theta_1
    
    #Prediction table 
    y_pred = h.predict(X_test)
    compare = pd.DataFrame({'Current Month Stock Price': y_test.flatten(), 'Linear_Regression_Prediction': y_pred.flatten()})
    
    #test data line 
    #fig,(ax1) = plt.subplots(1, figsize = (12,6))
    #ax1.scatter (X_test, y_test, s = 8)
    #plt.plot(X_test,y_pred, color = 'red', linewidth = 2)
    #plt.title ('Predicted points')
    #plt.xlabel('low')
    #plt.ylabel('high')
    #plt.show()
    
    return compare, y_pred, X_test


#Cost evaluators:MSE
def mean_square_error(predictions, targets):
    # Retrieving number of samples in dataset
    samples_num = len(predictions)
    
    # Summing square differences between predicted and expected values
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += (prediction - target)**2
        
    # Calculating mean and dividing by 2
    mae_error = (1.0 / (2*samples_num)) * accumulated_error
    
    return mae_error  

#Cost evaluators: MAE
def mean_absolute_error(predictions, targets):
    # Retrieving number of samples in dataset
    samples_num = len(predictions)
    
    # Summing absolute differences between predicted and expected values
    accumulated_error = 0.0
    for prediction, target in zip(predictions, targets):
        accumulated_error += np.abs(prediction - target)
        
    # Calculating mean
    mae_error = (1.0 / samples_num) * accumulated_error
    
    return mae_error   


#iterative cost with training line 
def interactive_cost_coef(dataset):
    #required librairies
    import streamlit as st
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    #linear regression
    #pick some random value to start with
    theta_0 = np.random.random()
    theta_1 = np.random.random()
    #defining the coordinates 
    X = dataset['low'].values.reshape(-1,1).astype('float32')
    y = dataset['high'].values.reshape(-1,1).astype('float32')

    #required sub functions 
    def hypothesis (theta_0,theta_1,X):
        return theta_1*X + theta_0


    def cost_function (X,y,theta_0,theta_1):
        m = len(X)
        summation = 0.0
        for i in range (m):
            summation += ((theta_1 * X[i] + theta_0) - y[i])**2
        return summation /(2*m)


    def gradient_descent(X,y,theta_0,theta_1,learning_rate):
        t0_deriv = 0
        t1_deriv = 0
        m = len(X)

        for i in range (m):
            t0_deriv += (theta_1 * X[i] + theta_0) - y[i]
            t1_deriv += ((theta_1 * X[i] + theta_0) - y[i])* X[i]
        theta_0 -= (1/m) * learning_rate * t0_deriv
        theta_1 -= (1/m) * learning_rate * t1_deriv

        return theta_0,theta_1


    def training (X, y, theta_0, theta_1, learning_rate, iters):
        cost_history = [0]
        t0_history = [0]
        t1_history = [0]

        for i in range(iters):
            theta_0,theta_1 = gradient_descent(X, y, theta_0, theta_1, learning_rate)
            t0_history.append(theta_0)
            t1_history.append(theta_1)
            cost = cost_function(X, y, theta_0, theta_1)
            cost_history.append(cost)
            if i%10 == 0:
                st.info("iter={}, theta_0={}, theta_1={}, cost= {}".format(i, theta_0, theta_1, cost))
        #st.warning((t0_history, t1_history, cost_history))
        return t0_history, t1_history, cost_history
    
    #25 iterations
    
    
    t0_history, t1_history, cost_history = training (X, y, theta_0, theta_1, 0.08, 100)
    #Plot the cost function
    plt.title('Cost Function C')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(cost_history)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.show()
    st.pyplot()
    
    return 200

    


# Optimal Transport Functions
#===============================
## Optimal Trasport Pediction 
# a and b are 1D histograms (sum to 1 and positive)
# M is the ground cost matrix

def pyOT_sinkhorn(dataset, size):
    #required librairies
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    import numpy as np
    import pandas as pd
    import streamlit as st
    import ot   #Optimal trasport pip install POT
    from sklearn.metrics.pairwise import euclidean_distances
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from layers import SinkhornDistance
    np.random.seed(42)
    
    #Parameters
    n = len(dataset)
    
    #Wasserstein distances
    a= dataset['low'].values.reshape(-1,1).astype('float32')
    b= dataset['high'].values.reshape(-1,1).astype('float32')
    
    #====================================================
    # Wrap with torch tensors
    x = torch.tensor(a[:size], dtype=torch.float)
    y = torch.tensor(b[:size], dtype=torch.float)

    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100)
    dist, P, C = sinkhorn(x, y)
    print("Sinkhorn distance: {:.3f}".format(dist.item()))
    
    return round(dist.item(),3), C, P
    
#Optimal Trasport Pediction 
# a and b are 1D histograms (sum to 1 and positive)
# M is the ground cost matrix

def pyOT_Distance_matrix(dataset, size):
    #required librairies
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    C= pyOT_sinkhorn(dataset, size)[1]
    #====================================================

    plt.imshow(C)
    plt.title('Distance matrix')
    plt.colorbar();
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.show()
    st.pyplot()
    
    return 200
    
#Optimal Trasport Pediction 
# a and b are 1D histograms (sum to 1 and positive)
# M is the ground cost matrix

def pyOT_coupline_matrix(dataset, size):
    #required librairies
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    P= pyOT_sinkhorn(dataset, size)[2]
    #====================================================

    plt.imshow(P)
    plt.title('Coupling matrix');
    plt.colorbar();
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.show()
    st.pyplot()
    
    return 200
    
    
#moving_probability_masses
def moving_probability_masses(dataset, size):
    #required librairies
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    import streamlit as st
    import ot   #Optimal trasport pip install POT
    import matplotlib.pyplot as plt
    np.random.seed(42)
    
    #Parameters
    n = len(dataset)
    
    #Presentation
    st.info('''
    Many problems in machine learning deal with the idea of making two probability distributions to be as close 
    as possible. In the simpler case where we only have observed variables x (say, Stock prices at a given time) coming from 
    an unknown distribution p(x), we’d like to find a model q(x|θ) (like a neural network) that is a good 
    approximation of p(x). It can be shown1 that minimizing KL(p∥q) is equivalent to minimizing the negative 
    log-likelihood, which is what we usually do when training a classifier, for example. 
    In the case of the Variational Autoencoder, we want the approximate posterior to be close to some 
    prior distribution, which we achieve, again, by minimizing the KL divergence between them.
    ''')
    #Wasserstein distances
    a= dataset['low'].values.reshape(-1,1).astype('float32')
    b= dataset['high'].values.reshape(-1,1).astype('float32')    
    
    plt.figure(figsize=(18, 9))
    plt.scatter(a[:size], a[:size], label='supp($p(x)$)')
    plt.scatter(b[:size], b[:size], label='supp($q(x)$)')
    plt.xlabel('No. of iterations')
    plt.ylabel('Probability Matching at the nth iteration')
    plt.legend();
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.show()
    st.pyplot()
    
    return 200
    
    
#Conclusive Dataframe
def comparative_dataframes(dataset):
    import numpy as np
    import matplotlib.pylab as pl
    import ot
    # necessary for 3d plot even if not used
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from matplotlib.collections import PolyCollection  # noqa
    
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        
    #%% parameters
    problems = []
    n = 100  # nb bins

    # bin positions
    x = np.arange(n, dtype=np.float64)

    #definining the datapoints:
    ##############################################################################
    # Gaussian Data
    # -------------

    # Gaussian distributions
    a1 = dataset['close'].to_numpy()
    a1= a1[:n]
    a2 = dataset["high"].to_numpy()
    a2 = a2[:n]


    # creating matrix A containing all distributions
    A = np.vstack((a1, a2)).T
    n_distributions = A.shape[1]

    # loss matrix + normalization
    M = ot.utils.dist0(n)
    M /= M.max()


    alpha = 0.5  # 0<=alpha<=1
    weights = np.array([1 - alpha, alpha])

    # l2bary
    bary_l2 = A.dot(weights)

    # wasserstein
    reg = 1e-3
    ot.tic()
    bary_wass = ot.bregman.barycenter(A, M, reg, weights)
    ot.toc()


    ot.tic()
    bary_wass2 = ot.lp.barycenter(A, M, weights, solver='interior-point', verbose=True)
    ot.toc()

    nbm = len(problems)
    nbm2 = (nbm // 2)


    pl.figure(2, (20, 6))
    pl.clf()
        
    return A, bary_l2, bary_wass, bary_wass2

def return_conclusive_dataframe(dataset):
    #Linear regression:
    Linear_Regrassion= pd.DataFrame(regression_model(dataset)[0])
    Linear_Regrassion= Linear_Regrassion.tail(30).reset_index()
    Linear_Regrassion = Linear_Regrassion.drop('index', 1)
    
    
    #Regular Euclidean
    Current_to_euler_cost = pd.DataFrame(comparative_dataframes(dataset)[0])
    Current_to_euler_cost = Current_to_euler_cost.tail(30).reset_index()
    #remove the old index column
    Current_to_euler_cost = Current_to_euler_cost.drop('index', 1)
    #convert int column name to str
    Current_to_euler_cost.columns = Current_to_euler_cost.columns.map(str)
    # rename the 0 columns
    Current_to_euler_cost= Current_to_euler_cost.rename({'0': 'With_Euler_Cost'}, axis='columns')
    Current_to_euler_cost = Current_to_euler_cost.rename({'1': 'Smoothened_Euclidean_Cost'}, axis='columns')
    
    #Extended Cost
    Current_to_extended_cost = pd.DataFrame(comparative_dataframes(dataset)[1])
    Current_to_extended_cost = Current_to_extended_cost.tail(30).reset_index()
    #remove the old index column
    Current_to_extended_cost = Current_to_extended_cost.drop('index', 1)
    #convert int column name to str
    Current_to_extended_cost.columns = Current_to_extended_cost.columns.map(str)
    # rename the 0 columns
    Current_to_extended_cost = Current_to_extended_cost.rename({'0': 'With_Extended_Cost'}, axis='columns')
    
    #Regularized Sinkhorn
    Current_to_Regularized_sinkhorn= pd.DataFrame(comparative_dataframes(dataset)[2])
    Current_to_Regularized_sinkhorn= Current_to_Regularized_sinkhorn.tail(30).reset_index()
    #remove the old index column
    Current_to_Regularized_sinkhorn = Current_to_Regularized_sinkhorn.drop('index', 1)
    #convert int column name to str
    Current_to_Regularized_sinkhorn.columns = Current_to_Regularized_sinkhorn.columns.map(str)
    # rename the 0 columns
    Current_to_Regularized_sinkhorn = Current_to_Regularized_sinkhorn.rename({'0': 'Regularized_Sinkhorn'}, axis='columns')

    #Stockastic Gradient Descent
    Current_to_sgd_wassertein= pd.DataFrame(comparative_dataframes(dataset)[3])
    Current_to_sgd_wassertein= Current_to_sgd_wassertein.tail(30).reset_index()
    #remove the old index column
    Current_to_sgd_wassertein = Current_to_sgd_wassertein.drop('index', 1)
    #convert int column name to str
    Current_to_sgd_wassertein.columns = Current_to_sgd_wassertein.columns.map(str)
    # rename the 0 columns
    Current_to_sgd_wassertein= Current_to_sgd_wassertein.rename({'0': 'SGD_Wasserstein'}, axis='columns')
    
    #Merging
    # compile the list of dataframes you want to merge
    data_frames = [Linear_Regrassion, 
                   Current_to_euler_cost, 
                   Current_to_extended_cost, 
                   Current_to_Regularized_sinkhorn, 
                   Current_to_sgd_wassertein]

    #Merging
    merged_df = pd.concat(data_frames, axis=1)
    
    return merged_df


#plotly graph
def plotly_3d(dataset):
    import os
    import glob
    import time
    import multiprocessing

    import streamlit as st
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    #from Inputs_Parallel import get_possible_scenarios

    # Side Bar #######################################################

    gcr_config = st.sidebar.slider(label="Ground Coverage Ratio Range Selection",
                                   min_value=1,
                                   max_value=29,
                                   step=1,
                                   value=(1, 30))

    sr_config = st.sidebar.slider(label="Sizing Ratio Range Selection",
                                  min_value=1.0,
                                  max_value=2.0,
                                  step=0.1,
                                  value=(1.0, 1.5))

    run_button = st.sidebar.button(label='Run Optimization')

    progress_bar = st.sidebar.progress(0)

    # App ###########################################################
    # Graphing Function #####
    z_data = return_conclusive_dataframe(dataset)
    z = z_data.values
    sh_0, sh_1 = z.shape
    x, y = np.linspace(1, 30, sh_0), np.linspace(1, 30, sh_1)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='Stock Price Forcasting With Smoothened Extended Cost : Sinkhorn, Wassertein',
                      scene = dict(
                      xaxis_title='X Various Extended Cost Functions',
                      yaxis_title='Y Over 30-Days Forcasting',
                      zaxis_title='Z Smoothed Stock Price Margin'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)
    
    return 200


# ========================================  INTORODUCTORY SPACE: ==================================================:
DATA_URL = df
st.markdown("# Distributional barycenter problem through data-driven flows in Continuous time- By Joseph Bunster")
st.markdown("Explore the dataset to know more about OT based Financial Time series Dataset Linear Regression")
img=Image.open('images/carbon.png')
st.image(img,width=700)

#Presentation paragraph
import subprocess
st.markdown('''
**Abstract**
A new method is proposed for the solution of the data-driven optimal transport barycenter problem and of the more general distributional barycenter problem that the article introduces. The method improves on previous approaches based
on adversarial games, by slaving the discriminator to the generator, minimizing the need for parameterizations and by allowing the adoption of general cost functions. It is applied to numerical examples, which include analyzing the
MNIST data set with a new cost function that penalizes non-isometric maps. 

***Keywords:*** Optimal transport, barycenter problem, pattern visualization, filtering, adversarial optimization
''')
path_to_pdf =  'files/Distributional barycenter problem through data-driven flows.pdf'
#action = subprocess.Popen([path_to_pdf], shell=True)
if st.button('Download the related research article'):
    subprocess.Popen([path_to_pdf], shell=True)

st.markdown("The data presented is of 5 different companies - **Microsoft, Apple, Tesla, Google and Amazon,** collected from Tiingo API **https://www.tiingo.com.**")


# Button to present Joseph Bunster:

if st.button("Learn more about Joseph Bunster and data processed"):
    img=Image.open('images/author.png')
    st.markdown("**Joseph Bunster ** is a hardworking mathematics professional, passionate about applying his technical background to solving real world problems. he enjoys challenges and thrive under pressure, these traits helped him successfully compete in the 2018 US National Collegiate Mathematics Championship where he placed 3rd in the United States. Bunster is currently enrolled in a Masters of Science in Mathematics Program at NYU-Courant, with an expected graduation date of January 2022. He has previous research experience in Financial Engineering,Optimal Control Theory, and Reinforcement Learning.Interests in Applied Math, Probability, Optimization and Finance..")
    st.image(img,width=200, caption="Joe Bunster 🤵‍")
    st.markdown("The data was collected and made available by **[Joseph Bunster](https://www.linkedin.com/in/joseph-bunster/)**.")
    images=Image.open('images/tiingo.png')
    st.image(images,width=600)
    #Ballons
    st.balloons()


# Project Abstract:
st.info(''' The optimal transport theory is the study of optimal transportation and allocation between measures.
The optimal transport problem was first introduced by Monge (1781) and formalized by Kantorovitch (1942), leading to the so called Monge-Kantorovitch transportation problem.
The goal is to look for a transport map transforming a probability density function into another while minimizing the cost of transport.
''')
img=Image.open("images/OT.jpg")
st.image(img,width=700)


# Designing the side panel:
st.sidebar.markdown("## Side Panel")
st.sidebar.markdown("Use this panel to explore the dataset and create own viz.")
st.header("Now, Explore Yourself the Time Series Dataset")


# I- Presenting the dataset
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading timeseries dataset...')

# Load 10,000 rows of data into the dataframe.
# df 
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading timeseries dataset...Completed!')
images=Image.open('images/bot.png')
st.image(images,width=150)


# Showing the original raw data
st.title('Quick  Explore')
st.sidebar.subheader(' Quick  Explore')
st.markdown("Tick the box on the side panel to explore the dataset.")


if st.sidebar.checkbox('Basic info'):
    if st.sidebar.checkbox('Quick Look'):
        st.subheader('Dataset Quick Look:')
        st.write(df.head())
    if st.sidebar.checkbox("Show Columns"):
        st.subheader('Show Columns List')
        all_columns = df.columns.to_list()
        st.write(all_columns)
   
    if st.sidebar.checkbox('Statistical Description'):
        st.subheader('Statistical Data Descripition')
        st.write(df.describe())
    if st.sidebar.checkbox('Missing Values?'):
        st.subheader('Missing values')
        st.write(df.isnull().sum())
if st.sidebar.checkbox('Dataset Another Quick Look'):
    st.subheader('Dataset Quick Look:')
    st.write(df.head())

    
    
    
    
  
 
    
# ========================================  Main Function SPACE: ==================================================:
        
def main():
    #data partitions
    apple = df[df['Ticker'].str.contains('AAPL')]
    microsoft = df[df['Ticker'].str.contains('MSFT')]
    amazon = df[df['Ticker'].str.contains('AMZN')]
    google = df[df['Ticker'].str.contains('Goog')]
    tesla = df[df['Ticker'].str.contains('TSLA')]
    
    # Exploring the datset:
    st.title('🛑 A- Let\'s explore the time series data per Company name: ')
    st.markdown("Novigate through this drop down to display the data per Stock company name")
    mydata = st.selectbox("Choose your dataset:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("Show Dataset"):
        if mydata == 'Microsoft':
            st.write(df[df['Ticker'].str.contains('MSFT')])
        elif mydata == 'Apple':
            st.write(df[df['Ticker'].str.contains('AAPL')])
        elif mydata == 'Tesla':
            st.write(df[df['Ticker'].str.contains('TSLA')])
        elif mydata == 'Google':
            st.write(df[df['Ticker'].str.contains('Goog')])            
        elif mydata == 'Amazon':
            st.write(df[df['Ticker'].str.contains('AMZN')])
        st.success('You successfully showcased the dataset per company_name for the last 6 months:')
        
    # Exploring the datset:
    st.title('🛑 B- OT Algorithm and Prerequisites: ')
    # I- Cost Function:
    #=================
    st.subheader("I - 📈 Visualization:")
    
    st.markdown("Navigate through this drop down to display the data per Stock company name")
    mydata = st.selectbox("Choose a company:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("Visualize"):
        if mydata == 'Apple':
            st.info("Computing the Euclidian distance for "+df[df['Ticker'].str.contains('AAPL')].Ticker[0]+" stock")
            # plot of Apple Price for last six month
            company = price_plotting(df, 'AAPL')[0]            
            time = price_plotting(df, 'AAPL')[1]
            plt.plot(company)
            plt.title(df[df['Ticker'].str.contains('AAPL')].Ticker[0]+" Price (for the last 6 months)")
            plt.xlabel("Time")
            plt.ylabel("Price in $")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            
            
            # Converting prices to returns 
            financial_returns(company)
            
            # graph of returns Z
            returns = financial_returns(company)
            plt.plot(returns, "g")
            plt.title("Z variable ~ Apple Returns in 6 months")
            plt.xlabel("Time")
            plt.ylabel(" % Return")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            st.write("The std of Z is: ", np.std(returns))
            
            plt.hist(returns, bins='auto')
            plt.title("Z variable ~ Apple Returns in 6 months")
            plt.ylabel("Frequency")
            plt.xlabel("% Return")
            plt.show()
            st.pyplot()
            

            plt.hist(returns, bins = 'auto')
            plt.title("Kernel Density Estimation ~ Apple Returns in 6 months")
            sns.kdeplot(returns)
            plt.show()
            st.pyplot()
            
            st.warning("Selling price Optimization for Apple")
            st.write(stochastic_gradient_descent(apple))
  
            
            
        elif mydata == 'Microsoft':
            st.info("Computing the Euclidian distance for "+df[df['Ticker'].str.contains('MSFT')].Ticker[0]+" stock")
            # plot of Apple Price for last six month
            company = price_plotting(df, 'MSFT')[0]
            time = price_plotting(df, 'MSFT')[1]
            plt.plot(company)
            plt.title(df[df['Ticker'].str.contains('MSFT')].Ticker[0]+" Price (for the last 6 months)")
            plt.xlabel("Time")
            plt.ylabel("Price in $")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            showPyplotGlobalUse = False
            st.pyplot()
            
            # Converting prices to returns 
            financial_returns(company)
            
            # graph of returns Z
            returns = financial_returns(company)
            plt.plot(returns, "g")
            plt.title("Z variable ~ Microsoft Returns in 6 months")
            plt.xlabel("Time")
            plt.ylabel(" % Return")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            st.write("The std of Z is: ", np.std(returns))
            
            plt.hist(returns, bins='auto')
            plt.title("Z variable ~ Microsoft Returns in 6 months")
            plt.ylabel("Frequency")
            plt.xlabel("% Return")
            plt.show()
            st.pyplot()
            

            plt.hist(returns, bins = 'auto')
            plt.title("Kernel Density Estimation ~ Microsoft Returns in 6 months")
            sns.kdeplot(returns)
            plt.show()
            st.pyplot()
            st.warning("Selling price Optimization for Microsoft")
            st.write(stochastic_gradient_descent(microsoft))
            
            
        elif mydata == 'Tesla':
            st.info("Computing the Euclidian distance for "+df[df['Ticker'].str.contains('TSLA')].Ticker[0]+" stock")
            # plot of Apple Price for last six month
            company = price_plotting(df, 'TSLA')[0]
            time = price_plotting(df, 'TSLA')[1]
            plt.plot(company)
            plt.title(df[df['Ticker'].str.contains('TSLA')].Ticker[0]+" Price (for the last 6 months)")
            plt.xlabel("Time")
            plt.ylabel("Price in $")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            showPyplotGlobalUse = False
            st.pyplot() 
            
            # Converting prices to returns 
            financial_returns(company)
            
            # graph of returns Z
            returns = financial_returns(company)
            plt.plot(returns, "g")
            plt.title("Z variable ~ Tesla Returns in 6 months")
            plt.xlabel("Time")
            plt.ylabel(" % Return")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            st.write("The std of Z is: ", np.std(returns))
            
            plt.hist(returns, bins='auto')
            plt.title("Z variable ~ Tesla Returns in 6 months")
            plt.ylabel("Frequency")
            plt.xlabel("% Return")
            plt.show()
            st.pyplot()
            

            plt.hist(returns, bins = 'auto')
            plt.title("Kernel Density Estimation ~ Tesla Returns in 6 months")
            sns.kdeplot(returns)
            plt.show()
            st.pyplot()
            
            st.warning("Selling price Optimization for Tesla")
            st.write(stochastic_gradient_descent(tesla))
            
            
        elif mydata == 'Google':
            st.info("Computing the Euclidian distance for "+df[df['Ticker'].str.contains('Goog')].Ticker[0]+" stock")
            # plot of Apple Price for last six month
            company = price_plotting(df, 'Goog')[0]
            time = price_plotting(df, 'Goog')[1]
            plt.plot(company)
            plt.title(df[df['Ticker'].str.contains('Goog')].Ticker[0]+" Price (for the last 6 months)")
            plt.xlabel("Time")
            plt.ylabel("Price in $")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            showPyplotGlobalUse = False
            st.pyplot()            
            # Converting prices to returns 
            financial_returns(company)
            
            # graph of returns Z
            returns = financial_returns(company)
            plt.plot(returns, "g")
            plt.title("Z variable ~ Google Returns in 6 months")
            plt.xlabel("Time")
            plt.ylabel(" % Return")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            st.write("The std of Z is: ", np.std(returns))
            
            plt.hist(returns, bins='auto')
            plt.title("Z variable ~ Google Returns in 6 months")
            plt.ylabel("Frequency")
            plt.xlabel("% Return")
            plt.show()
            st.pyplot()
            

            plt.hist(returns, bins = 'auto')
            plt.title("Kernel Density Estimation ~ Google Returns in 6 months")
            sns.kdeplot(returns)
            plt.show()
            st.pyplot()
            
            st.warning("Selling price Optimization for Google")
            st.write(stochastic_gradient_descent(google))
            
        elif mydata == 'Amazon':
            st.info("Computing the Euclidian distance for "+df[df['Ticker'].str.contains('AMZN')].Ticker[0]+" stock")
            # plot of Apple Price for last six month
            company = price_plotting(df, 'AMZN')[0]
            time = price_plotting(df, 'AMZN')[1]
            plt.plot(company)
            plt.title(df[df['Ticker'].str.contains('AMZN')].Ticker[0]+" Price (for the last 6 months)")
            plt.xlabel("Time")
            plt.ylabel("Price in $")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            showPyplotGlobalUse = False
            st.pyplot()  
            # Converting prices to returns 
            financial_returns(company)
            
            # graph of returns Z
            returns = financial_returns(company)
            plt.plot(returns, "g")
            plt.title("Z variable ~ Amazon Returns in 6 months")
            plt.xlabel("Time")
            plt.ylabel(" % Return")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            st.write("The std of Z is: ", np.std(returns))
            
            plt.hist(returns, bins='auto')
            plt.title("Z variable ~ Amazon Returns in 6 months")
            plt.ylabel("Frequency")
            plt.xlabel("% Return")
            plt.show()
            st.pyplot()
            

            plt.hist(returns, bins = 'auto')
            plt.title("Kernel Density Estimation ~ Amazon Returns in 6 months")
            sns.kdeplot(returns)
            plt.show()
            st.pyplot()
            
            st.warning("Selling price Optimization for Amazon")
            st.write(stochastic_gradient_descent(amazon))
            
        st.success('You successfully showcased the dataset plots per company_name for the last 6 months:')   
        

    # I- Cost Function:
    #=================
    st.subheader("II - 🧪 Cost Function:")
    st.warning('''
    It is a function that measures the performance of a Machine Learning model for given data. Cost Function quantifies the error between predicted values and expected values and presents it in the form of a single real number. Depending on the problem Cost Function can be formed in many different ways. The purpose of Cost Function is to be either:
- **Minimized** - then returned value is usually called cost, loss or error. The goal is to find the values of model parameters for which Cost Function return as small number as possible.
- **Maximized** - then the value it yields is named a reward. The goal is to find values of model parameters for which returned number is as large as possible.

For algorithms relying on Gradient Descent to optimize model parameters, every function has to be differentiable.

$$C(y(x,k), ρ)) = 1/N^{2} \sum_{1\le i\not\equiv j\le N} [[{\parallel y^{k}_i - y^{k}_j \parallel ^{2} }\div ({x^{k}_i - x^{k}_j \parallel ^{2} + \epsilon^{2})} - 1]] + \omega * 1/N \sum_{i=1}^{N} \parallel y^{k}_i - x^{k}_i \parallel ^{2}$$
    ''')

    # 1- Cost function 1: Euclidian Distance
    token_text = '<p style="color:green; font-size: 20px;"><b>1- Cost function 1: Euclidian Distance</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    st.markdown("Navigate through this drop down to display the cost function outcome per company name")
    mydata = st.selectbox("Select a company:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("Compute"):
        if mydata == 'Apple':
            close = apple.close
            high = apple.high
            # changing data from pandas data series to numpy array
            HIGH = np.array(high)
            CLOSE = np.array(close)
            st.success("Euclidian distance for Apple:")
            st.write(euclidian_distance(HIGH, CLOSE))
            
        elif mydata == 'Microsoft':
            close = microsoft.close
            high = microsoft.high
            # changing data from pandas data series to numpy array
            HIGH = np.array(high)
            CLOSE = np.array(close)
            st.success("Euclidian distance for Microsoft:")
            st.write(euclidian_distance(HIGH, CLOSE))            
            
        elif mydata == 'Tesla':
            close = tesla.close
            high = tesla.high
            # changing data from pandas data series to numpy array
            HIGH = np.array(high)
            CLOSE = np.array(close)
            st.success("Euclidian distance for Tesla:")
            st.write(euclidian_distance(HIGH, CLOSE))   
            
        elif mydata == 'Google':
            close = google.close
            high = google.high
            # changing data from pandas data series to numpy array
            HIGH = np.array(high)
            CLOSE = np.array(close)
            st.success("Euclidian distance for Google:")
            st.write(euclidian_distance(HIGH, CLOSE))
            
        elif mydata == 'Amazon':
            close = amazon.close
            high = amazon.high
            # changing data from pandas data series to numpy array
            HIGH = np.array(high)
            CLOSE = np.array(close)
            st.success("Euclidian distance for Amazon:")
            st.write(euclidian_distance(HIGH, CLOSE))            
            
            
            
    # 2- Cost function 2: Wasserstein Distance
    token_text = '<p style="color:green; font-size: 20px;"><b>2- Cost function 2: Wasserstein Distance</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    st.markdown("Navigate through this drop down to display the cost function outcome per company name")
    st.info('''
             If we assume the supports for p(x) and q(x) are time_Series_for_highest_stock_price and time_Series_for_lowest_stock_price, respectively, the cost matrix is: C

        With these definitions, the total cost can be calculated as the [Frobenius inner](https://en.wikipedia.org/wiki/Frobenius_inner_product) product between P
        and C : 

        $$<C,P>=\sum_{ij} C_{ij}P_{ij}$$

        As you might have noticed, there are actually multiple ways to move points from one support to the other, each one yielding different costs. The one above is just one example, but we are interested in the assignment that results in the smaller cost. This is the problem of optimal transport between two discrete distributions, and its solution is the lowest cost LC
        over all possible coupling matrices. This last condition introduces a constraint in the problem, because not any matrix is a valid coupling matrix. For a coupling matrix, all its columns must add to a vector containing the probability masses for p(x), and all its rows must add to a vector with the probability masses for q(x). In our example, these vectors contain 4 elements, all with a value of 1/4. More generally, we can let these two vectors be a and b

        , respectively, so the optimal transport problem can be written as: 

        $$L_{C}=min_{P}⟨C,P⟩$$

        subject to $$P1=a$$ 

        $$P^{⊤}1=b$$

        When the distance matrix is based on a valid [distance function](https://en.wikipedia.org/wiki/Metric_(mathematics)), the minimum cost is known as the Wasserstein distance.

            ''') 
    mydata = st.radio("Select a stock out of .. :",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if mydata == 'Apple':
            st.warning("Wasserstein distance for time series data:")
            count_points = st.slider('Select a range of time series data points', min(apple.high), max(apple.high), (min(apple.high), max(apple.high)))
            if count_points == count_points:
                st.write("- Values:", count_points)
                counter = int(round((max(count_points) - min(count_points)),0)*4)
                st.write("- Count of selected points is:", counter)
                st.success("Wasserstein Distance:")
                pyOT_sinkhorn(apple, counter)[0]
                st.success("Probability Distribution:")
                moving_probability_masses(apple, counter)
                st.success("Distance Matrix:")
                pyOT_Distance_matrix(apple, counter)
                st.success("Coupling Matrix:")
                pyOT_coupline_matrix(apple, counter)
    
    elif mydata == 'Microsoft':
            st.warning("Microsoft Wasserstein distance for time series data:")   
            count_points = st.slider('Select a range of time series data points', min(microsoft.high), max(microsoft.high), (min(microsoft.high), max(microsoft.high)))
            st.write("- Values:", count_points)
            counter = int(round((max(count_points) - min(count_points)),0)*4)
            st.write("- Count of selected points is:", counter)
            st.success("Wasserstein Distance:")
            pyOT_sinkhorn(microsoft, counter)[0]
            st.success("Probability Distribution:")
            moving_probability_masses(microsoft, counter)
            st.success("Distance Matrix:")
            pyOT_Distance_matrix(microsoft, counter)
            st.success("Coupling Matrix:")
            pyOT_coupline_matrix(microsoft, counter)      
    elif mydata == 'Tesla':
            st.warning("Tesla Wasserstein distance for time series data:")   
            count_points = st.slider('Select a range of time series data points', min(tesla.high), max(tesla.high), (min(tesla.high), max(tesla.high)))
            st.write("- Values:", count_points)
            counter = int(round((max(count_points) - min(count_points)),0)*4)
            st.write("- Count of selected points is:", counter)
            st.success("Wasserstein Distance:")
            pyOT_sinkhorn(tesla, counter)[0]
            st.success("Probability Distribution:")
            moving_probability_masses(tesla, counter)
            st.success("Distance Matrix:")
            pyOT_Distance_matrix(tesla, counter)
            st.success("Coupling Matrix:")
            pyOT_coupline_matrix(tesla, counter) 
            
    elif mydata == 'Google':
            st.warning("Google Wasserstein distance for time series data:")  
            count_points = st.slider('Select a range of time series data points', min(google.high), max(google.high), (min(google.high), max(google.high)))
            st.write("- Values:", count_points)
            counter = int(round((max(count_points) - min(count_points)),0)*4)
            st.write("- Count of selected points is:", counter)
            st.success("Wasserstein Distance:")
            pyOT_sinkhorn(google, counter)[0]
            st.success("Probability Distribution:")
            moving_probability_masses(google, counter)
            st.success("Distance Matrix:")
            pyOT_Distance_matrix(google, counter)
            st.success("Coupling Matrix:")
            pyOT_coupline_matrix(google, counter)
            
    elif mydata == 'Amazon':
            st.warning("Amazon Wasserstein distance for time series data:")
   
            count_points = st.slider('Select a range of time series data points', min(amazon.high), max(amazon.high), (min(amazon.high), max(amazon.high)))
            st.write("- Values:", count_points)
            counter = int(round((max(count_points) - min(count_points)),0)*4)
            st.write("- Count of selected points is:", counter)
            st.success("Wasserstein Distance:")
            pyOT_sinkhorn(amazon, counter)[0]
            st.success("Probability Distribution:")
            moving_probability_masses(amazon, counter)
            st.success("Distance Matrix:")
            pyOT_Distance_matrix(amazon, counter)
            st.success("Coupling Matrix:")
            pyOT_coupline_matrix(amazon, counter)    

    
 
    # 3- Cost function 3: Evaluating the cost function through MAE and MSE
    token_text = '<p style="color:green; font-size: 20px;"><b>3- Evaluating the cost function through MAE and MSE</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    mydata = st.selectbox("Evaluate the following dataset:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("Iterate.."):
        if mydata == 'Apple':
            st.success("Linear regression for Apple:")  
            #defining the coordinates 
            X = apple['low'].values.reshape(-1,1).astype('float32')
            y = apple['high'].values.reshape(-1,1).astype('float32')
            X_test = regression_model(apple)[2]
            y_pred = regression_model(apple)[1]

            #visualization
            fig,(ax1) = plt.subplots(1, figsize = (12,6))
            ax1.scatter (X, y, s = 8)
            plt.plot(X_test,y_pred, color = 'red', linewidth = 2)
            plt.title ('Min vs Max Stock Rate')
            plt.xlabel('Targeted Stock price')
            plt.ylabel('high')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            showPyplotGlobalUse = False
            st.pyplot()
            
            #Performance metrics
            st.success("Cost evaluators: MAE and MSE for Apple:")
            st.write("MAE is :", mean_absolute_error(regression_model(apple)[1][0], regression_model(apple)[2][0]))
            st.write("MSE is :", mean_square_error(regression_model(apple)[1][0], regression_model(apple)[2][0]))
            st.write(interactive_cost_coef(apple))
            
            
        elif mydata == 'Microsoft':
            st.success("Linear regression for Microsoft:")  
            #defining the coordinates 
            X = microsoft['low'].values.reshape(-1,1).astype('float32')
            y = microsoft['high'].values.reshape(-1,1).astype('float32')
            X_test = regression_model(microsoft)[2]
            y_pred = regression_model(microsoft)[1]

            #visualization
            fig,(ax1) = plt.subplots(1, figsize = (12,6))
            ax1.scatter (X, y, s = 8)
            plt.plot(X_test,y_pred, color = 'red', linewidth = 2)
            plt.title ('Min vs Max Stock Rate')
            plt.xlabel('Targeted Stock price')
            plt.ylabel('high')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            showPyplotGlobalUse = False
            st.pyplot()
            
            #Performance metrics
            st.success("Cost evaluators: MAE and MSE for Microsoft:")
            st.write("MAE is :", mean_absolute_error(regression_model(microsoft)[1][0], regression_model(microsoft)[2][0]))
            st.write("MSE is :", mean_square_error(regression_model(microsoft)[1][0], regression_model(microsoft)[2][0]))
            st.write(interactive_cost_coef(microsoft))           
            
        elif mydata == 'Tesla':
            st.success("Linear regression for Tesla:")  
            #defining the coordinates 
            X = tesla['low'].values.reshape(-1,1).astype('float32')
            y = tesla['high'].values.reshape(-1,1).astype('float32')
            X_test = regression_model(tesla)[2]
            y_pred = regression_model(appteslale)[1]

            #visualization
            fig,(ax1) = plt.subplots(1, figsize = (12,6))
            ax1.scatter (X, y, s = 8)
            plt.plot(X_test,y_pred, color = 'red', linewidth = 2)
            plt.title ('Min vs Max Stock Rate')
            plt.xlabel('Targeted Stock price')
            plt.ylabel('high')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            showPyplotGlobalUse = False
            st.pyplot()
            
            #Performance metrics
            st.success("Cost evaluators: MAE and MSE for Tesla:")
            st.write("MAE is :", mean_absolute_error(regression_model(tesla)[1][0], regression_model(tesla)[2][0]))
            st.write("MSE is :", mean_square_error(regression_model(tesla)[1][0], regression_model(tesla)[2][0]))
            st.write(interactive_cost_coef(tesla)) 
            
        elif mydata == 'Google':
            st.success("Linear regression for Google:")  
            #defining the coordinates 
            X = google['low'].values.reshape(-1,1).astype('float32')
            y = google['high'].values.reshape(-1,1).astype('float32')
            X_test = regression_model(google)[2]
            y_pred = regression_model(google)[1]

            #visualization
            fig,(ax1) = plt.subplots(1, figsize = (12,6))
            ax1.scatter (X, y, s = 8)
            plt.plot(X_test,y_pred, color = 'red', linewidth = 2)
            plt.title ('Min vs Max Stock Rate')
            plt.xlabel('Targeted Stock price')
            plt.ylabel('high')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            showPyplotGlobalUse = False
            st.pyplot()
            
            #Performance metrics
            st.success("Cost evaluators: MAE and MSE for Google:")
            st.write("MAE is :", mean_absolute_error(regression_model(google)[1][0], regression_model(google)[2][0]))
            st.write("MSE is :", mean_square_error(regression_model(google)[1][0], regression_model(google)[2][0]))
            st.write(interactive_cost_coef(google))
            
        elif mydata == 'Amazon':
            st.success("Linear regression for Amazon:")  
            #defining the coordinates 
            X = amazon['low'].values.reshape(-1,1).astype('float32')
            y = amazon['high'].values.reshape(-1,1).astype('float32')
            X_test = regression_model(amazon)[2]
            y_pred = regression_model(amazon)[1]

            #visualization
            fig,(ax1) = plt.subplots(1, figsize = (12,6))
            ax1.scatter (X, y, s = 8)
            plt.plot(X_test,y_pred, color = 'red', linewidth = 2)
            plt.title ('Min vs Max Stock Rate')
            plt.xlabel('Targeted Stock price')
            plt.ylabel('high')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            showPyplotGlobalUse = False
            st.pyplot()
            
            #Performance metrics
            st.success("Cost evaluators: MAE and MSE for Amazon:")
            st.write("MAE is :", mean_absolute_error(regression_model(amazon)[1][0], regression_model(amazon)[2][0]))
            st.write("MSE is :", mean_square_error(regression_model(amazon)[1][0], regression_model(amazon)[2][0]))
            st.write(interactive_cost_coef(amazon))               
    
    
    
    
    
    


    # III - Extension to general cost functions:
    #=========================================================
    st.subheader("III - 🔣 Extension to general cost functions:")
    #1- Cost function visualization 
    token_text = '<p style="color:green; font-size: 20px;"><b>1- Cost function visualization </b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    st.info('''
    We have so far restricted the cost component $$L_C$$ of the objective function to the expected value of a pairwise cost $$c(x, T(x, z))$$, as pertains optimal transport. 
    However, it is clear from the data-based formulations derived that the only requirement one must impose on $$L_C = C(T,ρ)$$ is that ρ should only appear through the expected value of functions, which can be replaced by their empirical counterpart when only samples $$(xi, zi)$$ of $$\rho$$ are known. Thus, for instance, inlieu of the pairwise cost $$L_C = \int_{}^{} c(x, T(x, z))ρ(x, z)dx dz$$ , one may propose cost functions involving two points and their images under a common factor $$z$$,
    
    $$
    L_C = \int_{}^{} c(x, T(x, z), x_2, T(x_2,z))ρ(x_1|z)ρ(x_2|z)v(z)dx_1 dx_2 dz
    $$
    
    ''')
    
    mydata = st.selectbox("There we go:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("Animate"):
        if mydata == 'Apple':
            st.success("Cost function visualization for Apple:")
            st.write(animated_plot_LR(apple))
            video_file = open("images/"+apple.Ticker[0]+".mp4", 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            if st.button("Wanna_see 3D"):
                st.info("Here is a 3D animation:")
                three_dimension_animation(apple)
                video_file2 = open("images/"+apple.Ticker[0]+"2.mp4", 'rb')
                video_bytes2 = video_file2.read()
                st.video(video_bytes2)
      
        elif mydata == 'Microsoft':
            st.success("Cost function visualization for Microsoft:")
            st.write(animated_plot_LR(microsoft))
            video_file = open("images/"+microsoft.Ticker[0]+".mp4", 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            #3D animation
            if st.button("Wanna_see 3D"):
                st.info("Here is a 3D animation:")
                three_dimension_animation(microsoft)
                video_file2 = open("images/"+microsoft.Ticker[0]+"2.mp4", 'rb')
                video_bytes2 = video_file2.read()
                st.video(video_bytes2)

        elif mydata == 'Tesla':
            st.success("Cost function visualization for Tesla:")
            st.write(animated_plot_LR(tesla))
            video_file = open("images/"+tesla.Ticker[0]+".mp4", 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            #3D animation
            if st.button("Wanna_see 3D"):
                st.info("Here is a 3D animation:")
                three_dimension_animation(tesla)
                video_file2 = open("images/"+tesla.Ticker[0]+"2.mp4", 'rb')
                video_bytes2 = video_file2.read()
                st.video(video_bytes2)
    
        elif mydata == 'Google':
            st.success("Cost function visualization for Google:")
            st.write(animated_plot_LR(google))
            video_file = open("images/"+google.Ticker[0]+".mp4", 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            #3D animation
            if st.button("Wanna_see 3D"):
                st.info("Here is a 3D animation:")
                three_dimension_animation(google)
                video_file2 = open("images/"+google.Ticker[0]+"2.mp4", 'rb')
                video_bytes2 = video_file2.read()
                st.video(video_bytes2)
        
        elif mydata == 'Amazon':
            st.success("Cost function visualization for Amazon:")
            st.write(animated_plot_LR(amazon))
            video_file = open("images/"+amazon.Ticker[0]+".mp4", 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            #3D animation
            if st.button("Wanna_see 3D"):
                st.info("Here is a 3D animation:")
                three_dimension_animation(amazon)
                video_file2 = open("images/"+amazon.Ticker[0]+"2.mp4", 'rb')
                video_bytes2 = video_file2.read()
                st.video(video_bytes2)
    
    #2- Training Loss
    token_text = '<p style="color:green; font-size: 20px;"><b>2- Learning Rate</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    mydata = st.selectbox("Training Loss:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("calculate"):
        if mydata == 'Apple':
            
            #get history
            import time
            history = training_loss(apple)
            my_bar = st.progress(0)
            for percent_complete in range(5):
                time.sleep(0.3)
                my_bar.progress(percent_complete + 1)
            # convert the training history to a dataframe
            history_df = pd.DataFrame(history.history)
            # use Pandas native plot method
            st.success("Training Loss plot for Apple:")
            history_df['loss'].plot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            #Display training log
            #st.write(training_log_catching(history_df))
            #plot the cost matrix
            #st.write(cost_matrix(apple))
            
            
        elif mydata == 'Microsoft':
            
            #get history
            import time
            history = training_loss(microsoft)
            my_bar = st.progress(0)
            for percent_complete in range(5):
                time.sleep(0.3)
                my_bar.progress(percent_complete + 1)
            # convert the training history to a dataframe
            history_df = pd.DataFrame(history.history)
            # use Pandas native plot method
            st.success("Training Loss plot for Microsoft:")
            plt.plot(history_df['loss'])
            #history_df['loss'].plot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            #Display training log
            #st.write(training_log_catching(history_df))
            #plot the cost matrix
            #st.write(cost_matrix(microsoft))           
            
        elif mydata == 'Tesla':
            
            #get history
            import time
            history = training_loss(tesla)
            my_bar = st.progress(0)
            for percent_complete in range(5):
                time.sleep(0.3)
                my_bar.progress(percent_complete + 1)
            # convert the training history to a dataframe
            history_df = pd.DataFrame(history.history)
            # use Pandas native plot method
            st.success("Cost Matrix plot for Tesla:")
            history_df['loss'].plot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            #Display training log
            #st.write(training_log_catching(history_df))
            #plot the cost matrix            
            #st.write(cost_matrix(tesla))             
            
        elif mydata == 'Google':
            
            #get history
            import time
            history = training_loss(google)
            my_bar = st.progress(0)
            for percent_complete in range(5):
                time.sleep(0.3)
                my_bar.progress(percent_complete + 1)
            # convert the training history to a dataframe
            history_df = pd.DataFrame(history.history)
            # use Pandas native plot method
            st.success("Training Loss plot for Google:")
            history_df['loss'].plot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            #Display training log
            #st.write(training_log_catching(history_df))
            #plot the cost matrix
            #st.write(cost_matrix(google))   
            
        elif mydata == 'Amazon':
            
            #get history
            import time
            history = training_loss(amazon)
            my_bar = st.progress(0)
            for percent_complete in range(5):
                time.sleep(0.3)
                my_bar.progress(percent_complete + 1)
            # convert the training history to a dataframe
            history_df = pd.DataFrame(history.history)
            # use Pandas native plot method
            st.success("Training Loss plot for Amazon:")
            history_df['loss'].plot()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            #Display training log
            #st.write(training_log_catching(history_df))
            #plot the cost matrix
            #st.write(cost_matrix(amazon))             

    # III- Conditional Density Function:
    #===================================
    st.subheader("V - 🔬 Conditional Kernel Density Estimation:")
    #1- Visualize the conditional density estimates
    token_text = '<p style="color:green; font-size: 20px;"><b>1- Visualize the conditional density estimates of stock returns</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    st.markdown("Navigate through this drop down to display the Conditional Density Function graph per company name")
    mydata = st.selectbox("Pick a company:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("plot"):
        if mydata == 'Apple':
            st.success("Conditional Density Function for Apple:")
            cde_test = nnkcde_function(apple)[0]
            z_grid = nnkcde_function(apple)[1]
            z_test = nnkcde_function(apple)[3]

            fig = plt.figure(figsize=(30, 20))
            for jj, cde_predicted in enumerate(cde_test[:12,:]):
                ax = fig.add_subplot(3, 4, jj + 1)
                plt.plot(z_grid, cde_predicted, label=r'$\hat{p}(z| x_{\rm obs})$')
                plt.axvline(z_test[jj], color='red', label=r'$z_{\rm obs}$')
                plt.xticks(size=16)
                plt.yticks(size=16)
                plt.xlabel(r'Redshift $z$', size=20)
                plt.ylabel('CDE', size=20)
                plt.legend(loc='upper right', prop={'size': 18})
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            
        if mydata == 'Microsoft':
            st.success("Conditional Density Function for Microsoft:")
            cde_test = nnkcde_function(microsoft)[0]
            z_grid = nnkcde_function(microsoft)[1]
            z_test = nnkcde_function(microsoft)[3]

            fig = plt.figure(figsize=(30, 20))
            for jj, cde_predicted in enumerate(cde_test[:12,:]):
                ax = fig.add_subplot(3, 4, jj + 1)
                plt.plot(z_grid, cde_predicted, label=r'$\hat{p}(z| x_{\rm obs})$')
                plt.axvline(z_test[jj], color='red', label=r'$z_{\rm obs}$')
                plt.xticks(size=16)
                plt.yticks(size=16)
                plt.xlabel(r'Redshift $z$', size=20)
                plt.ylabel('CDE', size=20)
                plt.legend(loc='upper right', prop={'size': 18})
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            
        if mydata == 'Google':
            st.success("Conditional Density Function for Google:")
            cde_test = nnkcde_function(google)[0]
            z_grid = nnkcde_function(google)[1]
            z_test = nnkcde_function(google)[3]

            fig = plt.figure(figsize=(30, 20))
            for jj, cde_predicted in enumerate(cde_test[:12,:]):
                ax = fig.add_subplot(3, 4, jj + 1)
                plt.plot(z_grid, cde_predicted, label=r'$\hat{p}(z| x_{\rm obs})$')
                plt.axvline(z_test[jj], color='red', label=r'$z_{\rm obs}$')
                plt.xticks(size=16)
                plt.yticks(size=16)
                plt.xlabel(r'Redshift $z$', size=20)
                plt.ylabel('CDE', size=20)
                plt.legend(loc='upper right', prop={'size': 18})
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
    
        if mydata == 'Tesla':
            st.success("Conditional Density Function for Tesla:")
            cde_test = nnkcde_function(tesla)[0]
            z_grid = nnkcde_function(tesla)[1]
            z_test = nnkcde_function(tesla)[3]

            fig = plt.figure(figsize=(30, 20))
            for jj, cde_predicted in enumerate(cde_test[:12,:]):
                ax = fig.add_subplot(3, 4, jj + 1)
                plt.plot(z_grid, cde_predicted, label=r'$\hat{p}(z| x_{\rm obs})$')
                plt.axvline(z_test[jj], color='red', label=r'$z_{\rm obs}$')
                plt.xticks(size=16)
                plt.yticks(size=16)
                plt.xlabel(r'Redshift $z$', size=20)
                plt.ylabel('CDE', size=20)
                plt.legend(loc='upper right', prop={'size': 18})
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()    
    
        if mydata == 'Amazon':
            st.success("Conditional Density Function for Amazon:")
            cde_test = nnkcde_function(amazon)[0]
            z_grid = nnkcde_function(amazon)[1]
            z_test = nnkcde_function(amazon)[3]

            fig = plt.figure(figsize=(30, 20))
            for jj, cde_predicted in enumerate(cde_test[:12,:]):
                ax = fig.add_subplot(3, 4, jj + 1)
                plt.plot(z_grid, cde_predicted, label=r'$\hat{p}(z| x_{\rm obs})$')
                plt.axvline(z_test[jj], color='red', label=r'$z_{\rm obs}$')
                plt.xticks(size=16)
                plt.yticks(size=16)
                plt.xlabel(r'Redshift $z$', size=20)
                plt.ylabel('CDE', size=20)
                plt.legend(loc='upper right', prop={'size': 18})
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()    
        
    
    #2- Nadaraya–Watson estimation
    #token_text = '<p style="color:green; font-size: 20px;"><b>2- Nadaraya–Watson estimation</b></p>'
    #st.markdown(token_text, unsafe_allow_html=True)
    #3- Selecting the bandwidth via cross-validation
    #token_text = '<p style="color:green; font-size: 20px;"><b>3- Selecting the bandwidth via cross-validation</b></p>'
    #st.markdown(token_text, unsafe_allow_html=True)
    #2- KDE smoothing: Scott's and Silverman Factors for Bandwidth Selection for Kernel Density
    token_text = '<p style="color:green; font-size: 20px;"><b>2- KDE smoothing: Scott\'s and Silverman Factors for Bandwidth Selection for Kernel Density</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    mydata = st.selectbox("Go for a company:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("Smooth it"):
        if mydata == 'Apple':
            st.success("Scott's and Silverman Factors for Bandwidth Selection for Kernel Density for Apple:")
            #definining the datapoints: The dataset with which `gaussian_kde` was initialized
            a = apple.high
            # The covariance matrix of `dataset`, scaled by the calculated bandwidt
            kde = stats.gaussian_kde(a)

            b = np.linspace(min(a)-20, max(a)+20, num=100)
            y1 = kde(b)

            kde.set_bandwidth(bw_method='silverman')
            y2 = kde(b)

            kde.set_bandwidth(bw_method=kde.factor / 3.)
            y3 = kde(b)

            #Plotting
            fig, ax = plt.subplots()
            #ax.plot(a, np.full(a.shape, 1 / (4. * a.size)), 'bo',label='Data points (rescaled)')
            ax.plot(b, y1, label='Scott (default)')
            ax.plot(b, y2, label='Silverman')
            ax.plot(b, y3, label='Const (1/3 * Silverman)')
            ax.legend()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()

        elif mydata == 'Microsoft':
            st.success("Scott's and Silverman Factors for Bandwidth Selection for Kernel Density for Microsoft:")
            #definining the datapoints: The dataset with which `gaussian_kde` was initialized
            a = microsoft.high
            # The covariance matrix of `dataset`, scaled by the calculated bandwidt
            kde = stats.gaussian_kde(a)

            b = np.linspace(min(a)-20, max(a)+20, num=100)
            y1 = kde(b)

            kde.set_bandwidth(bw_method='silverman')
            y2 = kde(b)

            kde.set_bandwidth(bw_method=kde.factor / 3.)
            y3 = kde(b)

            #Plotting
            fig, ax = plt.subplots()
            #ax.plot(a, np.full(a.shape, 1 / (4. * a.size)), 'bo',label='Data points (rescaled)')
            ax.plot(b, y1, label='Scott (default)')
            ax.plot(b, y2, label='Silverman')
            ax.plot(b, y3, label='Const (1/3 * Silverman)')
            ax.legend()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            
        elif mydata == 'Google':
            st.success("Scott's and Silverman Factors for Bandwidth Selection for Kernel Density for Google:")
            #definining the datapoints: The dataset with which `gaussian_kde` was initialized
            a = google.high
            # The covariance matrix of `dataset`, scaled by the calculated bandwidt
            kde = stats.gaussian_kde(a)

            b = np.linspace(min(a)-100, max(a)+100, num=100)
            y1 = kde(b)

            kde.set_bandwidth(bw_method='silverman')
            y2 = kde(b)

            kde.set_bandwidth(bw_method=kde.factor / 3.)
            y3 = kde(b)

            #Plotting
            fig, ax = plt.subplots()
            #ax.plot(a, np.full(a.shape, 1 / (4. * a.size)), 'bo',label='Data points (rescaled)')
            ax.plot(b, y1, label='Scott (default)')
            ax.plot(b, y2, label='Silverman')
            ax.plot(b, y3, label='Const (1/3 * Silverman)')
            ax.legend()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()            
    
        elif mydata == 'Amazon':
            st.success("Scott's and Silverman Factors for Bandwidth Selection for Kernel Density for Amazon:")
            #definining the datapoints: The dataset with which `gaussian_kde` was initialized
            a = amazon.high
            # The covariance matrix of `dataset`, scaled by the calculated bandwidt
            kde = stats.gaussian_kde(a)

            b = np.linspace(min(a)-100, max(a)+100, num=100)
            y1 = kde(b)

            kde.set_bandwidth(bw_method='silverman')
            y2 = kde(b)

            kde.set_bandwidth(bw_method=kde.factor / 3.)
            y3 = kde(b)

            #Plotting
            fig, ax = plt.subplots()
            #ax.plot(a, np.full(a.shape, 1 / (4. * a.size)), 'bo',label='Data points (rescaled)')
            ax.plot(b, y1, label='Scott (default)')
            ax.plot(b, y2, label='Silverman')
            ax.plot(b, y3, label='Const (1/3 * Silverman)')
            ax.legend()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()    
    
        elif mydata == 'Tesla':
            st.success("Scott's and Silverman Factors for Bandwidth Selection for Kernel Density for Tesla:")
            #definining the datapoints: The dataset with which `gaussian_kde` was initialized
            a = tesla.high
            # The covariance matrix of `dataset`, scaled by the calculated bandwidt
            kde = stats.gaussian_kde(a)

            b = np.linspace(min(a)-100, max(a)+100, num=100)
            y1 = kde(b)

            kde.set_bandwidth(bw_method='silverman')
            y2 = kde(b)

            kde.set_bandwidth(bw_method=kde.factor / 3.)
            y3 = kde(b)

            #Plotting
            fig, ax = plt.subplots()
            #ax.plot(a, np.full(a.shape, 1 / (4. * a.size)), 'bo',label='Data points (rescaled)')
            ax.plot(b, y1, label='Scott (default)')
            ax.plot(b, y2, label='Silverman')
            ax.plot(b, y3, label='Const (1/3 * Silverman)')
            ax.legend()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
        st.info('''
                    Bandwidth selection strongly influences the estimate obtained from the KDE (much more so than the actual shape of the kernel). Bandwidth selection can be done by a “rule of thumb”, by cross-validation, by “plug-in methods” or by other means; gaussian_kde uses a rule of thumb, the default is Scott’s Rule.    

                    **Description**

                    Use Scott's rule of thumb to determine the smoothing bandwidth for the kernel estimation of point process intensity.

                    ***Formula:***

                       n**(-1./(d+4))

                    Where:

                    - n: the number of data points
                    - d: the number of dimensions

                    ***NB***: In the case of unequally weighted points, we use neff

                    ''')    
        st.info('''
            **Silversman's Factor**

            (n * (d + 2) / 4.)**(-1. / (d + 4))

            Where:
            - n: the number of data points
            - d: the number of dimensions

            ***NB***: In the case of unequally weighted points, we use neff
        ''')
    
    # IV- Distributional Barycenter:
    #==============================
    st.subheader("V - 📊 Distributional Barycenter:")
    #1 - Barycenter interpolation with Wasserstein
    token_text = '<p style="color:green; font-size: 20px;"><b>1 - Barycenter interpolation with Wasserstein</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    
    st.markdown("Navigate through this drop down to display the Barycenter interpolation with Wasserstein graph per company name")
    mydata = st.selectbox("Tick a company:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("Get the chart"):
        if mydata == 'Apple':
            st.success("Barycenter interpolation with Wasserstein graph for Apple:")
            st.write(baycenter_interpolation(apple))
            
        if mydata == 'Microsoft':
            st.success("Barycenter interpolation with Wasserstein graph for Microsoft:")
            st.write(baycenter_interpolation(microsoft))  
            
        if mydata == 'Tesla':
            st.success("Barycenter interpolation with Wasserstein graph for Tesla:")
            st.write(baycenter_interpolation(tesla))  
            
        if mydata == 'Google':
            st.success("Barycenter interpolation with Wasserstein graph for Google:")
            st.write(baycenter_interpolation(google)) 
            
        if mydata == 'Amazon':
            st.success("Barycenter interpolation with Wasserstein graph for Amazon:")
            st.write(baycenter_interpolation(amazon)) 
            
    #2 - Wasserstein distances computation
    #token_text = '<p style="color:green; font-size: 20px;"><b>2 - Wasserstein distances computation</b></p>'
    #st.markdown(token_text, unsafe_allow_html=True)
    
    #2 - Wasserstein barycenter computation
    token_text = '<p style="color:green; font-size: 20px;"><b>2 - Wasserstein barycenter computation</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    st.markdown("Navigate through this drop down to display the Wasserstein barycenter computation graph per company name")
    mydata = st.selectbox("Select a ticker:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("Get the graph"):
        if mydata == 'Apple':
            st.success("Wasserstein barycenter computation graph for Apple:")
            st.write(baycenter_computation(apple))
            
        if mydata == 'Microsoft':
            st.success("Wasserstein barycenter computation graph for Microsoft:")
            st.write(baycenter_computation(microsoft))  
            
        if mydata == 'Tesla':
            st.success("Wasserstein barycenter computation graph for Tesla:")
            st.write(baycenter_computation(tesla))
            
        if mydata == 'Google':
            st.success("Wasserstein barycenter computation graph for Google:")
            st.write(baycenter_computation(google)) 
            
        if mydata == 'Amazon':
            st.success("Wasserstein barycenter computation graph for Amazon:")
            st.write(baycenter_computation(amazon))    
    
    
    #3 - Optimal Transport Matrix computation
    #token_text = '<p style="color:green; font-size: 20px;"><b>3 - Optimal Transport Matrix computation</b></p>'
    #st.markdown(token_text, unsafe_allow_html=True)

    # V- Optimal Transport Loss and Dual Variables:
    #==============================================
    st.subheader("VI - 📋 Optimal Transport & Alternative CDE:")
    st.info('''
    ***Optimal transport problem solver***

    The optimal transport theory is the study of optimal transportation and allocation between measures.

    The optimal transport problem was first introduced by Monge (1781) and formalized by Kantorovitch (1942), leading to the so called Monge-Kantorovitch transportation problem.

    The goal is to look for a transport map transforming a probability density function into another while minimizing the cost of transport.
    ''')
    # 1- Regularized OT with Sinkhorn
    token_text = '<p style="color:green; font-size: 20px;"><b>1- Regularized OT with Sinkhorn</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    st.info('''
    ***An alternative conditional density estimator***
    Given that the following matrix $$Z \in \mathbb{R}^{N\star N}$$ is a normalized version of similar kernels in z-space    $$Z_{ik}={K_b(z_k, z_i)}\div{\sum^{N}_{j=1}K_b(z_k,z_j)}$$ is not the only choice that makes a robust conditional density estimator. The core requirements for $$Z$$ is that the entries $$Z_{ik}$$ must be nonnegative and add up to zero row-wise to guarantee that the estimated $$ρT(y|z_k)$$ is positive and integrates to one.
    Therefore, a symmetric matrix $$Z$$ with nonnegative entries whose rows add up to one is necessarily bi-stochastic, a natural candidate is the unique bi-stochastic matrix $${Z}^{\sim}$$ that derives from the
symmetric and positive Kernel matrix $$K_{ik} = K_b(z_k,z_i)$$ through **Sinkhorn's factorization**: 
$${Z}^{\sim}= DKD$$; where $$D$$ is a diagonal matrix with positive diagonalentries.
    ''')
    mydata = st.selectbox("Regularized OT with Sinkhorn:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("voila "):
        if mydata == 'Apple':
            st.success("Regularized OT with Sinkhorn for Apple:")
            # data to be plotted
            xAxis = list(range(1, 121))
            yAxis = optimal_transport_sinkhorn(apple)
            # plotting
            plt.title("Optimal Transport Sinkhorn for Apple Data")
            plt.xlabel("nth point")
            plt.ylabel(" OT matrix Computation")
            plt.plot(xAxis, yAxis, color ="orange")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()
            
        elif mydata == 'Microsoft':
            st.success("Regularized OT with Sinkhorn for Microsoft:")
            # data to be plotted
            xAxis = list(range(1, 121))
            yAxis = optimal_transport_sinkhorn(microsoft)
            # plotting
            plt.title("Optimal Transport Sinkhorn for Microsoft Data")
            plt.xlabel("nth point")
            plt.ylabel(" OT matrix Computation")
            plt.plot(xAxis, yAxis, color ="orange")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()    

        elif mydata == 'Tesla':
            st.success("Regularized OT with Sinkhorn for Tesla:")
            # data to be plotted
            xAxis = list(range(1, 121))
            yAxis = optimal_transport_sinkhorn(tesla)
            # plotting
            plt.title("Optimal Transport Sinkhorn for Tesla Data")
            plt.xlabel("nth point")
            plt.ylabel(" OT matrix Computation")
            plt.plot(xAxis, yAxis, color ="orange")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()    
    
        elif mydata == 'Google':
            st.success("Regularized OT with Sinkhorn for Google:")
            # data to be plotted
            xAxis = list(range(1, 121))
            yAxis = optimal_transport_sinkhorn(google)
            # plotting
            plt.title("Optimal Transport Sinkhorn for Google Data")
            plt.xlabel("nth point")
            plt.ylabel(" OT matrix Computation")
            plt.plot(xAxis, yAxis, color ="orange")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()    
    
        elif mydata == 'Amazon':
            st.success("Regularized OT with Sinkhorn for Amazon:")
            # data to be plotted
            xAxis = list(range(1, 121))
            yAxis = optimal_transport_sinkhorn(amazon)
            # plotting
            plt.title("Optimal Transport Sinkhorn for Amazon Data")
            plt.xlabel("nth point")
            plt.ylabel(" OT matrix Computation")
            plt.plot(xAxis, yAxis, color ="orange")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.show()
            st.pyplot()    
    
    
    
    
    
    # 2- Compute the Wasserstein loss for Sinkhorn
    token_text = '<p style="color:green; font-size: 20px;"><b>2- Compute the Wasserstein loss for Sinkhorn</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    mydata = st.selectbox("Compute the Wasserstein loss for Sinkhorn:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if st.button("let's compute.. "):
        if mydata == 'Apple':
            st.success("Compute the Wasserstein loss for Sinkhorn for Apple:")
            st.write(use_case(apple))
            
        elif mydata == 'Microsoft':
            st.success("Compute the Wasserstein loss for Sinkhorn for Micorosoft:")
            st.write(use_case(microsoft))   
            
        elif mydata == 'Tesla':
            st.success("Compute the Wasserstein loss for Sinkhorn for Tesla:")
            st.write(use_case(tesla))            
            
        elif mydata == 'Google':
            st.success("Compute the Wasserstein loss for Sinkhorn for Google:")
            st.write(use_case(google))            
            
        elif mydata == 'Amazon':
            st.success("Compute the Wasserstein loss for Sinkhorn for Amazon:")
            st.write(use_case(amazon))            
            
            
            
            
    
    # 3- Visualization: Comparison between Sinkhorn and EMD
    #token_text = '<p style="color:green; font-size: 20px;"><b>3- Visualization: Comparison between Sinkhorn and EMD</b></p>'
    #st.markdown(token_text, unsafe_allow_html=True)
    
    # VI: Recapitulation and Conclusion:
    #=====================
    st.subheader("VII - 📙 Recapitulation and Conclusion:")
    #1- Comparative Dataframe of the Predictive Stock Price using OT vs LR vs Reg. OT
    token_text = '<p style="color:green; font-size: 20px;"><b>1- Comparative Dataframe of the Predictive Stock Price using OT vs LR vs Reg. OT</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    mydata = st.radio("Compute the price prediction for:",['Microsoft', 'Apple', 'Tesla', 'Google', 'Amazon'])
    if mydata == 'Apple':
            st.success("Stock Price Forecasting for Apple:")
            st.write(return_conclusive_dataframe(apple))
            st.write(plotly_3d(apple))
            
    elif mydata == 'Microsoft':
            st.success("Stock Price Forecasting for Microsoft:")
            st.write(return_conclusive_dataframe(microsoft))
            st.write(plotly_3d(microsoft))
            
    elif mydata == 'Tesla':
            st.success("Stock Price Forecasting for Tesla:")
            st.write(return_conclusive_dataframe(tesla))
            st.write(plotly_3d(tesla))
            
    elif mydata == 'Google':
            st.success("Stock Price Forecasting for Google:")
            st.write(return_conclusive_dataframe(google))
            st.write(plotly_3d(google))
    
    elif mydata == 'Amazon':
            st.success("Stock Price Forecasting for Amazon:")
            st.write(return_conclusive_dataframe(amazon))
            st.write(plotly_3d(amazon))
    

    
    
    
    #2- Investor Kit app v2
    #token_text = '<p style="color:green; font-size: 20px;"><b>2- Investor Kit app v2</b></p>'
    #st.markdown(token_text, unsafe_allow_html=True)
    #2- Glossary
    token_text = '<p style="color:green; font-size: 20px;"><b>2- Glossary</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    # Get some infor about what was mentioned.
    glossaire = [['Transportation theory', '''In mathematics and economics, transportation theory or transport theory is a name given to the study of optimal transportation and allocation of resources. The problem was formalized by the French mathematician Gaspard Monge in 1781. In the 1920s A.N. '''], 
                 ['Wasserstein metric', 'In mathematics, the Wasserstein distance or Kantorovich–Rubinstein metric is a distance function defined between probability distributions on a given metric space M.'], 
                 ['Loss function', 'In mathematical optimization and decision theory, a loss function or cost function is a function that maps an event or values of one or more variables onto a real number intuitively representing some "cost" associated with the event. An optimization problem seeks to minimize a loss function'],
                ['Sinkhorn Knopp','The Sinkhorn-Knopp algorithm takes a matrix A and finds diagonal matrices D and E such that if M = DAE the sum of each column and each row of M is unity. The method is, in effect, to alternately normalise the rows and the columns of the matrix. ... Such matrices have various applications, including web page ranking'],
                ['Bandwidth ','The bandwidth is a measure of how closely you want the density to match the distribution. bw the smoothing bandwidth to be used. The kernels are scaled such that this is the standard deviation of the smoothing kernel.'],
                ['Barycentric interpolation','Barycentric interpolation is a variant of Lagrange polynomial interpolation that is fast and stable. It deserves to be known as the standard method of polynomial interpolation. '],
                ['Coupling matrix ','The coupling coefficient of resonators is a dimensionless value that characterizes interaction of two resonators. Coupling coefficients are used in resonator filter theory. Resonators may be both electromagnetic and acoustic. Coupling coefficients together with resonant frequencies and external quality factors of resonators are the generalized parameters of filters. In order to adjust the frequency response of the filter it is sufficient to optimize only these generalized parameters. '],
                ['undersmoothing','A small bandwidth leads to undersmoothing: It means that the density plot will look like a combination of individual peeks (one peek per each sample element).'],
                ['oversmoothing','A huge bandwidth leads to oversmoothing: It means that the density plot will look like a unimodal distribution and hide all non-unimodal distribution properties (e.g., if a distribution is multimodal, we will not see it in the plot).'],
                ['Geodesic distance','A simple measure of the distance between two vertices in a graph is the shortest path between the vertices. Formally, the geodesic distance between two vertices is the length in terms of the number of edges of the shortest path between the vertices. For two vertices that are not connected in a graph, the geodesic distance is defined as infinite.'],
                ['Euclidean distance','In mathematics, the Euclidean distance between two points in Euclidean space is the length of a line segment between the two points. It can be calculated from the Cartesian coordinates of the points using the Pythagorean theorem, therefore occasionally being called the Pythagorean distance. '],
                ['Adversarial optimization','Adversarial Training (AT) is a technique that approximately solves a robust optimization problem to minimize the worst-case loss and is widely regarded as the most effective defense against such attacks.'],
                ['Linear regression estimation','In statistics, linear regression is a linear approach for modelling the relationship between a scalar response and one or more explanatory variables. The case of one explanatory variable is called simple linear regression; for more than one, the process is called multiple linear regression.'],]
    Glossary = pd.DataFrame(glossaire, columns = ['Terminologies', 'Definition'])
    Glossary = Glossary.sort_values('Terminologies', ascending=True)
    if st.button('Show Glossary'):
        st.table(Glossary.assign(hack='').set_index('hack'))

    #3- Conclusion
    token_text = '<p style="color:green; font-size: 20px;"><b>3- Conlusion</b></p>'
    st.markdown(token_text, unsafe_allow_html=True)
    st.markdown('''
    This work introduces the distributional barycenter problem, an extension of the optimal transport barycenter problem where the cost needs not be the expected value of a pairwise function, allowing more general costs needed in
applications, such as a new cost penalizing non-isometric maps.
A novel numerical algorithm is introduced for the solution of the barycenter problem. The algorithm avoids the difficulties typical of adversarial approaches by slaving the discriminator to the generator. This results in a simpler approach that looks for a minimum rather than a saddle point of the objective function.
    ''')

    # Sidebar: Footer
    st.sidebar.markdown("[Data Source](https://www.ttingo.com)")
    st.sidebar.info("Linkedin [Joseph Bunster](https://www.linkedin.com/in/joseph-bunster/) ")
    st.sidebar.info("Self Exploratory Visualization using Optimal Transport on Financial Time Series Data- Brought To you By [Jospeh Bunster](https://github.com/Joseph-Bunster)  ")
    st.sidebar.text("Built with  ❤️ by Joe Bunster")
    st.sidebar.image(
            "https://upload.wikimedia.org/wikipedia/commons/7/76/Banco_de_Jos%C3%A9_Bunster_%2830203401352%29.jpg",
            width=250, # Manually Adjust the width of the image as per requirement
        )


#Main function execution 
if __name__=='__main__':
    main()
