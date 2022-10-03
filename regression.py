from copyreg import constructor
import numpy as np
import matplotlib.pyplot as plt
import util2

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the prior distribution
    
    Outputs: None
    -----
    """
    mean = np.array([0,0])
    covariance = np.array([[beta, 0],[0, beta]])

    plotDensityGaussian(-1, 1, -1, 1, mean, covariance)
    plt.scatter(-0.1, -0.5)

    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.title("posterior distribution")
    plt.savefig(f'posterior-{0}.pdf', dpi = 1200)
    plt.show()
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    n = len(x)
    onesVec = np.ones((n, 1))
    X = np.hstack((onesVec, x))
    I = np.eye(2)
    mean = np.linalg.multi_dot((np.linalg.inv(np.dot(X.T, X) + (sigma2/beta)*I), X.T, z))
    mean = mean.ravel()
    covariance = np.linalg.inv(np.dot(X.T, X) + (sigma2/beta)*I)*sigma2
   
    plotDensityGaussian(-1, 1, -1, 1, mean, covariance)
    plt.scatter(-0.1, -0.5)

    plt.xlabel("a0")
    plt.ylabel("a1")
    plt.title("posterior distribution")
    plt.savefig(f'posterior-{len(x)}.pdf', dpi = 1200)
    plt.show()
    return (mean,covariance)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the prior distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    n = len(x)
    x = x[np.newaxis].T
    onesVec = np.ones((n, 1))

    X = np.hstack((onesVec, x))
    output = np.matmul(X, mu)

    covarianceDiagonals = np.linalg.multi_dot((X, Cov, X.T))
    covariances = np.sqrt(np.diagonal(covarianceDiagonals))

    plt.scatter(x, output)
    plt.errorbar(x, output, covariances)
    plt.scatter(x_train, z_train)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("prediction distribution")
    plt.savefig(f'prediction-{len(x_train)}.pdf', dpi = 1200)
    plt.show()
    return

def plotDensityGaussian(xMin, xMax, yMin, yMax, mean, cov):
    xAxis = np.linspace(xMin, xMax, 250)
    yAxis = np.linspace(yMin, yMax, 250)
    grid = np.meshgrid(xAxis, yAxis)
    xx, yy = grid
    points = np.reshape(grid, (2, -1)).T
    contour = util.density_Gaussian(mean, cov, points).T.reshape(xx.shape)
    plt.contour(xx, yy, contour)

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('C:/Users/ccall/OneDrive/Desktop/Year3Sem2/ECE368 - Probabilistic Reasoning/Lab 2/training.txt')
    # new inputs for prediction 
    x_test = np.arange(-4,4.01,0.2)

    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    trainingSamples = [1, 5, 100]
    for ns in trainingSamples:
        #ns  = 5
        
        # used samples
        x = x_train[0:ns]
        z = z_train[0:ns]
        
        # prior distribution p(a)
        priorDistribution(beta)
        
        # posterior distribution p(a|x,z)
        mu, Cov = posteriorDistribution(x,z,beta,sigma2)
        
        # distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    

    
    
    

    
