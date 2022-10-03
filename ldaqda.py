import numpy as np
import matplotlib.pyplot as plt
import util1

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """

    maleIsTrue = y == 1
    femaleIsTrue = y == 2
    maleValues = x[maleIsTrue][:]
    femaleValues = x[femaleIsTrue][:]

    mu_male = maleValues.mean(axis = 0)
    mu_female = femaleValues.mean(axis = 0)

    cov_male = np.cov(maleValues, rowvar = False)
    cov_female = np.cov(femaleValues, rowvar = False)
    cov = ((maleValues.shape[0]*cov_male) + (femaleValues.shape[0]*cov_female)) / len(y)

    print("cov: ", cov)

    #Plotting the data
    #np.meshgrid()
    #compute the density for each point in the meshgrid for each class
    #plt.contour
    xAxis = np.linspace(50, 80, 500)
    yAxis = np.linspace(80, 280, 500)
    grid = np.meshgrid(xAxis, yAxis)
    xx, yy = grid

    points = np.reshape(grid, (2, -1)).T
    print(points)
    print(points.shape[1])

    maleContour = util.density_Gaussian(mu_male, cov_male, points).T.reshape(xx.shape)
    femaleContour = util.density_Gaussian(mu_female, cov_female, points).T.reshape(xx.shape)

    LDAcontour = calcLDA(mu_male, mu_female, cov, points).T.reshape(xx.shape)
    QDAcontour = calcQDA(mu_male, mu_female, cov_male, cov_female, points).T.reshape(xx.shape)

    plt.style.use('seaborn-whitegrid')
    plt.plot(maleValues[:,0], maleValues[:,1], 'o', color = "blue")
    plt.plot(femaleValues[:,0], femaleValues[:,1], 'o', color = "red")

    plt.contour(xx, yy, maleContour, colors = "blue", alpha = 0.5)
    plt.contour(xx, yy, femaleContour, colors = "red", alpha = 0.5)

    plt.contour(xx, yy, LDAcontour, levels = [0], colors = 'black')

    plt.show()

    plt.style.use('seaborn-whitegrid')
    plt.plot(maleValues[:,0], maleValues[:,1], 'o', color = "blue")
    plt.plot(femaleValues[:,0], femaleValues[:,1], 'o', color = "red")

    plt.contour(xx, yy, maleContour, colors = "blue", alpha = 0.5)
    plt.contour(xx, yy, femaleContour, colors = "red", alpha = 0.5)

    plt.contour(xx, yy, QDAcontour, levels = [0], colors = 'black')

    plt.show()

    return (mu_male,mu_female,cov,cov_male,cov_female)  

def calcLDA(male_mean, female_mean,covariance_mat,x_set):
    w = np.matmul(np.linalg.inv(covariance_mat), (male_mean - female_mean))
    b = -0.5 * np.matmul((male_mean + female_mean).T, w)
    LDA = []
    for x in x_set:
        boundaryPoint = np.matmul(w.T, x) + b
        LDA.append(boundaryPoint)
    LDA_array = np.array(LDA)

    return LDA_array

def calcQDA(mean_male, mean_female, cov_male, cov_female, x_set):
    QDA = []
    for x in x_set:
        term1 = np.matmul( np.matmul((x - mean_male).T, np.linalg.inv(cov_male)), (x - mean_male))
        term2 = np.matmul( np.matmul((x - mean_female).T, np.linalg.inv(cov_female)), (x - mean_female))
        term3 = np.math.log(np.linalg.det(cov_female)/np.linalg.det(cov_male))
        boundaryPoint = term2 - term1 + term3
        QDA.append(boundaryPoint)
    QDA_array = np.array(QDA)

    return QDA_array

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate

    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis

    x: a N-by-2 2D array contains the height/weight data of the N samples  

    y: a N-by-1 1D array contains the labels of the N samples 

    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """

    numLabels = 0
    mis_lda = 0
    mis_qda = 0

    for i in range(x.shape[0]):
        xData = x[i,:]

        #linear calculations
        w = np.matmul(np.linalg.inv(cov), (mu_male - mu_female))
        b = -0.5 * np.matmul((mu_male + mu_female).T, w)
        boundaryPoint = np.matmul(w.T, xData) + b
        if(boundaryPoint > 0):
            classification = 1  #male
        else:
            classification = 2  #female
        if(classification != y[i]):
            mis_lda += 1

        #quadratic calculations
        term1 = np.matmul( np.matmul((xData - mu_male).T, np.linalg.inv(cov_male)), (xData - mu_male))
        term2 = np.matmul( np.matmul((xData - mu_female).T, np.linalg.inv(cov_female)), (xData - mu_female))
        term3 = np.math.log(np.linalg.det(cov_female)/np.linalg.det(cov_male))
        boundaryPoint = term2 - term1 + term3
        if(boundaryPoint > 0):
            classification = 1
        else:
            classification = 2
        if(classification != y[i]):
            mis_qda += 1

        numLabels += 1

    mis_lda /= numLabels
    mis_qda /= numLabels

    print("The misrate for LDA is ", mis_lda*100, "%")
    print("The misrate for QDA is ", mis_qda*100, "%")

    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file("C:/Users/ccall/OneDrive/Desktop/Year3Sem2/ECE368 - Probabilistic Reasoning/Lab 1/ldaqda/trainHeightWeight.txt")
    x_test, y_test = util.get_data_in_file("C:/Users/ccall/OneDrive/Desktop/Year3Sem2/ECE368 - Probabilistic Reasoning/Lab 1/ldaqda/testHeightWeight.txt")
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    
    

    
