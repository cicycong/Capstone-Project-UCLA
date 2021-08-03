from pipeline import runRFR,runLasso,runGB


def main():
    '''

    run pipeline here

    '''
    #run Random Forest Model pipeline
    runRFR()
    #run Lasso Regression Model pipeline
    runLasso()
    #run Gradient Boost Model pipeline
    runGB()




if __name__ == '__main__':
    main()
