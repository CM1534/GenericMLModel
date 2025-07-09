import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

# Load data #
def fileReader(path):
    data = pd.read_csv(path)
    filteredData = data.dropna(axis=0)
    return filteredData
#gets x,y
def makeModel(data, features=[]):
    y = data.Price
    x = data[features]
    return x,y

#splits data
def split(x, y, rand = 0):
    train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = rand)
    return train_x, val_x, train_y, val_y

#helper function, gets mean error
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y, rand = 0):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=rand)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)

#takes in data & number of nodes, finds best one
def findNumNodes(train_X, val_X, train_y, val_y, leafList = [5, 50, 500, 5000]):
    runningBest = -1
    runningNodes = -1
    for nodes in leafList:
        my_mae = get_mae(nodes, train_X, val_X, train_y, val_y)
        if my_mae < runningBest or runningBest == -1:
            runningBest = my_mae
            runningNodes = nodes
        #print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(nodes, my_mae))
    return runningNodes

#########################################################################################################
def forestModel(train_X, train_y, val_X, val_y):
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_X, train_y)
    preds = forest_model.predict(val_X)
    MAE = mean_absolute_error(val_y, preds)
    #print(MAE)
    return MAE
#########################################################################################################
#combines all helper functions
def model(path, features = [], rand = 0, leafList = [5, 50, 100, 250, 400, 500, 600, 750, 1000, 5000]):
    data = fileReader(path)
    x, y = makeModel(data, features)
    tX, vX, tY, vY = split(x, y, rand)
    numNodes = findNumNodes(tX, vX, tY, vY, leafList)
    finalModel = DecisionTreeRegressor(max_leaf_nodes = numNodes, random_state = rand)
    finalModel.fit(tX, tY)
    valPred = finalModel.predict(vX)
    valMAE = mean_absolute_error(valPred, vY)
    #print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(valMAE))

    forestMAE = forestModel(tX, tY, vX, vY)
    print("MAE with DecisionTree: %d\nMAE with RandForest: %d" %(valMAE, forestMAE))
    print("Max nodes for DecisionTree: %d" %(numNodes))

#########################################################################################################
#Edit here
path = 'melb_data.csv'
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']
model(path, features, 0)

