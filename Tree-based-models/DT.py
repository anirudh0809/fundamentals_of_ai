from feature_engineering import *
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# import pydotplus
# import graphviz 
#print(data_new['outcome'].unique())
''' Defining independent and dependent variable '''
X = data_new.drop(['outcome'], axis = 1)
Y = data_new['outcome']

''' splitting the data into train and validation sets'''
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, stratify=Y)

print("Count of training instances -",len(X_train))
print("Count of testing instances -",len(y_test))

pipeline_def = [('scaler', StandardScaler()), ('decision_tree', DecisionTreeClassifier())]
parameters = {'criterion' : ['gini','entropy'],
                'max_depth': np.arange(3,15)
                
            }
pipeline = Pipeline(pipeline_def)
print(pipeline)


#from tqdm import tqdm_notebook as tqdm 
import warnings
print('Training Start')
warnings.filterwarnings("ignore")


for cross_val in range(3,6):
    grid = GridSearchCV(pipeline, param_grid= parameters, cv=cross_val)
    grid.fit(X_train,y_train)
    print("Score for %d fold Cross Validation: %3.2f"%(cross_val,grid.score(X_test,y_test)))
    print("Best Parameters")
    print(grid.best_params_)

print("loop end")




