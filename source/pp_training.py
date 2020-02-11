
import swigpython3 as pilotbase #for bare-metal stuff
import pilotpython #nicer python classes
from pandas import DataFrame
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

def onInitialize(ctxt):
    global df
    df = None
    return pilotpython.READY_FOR_INPUT_DATA

def onProcess(ctxt, dr):
    global df
    
    context= pilotpython.Context(ctxt)
    data= pilotpython.DataRecord(dr)
    props= data.getProperties()
    
    if data.isNewData():       
        y = np.array(df['pdb_code'])
        X = df.drop(['pdb_code'], axis=1)

        kernel = 1.0 * RBF([1.0])
        gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)
        
        predtrainClass = gpc_rbf_isotropic.predict(X)
        predtrain = gpc_rbf_isotropic.predict_proba(X)

        props.defineFloatArrayProperty("Sepal Length", df["Sepal Length"].tolist())
        props.defineFloatArrayProperty("Sepal Width", df["Sepal Width"].tolist())
        props.defineStringArrayProperty("Class", df["Class"].tolist())
        props.defineStringArrayProperty("Predicted Class", predtrainClass.tolist())
        props.defineFloatArrayProperty("Iris-setosa Prob", predtrain[:,0].tolist())
        props.defineFloatArrayProperty("Iris-versicolor Prob", predtrain[:,1].tolist())
        props.defineFloatArrayProperty("Iris-virginica Prob", predtrain[:,2].tolist())
      
        data.routeTo(pilotpython.PASS_PORT)
        return pilotpython.DONE_PROCESSING_DATA

    else:
        numprops = props.getSize()
        
        names = set()
        d = {}
        for i in range(numprops):
            prop = props.getByIndex(i)
            name = prop.getName()
            value = prop.getValue()
            if (value.getClassTag() == "SciTegic.value.DoubleValue"):
                val = value.getFloat()
            else:
                val = value.getString()
            d[name] = val
            names.add(name)

        if df is None:
            df = DataFrame.from_dict([d])  # need to pass list of dicts rather than dict
        else:
            df = df.append([d])
            nrecs = df.shape[0]
            context.debugMessage(str(nrecs))
    
        data.routeTo(pilotpython.NO_PORT)      
        return pilotpython.READY_FOR_INPUT_THEN_NEW_DATA

def onFinalize(ctxt):
    pass
