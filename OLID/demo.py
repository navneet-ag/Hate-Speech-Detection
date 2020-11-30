from myclasses.Features import Features
from myclasses.Data import Data
import pickle
import sys

def load_model(path):
    with open(path, "rb") as fp:
        clf = pickle.load(fp)
    return clf
dataA=load_model("./models/subtaska/data_obj")
dataB=load_model("./models/subtaskb/data_obj")
dataC=load_model("./models/subtaskc/data_obj")
modelpathA="./models/subtaska/"
modelpathB="./models/subtaskb/"
modelpathC="./models/subtaskc/"
if len(sys.argv)!=1:
    modelpathA=modelpathA+sys.argv[1]
    modelpathB=modelpathB+sys.argv[1]
    modelpathC=modelpathC+sys.argv[1]

else: # Add best model hardcoded below
    modelpathA=modelpathA+"log"
    modelpathB=modelpathB+"log"
    modelpathC=modelpathC+"log"
modelA=load_model(modelpathA)
modelB=load_model(modelpathB)
modelC=load_model(modelpathC)
# xA=dataA.ui_helper(["Muslims are asshole"])
# xB=dataB.ui_helper(["Muslims are asshole"])
# xC=dataC.ui_helper(["Muslims are asshole"])
# print(xA.shape)
# print(modelA.predict(xA))
# print(modelB.predict(xB))
# print(modelC.predict(xC))
while True:
    print("Please Enter Your Text Sentence: ")
    sent=input()
    xA=dataA.ui_helper([sent])
    outA=modelA.predict(xA)
    # print(outA[0])
    print("Task A:")
    if  outA[0]!=("OFF"):
        print("The given Sentence is not classified as Offensive")
    else:
        print("The given Sentence is classified as Offensive")
        print("Task B:")
        xB=dataB.ui_helper([sent])
        outB=modelB.predict(xB)
        if outB[0]!=("TIN"):
            print("The given Sentence is not classified as Targeted")
        else:
            print("The given Sentence is classified as Targeted")
            print("Task C:")
            xC=dataC.ui_helper([sent])
            outC=modelC.predict(xC)
            # print(outC)
            if outC[0]==("OTH"):
                print("The given Sentence is classified as Targeted to others")
            elif outC[0]==("GRP"):
                print("The given Sentence is classified as Targeted to Group")
            else:
                print("The given Sentence is classified as Targeted to Individual")

    #     print("Task B:")
	# print("Task B:")
	# print("Task C:")