import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split

# Read the CSV file
LFTr = pd.read_csv("LF_train.csv")
LHTr = pd.read_csv("LH_train.csv")
RFTr = pd.read_csv("RF_train.csv")
RHTr = pd.read_csv("RH_train.csv")
LFTe = pd.read_csv("LF_test.csv")
LHTe = pd.read_csv("LH_test.csv")
RFTe = pd.read_csv("RF_test.csv")
RHTe = pd.read_csv("RH_test.csv")
sub = pd.read_csv("oldsubmission.csv")
realsub = pd.read_csv("submission.csv")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(363, 25)
        self.fc2 = nn.Linear(25, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


net = Net()
# Modifying
def cleanTrain(Tr, val):
    Tr.drop('dob', axis=1, inplace=True)
    Tr.drop('age', axis=1, inplace=True)
    Tr.drop('forceplate_date', axis=1, inplace=True)
    Tr.drop('gait', axis=1, inplace=True)
    Tr.drop('Gait', axis=1, inplace=True)
    Tr.drop('speed', axis=1, inplace=True)
    Tr.drop('Speed', axis=1, inplace=True)
    if val == 1:
        yTr = Tr['LF']
        Tr.drop('LF', axis=1, inplace=True)
    elif val == 2:
        yTr = Tr['LH']
        Tr.drop('LH', axis=1, inplace=True)
    elif val == 3:
        yTr = Tr['RF']
        Tr.drop('RF', axis=1, inplace=True)
    elif val == 4:
        yTr = Tr['RH']
        Tr.drop('RH', axis=1, inplace=True)
    xTr = Tr

    # xTr['speed'] = pd.to_numeric(xTr['speed'], errors="coerce")
    # xTr['Speed'] = pd.to_numeric(xTr['Speed'], errors="coerce")
    # invalidLF = LFxTr.loc[pd.isna(LFxTr['speed']), :].index
    # LFxTr.drop(invalidLF, axis=0, inplace=True)
    cols = []
    for col in xTr.columns:
        cols.append(col)
    # xTr[cols[3]] = xTr[cols[3]].astype(float)
    # colmean = xTr[cols[3]].mean()
    # xTr[cols[3]] = xTr[cols[3]].fillna(colmean)
    # xTr[cols[184]] = xTr[cols[184]].astype(float)
    # colmean = xTr[cols[184]].mean()
    # xTr[cols[184]] = xTr[cols[184]].fillna(colmean)
    fcount = 1
    for i in range(3, 123):
        if fcount > 4:
            xTr[cols[i]] = xTr[cols[i]].astype(float)
            colmean = xTr[cols[i]].mean()
            xTr[cols[i]] = xTr[cols[i]].fillna(colmean)
        if fcount < 8:
            fcount += 1
        else:
            fcount = 1
    scount = 1
    for i in range(183, 303):
        if scount > 4:
            xTr[cols[i]] = xTr[cols[i]].astype(float)
            colmean = xTr[cols[i]].mean()
            xTr[cols[i]] = xTr[cols[i]].fillna(colmean)
        if scount < 8:
            scount += 1
        else:
            scount = 1
    fscount = 1
    arr = [-1, 1]

    for i in range(3, 123):
        if fscount <= 4:
            xTr[cols[i]] = xTr[cols[i]].astype(float)
            xTr[cols[i+4]] = xTr[cols[i+4]].astype(float)
            colmean = xTr[cols[i]].mean()
            col4mean = xTr[cols[i+4]].mean()
            # random_choice = np.random.choice(arr)
            t = colmean + col4mean
            xTr[cols[i]] = xTr[cols[i]].fillna(t)
        if fscount < 8:
            fscount += 1
        else:
            fscount = 1
    sscount = 1
    for i in range(183, 303):
        if sscount <= 4:
            xTr[cols[i]] = xTr[cols[i]].astype(float)
            xTr[cols[i+4]] = xTr[cols[i+4]].astype(float)
            colmean = xTr[cols[i]].mean()
            col4mean = xTr[cols[i+4]].mean()
            # random_choice = np.random.choice(arr)
            t = colmean + col4mean
            xTr[cols[i]] = xTr[cols[i]].fillna(t)
        if sscount < 8:
            sscount += 1
        else:
            sscount = 1

    for i in range(123, 183):
        xTr[cols[i]] = xTr[cols[i]].astype(float)
        colmean = xTr[cols[i]].mean()
        xTr[cols[i]] = xTr[cols[i]].fillna(colmean)
    for i in range(303, len(cols)):
        xTr[cols[i]] = xTr[cols[i]].astype(float)
        colmean = xTr[cols[i]].mean()
        xTr[cols[i]] = xTr[cols[i]].fillna(colmean)
    xTr = torch.from_numpy(xTr.values).float()
    yTr = torch.from_numpy(yTr.values).float()

    # dclf = DecisionTreeClassifier(max_depth=1)
    # dclf = RandomForestClassifier(
    #     max_samples=30, n_estimators=100, min_samples_leaf=2, max_depth=1, max_features=1)
    # clf = AdaBoostClassifier(estimator=dclf, n_estimators=40)
    # clf = AdaBoostClassifier(
    #     estimator=dclf, n_estimators=250)
    # clf = AdaBoostClassifier(
    #     n_estimators=900, learning_rate=.9, random_state=0)
    # clf = clf.fit(xTr, yTr)
    output = net(xTr)
    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    for i in range(len(yTr)):
        xTr_e = xTr[i]
        yTr_e = yTr[i]
        optimizer.zero_grad()
        output = net(xTr_e)
        loss = criterion(output[0], yTr_e)
        loss.backward()
        optimizer.step()
    return net


LFclf = cleanTrain(LFTr, 1)
LHclf = cleanTrain(LHTr, 2)
RFclf = cleanTrain(RFTr, 3)
RHclf = cleanTrain(RHTr, 4)


def cleantest(Te):
    Te.drop('dob', axis=1, inplace=True)
    Te.drop('age', axis=1, inplace=True)
    Te.drop('forceplate_date', axis=1, inplace=True)
    Te.drop('gait', axis=1, inplace=True)
    Te.drop('Gait', axis=1, inplace=True)
    Te.drop('speed', axis=1, inplace=True)
    Te.drop('Speed', axis=1, inplace=True)
    xTe = Te

    # xTe['speed'] = pd.to_numeric(xTe['speed'], errors="coerce")
    # xTe['Speed'] = pd.to_numeric(xTe['Speed'], errors="coerce")
    cols = []
    for col in xTe.columns:
        cols.append(col)
    # xTe[cols[3]] = xTe[cols[3]].astype(float)
    # colmean = xTe[cols[3]].mean()
    # xTe[cols[3]] = xTe[cols[3]].fillna(colmean)
    # xTe[cols[184]] = xTe[cols[184]].astype(float)
    # colmean = xTe[cols[184]].mean()
    # xTe[cols[184]] = xTe[cols[184]].fillna(colmean)
    fcount = 1
    for i in range(3, 123):
        if fcount > 4:
            xTe[cols[i]] = xTe[cols[i]].astype(float)
            colmean = xTe[cols[i]].mean()
            xTe[cols[i]] = xTe[cols[i]].fillna(colmean)
        if fcount < 8:
            fcount += 1
        else:
            fcount = 1
    scount = 1
    for i in range(183, 303):
        if scount > 4:
            xTe[cols[i]] = xTe[cols[i]].astype(float)
            colmean = xTe[cols[i]].mean()
            xTe[cols[i]] = xTe[cols[i]].fillna(colmean)
        if scount < 8:
            scount += 1
        else:
            scount = 1
    fscount = 1
    arr = [-1, 1]

    for i in range(3, 123):
        if fscount <= 4:
            xTe[cols[i]] = xTe[cols[i]].astype(float)
            xTe[cols[i+4]] = xTe[cols[i+4]].astype(float)
            colmean = xTe[cols[i]].mean()
            col4mean = xTe[cols[i+4]].mean()
            t = colmean + col4mean
            xTe[cols[i]] = xTe[cols[i]].fillna(t)
        if fscount < 8:
            fscount += 1
        else:
            fscount = 1
    sscount = 1
    for i in range(183, 303):
        if sscount <= 4:
            xTe[cols[i]] = xTe[cols[i]].astype(float)
            xTe[cols[i+4]] = xTe[cols[i+4]].astype(float)
            colmean = xTe[cols[i]].mean()
            col4mean = xTe[cols[i+4]].mean()
            t = colmean + col4mean
            xTe[cols[i]] = xTe[cols[i]].fillna(t)
        if sscount < 8:
            sscount += 1
        else:
            sscount = 1

    for i in range(123, 183):
        xTe[cols[i]] = xTe[cols[i]].astype(float)
        colmean = xTe[cols[i]].mean()
        xTe[cols[i]] = xTe[cols[i]].fillna(colmean)
    for i in range(303, len(cols)):
        xTe[cols[i]] = xTe[cols[i]].astype(float)
        colmean = xTe[cols[i]].mean()
        xTe[cols[i]] = xTe[cols[i]].fillna(colmean)
    xTe = torch.from_numpy(xTe.values).float()
    return xTe


LFpreds = LFclf(cleantest(LFTe))
LFpreds = torch.round(LFpreds).detach().numpy().flatten()
LFpreds = pd.Series(LFpreds.astype(int))
# LFTe = LFTe["id"].flatten()
out = pd.DataFrame({"id": LFTe["id"], 'LF': LFpreds})
out.to_csv(
    "/Users/jinhongchen/Desktop/cs4780-spring-2023-kaggle-competition/LF_test_labels.csv")
LHpreds = LHclf(cleantest(LHTe))
LHpreds = torch.round(LHpreds).detach().numpy().flatten()
LHpreds = pd.Series(LHpreds.astype(int))
# LHTe = LHTe["id"].flatten()
out = pd.DataFrame({"id": LHTe["id"], 'LH': LHpreds})
out.to_csv(
    "/Users/jinhongchen/Desktop/cs4780-spring-2023-kaggle-competition/LH_test_labels.csv")
RFpreds = RFclf(cleantest(RFTe))
RFpreds = torch.round(RFpreds).detach().numpy().flatten()
RFpreds = pd.Series(RFpreds.astype(int))
# RFTe = RFTe["id"].flatten()
out = pd.DataFrame({"id": RFTe["id"], 'RF': RFpreds})
out.to_csv(
    "/Users/jinhongchen/Desktop/cs4780-spring-2023-kaggle-competition/RF_test_labels.csv")
RHpreds = LFclf(cleantest(RHTe))
RHpreds = torch.round(RHpreds).detach().numpy().flatten()
RHpreds = pd.Series(RHpreds.astype(int))
# RHTe = RHTe["id"].flatten()
out = pd.DataFrame({"id": RHTe["id"], 'RH': RHpreds})
out.to_csv(
    "/Users/jinhongchen/Desktop/cs4780-spring-2023-kaggle-competition/RH_test_labels.csv")

print(realsub['label'].sum()/len(realsub['label']))
print(sub['label'].sum()/len(sub['label']))

# def testing(clf, X, Y):
#     indices = np.random.choice(10, replace=True, size=len(X['id']))
#     print(len(indices))
#     xTr = X.iloc[indices[:10]]
#     yTr = Y.iloc[indices[:10]]
#     preds = clf.predict(xTr)
#     print(xTr)
#     print(yTr)
#     print(len(preds))
#     res = yTr-preds
#     total = res.sum()
#     ftotal = total*total
#     loss = ftotal/50
#     return loss


# LFloss = testing(LFclf, LFx, LFy)
# LHloss = testing(LHclf, LHx, LHy)
# RFloss = testing(RFclf, RFx, RFy)
# RHloss = testing(RHclf, RHx, RHy)
# print("LF training loss " + str(LFloss))
# print("LH training loss " + str(LHloss))
# print("RF training loss " + str(RFloss))
# print("RH training loss " + str(RHloss))


# def cleaning(Te):

#     Te.drop('dob', axis=1, inplace=True)
#     Te.drop('age', axis=1, inplace=True)
#     Te.drop('forceplate_date', axis=1, inplace=True)
#     Te.drop('gait', axis=1, inplace=True)
#     Te.drop('Gait', axis=1, inplace=True)
#     Te['speed'] = pd.to_numeric(Te['speed'], errors="coerce")
#     Te['Speed'] = pd.to_numeric(Te['Speed'], errors="coerce")
#     Tecols = []
#     for col in Te.columns:
#         Tecols.append(col)

#     for i in range(3, len(Tecols)):
#         Te[Tecols[i]] = Te[Tecols[i]].astype(float)
#         colmean = Te[Tecols[i]].mean()
#         Te[Tecols[i]] = Te[Tecols[i]].fillna(colmean)
#     return Te


# LFTe = cleaning(LFTe)
# LHTe = cleaning(LHTe)
# RFTe = cleaning(RFTe)
# RHTe = cleaning(RHTe)

# LFpred = LFclf.predict(LFTe)
# LHpred = LHclf.predict(LHTe)
# RFpred = RFclf.predict(RFTe)
# RHpred = RHclf.predict(RHTe)

# print(LFpred)
# print(LHpred)
# print(RFpred)
# print(RHpred)

##
# LHTr.drop('dob', axis=1, inplace=True)
# LHTr.drop('age', axis=1, inplace=True)
# LHTr.drop('forceplate_date', axis=1, inplace=True)
# LHTr.drop('gait', axis=1, inplace=True)
# LHTr.drop('Gait', axis=1, inplace=True)
# LHyTr = LHTr['LH']
# LHTr.drop('LH', axis=1, inplace=True)
# LHxTr = LHTr

# LHxTr['speed'] = pd.to_numeric(LHxTr['speed'], errors="coerce")
# LHxTr['Speed'] = pd.to_numeric(LHxTr['Speed'], errors="coerce")
# # invalidLH = LHxTr.loc[pd.isna(LHxTr['speed']), :].index
# # LHxTr.drop(invalidLH, axis=0, inplace=True)
# LHcols = []
# for col in LHxTr.columns:
#     LHcols.append(col)

# for i in range(3, len(LHcols)):
#     LHxTr[LHcols[i]] = LHxTr[LHcols[i]].astype(float)
#     colmean = LHxTr[LHcols[i]].mean()
#     LHxTr[LHcols[i]] = LHxTr[LHcols[i]].fillna(colmean)
# LHclf = tree.DecisionTreeClassifier()
# LHclf = LHclf.fit(LHxTr, LHyTr)

# RFTr.drop('dob', axis=1, inplace=True)
# RFTr.drop('age', axis=1, inplace=True)
# RFTr.drop('forceplate_date', axis=1, inplace=True)
# RFTr.drop('gait', axis=1, inplace=True)
# RFTr.drop('Gait', axis=1, inplace=True)
# RFyTr = RFTr['RF']
# RFTr.drop('RF', axis=1, inplace=True)
# RFxTr = RFTr

# RFxTr['speed'] = pd.to_numeric(RFxTr['speed'], errors="coerce")
# RFxTr['Speed'] = pd.to_numeric(RFxTr['Speed'], errors="coerce")
# # invalidRF = RFxTr.loc[pd.isna(RFxTr['speed']), :].index
# # RFxTr.drop(invalidRF, axis=0, inplace=True)
# RFcols = []
# for col in RFxTr.columns:
#     RFcols.append(col)

# for i in range(3, len(RFcols)):
#     RFxTr[RFcols[i]] = RFxTr[RFcols[i]].astype(float)
#     colmean = RFxTr[RFcols[i]].mean()
#     RFxTr[RFcols[i]] = RFxTr[RFcols[i]].fillna(colmean)
# RFclf = tree.DecisionTreeClassifier()
# RFclf = RFclf.fit(RFxTr, RFyTr)

# RHTr.drop('dob', axis=1, inplace=True)
# RHTr.drop('age', axis=1, inplace=True)
# RHTr.drop('forceplate_date', axis=1, inplace=True)
# RHTr.drop('gait', axis=1, inplace=True)
# RHTr.drop('Gait', axis=1, inplace=True)
# RHyTr = RHTr['RH']
# RHTr.drop('RH', axis=1, inplace=True)
# RHxTr = RHTr
# RHxTr['speed'] = pd.to_numeric(RHxTr['speed'], errors="coerce")
# RHxTr['Speed'] = pd.to_numeric(RHxTr['Speed'], errors="coerce")
# RHxTr['speed'] = .replace('no data', float('nan'), regex=True)
# RHxTr['speed'] = RHxTr['speed'].replace(
#     'Not able to walk', float('nan'), regex=True)
# RHxTr['Speed'] = RHxTr['speed'].replace(
#     'Not able to trot', float('nan'), regex=True)
# invalidRH = RHxTr.loc[pd.isna(RHxTr['speed']), :].index
# RHxTr.drop(invalidRH, axis=0, inplace=True)
# RHcols = []
# for col in RHxTr.columns:
#     RHcols.append(col)

# for i in range(3, len(RHcols)):
#     RHxTr[RHcols[i]] = RHxTr[RHcols[i]].astype(float)
#     colmean = RHxTr[RHcols[i]].mean()
#     RHxTr[RHcols[i]] = RHxTr[RHcols[i]].fillna(colmean)
# RHclf = tree.DecisionTreeClassifier()
# RHclf = RHclf.fit(LFxTr, LFyTr)
