import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import tslearn.early_classification as tsl
from tslearn.neural_network import TimeSeriesMLPClassifier
from tslearn.datasets import UCR_UEA_datasets

seed = 0
numpy.random.seed(seed)

X_train, Y_train, X_test, Y_test = UCR_UEA_datasets().load_dataset("ECG200")

# Scale time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)

model = tsl.NonMyopicEarlyClassifier(n_clusters=2, base_classifier=TimeSeriesMLPClassifier(solver="sgd",
                                                                                           hidden_layer_sizes=41,
                                                                                           random_state=1,
                                                                                           max_iter=10000),
                                     min_t=4, lamb=1, cost_time_parameter=0.01, random_state=None
                                     )

model.fit(X_train, Y_train)

prediction01, time_prediction01 = model.predict_proba_and_earliness(X_test)
auc01 = roc_auc_score(Y_test, numpy.amax(prediction01, axis=1))

model.set_params(cost_time_parameter=0.05)
prediction05, time_prediction05 = model.predict_proba_and_earliness(X_test)
auc05 = roc_auc_score(Y_test, numpy.amax(prediction05, axis=1))

model.set_params(cost_time_parameter=0.1)
prediction1, time_prediction1 = model.predict_proba_and_earliness(X_test)
auc1 = roc_auc_score(Y_test, numpy.amax(prediction1, axis=1))

print("The AUC for cost time 0.01 is", auc01, "for cost time 0.05 it is",
      auc05, "and for 0.1 is is", auc1)

plt.figure()
plt.plot(X_test[1], label="An example of ECG")
plt.axvline(x=time_prediction1[1], label="Time of prediction for cost time 0.1")
plt.axvline(x=time_prediction05[1], label="Time of prediction for cost time 0.05")
plt.axvline(x=time_prediction01[1], label="Time of prediction for cost time 0.01")
plt.legend()
plt.title("Early classification of an ECG's time series with different time cost")
plt.show()

plt.figure()
plt.hist(time_prediction01)
plt.title("Histogram of classification time from the non-myopic model on ECGTwoLead dataset")
plt.show()


