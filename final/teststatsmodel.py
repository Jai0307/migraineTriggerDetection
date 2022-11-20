# predict probabilities with a multinomial logistic regression model
from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
import statsmodels.discrete.discrete_model as sm



# define dataset
X, y = make_classification(n_samples=10000, n_features=5, n_informative=5, n_redundant=0, n_classes=3, random_state=100)

print(X.shape)
# print(y)

# define the multinomial logistic regression model
# model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model = sm.MNLogit(y, X)
model_fit = model.fit(method='bfgs', maxiter=10000, full_output=1, disp=1, gtol=1e-10)
summary = model_fit.summary()
print(summary)
# fit the model on the whole dataset
# model.fit(X, y)
# # define a single row of input data
# row = [1.89149379, -0.39847585, 1.63856893, 0.01647165, 1.51892395, -3.52651223, 1.80998823, 0.58810926, -0.02542177, -0.52835426]
# # predict a multinomial probability distribution
# yhat = model.predict_proba([row])
# print(model.score(X, y))
# # summarize the predicted probabilities
# print('Predicted Probabilities: %s' % yhat[0])