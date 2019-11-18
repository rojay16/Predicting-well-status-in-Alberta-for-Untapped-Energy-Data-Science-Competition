# Predicting-well-status-in-Alberta-for-Untapped-Energy-Data-Science-Competition

Keywords: Data science competition, Oil and gas analytics, predicting well status, predicting well production,gradient boosting, deep learning, tensorflow, xgboost, pandas, scikit-learn, cross validation, hyperparameter tuning.

This is the script submitted to the Untapped Energy reClaim Data competition : Classification challenge. 
The model created in this script placed 5th in the competition with over a 90 percent accuracy and a similar model place 4th in the Untapped Energy reClaim Data competition : Regression challenge.

In this competition I used data  with over 60 features (such as drilling date, well depth and location) from over 700,000 wells to predict the status of a well (classification) and initial production of the well (regression). xgboost and tensorflow were used to make the predictions and used I pandas and scikit-learn to preprocess the data. Feature engineering was done on the data removing fearues deemed superfluous and that could contribute to overfitting. Features were added that were thought could aid the model (such as the length of time between the well completion and the last report on the well). The numerical data was scaled to generate better solution for the optimization algorithm. Hyperparameter tuning was applied to the xgboost model finding the optimal tree depth and step length for the xgboost model. The neural net models created utilized regularization methods such as dropout and L1 regularization. 

