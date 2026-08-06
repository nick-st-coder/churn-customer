# churn-customer
This project's goal is to predict whether customer will discontinue his subscription with company or not. This will help company to decrease costs.

### Summary:

#### Trade-offs

The most obvious one is precision-recall trade-off, but in our case recall is valued more, since in our project cost of false negative error is very high. So i decided to find a balance between those two and final model managed to find 91% of churns, but it makes a lot of fake alarms, because of  50% precision score.
- that means model will mark loyal customers as churns pretty often -> company will spend additional money on "bringing back" customer, but thanks to this sacrifice model will also capture `91%` of the actual churns.

The second tradeoff is the model. 
- XGBoost model was the fastest across all of them, but showed significantly worse outputs than LGBM. On the other side the slowest model - Random Forest, showed the best results, but I decided to work with LGBM, since its perfomance was balanced, fast and still very accurate. 
---
#### Bisiness problem i solved

Model can help companies to determine whether customer will leave or not -> this will decreases costs of ads, campaings to make customer want to return, etc.

Let's imagine a scenario in which the company has 1 million customers. Using this churn prediction model:

- 464,000 customers would be flagged as being at risk of churning.

- 182,000 of them would be actual churners.

- 282,000 would be false positives.

- Assuming 40% of contacted churners accept a retention offer, approximately 72,800 customers would be retained.

- Gross annual revenue saved: $43,680,000

- Total retention campaign cost: $23,200,000

- Estimated net annual savings: $20,480,000

Although the model still misses approximately 9% of churning customers and the company spends money on retention offers for around 282,000 customers who would not have churned, the estimated benefit remains substantially higher than the campaign cost.

With exploratory data analysis i found out that the most common reason of customer leaving are high prices and in our cases discounts can help customer to change their minds.
---
#### The problems that i faced while doing this project:
- Program started to use default threshold (0.5) intead of custom one (0.25) which in result lead to wrong metrics scores.

Because of this most of my predictions has failed and i got F1 scores for both train and test at 0.000, but it was quickly fixed by switching back to custom threshold.

- First Dockers image size was 4GB, which for a not-deep-learning project is to high.

Fixed by forcing Docker to copy only neccesary folders and deleting some unnecessary packages (like nvidia and xgboost). Now the newest docker image is only 2.5GB

- Deployed model was failing to predict new data.

The issue was loading model with pyfunc, which has no `predict_proba` function. I had to switch to mlflow.lightgbm and load model with that instead.