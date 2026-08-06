# churn-customer
This project's goal is to predict whether customer will discontinue his subscription with company or not. Which will decrease company cost.

### Summary (based on analysis and modeling):

#### Trade-offs

The most obvious one is precision-recall trade-off, which in our case recall is higher than precision 
- that means model will mark loyal customers as churns pretty often -> company'll spend additional money on "bringing back" customer, but thanks to it model also will capture `91%` of all churn customers.
---
#### Bisiness problem i solved

Model can help telco company to determine whether customer will leave or not -> this will decreases costs of ads, campaings, etc.

Let's imagine a scenario in which there're 1 million of customers, with this new model i calculated that:

- 409,756 subscribers will be flagged by model

- 168,000 of them are actual churns

- 241,756 of them are fake alarms

- about 67,200 customers will be saved

- Gross Revenue Saved - $40,320,000

- Total Campaign Cost - $20,487,800

- Net Annual Savings - $19,832,200

Still model misses 16% of churns and company has to offer discounts for 240k fake alarms customers, but overall savings are higher than costs

- also with correlation analysis i found out that the most common reason of customer leaving is too high prices and in our cases discounts can help customer to stay
---
#### The problems that i faced while doing this project:
- Program started to use default threshold (0.5) intead of custom one (0.25) which in result lead to wrong metrics results

Because of this most of my predictions has failed and i got F1 scores for both train and test at 0.000, but it was quickly fixed by switching back to custom threshold

- First try Dockers image size was 4GB, which for not a deep learning project is to high.

Fixed by command Docker to copy only neccesary folders and deleting some unnecessary packages (like nvidia). Now the newest docker image is only 2.5GB

- Deployed model was failing to predict new data

The issue was loading model with pyfunc, which has now predict_proba function. Had to switch to mlflow.LightGBM loading of model instead.