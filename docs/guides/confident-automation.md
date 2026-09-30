# Automate the confident decisions

Every answer carries a calibrated probability, so a common deployment is to act automatically on the answers the model
is sure about and send the rest to a person. How much you can automate depends on the accuracy you need.

## Measured on 2,000 business decisions

opendecider-nano on the typed-decisions test split (400 cases across customer service, invoice processing, security
incidents and AI-agent traces; no OpenDecider model was trained on it):

| automate when p ≥ | automated | accuracy of the automated ones | sent to a person |
|---|---|---|---|
| (everything) | 100% | 0.796 | 0% |
| 0.5 | 76% | 0.873 | 24% |
| 0.6 | 51% | 0.941 | 49% |
| 0.7 | 32% | 0.972 | 68% |
| 0.8 | 21% | 0.998 | 79% |
| 0.9 | 13% | 0.996 | 87% |

Automating the answers with p ≥ 0.6 handles half the decisions at 94% accuracy. The gold labels come from a teacher
model and are themselves imperfect, so read the top rows as "agrees with the labels", not as a ceiling.

Run it yourself, on any model:

```bash
pip install opendecider pandas pyarrow
python examples/confident_automation.py                                            # nano, about a minute on a laptop CPU
python examples/confident_automation.py --model manjunathshiva/opendecider-small-td   # pip install "opendecider[small]"
```

## Across models

Accuracy on the most confident share of decisions:

| benchmark | model | all decisions | most confident 70% | most confident 50% |
|---|---|---|---|---|
| typed-decisions | opendecider-small-td | 0.792 | 0.893 | **0.949** |
| typed-decisions | opendecider-nano | 0.796 | 0.894 | 0.943 |
| typed-decisions | opendecider-medium-td | 0.788 | 0.896 | 0.948 |
| typed-decisions | opendecider-large-td | 0.801 | **0.901** | 0.947 |
| typed-decisions | TypeSafe Jev 1.13 | 0.754 | 0.839 | 0.882 |
| general (200) | opendecider-large-td | 0.750 | 0.807 | **0.870** |
| general (200) | TypeSafe Jev 1.13 | 0.730 | **0.829** | 0.860 |
| general (200) | opendecider-small | 0.735 | 0.800 | 0.830 |
| general (200) | opendecider-medium-td | 0.765 | 0.807 | 0.820 |
| general (200) | Laya | 0.545 | 0.543 | 0.550 |

On typed-decisions, automating the confident half gives 94–95% accuracy with OpenDecider against 88% with Jev. On the
general decisions Jev ranks its confidence better than the 4B and 30B models, but opendecider-large-td passes it. Laya's
confidence barely separates right from wrong answers here, so check any model's confidence on your own data before
thresholding on it.

## In code

```python
from opendecider import load, Choice

model = load("manjunathshiva/opendecider-nano")
AUTO = 0.6   # pick the threshold from a sample of your own labelled decisions

a = model.system_one(ticket, {"team": Choice("Which team should handle this?",
                                             {"billing": "charges, refunds", "technical": "bugs, outages"})})["answers"]["team"]
if a["confidence"] >= AUTO:
    route_to(a["choice"])
else:
    send_to_person(ticket, suggestion=a["choice"], probabilities=a["probabilities"])
```

Choose the threshold on a few hundred of your own labelled decisions: the right value depends on your data, your
questions and the cost of a wrong automatic action.
