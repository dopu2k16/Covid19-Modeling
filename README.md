# Abstract
In December 2019, a novel coronavirus was found
in a seafood wholesale market in Wuhan, China. World
Health Organization (WHO) officially named this coronavirus
as COVID-19. Since the first patient was hospitalized on
December 12, 2019, China has reported a total of 78,824
confirmed COVID-19 cases and 2,788 deaths as of February
28, 2020. The COVID-19 has been successfully contained
in China but is spreading all over the world. COVID-19
epidemic is prone to disrupt and crumble the existing health-
care infrastructures in both the developed and developing
world. COVID19 also impacts people’s daily life and country’s
economic development. In this paper, we adopt mathematical
epidemic models such as Susceptible-Infected-Recovery (SIR),
Susceptible-Infected-Recovery-Fatality/Deaths (SIR-F) to sim-
ulate the epidemic on the data available for the entire world
and future projections on the number of infections, deaths in
six specific countries (Italy, France, Spain, Germany, USA,
and India) across a time-frame of 7 days, 1 month, 3 months,
and 3 years in future. We analyzed the epidemic by extending
the SIR-F model with controlled parameters and simulating
the behavior on our default case study data. We also fit other
mathematical models such as exponential and logistic models
to C(t), the cumulative number of positive infections trajectory
function. In the latter section, we also used statistical machine
learning techniques such as Polynomial regression, support
vector machine regression, and simple neural network such
as multilayer perceptron to better understand and learns the
underlying pattern of the real epidemic growth and the virus
proliferation pattern. We found out that the predictions by
the logistic model was underreported, i.e, the actual trajectory
is more complex than the logistic model. However, we found
out that different models found to be better in modeling the
pandemic outbreak in respective countries. We also performed
data analysis to project the infection, recovery, and death
statistics from the real data and also calculated the growth
factor of pandemic outbreak in the countries and grouping them
respectively. To our future projections and analysis, we found
out USA, followed by India are gonna be the most affected
countries with each resulting into millions of positive infections
cases and deaths.

## Index Terms
COVID-19, epidemic modeling, data analysis,
SIR/SIR-F modeling, machine learning, polynomial regression,
support vector machine, logistic modeling, predictions.

## Install the dependencies
```python
pip install requirements.txt
```