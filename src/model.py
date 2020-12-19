import os, pandas, poirot, numpy, jax
from poirot import Normal, PoissonApprox, GaussianProcess, Matern52, InferenceState, dist, infer
import jax.numpy as np
from jax.nn import sigmoid
from jax.lax import scan
from jax.nn import softmax
from jax.scipy.special import logit
from jax import vmap
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

plt.style.use('seaborn')
matplotlib.rcParams['figure.dpi'] = 200

def up(i):
    return np.clip(i, 0)

def date(s):
    return datetime.strptime(s, '%Y-%m-%d').date()

# urlretrieve("https://covid.ourworldindata.org/data/owid-covid-data.csv",
#             os.path.abspath('') + "/../data/owid-covid-data.csv")

data = pandas.read_csv(os.path.abspath('') + "/../data/owid-covid-data.csv")
data.date = [date(d) for d in data.date]
data.new_cases[pandas.isna(data.new_cases)] = 0
data.new_deaths[pandas.isna(data.new_deaths)] = 0

serodata = pandas.read_csv(os.path.abspath('') + "/../data/serosurveys_selected.csv")
serodata = serodata[pandas.notna(serodata.date)]
serodata.date = [date(d) for d in serodata.date]

dates = sorted(data.date.unique())
countries = data.location.unique().tolist()
ndays = (max(data.date) - min(data.date)).days+1
startDate = min(data.date)

cases = numpy.zeros([len(countries), ndays])
deaths = numpy.zeros([len(countries), ndays])
population = numpy.zeros(len(countries))

for i, row in data.iterrows():
    country = countries.index(row.location)
    date = (row.date - startDate).days
    cases[country, date] = row.new_cases / row.population
    deaths[country, date] = row.new_deaths / row.population
    population[country] = row.population

cases = np.array(cases)
deaths = np.array(deaths)
population = np.array(population)

# cs = [countries.index(c) for c in ["United Kingdom"]]
cs = [countries.index(c) for c in \
      ["United Kingdom", "United States", "India",
       "Brazil", "Germany", "Italy"]]

cases = cases[cs, :]
deaths = deaths[cs, :]
totalCases = cases.cumsum(1)
totalDeaths = deaths.cumsum(1)
population = population[np.array(cs)].reshape([-1, 1])
countries = [countries[c] for c in cs]
ncountries = len(countries)

date = np.arange(0, ndays)
plen = 1
periodi = date[date % plen == 0]
period = date // plen
nperiods = period.max() + 1
weekday = np.array(range(0, ndays), dtype = int) % 7

serodata = serodata[serodata.country.isin(countries)]

serodate = np.array([dates.index(d) for d in serodata.date])
serocountry = np.array([countries.index(c) for c in serodata.country])
serosample = np.array(serodata['sample.size'])
seropos = np.array(serodata['result.pct'])/100

# Because the model is discrete, all rates are ∈ (0, 1)
# The equivalent continuous parameter is -log(1-x)

def sir(cases, beta, gamma, seed):
    state = np.array([1-seed, seed, 0, 0])
    ss = []
    def loop(state, i):
        S, I, R, caught = state
        infections = beta[i]*S # rate = βI = RγI
        recoveries = gamma*I
        S -= infections
        I += infections - recoveries
        R += recoveries
        caught += cases[up(i-1)] - caught*gamma
        state = np.array([S, I, R, caught])
        return state, state
    return scan(loop, state, date)[1]

def model(ps):
    rate, gamma, seed, scale = ps['rate'], ps['gamma'], ps['seed'], ps['scale']
    dist(rate, Normal(0, 10))
    dist(scale, Normal(0, 1))
    dist(gamma, Normal(-2, 0.1))
    dist(seed, Normal(-10, 2))

    rate = sigmoid(rate)
    gamma = sigmoid(gamma)
    seed = sigmoid(seed)
    scale = np.exp(ps['scale'])

    out = vmap(sir)(cases, rate[:,period], gamma, seed)
    I, R, caught = out[:,:,1], out[:,:,2], out[:,:,3]

    dist(seropos, Normal((I+R)[serocountry, serodate], scale[serocountry,0]))

    R = np.log(rate) - np.log(I[:,periodi]) - np.log(gamma)[:,None]
    gp = GaussianProcess(Matern52(0.5,60/plen))
    dist(R, gp.at(np.arange(0, nperiods), 1e-5))

    capacity, cChange = ps['capacity'], ps['cChange']
    dist(capacity, Normal(0, 1))
    dist(cChange, Normal(0, 5))
    capacity = sigmoid(date/30/cChange + capacity[:,weekday])

    dist(cases*population, PoissonApprox((I-caught)*population*capacity, scale[:,1,None]))

    return out, R

ps = {'rate': np.zeros([ncountries, nperiods])-10,
      'capacity': np.zeros([ncountries, 7]) - 1.5,
      'cChange': np.zeros([ncountries, 1]) + 1.5,
      'gamma': np.zeros(ncountries)-2.0, # recovery
      'scale': np.zeros([ncountries, 2]),
      'seed': np.zeros(ncountries) - 10}

poirot.loglikelihood(model, ps)

state = InferenceState(ps)
state = infer(model, state)
poirot.lls.pop()

i = countries.index("United Kingdom")
ps = state.mean()
out, R = model(ps)
np.exp(ps['seed'])*population[i]
1/sigmoid(ps['gamma'])
np.exp(ps['scale'])

plt.bar(dates, cases[i,:])

plotR(i)

plt.plot([dates[i] for i in periodi], np.exp(R)[i,:])
plt.plot(dates, sigmoid(ps['rate'])[i,period])
plt.plot(dates, out[i,:,0]) # S
plt.plot(dates, out[i,:,1]) # I
plt.plot(dates, out[i,:,2]) # R
plt.plot(dates, out[i,:,3]) # caught

plt.plot(dates, sigmoid((date/30)/ps['cChange'][0] + ps['capacity'][0,weekday]))

def plotR(i):
    x = [dates[i] for i in periodi]
    out, R = jax.vmap(model)(state.samples())
    S = out[:,:,:,0]
    R = np.exp(R[:, i, ...])*S[:,i,periodi]
    # plt.plot(x, np.mean(R, 0))
    for i in range(R.shape[0]):
        plt.plot(x, R[i,...], color = 'blue', alpha = 0.5, linewidth = 0.1)
    plt.ylim(0, 3)

# import pickle
# pickle.dump(state, open("state.p", "wb"))
# state = pickle.load(open("state.p", "rb"))
# jax.config.update("jax_debug_nans", True)
