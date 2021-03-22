import math

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pyabc
import pandas as pd
import time
from functools import partial


def wf_with_selection_and_mutation(wt_freq, population_size, fitness, mutation_rate):
    """
    This is the heart of the model: bi-allele Wright-Fisher with selection and mutation.
    Math adapted from: https://academic.oup.com/sysbio/article/66/1/e30/2670014#syw056M2
    Note that the order of selectiona and mutation is relevant to the results.
    """
    wt_freq = (wt_freq * fitness) / ((1 - wt_freq) + (wt_freq * fitness))     # selection
    wt_freq = wt_freq * (1 - mutation_rate) + (1 - wt_freq) * mutation_rate   # mutation
    wt_freq = np.random.binomial(population_size, wt_freq) / population_size  # bottleneck
    return wt_freq


def wf_multiple_generations(generations_number, wt_freq, population_size, fitness, mutation_rate, sequence_sample_size,
                            color=None, label=None, plot=True):
    freqs = [wt_freq]
    for i in range(generations_number - 1):
        wt_freq = wf_with_selection_and_mutation(wt_freq=wt_freq, population_size=population_size,
                                                 fitness=fitness, mutation_rate=mutation_rate)
        freqs.append(wt_freq)
    freqs = np.random.binomial(sequence_sample_size, freqs) / sequence_sample_size  # sequence sampling
    if plot:
        plt.plot(range(generations_number), freqs, color=color, label=label, alpha=0.1)
    return np.array(freqs)


def simulate_data(generations_number, wt_freqs, population_size, fitness, mutation_rate, sequence_sample_size,
                  color=None, label=None, plot=True):
    """
    This function has 3 uses:
        1. Its the model's simulator
        2. It is used to synthesize data for posterior predictive checks
        3. It's used to plot the data and get a better intuition to what is actually going on
    """
    if plot:
        plt.xlabel('Generation')
        plt.ylabel('Frequency')
    data = []
    if label is None:
        label = f"w={fitness}, mu={mutation_rate}"
    first_run = True
    for freq in wt_freqs:
        datum = wf_multiple_generations(generations_number=generations_number, wt_freq=freq, plot=plot,
                                        population_size=population_size, fitness=fitness, mutation_rate=mutation_rate,
                                        color=color, sequence_sample_size=sequence_sample_size,
                                        label=label if first_run else None)
        data.append(datum)
        first_run = False
    if plot:
        leg = plt.legend()
        for lh in leg.legendHandles:
            lh.set_alpha(1)
    return pd.DataFrame(data)


def smc_model(parameters, intial_freq, sequence_sample_size, pop_size, gen_num):
    mutation_rate = 10 ** parameters['mu']
    fitness = parameters['w']
    return {'a': simulate_data(generations_number=gen_num,  wt_freqs=intial_freq, population_size=pop_size,
                               mutation_rate=mutation_rate, fitness=fitness, sequence_sample_size=sequence_sample_size,
                               plot=False, color=False, label=False)}


def l1_distance(simulation, data):
    return abs(data['a'] - simulation['a']).sum().sum()  # double sum for multi allele compatibility


def run_smc(priors, data, epsilon, max_episodes, smc_population_size, sequence_sample_size, pop_size, gen_num,
            distance_function=l1_distance):
    start = time.time()
    initial_freq = data[0]
    try:
        iter(initial_freq)
    except:  # the avg method only gets one intial freq
        initial_freq = [initial_freq]
    model = partial(smc_model, intial_freq=initial_freq, sequence_sample_size=sequence_sample_size,
                    pop_size=pop_size,gen_num=gen_num)
    model.__name__ = 'model with params'  # SMC needs this for some reason...
    abc = pyabc.ABCSMC(
            model, priors, distance_function, smc_population_size)
    db_path = ("sqlite:///test.db")
    smc_post = abc.new(db_path, {'a': data})
    smc_post = abc.run(minimum_epsilon=epsilon, max_nr_populations=max_episodes)
    print("SMC run time: ", round(time.time()-start,2))
    print("Total number of SMC simulations: ", smc_post.total_nr_simulations)
    return smc_post


def infer_megapost(prior_dist, data, epsilon, max_episodes, smc_population_size, sequence_sample_size,
                   gen_num, pop_size):
    megapost = pd.DataFrame()
    for i, row in data.iterrows():
        smc_run = run_smc(prior_dist, row, epsilon, max_episodes, smc_population_size,
                          sequence_sample_size=sequence_sample_size, gen_num=gen_num, pop_size=pop_size)
        df, weights = smc_run.get_distribution()
        df['weights'] = weights
        megapost = pd.concat([megapost, df])
    return megapost


def plot_2d_kde_from_df(df, real_w, real_mu, ax, title):
    sns.kdeplot(data=df, x='mu', y='w', weights='weights', ax=ax)
    ax.plot(math.log10(real_mu), real_w, marker='o', color='red')
    ax.set_xlabel('log10(mu)')
    ax.set_ylim([-0.2,2.2])
    ax.set_xlim([-8, -2])
    ax.set_title(title)


def plot_kdes(fitness, mutation_rate, posts):
    fig, axes = plt.subplots(1, len(posts))
    i = 0
    for title, df in posts.items():
        try:
            plot_2d_kde_from_df(df, fitness, mutation_rate, axes[i], title)
        except:
            plot_2d_kde_from_df(df, fitness, mutation_rate, axes, title)  # for one plot plt returns the ax itself
        i += 1
    plt.gcf().set_size_inches((12, 4))
    plt.gcf().tight_layout()


def run_methods(mutation_rate, fitness, epsilon=0.005, max_episodes=10, w_prior=(0,2), mu_prior=(-7,5),
                smc_population_size=1000, wt_freqs=(0,), methods='All', model_ss=10 ** 5, data_ss=10 ** 5,
                pop_size=10**8, gen_num=10):
    """
    This is the main tool which runs multiple methods and graphs their posteriors.
    """
    if methods== 'All':
        methods = ['megapost', 'megadist', 'avg']
    print("Creating dataset...")
    data = simulate_data(generations_number=gen_num, wt_freqs=wt_freqs,  mutation_rate=mutation_rate,
                         population_size=pop_size, fitness=fitness, sequence_sample_size=data_ss,
                         plot=False)
    prior_dist = pyabc.Distribution(w=pyabc.RV("uniform", w_prior[0], w_prior[1]),
                                    mu=pyabc.RV("uniform", mu_prior[0], mu_prior[1]))
    posts = dict()
    if 'megapost' in methods:
        print("Inferring with mega posterior")
        posts['megapost'] = infer_megapost(prior_dist=prior_dist, data=data, epsilon=epsilon,
                                           max_episodes=max_episodes, smc_population_size=smc_population_size,
                                           sequence_sample_size=model_ss, gen_num=gen_num, pop_size=pop_size)
    if 'megadist' in methods:
        print("Inferring with mega distance function")
        megadist, ws = run_smc(priors=prior_dist, data=data, epsilon=len(wt_freqs)*epsilon, max_episodes=max_episodes,
                               smc_population_size=smc_population_size, gen_num=gen_num, pop_size=pop_size,
                               sequence_sample_size=model_ss).get_distribution()
        megadist['weights'] = ws
        posts['megadist'] = megadist
    if 'avg' in methods:
        print("Inferring from avgs")
        avg, ws = run_smc(priors=prior_dist, data=data.mean(), epsilon=epsilon, max_episodes=max_episodes,
                          smc_population_size=smc_population_size, gen_num=gen_num, pop_size=pop_size,
                          sequence_sample_size=model_ss).get_distribution()
        avg['weights'] = ws
        posts['avg'] = avg
    plot_kdes(fitness, mutation_rate, posts)
    return posts
