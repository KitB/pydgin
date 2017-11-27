import csv
import numpy as np


def do_epoch(groups, delta_days, delta_groups, lru, lrc):
    # Update population - sit die deel in sy eie funksie en roep dit twee keer 
    # vir die pyp 
    # die feature_transformation maak klaar txt files met 'n kolom vir die pyp
    for i, (group, group_delta) in enumerate(zip(groups, delta_groups)):
        if group_delta > 0:
            immigrants = np.zeros((group_delta, len(groups)))  # agent net sy array
            immigrants[:, i] = 1
            # implementeer my oorspronkike oplossing vir die pipe hier 
            # los die feature_transformation soos wat dit tans is, dit sal die drie 
            # kolomme gee wat jy wil he, implementeer dan die half/half ding van daar af 
            groups[i] = np.concatenate((groups[i], immigrants))
        elif group_delta < 0:
            indices = np.arange(group.shape[0])
            idx_to_keep = np.random.choice(indices,
                                           size=(group.shape[0] + group_delta),  # group_delta is negative
                                           replace=False)
            groups[i] = groups[i][idx_to_keep]

        # die deel stoor die groep waarvan die agent oorspronklik kom
        group_idx = np.concatenate([[i] * group.shape[0] for i, group in enumerate(groups)])
        everyone = np.concatenate(groups)

        # e.g.
        # groups = [[1, 2], [3, 4, 5], [6, 7, 8]]
        # everyone =  [1, 2, 3, 4, 5, 6, 7, 8]
        # group_idx = [0, 0, 1, 1, 1, 2, 2, 2]

    # Communicate
    # Select conversation partners
    pop_size = everyone.shape[0]
    selections = np.random.randint(pop_size, size=(delta_days, pop_size))
    # die array stoor die lukrake indekse van gesels maats 
    # dis die deel wat sal moet verander vir 'n gestrukrureerde koloniale situasie soos beskryf  
    # in Tria et al 2015, gebaseer op Chaundenson 2001

    for day in selections:
        # The following index manipulation implements "people don't speak to themselves"
        all_idx = np.arange(pop_size, dtype=np.int32)  # [0, 1, 2, 3, ..., pop_size]
        neq_idx = day != all_idx  # a boolean array of whether or not each index has "you == me"
        # this is element-wise ^

        us_idx = all_idx[neq_idx]
        # `day` might reasonably be called `day_idx`
        them_idx = day[neq_idx]

        us = everyone[us_idx]
        them = everyone[them_idx]

        our_langs = speak(us)
        their_langs = speak(them)

        learning_rates = np.where(our_langs == their_langs, lru, lrc)
        learning_rates.shape += (1,)  # Need to reshape so that it can be multiplied later

        us_updates = make_update_array(us, their_langs, learning_rates)
        them_updates = make_update_array(them, our_langs, learning_rates)

        us += us_updates
        them += them_updates

        everyone[us_idx] = us
        everyone[them_idx] = them

    # Reconstruct groups
    for i, group in enumerate(groups):
        group[:] = everyone[group_idx == i]

    return groups

def update_population():
    # bou die groepe, maar wat van die pyp? bou dit dan ander groupe?
        # opsies:
        # pyp van twee reeds bestaande opsies, 
        # pyp van een bestaande en een nuwe opsie, 
        # pyp van twee nie bestaande opsies 

        # eintlik net twee, of die opsie bestaan, of dit doen nie.
        # split op pyp 
        # deal met bestaande opsies
            # kry bestaande opsie se indeks 
    pass

    #return groups - ek dink


def make_update_array(population, columns, learning_rates):
    out = np.full_like(population, -1)  # an array of -1 the same shape as population
    columns_idx = (np.arange(out.shape[0]), columns)
    out[columns_idx] = 1  # Now it's positive on the columns selected
    out *= learning_rates

    # learning rate * (1 - value) when in columns, lr * value otherwise
    scales = np.copy(population)
    scales[columns_idx] = 1 - scales[columns_idx]

    out *= scales
    return out


def speak(population):
    cumulative_sums = np.cumsum(population, axis=1)
    # We add (1,) to the shape so that numpy knows to do elementwise greater-than later
    random_chances = np.random.random((population.shape[0], 1))
    return np.argmax(cumulative_sums > random_chances, axis=1)


class Model(object):
    def __init__(self, populations_path, epochs_path,
                 learning_rate_coordinated,
                 learning_rate_uncoordinated):
        # Skip the header row of the population, this means we only work on the outputs of
        # 'feature_transformation.py'
        self.populations = np.loadtxt(populations_path, dtype=np.int32,
                                      skiprows=1, delimiter=',', ndmin=2)
        self.epochs = np.loadtxt(epochs_path, dtype=np.int32)
        self.lr_c = learning_rate_coordinated
        self.lr_u = learning_rate_uncoordinated

        # We skipped the headers row above so lets open it with the csv library
        with open(populations_path, 'rb') as popfile:
            self.headers = csv.reader(popfile).next() # stoor die hearders, Model.headers roep dit 

    def run(self):
        n_epochs, n_langs = self.populations.shape
        groups = [np.ndarray(shape=(0, n_langs), dtype=np.int32)
                  for _ in xrange(self.populations.shape[1])]

        for time_delta, groups_delta in zip(self.epochs, self.populations):
            groups = do_epoch(groups, time_delta, groups_delta, self.lr_c, self.lr_u)

        self.groups = groups

    def calc_distribution(self):
        everyone = np.concatenate(self.groups)
        return np.average(everyone, axis=0)

    @property
    def winner(self):
        return self.headers[np.argmax(self.calc_distribution())]


def get_argparser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('population', type=str)
    ap.add_argument('time', type=str)
    ap.add_argument('lc', type=float, nargs='?', default=0.01)
    ap.add_argument('lu', type=float, nargs='?', default=0.001)
    return ap


def get_args():
    return get_argparser().parse_args()


def main():
    args = get_args()
    do_it(args.population, args.time, args.lc, args.lu)


def do_it(population, time, lc, lu):
    m = Model(population, time, lc, lu)
    m.run()
    return m.get_distribution()


if __name__ == '__main__':
    main()
