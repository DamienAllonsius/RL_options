import os
import matplotlib.pyplot as plt
import variables

class SaveData(object):

    def __init__(self, path, seed):
        self.seed = seed
        self.path = path + "_seed_"
        self.delete_file()
        
    def delete_file(self):
        if os.path.exists(self.path + str(self.seed)):
            os.remove(self.path + str(self.seed))
  
    def record_data(self, t, reward):
        f = open(self.path + str(self.seed), "a")
        f.write(str(t) + " " + str(reward) + "\n")
        f.close()

    def plot_data(self):
        """
        plot the average values of the total reward
        """
        X, Y = [], []
        for line in open(self.path + str(0), 'r'):
            values = [float(s) for s in line.split()]    
            X.append(values[0])
            Y.append(values[1])
            
        for seed in range(1, NUMBER_SEEDS):
            count = -1
            for line in open(self.path + str(seed), 'r'):
                values = [float(s) for s in line.split()]    
                count += 1
                Y[count] += values[1]

        for k in range(len(Y)):
            Y[k] /= NUMBER_SEEDS

        plt.plot(X, Y)
        plt.savefig(self.path + "IMG")

