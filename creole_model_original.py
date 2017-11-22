import random


class Agent:
    def __init__(self, lc, lu, langs):
        self.lc = lc
        self.lu = lu
        self.langs = langs  # probabilty array of all languages
        self.rand = random.random()

    def setNativeLanguage(self, nativeLang, nrLangs):
        # initialises the langs array with the correct numer of
        # languages while setting their probabilities to 0.
        # Sets the probability of the native language to 1.
        for i in xrange(nrLangs):
            self.langs.append(0)
        self.langs[nativeLang] = 1

    def getLangs(self):
        return self.langs

    def speak(self):
        # returns a random language based on language probability array
        qs = 0  # not sure why it's called this
        currentLang = -1
        while qs < self.rand:
            currentLang += 1
            qs += self.langs[currentLang]
        return currentLang

    def update(self, myLang, yourLang):
        # updates language probability array depending on language heard
        learning = 0
        if(myLang == yourLang):
            learning = self.lc
        else:
            learning = self.lu
        # if myLang !=0:  # no idea why this is necissary
        self.langs[yourLang] += learning * (1 - self.langs[yourLang])
        for i in xrange(len(self.langs)):
            if i != yourLang:
                self.langs[i] -= learning * self.langs[i]


class Model:

    def __init__(self, populationMatrix):
        self.population = populationMatrix
        self.agents = []
        self.groups = []  # the beginning of the error
        self.random = random.random()

    def run(self, timeVector, lc, lu):
        self.time = timeVector
        nrLangs = len(self.population[0])
        for h in xrange(nrLangs):  # creates a new group for each demographic
            self.groups.append([])
        i = 0
        for t in self.time:
            # print "Time interval ", i, " of ", len(self.time), t, " days"
            # Update population
            for j in xrange(nrLangs):
                nrAgents = self.population[i][j]
                if nrAgents > 0:  # population growth
                    for k in xrange(nrAgents):
                        langPArr = []
                        immigrant = Agent(lc, lu, langPArr)
                        immigrant.setNativeLanguage(j, nrLangs)
                        self.agents.append(immigrant)
                        self.groups[j].append(immigrant)  # added to both population and group
                elif nrAgents < 0:  # population decline
                    random.shuffle(self.groups[j])
                    end = len(self.groups[j])
                    beginning = end+nrAgents
                    del self.groups[j][beginning:end]  # only removed from group

            # Communicate
            for l in xrange(t):  # every agent communicates once a day
                nrAgents = len(self.agents)
                random.shuffle(self.agents)
                for me in self.agents:
                    r = random.randint(0, int(nrAgents)-1)
                    you = self.agents[r]
                    if me != you:
                        myLang = me.speak()
                        yourLang = you.speak()
                        me.update(myLang, yourLang)
                        you.update(yourLang, myLang)
            i += 1

        return self.agents


class Pidgin:

    def __init__(self, populationMatrixFile, timeVectorFile, lc, lu):
        self.pop = self.parseMatrix(populationMatrixFile)
        self.t = self.parseVector(timeVectorFile)
        self.lc = lc
        self.lu = lu
        agents = self.run(self.pop)
        print str(self.getDistribution(agents))

    def run(self, population):
        m = Model(population)
        return m.run(self.t, self.lc, self.lu)  # returns a list of agents

    def parseMatrix(self, matrixFile):
        with open(matrixFile) as file:
            lines = file.readlines()
            matrix = [[int(num) for num in line.split()] for line in lines]
        return matrix

    def parseVector(self, vectorFile):
        with open(vectorFile) as file:
            lines = file.readlines()
            vector = [int(line) for line in lines]
        return vector

    def getDistribution(self, agents):
        allDist = []
        for agent in agents:
            allDist.append(agent.getLangs())
        distTuple = zip(*allDist)
        sumDist = [sum(tup) for tup in distTuple]
        avDist = [dist/float(len(agents)) for dist in sumDist]
        return avDist


p = Pidgin("F81_pop.txt", "Time.txt", 0.01, 0.001)
