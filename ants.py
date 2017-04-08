import random as r

# endpoints: pointers to dictionary elements containing edges
class edge:
    def __init__(self, endpoints, cost, decay, pheromone):
        self.endpoints = endpoints
        self.desirability = 1/cost
        self.cost = cost

        self.pheromone = pheromone # initial value
        self.start_pher = pheromone # save initial value for adjustment
        self.decay_rate = decay
        self.decay_amnt = 1-decay

    # base pheromone will need to be set after initialization because
    # nearest neighbour uses edges
    def set_pher(self, pheromone):
        self.pheromone = pheromone
        self.start_pher = pheromone

    def update_local(self):
        self.pheromone = self.decay_amnt * self.pheromone + self.decay_rate * self.start_pher

    # winning edges get updated by the global rule
    def update_global(self, length):
        self.pheromone = self.decay_amnt * self.pheromone + self.decay_rate/length

    # get the other endpoint of the edge
    def other(self, endpoint):
        for e in self.endpoints:
            if e is not endpoint: return e

# Note: some ants will find dead ends, from my reading I understand that's okay
# We just try again
class ant:
    def __init__(self, start_pos, alpha, beta, q):
        # pos = pointer to vertex
        self.pos = start_pos
        self.visited = [start_pos] # record of nodes travelled
        self.path = []
        self.home = start_pos
        # coefficients, set arbitrarily
        self.al= alpha
        self.be = beta
        self.q = q # important! in range (0, 1)

    def probs(self, moves, candidates): 
        # allowing the denominator to contain the current candidate in the sum should
        # have a small effect on choice distribution and be faster
        # if I'm wrong blame this paper: http://people.idsia.ch/~luca/acs-bio97.pdf
        denom = sum(o.pheromone ** self.al * o.desirability ** self.be for o in candidates)
        return [e/denom for e in moves]

    # idea from this nice stack overflow answer
    # https://stackoverflow.com/questions/16489449/select-element-from-array-with-probability-proportional-to-its-value
    def cumul_probs(self, p):
        cp = [p[0]]
        i = 1
        while i < len(p):
            cp.append(p[i] + p[i-1])
            i += 1

        return cp
            
    def move(self):
        candidates = [e for e in self.pos if e.other(self.pos) not in self.visited]

        # fail out if there are no moves available
        if not candidates:
            for e in self.pos:
                if e.other(self.pos) is self.home:
                    self.path.append(e)
                    self.visited.append(self.home)
                    return self.path, self.visited # signal that the tour can be completed
            return False

        moves = [e.pheromone ** self.al * e.desirability ** self.be for e in candidates]
        move = None # so the interpreter doesn't complain

        if r.random() > self.q:
            p = self.probs(moves, candidates)
            cp = self.cumul_probs(p)
            selection = r.random() * cp[-1]
            for prob in cp:
                if prob >= selection:
                    move = self.pos.index(candidates[cp.index(prob)])
                    break
        else:
            move = self.pos.index(candidates[moves.index(max(moves))]) # select the best move

        self.pos[move].update_local()
        self.path.append(self.pos[move])
        nxt = self.pos[move].other(self.pos)
        self.visited.append(nxt)
        self.pos = nxt
        return True

# nearest neighbour heuristic
def nearest_neighbour(graph):
    start = graph[r.choice(list(graph))]
    pos = start
    visited = [pos]
    path = []

    candidates = [e for e in pos if e.other(pos) not in visited]
    
    while candidates:
        candidate = candidates[0]

        for e in candidates:
            if e.cost < candidate.cost:
                candidate = e

        path.append(candidate)
        nxt = candidate.other(pos)
        visited.append(nxt)
        pos = nxt

        candidates = [e for e in pos if e.other(pos) not in visited]

    for e in pos:
        if e.other(pos) is start:
            path.append(e)

    return path

# yields true if last few paths contain the same edges
def converging(paths):
    
    if len(paths) >= 3:
        i = len(paths) - 1 
        j = i - 1
        k = j - 1
        for edge in paths[i][0]:
            if edge not in paths[j][0]: return False
            if edge not in paths[k][0]: return False
        return True
    return False

# hey look it's the ACO
def ant_colony(graph, edges):
    # parameters
    al = 1
    be = 1
    p = 0.5

    # use nearest neighbour to generate starting pheromone
    heur_length = sum(e.cost for e in nearest_neighbour(graph))
    start_pher = len(graph) * heur_length

    for edge in edges: edge.set_pher(start_pher)

    paths = []

    # run the ACO until it starts to converge
    while not converging(paths):
        ants = [ant(graph[vertex], al, be, p) for vertex in graph]
        ants_complete = []
        
        # move all ants until they've completed their paths
        while len(ants_complete) < len(ants):
            for a in ants:
                if a not in ants_complete:
                    result = a.move()
                    if result != True:
                        ants_complete.append(a)

                        if result:
                            paths.append(result)

        # find the best of all paths
        current_cost = float('inf')
        candidate_path = None
        for path in paths:
            new_cost = sum(e.cost for e in path[0])

            if new_cost < current_cost and len(path[1]) > len(graph):
                current_cost = new_cost
                candidate_path = path

        # do a global update if a suitable candidate exists
        if candidate_path:
            for e in candidate_path[0]:
                e.update_global(heur_length)
            # print_path(candidate_path[0])
            paths.append(candidate_path)
    
        # print("completed a round")

    return paths[-1]

def gen_graph(edges_list):
    decay = 0.5
    graph = {}
    edges = []

    for e in edges_list:
        if e[0] not in graph:
            graph[e[0]] = []

        if e[1] not in graph:
            graph[e[1]] = []

        # no initial pheromone, it will be set later
        new_edge = edge([graph[e[0]], graph[e[1]]], e[2], decay, 0)

        graph[e[0]].append(new_edge)
        graph[e[1]].append(new_edge)

        edges.append(new_edge)

    return graph, edges


def solution_to_string(solution_readable):
    s = str(solution_readable)[1:-1]
    new_s = '{'
    nospace = False

    for c in s:
        if c not in ("'", '"'):
            if c == '[':
                new_s += '('
                nospace = True
            elif c == ']':
                new_s += ')'
                nospace = False

            if not (nospace and c == ' '):
                new_s += c

    return new_s + '}\n\n'

def print_graph(graph):
    for vertex in graph:
        print(vertex + ':')

        for edge in graph[vertex]:
            for v in graph:
                if graph[v] == edge.other(graph[vertex]):
                    print(' ' + v)
                    break

def print_path(path):
    global GRAPH

    for edge in path:
        print([v for v in GRAPH if GRAPH[v] in edge.endpoints])

    print('')

def main():
    f = open('in.txt', 'r')
    graph_strings = f.readlines()
    f.close()

    graphs_chars = [(g[1:-1]) for g in graph_strings if g != '\n']

    graphs = []
    
    for g in graphs_chars:
        graphs.append([])
        i = 0
        while i < len(g):
            if g[i] == '(':
                i += 1
                graphs[-1].append([])

                while g[i] != ')':
                    if g[i] not in (',', ' '):
                        graphs[-1][-1].append(g[i])

                    i += 1

                graphs[-1][-1][2] = int(''.join(graphs[-1][-1][2:]))
                graphs[-1][-1] = graphs[-1][-1][:3]
            i += 1

    
    solutions = []

    solutions_readable = []

    for edges_list in graphs:
        graph, edges = gen_graph(edges_list)

        global GRAPH
        GRAPH = graph

        solution = ant_colony(graph, edges)
        solutions.append(solution)

        solution_readable = [[], sum(e.cost for e in solution[0])]
        
        for v in solution[1]:
            for key in graph:
                if graph[key] is v:
                    solution_readable[0].append(key)
                    break

        solutions_readable.append(solution_readable)


    f = open('out.txt', 'w')
    for s in solutions_readable:
        f.write(solution_to_string(s))
    f.close()

if __name__ == "__main__":
    main()
