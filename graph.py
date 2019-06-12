from collections import defaultdict, deque

class DAG:
    def __init__(self):
        self.size = 0
        self.graph = defaultdict(list)

    # Adds an edge to a directed graph
    def AddDirected(self, src, dest, weight=1):
        if self.isReachable(dest, src):
            print('Cycle detected: {0} -> {1}'.format(src, dest))
            return False
        
        # Add an edge from src to dest. A new node
        # is added to the adjacency list of src
        self.graph[src].append([dest, weight])

        # If dest is not a node in graph
        # Add dest into graph keys 
        if dest not in self.graph.keys():
            self.graph[dest] = []

        # Set the number of vertices V to be equal to the number of keys
        self.size = len(self.graph.keys())

    # A recursive function used by topologicalSort 
    def topologicalSortUtil(self, v, visited, stack):
  
        # Mark the current node as visited.
        visited[v] = True
  
        # Recur for all the vertices adjacent to this vertex 
        for i in self.graph[v]:
            if visited[i[0]] == False:
                self.topologicalSortUtil(i[0], visited, stack) 
  
        # Push current vertex to stack which stores result 
        stack.append(v)

    # The function to find longest distances from a given vertex. 
    # It uses recursive topologicalSortUtil() to get topological 
    # sorting.
    def longestPath(self, src): 
        # Mark all the vertices as not visited 
        visited = {v:False for v in self.graph.keys()}
        stack = []

        # Initialize distances to all vertices as infinite and 
        # distance to source as 0 
        dist = {v:-float('inf') for v in self.graph.keys()}
        dist[src] = 0
  
        # Call the recursive helper function to store Topological 
        # Sort starting from all vertices one by one 
        for v in self.graph.keys(): 
            if visited[v] == False: 
                self.topologicalSortUtil(v, visited, stack)

        # Process vertices in topological order
        while stack:
            # Get the next vertex from topological order
            u = stack.pop()

            # Update distances of all adjacent vertices
            if dist[u] < float('inf'):
                for neighbor in self.graph[u]:
                    v = neighbor[0]
                    if dist[v] < dist[u] + neighbor[1]:
                        dist[v] = dist[u] + neighbor[1]

        return dist

    def isReachable(self, src, dest):
        visited = {v:False for v in self.graph.keys()}
        visited[src] = True

        queue = deque()
        queue.append(src)

        while queue:
            u = queue.popleft()

            if u == dest:
                return True

            for neighbor in self.graph[u]:
                v = neighbor[0]

                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return False

    def ToDo(self, node):
        dist = self.longestPath(node)
        todo = [x for x in dist.keys() if dist[x] != -float('inf')]
        todo = sorted(todo, key=lambda x: dist[x], reverse=True)
        return todo


if __name__ == '__main__':
    graph = DAG()
    graph.AddDirected('0', '5')
    graph.AddDirected('0', '4')
    graph.AddDirected('2', '5')
    graph.AddDirected('3', '2')
    graph.AddDirected('1', '3')
    graph.AddDirected('1', '4')
    graph.AddDirected('5', '1')
    
    node = '1'
    todo = graph.ToDo(node)
    print('To do list ({0}):'.format(node))
    for i in range(len(todo)):
        print('{0}. {1}'.format(i+1, todo[i]))
