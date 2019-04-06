#Originally written by econchick and modified by Rajdeep400 (check https://gist.github.com/econchick/4666413 )
#Some extra functionallity added by Clarkguilty.
from collections import defaultdict
class Graph:
      def __init__(self):
            self.nodes = set()
            self.edges = defaultdict(list)
            self.distances = {}
             
      def add_node(self, value):
            self.nodes.add(value)
      
      def add_edge(self, from_node, to_node, distance):
            self.edges[from_node].append(to_node)
            self.distances[(from_node, to_node)] = distance
          
def dijkstra(graph, initial):
      visited = {initial: 0}
      path = defaultdict(list)
      nodes = set(graph.nodes)
      
      while nodes: 
          min_node = None
          for node in nodes:
              if node in visited:
                  if min_node is None:
                      min_node = node
                  elif visited[node] < visited[min_node]:
                      min_node = node
      
          if min_node is None:
              break
      
          nodes.remove(min_node)
          current_weight = visited[min_node]
      
          for edge in graph.edges[min_node]:
              weight = current_weight + graph.distances[(min_node, edge)]
              if edge not in visited or weight < visited[edge]:
                  visited[edge] = weight
                  path[edge].append(min_node)
      
      return path

#Returns the length of the shortest path.
def minimalPathLength(graph,initial,final):
      length = 0
      conec = final
      while(conec != initial):
            conecT = dijkstra(graph,initial)[conec][-1]
            length += graph.distances[(conecT,conec)]
            conec = conecT
      return length

#Returns a the shortest path as a list from final to initial .
def minimalPath(graph,initial,final):
      length = 0
      
      conec = final
      path = [conec]
      while(conec != initial):
            conecT = dijkstra(graph,initial)[conec][-1]
            length += graph.distances[(conecT,conec)]
            conec = conecT
            path.append(conec)
      return path

def minimalPath2(graph,initial,final):
      length = 0
      path0 = dijkstra(graph,initial)
      conec = final
      path = [conec]
      while(conec != initial):
            conecT = path0[conec]
            length += graph.distances[(conecT[-1],conec)]
            conec = conecT[-1]
            path.append(conec)
      return path

def minimalPathLength2(graph,initial,final):
      length = 0
      path0 = dijkstra(graph,initial)
      conec = final
      path = [conec]
      while(conec != initial):
            conecT = path0[conec]
            length += graph.distances[(conecT[-1],conec)]
            conec = conecT[-1]
            path.append(conec)
      return length






