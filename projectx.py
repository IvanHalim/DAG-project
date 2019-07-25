import pandas as pd
import networkx as nx
import networkx.drawing.nx_pydot as pdt
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as py
from IPython.display import Image, display
from datetime import datetime, timedelta
from collections import defaultdict

def read_files(activity_file, edge_file, header=['infer','infer'],
               id=[False,False], coeffs=[1,4,1], key=True):
    
    # Read activity file and edge file into a pd.DataFrame
    df1 = pd.read_csv(activity_file, header=header[0])
    df2 = pd.read_csv(edge_file, header=header[1])
    
    # If either file contains an ID column, drop it
    if id[0]:
        df1 = df1.drop(df1.columns[0], axis=1)
    if id[1]:
        df2 = df2.drop(df2.columns[0], axis=1)
        
    # First we're going to convert the activity file into
    # a dictionary. Iterate over rows in dataframe.
    activity_dict = defaultdict(dict)
    for _, row in df1.iterrows():
        
        # The node is the first element of row
        # Optimistic time is the second element
        # Normal time is the third element
        # Pessimistic time is the fourth element
        node = row.iat[0]
        activity_dict[node]['Optimistic'] = row.iat[1]
        activity_dict[node]['Normal'] = row.iat[2]
        activity_dict[node]['Pessimistic'] = row.iat[3]
        
        # Since we haven't calculated the critical path
        # we're assuming the node is of type 'normal'
        # instead of 'critical'
        activity_dict[node]['Type'] = 'normal'
        activity_dict[node]['Status'] = 'To Do'
        activity_dict[node]['StartDate'] = None
        activity_dict[node]['FinishDate'] = None
        
        # Calculated the expected time using the formula:
        #
        #    E = (O + 4M + P) / 6
        #
        # where O = Optimistic Time
        #       M = Most Likely Time / Normal Time
        #       P = Pessimistic Time
        #
        # The coefficients can be changed in the 'coeffs'
        # parameter.
        #
        activity_dict[node]['Expected'] =\
            round((coeffs[0]*activity_dict[node]['Optimistic'] +\
                   coeffs[1]*activity_dict[node]['Normal'] +\
                   coeffs[2]*activity_dict[node]['Pessimistic'])\
                   / float(sum(coeffs)), 2)
        
        activity_dict[node]['Actual'] = activity_dict[node]['Expected']
    
    # Next we're going to convert the edge file into a list
    edge_list = []
    if key:
        # If the edge file uses parent IDs and child IDs
        # then we need to refer to its location in the
        # activity file
        for _, row in df2.iterrows():
            u = df1.iat[row.iat[0]-1, 0]
            v = df1.iat[row.iat[1]-1, 0]
            edge_list.append((u, v))
    else:
        # If the edge file uses parent nodes and child nodes
        # instead of IDs, simply add an edge between the nodes
        for _, row in df2.iterrows():
            u = row.iat[0]
            v = row.iat[1]
            edge_list.append((u, v))
            
    return activity_dict, edge_list


def to_activity_dict(activity_matrix, coeffs=[1,4,1]):
    """
    This function converts a 2-dimensional matrix into
    a dictionary of nodes and attributes
    """
    activity_dict = defaultdict(dict)
    for row in activity_matrix:
        
        # The node is the first element of row
        # Optimistic time is the second element
        # Normal time is the third element
        # Pessimistic time is the fourth element
        node = row[0]
        activity_dict[node]['Optimistic'] = row[1]
        activity_dict[node]['Normal'] = row[2]
        activity_dict[node]['Pessimistic'] = row[3]
        
        # Since we haven't calculated the critical path
        # we're assuming the node is of type 'normal'
        # instead of 'critical'
        activity_dict[node]['Type'] = 'normal'
        activity_dict[node]['Status'] = 'To Do'
        activity_dict[node]['StartDate'] = None
        activity_dict[node]['FinishDate'] = None
        
        # Calculated the expected time using the formula:
        #
        #    E = (O + 4M + P) / 6
        #
        # where O = Optimistic Time
        #       M = Most Likely Time / Normal Time
        #       P = Pessimistic Time
        #
        # The coefficients can be changed in the 'coeffs'
        # parameter.
        #
        activity_dict[node]['Expected'] =\
            round((coeffs[0]*activity_dict[node]['Optimistic'] +\
                   coeffs[1]*activity_dict[node]['Normal'] +\
                   coeffs[2]*activity_dict[node]['Pessimistic'])\
                   / float(sum(coeffs)), 2)
        
        activity_dict[node]['Actual'] = activity_dict[node]['Expected']
        
    return activity_dict


def to_edge_list(adjacency_list, reverse=False):
    """
    This function converts a dictionary of lists adjacency
    representation into a list of edge tuples.
    """
    if not reverse:
        edge_list = [(node, nbr) for node, neighbors in adjacency_list.items() for nbr in neighbors]
    else:
        edge_list = [(nbr, node) for node, neighbors in adjacency_list.items() for nbr in neighbors]
    return edge_list


def parse_graph(activities, edges, title='Project X', begin=None, deadline=None):
    """
    Create a dependency graph from a list of activities and edges
    """
    # Create a new Directed Graph and set its
    # Title, Begin and Deadline. The Begin date
    # is today's date if it is not provided.
    G = nx.DiGraph()
    G.graph['Title'] = title
    G.graph['Begin'] = str(datetime.now().date()) if begin is None else begin
    G.graph['Deadline'] = deadline
    
    # networkx.graph.add_nodes_from() only accepts
    # list of nodes or list of tuples. By using
    # dict.items(), we give one more option to
    # add nodes using a dictionary of nodes and
    # attributes.
    if isinstance(activities, dict):
        G.add_nodes_from(activities.items())
    else:
        G.add_nodes_from(activities)
        
    # Add edges into graph using an edge list
    G.add_edges_from(edges)
    
    return G
    
    
def display_graph(G, colors=None, index_col=None):
    """
    This function displays the visual representation
    of the dependency network
    """
    # Convert G to an AGraph to visualize it
    # using GraphViz. We can also use Matplotlib
    # but I find GraphViz to be prettier.
    A = pdt.to_pydot(G)
    
    # Set the node attributes for styling
    for i, node in enumerate(A.get_nodes()):
        node.set_shape('circle')
        node.set_style('filled')
        
        if colors is not None and index_col is not None:
            # If an index column and a colors dictionary is provided,
            # we want classify the nodes based on the attribute value
            # at index column and visualize the difference
            node.set_fillcolor(colors[node.get_attributes()[index_col]]['fillcolor'])
            node.set_fontcolor(colors[node.get_attributes()[index_col]]['fontcolor'])
        else:    
            # Otherwise, just set the color to default
            node.set_fillcolor('cyan')
            node.set_fontcolor('black')
            
    # Display the created AGraph
    plt = Image(A.create_png())
    display(plt)

    
def add_finish_node(G, leaf_nodes=None):
    """
    This function adds a 'Finish' node to the dependency network
    """
    # Remove 'Finish' node from G if G already has it
    if G.has_node('Finish'):
        G.remove_node('Finish')
        
    # If the leaf nodes are not provided, search for
    # all the lowest descendants in G.
    if leaf_nodes is None:
        leaf_nodes = [node for node in G.nodes() if not list(G.successors(node))]
        
    # Create a 'Finish' node
    G.add_node('Finish')
    G.node['Finish']['Optimistic'] = 0
    G.node['Finish']['Normal'] = 0
    G.node['Finish']['Pessimistic'] = 0
    G.node['Finish']['Expected'] = 0
    G.node['Finish']['Actual'] = 0
    G.node['Finish']['Type'] = 'normal'
    G.node['Finish']['Status'] = 'To Do'
    G.node['Finish']['StartDate'] = None
    G.node['Finish']['FinishDate'] = None
    
    # Add an edge from all the leaf nodes to 'Finish' node
    for node in leaf_nodes:
        G.add_edge(node, 'Finish')
    return G


def add_start_node(G, completed=None, finish_node='Finish'):
    """
    This function adds a 'Start' node to the dependency network
    """
    # Remove 'Start' node from G if G already has it
    if G.has_node('Start'):
        G.remove_node('Start')
        
    if completed is not None:
        # If some tasks have already been completed
        # then we're going to look for the highest
        # ancestors of finish node that have not been
        # completed.
        
        # First, we're going connect all the completed
        # nodes to a node called 'Completed'. All the
        # ancestors of 'Completed' are nodes we want
        # to exclude.
        G.add_node('Completed')
        for node in completed:
            G.add_edge(node, 'Completed')
        
        # Create a subgraph which contains only nodes
        # that have not been completed
        nodes = [node for node in nx.ancestors(G, finish_node) if node not in nx.ancestors(G, 'Completed')]
        nodes.remove('Completed')
        H = nx.DiGraph(G.subgraph(nodes))
        
        # The root nodes are the nodes in this new graph
        # which is the ancestors of the finish node and
        # does not have a predecessor.
        root_nodes = [node for node in nx.ancestors(H, finish_node) if not list(H.predecessors(node))]
        
        # Don't forget to remove the 'Completed' node from
        # the original graph
        G.remove_node('Completed')
    else:
        # Otherwise, the root nodes are simply the ancestors
        # of finish node which does not have a predecessor.
        root_nodes = [node for node in nx.ancestors(G, finish_node) if not list(G.predecessors(node))]
    
    # Create a 'Start' node
    G.add_node('Start')
    G.node['Start']['Optimistic'] = 0
    G.node['Start']['Normal'] = 0
    G.node['Start']['Pessimistic'] = 0
    G.node['Start']['Expected'] = 0
    G.node['Start']['Actual'] = 0
    G.node['Start']['Type'] = 'normal'
    G.node['Start']['Status'] = 'Available'
    G.node['Start']['StartDate'] = None
    G.node['Start']['FinishDate'] = None
    
    # Add an edge from 'Start' node to all the root nodes
    for node in root_nodes:
        G.add_edge('Start', node)
    return G


def create_subgraph(G, src='Start', dest='Finish', title=None, begin=None, deadline=None):
    
    # First we're going to search for all the nodes in between
    # src and dest, including the src and dest
    nodes = [node for node in nx.ancestors(G, dest) if node in nx.descendants(G, src)]
    nodes.extend([src, dest])
    
    # Then we create a new graph from G which only contains
    # the nodes we searched previously
    H = nx.DiGraph(G.subgraph(nodes))
    
    # Set the Title, Begin and Deadline for the new graph.
    # If the values are not provided, use the values from G.
    H.graph['Title'] = G.graph['Title'] if title is None else title
    H.graph['Begin'] = G.graph['Begin'] if begin is None else begin
    H.graph['Deadline'] = G.graph['Deadline'] if deadline is None else deadline
    return H


def init_graph(H, margin=0):
    """
    Calculates all the node attributes, such as
    ES, LS, EF, LF and Slack.
    """
    # Perform a forward and backward pass
    forward_pass(H)
    backward_pass(H)
    
    # Calculates the Slack
    calculate_slack(H)
    
    # Highlight critical activities
    highlight_critical_activities(H, margin=margin)
    
    return H


def forward_pass(H):
    """
    Calculate the Early Start and the Early Finish
    of the nodes in the graph
    """
    # Convert the begin date into a date object
    begin = datetime.strptime(H.graph['Begin'], '%Y-%m-%d').date()
    
    # Traverse the nodes in Topological order
    for node in nx.topological_sort(H):
        
        parents = list(H.predecessors(node))
        if not parents:
            # If the node has no predecessors, then
            # it is the first activity. Set the
            # Early Start to zero.
            H.node[node]['ES'] = 0
        else:
            # Otherwise, the Early Start of this
            # activity is the latest Early Finish
            # of its predecessors activities.
            H.node[node]['ES'] = max(H.node[x]['EF'] for x in parents)
        
        if H.node[node]['StartDate'] is not None:
            # If the start date is set, then
            # the start time cannot be less
            # than the start date
            start = datetime.strptime(H.node[node]['StartDate'], '%Y-%m-%d').date()
            H.node[node]['ES'] = max(H.node[node]['ES'], (start - begin).days)
        
        # By default, the Early Finish of this activity is
        # simply the Early Start plus the duration.
        H.node[node]['EF'] = round(H.node[node]['ES'] + H.node[node]['Actual'], 2)
        
        if H.node[node]['FinishDate'] is not None:
            # If a finish date is set, then the Early Finish
            # cannot be greater than the finish date but also
            # cannot be less than the Early Start.
            finish = datetime.strptime(H.node[node]['FinishDate'], '%Y-%m-%d').date()
            H.node[node]['EF'] = min(H.node[node]['EF'], (finish - begin).days)
            H.node[node]['EF'] = max(H.node[node]['EF'], H.node[node]['ES'])

    return H


def backward_pass(H):
    """
    Calculate the Late Start and the Late Finish
    of the nodes in the graph
    """
    # Convert the begin date into a date object
    begin = datetime.strptime(H.graph['Begin'], '%Y-%m-%d').date()
    
    # Traverse the nodes in reverse topological order
    for node in list(reversed(list(nx.topological_sort(H)))):
        
        children = list(H.successors(node))
        if not children:
            # If the node has no successors, then
            # it is the last activity.
            if H.graph['Deadline'] is None:
                # If a deadline is not set, then
                # the Late Finish is the same as
                # the Early Finish.
                H.node[node]['LF'] = H.node[node]['EF']
            else:
                # If a deadline is set, then the Late Finish
                # is the number of days between begin and deadline
                # but it cannot be less than the Early Finish
                deadline = datetime.strptime(H.graph['Deadline'], '%Y-%m-%d').date()
                H.node[node]['LF'] = max((deadline - begin).days, H.node[node]['EF'])
        else:
            # Otherwise, the Late Finish of this
            # activity is the minimum Late Start
            # of its successors activities.
            H.node[node]['LF'] = min(H.node[x]['LS'] for x in children)
            
        if H.node[node]['FinishDate'] is not None:
            # If a finish date has been set, then
            # the Late Finish cannot be greater than
            # the finish date, but it also cannot
            # be less than the Early Finish
            finish = datetime.strptime(H.node[node]['FinishDate'], '%Y-%m-%d').date()
            H.node[node]['LF'] = min(H.node[node]['LF'], (finish - begin).days)
            H.node[node]['LF'] = max(H.node[node]['LF'], H.node[node]['EF'])
        
        # The Late Start of this activity is simply
        # the Late Finish minus the duration. It cannot
        # be less than the Early Start.
        H.node[node]['LS'] = round(H.node[node]['LF'] - H.node[node]['Actual'], 2)
        H.node[node]['LS'] = max(H.node[node]['LS'], H.node[node]['ES'])
        
    return H


def calculate_slack(H):
    """
    Calculates the Slack of the activities
    in the graph
    """
    for node in H.nodes():
        # The Slack is calculated by finding the difference
        # between Early Start and Late Start or Early Finish
        # and Late Finish
        H.node[node]['Slack'] = round(H.node[node]['LF'] - H.node[node]['EF'], 2)
    return H


def reset(G, duration=None, margin=0):
    
    for node in G.nodes():
        if duration is not None:
            G.node[node]['Actual'] = G.node[node][duration]
        G.node[node]['StartDate'] = None
        G.node[node]['FinishDate'] = None
        G.node[node]['Status'] = 'To Do'

    G.node['Start']['Status'] = 'Available'    
    return init_graph(G, margin=margin)
        
    
def highlight_critical_activities(H, src='Start', dest='Finish', margin=0):
    """
    Label the nodes as 'critical' or 'normal' based on
    whether they lie on the critical path
    """    
    critical = critical_activities(H, src=src, dest=dest, margin=margin)
    for node in H.nodes():
        if node in critical:
            H.node[node]['Type'] = 'critical'
        else:
            H.node[node]['Type'] = 'normal'
    return H
        
    
def critical_paths(H, src='Start', dest='Finish', margin=0):
    """
    A function to find all the critical paths and the duration
    of the paths in graph H from source to destination
    """
    # This is a function to calculate the total duration of a path.
    path_duration = lambda path: sum(H.node[node]['Actual'] for node in path)
    
    # All simple paths between source and destination
    paths = list(nx.all_simple_paths(H, source=src, target=dest))
    
    # Find the maximum duration in the path between src and dest.
    max_duration = max(path_duration(path) for path in paths)
    
    # A critical path is a path between src and dest which has the maximum duration
    critical_paths = [path for path in paths if (max_duration - path_duration(path)) <= margin]
    
    return critical_paths, max_duration


def critical_activities(H, src='Start', dest='Finish', margin=0):
    """
    Return a set of activities that lie on the critical path
    """   
    critical = critical_paths(H, src=src, dest=dest, margin=margin)[0]
    return set(activity for path in critical for activity in path)


def display_gantt_chart(G, begin=None, colors=None, reverse_colors=False, index_col='Type',
                        show_colorbar=True, title=None, bar_width=0.2, showgrid_x=True,
                        showgrid_y=True, group_tasks=True):
    """
    This function displays a Gantt chart representation
    of the dependency network
    """
    # If a begin date is not provided, use the begin date of G
    if begin is None:
        begin = G.graph['Begin']
        
    # Convert date string into datetime format
    begin = datetime.strptime(begin, '%Y-%m-%d').date()
    
    # If a title is not provided, use the title of G
    if title is None:
        title = G.graph['Title']
        
    # Convert graph G into a dataframe format that ff.create_gantt() can read
    df = to_dataframe(G, begin, index_col)
    
    # Create Gantt chart
    fig = ff.create_gantt(df, colors=colors, reverse_colors=reverse_colors, index_col=index_col,
                          show_colorbar=show_colorbar, title=title, bar_width=bar_width,
                          showgrid_x=showgrid_x, showgrid_y=showgrid_y, group_tasks=group_tasks)
    
    # Convert Gantt chart into a figure widget and display it
    f = go.FigureWidget(fig)
    display(f)
    
    
def to_dataframe(G, begin, index_col):
    df = []
    for node in nx.topological_sort(G):
        task = dict(Task   = node,
                    Start  = str(begin + timedelta(days=G.node[node]['ES'])),
                    Finish = str(begin + timedelta(days=G.node[node]['EF'])))
        task[index_col] = G.node[node][index_col]
        
        slack = dict(Task   = node,
                     Start  = str(begin + timedelta(days=G.node[node]['EF'])),
                     Finish = str(begin + timedelta(days=G.node[node]['LF'])))
        slack[index_col] = 'slack'
        
        df.extend([task, slack])
    return df


def PERT(G, leaf_nodes=None, completed=None, duration=None, title=None,
         begin=None, deadline=None, margin=0):
    G = add_finish_node(G, leaf_nodes=leaf_nodes)
    G = add_start_node(G, completed=completed)
    H = create_subgraph(G, title=title, begin=begin, deadline=deadline)
    H = reset(H, duration=duration, margin=margin)
    return H


def update_progress(H, date=None):
    
    if date is None:
        # If a date is not provided then it is
        # assumed to be today.
        date = str(datetime.now().date())
        
    # Convert the date string into a datetime object
    date = datetime.strptime(date, '%Y-%m-%d').date()
    
    # Convert the begin date into a date object
    begin = datetime.strptime(H.graph['Begin'], '%Y-%m-%d').date()
    
    for node in H.nodes():
        start  = begin + timedelta(days=H.node[node]['ES'])
        finish = begin + timedelta(days=H.node[node]['EF'])
        late   = begin + timedelta(days=H.node[node]['LF'])
        
        if H.node[node]['Status'] in ('To Do', 'Available'):
            if start < date:
                H.node[node]['StartDate'] = str(date)
        
        elif H.node[node]['Status'] in ('In Progress', 'Late'):
            if finish < date:
                H.node[node]['Actual'] = (date - start).days
            if H.node[node]['FinishDate'] is not None:
                finish_date = datetime.strptime(H.node[node]['FinishDate'], '%Y-%m-%d').date()
                if finish_date < date:
                    H.node[node]['FinishDate'] = str(date)
            H.node[node]['Status'] = 'Late' if late <= date else 'In Progress'
                    
        elif H.node[node]['Status'] == 'Completed':
            if finish > date:
                H.node[node]['Actual'] = max((date - start).days, 0)
                
    init_graph(H)
    return H


def change_status(H, activity, status, replace=False):
    
    if status != 'To Do':
        for node in nx.ancestors(H, activity):
            if replace:
                H.node[node]['Status'] = 'Completed'
            elif H.node[node]['Status'] != 'Completed':
                raise ValueError('\'{0}\' is not completed'.format(node))
    
    if status != 'Completed':
        for node in nx.descendants(H, activity):
            if replace:
                H.node[node]['Status'] = 'To Do'
            elif H.node[node]['Status'] not in ('To Do', 'Available'):
                raise ValueError('\'{0}\' has been started'.format(node))
    
    H.node[activity]['Status'] = status
    
    available = available_tasks(H)
    for node in H.nodes():
        if node in available:
            H.node[node]['Status'] = 'Available'
        elif H.node[node]['Status'] == 'Available':
            H.node[node]['Status'] = 'To Do'
    
    return H


def available_tasks(H):
    tasks = []
    for node in H.nodes():
        if H.node[node]['Status'] not in ('In Progress', 'Late', 'Completed'):
            predecessors = list(H.predecessors(node))
            if not predecessors or all(H.node[n]['Status'] == 'Completed' for n in predecessors):
                tasks.append(node)
    return tasks