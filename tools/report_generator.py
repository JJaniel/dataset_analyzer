
import networkx as nx
import json

def generate_report(graph):
    """
    Analyzes the dataset graph and generates a JSON report.
    """
    report = {
        "dataset_groups": [],
        "graph_stats": {
            "total_datasets": graph.number_of_nodes(),
            "total_connections": graph.number_of_edges()
        }
    }

    # Find connected components (groups of related datasets)
    connected_components = list(nx.connected_components(graph))

    for i, component in enumerate(connected_components):
        group = {
            "group_id": i + 1,
            "datasets": list(component),
            "connections": []
        }

        # Get the subgraph for the current component
        subgraph = graph.subgraph(component)

        # Get the edges and their attributes
        for u, v, data in subgraph.edges(data=True):
            group["connections"].append({
                "source": u,
                "target": v,
                "weight": data.get('weight', 0),
                "label": data.get('label', '')
            })
        
        report["dataset_groups"].append(group)

    return json.dumps(report, indent=2)
