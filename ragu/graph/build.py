import networkx as nx
from cdlib import algorithms


# 3. Element Instances → Element Summaries
def summarize_elements(elements, client, config):
    summaries = []
    for index, element in enumerate(elements):
        print(f"Element index {index} of {len(elements)}:")
        response = client.chat.completions.create(
            model=config.llm.model,
            messages=[
                {"role": "system", "content": config.graph.summarize_elements.system_prompt},
                {"role": "user", "content": element}
            ]
        )
        print("Element summary:", response.choices[0].message.content)
        summary = response.choices[0].message.content
        summaries.append(summary)
    return summaries


# 4. Element Summaries → Graph Communities
def build_graph_from_summaries(summaries):
    G = nx.Graph()
    for index, summary in enumerate(summaries):
        print(f"Summary index {index} of {len(summaries)}:")
        lines = summary.split("\n")
        entities_section = False
        relationships_section = False
        entities = []
        for line in lines:
            if line.startswith("### Entities:") or line.startswith("**Entities:**"):
                entities_section = True
                relationships_section = False
                continue
            elif line.startswith("### Relationships:") or line.startswith("**Relationships:**"):
                entities_section = False
                relationships_section = True
                continue
            if entities_section and line.strip():
                if line[0].isdigit() and line[1] == ".":
                    line = line.split(".", 1)[1].strip()
                entity = line.strip()
                entity = entity.replace("**", "")
                entities.append(entity)
                G.add_node(entity)
            elif relationships_section and line.strip():
                parts = line.split("->")
                if len(parts) >= 2:
                    source = parts[0].strip()
                    target = parts[-1].strip()
                    relation = " -> ".join(parts[1:-1]).strip()
                    G.add_edge(source, target, label=relation)
    return G


# 5. Graph Communities → Community Summaries
def detect_communities(graph):
    communities = []
    index = 0
    for component in nx.connected_components(graph):
        print(
            f"Component index {index} of {len(list(nx.connected_components(graph)))}:")
        subgraph = graph.subgraph(component)
        if len(subgraph.nodes) > 1:  # Leiden algorithm requires at least 2 nodes
            try:
                sub_communities = algorithms.leiden(subgraph)
                for community in sub_communities.communities:
                    communities.append(list(community))
            except Exception as e:
                print(f"Error processing community {index}: {e}")
        else:
            communities.append(list(subgraph.nodes))
        index += 1
    print("Communities from detect_communities:", communities)
    return communities


def summarize_communities(communities, graph, client, config):
    community_summaries = []
    for index, community in enumerate(communities):
        print(f"Summarize Community index {index} of {len(communities)}:")
        subgraph = graph.subgraph(community)
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges(data=True))
        description = "Entities: " + ", ".join(nodes) + "\nRelationships: "
        relationships = []
        for edge in edges:
            relationships.append(
                f"{edge[0]} -> {edge[2]['label']} -> {edge[1]}")
        description += ", ".join(relationships)

        response = client.chat.completions.create(
            model=config.llm.model,
            messages=[
                {"role": "system",
                    "content": config.graph.summarize_communities.system_prompt},
                {"role": "user", "content": description}
            ]
        )
        summary = response.choices[0].message.content.strip()
        community_summaries.append(summary)
    return community_summaries


class GraphBuilder():
    def __init__(self, client, config):
        self.client = client
        self.config = config

    def __call__(self, elements, *args, **kwds):
        elements_summaries = summarize_elements(elements)
        G = build_graph_from_summaries(elements_summaries)
        communities = detect_communities(G)
        communities_summaries = summarize_communities(communities)
        
        return G, communities_summaries
