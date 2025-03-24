# Based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/

from collections import Counter


async def _find_most_related_edges_from_entities(node_datas, knowledge_graph):
    all_related_edges = []
    for node_d in node_datas:
        if knowledge_graph.graph.has_node(node_d["entity_name"]):
            all_related_edges.extend(list(knowledge_graph.graph.edges(node_d["entity_name"])))
    all_edges = []
    seen = set()
    for this_edges in all_related_edges:
        sorted_edge = tuple(sorted(this_edges))
        if sorted_edge not in seen:
            seen.add(sorted_edge)
            all_edges.append(sorted_edge)

    all_edges_pack = [knowledge_graph.get_edge(e[0], e[1]) for e in all_edges]
    all_edges_degree = [knowledge_graph.edge_degree(e[0], e[1]) for e in all_edges]

    all_edges_data = [
        {"source": k[0], "target": k[1], "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: x["rank"], reverse=True
    )

    return all_edges_data

async def _find_most_related_text_unit_from_entities(node_datas, chunks_db, knowledge_graph):
    text_units_id = []
    for node_d in node_datas:
        text_units_id.append(node_d["source_chunk_id"])
    edges = []
    for node_d in node_datas:
        if knowledge_graph.graph.has_node(node_d["entity_name"]):
            edges.extend(list(knowledge_graph.graph.edges(node_d["entity_name"])))

    all_one_hop_nodes = set()
    for edge in edges:
        if edge:
            all_one_hop_nodes.update([edge[1]])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = [knowledge_graph.graph.nodes.get(e) for e in all_one_hop_nodes]
    all_one_hop_text_units_lookup = {
        k: set(v["source_chunk_id"])
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units_id, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    chunks = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_units = [t["data"] for t in chunks]
    return all_text_units


async def _find_most_related_community_from_entities(node_datas: list, community_report_bd, level: int = 0):
        related_communities = []
        for node_d in node_datas:
            if "cluster_id" not in node_d:
                continue
            related_communities.append({"cluster": node_d["cluster_id"], "level": node_d["level"]})

        related_community_dup_keys = [
            str(dp["cluster"])
            for dp in related_communities
            if dp["level"] <= level
        ]
        related_community_keys_counts = dict(Counter(related_community_dup_keys))
        _related_community_datas = [await community_report_bd.get_by_id(k) for k in related_community_keys_counts.keys()]
        related_community_datas = {
            k: v
            for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
            if v is not None
        }
        related_community_keys = sorted(
            related_community_keys_counts.keys(),
            key=lambda k: (
                related_community_keys_counts[k],
                related_community_datas[k]["report_json"].get("rating", -1),
            ),
            reverse=True,
        )
        sorted_community_datas = [
            related_community_datas[k] for k in related_community_keys
        ]

        return sorted_community_datas