import pandas as pd
import sys

if len(sys.argv) < 2:
	print("Usage: python3 testPandas.py <filename>")
	sys.exit(1)

filename = sys.argv[1]

node_df = pd.read_table('nodes.csv', header=0, low_memory=False)
#adj_list = {id: [] for id in node_df['id:ID']}
edge_df = pd.read_table('rels.csv', header=0)
#for i, row in edge_df.iterrows():
    #src_node = row['start:START_ID']
    #dest_node = row['end:END_ID']
    #adj_list[src_node].append(dest_node)
#adj_df = pd.DataFrame({'id:ID': adj_list.keys(), 'neighbors': adj_list.values()})
#graph_df = adj_df.merge(node_df, left_index=True, right_on='id:ID')
#graph_df = graph_df.drop(columns=['id:ID_x', 'id:ID_y'])
#graph_df = graph_df.explode('neighbors')
#graph_df.to_csv(filename, index=False)

graph2 = edge_df.merge(node_df, left_index=True, right_on='id:ID')
#print(graph2[['start:START_ID', 'end:END_ID', 'type', 'name', 'type:TYPE']])
graph2 = graph2.drop(columns=['var', 'taint_src', 'taint_dst', 'id:ID'])
graph2 = graph2.rename(columns={'start:START_ID':'source', 'end:END_ID':'dest', 'type':'source_type', 'name':'source_name', 'type:TYPE':'edge_type'})
graph2.to_csv('graph_files/' + filename, index=False)
