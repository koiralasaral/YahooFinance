import matplotlib.pyplot as plt
import networkx as nx

# Erstelle einen gerichteten Graphen
G = nx.DiGraph()

# Ereignisknoten mit zusätzlichen Informationen zu den Wahrheitswerten
nodes = {
    "K": "K\n(Theseus zieht\ngegen Kreta, True)",
    "S": "S\n(Theseus zieht\nnicht gegen Skyros, False)",
    "T": "T\n(Theseus stirbt\nauf Skyros, False)",
    "V": "V\n(Skyros wird\nnicht verwüstet, False)"
}

# Knoten hinzufügen
for node in nodes:
    G.add_node(node)

# Füge Kanten hinzu und versehe sie mit Labels, die den jeweiligen Orakelspruch anzeigen.

# Ephyra: ¬K ⇒ V
# Hinweis: Diese Kante interpretiert man so, dass NICHT-K zu V führt. Da K True ist, ist die Bedingung nicht erfüllt.
G.add_edge("K", "V", label="Ephyra: ¬K⇒V", style="dotted")

# Ephyra: S ⇒ T
G.add_edge("S", "T", label="Ephyra: S⇒T", style="solid")

# Delphi: ¬T ⇒ V
# Dieser Spruch ist falsch – wir kennzeichnen die Kante daher z. B. mit einem gestrichelten Linienstil.
G.add_edge("T", "V", label="Delphi: ¬T⇒V (Falsch)", style="dashed")

# Dodona: V ⇒ (S ∧ K)
# Wir setzen hier zwei Kanten: von V zu S und von V zu K.
G.add_edge("V", "S", label="Dodona: V⇒S", style="solid")
G.add_edge("V", "K", label="Dodona: V⇒K", style="solid")

# Positioniere die Knoten
pos = nx.spring_layout(G, seed=42)

# Erstelle den Plot
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
nx.draw_networkx_labels(G, pos, labels=nodes, font_size=10)

# Zeichne die Kanten mit verschiedenen Stilarten
edge_styles = nx.get_edge_attributes(G, 'style')
for (u, v), style in edge_styles.items():
    if style == "dotted":
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], style='dotted',
                               arrowstyle="->", arrowsize=20, edge_color="gray")
    elif style == "dashed":
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], style='dashed',
                               arrowstyle="->", arrowsize=20, edge_color="red")
    else:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], style='solid',
                               arrowstyle="->", arrowsize=20, edge_color="gray")

# Kantenbeschriftungen
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=8)

plt.title("Diagramm der Orakelsprüche und logischen Zusammenhänge")
plt.axis("off")
plt.tight_layout()
plt.show()