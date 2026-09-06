from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite
from networkx.algorithms import community as nx_community
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_results"
FIG = OUT / "figures"
SEED = 42

# Lista deliberadamente compacta y reproducible. No se descarga ningún recurso externo.
STOPWORDS_ES = {
    "a", "al", "algo", "ante", "antes", "como", "con", "contra", "cual", "cuando",
    "de", "del", "desde", "donde", "el", "ella", "en", "entre", "era", "es", "esa",
    "ese", "eso", "esta", "este", "esto", "fue", "ha", "han", "hasta", "hay", "la",
    "las", "le", "les", "lo", "los", "mas", "me", "mi", "muy", "no", "o", "para",
    "pero", "por", "porque", "que", "se", "si", "sin", "son", "su", "sus", "tambien",
    "te", "un", "una", "uno", "y", "ya", "yo", "esos", "esas", "aqui", "asi", "todo",
    "todos", "toda", "todas", "ser", "tener", "hacer", "ver", "cuando", "esto", "esta",
}


def clean_id(value: object) -> str:
    """Preserva IDs, solo elimina espacios accidentales y normaliza Unicode."""
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()


def parse_count(value: object) -> float:
    """Convierte '2,390 vistas', '1.2 K' y vacíos a conteos numéricos."""
    if pd.isna(value) or not str(value).strip():
        return np.nan
    text = str(value).strip().lower().replace("\u00a0", " ")
    match = re.search(r"([0-9][0-9.,\s]*)(?:\s*([km]))?", text)
    if not match:
        return np.nan
    number, suffix = match.groups()
    number = number.replace(" ", "")
    # Si contiene ambos separadores, el último se toma como decimal; si no, son miles.
    if "." in number and "," in number:
        decimal = max(number.rfind("."), number.rfind(","))
        number = number[:decimal].replace(".", "").replace(",", "") + "." + number[decimal + 1:]
    elif suffix and ("." in number or "," in number):
        number = number.replace(",", ".")
    else:
        number = number.replace(",", "").replace(".", "")
    try:
        multiplier = {"k": 1_000, "m": 1_000_000}.get(suffix, 1)
        return float(number) * multiplier
    except ValueError:
        return np.nan


def parse_list(value: object) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return []
    return [str(x).strip() for x in parsed if str(x).strip()] if isinstance(parsed, list) else []


def clean_text(value: object) -> str:
    """Texto para frecuencias, no para sentimiento: minúsculas, sin URL/puntuación/números."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Conserva el término de hashtags y menciones, pero remueve sus marcadores.
    text = re.sub(r"([#@])([\wáéíóúüñ]+)", r" \2 ", text)
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS_ES and len(t) > 1]
    return " ".join(tokens)


def short(text: object, size: int = 42) -> str:
    """Etiqueta corta para gráficos y tablas."""
    text = str(text)
    return text if len(text) <= size else text[: size - 1].rstrip() + "…"


def top_frame(series: pd.Series, name: str, n: int = 15) -> pd.DataFrame:
    return series.head(n).rename(name).reset_index().rename(columns={"index": "elemento"})


def save_bar(data: pd.Series, title: str, xlabel: str, filename: str, n: int = 15, color: str = "#2878b5") -> None:
    values = data.head(n).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(values) * 0.38)))
    labels = [" | ".join(map(str, item)) if isinstance(item, tuple) else str(item) for item in values.index]
    ax.barh([short(l, 55) for l in labels], values.values, color=color)
    ax.set_title(title); ax.set_xlabel(xlabel)
    fig.tight_layout(); fig.savefig(FIG / filename, dpi=170); plt.close(fig)


def quality_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({"dataset": label, "variable": col, "tipo": str(df[col].dtype),
                     "faltantes": int(df[col].isna().sum()), "porcentaje_faltante": round(df[col].isna().mean()*100, 2),
                     "valores_unicos": int(df[col].nunique(dropna=True)),
                     "constante": df[col].nunique(dropna=False) <= 1})
    return pd.DataFrame(rows)


def iqr_outliers(series: pd.Series) -> tuple[int, float, float]:
    s = series.dropna()
    if s.empty: return 0, np.nan, np.nan
    q1, q3 = s.quantile([.25, .75]); low, high = q1 - 1.5*(q3-q1), q3 + 1.5*(q3-q1)
    return int(((s < low) | (s > high)).sum()), float(low), float(high)


def projection_edges(graph: nx.Graph, columns: tuple[str, str, str]) -> pd.DataFrame:
    rows = [(u.split(":", 1)[1], v.split(":", 1)[1], int(d["weight"])) for u, v, d in graph.edges(data=True)]
    return pd.DataFrame(rows, columns=list(columns)).sort_values(columns[2], ascending=False)


def network_metrics(graph: nx.Graph, label: str, exact_connectivity: bool = True) -> dict:
    """Métricas estructurales comparables entre la red bipartita y sus proyecciones."""
    degrees = np.array([d for _, d in graph.degree()], dtype=float)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest = graph.subgraph(components[0]) if components else graph
    if largest.number_of_nodes() < 2:
        connectivity = 0
    elif exact_connectivity:
        connectivity = nx.node_connectivity(largest)
    else:
        connectivity = nx.approximation.node_connectivity(largest)
    return {
        "red": label,
        "nodos": graph.number_of_nodes(),
        "aristas": graph.number_of_edges(),
        "densidad": round(nx.density(graph), 5),
        "grado_medio": round(float(degrees.mean()), 3) if len(degrees) else 0.0,
        "grado_mediano": float(np.median(degrees)) if len(degrees) else 0.0,
        "grado_maximo": int(degrees.max()) if len(degrees) else 0,
        "nodos_aislados": int((degrees == 0).sum()),
        "componentes": len(components),
        "nodos_componente_mayor": largest.number_of_nodes(),
        "porcentaje_componente_mayor": round(largest.number_of_nodes() / graph.number_of_nodes() * 100, 2),
        "transitividad": round(nx.transitivity(graph), 4),
        "clustering_promedio": round(nx.average_clustering(graph), 4),
        "conectividad_componente_mayor": int(connectivity),
    }


def degree_frame(graph: nx.Graph, label: str, kind_of: dict | None = None) -> pd.DataFrame:
    rows = []
    for node, degree in graph.degree():
        rows.append({"red": label, "tipo_nodo": (kind_of or {}).get(node, node.split(":", 1)[0]), "grado": degree})
    counts = pd.DataFrame(rows).groupby(["red", "tipo_nodo", "grado"]).size().rename("nodos").reset_index()
    return counts


def grouped_layout(graph: nx.Graph, group_of: dict, radius: float = 5.5) -> dict:
    """Ubica cada grupo por separado y reparte los centros en un círculo.

    Con estrellas muy densas el resorte de Fruchterman-Reingold amontona todo en el centro;
    separar los grupos deja ver cuántos cúmulos hay y qué nodos los conectan.
    """
    groups: dict = {}
    for node in graph.nodes():
        groups.setdefault(group_of.get(node, "otros"), []).append(node)
    order = sorted(groups, key=lambda key: len(groups[key]), reverse=True)
    biggest = max(len(members) for members in groups.values())
    positions = {}
    for index, key in enumerate(order):
        members = groups[key]
        angle = 2 * np.pi * index / len(order)
        center = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        scale = 0.45 + 1.25 * np.sqrt(len(members) / biggest)
        subgraph = graph.subgraph(members)
        local = (nx.spring_layout(subgraph, seed=SEED, scale=scale) if len(members) > 1
                 else {members[0]: np.zeros(2)})
        for node, point in local.items():
            positions[node] = center + point
    return positions


def draw_network(graph: nx.Graph, filename: str, title: str, color_by: dict | None = None,
                 palette: dict | None = None, sizes: dict | None = None, labels: dict | None = None,
                 figsize: tuple[int, int] = (14, 10), k: float | None = None, pos: dict | None = None) -> None:
    """Dibuja la red completa: los nodos aislados también se grafican."""
    pos = pos or nx.spring_layout(graph, seed=SEED, k=k)
    fig, ax = plt.subplots(figsize=figsize)
    if graph.number_of_edges():
        weights = [d.get("weight", 1) for _, _, d in graph.edges(data=True)]
        widths = [0.35 + 0.45 * np.log1p(w) for w in weights]
        # Con miles de aristas superpuestas hace falta bajar la opacidad para ver los nodos.
        alpha = .22 if graph.number_of_edges() < 600 else .06
        nx.draw_networkx_edges(graph, pos, alpha=alpha, width=widths, ax=ax)
    groups: dict = {}
    for node in graph.nodes():
        groups.setdefault((color_by or {}).get(node, "nodo"), []).append(node)
    for group, nodes in sorted(groups.items(), key=lambda item: str(item[0])):
        nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_size=[(sizes or {}).get(n, 26) for n in nodes],
                               node_color=(palette or {}).get(group, "#2878b5"), label=str(group),
                               linewidths=0.2, edgecolors="white", ax=ax)
    if labels:
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, ax=ax)
    ax.set_title(title)
    if len(groups) > 1 and len(groups) <= 12:
        ax.legend(scatterpoints=1, fontsize=8, loc="lower left")
    ax.axis("off"); fig.tight_layout(); fig.savefig(FIG / filename, dpi=180); plt.close(fig)


def sentiment_scores(texts: list[str]) -> pd.DataFrame:
    """RoBERTuito ajustado en TASS: modelo entrenado con texto social en español."""
    from pysentimiento import create_analyzer

    analyzer = create_analyzer(task="sentiment", lang="es")
    predictions = analyzer.predict(texts)
    rows = []
    for prediction in predictions:
        probabilities = prediction.probas
        rows.append({"sentimiento": prediction.output,
                     "prob_positivo": round(probabilities["POS"], 4),
                     "prob_neutro": round(probabilities["NEU"], 4),
                     "prob_negativo": round(probabilities["NEG"], 4),
                     "puntaje": round(probabilities["POS"] - probabilities["NEG"], 4)})
    return pd.DataFrame(rows)


def sentiment_summary(frame: pd.DataFrame, keys: list[str], minimum: int = 10) -> pd.DataFrame:
    """Resume sentimiento solo donde el tamaño de muestra lo permite."""
    grouped = frame.groupby(keys, dropna=False)
    summary = grouped.agg(comentarios=("comment_id", "count"), puntaje_medio=("puntaje", "mean"),
                          positivos=("sentimiento", lambda s: (s == "POS").sum()),
                          neutros=("sentimiento", lambda s: (s == "NEU").sum()),
                          negativos=("sentimiento", lambda s: (s == "NEG").sum())).reset_index()
    for column in ["positivos", "neutros", "negativos"]:
        summary[f"pct_{column}"] = (summary[column] / summary["comentarios"] * 100).round(1)
    summary["puntaje_medio"] = summary["puntaje_medio"].round(3)
    return summary[summary.comentarios >= minimum].sort_values("comentarios", ascending=False)


def main() -> None:
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    resumen: dict = {}
    videos = pd.read_csv(ROOT / "youtube_videos.csv")
    comments = pd.read_csv(ROOT / "youtube_comments.csv")
    raw_videos, raw_comments = videos.copy(), comments.copy()

    # ------------------------------------------------------------------ 1 y 2
    # Identificadores estables: no se reemplazan por nombres/handles visibles.
    for df, columns in [(videos, ["video_id", "channel_id"]),
                        (comments, ["video_id", "comment_id", "channel_id", "author_channel_id"])]:
        for col in columns: df[col] = df[col].map(clean_id)
    for df, cols in [(videos, ["channel_name", "channel_handle", "owner_handle"]),
                     (comments, ["channel_name", "author_name", "author_handle"])]:
        for col in cols: df[col] = df[col].fillna("").astype(str).str.strip()

    videos["view_count"] = pd.to_numeric(videos["view_count"], errors="coerce")
    videos["view_count_from_text"] = videos["view_count_text"].map(parse_count)
    comments["like_count"] = comments["like_count_text"].map(parse_count)
    comments["reply_count"] = pd.to_numeric(comments["reply_count"], errors="coerce")
    comments["texto_original"] = comments["text"].fillna("").astype(str)
    comments["texto_limpio"] = comments["texto_original"].map(clean_text)
    comments["hashtags"] = comments["texto_original"].str.findall(r"(?<!\w)#([\wáéíóúüñ]+)", flags=re.I).map(lambda x: [a.lower() for a in x])
    videos["keywords_list"] = videos["keywords"].map(parse_list)
    videos["query_hits_list"] = videos["query_hits"].map(parse_list)

    # Integración: mantener cada comentario y comprobar si hay video coincidente.
    video_attrs = videos[["video_id", "title", "channel_id", "channel_name", "category", "view_count"]].rename(
        columns={"title": "video_title_catalog", "channel_id": "video_channel_id", "channel_name": "video_channel_name"})
    integrated = comments.merge(video_attrs, on="video_id", how="left", validate="many_to_one", indicator="estado_integracion")
    associated = int((integrated["estado_integracion"] == "both").sum())

    # Diagnóstico de consistencia de representaciones visibles para IDs estables.
    consistency = pd.concat([
        videos.groupby("channel_id").agg(nombres=("channel_name", "nunique"), handles=("channel_handle", "nunique")).reset_index().assign(entidad="canal"),
        comments.groupby("author_channel_id").agg(nombres=("author_name", "nunique"), handles=("author_handle", "nunique")).reset_index().rename(columns={"author_channel_id":"channel_id"}).assign(entidad="autor"),
    ], ignore_index=True)
    consistency["inconsistente"] = (consistency["nombres"] > 1) | (consistency["handles"] > 1)

    quality = pd.concat([quality_table(raw_videos, "videos"), quality_table(raw_comments, "comments")])
    quality.to_csv(OUT / "diagnostico_calidad.csv", index=False, encoding="utf-8-sig")
    consistency.to_csv(OUT / "consistencia_identificadores.csv", index=False, encoding="utf-8-sig")
    outlier_rows = []
    for dataset, col, s in [("videos", "view_count", videos.view_count), ("comments", "like_count", comments.like_count), ("comments", "reply_count", comments.reply_count)]:
        n, low, high = iqr_outliers(s); outlier_rows.append({"dataset":dataset,"variable":col,"atipicos_iqr":n,"limite_inferior":low,"limite_superior":high})
    pd.DataFrame(outlier_rows).to_csv(OUT / "valores_atipicos_iqr.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------ 3
    comments_per_video = integrated.groupby(["video_id", "video_title_catalog", "video_channel_name"], dropna=False).agg(
        comentarios=("comment_id", "count"), autores_unicos=("author_channel_id", "nunique"), me_gusta=("like_count", "sum"), respuestas=("reply_count", "sum"), visualizaciones=("view_count", "first")).sort_values("comentarios", ascending=False)
    comments_per_channel = integrated.groupby(["video_channel_id", "video_channel_name"], dropna=False).agg(
        comentarios=("comment_id", "count"), autores_unicos=("author_channel_id", "nunique"), videos=("video_id", "nunique")).sort_values("comentarios", ascending=False)
    videos_per_channel = videos.groupby(["channel_id", "channel_name"], dropna=False).size().sort_values(ascending=False).rename("videos")
    word_counts = Counter(w for text in comments.texto_limpio for w in text.split())
    bigram_counts = Counter(" ".join(pair) for text in comments.texto_limpio for pair in zip(text.split(), text.split()[1:]))
    hashtags = Counter(h for tags in comments.hashtags for h in tags)
    queries = Counter(q for qs in videos.query_hits_list for q in qs)
    categories = videos.category.value_counts(dropna=False)
    words_series = pd.Series(word_counts).sort_values(ascending=False)
    bigrams_series = pd.Series(bigram_counts).sort_values(ascending=False)

    descriptive = pd.concat([
        videos.view_count.describe().rename("visualizaciones_video"),
        comments.like_count.describe().rename("me_gusta_comentario"),
        comments.reply_count.describe().rename("respuestas_comentario"),
        comments_per_video.comentarios.describe().rename("comentarios_por_video"),
        comments_per_video.autores_unicos.describe().rename("autores_por_video"),
        videos_per_channel.describe().rename("videos_por_canal"),
    ], axis=1).round(2)
    descriptive.to_csv(OUT / "estadisticas_descriptivas.csv", encoding="utf-8-sig")

    for filename, frame in {"comentarios_por_video.csv":comments_per_video.reset_index(), "comentarios_por_canal.csv":comments_per_channel.reset_index(), "videos_por_canal.csv":videos_per_channel.reset_index(), "palabras_frecuentes.csv":top_frame(words_series, "frecuencia", 100), "bigramas_frecuentes.csv":top_frame(bigrams_series, "frecuencia", 100), "hashtags_frecuentes.csv":top_frame(pd.Series(hashtags).sort_values(ascending=False), "frecuencia", 100), "consultas_frecuentes.csv":top_frame(pd.Series(queries).sort_values(ascending=False), "frecuencia", 100), "categorias.csv":categories.rename("videos").reset_index().rename(columns={"category":"categoria", "index":"categoria"})}.items():
        frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    save_bar(comments_per_video["comentarios"], "Videos con mayor participación", "Comentarios", "comentarios_por_video.png")
    save_bar(comments_per_channel["comentarios"], "Canales con mayor participación", "Comentarios", "comentarios_por_canal.png")
    save_bar(categories, "Videos por categoría", "Videos", "categorias.png")
    save_bar(words_series, "Palabras frecuentes en comentarios", "Frecuencia", "palabras_frecuentes.png")
    save_bar(bigrams_series, "Bigramas frecuentes en comentarios", "Frecuencia", "bigramas_frecuentes.png", color="#7570b3")
    fig, ax = plt.subplots(figsize=(8, 5)); plot = comments_per_video.dropna(subset=["visualizaciones"])
    ax.scatter(plot.visualizaciones, plot.comentarios, alpha=.7, color="#d95f02"); ax.set_xscale("symlog"); ax.set_xlabel("Visualizaciones (escala symlog)"); ax.set_ylabel("Comentarios"); ax.set_title("Popularidad observada y participación"); fig.tight_layout(); fig.savefig(FIG / "visualizaciones_vs_comentarios.png", dpi=170); plt.close(fig)
    cloud = WordCloud(width=1400, height=700, background_color="white", colormap="viridis", random_state=SEED)
    cloud.generate_from_frequencies(word_counts)
    cloud.to_file(str(FIG / "nube_palabras.png"))

    # ------------------------------------------------------------------ 4
    # Arista = ocurrencia observada de un autor comentando en un video; peso = número de comentarios.
    edge_table = comments.groupby(["author_channel_id", "video_id"], as_index=False).agg(
        peso_comentarios=("comment_id", "count"), primer_autor_nombre=("author_name", "first"), primer_titulo_video=("video_title", "first"))
    author_nodes = comments.groupby("author_channel_id", as_index=False).agg(nombre_visible=("author_name", "first"), handle=("author_handle", "first"), comentarios=("comment_id", "count"), videos_distintos=("video_id", "nunique")).rename(columns={"author_channel_id":"node_id"}).assign(tipo_nodo="autor")
    video_nodes = videos.groupby("video_id", as_index=False).agg(titulo=("title", "first"), canal_id=("channel_id", "first"), canal=("channel_name", "first"), visualizaciones=("view_count", "first"), categoria=("category", "first")).rename(columns={"video_id":"node_id"}).assign(tipo_nodo="video")
    nodes = pd.concat([author_nodes, video_nodes], ignore_index=True, sort=False)
    nodes.to_csv(OUT / "red_bipartita_nodos.csv", index=False, encoding="utf-8-sig")
    edge_table.rename(columns={"author_channel_id":"source_autor", "video_id":"target_video"}).to_csv(OUT / "red_bipartita_aristas.csv", index=False, encoding="utf-8-sig")
    graph = nx.Graph()
    graph.add_nodes_from(("autor:"+row.node_id, {"bipartite":"autor"}) for row in author_nodes.itertuples())
    graph.add_nodes_from(("video:"+row.node_id, {"bipartite":"video"}) for row in video_nodes.itertuples())
    graph.add_weighted_edges_from(("autor:"+r.author_channel_id, "video:"+r.video_id, r.peso_comentarios) for r in edge_table.itertuples())
    nx.write_gexf(graph, OUT / "red_bipartita.gexf")
    draw_network(graph, "red_bipartita_completa.png", "Red bipartita completa autor–video",
                 color_by={n: ("Autores" if d["bipartite"] == "autor" else "Videos") for n, d in graph.nodes(data=True)},
                 palette={"Autores": "#1b9e77", "Videos": "#d95f02"},
                 sizes={n: (22 if d["bipartite"] == "autor" else 46) for n, d in graph.nodes(data=True)}, k=1.1)

    # ------------------------------------------------------------------ 5
    author_ids = [n for n, d in graph.nodes(data=True) if d["bipartite"] == "autor"]
    video_ids = [n for n, d in graph.nodes(data=True) if d["bipartite"] == "video"]
    proj_authors = bipartite.weighted_projected_graph(graph, author_ids)
    proj_videos = bipartite.weighted_projected_graph(graph, video_ids)
    projection_edges(proj_authors, ("autor_a", "autor_b", "videos_compartidos")).to_csv(OUT / "proyeccion_autor_autor_aristas.csv", index=False, encoding="utf-8-sig")
    projection_edges(proj_videos, ("video_a", "video_b", "autores_compartidos")).to_csv(OUT / "proyeccion_video_video_aristas.csv", index=False, encoding="utf-8-sig")

    multi_video = {"autor:" + r.node_id for r in author_nodes.itertuples() if r.videos_distintos > 1}
    # Cada autor se agrupa por el video donde más comentó, que es el cúmulo al que pertenece.
    main_video = comments.groupby("author_channel_id").video_id.agg(lambda s: s.value_counts().index[0])
    draw_network(proj_authors, "proyeccion_autor_autor.png", "Proyección autor–autor (peso = videos compartidos)",
                 color_by={n: ("Comentó en más de un video" if n in multi_video else "Comentó en un solo video") for n in proj_authors},
                 palette={"Comentó en más de un video": "#d95f02", "Comentó en un solo video": "#1b9e77"},
                 sizes={n: (90 if n in multi_video else 24) for n in proj_authors},
                 pos=grouped_layout(proj_authors, {"autor:" + a: v for a, v in main_video.items()}))
    connected_videos = {n for n in proj_videos if proj_videos.degree(n) > 0}
    video_title = {"video:" + r.node_id: r.titulo for r in video_nodes.itertuples()}
    draw_network(proj_videos, "proyeccion_video_video.png", "Proyección video–video (peso = autores compartidos)",
                 color_by={n: ("Comparte autores" if n in connected_videos else "Sin autores compartidos") for n in proj_videos},
                 palette={"Comparte autores": "#d95f02", "Sin autores compartidos": "#bdbdbd"},
                 sizes={n: (110 if n in connected_videos else 18) for n in proj_videos})
    linked_videos = proj_videos.subgraph(connected_videos)
    draw_network(linked_videos, "proyeccion_video_video_conectada.png",
                 "Videos que comparten al menos un autor", figsize=(11, 8), k=1.4,
                 color_by={n: "Video" for n in linked_videos}, palette={"Video": "#d95f02"},
                 sizes={n: 260 for n in linked_videos},
                 labels={n: short(video_title.get(n, n), 34) for n in linked_videos})

    # ------------------------------------------------------------------ 6
    node_kind = {n: d["bipartite"] for n, d in graph.nodes(data=True)}
    topology = pd.DataFrame([
        network_metrics(graph, "bipartita autor-video"),
        network_metrics(proj_authors, "proyección autor-autor", exact_connectivity=False),
        network_metrics(proj_videos, "proyección video-video", exact_connectivity=False),
    ])
    topology.to_csv(OUT / "metricas_topologia.csv", index=False, encoding="utf-8-sig")
    degrees = pd.concat([degree_frame(graph, "bipartita autor-video", node_kind),
                         degree_frame(proj_authors, "proyección autor-autor", {n: "autor" for n in proj_authors}),
                         degree_frame(proj_videos, "proyección video-video", {n: "video" for n in proj_videos})])
    degrees.to_csv(OUT / "distribucion_grados.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (label, values, color) in zip(axes, [
            ("Autores (red bipartita)", [graph.degree(n) for n in author_ids], "#1b9e77"),
            ("Videos (red bipartita)", [graph.degree(n) for n in video_ids], "#d95f02"),
            ("Autores (proyección autor–autor)", [proj_authors.degree(n) for n in proj_authors], "#7570b3")]):
        counts = pd.Series(values).value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=color)
        ax.set_yscale("log"); ax.set_title(label, fontsize=14); ax.tick_params(labelsize=12)
        ax.set_xlabel("Grado", fontsize=13); ax.set_ylabel("Nodos (escala log)", fontsize=13)
    fig.suptitle("Distribución de grados", fontsize=16); fig.tight_layout(); fig.savefig(FIG / "distribucion_grados.png", dpi=170); plt.close(fig)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    isolated_videos = [n for n in video_ids if graph.degree(n) == 0]
    peripheral = pd.DataFrame([
        {"grupo": "videos sin comentarios recolectados", "nodos": len(isolated_videos)},
        {"grupo": "autores con un solo video comentado", "nodos": int((author_nodes.videos_distintos == 1).sum())},
        {"grupo": "componentes con menos de 5 nodos", "nodos": sum(1 for c in components if len(c) < 5)},
        {"grupo": "nodos fuera de la componente mayor", "nodos": graph.number_of_nodes() - len(components[0])},
    ])
    peripheral.to_csv(OUT / "nodos_perifericos.csv", index=False, encoding="utf-8-sig")
    bipartite_clustering = float(np.mean(list(bipartite.clustering(graph.subgraph(components[0])).values())))

    # ------------------------------------------------------------------ 7
    # Se excluyen los videos sin comentarios: son nodos sin información relacional.
    observed = graph.subgraph([n for n, d in graph.degree() if d > 0]).copy()
    partition = sorted(nx_community.louvain_communities(observed, weight="weight", seed=SEED), key=len, reverse=True)
    modularity = nx_community.modularity(observed, partition, weight="weight")
    community_of = {node: index for index, group in enumerate(partition, start=1) for node in group}
    community_frame = pd.DataFrame([
        {"node_id": node.split(":", 1)[1], "tipo_nodo": node_kind[node], "comunidad": community_of[node],
         "grado": observed.degree(node),
         "etiqueta": video_title.get(node, "") if node_kind[node] == "video" else ""}
        for node in observed.nodes()])
    community_frame.to_csv(OUT / "comunidades_nodos.csv", index=False, encoding="utf-8-sig")

    integrated["comunidad"] = integrated.video_id.map(lambda v: community_of.get("video:" + v))
    comments["comunidad"] = integrated["comunidad"]
    community_rows = []
    for cid, group in integrated.dropna(subset=["comunidad"]).groupby("comunidad"):
        members = partition[int(cid) - 1]
        palabras = Counter(w for text in group.texto_limpio for w in text.split())
        community_rows.append({
            "comunidad": int(cid),
            "nodos": len(members),
            "autores": sum(1 for n in members if node_kind[n] == "autor"),
            "videos": sum(1 for n in members if node_kind[n] == "video"),
            "comentarios": len(group),
            "canales": " | ".join(sorted(set(group.video_channel_name))),
            "video_principal": short(group.video_title_catalog.mode().iat[0], 60),
            "me_gusta": int(group.like_count.fillna(0).sum()),
            "palabras_frecuentes": ", ".join(w for w, _ in palabras.most_common(8)),
        })
    community_summary = pd.DataFrame(community_rows).sort_values("comentarios", ascending=False)

    colors = plt.get_cmap("tab10").colors
    principal = list(community_summary.comunidad.head(len(colors)))
    palette = {f"Comunidad {cid}": "#%02x%02x%02x" % tuple(int(255*c) for c in colors[i]) for i, cid in enumerate(principal)}
    palette["Otras comunidades"] = "#c7c7c7"
    draw_network(observed, "comunidades.png", f"Comunidades detectadas con Louvain (modularidad = {modularity:.3f})",
                 color_by={n: (f"Comunidad {community_of[n]}" if community_of[n] in principal else "Otras comunidades") for n in observed},
                 palette=palette,
                 sizes={n: (90 if node_kind[n] == "video" else 22) for n in observed},
                 pos=grouped_layout(observed, community_of))

    # ------------------------------------------------------------------ 8
    degree_centrality = nx.degree_centrality(observed)
    betweenness = nx.betweenness_centrality(observed, seed=SEED)
    closeness = nx.closeness_centrality(observed)
    pagerank = nx.pagerank(observed, weight="weight")
    try:
        eigenvector = nx.eigenvector_centrality(observed, max_iter=1000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: np.nan for n in observed}
    articulation = set(nx.articulation_points(observed))
    centrality = pd.DataFrame({
        "node_id": [n.split(":", 1)[1] for n in observed],
        "tipo_nodo": [node_kind[n] for n in observed],
        "etiqueta": [video_title.get(n, "") if node_kind[n] == "video" else "" for n in observed],
        "grado": [observed.degree(n) for n in observed],
        "comunidad": [community_of[n] for n in observed],
        "centralidad_grado": [round(degree_centrality[n], 5) for n in observed],
        "intermediacion": [round(betweenness[n], 5) for n in observed],
        "cercania": [round(closeness[n], 5) for n in observed],
        "pagerank": [round(pagerank[n], 5) for n in observed],
        "vector_propio": [round(eigenvector[n], 5) for n in observed],
        "punto_articulacion": [n in articulation for n in observed],
    })
    authors_centrality = centrality[centrality.tipo_nodo == "autor"].sort_values("intermediacion", ascending=False)
    videos_centrality = centrality[centrality.tipo_nodo == "video"].sort_values("intermediacion", ascending=False)
    authors_centrality.merge(author_nodes[["node_id", "nombre_visible", "comentarios", "videos_distintos"]], on="node_id", how="left").to_csv(OUT / "centralidad_autores.csv", index=False, encoding="utf-8-sig")
    videos_centrality.merge(video_nodes[["node_id", "canal", "visualizaciones", "categoria"]], on="node_id", how="left").to_csv(OUT / "centralidad_videos.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"node_id": [n.split(":", 1)[1] for n in articulation],
                  "tipo_nodo": [node_kind[n] for n in articulation],
                  "etiqueta": [video_title.get(n, "") for n in articulation]}).to_csv(OUT / "puntos_articulacion.csv", index=False, encoding="utf-8-sig")

    save_bar(videos_centrality.set_index("etiqueta").intermediacion, "Videos con mayor intermediación", "Intermediación", "centralidad_videos.png", n=10, color="#d95f02")
    bridges = authors_centrality[authors_centrality.intermediacion > 0]
    if len(bridges):
        save_bar(bridges.set_index("node_id").intermediacion, "Autores puente (intermediación > 0)", "Intermediación", "centralidad_autores.png", n=10, color="#1b9e77")

    # ------------------------------------------------------------------ 9
    sentiment = sentiment_scores(integrated.texto_original.tolist())
    integrated = pd.concat([integrated.reset_index(drop=True), sentiment], axis=1)
    integrated[["comment_id", "video_id", "author_channel_id", "video_channel_name", "comunidad",
                "sentimiento", "prob_positivo", "prob_neutro", "prob_negativo", "puntaje",
                "texto_original"]].to_csv(OUT / "sentimiento_comentarios.csv", index=False, encoding="utf-8-sig")
    sentiment_video = sentiment_summary(integrated, ["video_id", "video_title_catalog", "video_channel_name"])
    sentiment_channel = sentiment_summary(integrated, ["video_channel_id", "video_channel_name"])
    sentiment_community = sentiment_summary(integrated.dropna(subset=["comunidad"]), ["comunidad"])
    sentiment_video.to_csv(OUT / "sentimiento_por_video.csv", index=False, encoding="utf-8-sig")
    sentiment_channel.to_csv(OUT / "sentimiento_por_canal.csv", index=False, encoding="utf-8-sig")
    sentiment_community.to_csv(OUT / "sentimiento_por_comunidad.csv", index=False, encoding="utf-8-sig")

    distribution = integrated.sentimiento.value_counts().reindex(["POS", "NEU", "NEG"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Positivo", "Neutro", "Negativo"], distribution.values, color=["#1b9e77", "#999999", "#d95f02"])
    for x, value in enumerate(distribution.values):
        ax.text(x, value + 3, f"{value:.0f} ({value/len(integrated)*100:.1f} %)", ha="center", fontsize=9)
    ax.set_ylabel("Comentarios"); ax.set_title("Distribución de sentimiento en los comentarios")
    fig.tight_layout(); fig.savefig(FIG / "sentimiento_distribucion.png", dpi=170); plt.close(fig)

    def stacked_sentiment(frame: pd.DataFrame, labels: list[str], title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(10, max(3.5, len(frame) * 0.55)))
        base = np.zeros(len(frame))
        for column, color, name in [("pct_positivos", "#1b9e77", "Positivo"), ("pct_neutros", "#999999", "Neutro"), ("pct_negativos", "#d95f02", "Negativo")]:
            ax.barh(labels, frame[column].values, left=base, color=color, label=name)
            base = base + frame[column].values
        ax.set_xlabel("Porcentaje de comentarios"); ax.set_title(title); ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(FIG / filename, dpi=170); plt.close(fig)

    if len(sentiment_video):
        stacked_sentiment(sentiment_video, [short(t, 40) for t in sentiment_video.video_title_catalog],
                          "Sentimiento por video (10 o más comentarios)", "sentimiento_por_video.png")
    if len(sentiment_community):
        stacked_sentiment(sentiment_community, [f"Comunidad {int(c)}" for c in sentiment_community.comunidad],
                          "Sentimiento por comunidad (10 o más comentarios)", "sentimiento_por_comunidad.png")
    community_summary = community_summary.merge(
        sentiment_community[["comunidad", "puntaje_medio", "pct_positivos", "pct_neutros", "pct_negativos"]],
        on="comunidad", how="left")
    community_summary.to_csv(OUT / "comunidades_resumen.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------- Resumen general
    active_videos = int(np.ceil(len(comments_per_video) * .10)); active_channels = int(np.ceil(len(comments_per_channel) * .10))
    author_video_degree = edge_table.groupby("author_channel_id").video_id.nunique().sort_values(ascending=False)
    author_channel_participation = integrated[["author_channel_id", "video_channel_id", "video_channel_name"]].drop_duplicates()
    cross_channel_authors = author_channel_participation.groupby("author_channel_id").agg(
        canales=("video_channel_id", "nunique"), nombres_canales=("video_channel_name", lambda x: " | ".join(sorted(set(x))))
    ).query("canales > 1").sort_values("canales", ascending=False)
    cross_channel_authors.to_csv(OUT / "autores_multicanal.csv", encoding="utf-8-sig")

    resumen.update({
        "videos": len(videos), "comentarios": len(comments), "canales": int(videos.channel_id.nunique()),
        "autores": int(comments.author_channel_id.nunique()), "comentarios_asociados": associated,
        "videos_con_comentarios": int(comments.video_id.nunique()),
        "dimensiones_videos": list(raw_videos.shape), "dimensiones_comentarios": list(raw_comments.shape),
        "video_id_duplicados": int(videos.duplicated("video_id").sum()),
        "comment_id_duplicados": int(comments.duplicated("comment_id").sum()),
        "textos_modificados": int((comments.texto_original != comments.texto_limpio).sum()),
        "textos_vacios_tras_limpieza": int(comments.texto_limpio.eq("").sum()),
        "duplicados_texto_original": int(comments.texto_original.duplicated().sum()),
        "duplicados_texto_limpio": int(comments.texto_limpio.duplicated().sum()),
        "ids_inconsistentes": int(consistency.inconsistente.sum()),
        "concentracion_top10_videos": round(comments_per_video.comentarios.head(active_videos).sum()/len(comments)*100, 1),
        "concentracion_top10_canales": round(comments_per_channel.comentarios.head(active_channels).sum()/len(comments)*100, 1),
        "spearman_vistas_comentarios": round(float(comments_per_video[["visualizaciones", "comentarios"]].corr(method="spearman").iloc[0, 1]), 3),
        "autores_multivideo": int((author_video_degree > 1).sum()), "autores_multicanal": len(cross_channel_authors),
        "aristas_bipartita": graph.number_of_edges(),
        "aristas_autor_autor": proj_authors.number_of_edges(), "aristas_video_video": proj_videos.number_of_edges(),
        "clustering_bipartito_componente_mayor": round(bipartite_clustering, 4),
        "comunidades": len(partition), "modularidad": round(modularity, 4),
        "tamanos_comunidades": [len(c) for c in partition],
        "puntos_articulacion": len(articulation),
        "videos_articuladores": int(sum(1 for n in articulation if node_kind[n] == "video")),
        "autores_articuladores": int(sum(1 for n in articulation if node_kind[n] == "autor")),
        "sentimiento": {k: int(v) for k, v in distribution.items()},
        "puntaje_sentimiento_medio": round(float(integrated.puntaje.mean()), 3),
        "topologia": topology.to_dict(orient="records"),
        "top_videos": comments_per_video.head(6).reset_index().to_dict(orient="records"),
        "top_canales": comments_per_channel.head(6).reset_index().to_dict(orient="records"),
        "comunidades_principales": community_summary.head(6).to_dict(orient="records"),
        "sentimiento_por_canal": sentiment_channel.to_dict(orient="records"),
        "sentimiento_por_video": sentiment_video.to_dict(orient="records"),
        "top_palabras": words_series.head(15).to_dict(), "top_bigramas": bigrams_series.head(10).to_dict(),
        "categorias": categories.head(6).to_dict(),
        "autores_puente": authors_centrality.head(10).to_dict(orient="records"),
        "videos_centrales": videos_centrality.head(10).to_dict(orient="records"),
        "perifericos": peripheral.to_dict(orient="records"),
    })
    (OUT / "resumen_metricas.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    integrated.to_csv(OUT / "datos_integrados_limpios.csv", index=False, encoding="utf-8-sig")
    (OUT / "requirements.txt").write_text("pandas\nnumpy\nmatplotlib\nnetworkx\nwordcloud\npysentimiento\n", encoding="utf-8")
    print(f"Analisis terminado. Resultados: {OUT}")
    print(f"Red bipartita: {graph.number_of_nodes()} nodos y {graph.number_of_edges()} aristas; "
          f"{len(partition)} comunidades (modularidad {modularity:.3f}).")


if __name__ == "__main__":
    main()
