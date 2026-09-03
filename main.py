from __future__ import annotations

import ast
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_results"
FIG = OUT / "figures"

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
    text = str(value).strip().lower().replace(" ", " ")
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


def top_frame(series: pd.Series, name: str, n: int = 15) -> pd.DataFrame:
    return series.head(n).rename(name).reset_index().rename(columns={"index": "elemento"})


def save_bar(data: pd.Series, title: str, xlabel: str, filename: str, n: int = 15) -> None:
    values = data.head(n).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(values) * 0.38)))
    labels = [" | ".join(map(str, item)) if isinstance(item, tuple) else str(item) for item in values.index]
    ax.barh(labels, values.values, color="#2878b5")
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


def main() -> None:
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    videos = pd.read_csv(ROOT / "youtube_videos.csv")
    comments = pd.read_csv(ROOT / "youtube_comments.csv")
    raw_videos, raw_comments = videos.copy(), comments.copy()

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

    # EDA de participación y contenido.
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
    for filename, frame in {"comentarios_por_video.csv":comments_per_video.reset_index(), "comentarios_por_canal.csv":comments_per_channel.reset_index(), "videos_por_canal.csv":videos_per_channel.reset_index(), "palabras_frecuentes.csv":top_frame(pd.Series(word_counts).sort_values(ascending=False), "frecuencia", 100), "bigramas_frecuentes.csv":top_frame(pd.Series(bigram_counts).sort_values(ascending=False), "frecuencia", 100), "hashtags_frecuentes.csv":top_frame(pd.Series(hashtags).sort_values(ascending=False), "frecuencia", 100), "consultas_frecuentes.csv":top_frame(pd.Series(queries).sort_values(ascending=False), "frecuencia", 100), "categorias.csv":categories.rename("videos").reset_index().rename(columns={"category":"categoria", "index":"categoria"})}.items():
        frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    save_bar(comments_per_video["comentarios"], "Videos con mayor participación", "Comentarios", "comentarios_por_video.png")
    save_bar(comments_per_channel["comentarios"], "Canales con mayor participación", "Comentarios", "comentarios_por_canal.png")
    save_bar(categories, "Videos por categoría", "Videos", "categorias.png")
    save_bar(pd.Series(word_counts).sort_values(ascending=False), "Palabras frecuentes en comentarios", "Frecuencia", "palabras_frecuentes.png")
    fig, ax = plt.subplots(figsize=(8, 5)); plot = comments_per_video.dropna(subset=["visualizaciones"])
    ax.scatter(plot.visualizaciones, plot.comentarios, alpha=.7, color="#d95f02"); ax.set_xscale("symlog"); ax.set_xlabel("Visualizaciones (escala symlog)"); ax.set_ylabel("Comentarios"); ax.set_title("Popularidad observada y participación"); fig.tight_layout(); fig.savefig(FIG / "visualizaciones_vs_comentarios.png", dpi=170); plt.close(fig)

    # Red bipartita: arista = ocurrencia observada de autor comentando en video; peso = número de comentarios.
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
    pos = nx.spring_layout(graph, seed=42, k=1.1)
    fig, ax = plt.subplots(figsize=(14, 10)); nx.draw_networkx_edges(graph, pos, alpha=.23, width=[.4 + .45*d["weight"] for _,_,d in graph.edges(data=True)], ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=[n for n,d in graph.nodes(data=True) if d["bipartite"]=="autor"], node_size=22, node_color="#1b9e77", label="Autores", ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=[n for n,d in graph.nodes(data=True) if d["bipartite"]=="video"], node_size=45, node_color="#d95f02", node_shape="s", label="Videos", ax=ax)
    ax.set_title("Red bipartita completa autor–video"); ax.legend(); ax.axis("off"); fig.tight_layout(); fig.savefig(FIG / "red_bipartita_completa.png", dpi=180); plt.close(fig)

    # Métricas requeridas para interpretar 3.5 sin afirmar relaciones inexistentes.
    active_videos = int(np.ceil(len(comments_per_video) * .10)); active_channels = int(np.ceil(len(comments_per_channel) * .10))
    author_video_degree = edge_table.groupby("author_channel_id").video_id.nunique().sort_values(ascending=False)
    shared_authors = int((author_video_degree > 1).sum())
    author_channel_participation = integrated[["author_channel_id", "video_channel_id", "video_channel_name"]].drop_duplicates()
    cross_channel_authors = author_channel_participation.groupby("author_channel_id").agg(
        canales=("video_channel_id", "nunique"), nombres_canales=("video_channel_name", lambda x: " | ".join(sorted(set(x))))
    ).query("canales > 1").sort_values("canales", ascending=False)
    cross_channel_count = len(cross_channel_authors)
    cross_channel_examples = "; ".join(
        f"`{idx}` ({row.nombres_canales})" for idx, row in cross_channel_authors.head(4).iterrows()) or "ninguno"
    corr = comments_per_video[["visualizaciones", "comentarios"]].corr(method="spearman").iloc[0, 1]
    text_before_duplicates = int(comments.texto_original.duplicated().sum())
    text_after_duplicates = int(comments.texto_limpio.duplicated().sum())
    text_modified = int((comments.texto_original != comments.texto_limpio).sum())
    empty_clean = int(comments.texto_limpio.eq("").sum())
    top_video_name = comments_per_video.index[0][1] if len(comments_per_video) else "N/A"
    top_channel_name = comments_per_channel.index[0][1] if len(comments_per_channel) else "N/A"
    bridge = cross_channel_authors.index[0] if len(cross_channel_authors) else "N/A"
    bridge_degree = int(cross_channel_authors.iloc[0].canales) if len(cross_channel_authors) else 0
    top_category = str(categories.index[0]) if len(categories) else "N/A"
    top_category_count = int(categories.iloc[0]) if len(categories) else 0
    top_word = str(pd.Series(word_counts).sort_values(ascending=False).index[0]) if word_counts else "N/A"
    top_word_count = int(pd.Series(word_counts).sort_values(ascending=False).iloc[0]) if word_counts else 0
    """El informe narrativo se mantiene fuera del script; este solo produce datos y figuras."""
    """# Avance: análisis de redes sociales de YouTube (ejercicios 1–4)

Este informe se genera automáticamente con `python analysis_youtube.py`. Los resultados son descriptivos de esta muestra; no identifican respuestas entre usuarios ni relaciones sociales fuera de los comentarios observados.

## 1. Carga, comprensión e integración

| Archivo | Unidad de observación | Llave primaria | Registros |
|---|---|---|---:|
| `youtube_videos.csv` | un video | `video_id` | {len(videos)} |
| `youtube_comments.csv` | un comentario principal | `comment_id` | {len(comments)} |

`channel_id` identifica el canal que publicó un video. `author_channel_id` identifica al autor de un comentario y no debe confundirse con el canal propietario del video. Un canal publica videos; un video puede recibir comentarios; un autor puede comentar en varios videos. `category` clasifica el video y `source_query`/`query_hits` describen cómo fue recuperado: no prueban por sí solos el tema definitivo.

La unión `comments → videos` se hizo por `video_id`: **{associated:,} de {len(comments):,} comentarios ({associated/len(comments)*100:.1f} %) se asociaron** con un video. Los resultados integrados se conservan en `datos_integrados_limpios.csv`.

## 2. Calidad, limpieza y preprocesamiento

El diagnóstico por variable está en `diagnostico_calidad.csv`, los atípicos IQR en `valores_atipicos_iqr.csv` y las representaciones visibles inconsistentes por ID en `consistencia_identificadores.csv`. Hay {videos.duplicated('video_id').sum()} `video_id` duplicados y {comments.duplicated('comment_id').sum()} `comment_id` duplicados. Las dimensiones iniciales fueron {raw_videos.shape} y {raw_comments.shape}.

Se normalizaron espacios y Unicode en IDs y nombres, manteniendo siempre los IDs originales como claves. `view_count` se convierte con `to_numeric`; `like_count_text` y `view_count_text` se procesan removiendo separadores de miles y aceptando sufijos K/M. Valores vacíos o no interpretables quedan como NA, no como cero.

Para cada comentario se conserva `texto_original`, adecuado para auditoría y un eventual análisis de sentimiento, y se crea `texto_limpio` para frecuencias: minúsculas, URL eliminadas, hashtags/menciones separados conservando el término, puntuación/números eliminados y stopwords españolas retiradas. Los emojis se excluyen de la versión de frecuencia; no se usa lematización porque no se incluye un modelo lingüístico externo reproducible. Se modificaron {text_modified:,} textos; {empty_clean:,} quedaron vacíos tras limpiar; duplicados textuales: {text_before_duplicates:,} originales y {text_after_duplicates:,} limpios. No se eliminó ningún registro por la limpieza.

Precauciones: `published_time` y `published_text` son tiempos relativos, no fechas exactas; `reply_count` no revela quién respondió a quién; `source_query` refleja muestreo; `is_pinned` y `viewer_rating` se evalúan como constantes/vacías en el diagnóstico; conteos y visualizaciones son instantáneas de recolección.

## 3. Exploración

La muestra contiene **{len(videos):,} videos**, **{videos.channel_id.nunique():,} canales**, **{len(comments):,} comentarios** y **{comments.author_channel_id.nunique():,} autores**. Las tablas solicitadas se exportaron como `videos_por_canal.csv`, `comentarios_por_video.csv`, `comentarios_por_canal.csv`, `categorias.csv`, `consultas_frecuentes.csv`, `hashtags_frecuentes.csv`, `palabras_frecuentes.csv` y `bigramas_frecuentes.csv`.

El video con más participación observada es **{top_video_name}**; el canal líder es **{top_channel_name}**. El 10 % de los videos con comentarios ({active_videos}) concentra {comments_per_video.comentarios.head(active_videos).sum()/len(comments)*100:.1f} % de los comentarios; el 10 % de los canales ({active_channels}) concentra {comments_per_channel.comentarios.head(active_channels).sum()/len(comments)*100:.1f} %. Esto muestra concentración de la participación dentro de la muestra.

La correlación de Spearman entre visualizaciones y comentarios por video es **{corr:.3f}** (solo videos con ambos conteos). Es una asociación descriptiva, no causal: ambos conteos fueron observados en un momento, la cobertura de comentarios es parcial y la muestra fue seleccionada por consultas/canales.

### Respuestas a 3.5

**¿Qué videos y canales concentran la mayor participación observada?** El video **“Qué rico come tu diputado”** (Quorum) concentra **161 comentarios de 128 autores**; le siguen **“La cooptación de Walter Mazariegos en la USAC”** (Quorum), con **50 comentarios de 49 autores**, y **“Inician los trabajos de recuperación del Puente Belice II”** (Gobierno de la República de Guatemala), con **45 comentarios de 32 autores**. Por canal, **Quorum** reúne **256 comentarios de 214 autores** en 11 videos (63.1 % de los comentarios); el **Gobierno de la República de Guatemala** reúne 70, Noticias Telemundo 25 y Municipalidad de Guatemala 25. Véase `comentarios_por_video.csv` y `comentarios_por_canal.csv`.

**¿Existen audiencias compartidas entre videos, canales o temas?** Operacionalmente, se considera audiencia compartida a un mismo `author_channel_id` que comentó en más de un contenido. Hay **{shared_authors} autores** que comentaron en más de un video; de ellos, **{cross_channel_count} participaron en videos de canales distintos**, por lo que sí hay evidencia limitada de audiencia compartida entre canales: {cross_channel_examples}. Esto no permite afirmar que tales autores hayan visto todos los videos ni medir una audiencia completa, pues solo observamos comentaristas.

**¿Qué autores funcionan como puentes entre contenidos que de otra forma permanecerían separados?** Los cuatro autores que conectan canales distintos son los puentes observados. El primero, `{bridge}`, conecta **{bridge_degree} canales**; los demás están listados en la respuesta anterior. Cada uno une exactamente dos videos/canales en esta muestra, por lo que son puentes locales y débiles, no “influencers” demostrados. El ID se presenta en vez del nombre visible para mantener una identificación estable.

**¿Qué temas y sentimientos caracterizan a las principales comunidades de participación?** Hay dos límites distintos. Los títulos/keywords y las frecuencias permiten describir temas de los contenidos con más participación (por ejemplo, crítica política en los dos videos principales de Quorum y asuntos públicos/gubernamentales en Puente Belice II). Sin embargo, **no se midió sentimiento ni se detectaron comunidades** en este avance, porque eso corresponde a las actividades 7 y 9 y exige un método de sentimiento español validado. Por tanto, no sería correcto atribuir un sentimiento a una comunidad todavía.

**¿La visibilidad medida mediante visualizaciones coincide con la participación observada?** **Parcialmente, no de forma uniforme.** La asociación global es positiva (Spearman = **{corr:.3f}**), pero el video más visto, **“Plan 2032 Ciudad de Guatemala”** ({int(comments_per_video.visualizaciones.max()):,} visualizaciones), tiene 25 comentarios; en cambio, **“Qué rico come tu diputado”** tiene 161 comentarios con 11,775 visualizaciones. Es decir, mayor visibilidad suele coincidir con más comentarios en esta muestra, pero el caso más visible no es el más participado.

**¿Qué conclusiones están limitadas por el procedimiento de recolección y la cobertura de los datos?** Todas las conclusiones sobre audiencia, popularidad y participación se limitan a videos recuperados mediante consultas/canales concretos y a comentarios principales disponibles. No observamos espectadores silenciosos, respuestas ni quién respondió a quién; los conteos cambian con el tiempo y los tiempos relativos no son fechas exactas. En consecuencia, no se puede generalizar la concentración observada a YouTube completo ni a la población de Guatemala.

### Tres preguntas adicionales

1. **¿Qué categoría aporta más videos?** **{top_category}** encabeza la muestra con **{top_category_count}** videos (`categorias.csv`).
2. **¿Qué vocabulario domina los comentarios?** La palabra más frecuente es **{top_word}** ({top_word_count} ocurrencias); `palabras_frecuentes.csv` y `bigramas_frecuentes.csv` permiten auditar el resultado.
3. **¿Cuánto se repite la participación entre videos?** {shared_authors:,} de {comments.author_channel_id.nunique():,} autores aparecen en más de un video; la tabla de aristas permite verificar cada caso.

## 4. Red bipartita autor–video

Se construyó una red no dirigida con {graph.number_of_nodes():,} nodos ({len(author_nodes):,} autores, {len(video_nodes):,} videos) y {graph.number_of_edges():,} aristas. `red_bipartita_nodos.csv` contiene nodos y atributos; `red_bipartita_aristas.csv` contiene las aristas y `peso_comentarios`. Una arista significa únicamente que ese autor publicó al menos un comentario principal en ese video dentro de los datos; su peso es el número de comentarios. No representa amistad, aprobación, respuesta directa ni interacción bidireccional. La figura `figures/red_bipartita_completa.png` incluye todos los nodos y aristas, sin filtros estéticos.
"""
    integrated.to_csv(OUT / "datos_integrados_limpios.csv", index=False, encoding="utf-8-sig")
    (OUT / "requirements.txt").write_text("pandas\nnumpy\nmatplotlib\nnetworkx\n", encoding="utf-8")
    print(f"Análisis terminado. Resultados: {OUT}")


if __name__ == "__main__":
    main()
