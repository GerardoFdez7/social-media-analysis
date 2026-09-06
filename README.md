# Analítica de Redes Sociales

Laboratorio 6 (CC3084 – Data Science, UVG). El análisis parte de `youtube_videos.csv` y
`youtube_comments.csv` y cubre los ejercicios 1 a 10: integración, limpieza, exploración,
red bipartita autor–video, proyecciones, topología, comunidades, centralidad y sentimiento.

## Requisitos

Python 3.10 o superior y las dependencias listadas en `analysis_results/requirements.txt`
(pandas, numpy, matplotlib, networkx, wordcloud y pysentimiento). La primera ejecución de
`pysentimiento` descarga el modelo `robertuito-sentiment-analysis` desde Hugging Face
(aproximadamente 450 MB) y lo deja en caché, por lo que ese paso necesita conexión a
internet una sola vez.

```powershell
python -m pip install pandas numpy matplotlib networkx wordcloud pysentimiento
```

## Ejecución

Desde la raíz del repositorio:

```powershell
python main.py
```

El script crea la carpeta `analysis_results/` con:

- `datos_integrados_limpios.csv`: comentarios unidos a los atributos del video, con
  `texto_original`, `texto_limpio`, conteos numéricos, comunidad y sentimiento.
- Diagnóstico de calidad: `diagnostico_calidad.csv`, `valores_atipicos_iqr.csv`,
  `consistencia_identificadores.csv`, `estadisticas_descriptivas.csv`.
- Exploración: `comentarios_por_video.csv`, `comentarios_por_canal.csv`,
  `videos_por_canal.csv`, `categorias.csv`, `consultas_frecuentes.csv`,
  `hashtags_frecuentes.csv`, `palabras_frecuentes.csv`, `bigramas_frecuentes.csv`.
- Redes: `red_bipartita_nodos.csv`, `red_bipartita_aristas.csv`, `red_bipartita.gexf`
  (para Gephi), `proyeccion_autor_autor_aristas.csv`, `proyeccion_video_video_aristas.csv`,
  `metricas_topologia.csv`, `distribucion_grados.csv`, `nodos_perifericos.csv`.
- Comunidades y centralidad: `comunidades_nodos.csv`, `comunidades_resumen.csv`,
  `centralidad_autores.csv`, `centralidad_videos.csv`, `puntos_articulacion.csv`,
  `autores_multicanal.csv`.
- Sentimiento: `sentimiento_comentarios.csv`, `sentimiento_por_video.csv`,
  `sentimiento_por_canal.csv`, `sentimiento_por_comunidad.csv`.
- `resumen_metricas.json` con las cifras citadas en el informe y `figures/` con las
  visualizaciones en PNG.

La semilla (`SEED = 42`) fija los resultados de Louvain, de los layouts de red y de la nube
de palabras. La lista de stopwords en español está incluida en el código para no depender de
descargas adicionales.
