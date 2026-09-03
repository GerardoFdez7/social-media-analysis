# Analítica de Redes Sociales

Ejecute el análisis desde la raíz del repositorio:

```powershell
python -m pip install -r analysis_results/requirements.txt
python main.py
```

El script usa `youtube_videos.csv` y `youtube_comments.csv`, y crea `analysis_results/` con datos limpios e integrados, diagnóstico, tablas de exploración, tablas de nodos/aristas, un archivo GEXF y figuras. No descarga recursos externos: la lista de stopwords está incluida en el código para mantener la ejecución reproducible.
