# Chat local con PDF

Interfaz Streamlit con selector de modelos Ollama, carga de PDF, respuestas
progresivas, historial y descarga de conversación. Python 3.10 o superior.

## Iniciar en macOS

Descomprime el ZIP y abre una terminal en la carpeta `chat_pdf`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

Abre http://localhost:8501. Ollama debe estar ejecutándose en la misma
computadora, en http://localhost:11434. Si no está abierto, ejecuta
`ollama serve` en otra terminal. Si ya está abierto no lo ejecutes otra vez.

1. Elige un modelo; por defecto se selecciona `mistral:7b-instruct` si existe.
2. Selecciona el PDF y pulsa **Cargar documento**.
3. Escribe “Explícame de qué trata”, “Amplía el segundo punto” o “Dame un ejemplo”.
4. Usa **Nueva conversación** para empezar de nuevo con el mismo PDF.

## Funcionamiento y límites

Esta versión manda el texto completo y todo el historial en cada pregunta.
No utiliza embeddings, una base vectorial ni LangChain. `nomic-embed-text`
no se necesita. El historial se conserva durante la sesión de Streamlit;
recargar la página o reiniciar el proceso puede borrarlo. Puedes descargarlo.
Cambiar o quitar el archivo limpia el documento y la conversación anteriores.
Cambiar de modelo conserva el historial, pero puede cambiar la calidad.

Se consulta `/api/show` para limitar las opciones de contexto a la capacidad
declarada del modelo. Aumentar contexto requiere más RAM; la capacidad teórica
del modelo no garantiza que tu computadora pueda ejecutarla.

El presupuesto de texto utiliza bytes UTF-8 más márgenes como estimación
conservadora; no es el conteo exacto del tokenizer y puede rechazar un texto
que técnicamente cabría. No se recorta el documento ni se eliminan turnos
automáticamente. Si el texto no cabe, aumenta contexto, limpia el historial
o usa un documento más corto. Para PDFs extensos se necesitaría una versión
con RAG conversacional o procesamiento por secciones.

Solo se extrae texto: no hay OCR, interpretación de imágenes ni comprensión
garantizada de tablas o columnas. Las páginas citadas corresponden a su orden
físico en el PDF. Revisa el texto extraído si el contenido parece incompleto.
Los PDFs con contraseña deben desbloquearse previamente. Máximo: 25 MB.

El documento se procesa en memoria y se envía al Ollama de localhost; la app
no utiliza APIs de nube ni guarda el PDF en disco. La instalación inicial
requiere Internet. No se mide aquí la política de logs de tu instalación de Ollama.

Edita `SYSTEM_PROMPT` en `app.py` para cambiar el comportamiento. Streamlit
recarga el código; usa Nueva conversación si quieres evaluar el nuevo prompt
sin respuestas anteriores que condicionen el diálogo.

## Validación

Se verificaron la sintaxis y la lógica de extracción, presupuesto de contexto,
historial y manejo de streaming mediante pruebas locales con respuestas
simuladas. No se ejecutó inferencia contra tu Ollama ni se midió su rendimiento.

Referencias de API:
- https://docs.ollama.com/api/chat
- https://docs.ollama.com/api/tags
- https://docs.ollama.com/api/show
