
# Comparativa: RAG Tradicional vs Razonador Semántico

Este script demuestra la diferencia entre un sistema RAG clásico y el Razonador Semántico creado por **agunet**.

## Requisitos
- Python 3
- torch, faiss, sentence-transformers, requests
- Ollama corriendo (modelo gemma3:12b)
- Archivo del manipulador: `manipulador_latente.pth`

## Uso
1. Asegurate de tener Ollama activo.
2. Ejecutá el script:

```
python3 comparativa_razonador.py
```

Modificá la variable `consulta` para probar diferentes preguntas.
