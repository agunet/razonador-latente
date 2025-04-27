
import torch
import faiss
import requests
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "gemma3:12b"

DOCUMENTOS = [
    "La inteligencia artificial permite automatizar procesos complejos.",
    "El aprendizaje automático analiza datos para mejorar decisiones.",
    "La ciencia de datos identifica patrones ocultos en grandes volúmenes de información.",
    "La computación en la nube facilita el acceso remoto a recursos tecnológicos.",
    "Los algoritmos optimizan tareas repetitivas en diversas industrias.",
    "El procesamiento de lenguaje natural ayuda a las máquinas a comprender el lenguaje humano.",
    "La ciberseguridad protege activos digitales frente a amenazas.",
    "La robótica mejora la eficiencia mediante la automatización física.",
    "Blockchain asegura la integridad de transacciones sin intermediarios.",
    "La realidad aumentada potencia la interacción entre humanos y sistemas digitales."
]

# Cargar SBERT y Manipulador
sbert = SentenceTransformer(SBERT_MODEL).to(DEVICE)

class ManipuladorLatente(nn.Module):
    def __init__(self, input_dim, output_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, output_dim)   # Salida fija en 768
        )

    def forward(self, x):
        return self.mlp(x)

manipulador = ManipuladorLatente(input_dim=384).to(DEVICE)
manipulador.load_state_dict(torch.load("manipulador_latente.pth"))
manipulador.eval()

# Crear FAISS para ambos métodos
embeddings_sbert = sbert.encode(DOCUMENTOS, convert_to_tensor=True).to(DEVICE)
index_faiss = faiss.IndexFlatL2(384)
index_faiss.add(embeddings_sbert.cpu().detach().numpy())

embeddings_manip = manipulador(embeddings_sbert).cpu().detach().numpy()
index_semantico = faiss.IndexFlatL2(768)
index_semantico.add(embeddings_manip)

def ollama_responder(contexto, pregunta, modo):
    contexto_texto = "\n".join(f"- {t}" for t in contexto)
    if modo == "rag":
        prompt = f"Responde brevemente: {pregunta}\nContexto:\n{contexto_texto}"
    else:
        prompt = f"""Usa el siguiente contexto para responder la pregunta de forma clara y razonada:

Contexto:
{contexto_texto}

Pregunta:
{pregunta}

Respuesta:"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt},
        stream=True
    )
    respuesta = ""
    for linea in response.iter_lines():
        if linea:
            data = json.loads(linea.decode('utf-8'))
            respuesta += data.get("response", "")
    return respuesta.strip()

def comparar_sistemas(pregunta, k=3):
    print(f"\n🔹 Consulta: {pregunta}\n")

    # RAG Tradicional
    query_emb = sbert.encode([pregunta], convert_to_tensor=True).to(DEVICE).cpu().detach().numpy()
    _, idxs = index_faiss.search(query_emb, k)
    textos_rag = [DOCUMENTOS[i] for i in idxs[0]]

    print("── RAG Tradicional (Coseno) ──")
    for t in textos_rag:
        print(f"- {t}")
    resp_rag = ollama_responder(textos_rag, pregunta, modo="rag")
    print(f"Respuesta:\n{resp_rag}\n")

    # Razonador Semántico
    query_manip = manipulador(torch.tensor(query_emb).to(DEVICE)).cpu().detach().numpy()
    _, idxs_sem = index_semantico.search(query_manip, k)
    textos_sem = [DOCUMENTOS[i] for i in idxs_sem[0]]

    print("── Razonador Semántico ──")
    for t in textos_sem:
        print(f"- {t}")
    resp_sem = ollama_responder(textos_sem, pregunta, modo="razonador")
    print(f"Respuesta:\n{resp_sem}")

if __name__ == "__main__":
    consulta = "¿Cómo ayuda la inteligencia artificial en el sector salud?"
    comparar_sistemas(consulta)
