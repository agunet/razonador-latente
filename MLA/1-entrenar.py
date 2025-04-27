# Entrenamiento de un Manipulador Latente con SBERT (Salida 768)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import json

# ─── Configuración ───
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SBERT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
OUTPUT_DIM = 768   # Dimensión fija de salida

# ─── Cargar modelo SBERT ───
print("🔹 Cargando modelo SBERT...")
sbert = SentenceTransformer(SBERT_MODEL).to(DEVICE)
EMBEDDING_DIM = sbert.get_sentence_embedding_dimension()
print(f"✅ SBERT listo con dimensión: {EMBEDDING_DIM}")

# ─── Dataset Q&A ───
class QADataset(Dataset):
    def __init__(self, path_json):
        with open(path_json, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["question"], item["answer"]

dataset = QADataset("qa_dataset.json")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ─── Definir Manipulador Latente con salida 768 ───
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

manipulador = ManipuladorLatente(EMBEDDING_DIM, OUTPUT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(manipulador.parameters(), lr=LR)
criterion = nn.MSELoss()

# ─── Entrenamiento ───
print("🚀 Comenzando entrenamiento del Manipulador Latente...")
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    manipulador.train()

    for preguntas, respuestas in loader:
        emb_preg = sbert.encode(preguntas, convert_to_tensor=True).to(DEVICE)
        emb_resp = sbert.encode(respuestas, convert_to_tensor=True).to(DEVICE)

        # Proyectamos emb_resp también a 768 para comparar
        if emb_resp.shape[1] != OUTPUT_DIM:
            proyector = nn.Linear(emb_resp.shape[1], OUTPUT_DIM).to(DEVICE)
            emb_resp = proyector(emb_resp)

        pred = manipulador(emb_preg)
        loss = criterion(pred, emb_resp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"📚 Epoch {epoch}/{EPOCHS} — Loss promedio: {avg_loss:.6f}")

# ─── Guardar el modelo ───
torch.save(manipulador.state_dict(), "manipulador_salida_768.pth")
print("✅ Manipulador Latente entrenado y guardado como 'manipulador_salida_768.pth'")
