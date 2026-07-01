import sys
import os
import torch
from PIL import Image
from torchvision import transforms

from src.train_model import CNN, IMG_SIZE, DEVICE

# Configuração
MODEL_PATH = "pesos_modelo.pth"
CLASS_NAMES = ["Goat", "Sheep"] 

EXTENSOES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp")

def carregar_modelo(model_path):
    model = CNN(num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def get_transform():
    # Mesma normalização usada no treino/validação
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def prever_imagem(model, image_path, transform):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)  # adiciona dimensão de batch

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()  # probabilidade da classe 1 (CLASS_NAMES[1])

    classe_predita = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confianca = prob if prob > 0.5 else 1 - prob

    return classe_predita, confianca, prob


def main():
    if len(sys.argv) != 2:
        print("Uso: python testar_modelo.py <imagem_ou_pasta>")
        sys.exit(1)

    caminho = sys.argv[1]

    if not os.path.exists(MODEL_PATH):
        print(f"Modelo '{MODEL_PATH}' não encontrado. Treine o modelo primeiro")
        sys.exit(1)

    model = carregar_modelo(MODEL_PATH)
    transform = get_transform()

    # Caminho é uma única imagem
    if os.path.isfile(caminho):
        classe, confianca, prob = prever_imagem(model, caminho, transform)
        print(f"{os.path.basename(caminho)}: {classe} (confiança: {confianca:.2%}, "
              f"prob. bruta de '{CLASS_NAMES[1]}': {prob:.4f})")

    # Caminho é uma pasta com várias imagens
    elif os.path.isdir(caminho):
        arquivos = [f for f in os.listdir(caminho) if f.lower().endswith(EXTENSOES_VALIDAS)]
        if not arquivos:
            print(f"Nenhuma imagem encontrada em '{caminho}'.")
            sys.exit(1)

        print(f"Testando {len(arquivos)} imagens...\n")
        for nome_arquivo in sorted(arquivos):
            caminho_completo = os.path.join(caminho, nome_arquivo)
            classe, confianca, prob = prever_imagem(model, caminho_completo, transform)
            print(f"{nome_arquivo}: {classe} (confiança: {confianca:.2%})")

    else:
        print(f"Caminho '{caminho}' não é um arquivo nem uma pasta válida.")
        sys.exit(1)


if __name__ == "__main__":
    main()