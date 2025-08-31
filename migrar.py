import os
import shutil
import time

def migrar_arquivos(origem, destino):
    # Garante que o diretório de destino existe
    os.makedirs(destino, exist_ok=True)

    agora = time.time()  # timestamp atual

    for raiz, _, arquivos in os.walk(origem):
        # recria a estrutura de pastas dentro do destino
        rel_path = os.path.relpath(raiz, origem)
        dest_dir = os.path.join(destino, rel_path)
        os.makedirs(dest_dir, exist_ok=True)

        for arquivo in arquivos:
            caminho_origem = os.path.join(raiz, arquivo)
            caminho_destino = os.path.join(dest_dir, arquivo)

            # copia o arquivo
            shutil.copy2(caminho_origem, caminho_destino)

            # atualiza data de acesso e modificação
            os.utime(caminho_destino, (agora, agora))

    print(f"Migração concluída de '{origem}' para '{destino}'.")

# Exemplo de uso
if __name__ == "__main__":
    origem = "/home/wsi/repositorios/GitHub/wsi-koko-rvc-api/files/"
    destino = "/home/wsi/repositorios/GitHub/wsi-koko-rvc-api/files2"
    migrar_arquivos(origem, destino)
