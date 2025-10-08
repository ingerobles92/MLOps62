# 🧩 Guía para crear y configurar el contenedor MLOps62

Esta guía explica paso a paso cómo replicar el entorno completo del proyecto **MLOps62**, desde GitHub hasta Docker, para trabajar con los mismos datasets y notebooks de EDA, modelado y DVC.

---

## 1️⃣ Instrucciones de GitHub

### Opción A — Hacer **Fork** del repositorio (recomendado)
Esto crea tu **copia personal** del proyecto en tu cuenta de GitHub.

1. Entra a  
   👉 [https://github.com/ingerobles92/MLOps62](https://github.com/ingerobles92/MLOps62)
2. Haz clic en el botón **“Fork”** (arriba a la derecha).
3. GitHub creará una copia bajo tu usuario, por ejemplo:  
   `https://github.com/TU_USUARIO/MLOps62`
4. Desde esa copia puedes subir cambios libremente sin afectar el repo original.

### Opción B — Clonar directamente (solo lectura)
Si no necesitas modificar el código:
```bash
git clone https://github.com/ingerobles92/MLOps62.git
```

---

## 2️⃣ Setup inicial en tu computadora

### 2.1 Instalar Docker
Asegúrate de tener **Docker Engine + Docker Compose v2**.

#### En Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg]   https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" |   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable docker
sudo systemctl start docker
```

Verifica:
```bash
docker --version
docker compose version
```

### 2.2 Instalar Visual Studio Code
```bash
sudo apt install wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /usr/share/keyrings/
sudo sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install -y code
```

Abre VSCode y agrega las extensiones:
- **Dev Containers**
- **Python**
- **Docker**

---

## 2.3 Instalar Git (host) y preparar credenciales de GitHub
**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
git --version
```

**Configura tu identidad (host):**
```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu_correo@dominio.com"
```

**Guarda tu Personal Access Token (PAT) para no reingresarlo:**
1) Crea el token en https://github.com/settings/tokens (scopes: `repo`, `workflow`).
2) Guarda credenciales:
```bash
git config --global credential.helper store
# La primera vez que hagas pull/push, escribe usuario y token; quedará en ~/.git-credentials
```

> **En el contenedor**, si haces `git push` desde dentro, repite la configuración:
```bash
git config --global credential.helper store
git config --global user.name "Tu Nombre"
git config --global user.email "tu_correo@dominio.com"
```

---

## 3️⃣ Clonar el repositorio

Crea la misma estructura usada en clase para homogeneidad:
```bash
cd ~/Documents
mkdir -p MLOps
cd MLOps
git clone https://github.com/TU_USUARIO/MLOps62.git
cd MLOps62
```

---

## 4️⃣ Crear el contenedor

Dentro del folder `MLOps62`:
```bash
docker compose build --no-cache
docker compose up -d
```

Verifica que esté corriendo:
```bash
docker compose ps
```

Deberías ver algo como:
```
mlops62-mlops-app   ...  0.0.0.0:8888->8888/tcp
mlops62-mlflow      ...  0.0.0.0:9001->9001/tcp
```

Para entrar al contenedor:
```bash
docker compose exec mlops-app bash
```

---

## 5️⃣ Configurar credenciales dentro y fuera del contenedor

### 5.1 Credenciales de AWS
Cada integrante tiene sus propias llaves AWS (Access Key y Secret Key).

Dentro del contenedor:
```bash
export AWS_ACCESS_KEY_ID="TU_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="TU_SECRET_KEY"
export AWS_DEFAULT_REGION="us-west-2"
```

Para mantenerlas permanentes, puedes agregarlas al archivo de perfil:
```bash
echo 'export AWS_ACCESS_KEY_ID="TU_ACCESS_KEY"' >> ~/.bashrc
echo 'export AWS_SECRET_ACCESS_KEY="TU_SECRET_KEY"' >> ~/.bashrc
echo 'export AWS_DEFAULT_REGION="us-west-2"' >> ~/.bashrc
source ~/.bashrc
```

### 5.2 Token de GitHub (dentro del contenedor)
Crea un **Personal Access Token (PAT)** en:  
👉 [https://github.com/settings/tokens](https://github.com/settings/tokens)

Luego dentro del contenedor:
```bash
git config --global credential.helper store
git config --global user.name "Tu Nombre"
git config --global user.email "tu_correo@dominio.com"

# La primera vez que hagas push te pedirá usuario/token y quedará guardado
git pull
git push
```

### 5.3 Configurar credenciales de GitHub fuera del contenedor (recomendado)
Esto asegura que Git funcione correctamente también en tu máquina local.

1. Crea un token en [https://github.com/settings/tokens](https://github.com/settings/tokens)
2. Guarda las credenciales en tu sistema:
```bash
git config --global credential.helper store
git config --global user.name "Tu Nombre"
git config --global user.email "tu_correo@dominio.com"
```
3. Ejecuta un `git pull` o `git push` una vez; se guardará tu token en `~/.git-credentials`.

Verifica:
```bash
cat ~/.git-credentials
```

---

## 6️⃣ Verificar DVC, AWS y GitHub antes de iniciar Jupyter

Asegúrate de que DVC esté conectado a tu remoto S3:
```bash
dvc remote list
# Debería mostrar:
# s3remote        s3://mlopsequipo62/mlops  (default)
```

Luego sincroniza:
```bash
dvc pull
```

Verifica que los datasets estén en `/work/data/raw`:
```bash
ls -lh data/raw
```

Si ves los archivos `.csv` descargados correctamente, tu entorno está listo.

---

## 7️⃣ Iniciar JupyterLab

Dentro del contenedor:
```bash
jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token=''
```

Abre en tu navegador:
👉 [http://localhost:8888](http://localhost:8888)

---

## ✅ Estructura esperada del proyecto

```
MLOps62/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── interim/
│
├── notebooks/
│   └── EDA/
│
├── src/
│   └── ...
│
├── docker-compose.yml
├── requirements.txt
├── .dvc/
└── README.md
```

---

## 🧠 Notas finales

- Cada integrante debe trabajar en su propia **rama** o fork.
- Antes de iniciar una nueva sesión, ejecuta:
  ```bash
  docker compose up -d
  docker compose exec mlops-app bash
  source ~/.bashrc
  ```
- Si se corrompe el contenedor, puedes reconstruirlo:
  ```bash
  docker compose down --volumes
  docker compose build --no-cache
  docker compose up -d
  ```
