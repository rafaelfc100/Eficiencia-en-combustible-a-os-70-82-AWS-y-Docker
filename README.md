# Eficiencia en Combustible: Años 70-82 - AWS y Docker

## Descripción del Proyecto
Sistema completo de Machine Learning para clasificar automóviles según su eficiencia de combustible utilizando un modelo Perceptrón Multicapa (MLP). El proyecto abarca todo el ciclo de desarrollo de una aplicación de inteligencia artificial, desde el análisis exploratorio de datos hasta el despliegue en producción en la nube de AWS. La aplicación permite a usuarios predecir si un automóvil de los años 1970-1982 es eficiente o ineficiente en consumo de combustible basándose en sus características técnicas.

## Objetivo
Clasificar automóviles históricos (1970-1982) en dos categorías:
- **Auto más eficiente, bajo consumo de combustible**
- **Auto menos eficiente, alto consumo de combustible**

## Arquitectura del Sistema
```
┌─────────────────┐    HTTP Request    ┌─────────────────┐
│   Frontend      │───────────────────▶│    Backend      │
│   (HTML/JS/CSS) │                    │   (Flask API)   │
│   Nginx Docker  │◀───────────────────│  Python Docker  │
│   Puerto 80     │   JSON Response    │   Puerto 5000   │
└─────────────────┘                    └─────────────────┘
         │                                       │
         │          AWS EC2 Instances            │
         └───────────────────────────────────────┘
```

## Tecnologías Utilizadas

### Machine Learning & Data Science
- Scikit-learn: MLPClassifier para el modelo de red neuronal
- Pandas & NumPy: Análisis y manipulación de datos
- Matplotlib & Seaborn: Visualización de resultados
- StandardScaler: Normalización de características

### Backend Development
- Flask: Framework para la API REST
- Flask-CORS: Configuración para permitir solicitudes cruzadas
- Joblib: Serialización de modelos entrenados

### Frontend Development
- HTML5: Estructura de la interfaz web
- CSS3: Estilos y diseño responsivo
- JavaScript (ES6): Lógica del cliente y comunicación con la API

### Infraestructura y DevOps
- AWS EC2: Servidores virtuales en la nube
- Docker: Contenedores para reproducibilidad
- Docker Compose: Orquestación de múltiples servicios
- Nginx: Servidor web para el frontend
- SSH/SCP: Conexión segura y transferencia de archivos

## Dataset

### Características
- Nombre: Auto MPG Dataset
- Período: 1970-1982
- Muestras: 398 vehículos
- Características: 7 variables técnicas
- Fuente: UCI Machine Learning Repository

### Variables del Dataset
1. mpg: Millas por galón (variable objetivo transformada)
2. cylinders: Número de cilindros (3-8)
3. displacement: Desplazamiento del motor (pulgadas cúbicas)
4. horsepower: Caballos de fuerza
5. weight: Peso del vehículo (libras)
6. acceleration: Aceleración 0-60 mph (segundos)
7. model_year: Año del modelo (70-82)
8. origin: Origen del vehículo (1=USA, 2=Europa, 3=Japón)

## Preprocesamiento de Datos

### Limpieza y Preparación
1. Imputación de valores nulos: 6 valores faltantes en 'horsepower' reemplazados con la mediana
2. Eliminación de columna no predictiva: 'car_name' removida del análisis
3. Transformación de variable objetivo: Conversión de 'mpg' (continua) a variable categórica binaria usando la mediana como umbral
4. Normalización: Aplicación de StandardScaler para escalar todas las características

### División de Datos
- Conjunto de entrenamiento: 80% de los datos
- Conjunto de prueba: 20% de los datos
- Validación cruzada: 5 folds para optimización de hiperparámetros

## Modelo de Machine Learning

### Arquitectura MLP
El modelo implementado es un Perceptrón Multicapa (MLP) con la siguiente configuración óptima obtenida mediante búsqueda aleatoria:

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 2 capas ocultas
    activation='relu',            # Función de activación
    solver='adam',               # Optimizador
    alpha=0.001,                 # Regularización L2
    learning_rate_init=0.01,     # Tasa de aprendizaje
    max_iter=500,                # Máximo de iteraciones
    random_state=42              # Semilla para reproducibilidad
)
```

### Hiperparámetros Optimizados
Se realizó una búsqueda aleatoria (Randomized Search) sobre:
- Número de capas ocultas (1-3 capas)
- Neuronas por capa (16-128 neuronas)
- Función de activación (relu, tanh, logistic)
- Optimizador (adam, sgd)
- Tasa de aprendizaje (0.0001-0.1)
- Regularización L2 (0.0001-0.01)

## Backend - API Flask

### Endpoints Disponibles
1. GET / - Verificación de estado del servidor
2. POST /predict - Predicción de eficiencia de combustible

### Estructura de la Solicitud
```json
{
  "features": [cylinders, displacement, horsepower, weight, acceleration, model_year, origin]
}
```

### Estructura de la Respuesta
```json
{
  "prediction": "Auto más eficiente, bajo consumo de combustible"
}
```

### Configuración CORS
Implementación de Flask-CORS para permitir solicitudes desde el frontend desplegado en un dominio diferente.

## Frontend - Interfaz Web

### Componentes Principales
1. Formulario de entrada: 7 campos numéricos para las características del vehículo
2. Validación en tiempo real: Restricciones de rango para cada campo
3. Indicador visual: Imágenes que cambian según el resultado de la predicción
4. Mensajes de error: Retroalimentación clara para el usuario

### Características Técnicas
- Diseño responsivo que funciona en dispositivos móviles y de escritorio
- Comunicación asíncrona con la API usando Fetch API
- Manejo de errores de conexión y validación
- Interfaz intuitiva con retroalimentación visual inmediata

## Despliegue en AWS EC2

### Configuración de Instancias
Se utilizaron dos instancias EC2 t3.micro (nivel gratuito):

1. Instancia Backend (backend-linux)
   - Sistema operativo: Amazon Linux 2023
   - Puerto abierto: 5000 (API Flask)
   - Configuración de seguridad: SSH restringido a IP específica

2. Instancia Frontend (frontend-linux)
   - Sistema operativo: Amazon Linux 2023
   - Puerto abierto: 80 (HTTP)
   - Configuración de seguridad: SSH restringido a IP específica

### Security Groups
Configuración de grupos de seguridad para controlar el tráfico de red:

**Backend Security Group:**
- Entrada: Puerto 5000 (0.0.0.0/0) - Acceso público a la API
- Entrada: Puerto 22 (Mi IP) - Acceso SSH restringido
- Salida: Todo el tráfico permitido

**Frontend Security Group:**
- Entrada: Puerto 80 (0.0.0.0/0) - Acceso público HTTP
- Entrada: Puerto 22 (Mi IP) - Acceso SSH restringido
- Salida: Todo el tráfico permitido

## Implementación con Docker

### Contenedor Backend
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY model ./model
EXPOSE 5000
CMD ["python", "app.py"]
```

### Contenedor Frontend
```dockerfile
FROM nginx:latest
COPY . /usr/share/nginx/html
EXPOSE 80
```

### Docker Compose (Desarrollo Local)
```yaml
version: "3.9"
services:
  backend:
    build: ./backend
    container_name: backend_app
    ports:
      - "5000:5000"
    networks:
      - autos_net

  frontend:
    build: ./frontend
    container_name: frontend_app
    ports:
      - "80:80"
    depends_on:
      - backend
    networks:
      - autos_net

networks:
  autos_net:
    driver: bridge
```

## Resultados del Modelo

### Métricas de Evaluación
| Métrica | Clase 0 (Eficiente) | Clase 1 (No Eficiente) | Global |
|---------|---------------------|------------------------|--------|
| Precisión | 81.4% | 97.2% | - |
| Recall | 97.2% | 81.4% | - |
| F1-Score | 88.6% | 88.6% | - |
| Exactitud | - | - | 88.6% |
| AUC-ROC | - | - | 0.91 |

### Matriz de Confusión
- Verdaderos Positivos (VP): 35
- Falsos Positivos (FP): 8
- Falsos Negativos (FN): 1
- Verdaderos Negativos (VN): 35

### Interpretación de Resultados
El modelo demostró un excelente desempeño con una exactitud global del 88.6% y un AUC-ROC de 0.91, indicando alta capacidad discriminativa. Ambas clases presentan el mismo F1-Score (88.6%), demostrando equilibrio en el desempeño del clasificador.

## Instalación y Uso

### Requisitos Previos
- Python 3.10+
- Docker y Docker Compose
- Cuenta de AWS (para despliegue en la nube)

### Ejecución Local
```bash
# 1. Clonar el repositorio
git clone [url-del-repositorio]
cd PF_AWS

# 2. Iniciar servicios con Docker Compose
docker-compose up --build

# 3. Acceder a la aplicación
# Frontend: http://localhost
# Backend: http://localhost:5000
```

### Despliegue en AWS EC2
```bash
# 1. Conectar a la instancia backend
ssh -i "clave.pem" ec2-user@IP_BACKEND

# 2. Instalar Docker
sudo yum install docker -y
sudo systemctl start docker
sudo usermod -a -G docker ec2-user

# 3. Transferir archivos
scp -i "clave.pem" -r ./backend ec2-user@IP_BACKEND:/home/ec2-user/

# 4. Desplegar backend
cd backend
docker build -t backend-app .
docker run -d -p 5000:5000 backend-app

# 5. Repetir pasos para el frontend con IP correspondiente
```

## Estructura del Proyecto
```
PF_AWS/
├── backend/
│   ├── app.py              # API Flask con CORS
│   ├── Dockerfile          # Contenedor Python
│   ├── requirements.txt    # Dependencias (Flask, scikit-learn, etc.)
│   ├── model/
│   │   ├── model.pkl      # Modelo ML entrenado
│   │   ├── scaler.pkl     # Scaler para normalización
│   │   └── classes.pkl    # Mapeo de etiquetas
│   └── train_model.py     # Script de entrenamiento
├── frontend/
│   ├── index.html         # Interfaz principal
│   ├── app.js             # Lógica JavaScript
│   ├── styles.css         # Estilos CSS
│   ├── Dockerfile         # Contenedor Nginx
│   └── img/               # Imágenes de resultados
└── docker-compose.yml     # Configuración de servicios
```

## Problemas Resueltos y Aprendizajes

### Desafíos Técnicos Superados
1. **Configuración de CORS**: Implementación correcta de Flask-CORS para permitir comunicación entre frontend y backend en diferentes dominios
2. **Gestión de IPs dinámicas**: Manejo de cambios en direcciones IP públicas al reiniciar instancias EC2
3. **Configuración de Security Groups**: Ajuste preciso de reglas de firewall para permitir tráfico específico
4. **Comunicación entre contenedores**: Establecimiento de conexión efectiva entre servicios Docker en instancias separadas
5. **Transferencia segura de archivos**: Uso de SCP con autenticación por clave SSH

### Lecciones Aprendidas
- La importancia de la configuración de CORS en aplicaciones web modernas
- Estrategias para manejar IPs dinámicas en entornos cloud
- Mejores prácticas para seguridad en AWS EC2
- Ventajas de la arquitectura separada frontend/backend
- Procesos de despliegue continuo en infraestructura cloud

## URLs de Producción
- **Frontend en producción**: http://3.16.157.250/
- **Backend en producción**: http://3.144.186.31:5000/
- **Endpoint de prueba**: http://3.144.186.31:5000/predict

## Mejoras Futuras
1. Implementación de HTTPS/SSL para comunicación segura
2. Adición de sistema de autenticación y autorización
3. Integración con base de datos para almacenamiento de predicciones
4. Implementación de sistema de logging y monitoreo
5. Creación de API de administración para gestión del modelo
6. Implementación de CI/CD para despliegue automatizado

## Autor
**Rafael Alejandro Frías Cortez**  
Licenciatura en Ingeniería de Datos e Inteligencia Artificial  
División de Ingenierías Campus Irapuato-Salamanca  
Universidad de Guanajuato  
Email: ra.friascortez@ugto.mx

## Licencia
Este proyecto está desarrollado con fines educativos como parte de la formación académica en Ingeniería de Datos e Inteligencia Artificial.

## Referencias
- UCI Machine Learning Repository: Auto MPG Dataset
- Documentación oficial de Flask
- Documentación oficial de AWS EC2
- Documentación oficial de Docker
- Scikit-learn: Machine Learning in Python

---
*Proyecto académico desarrollado como demostración de habilidades en Machine Learning, desarrollo web y despliegue en la nube.*
