# U-Net Image Segmentation Project

Este proyecto implementa un modelo de segmentación de imágenes utilizando la arquitectura U-Net con PyTorch.

## Estructura del Proyecto

```
project/
│── data/
│   ├── images/ (Coloca tus imágenes aquí)
│   ├── masks/  (Coloca tus máscaras aquí)
│
│── src/
│   ├── dataset.py
│   ├── unet.py
│   ├── train.py
│   ├── utils.py
│   ├── metrics.py
│
│── notebooks/
│   ├── visualize_training.ipynb
│
│── requirements.txt
│── README.md
```

## Instalación

1.  Clona este repositorio o descarga los archivos.
2.  Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Preparación del Dataset

1.  Coloca tus imágenes originales en `data/images/`.
2.  Coloca las máscaras de segmentación correspondientes en `data/masks/`.
    *   Las máscaras deben tener el mismo nombre de archivo que las imágenes.
    *   Las máscaras deben ser imágenes en escala de grises donde los píxeles de interés tienen valor > 0.

## Entrenamiento

Para entrenar el modelo, ejecuta el script `train.py`:

```bash
python src/train.py --epochs 20 --batch_size 8 --lr 0.001
```

Argumentos disponibles:
*   `--epochs`: Número de épocas de entrenamiento (default: 20).
*   `--batch_size`: Tamaño del batch (default: 4).
*   `--lr`: Learning rate (default: 1e-4).
*   `--img_dir`: Directorio de imágenes (default: `data/images`).
*   `--mask_dir`: Directorio de máscaras (default: `data/masks`).
*   `--checkpoint_dir`: Directorio para guardar modelos (default: `checkpoints`).

El mejor modelo se guardará automáticamente en `checkpoints/best_model.pth`.

## Visualización

Abre el notebook `notebooks/visualize_training.ipynb` para visualizar los resultados del modelo entrenado.
