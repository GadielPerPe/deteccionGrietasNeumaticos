import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

def detect_cracks_with_labels(img_path,
                              clahe_clip=2.0,
                              clahe_grid=(8,8),
                              bh_kernel=(21,21),
                              otsu=True,
                              canny_sigma=0.33,
                              close_iter=3,
                              open_iter=1,
                              min_length=30):
    # 1. Leer y convertir a gris + CLAHE
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    gray = clahe.apply(gray)

    # 2. Black-hat morphology
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, bh_kernel)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)

    # 3. Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    clean_bh = cv2.morphologyEx(bh, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    clean_bh = cv2.morphologyEx(clean_bh, cv2.MORPH_OPEN,  kernel, iterations=open_iter)

    # 4. Binarización (Otsu o Canny)
    if otsu:
        _, binarized = cv2.threshold(clean_bh, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        v = np.median(clean_bh)
        low  = int(max(0, (1.0 - canny_sigma) * v))
        high = int(min(255, (1.0 + canny_sigma) * v))
        binarized = cv2.Canny(clean_bh, low, high)

    # 5. Esqueletización
    skeleton = skeletonize(binarized > 0).astype(np.uint8) * 255

    # 6. Detección y filtrado de contornos
    cnts, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = img.copy()
    for c in cnts:
        L = cv2.arcLength(c, False)
        if L < min_length:
            continue
        # color según longitud
        if L < 100:
            color = (0,255,0)        # verde
        elif L < 300:
            color = (0,165,255)      # naranja
        else:
            color = (0,0,255)        # rojo
        cv2.drawContours(out, [c], -1, color, 2)

    # 7. Preparar imágenes para etiquetar
    bh_viz    = cv2.cvtColor(bh,        cv2.COLOR_GRAY2BGR)
    clean_viz = cv2.cvtColor(clean_bh,  cv2.COLOR_GRAY2BGR)
    bin_viz   = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
    skl_viz   = cv2.cvtColor(skeleton,  cv2.COLOR_GRAY2BGR)

    # Cada tupla: (imagen BGR, título en español)
    top_imgs = [
        (img,       "Original"),
        (out,       "Contornos")
    ]
    bot_imgs = [
        (bh_viz,    "Black-hat"),
        (clean_viz, "Limpieza"),
        (bin_viz,   "Binarización"),
        (skl_viz,   "Esqueletización")
    ]

    # 8. Mostrar con Matplotlib usando GridSpec
    fig = plt.figure(figsize=(20, 10))
    gs  = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.2)

    # Fila superior: dos imágenes y dos ejes ocultos
    for i, (im, title) in enumerate(top_imgs):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    for i in range(len(top_imgs), 4):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")

    # Fila inferior: cuatro etapas procesadas
    for i, (im, title) in enumerate(bot_imgs):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.suptitle("Pipeline de Detección de Grietas con Etiquetas", fontsize=16)
    plt.show()


if __name__ == "__main__":
    detect_cracks_with_labels(
        r"C:\Users\carlo\OneDrive\Documents\ProyectoResidencias\NeumaticosAgrietados\testing_data\cracked\Cracked-3.jpg",
        clahe_clip=2.0,
        clahe_grid=(8,8),
        bh_kernel=(21,21),
        otsu=True,
        canny_sigma=0.33,
        close_iter=3,
        open_iter=1,
        min_length=30
    )