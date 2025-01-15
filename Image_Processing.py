import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Non Maximum Suppression with Intersection over Union
def non_maximum_suppression(contours, threshold=0.5):
    selected_contours = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        selected = True

        for selected_contour in selected_contours:
            x_sel, y_sel, w_sel, h_sel = cv2.boundingRect(selected_contour)
            area_sel = w_sel * h_sel

            # Calculate intersection over union
            x_intersect = max(x, x_sel)
            y_intersect = max(y, y_sel)
            x2_intersect = min(x + w, x_sel + w_sel)
            y2_intersect = min(y + h, y_sel + h_sel)

            intersect_area = max(0, x2_intersect - x_intersect) * max(
                0, y2_intersect - y_intersect
            )
            iou = intersect_area / (area + area_sel - intersect_area)

            if iou > threshold:
                selected = False
                break

        if selected:
            selected_contours.append(contour)

    return selected_contours


# Read the image
img = cv2.imread("jalan.jpg")

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Image Enhancement (Point Processing)

# Apply histogram equalization to enhance contrast
normalized_img = cv2.equalizeHist(gray)

# Apply median filter to reduce noise
median_filtered_img = cv2.medianBlur(normalized_img, 9)  # Adjust kernel size

## Image Restoration

# Apply deblurring filter to correct blurriness
# (Use an appropriate deblurring technique for the type of blur in the image)
# Example: deblurred_img = your_deblurring_function(median_filtered_img)
# You need to implement or use a suitable deblurring method here

## Egg Detection (Similar to previous code)

# Apply Gaussian filter to improve image quality
blur = cv2.GaussianBlur(median_filtered_img, (5, 5), 0)

# Detect edges using Canny method
edges = cv2.Canny(blur, 30, 90)  # Adjust threshold values if needed

# Morphological processing to clean up the edges
kernel = np.ones((5, 5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.erode(edges, kernel, iterations=1)

# Find contours (outer boundaries only)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
min_contour_area = 100  # Set as needed
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Apply Non Maximum Suppression
filtered_contours_nms = non_maximum_suppression(filtered_contours)

## Plot results

with PdfPages("D:/Kepentingan_Negara/kuliah/pcd/gambar/hasil.pdf") as pdf:
    plt.figure(figsize=(15, 8))

    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    # Histogram Equalization
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2RGB))
    plt.title("Histogram Equalization")
    plt.axis("off")

    # Median Filter
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(median_filtered_img, cv2.COLOR_GRAY2RGB))
    plt.title("Median Filter")
    plt.axis("off")

    # Edge Detection
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
    plt.title("Edge Detection")
    plt.axis("off")

    # Edge Detection after Median Filter
    plt.subplot(2, 3, 5)
    edges_median = cv2.Canny(median_filtered_img, 30, 90)
    plt.imshow(cv2.cvtColor(edges_median, cv2.COLOR_GRAY2RGB))
    plt.title("Edge Detection after Median Filter")
    plt.axis("off")

    # Contour Detection
    plt.subplot(2, 3, 6)
    contour_img = np.copy(img)
    cv2.drawContours(contour_img, filtered_contours_nms, -1, (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title("Contour Detection with NMS")
    plt.axis("off")

    plt.tight_layout()
    pdf.savefig()  # Save the current figure into the pdf page
    plt.close()

    # Count the number of objects and holes
    num_objects_before_nms = len(contours)
    num_objects_after_nms = len(filtered_contours_nms)

    # Histogram Citra Asli dan Citra Hasil Normalisasi
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(gray.ravel(), bins=256, color="gray")
    plt.title("Histogram Citra Asli")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Jumlah Piksel")
    plt.subplot(1, 2, 2)
    plt.hist(normalized_img.ravel(), bins=256, color="gray")
    plt.title("Histogram Citra Hasil Normalisasi")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Jumlah Piksel")
    plt.tight_layout()
    pdf.savefig()  # Save the current figure into the pdf page
    plt.close()

    # Histogram Citra Hasil Median Filter
    plt.figure(figsize=(6, 4))
    plt.hist(median_filtered_img.ravel(), bins=256, color="gray")
    plt.title("Histogram Citra Hasil Median Filter")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Jumlah Piksel")
    pdf.savefig()  # Save the current figure into the pdf page
    plt.close()

    # Jumlah Kontur Sebelum dan Sesudah Filtrasi Berdasarkan Luas
    areas_before = [cv2.contourArea(cnt) for cnt in contours]
    areas_after = [cv2.contourArea(cnt) for cnt in filtered_contours]
    plt.figure(figsize=(10, 5))
    plt.bar(
        ["Sebelum Filtrasi", "Sesudah Filtrasi"],
        [len(areas_before), len(areas_after)],
        color=["blue", "green"],
    )
    plt.title("Jumlah Kontur Sebelum dan Sesudah Filtrasi")
    plt.xlabel("Proses")
    plt.ylabel("Jumlah Kontur")
    pdf.savefig()  # Save the current figure into the pdf page
    plt.close()

    # Perbandingan Jumlah Objek Sebelum dan Setelah Non Maximum Suppression
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Sebelum NMS", "Setelah NMS"],
        [num_objects_before_nms, num_objects_after_nms],
        color=["blue", "green"],
    )
    plt.title("Perbandingan Jumlah Objek Sebelum dan Setelah NMS")
    plt.xlabel("Proses")
    plt.ylabel("Jumlah")
    pdf.savefig()  # Save the current figure into the pdf page
    plt.close()
