from flask import Flask, render_template, request, send_file
import os
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Base path to your processed .npy data
BASE_DIR = "processed"

@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    error = None

    if request.method == "POST":
        patient_id = request.form["patient_id"]
        slice_idx = request.form["slice_idx"]

        folder = os.path.join(BASE_DIR, patient_id)
        img_file = os.path.join(folder, f"img_{slice_idx}.npy")
        mask_file = os.path.join(folder, f"mask_{slice_idx}.npy")

        if not os.path.exists(img_file) or not os.path.exists(mask_file):
            error = "Selected slice not found. Please try another index."
        else:
            # Load and display slice
            img = np.load(img_file)
            mask = np.load(mask_file)

            # Save output figure
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap="gray")
            plt.title("FLAIR Slice")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap="hot")
            plt.title("Segmentation Mask")
            plt.axis("off")

            plt.tight_layout()
            output_path = "static/images/slice.png"
            # Ensure output directory exists
            os.makedirs("static/images", exist_ok=True)

            plt.savefig(output_path)
            plt.close()
            image_path = output_path

    return render_template("index.html", image_path=image_path, error=error)

if __name__ == "__main__":
    app.run(debug=True)
