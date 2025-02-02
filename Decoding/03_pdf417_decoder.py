import cv2
import numpy as np
import subprocess
import os
import sys

# Paths to required files
javase_jar = "javase-3.5.0.jar"
core_jar = "core-3.5.0.jar"
jcommander_jar = "jcommander-1.82.jar"

barcode_image = "deskewed_image.png"  # The deskewed image to process

# Validate required files
for file in [javase_jar, core_jar, jcommander_jar, barcode_image]:
    if not os.path.exists(file):
        sys.exit(f"Error: {file} not found!")

# Preprocess the image before passing to ZXing
def preprocess_image(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to read the image. It may be corrupted or an unsupported format.")
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply Otsu's thresholding to make the image binary
        _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Save and return the thresholded image
        thresholded_image_path = "thresholded_image.png"
        cv2.imwrite(thresholded_image_path, thresholded_image)
        return thresholded_image_path
    except Exception as e:
        sys.exit(f"Image Processing Error: {str(e)}")

processed_image = preprocess_image(barcode_image)

# Docker command to detect the barcode and get its position
docker_command = [
    "docker", "run", "--rm",
    "-v", f"{os.getcwd()}:/app",
    "openjdk:17",
    "java", "-cp",
    f"/app/{javase_jar}:/app/{core_jar}:/app/{jcommander_jar}",
    "com.google.zxing.client.j2se.CommandLineRunner",
    f"/app/{processed_image}"
]

try:
    # Run the Docker command and capture both stdout and stderr
    result = subprocess.run(docker_command, capture_output=True, text=True, check=True, encoding='utf-8')
    output = result.stdout.strip()
    
    if not output:
        raise RuntimeError("ZXing did not return any output. Possible reasons: invalid barcode, poor image quality, or missing dependencies.")
    
    print("Decoded Output:")
    print(output)

except subprocess.CalledProcessError as e:
    sys.exit(f"ZXing Error: {e.stderr.strip() or 'Unknown error occurred while processing the barcode.'}")
except Exception as e:
    sys.exit(f"Unexpected Error: {str(e)}")

# Parse the ZXing output for barcode position
points = []
for line in output.splitlines():
    if line.startswith("  Point"):
        try:
            parts = line.split(":")[1].strip().replace("(", "").replace(")", "").split(",")
            points.append((int(float(parts[0])), int(float(parts[1]))))
        except ValueError:
            sys.exit(f"Error parsing point data: {line}")

# If points are found, draw a bounding polygon
if len(points) >= 4:
    try:
        # Load the image with OpenCV
        image = cv2.imread(barcode_image)
        if image is None:
            raise ValueError("Unable to read the image for annotation.")

        # Draw a polygon connecting the points
        points_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        print(f"Drawing polygon with points: {points}")

        cv2.polylines(image, [points_array], isClosed=True, color=(0, 255, 0), thickness=2)

        # Save and display the annotated image
        annotated_image_path = "annotated_barcode.png"
        cv2.imwrite(annotated_image_path, image)
        print(f"Annotated image saved as {annotated_image_path}")

        # Display the image
        cv2.imshow("Detected Barcode", image)
        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        sys.exit(f"Error during annotation: {str(e)}")
else:
    sys.exit("Error: No bounding box points detected. ZXing failed to identify the barcode in the image.")
