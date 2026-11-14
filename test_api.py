
import requests

image_path = "test_face.jpg"

with open(image_path, "rb") as image:
    response = requests.post(
        "http://127.0.0.1:5000/api/analyze",
        files={
            "frame": ("test_face.jpg", image, "image/jpeg")
        }
    )

print("Status:", response.status_code)
print("Response:", response.text)
