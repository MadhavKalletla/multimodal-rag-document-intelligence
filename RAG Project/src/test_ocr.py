import pytesseract
from PIL import Image

# 👇 Tell Python where Tesseract.exe is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 👇 Load any test image that has text (e.g., test.png or sample.jpg)
img = Image.open("testimage.png")  # replace with your actual image name

# 👇 Extract text using OCR
text = pytesseract.image_to_string(img)

print("✅ OCR Output:")
print("---------------------------")
print(text)
