
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "images"
OUTPUT = ROOT / "data" / "image_contact_sheet.jpg"

images = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])

thumb_width = 220
thumb_height = 180
label_height = 30
columns = 4
rows = (len(images) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (columns * thumb_width, rows * (thumb_height + label_height)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for index, path in enumerate(images):

    try:
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_width - 10, thumb_height - 10))

        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)

        sheet.paste(image, (x + 5, y + 5))

        draw.text(
            (x + 5, y + thumb_height),
            path.name,
            fill="black"
        )

    except Exception as error:
        print("Skipping:", path.name, error)

sheet.save(OUTPUT, quality=95)

print("Contact sheet saved:", OUTPUT)
print("Images included:", len(images))
