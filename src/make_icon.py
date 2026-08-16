"""Draw an arc-reactor style icon and write build/icon.ico (+ icon.png)."""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "build"
SIZE = 256

NAVY = (11, 16, 32, 255)
CYAN = (34, 211, 238, 255)
CYAN_DIM = (14, 116, 144, 255)
CORE = (200, 245, 255, 255)


def draw() -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # dark navy rounded square
    d.rounded_rectangle((8, 8, SIZE - 8, SIZE - 8), radius=52, fill=NAVY)
    m = SIZE / 2
    # concentric cyan rings
    for r, w, col in ((92, 8, CYAN_DIM), (68, 7, CYAN), (46, 6, CYAN)):
        d.ellipse((m - r, m - r, m + r, m + r), outline=col, width=w)
    # segmented outer ticks (arc-reactor feel)
    import math
    for i in range(10):
        a = math.radians(i * 36)
        x1, y1 = m + 100 * math.cos(a), m + 100 * math.sin(a)
        x2, y2 = m + 112 * math.cos(a), m + 112 * math.sin(a)
        d.line((x1, y1, x2, y2), fill=CYAN_DIM, width=6)
    # glow + bright core
    d.ellipse((m - 30, m - 30, m + 30, m + 30), fill=(34, 211, 238, 90))
    d.ellipse((m - 18, m - 18, m + 18, m + 18), fill=CORE)
    return img


def main():
    OUT.mkdir(exist_ok=True)
    img = draw()
    img.save(OUT / "icon.png")
    img.save(OUT / "icon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("wrote", OUT / "icon.ico")


if __name__ == "__main__":
    main()
