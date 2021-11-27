import numpy as np
from PIL import Image, ImageDraw, ImageFont

def make_dummy_image_with_text(text, font, grid_size):
    center_h = 0
    center_w = grid_size[1] * 16 // 2
    
    img_coords = np.meshgrid(np.arange(grid_size[0] * 16), np.arange(grid_size[1] * 16), indexing='ij')
    delta = np.sqrt(np.square(img_coords[0] - center_h) + np.square(img_coords[1] - center_w))

    yellow_intensity = np.maximum(1 - delta / np.percentile(delta, 80), 0.0)
    yellow_intensity = yellow_intensity[..., None]
    green = (45, 136, 127)
    yellow = (241, 248, 153)
    res = np.array(green)[None, None] * (1.0 - yellow_intensity) + np.array(yellow)[None, None] * yellow_intensity

    plot_color = (19, 68, 33)
    sinusoids = np.abs(np.random.randn(5))
    t_vals = np.arange(grid_size[1] * 16 / 8)
    fake_plot_vals = np.abs((np.cos(t_vals[:, None] * sinusoids[None] + np.random.random() * np.pi)).sum(-1))

    for i, r_i in enumerate(fake_plot_vals):
        height = int(20 * np.power(r_i / 5.0, 1.5))
        height = min(height, grid_size[0] * 16)
        w0 = 10 + i * 8
        w1 = w0 + 6
        H = grid_size[0] * 16
        h0 = H - height - 10
        h1 = H - 10
        res[h0:h1, w0:w1] = plot_color

    pimg = Image.fromarray(res.astype(np.uint8))

    font_i = ImageFont.truetype(
        font=font, size=24
    )
    txt_w, txt_h = font_i.getsize(text)

    draw = ImageDraw.Draw(pimg, mode='RGBA')
    x1 = center_w - txt_w // 2
    x2 = x1 + txt_w
    y1 = 10
    y2 = 10 + txt_h

    draw.rectangle([(x1, y1), (x1 + txt_w, y1 + txt_h)], fill=(255, 255, 255, 20))
    draw.text((x1, y1), text=text, fill=(0,0,0), font=font_i)

    draw.text(
        (10, grid_size[0] * 16 - 50), 
        text='the sounds channel', 
        fill=(plot_color[0] // 2, plot_color[1] // 2, plot_color[2] // 2),
        font=ImageFont.truetype(font=font, size=12)
    )
    return pimg
