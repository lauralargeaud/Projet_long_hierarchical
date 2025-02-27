import plotly.express as px
import pandas as pd
from PIL import Image

# For custom color palettes
import palettable as pt
import colorsys

def plot_hierarchical_perfs(perfs_csv,
                            metric_to_plot,
                            cmap_list=None,
                            show=False,
                            html_output=None,
                            png_output=None,
                            remove_lines=False,
                            font_size=32):

    # Load data from exported csv:
    data = pd.read_csv(perfs_csv)

    if not cmap_list:
        cmap_list = 'fall_r'

    fig = px.sunburst(data,
                        names = 'Name',
                        parents = 'Parent',
                        values = 'Count',
                        color = metric_to_plot,
                        branchvalues = 'total',
                        color_continuous_scale = cmap_list,
                        # hover_name = 'Name',
                        # hover_data={'Name': False, 'Parent': False, 'Count': True, 'F1-score': ':.2f'},
                        hover_data = ['Taxon_level'],
                        template = 'presentation'
                        )
    fig.update_layout(hoverlabel=dict(font_size=24, font_family="Rockwell"))
    fig.update_traces(hovertemplate=
                        '<b>%{label}</b> (%{customdata[0]})<br><br>' +
                        'Total Images = %{value}<br>' +
                        'F1-score = %{color:.2f}<br>')

    # Change max font size:
    fig.update_layout(font=dict(size=font_size))

    if remove_lines:
        fig.update_traces(marker=dict(
            line=dict(
                color="rgba(0,0,0,0)",  # Transparent lines
                width=0                # Line width set to 0
            )
        ))
    else:
        fig.update_traces(marker=dict(
            line=dict(
                color="rgba(20,20,20,0.35)",  # slightly Transparent lines
                width=1.2
            )
        ))

    # Custom bar scale:
    fig.update_layout(coloraxis_colorbar=dict(
        title=metric_to_plot,
        thicknessmode="fraction", thickness=0.05,
        lenmode="fraction", len=0.4,
        xanchor="left", x=1,
        tickvals=[0, 0.5, 1],
        ticktext=["0", "0.5", "1"],
        tickmode="array",
        tickfont=dict(size=26, family="Rockwell"),
        title_font=dict(size=28, family="Arial"),
    ))

    if show:
        fig.show()
    if html_output:
        fig.write_html(html_output)
    if png_output:
        # Set background color to transparent
        fig.update_layout({
            # 'plot_bgcolor': 'rgba(0, 0, 0, 0)', # not sure if useful
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        scale = 2
        fig.write_image(png_output, width=1100, height=1000, scale=scale)
        def trim_image(image_path, png_output, trim_pixels):
            with Image.open(image_path) as img:
                width, height = img.size
                left = trim_pixels[0]
                top = trim_pixels[1]
                right = width - trim_pixels[2]
                bottom = height - trim_pixels[3]
                img_cropped = img.crop((left, top, right, bottom))
                img_cropped.save(png_output)
        trim_pixels = [ scale * 85, # Left
                        scale * 60, # Top
                        scale * 20, # Right # TODO (was 80 before)
                        scale * 80] # Bottom
        trim_image(png_output, png_output, trim_pixels)


def get_custom_color_list(saturation_factor=1.0):
    color_list = pt.cartocolors.diverging.Fall_7.hex_colors # DEFAULT
    # color_list = pt.cartocolors.diverging.Geyser_7.hex_colors # ELECTRIC
    # color_list = pt.cartocolors.diverging.Temps_3.hex_colors # LOLIPOP
    
    color_list = color_list[::-1] # Reverse color order

    # Increase saturation of colors
    def increase_saturation(color, factor):
        r, g, b = [x / 255.0 for x in color]
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        s = min(1, s * factor)
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return [int(x * 255) for x in (r, g, b)]

    if saturation_factor != 1.0:
        color_list = [increase_saturation([int(color[i:i+2], 16) for i in (1, 3, 5)], saturation_factor) for color in color_list]
        color_list = ['#%02x%02x%02x' % tuple(color) for color in color_list]

    return color_list


# Usage Example:
run = False
if run:
    color_list = get_custom_color_list(saturation_factor=1.25)
    plot_hierarchical_perfs(perfs_csv="test.csv",
                            metric_to_plot="F1-score",
                            cmap_list=color_list,
                            show=True,
                            html_output="./export.html",
                            png_output="./export.png",
                            remove_lines=False,
                            font_size=32)