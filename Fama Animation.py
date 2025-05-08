import yfinance as yf  # Although not directly used for cylinder dimensions
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle, Circle

# --- Cylinder Setup ---
cylinder_radius = 1.0
cylinder_height = 5.0
liquid_color = 'blue'

# --- Fama-French Data for Liquid Level ---
start_date_anim = "2020-01-01"
end_date_anim = "2024-01-01"

try:
    ff_data_anim = web.get_data_famafrench('F-F_Research_Data_5_Factors_2x3', start=start_date_anim, end=end_date_anim)[0] / 100
    ff_data_anim.index = ff_data_anim.index.to_timestamp()
    liquid_level_data = ff_data_anim['Mkt-RF']  # Using 'Mkt-RF' as the liquid level driver

    fig, ax = plt.subplots()
    ax.set_xlim(-cylinder_radius - 0.5, cylinder_radius + 0.5)
    ax.set_ylim(0, cylinder_height + 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Cylinder base and walls
    base = Circle((0, 0), cylinder_radius, color='gray', alpha=0.5)
    wall_left = Rectangle((-cylinder_radius, 0), 0.1, cylinder_height, color='gray', alpha=0.5)
    wall_right = Rectangle((cylinder_radius - 0.1, 0), 0.1, cylinder_height, color='gray', alpha=0.5)
    top_ellipse = Circle((0, cylinder_height), cylinder_radius, color='gray', alpha=0.5)
    ax.add_patch(base)
    ax.add_patch(wall_left)
    ax.add_patch(wall_right)
    ax.add_patch(top_ellipse)

    liquid = Rectangle((-cylinder_radius, 0), 2 * cylinder_radius, 0, color=liquid_color)
    ax.add_patch(liquid)
    time_text = ax.text(0.5, cylinder_height + 0.5, '', ha='center')

    def animate_liquid(i):
        if i < len(liquid_level_data):
            level_factor = liquid_level_data.iloc[i]
            # Scale the factor to the cylinder height (you might need to adjust the scaling)
            liquid_height = cylinder_height * (level_factor + abs(liquid_level_data.min())) / (abs(liquid_level_data.max()) + abs(liquid_level_data.min()))
            liquid.set_height(max(0, liquid_height))
            time_text.set_text(liquid_level_data.index[i].strftime('%Y-%m'))
        return liquid, time_text

    ani = animation.FuncAnimation(fig, animate_liquid, frames=len(liquid_level_data), interval=200, blit=True)
    plt.title('Fama-French Mkt-RF Factor as Blue Liquid Level')
    plt.show()

except Exception as e:
    print(f"Error during liquid level animation: {e}")