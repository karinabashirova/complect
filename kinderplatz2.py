import os
import shutil
from datetime import datetime

import os
import shutil
import numpy as np
import pandas as pd
import calendar
from datetime import datetime
from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

# import plotly.graph_objects as go
#
# df = pd.DataFrame()
# df['ids'] = [1, 1, 1, 2, 2, 2]
# df['parents'] = [3, 3, 2, 5, 5, 6]
# df['labels'] = [1, 2, 3, 6, 5, 8]
# fig = go.Figure()
# fig.add_trace(go.Treemap(
#     ids=df.ids,
#     labels=df.labels,
#     parents=df.parents,
#     maxdepth=3,
# ))
# fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
#
# fig.show()
#
#
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/96c0bd/sunburst-coffee-flavors-complete.csv')
# print(df)


print()


# import time
#
# import plotly.graph_objects as go
#
# import numpy as np
#
# N = 1000
#
# # Create figure
# start_time = time.time()
# fig = go.Figure()
# x = np.random.randn(N)
#
# fig.add_trace(
#     go.Scattergl(
#         x=x,
#         y=np.arange(N),
#         mode='markers+lines',
#         marker=dict(
#             line=dict(
#                 width=1,
#                 color='DarkSlateGrey')
#         )
#     )
# )
#
# fig.show()
# print(time.time()-start_time)
#
#
# start_time = time.time()
# fig = go.Figure()
# fig.add_trace(
#     go.Scatter(
#         x=x,
#         y=np.arange(N),
#         mode='markers+lines',
#         marker=dict(
#             line=dict(
#                 width=1,
#                 color='DarkSlateGrey')
#         )
#     )
# )
#
# fig.show()
# print(time.time()-start_time)


# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# from IPython.display import display, HTML
# import chart_studio.plotly as py
#
# data = [
#     go.Scatter(
#         x=[1, 2, 3],
#         y=[1, 3, 1]
#     )
# ]
# print(data)
# py.plot(data, filename='privacy-public')
# py.iplot(data, filename='privacy-public', sharing='public')

def generate_sales_data(month: int) -> pd.DataFrame:
    # Date range from first day of month until last
    # Use ```calendar.monthrange(year, month)``` to get the last date
    dates = pd.date_range(
        start=datetime(year=2020, month=month, day=1),
        end=datetime(year=2020, month=month, day=calendar.monthrange(2020, month)[1])
    )

    # Sales numbers as a random integer between 1000 and 2000
    sales = np.random.randint(low=1000, high=2000, size=len(dates))

    # Combine into a single dataframe
    return pd.DataFrame({
        'Date': dates,
        'ItemsSold': sales
    })


def plot(data: pd.DataFrame, filename: str) -> None:
    plt.figure(figsize=(12, 4))
    plt.grid(color='#F2F2F2', alpha=1, zorder=0)
    plt.plot(data['Date'], data['ItemsSold'], color='#087E8B', lw=3, zorder=5)
    plt.title(f'Sales 2020/{data["Date"].dt.month[0]}', fontsize=17)
    plt.xlabel('Period', fontsize=13)
    plt.xticks(fontsize=9)
    plt.ylabel('Number of items sold', fontsize=13)
    plt.yticks(fontsize=9)
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    return


december = generate_sales_data(month=12)
plot(data=december, filename='december.png')

PLOT_DIR = 'plots'


def construct():
    # Delete folder if exists and create it again
    try:
        shutil.rmtree(PLOT_DIR)
        os.mkdir(PLOT_DIR)
    except FileNotFoundError:
        os.mkdir(PLOT_DIR)

    # Iterate over all months in 2020 except January
    for i in range(2, 13):
        # Save visualization
        plot(data=generate_sales_data(month=i), filename=f'{PLOT_DIR}//{i}.png')

    # Construct data shown in document
    counter = 0
    pages_data = []
    temp = []
    # Get all plots
    files = os.listdir(PLOT_DIR)
    # Sort them by month - a bit tricky because the file names are strings
    files = sorted(os.listdir(PLOT_DIR), key=lambda x: int(x.split('.')[0]))
    # Iterate over all created visualization
    for fname in files:
        # We want 3 per page
        if counter == 3:
            pages_data.append(temp)
            temp = []
            counter = 0

        temp.append(f'{PLOT_DIR}/{fname}')
        counter += 1

    return [*pages_data, temp]


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297

    def header(self):
        self.set_font('Arial', 'B', 11)
        self.cell(self.WIDTH - 80)
        self.cell(30, 1, 'Realised and implied vols', 0, 0, 'R')
        self.ln(20)

    def footer(self):
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def page_body(self, images):
        # Determine how many plots there are per page and set positions
        # and margins accordingly
        if len(images) == 3:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
            self.image(images[2], 15, self.WIDTH / 2 + 90, self.WIDTH - 30)
        elif len(images) == 2:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        else:
            self.image(images[0], 15, 25, self.WIDTH - 30)

    def print_page(self, images):
        # Generates the report
        self.add_page()
        self.page_body(images)


plots_per_page = construct()
print(plots_per_page)

pdf = PDF()

for elem in plots_per_page:
    pdf.print_page(elem)

pdf.output('SalesRepot.pdf', 'F')
