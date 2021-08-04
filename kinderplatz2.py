# import pandas as pd
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

