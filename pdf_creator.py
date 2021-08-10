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


class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 200

    def header(self):
        self.set_font('Arial', 'B', 11)
        self.cell(self.WIDTH / 2)
        self.cell(-1, 1, 'Realised and implied volatility', 0, 0, 'R')
        self.ln(20)

    def footer(self):
        self.set_y(-10)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, str(self.page_no()), 0, 0, 'C')

    def page_body_two_pictures(self, images, title):
        self.set_font('Arial', size=10)
        self.cell(w=120, h=1, txt=title, border=0, ln=0, align='C')

        self.image(name=images[0], x=5, y=45, w=self.WIDTH - 70)
        self.image(name=images[1], x=self.WIDTH / 2 + 50, y=45, w=self.WIDTH - 70)

    def page_body_one_picture_one_text(self, images, title):
        self.set_font('Arial', size=10)
        self.image(name=images[0], x=20, y=20, w=240)
        self.image(name=images[1], x=10, y=self.HEIGHT * 0.29, w=self.WIDTH * 1.3)

    def page_body_one_picture_one_title(self, image, title):
        self.set_font('Arial', size=10)
        self.cell(w=150, h=1, txt=title, border=0, ln=0, align='C')

        self.image(name=image, x=10, y=40, w=self.WIDTH * 1.3)

    def print_page(self, images):
        self.add_page(orientation='L')
        self.page_body_two_pictures(images[:2], title='Realised volatility histogram, black line - implied volatility')

        self.add_page(orientation='L')
        self.page_body_two_pictures(images[2:4],
                                    title='RV(BTC)/RV(ETH), Histogram of RV(BTC)/RV(ETH), black line - IV(BTC)/IV(ETH)')

        self.add_page(orientation='L')
        self.page_body_one_picture_one_text(images[4:6],
                                            title='IV(ETH) from beta*IV(BTC), beta from linear regression of spot returns')

    def print_page2(self, images):
        # Generates the report
        self.add_page(orientation='L')
        self.page_body_one_picture_one_title(images[0], title='BTC volatility')

        self.add_page(orientation='L')
        self.page_body_one_picture_one_title(images[1],
                                             title='BTC, Black line - IV, X axis - days for RV counting/days before expiration, Y axis - volatility')

        self.add_page(orientation='L')
        self.page_body_one_picture_one_title(images[2],
                                             title='BTC, Black line - IV, X axis - days for RV counting/days before expiration, Y axis - volatility')

        self.add_page(orientation='L')
        self.page_body_one_picture_one_title(images[3], title='ETH volatility')

        self.add_page(orientation='L')
        self.page_body_one_picture_one_title(images[4],
                                             title='ETH, Black line - IV, X axis - days for RV counting/days before expiration, Y axis - volatility')

        self.add_page(orientation='L')
        self.page_body_one_picture_one_title(images[5],
                                             title='ETH, Black line - IV, X axis - days for RV counting/days before expiration, Y axis - volatility')


file_names_for_quantiles = ['plots//quantiles_for_BTC.png', 'plots//quantiles_for_ETH.png',
                            'plots//ratio_RV.png', 'plots//quantiles_for_ratio.png',
                            'plots//text.png', 'plots//beta_regression.png']
file_names_for_rolling_sizes = ['plots//volatility_BTC.png',
                                'plots//quantile_in_RV_for_different_rolling_size_BTC.png',
                                'plots//quantile_in_RV_for_different_rolling_size_BTC_max.png',

                                'plots//volatility_ETH.png',
                                'plots//quantile_in_RV_for_different_rolling_size_ETH.png',
                                'plots//quantile_in_RV_for_different_rolling_size_ETH_max.png',
                                ]


def create_report(report_name='quantiles.pdf'):
    pdf = PDF()
    pdf.print_page(file_names_for_quantiles)
    pdf.print_page2(file_names_for_rolling_sizes)

    pdf.output(name=report_name)
    print('Report is ready')
