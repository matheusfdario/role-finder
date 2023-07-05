# -*- coding: utf-8 -*-
"""
Módulo ``Cmaps``
================

Possui mapas de cor que podem ser utilizados pela interface.

.. raw:: html

    <hr>

"""

import numpy as np


civa = np.asarray([
   [0., 254., 255.], [0., 250., 255.], [0., 246., 255.], [0., 242., 255.], [0., 238., 255.], [0., 235.75, 255.],
   [0., 233.5, 255.], [0., 231.25, 255.], [0., 229., 255.], [0., 225., 255.], [0., 221., 255.], [0., 217., 255.],
   [0., 213., 255.], [0., 209., 255.], [0., 205., 255.], [0., 201., 255.], [0., 197., 255.], [0., 193., 255.],
   [0., 189., 255.], [0., 185., 255.], [0., 181., 255.], [0., 178., 255.], [0., 175., 255.], [0., 172., 255.],
   [0., 169., 255.], [0., 165., 255.], [0., 161., 255.], [0., 157., 255.], [0., 153., 255.], [0., 149., 255.],
   [0., 145., 255.], [0., 141., 255.], [0., 137., 255.], [0., 132.75, 255.], [0., 128.5, 255.],
   [0., 124.25, 255.], [0., 120., 255.], [0., 116.25, 255.], [0., 112.5, 255.], [0., 108.75, 255.],
   [0., 105., 255.], [0., 101.25, 255.], [0., 97.5, 255.], [0., 93.75, 255.], [0., 90., 255.], [0., 86.25, 255.],
   [0., 82.5, 255.], [0., 78.75, 255.], [0., 75., 255.], [0., 70.5, 255.], [0., 66., 255.], [0., 61.5, 255.],
   [0., 57., 255.], [1.5, 49.75, 255.], [3., 42.5, 255.], [4.5, 35.25, 255.], [6., 28., 255.], [8., 25.25, 255.],
   [10., 22.5, 255.], [12., 19.75, 255.], [14., 17., 255.], [18., 13.5, 255.], [22., 10., 255.], [26., 6.5, 255.],
   [30., 3., 255.], [36.5, 2.25, 255.], [43., 1.5, 255.], [49.5, 0.75, 255.], [56., 0., 255.], [63.25, 0., 255.],
   [70.5, 0., 255.], [77.75, 0., 255.], [85., 0., 255.], [89.75, 0., 255.], [94.5, 0., 255.], [99.25, 0., 255.],
   [104., 0., 255.], [106., 0., 255.], [108., 0., 255.], [110., 0., 255.], [112., 0., 255.], [115.5, 0., 255.],
   [119., 0., 255.], [122.5, 0., 255.], [126., 0., 255.], [127.5, 0., 255.], [129., 0., 255.], [130.5, 0., 255.],
   [132., 0., 255.], [134., 0., 255.], [136., 0., 255.], [138., 0., 255.], [140., 0., 255.], [141.5, 0., 255.],
   [143., 0., 255.], [144.5, 0., 255.], [146., 0., 255.], [147.5, 0., 255.], [149., 0., 255.], [150.5, 0., 255.],
   [152., 0., 255.], [153.75, 0., 255.], [155.5, 0., 255.], [157.25, 0., 255.], [159., 0., 255.],
   [160.5, 0., 255.], [162., 0., 255.], [163.5, 0., 255.], [165., 0., 255.], [166.5, 0., 255.], [168., 0., 255.],
   [169.5, 0., 255.], [171., 0., 255.], [172.75, 0., 255.], [174.5, 0., 255.], [176.25, 0., 255.],
   [178., 0., 255.], [179.5, 0., 255.], [181., 0., 255.], [182.5, 0., 255.], [184., 0., 255.], [186., 0., 255.],
   [188., 0., 255.], [190., 0., 255.], [192., 0., 255.], [193.75, 0., 255.], [195.5, 0., 255.],
   [197.25, 0., 255.], [199., 0., 255.], [200.5, 0., 255.], [202., 0., 255.], [203.5, 0., 255.], [205., 0., 255.],
   [207., 0., 255.], [209., 0., 255.], [211., 0., 255.], [213., 0., 255.], [213.75, 0., 255.], [214.5, 0., 255.],
   [215.25, 0., 255.], [216., 0., 255.], [217.25, 0., 255.], [218.5, 0., 255.], [219.75, 0., 255.],
   [221., 0., 255.], [222.25, 0., 255.], [223.5, 0., 255.], [224.75, 0., 255.], [226., 0., 255.],
   [227.5, 0., 255.], [229., 0., 255.], [230.5, 0., 255.], [232., 0., 255.], [233.75, 0., 255.],
   [235.5, 0., 255.], [237.25, 0., 255.], [239., 0., 255.], [240.5, 0., 254.75], [242., 0., 254.5],
   [243.5, 0., 254.25], [245., 0., 254.], [245.75, 0., 253.75], [246.5, 0., 253.5], [247.25, 0., 253.25],
   [248., 0., 253.], [248.25, 0., 252.5], [248.5, 0., 252.], [248.75, 0., 251.5], [249., 0., 251.],
   [248.25, 0., 249.75], [247.5, 0., 248.5], [246.75, 0., 247.25], [246., 0., 246.], [244.5, 0., 244.5],
   [243., 0., 243.], [241.5, 0., 241.5], [240., 0., 240.], [238.25, 0., 238.], [236.5, 0., 236.],
   [234.75, 0., 234.], [233., 0., 232.], [231.5, 0.25, 230.25], [230., 0.5, 228.5], [228.5, 0.75, 226.75],
   [227., 1., 225.], [225.75, 1.25, 222.5], [224.5, 1.5, 220.], [223.25, 1.75, 217.5], [222., 2., 215.],
   [220.25, 3.25, 212.5], [218.5, 4.5, 210.], [216.75, 5.75, 207.5], [215., 7., 205.], [213.5, 8., 203.25],
   [212., 9., 201.5], [210.5, 10., 199.75], [209., 11., 198.], [207.25, 11.75, 197.], [205.5, 12.5, 196.],
   [203.75, 13.25, 195.], [202., 14., 194.], [200.5, 13.25, 192.5], [199., 12.5, 191.], [197.5, 11.75, 189.5],
   [196., 11., 188.], [195.25, 9., 187.], [194.5, 7., 186.], [193.75, 5., 185.], [193., 3., 184.],
   [192.5, 2.25, 181.], [192., 1.5, 178.], [191.5, 0.75, 175.], [191., 0., 172.], [191., 0., 167.25],
   [191., 0., 162.5], [191., 0., 157.75], [191., 0., 153.], [191., 0., 148.], [191., 0., 143.], [191., 0., 138.],
   [191., 0., 133.], [191., 0., 129.], [191., 0., 125.], [191., 0., 121.], [191., 0., 117.], [191., 0., 111.5],
   [191., 0., 106.], [191., 0., 100.5], [191., 0., 95.], [191., 0., 89.5], [191., 0., 84.], [191., 0., 78.5],
   [191., 0., 73.], [190.75, 0., 67.], [190.5, 0., 61.], [190.25, 0., 55.], [190., 0., 49.], [189.25, 0., 42.25],
   [188.5, 0., 35.5], [187.75, 0., 28.75], [187., 0., 22.], [186., 1.5, 17.5], [185., 3., 13.], [184., 4.5, 8.5],
   [183., 6., 4.], [182.25, 6.75, 3.5], [181.5, 7.5, 3.], [180.75, 8.25, 2.5], [180., 9., 2.],
   [179.5, 12.75, 1.5], [179., 16.5, 1.], [178.5, 20.25, 0.5], [178., 24., 0.], [178., 26.75, 0.],
   [178., 29.5, 0.], [178., 32.25, 0.], [178., 35., 0.], [178., 37.5, 0.], [178., 40., 0.], [178., 42.5, 0.],
   [178., 45., 0.], [178., 46.75, 0.], [178., 48.5, 0.], [178., 50.25, 0.], [178., 52., 0.], [178., 53.75, 0.],
   [178., 55.5, 0.], [178., 57.25, 0.], [178., 59., 0.], [178., 60.75, 0.], [178., 62.5, 0.], [178., 64.25, 0.],
   [178., 66., 0.], [178., 67., 0.], [178., 68., 0.], [178., 69., 0.], [178., 70., 0.], [178., 71.75, 0.],
   [178., 73.5, 0.], [178., 75.25, 0.], [178., 77., 0.], [178., 78.5, 0.], [178., 80., 0.], [178., 81.5, 0.],
   [178., 83., 0.], [178., 85., 0.], [178., 87., 0.], [178., 89., 0.], [178., 91., 0.], [178., 92., 0.],
   [178., 93., 0.], [178., 94., 0.], [178., 95., 0.], [178., 96.75, 0.], [178., 98.5, 0.], [178., 100.25, 0.],
   [178., 102., 0.], [178., 103.5, 0.], [178., 105., 0.], [178., 106.5, 0.], [178., 108., 0.], [178., 109.75, 0.],
   [178., 111.5, 0.], [178., 113.25, 0.], [178., 115., 0.], [178., 116.75, 0.], [178., 118.5, 0.],
   [178., 120.25, 0.], [178., 122., 0.], [178., 123.25, 0.], [178., 124.5, 0.], [178., 125.75, 0.],
   [178., 127., 0.], [178., 129.25, 0.], [178., 131.5, 0.], [178., 133.75, 0.], [178., 136., 0.],
   [178., 137., 0.], [178., 138., 0.], [178., 139., 0.], [178., 140., 0.], [178., 142.25, 0.], [178., 144.5, 0.],
   [178., 146.75, 0.], [178., 149., 0.], [178., 150., 0.], [178., 151., 0.], [178., 152., 0.], [178., 153., 0.],
   [178., 155.25, 0.], [178., 157.5, 0.], [178., 159.75, 0.], [178., 162., 0.], [178., 162.75, 0.],
   [178., 163.5, 0.], [178., 164.25, 0.], [178., 165., 0.], [177.75, 166.5, 0.], [177.5, 168., 0.],
   [177.25, 169.5, 0.], [177., 171., 0.], [176.5, 171.5, 0.], [176., 172., 0.], [175.5, 172.5, 0.],
   [175., 173., 0.], [174.5, 173., 0.], [174., 173., 0.], [173.5, 173., 0.], [173., 173., 0.],
   [170.75, 170.75, 0.], [168.5, 168.5, 0.], [166.25, 166.25, 0.], [164., 164., 0.], [163.25, 163.25, 0.],
   [162.5, 162.5, 0.], [161.75, 161.75, 0.], [161., 161., 0.], [158.75, 158.75, 0.], [156.5, 156.5, 0.],
   [154.25, 154.25, 0.], [152., 152., 0.], [150.75, 150.75, 0.], [149.5, 149.5, 0.], [148.25, 148.25, 0.],
   [147., 147., 0.], [144.75, 144.75, 0.], [142.5, 142.5, 0.], [140.25, 140.25, 0.], [138., 138., 0.],
   [136.75, 136.75, 0.], [135.5, 135.5, 0.], [134.25, 134.25, 0.], [133., 133., 0.], [131., 131., 0.],
   [129., 129., 0.], [127., 127., 0.], [125., 125., 0.], [124.25, 124.25, 0.], [123.5, 123.5, 0.],
   [122.75, 122.75, 0.], [122., 122., 0.], [120., 120., 0.], [118., 118., 0.], [116., 116., 0.], [114., 114., 0.],
   [112.25, 112.25, 0.], [110.5, 110.5, 0.], [108.75, 108.75, 0.], [107., 107., 0.], [105.75, 105.75, 0.],
   [104.5, 104.5, 0.], [103.25, 103.25, 0.], [102., 102., 0.], [99.25, 99.25, 0.], [96.5, 96.5, 0.],
   [93.75, 93.75, 0.], [91., 91., 0.], [89.5, 89.5, 0.], [88., 88., 0.], [86.5, 86.5, 0.], [85., 85., 0.],
   [85., 85., 0.], [85., 85., 0.], [85., 85., 0.], [85., 85., 0.], [86.5, 86.5, 0.], [88., 88., 0.],
   [89.5, 89.5, 0.], [91., 91., 0.], [93.75, 93.75, 0.], [96.5, 96.5, 0.], [99.25, 99.25, 0.], [102., 102., 0.],
   [103.25, 103.25, 0.], [104.5, 104.5, 0.], [105.75, 105.75, 0.], [107., 107., 0.], [109., 109., 0.],
   [111., 111., 0.], [113., 113., 0.], [115., 115., 0.], [116.75, 116.75, 0.], [118.5, 118.5, 0.],
   [120.25, 120.25, 0.], [122., 122., 0.], [123., 123., 0.], [124., 124., 0.], [125., 125., 0.], [126., 126., 0.],
   [127.75, 127.75, 0.], [129.5, 129.5, 0.], [131.25, 131.25, 0.], [133., 133., 0.], [134.25, 134.25, 0.],
   [135.5, 135.5, 0.], [136.75, 136.75, 0.], [138., 138., 0.], [140.25, 140.25, 0.], [142.5, 142.5, 0.],
   [144.75, 144.75, 0.], [147., 147., 0.], [148.25, 148.25, 0.], [149.5, 149.5, 0.], [150.75, 150.75, 0.],
   [152., 152., 0.], [154.25, 154.25, 0.], [156.5, 156.5, 0.], [158.75, 158.75, 0.], [161., 161., 0.],
   [161.75, 161.75, 0.], [162.5, 162.5, 0.], [163.25, 163.25, 0.], [164., 164., 0.], [166.25, 166.25, 0.],
   [168.5, 168.5, 0.], [170.75, 170.75, 0.], [173., 173., 0.], [173.5, 173., 0.], [174., 173., 0.],
   [174.5, 173., 0.], [175., 173., 0.], [175.5, 172.5, 0.], [176., 172., 0.], [176.5, 171.5, 0.],
   [177., 171., 0.], [177.25, 169.5, 0.], [177.5, 168., 0.], [177.75, 166.5, 0.], [178., 165., 0.],
   [178., 164.25, 0.], [178., 163.5, 0.], [178., 162.75, 0.], [178., 162., 0.], [178., 160., 0.],
   [178., 158., 0.], [178., 156., 0.], [178., 154., 0.], [178., 152.5, 0.], [178., 151., 0.],
   [178., 149.5, 0.], [178., 148., 0.], [178., 146.75, 0.], [178., 145.5, 0.], [178., 144.25, 0.],
   [178., 143., 0.], [178., 140.5, 0.], [178., 138., 0.], [178., 135.5, 0.], [178., 133., 0.],
   [178., 131.75, 0.], [178., 130.5, 0.], [178., 129.25, 0.], [178., 128., 0.], [178., 126.25, 0.],
   [178., 124.5, 0.], [178., 122.75, 0.], [178., 121., 0.], [178., 119.25, 0.], [178., 117.5, 0.],
   [178., 115.75, 0.], [178., 114., 0.], [178., 113., 0.], [178., 112., 0.], [178., 111., 0.], [178., 110., 0.],
   [178., 107.75, 0.], [178., 105.5, 0.], [178., 103.25, 0.], [178., 101., 0.], [178., 100., 0.], [178., 99., 0.],
   [178., 98., 0.], [178., 97., 0.], [178., 95., 0.], [178., 93., 0.], [178., 91., 0.], [178., 89., 0.],
   [178., 87.5, 0.], [178., 86., 0.], [178., 84.5, 0.], [178., 83., 0.], [178., 81.5, 0.], [178., 80., 0.],
   [178., 78.5, 0.], [178., 77., 0.], [178., 75., 0.], [178., 73., 0.], [178., 71., 0.], [178., 69., 0.],
   [178., 68.25, 0.], [178., 67.5, 0.], [178., 66.75, 0.], [178., 66., 0.], [178., 63.5, 0.], [178., 61., 0.],
   [178., 58.5, 0.], [178., 56., 0.], [178., 55., 0.], [178., 54., 0.], [178., 53., 0.], [178., 52., 0.],
   [178., 50., 0.], [178., 48., 0.], [178., 46., 0.], [178., 44., 0.], [178., 41.5, 0.], [178., 39., 0.],
   [178., 36.5, 0.], [178., 34., 0.], [178., 31.75, 0.], [178., 29.5, 0.], [178., 27.25, 0.], [178., 25., 0.],
   [178.75, 21.5, 0.75], [179.5, 18., 1.5], [180.25, 14.5, 2.25], [181., 11., 3.], [181.5, 9.5, 3.5],
   [182., 8., 4.], [182.5, 6.5, 4.5], [183., 5., 5.], [184., 3.75, 7.25], [185., 2.5, 9.5], [186., 1.25, 11.75],
   [187., 0., 14.], [187.75, 0., 23.25], [188.5, 0., 32.5], [189.25, 0., 41.75], [190., 0., 51.],
   [190.25, 0., 54.75], [190.5, 0., 58.5], [190.75, 0., 62.25], [191., 0., 66.], [191., 0., 73.5],
   [191., 0., 81.], [191., 0., 88.5], [191., 0., 96.], [191., 0., 101.25], [191., 0., 106.5], [191., 0., 111.75],
   [191., 0., 117.], [191., 0., 120.25], [191., 0., 123.5], [191., 0., 126.75], [191., 0., 130.],
   [191., 0., 136.25], [191., 0., 142.5], [191., 0., 148.75], [191., 0., 155.], [191., 0., 157.5],
   [191., 0., 160.], [191., 0., 162.5], [191., 0., 165.], [191.5, 1., 170.], [192., 2., 175.], [192.5, 3., 180.],
   [193., 4., 185.], [193.75, 5.75, 185.75], [194.5, 7.5, 186.5], [195.25, 9.25, 187.25], [196., 11., 188.],
   [197.75, 11.75, 189.5], [199.5, 12.5, 191.], [201.25, 13.25, 192.5], [203., 14., 194.], [204.5, 13.25, 195.],
   [206., 12.5, 196.], [207.5, 11.75, 197.], [209., 11., 198.], [210.75, 10., 199.75], [212.5, 9., 201.5],
   [214.25, 8., 203.25], [216., 7., 205.], [217.5, 5.75, 207.75], [219., 4.5, 210.5], [220.5, 3.25, 213.25],
   [222., 2., 216.], [222.75, 1.75, 217.25], [223.5, 1.5, 218.5], [224.25, 1.25, 219.75], [225., 1., 221.],
   [227.25, 0.75, 224.], [229.5, 0.5, 227.], [231.75, 0.25, 230.], [234., 0., 233.], [235.25, 0., 234.75],
   [236.5, 0., 236.5], [237.75, 0., 238.25], [239., 0., 240.], [240.5, 0., 241.5], [242., 0., 243.],
   [243.5, 0., 244.5], [245., 0., 246.], [246., 0., 247.25], [247., 0., 248.5], [248., 0., 249.75],
   [249., 0., 251.], [248.75, 0., 251.5], [248.5, 0., 252.], [248.25, 0., 252.5], [248., 0., 253.],
   [247., 0., 253.5], [246., 0., 254.], [245., 0., 254.5], [244., 0., 255.], [242.75, 0., 255.],
   [241.5, 0., 255.], [240.25, 0., 255.], [239., 0., 255.], [237., 0., 255.], [235., 0., 255.], [233., 0., 255.],
   [231., 0., 255.], [229.75, 0., 255.], [228.5, 0., 255.], [227.25, 0., 255.], [226., 0., 255.],
   [225., 0., 255.], [224., 0., 255.], [223., 0., 255.], [222., 0., 255.], [220.5, 0., 255.], [219., 0., 255.],
   [217.5, 0., 255.], [216., 0., 255.], [215.25, 0., 255.], [214.5, 0., 255.], [213.75, 0., 255.],
   [213., 0., 255.], [211.5, 0., 255.], [210., 0., 255.], [208.5, 0., 255.], [207., 0., 255.], [205., 0., 255.],
   [203., 0., 255.], [201., 0., 255.], [199., 0., 255.], [197.75, 0., 255.], [196.5, 0., 255.],
   [195.25, 0., 255.], [194., 0., 255.], [191.5, 0., 255.], [189., 0., 255.], [186.5, 0., 255.], [184., 0., 255.],
   [182.25, 0., 255.], [180.5, 0., 255.], [178.75, 0., 255.], [177., 0., 255.], [175.5, 0., 255.],
   [174., 0., 255.], [172.5, 0., 255.], [171., 0., 255.], [169.5, 0., 255.], [168., 0., 255.], [166.5, 0., 255.],
   [165., 0., 255.], [163.75, 0., 255.], [162.5, 0., 255.], [161.25, 0., 255.], [160., 0., 255.],
   [158., 0., 255.], [156., 0., 255.], [154., 0., 255.], [152., 0., 255.], [150.75, 0., 255.], [149.5, 0., 255.],
   [148.25, 0., 255.], [147., 0., 255.], [145.25, 0., 255.], [143.5, 0., 255.], [141.75, 0., 255.],
   [140., 0., 255.], [137.75, 0., 255.], [135.5, 0., 255.], [133.25, 0., 255.], [131., 0., 255.],
   [129., 0., 255.], [127., 0., 255.], [125., 0., 255.], [123., 0., 255.], [119.75, 0., 255.], [116.5, 0., 255.],
   [113.25, 0., 255.], [110., 0., 255.], [108.5, 0., 255.], [107., 0., 255.], [105.5, 0., 255.], [104., 0., 255.],
   [99., 0., 255.], [94., 0., 255.], [89., 0., 255.], [84., 0., 255.], [76.75, 0., 255.], [69.5, 0., 255.],
   [62.25, 0., 255.], [55., 0., 255.], [48.5, 1., 255.], [42., 2., 255.], [35.5, 3., 255.], [29., 4., 255.],
   [25., 7.5, 255.], [21., 11., 255.], [17., 14.5, 255.], [13., 18., 255.], [11., 20.75, 255.], [9., 23.5, 255.],
   [7., 26.25, 255.], [5., 29., 255.], [3.75, 36., 255.], [2.5, 43., 255.], [1.25, 50., 255.], [0., 57., 255.],
   [0., 61.75, 255.], [0., 66.5, 255.], [0., 71.25, 255.], [0., 76., 255.], [0., 78.25, 255.], [0., 80.5, 255.],
   [0., 82.75, 255.], [0., 85., 255.], [0., 90.25, 255.], [0., 95.5, 255.], [0., 100.75, 255.], [0., 106., 255.],
   [0., 108.75, 255.], [0., 111.5, 255.], [0., 114.25, 255.], [0., 117., 255.], [0., 122., 255.],
   [0., 127., 255.], [0., 132., 255.], [0., 137., 255.], [0., 141.25, 255.], [0., 145.5, 255.],
   [0., 149.75, 255.], [0., 154., 255.], [0., 156.5, 255.], [0., 159., 255.], [0., 161.5, 255.], [0., 164., 255.],
   [0., 168.75, 255.], [0., 173.5, 255.], [0., 178.25, 255.], [0., 183., 255.], [0., 185., 255.],
   [0., 187., 255.], [0., 189., 255.], [0., 191., 255.], [0., 196.75, 255.], [0., 202.5, 255.],
   [0., 208.25, 255.], [0., 214., 255.], [0., 217.25, 255.], [0., 220.5, 255.], [0., 223.75, 255.],
   [0., 227., 255.], [0., 230.25, 255.], [0., 233.5, 255.], [0., 236.75, 255.], [0., 240., 255.],
   [0., 243.5, 255.], [0., 247., 255.], [0., 250.5, 255.], [0., 254., 255.], [0., 254., 255.], [0., 254., 255.],
   [0., 254., 255.]])