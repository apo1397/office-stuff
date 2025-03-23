import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_word_cloud(text_data, font_path='/Users/apoorvabhishek/Documents/fonts/Poppins-Bold.ttf', background_color='white', 
                     width=1600, height=800, colormap='tab10',
                     min_font_size=8, max_font_size=None,relative_scaling=0.3,  # Reduced to make size differences less extreme
                     prefer_horizontal=0.6):
    """
    Create a word cloud from text data with weights.
    
    Parameters:
    text_data (str): Data in format "word percentage\n" (tab-separated)
    font_path (str): Path to custom font file (optional)
    background_color (str): Background color of word cloud
    width (int): Width of the output image
    height (int): Height of the output image
    colormap (str): Matplotlib colormap name for words
    min_font_size (int): Minimum font size for words
    max_font_size (int): Maximum font size for words (optional)
    """
    # Parse input data
    word_weights = defaultdict(float)
    for line_number, line in enumerate(text_data.strip().split('\n'), 1):
        line = line.strip()
        if not line:  # Skip empty lines
            logging.info(f"Skipping empty line at line number {line_number}")
            continue
        try:
            word, weight_str = line.split('\t')
            weight = float(weight_str.strip('%'))
            word_weights[word] += weight
        except ValueError as e:
            logging.error(f"Error processing line {line_number}: {line}.  Error: {e}")
            continue  # Skip malformed lines
    
    # Convert to dictionary format required by WordCloud
    word_dict = {word: weight for word, weight in word_weights.items()}
    
    # Create and configure the WordCloud object
    wc = WordCloud(
        background_color=background_color,
        width=width,
        height=height,
        colormap=colormap,
        font_path=font_path,
        prefer_horizontal=prefer_horizontal,
        relative_scaling=relative_scaling,
        min_font_size=min_font_size,
        max_font_size=max_font_size
    )
    
    # Generate the word cloud
    wc.generate_from_frequencies(word_dict)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    
    return wc  # Return the WordCloud object for potential saving

# Sample usage
sample_data = """
United States	0.013691
Nepal	0.011792
Nepal	0.005647
Indonesia	0.030862
	
United Arab Emirates	0.022099
Indonesia	0.005505
United States	0.023915
United Arab Emirates	0.007071
Canada	0.004819
	
United Arab Emirates	0.005199
Bangladesh	0.004467
United Arab Emirates	0.013891
	
United Arab Emirates	0.006059
United Arab Emirates	0.004463
Brazil	0.002805
South Korea	0.001638
United Kingdom	0.004646
United Kingdom	0.004736
Pakistan	0.006978
Germany	0.001468
Philippine Continental Shelf	0.005861
Russia	0.004607
United Kingdom	0.006731
Canada	0.003306
	
Australia	0.006833
United States	0.003409
Canada	0.004688
Pakistan	0.006496
Brazil	0.009712
Iran	0.020527
Bangladesh	0.004433
	
Bangladesh	0.002142
	
Canada	0.006336
United Arab Emirates	0.0041
	
United Arab Emirates	0.025559
Nepal	0.004673
	
	
Canada	0.017429
Indonesia	0.006171
	
	
United States	0.008021
	
Saudi Arabia	0.00501
Italy	0.012191
United Kingdom	0.004807
	
	
Russia	0.030499
	
Indonesia	0.004032
Nepal	0.003885
Nepal	0.009211
United States	0.00541
	
Philippine Continental Shelf	0.011852
	
Indonesia	0.004857
Indonesia	0.004612
United States	0.013196
	
United Kingdom	0.005641
United States	0.020421
United States	0.020091
Nigeria	0.018095
Canada	0.01617
United States	0.008287
Indonesia	0.003591
	
	
	
United Kingdom	0.004695
Mauritius	0.003013
United States	0.013889
Hungary	0.003903
Egypt	0.008907
Pakistan	0.006413
	
	
Pakistan	0.006797
South Africa	0.015271
Bangladesh	0.016509
United Arab Emirates	0.006082
Brazil	0.034191
	
United Kingdom	0.022099
United States	0.00842
Brazil	0.03041
Pakistan	0.00825
Pakistan	0.00514
	
United States	0.006499
United Kingdom	0.005955
United States	0.015536
	
United States	0.011161
Pakistan	0.007364
United States	0.003741
United Kingdom	0.002574
Indonesia	0.005226
Bangladesh	0.004736
United States	0.007401
United Arab Emirates	0.001762
Indonesia	0.009769
South Korea	0.00499
Bangladesh	0.008373
United Arab Emirates	0.003636
	
Canada	0.006833
Indonesia	0.003409
Pakistan	0.007031
Canada	0.006838
Turkey	0.013528
Russia	0.0226
Pakistan	0.0057
	
United Arab Emirates	0.004641
	
United Kingdom	0.006336
Italy	0.005125
	
United States	0.032481
United Arab Emirates	0.005032
United Arab Emirates	0.001078
	
United Arab Emirates	0.018519
Canada	0.006708
	
	
Indonesia	0.008021
	
Qatar	0.005611
United Kingdom	0.014868
Bhutan	0.007865
	
	
Azerbaijan	0.060832
	
Pakistan	0.010081
United Arab Emirates	0.007285
United Arab Emirates	0.011842
United Arab Emirates	0.008873
	
United Arab Emirates	0.012404
	
United Kingdom	0.007947
United Arab Emirates	0.005451
Bulgaria	0.021017
	
Nepal	0.008205
Turkey	0.03127
Turkey	0.031756
United Kingdom	0.020952
United Kingdom	0.02383
United Arab Emirates	0.008594
United Kingdom	0.006822
	
	
	
United States	0.005122
United States	0.007231
Bangladesh	0.018519
Pakistan	0.004684
United States	0.016627
United Kingdom	0.0074
	
	
United States	0.009515
Pakistan	0.010095
Pakistan	0.001877
United Arab Emirates	0.011563
	
	
Nepal	0.011031
Bangladesh	0.004079
Bangladesh	0.005008
Bangladesh	0.009491
United States	0.006233
	
United Kingdom	0.007045
Bangladesh	0.003273
Nepal	0.004108
France	0.008272
Brazil	0.007735
Bangladesh	0.007073
	
	
Nepal	0.003233
Kenya	0.021064
United States	0.019261
United States	0.007819
Uzbekistan	0.042663
	
Australia	0.024862
Pakistan	0.013925
Iran	0.153823
Bangladesh	0.008839
United Kingdom	0.008673
	
Bangladesh	0.006499
United Arab Emirates	0.009429
Pakistan	0.018461
	
Pakistan	0.01148
Bangladesh	0.007364
Pakistan	0.004364
United Arab Emirates	0.004679
United States	0.007549
United Arab Emirates	0.006051
Bangladesh	0.019032
United Kingdom	0.002055
United States	0.022143
United States	0.015739
United Arab Emirates	0.00903
United Kingdom	0.004298
	
United Kingdom	0.013264
Pakistan	0.004545
United Arab Emirates	0.0075
United Arab Emirates	0.011282
Pakistan	0.018384
Brazil	0.037321
Brazil	0.0057
	
Nepal	0.004641
	
United States	0.010657
United Kingdom	0.005466
	
Bangladesh	0.049521
Canada	0.00683
Nepal	0.001232
	
United Kingdom	0.031808
United Arab Emirates	0.010196
	
	
Bangladesh	0.027786
	
United Kingdom	0.006613
United Arab Emirates	0.014868
United States	0.010706
	
	
Iran	0.110227
United Arab Emirates	0.00117
United States	0.010081
Pakistan	0.007771
Pakistan	0.017982
Bangladesh	0.02467
	
United Kingdom	0.016538
	
United Arab Emirates	0.010155
Bangladesh	0.008386
Bangladesh	0.022483
	
United Arab Emirates	0.008718
Pakistan	0.033184
Pakistan	0.033052
United Arab Emirates	0.03
Pakistan	0.024468
Pakistan	0.012584
United Arab Emirates	0.008977
	
	
	
Bangladesh	0.008536
Bangladesh	0.015366
Indonesia	0.037037
United States	0.006245
Indonesia	0.023951
Bangladesh	0.00962
	
	
Nepal	0.012234
United States	0.010656
Bangladesh	0.002346
Pakistan	0.013428
	
	
United States	0.011792
United Kingdom	0.005562
United Arab Emirates	0.006439
Pakistan	0.009922
Pakistan	0.006233
	
United Arab Emirates	0.007045
Nepal	0.005456
Bangladesh	0.004695
Malaysia	0.017923
Indonesia	0.009945
United States	0.009582
	
	
United Arab Emirates	0.004096
Bangladesh	0.005518
United Arab Emirates	0.007178
	
Brazil	0.008876
Nepal	0.01093
United Kingdom	0.006308
	
Japan	0.002309
Pakistan	0.009397
Brazil	0.02308
Bangladesh	0.017552
Pakistan	0.020557
Nepal	0.001838
Bangladesh	0.009763
Pakistan	0.005067
	
	
Nepal	0.019253
Nepal	0.004122
	
United States	0.007684
Indonesia	0.243813
Pakistan	0.023192
Bangladesh	0.009557
Turkey	0.151286
	
United States	0.118785
United Arab Emirates	0.018459
Turkey	0.182167
United States	0.011196
United States	0.015098
	
Nepal	0.02253
United States	0.010918
Indonesia	0.029976
	
Bangladesh	0.05102
United States	0.008034
Bangladesh	0.004988
United States	0.009359
United Arab Emirates	0.007549
United States	0.00684
Nepal	0.032142
United States	0.006166
Nepal	0.060892
Nepal	0.048752
United States	0.022164
United States	0.007934
	
United States	0.034164
Bangladesh	0.006818
United States	0.011562
United States	0.015726
Indonesia	0.021159
Turkey	0.226
Indonesia	0.0095
	
United States	0.00714
	
Bangladesh	0.011233
United States	0.013324
	
Pakistan	0.073482
United States	0.0133
United States	0.002927
	
United States	0.044444
United States	0.037832
	
	
Pakistan	0.028359
	
United Arab Emirates	0.030862
United States	0.016949
Nepal	0.115796
	
	
Turkey	0.163269
United States	0.001871
United Arab Emirates	0.012097
Bangladesh	0.016513
United States	0.019298
Pakistan	0.032893
	
United States	0.051544
	
United States	0.015453
United States	0.013417
Turkey	0.025904
	
United States	0.010769
Indonesia	0.052967
Indonesia	0.052495
United States	0.045714
United States	0.037234
Canada	0.016575
United States	0.01939
	
United States	0.001649
	
United Arab Emirates	0.02134
Nepal	0.067792
Brazil	0.060185
Italy	0.007026
Iran	0.096397
United States	0.0148
	
	
Bangladesh	0.014046
Bangladesh	0.012339
Nepal	0.005162
United States	0.024245
	
	
Bangladesh	0.013313
United States	0.00964
United States	0.008824
United States	0.02157
Nepal	0.008459
	
United States	0.009393
United States	0.009274
United States	0.005869
United States	0.024816
Pakistan	0.01768
Pakistan	0.013689
	
	
United States	0.005389
United States	0.007465
United States	0.011419
	
United States	0.011834
Bangladesh	0.01846
United Arab Emirates	0.023129
	
United States	0.005196
Nepal	0.012161
United States	0.035944
Pakistan	0.018963
United States	0.026525
United States	0.005882
United States	0.021618
Bangladesh	0.006891
	
	
Bangladesh	0.021518
United Arab Emirates	0.007832
United States	0.001235
Pakistan	0.010758
United States	0.012593
Bangladesh	0.015661
United States	0.016881
United States	0.040352
United States	0.01272
United States	0.011137
	
United States	0.016171
India	0.936142
"""

# Example usage with different font sizes
# Default minimum font size (8)
wc = create_word_cloud(sample_data)

# Larger minimum font size
# wc = create_word_cloud(sample_data, min_font_size=12)

# Both minimum and maximum font size
# wc = create_word_cloud(sample_data, min_font_size=12, max_font_size=50)

# To save the word cloud
# wc.to_file('wordcloud.png')