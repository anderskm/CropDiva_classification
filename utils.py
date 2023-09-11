import numpy as np

def print_annotation_stats(df):
    print('\nNumber of images with each label:')
    df_img_per_class = df.groupby(['label','UploadID'])['label'].count().unstack().fillna(0).astype(np.uint16)
    df_img_per_class['Sum'] = df_img_per_class.sum(axis=1)
    df_img_per_class = df_img_per_class.append(df_img_per_class.sum(axis=0).rename('Sum'))
    print(df_img_per_class)

    # Number of (unique) images
    print('\nNumber of images:')
    print(df['ImageID'].nunique())
    df_img = df.groupby(['ImageID','label'])['label'].count().unstack().fillna(0).astype(np.uint16)
    print(df_img.nunique(axis=0))

    print('\nAverage area:')
    print(df.groupby(['label'])['area'].mean())

    # print('\nNumber of polygons with each label:')
    # df_poly_per_class = df.groupby(['label','Date'])['N_polygons'].sum().unstack().fillna(0).astype(np.uint16)
    # df_poly_per_class['Sum'] = df_poly_per_class.sum(axis=1)
    # df_poly_per_class = df_poly_per_class.append(df_poly_per_class.sum(axis=0).rename('Sum'))
    # print(df_poly_per_class)

    # print('\nNumber of clusters:')
    # print(df['cluster'].nunique())

    # print('\nNumber of clusters each label appear in:')
    # print(df.groupby(['label'])['cluster'].nunique())

    # print('\nNumber of images per cluster each label appear in:')
    # print(df.groupby(['label'])['label'].count()/df.groupby(['label'])['cluster'].nunique())

    # TODO: Polygon sizes/area per species

def dataframe_filtering(df, mask):
    df_keep = df[mask]
    df_removed = df[~mask]
    return df_keep, df_removed

def countP(n, k):
     
    # Table to store results of subproblems
    dp = [[0 for i in range(k + 1)] 
             for j in range(n + 1)]
 
    # Base cases
    for i in range(n + 1):
        dp[i][0] = 0
 
    for i in range(k + 1):
        dp[0][k] = 0
 
    # Fill rest of the entries in 
    # dp[][] in bottom up manner
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            if (j == 1 or i == j):
                dp[i][j] = 1
            else:
                dp[i][j] = (j * dp[i - 1][j] +
                                dp[i - 1][j - 1])
                 
    return dp[n][k]